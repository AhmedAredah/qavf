"""Adversarial invariants for statistical and temporal QA rules."""

import time

import numpy as np
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.rules.temporal_checks import CumulativeRule, StepChangeRule, _trend_outlier_test
from qavf.stats import detect_outliers, flag_outliers, zscore_min_samples
from qavf.testing import rule_results, run_rule


def series(values, **extra):
    return {"s": pl.DataFrame({"t": range(len(values)), "v": values})}, {
        "series": [{"dataset": "s", "time_column": "t", "value": "v"}],
        **extra,
    }


@pytest.mark.parametrize("ref,cur", [([1.0] * 100, [1e9] * 100), ([0.0] * 99 + [1.0], [1.0] * 100)])
def test_degenerate_numeric_psi_detects_support_changes(ref, cur):
    r = run_rule(
        "STA-002",
        {"ref": pl.DataFrame({"x": ref}), "cur": pl.DataFrame({"x": cur})},
        {"comparisons": [{"dataset": "cur", "reference": "ref", "column": "x"}]},
    )
    assert r.status is Status.FAIL and r.metrics["psi"] > 0.2


def test_stable_psi_keeps_auditable_metrics():
    f = pl.DataFrame({"x": [1.0] * 100})
    r = run_rule(
        "STA-002",
        {"ref": f, "cur": f},
        {"comparisons": [{"dataset": "cur", "reference": "ref", "column": "x"}]},
    )
    assert r.status is Status.PASS and r.metrics["psi"] == 0
    assert r.metrics["reference_values"] == 100


@pytest.mark.parametrize("weights", [[100.0, 1.0, 100.0, 100.0], [1.0, 100.0, 100.0, 100.0]])
def test_duplicate_denominator_is_never_order_dependent(weights):
    r = run_rule(
        "SPA-002",
        {
            "n": pl.DataFrame({"id": [1, 2, 3], "v": [100.0] * 3}),
            "d": pl.DataFrame({"id": [1, 1, 2, 3], "w": weights}),
        },
        {
            "metrics": [
                {
                    "dataset": "n",
                    "value": "v",
                    "entity_column": "id",
                    "normalize_by": {"dataset": "d", "value": "w", "aggregate": "none"},
                }
            ]
        },
    )
    assert r.status is Status.ERROR and "denominator" in r.message


@pytest.mark.parametrize("gap", [0.5, 1.0, 2.0, 3.0])
def test_sign_reversal_cannot_be_zero_compound_change(gap):
    r = run_rule(
        "TMP-001",
        {"s": pl.DataFrame({"t": [0.0, gap], "v": [100.0, -100.0]})},
        {"series": [{"dataset": "s", "time_column": "t", "value": "v"}]},
    )
    assert r.status is Status.FAIL and r.affected_records[0]["change_rate"] is None


def test_level_shift_material_candidate_not_hidden_by_tied_score():
    r = run_rule("TMP-008", *series([10.0] * 4 + [11.0] * 5 + [20.0] * 4))
    assert r.status is Status.FAIL and r.affected_records[0]["median_after"] == 20


def test_exact_level_shift_budget_reports_incomplete():
    r = run_rule("TMP-008", *series([10.0] * 20 + [20.0] * 20, max_work=100))
    assert r.status is Status.ERROR and "max_work" in r.message


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
def test_trend_verdict_is_invariant_to_extreme_unit_scale(scale):
    r = run_rule(
        "TMP-003",
        *series(
            [v * scale for v in [1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0, 8.0]], log_transform=False
        ),
    )
    assert r.status is Status.FAIL and r.affected_count == 1


def test_linear_time_studentized_kernel_matches_independent_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(22)
    t = np.arange(40, dtype=float)
    v = 5 + t * 0.2 + rng.normal(0, 1, len(t))
    v[12] += 5
    ours = _trend_outlier_test(t, v, False)
    assert ours is not None
    reference = np.asarray(sm.OLS(v, sm.add_constant(t)).fit().outlier_test(method="bonferroni"))
    np.testing.assert_allclose(ours.residuals, reference[:, 0], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(ours.adjusted_p, reference[:, 2], rtol=1e-9, atol=1e-10)


def test_normalization_overflow_is_disclosed_with_valid_peers():
    results = rule_results(
        "SPA-001",
        {
            "n": pl.DataFrame({"id": range(5), "v": [1.0] * 4 + [1e308]}),
            "d": pl.DataFrame({"id": range(5), "w": [1.0] * 4 + [1e-308]}),
        },
        {
            "metrics": [
                {
                    "dataset": "n",
                    "value": "v",
                    "entity_column": "id",
                    "normalize_by": {"dataset": "d", "value": "w"},
                }
            ],
            "min_entities": 3,
        },
    )
    assert any(r.status is Status.ERROR and r.affected_count == 1 for r in results)


def test_huge_threshold_has_bounded_calculation_and_scan():
    start = time.monotonic()
    n = zscore_min_samples(1e308)
    assert n > 10**600
    scan = flag_outliers(
        pl.DataFrame({"x": [1.0, 2.0, 3.0]}), "x", method="zscore", threshold=1e308
    )
    assert scan.tested == 0
    assert time.monotonic() - start < 2


@pytest.mark.parametrize("rule", [StepChangeRule, CumulativeRule])
@pytest.mark.parametrize("limit", [0, 3])
def test_rule_evidence_bounded_before_engine_serialization(rule, limit):
    frames, params = series([float(10000 - i) for i in range(10000)])
    if rule is StepChangeRule:
        params["max_change_rate"] = 1e-8
    ctx = DataContext(frames=frames, evidence_limit=limit)
    try:
        findings = rule(params).evaluate(ctx)
    finally:
        ctx.close()
    failed = next(f for f in findings if f.assessed and not f.passed)
    assert failed.affected_count == 9999 and len(failed.affected) == limit


def test_float_max_grouped_bounds_match_array():
    x = [1.0, 1.0, 1.0, 1.0, np.finfo(float).max]
    arr = detect_outliers(x)
    grouped = flag_outliers(pl.DataFrame({"x": x}), "x")
    assert grouped.frame[grouped.columns.group_median][0] == pytest.approx(1.0)
    assert grouped.frame[grouped.columns.lower_bound][0] == pytest.approx(arr.lower)
    assert grouped.frame[grouped.columns.upper_bound][0] == pytest.approx(arr.upper)


def test_partial_untested_groups_are_explicit():
    f = pl.DataFrame({"g": ["ok"] * 8 + ["small"], "x": [1.0] * 8 + [1e20]})
    result = rule_results(
        "STA-001", {"s": f}, {"targets": [{"dataset": "s", "value": "x", "group_by": ["g"]}]}
    )
    assert any(r.status is Status.ERROR and r.affected_count == 1 for r in result)


def test_reference_only_is_not_a_successful_coverage_check():
    r = run_rule(
        "SPA-003",
        {"ref": pl.DataFrame({"id": [1, 2, 3]})},
        {"entity_column": "id", "reference_dataset": "ref"},
    )
    assert r.status is Status.ERROR


def test_legitimate_constants_excluded_from_both_stale_subchecks():
    f = pl.DataFrame(
        {"site": ["a"] * 3 + ["b"] * 3, "year": [2020, 2021, 2022] * 2, "v": [0.0] * 6}
    )
    r = run_rule(
        "TMP-005",
        {"s": f},
        {"series": [{"dataset": "s", "value": "v", "group_by": ["site"]}], "ignore_values": [0.0]},
    )
    assert r.status is Status.PASS


def test_lossy_large_integer_outlier_input_is_not_certified_clean():
    frame = pl.DataFrame({"v": [2**60 + i for i in range(8)]})
    with pytest.raises(ValueError, match="exactly"):
        flag_outliers(frame, "v")
    r = run_rule("STA-001", {"s": frame}, {"targets": [{"dataset": "s", "value": "v"}]})
    assert r.status is Status.ERROR and "exactly" in r.message


def test_cumulative_native_large_integers_remain_exact():
    frame = pl.DataFrame({"t": [0, 1], "v": [2**60 + 1, 2**60]})
    r = run_rule(
        "TMP-006", {"s": frame}, {"series": [{"dataset": "s", "value": "v", "time_column": "t"}]}
    )
    assert r.status is Status.FAIL and r.affected_records[0]["decrease"] == 1


def test_entirely_nonfinite_group_remains_visible():
    f = pl.DataFrame({"g": ["ok"] * 8 + ["missing"], "x": [1.0] * 8 + [None]})
    results = rule_results(
        "STA-001", {"s": f}, {"targets": [{"dataset": "s", "value": "x", "group_by": ["g"]}]}
    )
    assert any(r.status is Status.ERROR and r.affected_count == 1 for r in results)


@pytest.mark.parametrize("rule", ["TMP-001", "TMP-006"])
def test_singleton_temporal_groups_are_explicit(rule):
    frame = pl.DataFrame({"g": ["a", "a", "b"], "t": [0, 1, 0], "v": [1.0, 1.0, 1.0]})
    results = rule_results(
        rule,
        {"s": frame},
        {"series": [{"dataset": "s", "value": "v", "time_column": "t", "group_by": ["g"]}]},
    )
    assert any(r.status is Status.ERROR and r.affected_count == 1 for r in results)


def test_implicit_spatial_partitions_with_too_few_entities_are_visible():
    frame = pl.DataFrame({"g": ["a"] * 3 + ["b"], "e": [1, 2, 3, 1], "v": [1.0] * 4})
    results = rule_results(
        "SPA-002",
        {"s": frame},
        {"metrics": [{"dataset": "s", "value": "v", "entity_column": "e", "group_by": ["g"]}]},
    )
    assert any(r.status is Status.ERROR and r.affected_count == 1 for r in results)


def test_decimal_spatial_outlier_statistics_have_explicit_float_conversion():
    from decimal import Decimal

    frame = pl.DataFrame({"id": range(8), "v": [Decimal("1.0")] * 7 + [Decimal("100.0")]})
    r = run_rule(
        "SPA-001",
        {"s": frame},
        {"metrics": [{"dataset": "s", "value": "v", "entity_column": "id"}]},
    )
    assert r.status is Status.FAIL and r.affected_count == 1
