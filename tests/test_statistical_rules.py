"""Regression, reference and metamorphic coverage for statistical QA integrity."""

import math
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.rules.geo_checks import CoordinateRule
from qavf.rules.statistical_checks import ColumnOutlierRule
from qavf.rules.temporal_checks import _compound_change, _trend_outlier_test
from qavf.stats import detect_outliers, flag_outliers
from qavf.testing import rule_results, run_rule


def series_params(**extra):
    return {"series": [{"dataset": "s", "time_column": "t", "value": "v"}], **extra}


@pytest.mark.parametrize(
    "name",
    [
        "__qavf_original_median",
        "__qavf_relative_deviation",
        "relative_deviation",
        "score",
        "__qavf_row",
    ],
)
def test_outlier_verdict_and_original_evidence_independent_of_extra_columns(name):
    frame = pl.DataFrame({"v": [1.0] * 7 + [100.0], "g": ["a"] * 8})

    def evaluate(data, value="v", group="g"):
        return run_rule(
            "STA-001",
            {"s": data},
            {
                "targets": [
                    {
                        "dataset": "s",
                        "value": value,
                        "group_by": [group],
                        "min_relative_deviation": 0.1,
                    }
                ]
            },
        )

    baseline = evaluate(frame)
    result = evaluate(frame.with_columns(pl.lit(100.0).alias(name)))
    assert result.status is baseline.status is Status.FAIL
    assert result.affected_count == baseline.affected_count == 1
    assert result.total_records == baseline.total_records == 8
    assert result.metrics == baseline.metrics
    expected, actual = dict(baseline.affected_records[0]), dict(result.affected_records[0])
    assert expected["v"] == 100.0 and expected["g"] == "a"
    assert expected["relative_deviation"] == 99.0
    # The focused evidence drops unrelated columns. Even an unrelated index-name
    # collision must preserve the position and every retained diagnostic value.
    row_name = "qavf___qavf_row" if name == "__qavf_row" else "__qavf_row"
    assert actual.pop(row_name) == expected.pop("__qavf_row") == 7
    assert actual == expected

    # The same names remain user data when they are checked values or group keys;
    # diagnostics must take a separate name instead of replacing that raw data.
    for role in ("v", "g"):
        value, group = (name, "g") if role == "v" else ("v", name)
        collision = evaluate(frame.rename({role: name}), value, group)
        assert collision.status is Status.FAIL and collision.affected_count == 1
        assert collision.total_records == 8
        record = collision.affected_records[0]
        assert record[value] == 100.0 and record[group] == "a"
        assert record[row_name] == 7
        for diagnostic in (
            "score", "lower_bound", "upper_bound", "group_median", "relative_deviation"
        ):
            output = f"qavf_{diagnostic}" if diagnostic == name else diagnostic
            assert record[output] == expected[diagnostic]


@pytest.mark.parametrize("name", ["__qavf_log_value", "log_value", "__qavf_relative_deviation"])
def test_log_transform_preserves_group_identity(name):
    f = pl.DataFrame({"v": [1.0] * 7 + [100.0], name: ["a"] * 8})
    r = run_rule(
        "STA-001",
        {"s": f},
        {
            "targets": [
                {
                    "dataset": "s",
                    "value": "v",
                    "group_by": [name],
                    "transform": "log",
                    "min_relative_deviation": 0.1,
                }
            ]
        },
    )
    assert r.status is Status.FAIL and r.affected_count == 1
    assert r.affected_records[0][name] == "a"


@pytest.mark.parametrize(
    "name",
    ["__qavf_value", "__qavf_previous_value", "__qavf_previous_period", "__qavf_change_rate"],
)
@pytest.mark.parametrize("rule", ["TMP-001", "TMP-006"])
def test_temporal_generated_columns_do_not_overwrite_keys(name, rule):
    f = pl.DataFrame({"t": [0, 1], "v": [100, 1], name: ["a", "a"]})
    p = {"series": [{"dataset": "s", "value": "v", "time_column": "t", "group_by": [name]}]}
    r = run_rule(rule, {"s": f}, p)
    assert r.status is Status.FAIL and r.affected_count == 1
    assert r.affected_records[0][name] == "a"


@pytest.mark.parametrize("name", ["__qavf_row", "__qavf_problem", "_row", "problem"])
def test_geo_diagnostic_columns_preserve_original_values(name):
    f = pl.DataFrame({"latitude": [91.0], "longitude": [0.0], name: ["original"]})
    r = run_rule("GEO-001", {"s": f}, {"points": [{"dataset": "s"}]})
    assert r.status is Status.FAIL and r.affected_records[0][name] == "original"


@pytest.mark.parametrize(
    "values",
    [
        np.array([2**70] * 7 + [2**70 + 1], dtype=object),
        [Decimal("1e20")] * 7 + [Decimal("100000000000000000001")],
    ],
)
@pytest.mark.parametrize("method", ["iqr", "zscore", "modified_zscore"])
def test_object_numeric_arrays_cannot_bypass_lossless_conversion(values, method):
    with pytest.raises(ValueError, match="exactly"):
        detect_outliers(values, method)


@pytest.mark.parametrize("values", [[True, False, True], ["1", "2", "9"]])
def test_array_dtype_contract_matches_frame_numeric_contract(values):
    with pytest.raises(ValueError, match="numeric"):
        detect_outliers(values)


@pytest.mark.parametrize(
    "values", [[1e-300] * 6 + [1e-290, 1e308], [1e-300] * 6 + [-1e-290, 1e308]]
)
def test_mixed_range_is_explicitly_unassessed_in_both_interfaces(values):
    with pytest.raises(ValueError, match="precision loss"):
        detect_outliers(values, "iqr")
    scan = flag_outliers(pl.DataFrame({"v": values}), "v", method="iqr")
    assert scan.tested == 0 and scan.scores.null_count() == len(values)
    r = run_rule(
        "STA-001",
        {"s": pl.DataFrame({"v": values})},
        {"targets": [{"dataset": "s", "value": "v", "direction": "low"}]},
    )
    assert r.status is Status.ERROR


def test_numeric_uncertainty_in_one_group_does_not_hide_another_failure():
    values = [1e-300] * 6 + [-1e-290, 1e308] + [1.0] * 7 + [100.0]
    f = pl.DataFrame({"v": values, "g": ["uncertain"] * 8 + ["bad"] * 8})
    results = rule_results(
        "STA-001", {"s": f}, {"targets": [{"dataset": "s", "value": "v", "group_by": ["g"]}]}
    )
    assert any(r.status is Status.ERROR and r.affected_count == 8 for r in results)
    assert any(r.status is Status.FAIL and r.affected_count == 1 for r in results)


@pytest.mark.parametrize("exponent", [-300, -100, 0, 100, 300])
def test_iqr_agrees_with_independent_fences_under_unit_changes(exponent):
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 20.0]) * 10.0**exponent
    q1, q3 = np.quantile(values, [0.25, 0.75])
    expected = (values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))
    result = detect_outliers(values, "iqr")
    np.testing.assert_array_equal(result.mask, expected)
    assert (
        flag_outliers(pl.DataFrame({"v": values}), "v", method="iqr").flags.to_list()
        == expected.tolist()
    )


@pytest.mark.parametrize(
    "axis,bound", [("latitude", 90), ("latitude", -90), ("longitude", 180), ("longitude", -180)]
)
def test_decimal_coordinate_boundaries_are_compared_before_conversion(axis, bound):
    outside = Decimal(bound) + (Decimal("1e-18") if bound > 0 else -Decimal("1e-18"))
    f = pl.DataFrame({"latitude": [Decimal(1)], "longitude": [Decimal(1)]}).with_columns(
        pl.Series(axis, [outside])
    )
    r = run_rule("GEO-001", {"s": f}, {"points": [{"dataset": "s"}]})
    assert r.status is Status.FAIL


def test_decimal_bbox_keeps_fractional_boundary_and_original_coordinate():
    v = Decimal("1.000000000000000001")
    f = pl.DataFrame({"latitude": [v], "longitude": [Decimal(1)]})
    p = {
        "points": [
            {
                "dataset": "s",
                "bbox": {"min_lat": 0, "max_lat": Decimal(1), "min_lon": 0, "max_lon": 2},
            }
        ]
    }
    r = run_rule("GEO-001", {"s": f}, p)
    assert r.status is Status.FAIL and r.affected_records[0]["latitude"] == v


@pytest.mark.parametrize("values,limit", [([-(2**63), 0], 0.25), ([2**60, 2**60 + 1], 1e-20)])
def test_exact_integer_steps_do_not_disappear(values, limit):
    r = run_rule(
        "TMP-001",
        {"s": pl.DataFrame({"t": [0, 1], "v": values})},
        series_params(max_change_rate=limit),
    )
    assert r.status is Status.FAIL and r.affected_count == 1


@pytest.mark.parametrize("start,end", [(1e-200, 1e200), (1e200, 1e-200)])
def test_compound_rates_match_log_reference_without_ratio_overflow(start, end):
    expected = math.expm1((math.log(end) - math.log(start)) / 10000)
    assert _compound_change(start, end, 0, 10000) == pytest.approx(expected)
    r = run_rule(
        "TMP-001", {"s": pl.DataFrame({"t": [0, 10000], "v": [start, end]})}, series_params()
    )
    assert r.status is Status.PASS


def test_signed_time_difference_cannot_wrap():
    r = run_rule(
        "TMP-001",
        {"s": pl.DataFrame({"t": [-(2**63), 2**63 - 1], "v": [1.0, 4.0]})},
        series_params(),
    )
    assert r.status is Status.PASS


@pytest.mark.parametrize("offset", [0, 2**60])
def test_compound_rate_preserves_small_change_at_large_time_offset(offset):
    r = run_rule(
        "TMP-001",
        {"s": pl.DataFrame({"t": [offset, offset + 1], "v": [2**60, 2**60 + 1]})},
        series_params(max_change_rate=1e-20),
    )
    assert r.status is Status.FAIL
    assert r.affected_records[0]["change_rate"] == pytest.approx(2**-60)


@pytest.mark.parametrize("how", ["mean", "weighted_mean"])
def test_aggregation_never_certifies_a_rounded_cumulative_drop(how):
    f = pl.DataFrame({"t": [0, 1], "v": [2**60 + 1, 2**60], "w": [1.0, 1.0]})
    spec = {"dataset": "s", "value": "v", "time_column": "t", "aggregate": how}
    if how == "weighted_mean":
        spec["weight"] = "w"
    r = run_rule("TMP-006", {"s": f}, {"series": [spec]})
    assert r.status is Status.ERROR


def test_material_outlier_near_perfect_trend_matches_exact_normal_equations():
    t = np.arange(8, dtype=float)
    v = 1 + t * 1e8
    v[0] = 2.0
    result = _trend_outlier_test(t, v, False)
    assert result is not None and result.adjusted_p[0] < 0.05
    assert result.fitted[0] == pytest.approx(17 / 12, abs=1e-8)
    assert result.relative_deviation[0] == pytest.approx(7 / 17, abs=1e-8)
    r = run_rule(
        "TMP-003", {"s": pl.DataFrame({"t": t, "v": v})}, series_params(log_transform=False)
    )
    assert r.status is Status.FAIL and r.affected_count == 1


@pytest.mark.parametrize("offset", [-1e12, 0.0, 1e12])
@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_trend_statistics_invariant_to_time_translation_and_value_units(offset, scale):
    t = np.arange(40, dtype=float)
    v = 5 + t * 0.2 + np.random.default_rng(22).normal(size=40)
    v[12] += 5
    ref = _trend_outlier_test(t, v, False)
    got = _trend_outlier_test(t + offset, v * scale, False)
    assert ref is not None and got is not None
    np.testing.assert_allclose(got.residuals, ref.residuals, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got.adjusted_p, ref.adjusted_p, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("period", [2**60 + 1, Decimal("1.000000000000000001")])
def test_expected_periods_preserve_exact_numeric_identity(period):
    f = pl.DataFrame({"t": [period], "v": [1]})
    assert (
        run_rule("TMP-004", {"s": f}, series_params(expected_periods=[period])).status
        is Status.PASS
    )


def test_fractional_expected_group_is_not_truncated_to_existing_integer():
    f = pl.DataFrame({"g": [1, 1], "t": [0, 1], "v": [1, 1]})
    p = {
        "series": [{"dataset": "s", "value": "v", "time_column": "t", "group_by": ["g"]}],
        "expected_groups": [{"g": 1.9}],
    }
    assert run_rule("TMP-004", {"s": f}, p).status is Status.ERROR


@pytest.mark.parametrize("value,tolerance", [(Decimal("0.0"), 0.0), (Decimal("1.05"), 0.1)])
def test_decimal_stale_ignore_values_use_shared_exact_tolerance(value, tolerance):
    f = pl.DataFrame({"t": [0, 1, 2], "v": [value] * 3})
    ignore = 0 if value == 0 else 1
    r = run_rule("TMP-005", {"s": f}, series_params(ignore_values=[ignore], atol=tolerance))
    assert r.status is Status.PASS


@pytest.mark.parametrize("magnitude", [2**63 - 1, 2**110])
def test_cumulative_evidence_subtraction_is_arbitrary_precision(magnitude):
    f = pl.DataFrame({"t": [0, 1], "v": pl.Series([magnitude, -magnitude], dtype=pl.Int128)})
    r = run_rule("TMP-006", {"s": f}, series_params())
    assert r.status is Status.FAIL and r.affected_records[0]["decrease"] == 2 * magnitude


TEMPORAL_CASES = [
    ("TMP-001", [100.0, 1.0], {}),
    ("TMP-002", [3.0, 2.0, 1.0], {"expected_trend": "increasing"}),
    ("TMP-003", [1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0, 8.0], {}),
    ("TMP-005", [1.0, 1.0, 1.0], {}),
    ("TMP-006", [2.0, 1.0], {}),
    ("TMP-008", [1.0] * 4 + [10.0] * 4, {}),
]


@pytest.mark.parametrize("rule,values,extra", TEMPORAL_CASES)
@pytest.mark.parametrize("bad_first", [False, True])
def test_independent_temporal_failure_survives_malformed_contract_in_either_order(
    rule, values, extra, bad_first
):
    frames = {
        "good": pl.DataFrame({"t": range(len(values)), "v": values}),
        "text": pl.DataFrame({"t": [0, 1], "v": ["a", "b"]}),
    }
    specs = [
        {"dataset": name, "value": "v", "time_column": "t", **extra} for name in ("good", "text")
    ]
    if bad_first:
        specs.reverse()
    results = rule_results(rule, frames, {"series": specs})
    assert any(r.status is Status.ERROR for r in results)
    assert any(r.status is Status.FAIL for r in results)


@pytest.mark.parametrize("rule", [CoordinateRule, ColumnOutlierRule])
@pytest.mark.parametrize("limit", [0, 2])
def test_wide_evidence_attached_only_to_bounded_sample(rule, limit):
    data = (
        {"latitude": [91.0] * 100, "longitude": [0.0] * 100}
        if rule is CoordinateRule
        else {"v": [1.0] * 800 + [100.0] * 100}
    )
    n = len(next(iter(data.values())))
    f = pl.DataFrame({**data, **{f"unrelated_{i}": list(range(n)) for i in range(20)}})
    params = (
        {"points": [{"dataset": "s"}]}
        if rule is CoordinateRule
        else {"targets": [{"dataset": "s", "value": "v"}]}
    )
    ctx = DataContext(frames={"s": f}, evidence_limit=limit)
    try:
        findings = rule(params).evaluate(ctx)
    finally:
        ctx.close()
    failure = next(x for x in findings if x.assessed and not x.passed)
    assert failure.affected_count == 100
    assert len(failure.affected) == limit


def test_log_rounding_cannot_certify_distinct_high_values_clean():
    values = [1e300] * 7 + [np.nextafter(1e300, math.inf)]
    r = run_rule(
        "STA-001",
        {"s": pl.DataFrame({"v": values})},
        {"targets": [{"dataset": "s", "value": "v", "transform": "log"}]},
    )
    assert r.status is Status.ERROR
    assert _trend_outlier_test(np.arange(8), np.array(values), True) is None


def test_exact_level_shift_budget_is_shared_by_groups():
    f = pl.DataFrame(
        {"g": ["a"] * 8 + ["b"] * 8, "t": list(range(8)) * 2, "v": ([1.0] * 4 + [10.0] * 4) * 2}
    )
    p = {
        "series": [{"dataset": "s", "value": "v", "time_column": "t", "group_by": ["g"]}],
        "max_work": 8,
    }
    results = rule_results("TMP-008", {"s": f}, p)
    assert any(r.status is Status.ERROR and "budget" in r.message for r in results)
    assert any(r.status is Status.FAIL and r.affected_count == 1 for r in results)


@pytest.mark.parametrize("method", ["iqr", "modified_zscore"])
def test_decimal_array_and_frame_share_exact_float_precision_contract(method):
    values = [Decimal("0.1")] * 7 + [Decimal("10.1")]
    with pytest.raises(ValueError, match="exactly"):
        detect_outliers(values, method)
    with pytest.raises(ValueError, match="exactly"):
        flag_outliers(pl.DataFrame({"v": values}), "v", method=method)


@pytest.mark.parametrize("rule", ["TMP-007", "TMP-008"])
def test_numerically_untested_temporal_group_has_no_pass_or_assessed_count(rule):
    values = [1e-300] * 6 + [-1e-290, 1e308]
    f = pl.DataFrame({"t": np.arange(8) * 12, "v": values})
    results = rule_results(rule, {"s": f}, series_params())
    assert all(r.status is Status.ERROR for r in results)
    assert all(r.total_records in (None, 0) for r in results)


def test_numeric_configuration_preserves_binary_float_identity():
    from qavf.rules.geo_checks import BoundingBox
    from qavf.rules.temporal_checks import MissingPeriodParams, StaleValueParams

    point = 0.1
    periods = MissingPeriodParams(expected_periods=[point])
    ignored = StaleValueParams(ignore_values=[point])
    box = BoundingBox(min_lat=0, max_lat=point, min_lon=0, max_lon=1)
    assert type(periods.expected_periods[0]) is float
    assert type(ignored.ignore_values[0]) is float
    assert type(box.max_lat) is float
    frame = pl.DataFrame({"t": [point], "v": [1]})
    assert (
        run_rule("TMP-004", {"s": frame}, series_params(expected_periods=[point])).status
        is Status.PASS
    )
    coordinates = pl.DataFrame({"latitude": [point], "longitude": [0.5]})
    result = run_rule(
        "GEO-001", {"s": coordinates}, {"points": [{"dataset": "s", "bbox": box.model_dump()}]}
    )
    assert result.status is Status.PASS


@pytest.mark.parametrize("uncertain_first", [True, False])
def test_independent_season_failure_survives_uncertain_season_in_either_order(uncertain_first):
    uncertain = [1e-300] * 6 + [1e-290, 1e308]
    outlier = [1.0] * 7 + [100.0]
    first, second = (uncertain, outlier) if uncertain_first else (outlier, uncertain)
    values = [value for pair in zip(first, second, strict=True) for value in pair]
    frame = pl.DataFrame({"t": range(16), "v": values})
    results = rule_results("TMP-007", {"s": frame}, series_params(season_length=2))
    assert any(r.status is Status.ERROR for r in results)
    failure = next(r for r in results if r.status is Status.FAIL)
    assert failure.affected_count == 1 and failure.total_records == 8


def test_boolean_expectation_cannot_become_numeric_period_one():
    from pydantic import ValidationError

    from qavf.rules.temporal_checks import MissingPeriodParams

    with pytest.raises(ValidationError):
        MissingPeriodParams(expected_periods=[True])
