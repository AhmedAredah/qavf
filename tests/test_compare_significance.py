"""Significance in comparison levels: constructed data with known answers.

A real shift is significant and fails; noise alone is material but not
significant and passes under the default policy; a tiny shift over many values
is significant but immaterial and passes too (while ``significant`` fails it).
Measures without a sound test fall back to materiality, explicitly; tests that
cannot run are not assessed, never passed.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf import File, ResourceLimits, Status, compare_datasets, validate
from qavf.compare import DatasetComparison
from qavf.execution.os_limits import SUPPORTED_PLATFORMS

pytest.importorskip("scipy")

RNG = np.random.default_rng(20251021)


def _check(result, level, name):
    return next(c for c in result.level(level).checks if c.check == name)


def _grouped(shift: dict[str, float], n: int = 60, noise: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    groups = sorted(shift)
    baseline = pl.DataFrame(
        {"g": np.repeat(groups, n), "x": rng.normal(100.0, 5.0, n * len(groups))}
    )
    moved = np.repeat([shift[g] for g in groups], n) + rng.normal(0, noise, n * len(groups))
    candidate = baseline.with_columns(pl.col("x") + pl.Series(moved))
    return baseline, candidate


def _resampled(shift: dict[str, float], n: int = 60, seed: int = 1):
    """Candidate values drawn afresh from the baseline distribution, plus a shift."""
    rng = np.random.default_rng(seed)
    groups = sorted(shift)
    baseline = pl.DataFrame({"g": np.repeat(groups, n), "x": rng.normal(100, 5, n * len(groups))})
    candidate = pl.DataFrame(
        {
            "g": np.repeat(groups, n),
            "x": rng.normal(100, 5, n * len(groups)) + np.repeat([shift[g] for g in groups], n),
        }
    )
    return baseline, candidate


# ------------------------------------------------------------------ groups
def test_a_real_shift_fails_and_noise_passes_under_the_default_policy():
    # Every value moves a little (material), only group "a" moves systematically.
    baseline, candidate = _grouped({"a": 6.0, "b": 0.0, "c": 0.0, "d": 0.0}, noise=0.5)
    result = compare_datasets(
        baseline, candidate, levels=[{"level": "groups", "group_by": ["g"],
                                      "measures": ["mean(x)", "median(x)", "sum(x)"]}],
    )  # fmt: skip
    for label, test in (
        ("mean(x)", "welch_t"),
        ("median(x)", "mann_whitney"),
        ("sum(x)", "bootstrap"),
    ):
        check = _check(result, "groups", label)
        assert not check.passed and check.assessed
        assert [r["g"] for r in check.affected_records] == ["a"]
        record = check.affected_records[0]
        assert record["test"] == test and record["adjusted_p"] < 0.05
        assert record["adjusted_p"] >= record["p_value"]
        assert check.metrics["material"] == 4  # every group moved, only "a" significantly
        significance = check.significance
        assert significance.method == test and significance.family_size == 4
        assert (
            significance.correction == "bh" and significance.decision == "material_and_significant"
        )
        assert significance.failed == 1
        [measure] = [m for m in result.level("groups").measures if m.label == label]
        assert measure.significance == significance


def test_a_tiny_shift_over_many_values_is_significant_but_immaterial():
    baseline, candidate = _grouped({"a": 0.3}, n=20_000, noise=0.01)  # 6 standard errors
    levels = [{"level": "groups", "measures": ["mean(x)"], "rtol": 0.01}]
    default = compare_datasets(baseline, candidate, levels=levels)
    check = _check(default, "groups", "mean(x)")
    assert check.passed and check.metrics["material"] == 0
    assert check.significance.tests == []  # immaterial: no test needed
    significant = compare_datasets(baseline, candidate, levels=levels, decision="significant")
    strict = _check(significant, "groups", "mean(x)")
    assert not strict.passed and strict.significance.tests[0].significant
    material = compare_datasets(baseline, candidate, levels=levels, decision="material")
    assert _check(material, "groups", "mean(x)").passed


def test_material_policy_runs_no_test_and_reports_as_before():
    baseline, candidate = _resampled({"a": 6.0, "b": 0.0})
    levels = [{"level": "groups", "group_by": ["g"], "measures": ["mean(x)"]}]
    result = compare_datasets(baseline, candidate, levels=levels, decision="material")
    check = _check(result, "groups", "mean(x)")
    assert check.significance is None and result.level("groups").measures[0].significance is None
    assert check.message.startswith("mean(x) changed materially in 2 groups of 2")
    assert check.metrics == {}


def test_measures_without_a_sound_test_are_judged_by_materiality_explicitly():
    baseline, candidate = _resampled({"a": 0.0, "b": 0.0})
    levels = [{"level": "groups", "group_by": ["g"], "measures": ["max(x)", "distinct_count(x)"]}]
    result = compare_datasets(baseline, candidate, levels=levels)
    for label in ("max(x)", "distinct_count(x)"):
        check = _check(result, "groups", label)
        significance = check.significance
        assert not significance.applicable and significance.method == "none"
        assert "no sound test" in significance.reason
        assert "judged by materiality" in check.message
    assert not _check(result, "groups", "max(x)").passed  # new draws: maxima differ
    declined = compare_datasets(
        baseline, candidate,
        levels=[{"level": "groups", "group_by": ["g"],
                 "measures": [{"measure": "mean", "column": "x", "test": "none"}]}],
    )  # fmt: skip
    assert not _check(declined, "groups", "mean(x)").significance.applicable


def test_a_test_that_cannot_run_leaves_the_check_not_assessed():
    baseline = pl.DataFrame({"g": ["a", "b"], "x": [1.0, 2.0]})
    candidate = pl.DataFrame({"g": ["a", "b"], "x": [5.0, 2.0]})
    levels = [{"level": "groups", "group_by": ["g"], "measures": ["mean(x)"]}]
    check = _check(compare_datasets(baseline, candidate, levels=levels), "groups", "mean(x)")
    assert not check.assessed and not check.passed
    assert "at least 2 values" in check.message and check.affected_count == 1
    assert check.affected_records[0]["g"] == "a"
    budget = compare_datasets(
        *_resampled({"a": 6.0}), levels=[{"level": "groups", "measures": ["sum(x)"]}], max_work=50
    )
    assert "max_work" in _check(budget, "groups", "sum(x)").message


def test_count_and_share_measures_use_poisson_and_two_proportion_tests():
    baseline = pl.DataFrame({"g": ["a"] * 400 + ["b"] * 400 + ["c"] * 200})
    candidate = pl.DataFrame({"g": ["a"] * 520 + ["b"] * 395 + ["c"] * 205})
    levels = [{"level": "groups", "group_by": ["g"], "measures": ["count", "share"]}]
    result = compare_datasets(baseline, candidate, levels=levels)
    count, share = _check(result, "groups", "count"), _check(result, "groups", "share")
    assert [r["g"] for r in count.affected_records] == ["a"]
    assert count.affected_records[0]["test"] == "poisson"
    assert count.affected_records[0]["effect"] == pytest.approx(520 / 400)
    assert share.significance.method == "two_proportion" and not share.passed
    assert share.affected_records[0]["g"] == "a"


def test_equivalence_passes_only_when_shown():
    close = _grouped({"a": 0.02}, n=2000, noise=0.05)
    levels = [{"level": "groups", "measures": ["mean(x)"], "atol": 1.0}]
    shown = compare_datasets(*close, levels=levels, decision="equivalence")
    assert _check(shown, "groups", "mean(x)").passed
    tiny = _grouped({"a": 0.02}, n=3, noise=0.05)
    noisy = compare_datasets(*tiny, levels=levels, decision="equivalence")
    check = _check(noisy, "groups", "mean(x)")
    assert not check.passed and check.assessed  # not shown is a failure, never a pass
    counts = compare_datasets(
        pl.DataFrame({"g": ["a"] * 50}), pl.DataFrame({"g": ["a"] * 50}),
        levels=[{"level": "groups", "measures": ["count"]}], decision="equivalence",
    )  # fmt: skip
    assert not _check(counts, "groups", "count").assessed  # Poisson has no equivalence form


def test_level_settings_override_the_comparison():
    baseline, candidate = _resampled({"a": 2.0, "b": 0.0, "c": 0.0}, n=40, seed=3)
    levels = [
        {"level": "groups", "group_by": ["g"], "measures": ["mean(x)"]},
        {"level": "groups", "name": "strict", "group_by": ["g"], "measures": ["mean(x)"],
         "correction": "bonferroni", "alpha": 0.001, "decision": "significant"},
    ]  # fmt: skip
    result = compare_datasets(baseline, candidate, levels=levels)
    strict = _check(result, "strict", "mean(x)").significance
    assert (strict.correction, strict.alpha, strict.decision) == (
        "bonferroni", 0.001, "significant",
    )  # fmt: skip
    default = _check(result, "groups", "mean(x)").significance
    assert (default.correction, default.alpha) == ("bh", 0.05)


def test_resampled_p_values_are_reproducible_by_seed():
    baseline, candidate = _resampled({"a": 1.0, "b": 0.5}, n=30)
    levels = [{"level": "groups", "group_by": ["g"], "measures": ["sum(x)"]}]
    first = compare_datasets(baseline, candidate, levels=levels, seed=4)
    assert first == compare_datasets(baseline, candidate, levels=levels, seed=4)
    other = compare_datasets(baseline, candidate, levels=levels, seed=5)
    tests = [_check(r, "groups", "sum(x)").significance.tests for r in (first, other)]
    assert [t.p_value for t in tests[0]] != [t.p_value for t in tests[1]]
    assert [t.seed for t in tests[0]] == [4, 5]


@pytest.mark.parametrize(
    "measure,problem",
    [
        ({"measure": "mean", "column": "x", "test": "paired_t"}, "records level"),
        ({"measure": "count", "test": "bootstrap"}, "has no column"),
        ({"measure": "count", "column": "x", "test": "bootstrap"}, "sample statistic"),
        ({"measure": "mean", "column": "x", "test": "poisson"}, "not counts"),
        ({"measure": "mean", "column": "x", "test": "two_proportion"}, "not a share"),
        ({"measure": "mean", "column": "x", "test": "nope"}, "unknown statistical test"),
    ],
)
def test_incompatible_tests_are_configuration_errors(measure, problem):
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValidationError, match=problem):
        compare_datasets(frame, frame, levels=[{"level": "groups", "measures": [measure]}])


def test_without_scipy_the_default_is_not_assessed_and_material_still_works(monkeypatch):
    baseline, candidate = _resampled({"a": 6.0})
    for module in ("scipy", "scipy.stats"):
        monkeypatch.setitem(sys.modules, module, None)
    levels = [{"level": "groups", "measures": ["mean(x)"]}]
    check = _check(compare_datasets(baseline, candidate, levels=levels), "groups", "mean(x)")
    assert not check.assessed and "qavf[stats]" in check.message
    core = compare_datasets(baseline, candidate, levels=levels, decision="material")
    assert _check(core, "groups", "mean(x)").assessed


# ---------------------------------------------------------------- temporal
def test_temporal_buckets_are_tested_from_each_buckets_values():
    days = [date(2025, 1, 1) + timedelta(days=d) for d in range(6)]
    rows = [(day, value) for day in days for value in RNG.normal(50, 4, 80)]
    baseline = pl.DataFrame(rows, schema=["day", "x"], orient="row")
    redrawn = [
        (day, value + (8.0 if day == days[3] else 0.0))
        for day in days
        for value in RNG.normal(50, 4, 80)
    ]
    candidate = pl.DataFrame(redrawn, schema=["day", "x"], orient="row")
    levels = [{"level": "temporal", "time_column": "day", "bucket": "day", "measures": ["mean(x)"]}]
    result = compare_datasets(baseline, candidate, levels=levels)
    check = _check(result, "temporal", "mean(x)")
    assert [r["bucket"] for r in check.affected_records] == [days[3]]
    assert check.significance.family_size == 6 and check.metrics["material"] == 6
    assert result.level("temporal").measures[0].significance.failed == 1


# ----------------------------------------------------------------- records
def test_records_shift_check_tests_each_numeric_column():
    n = 300
    base = pl.DataFrame(
        {"k": range(n), "g": ["a", "b", "c"] * 100, "x": RNG.normal(10, 2, n),
         "y": RNG.normal(5, 1, n)}
    )  # fmt: skip
    cand = base.with_columns(
        pl.col("x") + 0.4 + pl.Series(RNG.normal(0, 0.3, n)),
        pl.col("y") + pl.Series(RNG.normal(0, 0.3, n)),
    )
    result = compare_datasets(base, cand, keys="k")
    records = result.level("records")
    shift = _check(result, "records", "shift")
    assert not shift.passed and records.shifted == ["x"]
    assert [r["column"] for r in shift.affected_records] == ["x"]
    assert shift.affected_records[0]["effect"] == pytest.approx(0.4, abs=0.1)
    assert not _check(result, "records", "values").passed  # every value still changed
    by_column = {c.column: c.significance for c in records.columns}
    assert by_column["x"].failed == 1 and by_column["y"].failed == 0
    assert records.significance.family_size == 2
    grouped = compare_datasets(
        base, cand, keys="k", levels=[{"level": "records", "shift_by": ["g"], "test": "wilcoxon"}]
    )
    assert len(_check(grouped, "records", "shift").affected_records) == 3
    plain = compare_datasets(base, cand, keys="k", decision="material")
    assert [c.check for c in plain.level("records").checks] == ["rows", "values"]
    with pytest.raises(ValidationError, match="paired test"):
        compare_datasets(base, cand, keys="k", levels=[{"level": "records", "test": "welch_t"}])


# ------------------------------------------------------------ series_shape
def _series(slope_b: float, slope_c: float, n: int = 48, seed: int = 2, lag: int = 0):
    rng = np.random.default_rng(seed)
    days = [date(2025, 1, 1) + timedelta(days=d) for d in range(n)]
    noise = rng.normal(0, 1, n + 1)
    signal = np.sin(np.arange(n + 1) / 3) * 10
    base = signal[1:] + slope_b * np.arange(n) + noise[1:]
    cand = (signal[1 - lag : n + 1 - lag] if lag else signal[1:]) + slope_c * np.arange(n)
    cand = cand + rng.normal(0, 1, n)
    return (
        pl.DataFrame({"day": days, "x": base}),
        pl.DataFrame({"day": days, "x": cand}),
    )


SHAPE = {"level": "series_shape", "time_column": "day", "bucket": "day", "measure": "sum(x)",
         "peak": None, "level_shift": None}  # fmt: skip


def test_series_shape_trend_needs_a_significant_slope_difference():
    steeper = compare_datasets(*_series(0.0, 0.5), levels=[SHAPE])
    trend = _check(steeper, "series_shape", "trend")
    assert not trend.passed and trend.affected_records[0]["test"] == "slope_difference"
    assert trend.significance.method == "slope_difference"
    [series] = steeper.level("series_shape").series
    assert {t.label for t in series.significance} >= {"trend"}
    noisy = compare_datasets(
        *_series(0.02, 0.03), levels=[{**SHAPE, "trend": {"rtol": 0.1}}]
    )  # material change of tiny slopes, not significant
    assert _check(noisy, "series_shape", "trend").passed


def test_series_shape_lag_and_correlation_are_tested():
    shifted = compare_datasets(*_series(0.0, 0.0, lag=1), levels=[SHAPE])
    lag = _check(shifted, "series_shape", "lag")
    assert not lag.passed and lag.affected_records[0]["test"] == "lag"
    unrelated = pl.DataFrame({"day": _series(0, 0)[0]["day"], "x": RNG.normal(0, 1, 48)})
    result = compare_datasets(_series(0, 0)[0], unrelated, levels=[SHAPE])
    correlation = _check(result, "series_shape", "correlation")
    assert not correlation.passed and correlation.affected_records[0]["test"] == "fisher_z"
    assert correlation.significance.tests[0].dependence == "effective_n"
    assessments = {a.check: a for a in result.level("series_shape").assessments}
    assert assessments["correlation"].significance == correlation.significance


# ------------------------------------------------------------- CMP-001, JSON
def _cmp001(baseline: str, candidate: str, **comparison):
    return {
        "datasets": {"v1": {}, "v2": {}},
        "rules": {
            "CMP-001": {
                "params": {
                    "comparisons": [
                        {
                            "name": "means", "baseline": baseline, "candidate": candidate,
                            "levels": [{"level": "groups", "group_by": ["g"],
                                        "measures": ["mean(x)"]}],
                            **comparison,
                        }
                    ]
                }
            }
        },
        "engine": {"run_unconfigured_rules": False},
    }  # fmt: skip


def test_cmp001_reports_significance_through_validate():
    baseline, candidate = _resampled({"a": 6.0, "b": 0.0})
    report = validate({"v1": baseline, "v2": candidate}, config=_cmp001("v1", "v2"))
    [result] = [r for r in report.results_for("CMP-001") if r.check == "means: groups: mean(x)"]
    assert result.status is Status.FAIL
    assert result.metrics["test"] == "welch_t" and result.metrics["family_size"] == 2
    assert [r["g"] for r in result.affected_records] == ["a"]
    assert "adjusted_p" in result.affected_records[0]


def test_results_round_trip_and_isolated_runs_agree(tmp_path):
    baseline, candidate = _resampled({"a": 6.0, "b": 0.0})
    baseline.write_parquet(tmp_path / "v1.parquet")
    candidate.write_parquet(tmp_path / "v2.parquet")
    levels = [{"level": "groups", "group_by": ["g"], "measures": ["mean(x)", "sum(x)"]}]
    local = compare_datasets(tmp_path / "v1.parquet", tmp_path / "v2.parquet", levels=levels)
    again = DatasetComparison.model_validate_json(local.model_dump_json())
    assert again == local
    if sys.platform not in SUPPORTED_PLATFORMS:
        return
    isolated = compare_datasets(
        File(tmp_path / "v1.parquet"),
        File(tmp_path / "v2.parquet"),
        levels=levels,
        resources=ResourceLimits(isolate=True, wall_time=120),
    )
    assert isolated.model_dump(exclude={"execution_limits"}) == local.model_dump(
        exclude={"execution_limits"}
    )


def test_series_shape_tests_that_cannot_run_are_not_assessed():
    baseline, candidate = _series(0.0, 0.5)
    sites = [pl.lit("a").alias("site")]
    levels = [{**SHAPE, "group_by": ["site"], "max_work": 10}]
    result = compare_datasets(
        baseline.with_columns(sites), candidate.with_columns(sites), levels=levels
    )
    trend = _check(result, "series_shape", "trend")
    assert not trend.assessed and "max_work" in trend.message
    assert trend.affected_records[0]["site"] == "a"
    assert trend.significance.tests[0].label == "site=a"
