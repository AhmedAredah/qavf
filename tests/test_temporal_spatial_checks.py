"""Temporal (TMP-*), spatial (SPA-*) and statistical (STA-*) rules on the demo data."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import polars as pl
import pytest

from qavf.demo import inject_defects
from qavf.engine.severity import Status
from qavf.testing import run_rule

COUNTS = {"dataset": "counts", "value": "aadt", "group_by": ["site_id"], "time_column": "year"}
CRASH_RATE = {
    "dataset": "crashes",
    "value": "crashes",
    "entity_column": "site_id",
    "group_by": ["year"],
    "normalize_by": {"dataset": "counts", "value": "aadt"},
}
REGIONS = {
    "North": ["S01", "S02", "S03", "S04", "S05", "S06"],
    "South": ["S07", "S08", "S09", "S10", "S11", "S12"],
}
CONFIGURED = {
    "TMP-001": {"series": [COUNTS], "max_change_rate": 0.15},
    "TMP-002": {"series": [{**COUNTS, "expected_trend": "increasing"}]},
    "TMP-003": {"series": [COUNTS]},
    "TMP-004": {"series": [COUNTS]},
    "SPA-001": {"metrics": [CRASH_RATE]},
    "SPA-002": {"metrics": [CRASH_RATE], "groups": REGIONS, "max_ratio": 1.6},
    "SPA-003": {"entity_column": "site_id", "time_column": "year", "reference_dataset": "sites"},
    "STA-001": {
        "targets": [{"dataset": "counts", "value": "truck_share", "group_by": ["year"]}],
        "threshold": 3.0,
    },
}


HAS_STATS = (
    importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("statsmodels") is not None
)
needs_stats = pytest.mark.skipif(not HAS_STATS, reason="needs the qavf[stats] extra")
STATS_RULES = {"TMP-002", "TMP-003"}


@pytest.mark.parametrize(
    "rule_id",
    [pytest.param(r, marks=needs_stats) if r in STATS_RULES else r for r in sorted(CONFIGURED)],
)
def test_clean_demo_passes(rule_id: str, demo, registry) -> None:
    result = run_rule(rule_id, demo, CONFIGURED[rule_id], registry=registry)
    assert result.status is Status.PASS, result.message


def _series(values: list[float], times: list[int] | None = None) -> dict[str, pl.DataFrame]:
    times = times or list(range(2020, 2020 + len(values)))
    return {"t": pl.DataFrame({"year": times, "g": ["a"] * len(values), "v": values})}


def _params(**extra) -> dict:
    return {"series": [{"dataset": "t", "value": "v", "group_by": ["g"], **extra}]}


# ------------------------------------------------------------------- temporal
def test_step_change_annualises_multi_period_steps() -> None:
    frames = _series([100.0, 121.0], times=[2020, 2022])  # +10 %/yr compounded
    assert run_rule("TMP-001", frames, {**_params(), "max_change_rate": 0.12}).status is Status.PASS
    result = run_rule("TMP-001", frames, {**_params(), "max_change_rate": 0.08})
    record = result.affected_records[0]
    assert record["change_rate"] == pytest.approx(0.10)
    assert (record["previous_period"], record["year"]) == (2020, 2022)
    assert (
        run_rule("TMP-001", _series([100.0, 105.0]), _params(max_change_rate=0.01)).status
        is Status.FAIL
    )


def test_step_change_flags_drops_and_steps_from_zero() -> None:
    result = run_rule("TMP-001", _series([100.0, 40.0, 0.0, 5.0]), _params())
    assert [r["year"] for r in result.affected_records] == [2021, 2022, 2023]
    assert result.affected_records[2]["change_rate"] is None
    quiet = run_rule("TMP-001", _series([100.0, 40.0, 0.0, 5.0]), {**_params(), "min_value": 1000})
    assert quiet.status is Status.PASS


def test_step_change_on_demo_defect(demo) -> None:
    result = run_rule("TMP-001", inject_defects(demo, ["aadt_spike"]), CONFIGURED["TMP-001"])
    assert {(r["site_id"], r["year"]) for r in result.affected_records} == {
        ("S03", 2021),
        ("S03", 2022),
    }


@needs_stats
def test_trend_direction(demo) -> None:
    falling = _series([10.0, 9.0, 8.5, 7.0, 6.0])
    result = run_rule("TMP-002", falling, _params(expected_trend="increasing"))
    assert result.status is Status.FAIL and result.affected_records[0]["kendall_tau"] == -1.0
    assert run_rule("TMP-002", falling, _params(expected_trend="decreasing")).status is Status.PASS
    assert run_rule("TMP-002", falling, _params()).status is Status.SKIPPED
    noisy = _series([10.0, 12.0, 11.0, 13.0, 15.0])
    assert run_rule("TMP-002", noisy, _params(expected_trend="increasing")).status is Status.PASS
    declining = run_rule("TMP-002", inject_defects(demo, ["declining_site"]), CONFIGURED["TMP-002"])
    assert [r["site_id"] for r in declining.affected_records] == ["S08"]


@needs_stats
def test_trend_outlier_needs_significance_and_materiality() -> None:
    rng = np.random.default_rng(3)
    years = list(range(2020, 2032))
    trend = [1000.0 * 0.97 ** (y - 2020) * (1 + rng.normal(0, 0.002)) for y in years]
    assert run_rule("TMP-003", _series(trend, years), _params()).status is Status.PASS

    spiked = list(trend)
    spiked[6] *= 1.4
    (record,) = run_rule("TMP-003", _series(spiked, years), _params()).affected_records
    assert (
        record["year"] == 2026
        and record["relative_deviation"] > 0.3
        and record["adjusted_p"] < 0.05
    )

    blip = list(trend)
    blip[6] *= 1.03  # detectable on this near-perfect trend, but immaterial
    assert run_rule("TMP-003", _series(blip, years), _params()).status is Status.PASS
    sensitive = {**_params(), "min_relative_deviation": 0.0}
    assert run_rule("TMP-003", _series(blip, years), sensitive).status is Status.FAIL


@needs_stats
def test_trend_outlier_ignores_short_and_perfect_series() -> None:
    assert run_rule("TMP-003", _series([1.0, 2.0, 50.0]), _params()).total_records == 0
    perfect = _series([float(2 * i + 1) for i in range(8)])
    assert run_rule("TMP-003", perfect, {**_params(), "log_transform": False}).status is Status.PASS


def test_trend_rules_explain_the_missing_stats_extra(monkeypatch) -> None:
    for module in ("scipy", "scipy.stats", "statsmodels", "statsmodels.api"):
        monkeypatch.setitem(sys.modules, module, None)
    result = run_rule("TMP-003", _series([1.0, 2.0, 3.0, 4.0, 5.0]), _params())
    assert result.status is Status.ERROR and "qavf[stats]" in result.message
    direction = run_rule("TMP-002", _series([1.0, 2.0, 3.0]), _params(expected_trend="increasing"))
    assert direction.status is Status.ERROR and "qavf[stats]" in direction.message


def test_missing_periods(demo) -> None:
    result = run_rule("TMP-004", inject_defects(demo, ["missing_count"]), CONFIGURED["TMP-004"])
    assert result.affected_records == [
        {"series": "counts.aadt", "site_id": "S11", "missing_periods": [2022]}
    ]
    everywhere = run_rule("TMP-004", demo, {**CONFIGURED["TMP-004"], "expected_periods": [2015]})
    assert everywhere.affected_count == 12


# -------------------------------------------------------------------- spatial
def test_entity_outlier_on_normalised_metric(demo) -> None:
    result = run_rule("SPA-001", inject_defects(demo, ["aadt_spike"]), CONFIGURED["SPA-001"])
    assert {(r["site_id"], r["year"]) for r in result.affected_records} == {("S03", 2021)}
    assert result.affected_records[0]["relative_deviation"] < -0.5  # crash rate collapsed


@pytest.mark.parametrize("method", ["zscore", "modified_zscore", "iqr"])
def test_entity_outlier_methods_and_thresholds(method: str) -> None:
    frame = pl.DataFrame(
        {
            "site": list(range(12)),
            "year": [2020] * 12,
            "x": [*([10.0, 10.5, 9.5, 10.2] * 3)[:11], 40.0],
        }
    )
    params = {
        "metrics": [{"dataset": "t", "value": "x", "entity_column": "site", "group_by": ["year"]}],
        "method": method,
    }
    assert [r["site"] for r in run_rule("SPA-001", {"t": frame}, params).affected_records] == [11]
    relaxed = run_rule("SPA-001", {"t": frame}, {**params, "threshold": 1000})
    # No z-score in 12 values can reach 1000, so that test is not assessable at all.
    assert relaxed.status is (Status.ERROR if method == "zscore" else Status.PASS)
    # Too few entities to compare: configured but not assessable.
    assert run_rule("SPA-001", {"t": frame.head(4)}, params).status is Status.ERROR


def test_entity_outlier_ignores_immaterial_spread() -> None:
    frame = pl.DataFrame(
        {"e": list(range(8)), "x": [100.0, 100.01, 99.99, 100.0, 100.02, 99.98, 100.0, 101.0]}
    )
    params = {"metrics": [{"dataset": "t", "value": "x", "entity_column": "e"}]}
    assert run_rule("SPA-001", {"t": frame}, params).status is Status.PASS
    exact = run_rule("SPA-001", {"t": frame}, {**params, "min_relative_deviation": 0})
    assert [r["e"] for r in exact.affected_records] == [7]


def test_group_comparison(demo) -> None:
    result = run_rule("SPA-002", inject_defects(demo, ["aadt_spike"]), CONFIGURED["SPA-002"])
    assert {(r["site_id"], r["group"]) for r in result.affected_records} == {("S03", "North")}
    south_only = {**CONFIGURED["SPA-002"], "groups": {"South": REGIONS["South"]}}
    assert (
        run_rule("SPA-002", inject_defects(demo, ["aadt_spike"]), south_only).status is Status.PASS
    )


def test_coverage(demo) -> None:
    result = run_rule("SPA-003", inject_defects(demo, ["missing_count"]), CONFIGURED["SPA-003"])
    assert result.affected_records == [
        {"dataset": "counts", "year": 2022, "missing_entities": ["S11"]}
    ]
    explicit = run_rule(
        "SPA-003",
        demo,
        {"entity_column": "site_id", "expected_entities": ["S01", "S99"], "datasets": ["sites"]},
    )
    assert explicit.affected_records == [{"dataset": "sites", "missing_entities": ["S99"]}]
    assert run_rule("SPA-003", demo, {}).status is Status.SKIPPED


# ---------------------------------------------------------------- statistical
def test_column_outliers_per_target_settings() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(50, 2, size=40).tolist()
    values[5] = 500.0
    frame = pl.DataFrame({"g": ["a"] * 20 + ["b"] * 20, "x": values, "id": list(range(40))})
    target = {"dataset": "t", "value": "x", "group_by": ["g"]}

    result = run_rule("STA-001", {"t": frame}, {"targets": [target], "threshold": 3.0})
    (record,) = result.affected_records
    assert record["__qavf_row"] == 5 and record["method"] == "iqr" and result.total_records == 40
    for method in ("zscore", "modified_zscore"):
        hit = run_rule("STA-001", {"t": frame}, {"targets": [{**target, "method": method}]})
        assert [r["__qavf_row"] for r in hit.affected_records] == [5], method
    assert (
        run_rule("STA-001", {"t": frame}, {"targets": [{**target, "threshold": 500}]}).status
        is Status.PASS
    )
    small = run_rule("STA-001", {"t": frame}, {"targets": [target], "min_group_size": 30})
    assert small.status is Status.ERROR and small.total_records == 0  # nothing testable
