"""Aggregation and regression tolerance contracts.

Explicit aggregation for non-additive metrics and per-column regression
tolerances with structured comparison metrics.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.rules.helpers import aggregate
from qavf.rules.spatial_checks import SpatialMetric
from qavf.rules.temporal_checks import SeriesSpec
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


def _run(frames, **config):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False, "parallel": False},
            "datasets": {name: {} for name in frames},
            **config,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


# =========================================================================
# Aggregations other than sum, with explicit missing-value semantics
# =========================================================================
GROUPS = pl.DataFrame(
    {
        "g": ["a", "a", "a", "b", "b", "c"],
        "v": [1.0, 3.0, None, 4.0, math.nan, None],
        "w": [1.0, 3.0, 5.0, 2.0, 1.0, 1.0],
        "t": [2, 1, 3, 1, 2, 1],
    }
)


@pytest.mark.parametrize(
    ("how", "extra", "expected"),
    [
        ("sum", {}, {"a": 4.0, "b": 4.0, "c": None}),
        ("mean", {}, {"a": 2.0, "b": 4.0, "c": None}),
        ("count", {}, {"a": 2.0, "b": 1.0, "c": 0.0}),
        ("min", {}, {"a": 1.0, "b": 4.0, "c": None}),
        ("max", {}, {"a": 3.0, "b": 4.0, "c": None}),
        ("first", {"order_by": "t"}, {"a": 3.0, "b": 4.0, "c": None}),
        ("last", {"order_by": "t"}, {"a": 1.0, "b": 4.0, "c": None}),
        ("weighted_mean", {"weight": "w"}, {"a": 2.5, "b": 4.0, "c": None}),
    ],
)
def test_aggregations(how, extra, expected) -> None:
    result = aggregate(GROUPS, ["g"], "v", "x", how, **extra)
    assert dict(zip(result["g"], result["x"], strict=True)) == expected


def test_weighted_mean_rejects_negative_and_zero_weights() -> None:
    frame = pl.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 5.0], "w": [1.0, -1.0, 0.0]})
    result = aggregate(frame, ["g"], "v", "x", "weighted_mean", weight="w")
    assert result["x"].to_list() == [None, None]


@pytest.mark.parametrize(
    "config",
    [
        {"aggregate": "weighted_mean"},
        {"aggregate": "first"},
        {"aggregate": "mean", "weight": "w"},
        {"aggregate": "sum", "order_by": "t"},
    ],
)
def test_aggregation_settings_are_validated(config) -> None:
    with pytest.raises(ValidationError):
        SeriesSpec.model_validate({"dataset": "s", "value": "v", **config})
    with pytest.raises(ValidationError):
        SpatialMetric.model_validate({"dataset": "s", "value": "v", "entity_column": "e", **config})


def test_a_rate_with_changing_observation_counts_is_not_a_trend_when_averaged() -> None:
    # The rate is constant (5%); only the number of observations per year grows.
    rows = [{"year": 2020 + y, "rate": 0.05} for y in range(5) for _ in range(1 + 2 * y)]
    frame = pl.DataFrame(rows)
    series = {"dataset": "s", "value": "rate"}
    as_sum = _run({"s": frame}, rules={"TMP-001": {"params": {"series": [series]}}})
    assert as_sum.get("TMP-001").status is Status.FAIL  # summing invents growth
    as_mean = _run(
        {"s": frame},
        rules={"TMP-001": {"params": {"series": [{**series, "aggregate": "mean"}]}}},
    )
    result = as_mean.get("TMP-001")
    assert result.status is Status.PASS and "(mean)" in str(result.metrics["series"])


def test_aggregate_none_requires_one_row_per_period() -> None:
    frame = pl.DataFrame({"year": [2020, 2020, 2021], "v": [1.0, 1.0, 1.0]})
    series = {"dataset": "s", "value": "v", "aggregate": "none"}
    report = _run({"s": frame}, rules={"TMP-004": {"params": {"series": [series]}}})
    [blocked] = [r for r in report.results if r.status is Status.ERROR]
    assert "aggregate: none" in blocked.message
    unique = pl.DataFrame({"year": [2020, 2021], "v": [1.0, 1.0]})
    report = _run({"s": unique}, rules={"TMP-004": {"params": {"series": [series]}}})
    assert report.get("TMP-004").status is Status.PASS


def test_spatial_metrics_and_normalisers_aggregate() -> None:
    sites = pl.DataFrame(
        {"site": list("abcde") * 2, "pm": [10.0, 11, 9, 10, 10, 10, 11, 9, 10, 10]}
    )
    activity = pl.DataFrame({"site": list("abcde"), "hours": [1.0] * 5})
    metric = {
        "dataset": "m",
        "value": "pm",
        "entity_column": "site",
        "aggregate": "mean",
        "normalize_by": {"dataset": "n", "value": "hours", "aggregate": "max"},
    }
    report = _run({"m": sites, "n": activity}, rules={"SPA-002": {"params": {"metrics": [metric]}}})
    result = report.get("SPA-002")
    assert result.status is Status.PASS and result.total_records == 5


def test_weight_and_order_columns_must_exist() -> None:
    series = {"dataset": "s", "value": "v", "aggregate": "weighted_mean", "weight": "nope"}
    report = _run(
        {"s": pl.DataFrame({"year": [2020], "v": [1.0]})},
        rules={"TMP-004": {"params": {"series": [series]}}},
    )
    [blocked] = [r for r in report.results if r.status is Status.ERROR]
    assert "nope" in blocked.message


# =========================================================================
# Per-column tolerances and structured comparison metrics
# =========================================================================
import json  # noqa: E402
import sys  # noqa: E402

import qavf.regression  # noqa: E402,F401
from qavf.rules.regression_checks import BaselineSpec  # noqa: E402

compare_module = sys.modules["qavf.regression.compare_outputs"]
comparison_execution = sys.modules["qavf.regression._comparison"]
BASE = pl.DataFrame({"k": [1, 2, 3], "share": [0.10, 0.20, 0.30], "mass": [100.0, 200.0, 300.0]})


def test_a_column_can_have_its_own_tolerance() -> None:
    candidate = BASE.with_columns(pl.col("share") + 0.0005, pl.col("mass") + 1e-12)
    loose = compare_module.compare_frames(
        BASE, candidate, ["k"], column_tolerances={"share": (0.0, 0.001)}
    )
    assert loose.equal and loose.column_tolerances == {"share": (0.0, 0.001)}
    strict = compare_module.compare_frames(BASE, candidate, ["k"])
    assert strict.values_changed == 3


def test_an_exact_column_ignores_the_table_tolerance() -> None:
    candidate = BASE.with_columns(pl.col("mass") * (1 + 1e-9))
    assert compare_module.compare_frames(BASE, candidate, ["k"]).equal
    exact = compare_module.compare_frames(
        BASE, candidate, ["k"], column_tolerances={"mass": (0.0, 0.0)}
    )
    assert exact.values_changed == 3
    assert all((d.rtol, d.atol) == (0.0, 0.0) for d in exact.differences)


@pytest.mark.parametrize(
    "tolerances",
    [{"nope": (0.0, 0.1)}, {"k": (0.0, 0.1)}, {"share": (-1.0, 0.0)}, {"share": (0.0, math.inf)}],
)
def test_column_tolerances_are_validated(tolerances) -> None:
    with pytest.raises(ValueError):
        compare_module.compare_frames(BASE, BASE, ["k"], column_tolerances=tolerances)


@pytest.mark.parametrize(
    "columns",
    [{"share": {"exact": True, "atol": 0.1}}, {"k": {"atol": 0.1}}, {"share": {"rtol": -1}}],
)
def test_baseline_column_settings_are_validated(columns) -> None:
    with pytest.raises(ValidationError):
        BaselineSpec.model_validate(
            {"dataset": "out", "path": "b.csv", "keys": ["k"], "columns": columns}
        )


@pytest.mark.parametrize("flag", [True, False])
def test_column_tolerances_apply_on_both_comparison_paths(monkeypatch, flag) -> None:
    monkeypatch.setattr(comparison_execution, "_TOLERANCE_PREFILTER", flag)
    candidate = BASE.with_columns(pl.col("share") + 0.0005, pl.col("mass") * (1 + 1e-9))
    result = compare_module.compare_frames(
        BASE, candidate, ["k"], column_tolerances={"share": (0.0, 0.001), "mass": (0.0, 0.0)}
    )
    assert {d.column for d in result.differences} == {"mass"} and result.values_changed == 3


def test_reg001_applies_column_tolerances_and_keeps_structured_metrics(tmp_path) -> None:
    path = tmp_path / "baseline.csv"
    BASE.write_csv(path)
    candidate = BASE.with_columns(pl.col("share") + 0.0005, pl.col("mass") + 5.0)
    spec = {
        "dataset": "out",
        "path": str(path),
        "keys": ["k"],
        "columns": {"share": {"atol": 0.001}, "mass": {"atol": 1.0}},
    }
    report = _run({"out": candidate}, rules={"REG-001": {"params": {"baselines": [spec]}}})
    result = report.get("REG-001")
    assert result.status is Status.FAIL and result.affected_count == 3
    comparison = result.metrics["comparisons"]["out"]
    assert comparison["values_changed"] == 3 and comparison["keys"] == ["k"]
    # Only atol was set: rtol is inherited from the baseline/rule default.
    assert comparison["column_tolerances"]["share"] == [1e-6, 0.001]
    assert comparison["max_abs_diff"] == pytest.approx(5.0) and comparison["truncated"] is False
    record = result.affected_records[0]
    assert record["column"] == "mass" and record["abs_diff"] == pytest.approx(5.0)
    assert (record["rtol"], record["atol"]) == (1e-6, 1.0)
    json.loads(report.model_dump_json())
