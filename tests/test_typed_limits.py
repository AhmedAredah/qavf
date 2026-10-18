"""Typed limits, unassessed evaluation problems and parameters.

Categorical and Enum text, evaluation problems reported as unassessed rather
than as violations, isolation of checks that fail at runtime, Decimal
tolerances and validation of comparison and statistical parameters.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Severity, Status
from qavf.regression.compare_outputs import compare_frames
from qavf.rules.input_checks import _column_eager, _column_values
from qavf.rules.regression_checks import BaselineParams, BaselineSpec
from qavf.rules.spatial_checks import GroupComparisonParams
from qavf.rules.statistical_checks import ColumnOutlierParams
from qavf.rules.temporal_checks import StepChangeParams
from qavf.settings import QAVFSettings
from qavf.stats import (
    detect_outliers,
    flag_outliers,
    iqr_outliers,
    modified_zscore_outliers,
    zscore_outliers,
)


def _results(registry, frames, datasets) -> dict[str | None, Any]:
    settings = QAVFSettings.model_validate(
        {"datasets": datasets, "plugins": {"entry_points": False}}
    )
    report = QARunner(settings, registry).run(frames=frames)
    return {r.check: r for r in report.results if r.rule_id == "INP-003"}


# =========================================================================
# Categorical and Enum columns get the same text checks as strings
# =========================================================================
TEXT_DTYPES = [pl.String, pl.Categorical, pl.Enum(["AA", "BB", "c"])]


@pytest.mark.parametrize("dtype", TEXT_DTYPES, ids=str)
@pytest.mark.parametrize(
    ("limits", "bad"),
    [
        ({"regex": "[A-Z]{2}"}, 1),
        ({"min_length": 2}, 1),
        ({"max_length": 1}, 2),
        ({"isin": ["AA", "BB"]}, 1),
    ],
)
def test_text_limits_apply_to_every_string_dtype(registry, dtype, limits, bad) -> None:
    frame = pl.DataFrame({"s": ["AA", "BB", "c"]}, schema={"s": dtype})
    results = _results(
        registry, {"t": frame}, {"t": {"schema": {"columns": {"s": {"dtype": "str", **limits}}}}}
    )
    failed = [r for r in results.values() if r.failed]
    assert [r.status for r in failed] == [Status.FAIL]
    assert failed[0].affected_count == bad and failed[0].unit == "rows"


# =========================================================================
# A check that cannot be evaluated is unassessed, never a violation
# =========================================================================
def test_limit_incompatible_with_the_data_type_is_unassessed(registry) -> None:
    frame = pl.DataFrame({"a": ["1", "2"]})
    results = _results(
        registry,
        {"t": frame},
        {"t": {"schema": {"columns": {"a": {"checks": [{"ge": 0, "severity": "INFO"}]}}}}},
    )
    [result] = [r for r in results.values() if r.failed]
    assert result.status is Status.ERROR and "Not assessed" in result.message
    assert "cannot apply" in result.message
    assert result.severity >= Severity.ERROR  # never inherits the check's INFO


def test_aware_bound_on_a_naive_column_is_unassessed(registry) -> None:
    frame = pl.DataFrame({"t": [datetime(2024, 1, 1)]})
    results = _results(
        registry,
        {"t": frame},
        {
            "t": {
                "schema": {
                    "columns": {
                        "t": {
                            "checks": [
                                {"ge": datetime(2020, 1, 1, tzinfo=UTC), "severity": "WARNING"}
                            ]
                        }
                    }
                }
            }
        },
    )
    [result] = [r for r in results.values() if r.failed]
    assert result.status is Status.ERROR and result.severity >= Severity.ERROR
    assert "time zone" in result.message


@pytest.mark.parametrize("bound", [date(2020, 1, 1), datetime(2020, 1, 1)])
def test_naive_bounds_are_read_in_the_column_time_zone(registry, bound) -> None:
    aware = pl.DataFrame({"t": [datetime(2024, 1, 1, tzinfo=UTC)]}).with_columns(
        pl.col("t").dt.convert_time_zone("America/Chicago")
    )
    results = _results(registry, {"t": aware}, {"t": {"schema": {"columns": {"t": {"ge": bound}}}}})
    assert not any(r.failed for r in results.values())
    late = {"t": {"schema": {"columns": {"t": {"ge": datetime(2023, 12, 31, 18, 1)}}}}}
    [failed] = [r for r in _results(registry, {"t": aware}, late).values() if r.failed]
    assert failed.status is Status.FAIL  # 2024-01-01 00:00 UTC is 2023-12-31 18:00 in Chicago


def test_aware_bound_compares_instants_across_zones(registry) -> None:
    aware = pl.DataFrame({"t": [datetime(2024, 1, 1, tzinfo=UTC)]}).with_columns(
        pl.col("t").dt.convert_time_zone("America/Chicago")
    )
    ok = {"t": {"schema": {"columns": {"t": {"ge": datetime(2024, 1, 1, tzinfo=UTC)}}}}}
    assert not any(r.failed for r in _results(registry, {"t": aware}, ok).values())


# =========================================================================
# One check failing at runtime never cancels its siblings
# =========================================================================
def test_runtime_failure_of_a_column_statistic_is_isolated(registry, monkeypatch) -> None:
    real = _column_values._statistic

    def flaky(series, statistic):
        if statistic == "mean":
            raise RuntimeError("backend exploded")
        return real(series, statistic)

    monkeypatch.setattr(_column_values, "_statistic", flaky)
    frame = pl.DataFrame({"a": [-1.0, 2.0], "b": [1.0, 2.0]})
    results = _results(
        registry,
        {"t": frame},
        {"t": {"schema": {"columns": {"a": {"ge": 0}, "b": {"mean": {"ge": 0}}}}}},
    )
    assert results["t.a: ge 0"].status is Status.FAIL
    blocked = (
        results["t.b: mean ≥ 0"]
        if "t.b: mean ≥ 0" in results
        else next(r for k, r in results.items() if k and k.startswith("t.b"))
    )
    assert blocked.status is Status.ERROR and "backend exploded" in blocked.message


def test_runtime_failure_inside_row_validation_is_isolated(registry, monkeypatch) -> None:
    real = _column_eager.row_tests

    def poisoned(constraints, dtype):
        tests = real(constraints, dtype)
        if "le" in tests:
            label, _ = tests["le"]

            def boom(column):
                raise RuntimeError("predicate exploded")

            tests["le"] = (label, boom)
        return tests

    monkeypatch.setattr(_column_eager, "row_tests", poisoned)
    frame = pl.DataFrame({"a": [-1.0, 2.0], "b": [1.0, 2.0]})
    results = _results(
        registry,
        {"t": frame},
        {"t": {"schema": {"columns": {"a": {"ge": 0}, "b": {"le": 10}}}}},
    )
    assert results["t.a: ge 0"].status is Status.FAIL
    assert results["t.b: le 10"].status is Status.ERROR
    assert "predicate exploded" in results["t.b: le 10"].message


# =========================================================================
# Decimal values use the numeric tolerance, exactly
# =========================================================================
def _dec(value: str, scale: int = 5) -> pl.DataFrame:
    return pl.DataFrame(
        {"k": [1], "v": [Decimal(value)]}, schema={"k": pl.Int64, "v": pl.Decimal(30, scale)}
    )


def test_decimal_within_tolerance_matches() -> None:
    assert compare_frames(_dec("100.00000"), _dec("100.00001"), ["k"]).equal


def test_decimal_outside_tolerance_reports_exact_differences() -> None:
    result = compare_frames(_dec("100.00000"), _dec("101.00000"), ["k"])
    assert not result.equal
    assert result.max_abs_diff == pytest.approx(1.0) and result.max_rel_diff == pytest.approx(0.01)


def test_decimal_mixed_scales_and_exact_comparison() -> None:
    assert compare_frames(_dec("100.00000"), _dec("100.0000", 4), ["k"]).equal
    assert not compare_frames(_dec("100.00000"), _dec("100.00001"), ["k"], rtol=0, atol=0).equal


def test_large_decimals_are_compared_without_float_rounding() -> None:
    big = "12345678901234567890.00001"
    other = "12345678901234567890.00002"
    assert not compare_frames(_dec(big), _dec(other), ["k"], rtol=0, atol=0).equal
    assert compare_frames(_dec(big), _dec(other), ["k"], rtol=0, atol=Decimal("0.00001")).equal


def test_decimal_against_float_uses_the_tolerance() -> None:
    floats = pl.DataFrame({"k": [1], "v": [100.000001]})
    assert compare_frames(_dec("100.00000"), floats, ["k"]).equal


# =========================================================================
# Comparison and statistical parameters are validated
# =========================================================================
SAME = pl.DataFrame({"k": [1], "v": [1.0]})
CHANGED = pl.DataFrame({"k": [1], "v": [999999.0]})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rtol": -1},
        {"rtol": math.inf},
        {"atol": math.nan},
        {"atol": -1e-9},
        {"max_differences": -1},
        {"ignore_columns": ["k"]},
        {"ignore_columns": ["nope"]},
    ],
)
def test_invalid_comparison_arguments_are_rejected_before_any_work(kwargs) -> None:
    with pytest.raises(ValueError):
        compare_frames(SAME, SAME, ["k"], **kwargs)


def test_repeated_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="repeat"):
        compare_frames(SAME, SAME, ["k", "k"])


def test_ignored_column_present_on_one_side_is_fine() -> None:
    extra = SAME.with_columns(pl.lit("run-1").alias("run_id"))
    assert compare_frames(SAME, extra, ["k"], ignore_columns=["run_id"]).equal


@pytest.mark.parametrize(
    ("model", "config"),
    [
        (BaselineParams, {"rtol": math.inf}),
        (BaselineParams, {"atol": math.nan}),
        (BaselineSpec, {"dataset": "x", "path": "p", "keys": ["k"], "atol": math.inf}),
        (BaselineSpec, {"dataset": "x", "path": "p", "keys": ["k"], "ignore_columns": ["k"]}),
        (BaselineSpec, {"dataset": "x", "path": "p", "keys": ["k", "k"]}),
        (ColumnOutlierParams, {"threshold": math.inf}),
        (GroupComparisonParams, {"max_ratio": math.inf}),
        (StepChangeParams, {"max_change_rate": math.inf}),
    ],
)
def test_rule_parameters_reject_non_finite_and_contradictory_values(model, config) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(config)


VALUES = [1.0, 2.0, 3.0, 4.0, 100.0, 2.0, 3.0, 1.0]


@pytest.mark.parametrize("threshold", [math.nan, math.inf, 0.0, -1.0])
@pytest.mark.parametrize(
    "call",
    [
        lambda t: zscore_outliers(VALUES, t),
        lambda t: modified_zscore_outliers(VALUES, t),
        lambda t: iqr_outliers(VALUES, t),
        lambda t: detect_outliers(VALUES, "iqr", t),
        lambda t: flag_outliers(pl.DataFrame({"v": VALUES}), "v", method="iqr", threshold=t),
    ],
    ids=["zscore", "modified", "iqr", "detect", "flag"],
)
def test_statistical_thresholds_must_be_finite_and_positive(call, threshold) -> None:
    with pytest.raises(ValueError, match="threshold"):
        call(threshold)


def test_array_input_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="1-D"):
        zscore_outliers(np.array([VALUES, VALUES]))
    with pytest.raises(ValueError, match="1-D"):
        detect_outliers(np.array(5.0))


def test_group_size_must_be_a_non_negative_integer() -> None:
    with pytest.raises(ValueError, match="min_group_size"):
        flag_outliers(pl.DataFrame({"v": VALUES}), "v", min_group_size=-5)
