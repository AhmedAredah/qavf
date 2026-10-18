"""Exact scalar results survive streaming boundaries without whole-column reads."""

from fractions import Fraction
from math import isfinite
from statistics import stdev

import polars as pl
import pytest

from qavf._float_accumulator import ExactFloatAccumulator
from qavf.engine.context import DataContext
from qavf.numeric import stable_float_statistic
from qavf.rules.input_checks import ColumnCheckRule
from qavf.settings import QAVFSettings


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 101])
def test_exact_accumulator_matches_independent_fraction_oracle_across_partitions(chunk):
    values = [1e308, 1.0, -1e308, None, float("nan"), 5e-324, -0.0] * 13
    usable = [v for v in values if v is not None and isfinite(v)]
    expected = sum(map(Fraction, usable), Fraction())
    accumulator = ExactFloatAccumulator()
    for offset in range(0, len(values), chunk):
        accumulator.update(iter(values[offset : offset + chunk]))
    assert accumulator.total() == expected
    assert accumulator.mean() == expected / len(usable)
    assert accumulator.count == len(usable)


def test_mean_accepts_one_pass_values_without_a_second_scan():
    assert stable_float_statistic(iter([1e16, 1.0, -1e16]), "mean") == Fraction(1, 3)


@pytest.mark.parametrize("values", [[1e16, 1e16 + 2, 1e16 + 4], [1e308, -1e308], [0.0, 5e-324]])
def test_deviation_preserves_exact_variance_with_one_pass_input(values):
    assert stable_float_statistic(iter(values), "std") == stdev(values)


def test_deviation_rejects_complete_underflow_without_rescanning():
    with pytest.raises(ValueError, match="underflows"):
        stable_float_statistic(iter([0.0] * 100 + [5e-324]), "std")


@pytest.mark.parametrize("values", [[], [None, float("nan")], [1.0]])
def test_deviation_insufficient_observations_are_undefined(values):
    assert stable_float_statistic(iter(values), "std") is None


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_nonfinite_accumulator_rejects_incomplete_reduction(value):
    accumulator = ExactFloatAccumulator()
    accumulator.update([2.0])
    with pytest.raises(ValueError, match="non-finite"):
        accumulator.update([1.0, value])
    # A failed chunk cannot publish part of its contribution.
    assert accumulator.total() == 2


def native_context(tmp_path, values, check, *, markers=(), row_budget=None):
    path = tmp_path / "values.parquet"
    pl.DataFrame(
        {"x": values, "selected": [1] * len(values), "payload": ["unused" * 100] * len(values)},
        schema_overrides={"x": pl.Float64},
    ).write_parquet(path)
    source = {"type": "file", "path": str(path)}
    if row_budget is not None:
        source["row_budget"] = row_budget
    settings = QAVFSettings.model_validate(
        {
            "datasets": {
                "d": {
                    "source": source,
                    "schema": {
                        "columns": {
                            "x": {"nullable": True, "missing_values": markers, "checks": [check]}
                        }
                    },
                }
            }
        }
    )
    return DataContext.from_settings(settings)


def test_native_exact_sum_streams_only_dependency_columns(tmp_path, monkeypatch):
    values = [1e16, *([1.0] * 100_001), -1e16]
    check = {"name": "exact total", "sum": {"ge": 100_001, "le": 100_001}}
    with native_context(tmp_path, values, check) as ctx:
        from qavf.engine.sql_read import SqlReadSession

        observed = []
        batches = SqlReadSession.batches

        def bounded(self, statement, *args, **kwargs):
            for frame in batches(self, statement, *args, **kwargs):
                assert frame.columns == ["x"]
                assert frame.height < len(values)
                observed.append(frame.height)
                yield frame

        monkeypatch.setattr(SqlReadSession, "batches", bounded)
        monkeypatch.setattr(ctx, "get", lambda *a: pytest.fail("whole dataset loaded"))
        monkeypatch.setattr(ctx, "project", lambda *a: pytest.fail("whole column collected"))
        result = ColumnCheckRule().evaluate(ctx)
        assert all(f.passed for f in result), [f.message for f in result]
        assert sum(observed) == len(values)
        assert len(observed) > 1
        assert not ctx.is_materialized("d")


@pytest.mark.parametrize("scope,expected", [("selected = 1", True), ("FALSE", False)])
def test_native_sum_mean_missing_scope_and_empty_are_exact(tmp_path, scope, expected):
    check = {
        "name": "exact scoped statistics",
        "where": scope,
        "sum": {"ge": 1, "le": 1},
        "mean": {"ge": "0.333333333333333333", "le": "0.333333333333333334"},
        "max_null_fraction": 0.5,
    }
    with native_context(
        tmp_path, [1e16, 1.0, -1e16, None, float("nan"), -999.0], check, markers=[-999]
    ) as ctx:
        result = ColumnCheckRule().evaluate(ctx)
        assert all(f.assessed for f in result), [f.message for f in result]
        assert all(f.passed for f in result) is expected


def test_native_scalar_result_does_not_bypass_complete_source_row_budget(tmp_path):
    check = {"name": "empty subset", "where": "FALSE", "sum": {"ge": 0}}
    with native_context(tmp_path, [1.0] * 10, check, row_budget=3) as ctx:
        result = ColumnCheckRule().evaluate(ctx)
        assert any(not f.assessed for f in result)
        assert any("budget" in f.message.lower() for f in result)


def test_native_deviation_streams_all_batches_and_preserves_other_statistics(tmp_path, monkeypatch):
    values = [1e16, 1e16 + 2, 1e16 + 4] * 30_000 + [None, float("nan")]
    deviation = stdev(values[:-2])
    check = {
        "name": "offset-safe streamed statistics",
        "mean": {"ge": 1e16 + 2, "le": 1e16 + 2},
        "std": {"ge": deviation - 1e-12, "le": deviation + 1e-12},
        "max_null_fraction": 0.001,
    }
    with native_context(tmp_path, values, check) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *a: pytest.fail("whole dataset loaded"))
        monkeypatch.setattr(ctx, "project", lambda *a: pytest.fail("whole column collected"))
        result = ColumnCheckRule().evaluate(ctx)
        assert all(f.passed for f in result), [f.message for f in result]
        assert not ctx.is_materialized("d")


@pytest.mark.parametrize("spilled", [False, True])
def test_native_scalar_dispatch_survives_concurrent_materialization(tmp_path, monkeypatch, spilled):
    check = {
        "name": "exact scoped fallback",
        "where": "selected = 1",
        "sum": {"ge": 1, "le": 1},
        "mean": {"ge": "0.333333333333333333", "le": "0.333333333333333334"},
        "max_null_fraction": 0.5,
    }
    with native_context(tmp_path, [1e16, 1.0, -1e16, None, -999.0], check, markers=[-999]) as ctx:
        if spilled:
            ctx._frames.limit = 0
        original_consume, original_project = ctx.consume_sql, ctx.project
        attempts, projected = [], []

        def interleaved(*args, **kwargs):
            # Another rule completes an eager read after this rule dispatches as
            # native but before its streaming operation starts. No timing needed.
            assert not ctx.is_materialized("d")
            ctx.get("d")
            attempts.append(1)
            return original_consume(*args, **kwargs)

        def project(dataset, columns):
            result = original_project(dataset, columns)
            projected.append(result.columns)
            return result

        monkeypatch.setattr(ctx, "consume_sql", interleaved)
        monkeypatch.setattr(ctx, "project", project)
        results = ColumnCheckRule().evaluate(ctx)
        assert all(f.assessed and f.passed for f in results), [f.message for f in results]
        assert attempts == [1]
        assert projected == [["x", "selected"]]
        assert ctx.resource_usage()["active_connection_reservations"] <= 1
