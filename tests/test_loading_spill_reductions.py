"""Simple reductions retain only their results when a frame cache owns IPC spill."""

from contextlib import contextmanager
from datetime import UTC, datetime

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules._execution import key_frequencies
from qavf.rules.freshness_checks import FreshnessRule
from qavf.rules.input_checks._datasets import _row_count
from qavf.rules.spatial_checks import CoverageRule
from qavf.settings import DatasetSpec


@contextmanager
def reduced_context(frame, storage, tmp_path, monkeypatch, *, columns=None, maximum=1):
    options = {"cache_max_bytes": 0} if storage == "spill" else {}
    source = frame.lazy() if storage == "lazy" else frame
    spec = DatasetSpec.model_validate({"schema": {"columns": columns or {}}})
    with DataContext(
        frames={"d": source}, specs={"d": spec}, temp_directory=tmp_path, **options
    ) as ctx:
        collected = []
        original = ctx._check_materialization

        def check_result(name, result):
            collected.append(result)
            assert result.height <= maximum
            assert "payload" not in result.columns
            return original(name, result)

        def no_full_read(*args, **kwargs):
            pytest.fail("a scalar or key reduction must not reload its complete input")

        monkeypatch.setattr(ctx, "get", no_full_read)
        monkeypatch.setattr(ctx, "project", no_full_read)
        monkeypatch.setattr(ctx, "_check_materialization", check_result)
        yield ctx
        assert collected
        if storage == "spill":
            usage = ctx.resource_usage()
            assert usage["frame_spill_bytes"] > 0
            assert usage["frame_spill_read_count"] == 0
            assert usage["frame_spill_reduce_count"] == 1


@pytest.mark.parametrize("storage", ["eager", "lazy", "spill"])
def test_row_count_is_complete_without_wide_reload(storage, tmp_path, monkeypatch):
    frame = pl.DataFrame({"payload": ["x" * 4096] * 257})
    with reduced_context(frame, storage, tmp_path, monkeypatch) as ctx:
        assert _row_count(ctx, "d") == 257


@pytest.mark.parametrize("storage", ["eager", "lazy", "spill"])
def test_key_counts_keep_duplicate_and_null_keys(storage, tmp_path, monkeypatch):
    frame = pl.DataFrame(
        {"key": ["a", None, "b", "a", None] * 100, "payload": ["x" * 4096] * 500}
    )
    with reduced_context(frame, storage, tmp_path, monkeypatch, maximum=3) as ctx:
        result = key_frequencies(ctx, "d", ["key"], "rows")
        assert result.to_dicts() == [
            {"key": "a", "rows": 200},
            {"key": None, "rows": 200},
            {"key": "b", "rows": 100},
        ]
        assert result.schema["rows"] == pl.UInt64


@pytest.mark.parametrize("storage", ["eager", "lazy", "spill"])
def test_coverage_reduces_duplicate_entities_before_collection(storage, tmp_path, monkeypatch):
    frame = pl.DataFrame(
        {
            "entity": ["a", "b", "a", "NONE"] * 100,
            "year": [2025, 2025, 2026, 2026] * 100,
            "payload": ["x" * 4096] * 400,
        }
    )
    with reduced_context(
        frame,
        storage,
        tmp_path,
        monkeypatch,
        columns={"entity": {"missing_values": ["NONE"]}},
        maximum=4,
    ) as ctx:
        [result] = CoverageRule(
            {
                "datasets": ["d"],
                "entity_column": "entity",
                "time_column": "year",
                "expected_entities": ["a", "b"],
                "expected_periods": [2025, 2026],
            }
        ).evaluate(ctx)
        assert result.assessed and not result.passed
        assert result.affected_count == 1 and result.total_records == 2
        assert result.affected == [
            {"dataset": "d", "year": 2026, "missing_entities": ["b"]}
        ]


@pytest.mark.parametrize("storage", ["eager", "lazy", "spill"])
def test_freshness_scalar_ignores_declared_marker(storage, tmp_path, monkeypatch):
    measured, marker = datetime(2026, 1, 1), datetime(2099, 1, 1)
    frame = pl.DataFrame(
        {"at": [measured, marker, None] * 100, "payload": ["x" * 4096] * 300}
    )
    with reduced_context(
        frame, storage, tmp_path, monkeypatch, columns={"at": {"missing_values": [marker]}}
    ) as ctx:
        [result] = FreshnessRule(
            {
                "now": datetime(2026, 1, 2, tzinfo=UTC),
                "checks": [{"dataset": "d", "column": "at", "max_age": "PT12H"}],
            }
        ).evaluate(ctx)
        assert result.assessed and not result.passed
        assert len(result.affected) == 1
        assert result.affected[0]["latest"] == measured
        assert result.affected[0]["age_hours"] == 24.0
