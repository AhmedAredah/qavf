"""Reduced materialization preserves complete temporal and reconciliation contracts."""

from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules._grouped import grouped_aggregate, normalized_relation
from qavf.rules.freshness_checks import FreshnessRule
from qavf.rules.reconciliation_checks import SharesSumRule
from qavf.rules.temporal_checks import CumulativeRule
from qavf.settings import QAVFSettings


def context(frame, backend, tmp_path, *, columns=None, evidence=2):
    spec = {"schema": {"columns": columns or {}}}
    frames = None
    if backend == "eager":
        frames = {"d": frame}
    else:
        if backend == "lazy":
            # A typed local file is a verified repeatable lazy source. Arbitrary
            # caller-supplied lazy plans deliberately keep a single projection.
            frame = frame.with_columns(pl.lit(1, dtype=pl.Int128).alias("typed_only"))
        path = tmp_path / "values.parquet"
        frame.write_parquet(path)
        spec["source"] = (
            {"type": "query", "sql": f"SELECT * FROM read_parquet('{path}')"}
            if backend == "native_query" else {"type": "file", "path": str(path)}
        )
    settings = QAVFSettings.model_validate(
        {"datasets": {"d": spec}, "engine": {"native_threads": 1, "max_workers": 1}}
    )
    result = DataContext.from_settings(settings, frames=frames, threads=1)
    result.evidence_limit = evidence
    return result


def reduced_only(monkeypatch, ctx, maximum):
    def forbidden(*args, **kwargs):
        pytest.fail("a source reduction loaded the complete input")

    monkeypatch.setattr(ctx, "get", forbidden)
    for name in ("collect", "sql"):
        original = getattr(ctx, name)

        def checked(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            assert result.height <= maximum, result.shape
            assert not result.height or "unused" not in result.columns
            return result

        monkeypatch.setattr(ctx, name, checked)


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("how", ["sum", "count", "min", "max", "none"])
def test_temporal_reduces_repeated_rows_before_collection(backend, how, tmp_path, monkeypatch):
    repeats = 1 if how == "none" else 200
    frame = pl.DataFrame(
        [
            (group, year, year, "x" * 128)
            for group in range(3)
            for year in range(5)
            for _ in range(repeats)
        ],
        schema=["g", "year", "v", "unused"],
        orient="row",
    )
    with context(frame, backend, tmp_path) as ctx:
        reduced_only(monkeypatch, ctx, 15)
        found = CumulativeRule(
            params={
                "series": [
                    {
                        "dataset": "d",
                        "value": "v",
                        "group_by": ["g"],
                        "aggregate": how,
                    }
                ]
            }
        ).evaluate(ctx)
        assert len(found) == 1 and found[0].passed
        assert found[0].total_records == 12


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_invalid_times_exclude_complete_null_equal_groups(backend, tmp_path, monkeypatch):
    frame = pl.DataFrame(
        {
            "g": ["bad", "bad", "NONE", "NONE", "good", "good"],
            "year": [1.0, None, 1.0, float("inf"), 1.0, 2.0],
            "v": [1, 2, 1, 2, 1, 2],
        }
    )
    with context(frame, backend, tmp_path, columns={"g": {"missing_values": ["NONE"]}}) as ctx:
        reduced_only(monkeypatch, ctx, 2)
        found = CumulativeRule(
            params={
                "series": [
                    {
                        "dataset": "d",
                        "value": "v",
                        "group_by": ["g"],
                    }
                ]
            }
        ).evaluate(ctx)
        assert len(found) == 2
        assert not found[0].assessed and found[0].affected_count == 2
        assert found[1].passed and found[1].total_records == 1


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("missing", ["fail", "ignore"])
def test_share_reduction_retains_null_keys_and_custom_missing(
    backend, missing, tmp_path, monkeypatch
):
    frame = pl.DataFrame({"g": [None, None, "ok", "ok"], "v": [1, -99, 1, 1]})
    with context(frame, backend, tmp_path, columns={"v": {"missing_values": [-99]}}) as ctx:
        reduced_only(monkeypatch, ctx, 2)
        found = SharesSumRule(
            params={
                "shares": [
                    {
                        "dataset": "d",
                        "value": "v",
                        "group_by": ["g"],
                        "total": 2,
                        "missing": missing,
                        "atol": 0,
                    }
                ]
            }
        ).evaluate(ctx)
        assert len(found) == 1 and not found[0].passed
        assert found[0].affected_count == 1 and found[0].total_records == 2
        sample = found[0].affected
        assert sample["g"].to_list() == [None]
        assert sample["sum"].to_list() == [1]
        assert sample["missing_shares"].to_list() == [1]


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("values", [[None, None], []])
def test_share_empty_and_null_only_are_not_clean(backend, values, tmp_path):
    frame = pl.DataFrame({"v": pl.Series(values, dtype=pl.Int64)})
    with context(frame, backend, tmp_path) as ctx:
        found = SharesSumRule(params={"shares": [{"dataset": "d", "value": "v"}]}).evaluate(ctx)
        assert len(found) == 1 and not found[0].passed
        assert found[0].assessed == bool(values)


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_decimal_reduction_retains_scale_and_exact_total(backend, tmp_path):
    frame = pl.DataFrame(
        {
            "g": [1, 1],
            "v": pl.Series(
                [Decimal("9007199254740993.25"), Decimal("0.25")], dtype=pl.Decimal(38, 2)
            ),
        }
    )
    with context(frame, backend, tmp_path) as ctx:
        relation = normalized_relation(ctx, "d", ["g", "v"])
        result = grouped_aggregate(relation, ["g"], "v", "total")
        assert result["total"].to_list() == [Decimal("9007199254740993.50")]
        assert result.schema["total"].scale == 2


def test_integer_sum_bound_is_checked_before_grouped_collection(tmp_path):
    frame = pl.DataFrame({"g": [1, 1], "v": pl.Series([2**126, 2**126], dtype=pl.Int128)})
    with context(frame, "lazy", tmp_path) as ctx:
        relation = normalized_relation(ctx, "d", ["g", "v"])
        with pytest.raises(ValueError, match="exceed accumulator precision"):
            grouped_aggregate(relation, ["g"], "v", "total")


def test_native_freshness_normalizes_only_its_selected_column(tmp_path, monkeypatch):
    marker = datetime(2099, 1, 1)
    frame = pl.DataFrame({"at": [datetime(2026, 1, 1), marker], "unused": [1, 2]})
    with context(
        frame,
        "native_query",
        tmp_path,
        columns={
            "at": {"missing_values": [marker]},
            "unused": {"missing_values": ["wrong-type"]},
        },
    ) as ctx:
        reduced_only(monkeypatch, ctx, 1)
        found = FreshnessRule(
            params={
                "now": datetime(2026, 1, 1, tzinfo=UTC),
                "checks": [
                    {
                        "dataset": "d",
                        "column": "at",
                        "max_age": "P1D",
                    }
                ],
            }
        ).evaluate(ctx)
        assert len(found) == 1 and found[0].passed
