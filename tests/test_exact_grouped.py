"""Sorted-batch reductions preserve exact results across chunk and dtype boundaries."""

from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules._exact_grouped import exact_grouped
from qavf.rules.helpers import aggregate
from qavf.rules.temporal_checks import CumulativeRule


@pytest.mark.parametrize("how", ["sum", "mean", "weighted_mean"])
@pytest.mark.parametrize("storage", ["frame", "lazy", "spill"])
def test_exact_grouped_matches_independent_small_oracle(how, storage, tmp_path):
    frame = pl.DataFrame(
        {
            "g": ["a"] * 3 + ["b"] * 3 + ["empty"],
            "v": [1e308, -1e308, 3.0, 1.0, 2.0, 6.0, None],
            "w": [1.0] * 7,
        }
    )
    with (
        DataContext(
            frames={"data": frame.lazy() if storage == "lazy" else frame},
            cache_max_bytes=0 if storage == "spill" else 2**20,
            temp_directory=tmp_path,
        ) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        actual = exact_grouped(
            ctx,
            observation,
            observation.scan(),
            ["g"],
            "v",
            "result",
            how,
            weight="w" if how == "weighted_mean" else None,
            batch_rows=2,
        )
    expected = [3.0, 9.0, None] if how == "sum" else [1.0, 3.0, None]
    assert actual.sort("g")["result"].to_list() == expected
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("how", ["mean", "weighted_mean"])
def test_exact_grouped_rejects_lossy_integer_observations(how, tmp_path):
    frame = pl.DataFrame({"v": [2**53 + 1, 2**53], "w": [1, 1]})
    with (
        DataContext(frames={"data": frame}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
        pytest.raises(ValueError, match="exactly"),
    ):
        exact_grouped(
            ctx,
            observation,
            observation.scan(),
            [],
            "v",
            "result",
            how,
            weight="w" if how == "weighted_mean" else None,
            batch_rows=1,
        )
    assert not list(tmp_path.iterdir())


def test_weighted_products_do_not_overflow_before_division(tmp_path):
    frame = pl.DataFrame({"v": [1e308, 1e308], "w": [1e308, 1e308]})
    with (
        DataContext(frames={"data": frame}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = exact_grouped(
            ctx,
            observation,
            observation.scan(),
            [],
            "v",
            "result",
            "weighted_mean",
            weight="w",
            batch_rows=1,
        )
    assert result.item() == 1e308


def test_numeric_aggregates_keep_nan_null_and_negative_weight_policy(tmp_path):
    frame = pl.DataFrame(
        {
            "g": ["missing"] * 2 + ["negative"] * 2,
            "v": [None, float("nan"), 2.0, 3.0],
            "w": [1.0, 1.0, 1.0, -1.0],
        }
    )
    expected = aggregate(frame, ["g"], "v", "result", "weighted_mean", weight="w").sort("g")
    with (
        DataContext(frames={"data": frame}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = exact_grouped(
            ctx,
            observation,
            observation.scan(),
            ["g"],
            "v",
            "result",
            "weighted_mean",
            weight="w",
            batch_rows=1,
        )
    assert result.sort("g").equals(expected)


def test_temporal_float_rule_avoids_full_projected_frame(monkeypatch, tmp_path):
    rows = 30000
    frame = pl.DataFrame({"year": [1] * rows + [2] * rows, "value": [1.0] * rows + [0.5] * rows})
    with DataContext(
        frames={"data": frame.lazy()}, evidence_limit=1, temp_directory=tmp_path
    ) as ctx:

        def no_project(*args, **kwargs):
            pytest.fail("exact float totals must consume scoped batches")

        monkeypatch.setattr(ctx, "project", no_project)
        [result] = CumulativeRule({"series": [{"dataset": "data", "value": "value"}]}).evaluate(ctx)
        assert result.assessed and not result.passed
        assert result.affected_count == 1
        assert result.affected[0]["decrease"] == Decimal(15000)


@pytest.mark.parametrize(
    "key",
    [
        pl.Series("g", [1, 2, 1], dtype=pl.Datetime("ns")),
        pl.Series("g", [[1], [2], [1]], dtype=pl.List(pl.Int64)),
        pl.Series("g", [{"a": 1}, {"a": 2}, {"a": 1}]),
    ],
)
def test_exact_grouping_retains_native_nested_and_nanosecond_keys(key, tmp_path):
    frame = pl.DataFrame({"g": key, "v": [1.0, 2.0, 3.0]})
    expected = frame.group_by("g").agg(pl.col("v").sum().alias("result")).sort("g")
    with (
        DataContext(frames={"data": frame.lazy()}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = exact_grouped(
            ctx, observation, observation.scan(), ["g"], "v", "result", "sum", batch_rows=1
        )
    assert result.sort("g").equals(expected)


@pytest.mark.parametrize("how,expected", [("first", 1.0), ("last", 3.0)])
def test_spatial_normalizer_projects_lexicographic_secondary_order(how, expected, tmp_path):
    from qavf.rules._spatial_aggregation import spatial_aggregate
    from qavf.rules.spatial_checks import Normalizer

    frame = pl.DataFrame(
        {"g": ["a", "a", "a"], "v": [3.0, 1.0, 2.0], "order": [0, 0, 0], "secondary": [3, 1, 2]}
    )
    spec = Normalizer(dataset="data", value="v", aggregate=how, order_by=["order", "secondary"])
    with DataContext(frames={"data": frame.lazy()}, temp_directory=tmp_path) as ctx:
        prepared = spatial_aggregate(ctx, "data", ["g"], "v", "result", how, None, spec.order_by)
    assert prepared.error is None
    assert prepared.values["result"].item() == expected


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Datetime("ns"), pl.List(pl.Int64)])
@pytest.mark.parametrize("rational", [False, True])
def test_empty_grouped_result_retains_typed_schema(dtype, rational, tmp_path):
    frame = pl.DataFrame(schema={"g": dtype, "v": pl.Float64})
    with (
        DataContext(frames={"data": frame.lazy()}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = exact_grouped(
            ctx,
            observation,
            observation.scan(),
            ["g"],
            "v",
            "result",
            "sum",
            rational=rational,
            presence_alias="rows",
        )
    assert result.is_empty()
    assert result.schema == {
        "g": dtype,
        "result": pl.String if rational else pl.Float64,
        "rows": pl.UInt64,
    }
    assert not list(tmp_path.iterdir())


def test_exact_grouped_rejects_cumulative_output_at_bounded_flush(tmp_path, monkeypatch):
    from qavf.engine.resources import ResourceLimitError

    frame = pl.DataFrame({"g": range(100), "v": [1.0] * 100})
    updates = []
    from qavf.rules import _exact_grouped

    original = _exact_grouped._State.add

    def record(self, value, weight, **kwargs):
        updates.append(value)
        return original(self, value, weight, **kwargs)

    monkeypatch.setattr(_exact_grouped._State, "add", record)
    with (
        DataContext(
            frames={"data": frame.lazy()}, temp_directory=tmp_path, memory_budget_bytes=128
        ) as ctx,
        ctx.observe("data", frame.columns) as observation,
        pytest.raises(ResourceLimitError, match="exact aggregate output"),
    ):
        exact_grouped(
            ctx, observation, observation.scan(), ["g"], "v", "result", "sum", batch_rows=1
        )
    assert len(updates) < frame.height
    assert not list(tmp_path.iterdir())
