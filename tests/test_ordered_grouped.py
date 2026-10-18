"""Ordered scalar selection keeps exact values without collecting member rows."""

from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules._ordered_grouped import ordered_grouped
from qavf.rules._spatial_aggregation import spatial_aggregate
from qavf.rules.helpers import aggregate
from qavf.rules.temporal_checks import CumulativeRule


@pytest.mark.parametrize("storage", ["eager", "lazy", "spill"])
@pytest.mark.parametrize("how", ["first", "last"])
@pytest.mark.parametrize("grouped", [True, False])
def test_ordered_reducer_matches_missing_and_tuple_order_contract(storage, how, grouped, tmp_path):
    frame = pl.DataFrame(
        {
            "g": ["a"] * 6 + ["empty"] * 2,
            "v": [None, float("nan"), 3.0, 1.0, 2.0, 2.0, None, None],
            "order": [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0, 1.0, 1.0],
            "secondary": [0, 0, 0, 1, 2, 2, 1, 2],
        }
    )
    keys = ["g"] if grouped else []
    expected = aggregate(frame, keys, "v", "result", how, order_by=("order", "secondary"))
    with (
        DataContext(
            frames={"data": frame.lazy() if storage == "lazy" else frame},
            cache_max_bytes=0 if storage == "spill" else 2**20,
            temp_directory=tmp_path,
        ) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        actual = ordered_grouped(
            observation, observation.scan(), keys, "v", "result", how, ("order", "secondary")
        )
    assert actual.equals(expected)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "values",
    [
        pl.Series("v", [2**63 - 1, 2**53 + 1], dtype=pl.Int64),
        pl.Series("v", [2**64 - 1, 2**64 - 2], dtype=pl.UInt64),
        pl.Series(
            "v", [Decimal("999999999999999999.123"), Decimal("0.001")], dtype=pl.Decimal(30, 3)
        ),
    ],
)
@pytest.mark.parametrize("how,selected", [("first", 1), ("last", 0)])
def test_ordered_reducer_preserves_exact_numeric_identity(values, how, selected, tmp_path):
    frame = pl.DataFrame({"g": ["a", "a"], "v": values, "order": [1, 0]})
    with (
        DataContext(frames={"data": frame.lazy()}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = ordered_grouped(
            observation, observation.scan(), ["g"], "v", "result", how, "order"
        )
    assert result["result"].dtype == values.dtype
    assert result["result"].item() == values[selected]


@pytest.mark.parametrize("how,direction", [("first", 1), ("last", -1)])
def test_selected_conflicts_rejected_but_nonselected_conflicts_ignored(how, direction, tmp_path):
    frame = pl.DataFrame(
        {"g": [1, 1, 1, 1], "order": [0, 0, direction, direction], "v": [5, 5, 10, 20]}
    )
    with (
        DataContext(frames={"data": frame.lazy()}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = ordered_grouped(
            observation, observation.scan(), ["g"], "v", "result", how, "order"
        )
        assert result["result"].item() == 5
        with pytest.raises(ValueError, match="conflicting values tied"):
            ordered_grouped(
                observation,
                observation.scan(),
                ["g"],
                "v",
                "result",
                "last" if how == "first" else "first",
                "order",
            )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("grouped", [True, False])
def test_empty_ordered_result_retains_measurement_dtype(grouped, tmp_path):
    frame = pl.DataFrame(schema={"g": pl.Datetime("ns"), "v": pl.UInt64, "order": pl.Int64})
    keys = ["g"] if grouped else []
    with (
        DataContext(frames={"data": frame.lazy()}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = ordered_grouped(
            observation, observation.scan(), keys, "v", "result", "first", "order"
        )
    assert result.height == (0 if grouped else 1)
    assert result["result"].dtype == pl.UInt64
    if not grouped:
        assert result.item() is None


@pytest.mark.parametrize("storage", ["eager", "lazy"])
@pytest.mark.parametrize("how", ["first", "last"])
def test_high_row_count_ordered_rules_never_project_members(monkeypatch, storage, how, tmp_path):
    n = 30000
    frame = pl.DataFrame(
        {
            "year": [1] * n + [2] * n,
            "entity": ["a"] * (2 * n),
            "order": [0] * (2 * n),
            "secondary": list(range(n)) * 2,
            "value": [2.0] * n + [1.0] * n,
        }
    )
    calls = []
    lazy = frame.lazy().map_batches(lambda batch: (calls.append(1), batch)[1])
    with DataContext(
        frames={"data": lazy if storage == "lazy" else frame}, temp_directory=tmp_path
    ) as ctx:
        monkeypatch.setattr(
            ctx, "project", lambda *args, **kwargs: pytest.fail("member projection")
        )
        [finding] = CumulativeRule(
            {
                "series": [
                    {
                        "dataset": "data",
                        "value": "value",
                        "aggregate": how,
                        "order_by": ["order", "secondary"],
                    }
                ]
            }
        ).evaluate(ctx)
        assert finding.assessed and not finding.passed
        assert finding.affected_count == 1
        prepared = spatial_aggregate(
            ctx, "data", ["entity", "year"], "value", "result", how, None, ("order", "secondary")
        )
        assert prepared.error is None
        assert prepared.values.sort("year")["result"].to_list() == [2.0, 1.0]
    assert calls == ([1, 1] if storage == "lazy" else [])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dtype", [pl.Null, pl.Float64])
def test_all_missing_groups_keep_null_measurements(dtype, tmp_path):
    frame = pl.DataFrame(
        {"g": [None, "a"], "v": pl.Series([None, None], dtype=dtype), "order": [1, 2]}
    )
    with (
        DataContext(frames={"data": frame.lazy()}, temp_directory=tmp_path) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        result = ordered_grouped(
            observation, observation.scan(), ["g"], "v", "result", "last", "order"
        )
    assert result.shape == (2, 2)
    assert result["result"].null_count() == 2
    assert result["result"].dtype == dtype
