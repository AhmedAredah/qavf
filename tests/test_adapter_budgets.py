"""Public adapter budgets reject overflow before full materialization."""

from unittest.mock import patch

import duckdb
import polars as pl
import pytest

import qavf.inputs.adapters as adapters
from qavf import Loader, Query, validate


def _config(budget):
    return {
        "datasets": {
            "data": {
                "schema": {"columns": {"x": {"ge": 0}}},
                "load_policy": {"row_budget": budget},
            }
        },
        "plugins": {"entry_points": False},
        "engine": {"parallel": False},
    }


@pytest.mark.parametrize("rows", [10, 11, 100_000])
def test_borrowed_query_limits_materialization_and_retains_connection(rows):
    checked = []
    original = adapters.check_budget

    def observe(frame, budget, what):
        checked.append(frame.height)
        return original(frame, budget, what)

    with (
        duckdb.connect() as connection,
        patch("qavf.inputs.adapters.check_budget", side_effect=observe),
    ):
        report = validate(
            Query("SELECT range AS x FROM range($rows)", connection, {"rows": rows}),
            config=_config(10),
        )
        assert connection.execute("SELECT 42").fetchone() == (42,)
    assert checked and max(checked) == min(rows, 11)
    assert report.exit_code() == (0 if rows == 10 else 1)
    if rows > 10:
        assert any("row_budget" in result.message for result in report.results)


@pytest.mark.parametrize("rows", [10, 11, 100_000])
def test_lazy_loader_limits_materialization_without_accepting_prefix(rows):
    collected = []
    calls = []
    original = pl.LazyFrame.collect

    def observe(self, *args, **kwargs):
        frame = original(self, *args, **kwargs)
        collected.append(frame.height)
        return frame

    def load():
        calls.append(1)
        return pl.LazyFrame().select(pl.int_range(0, rows).alias("x"))

    with patch.object(pl.LazyFrame, "collect", observe):
        report = validate(Loader(load), config=_config(10))
    assert calls == [1]
    assert collected and max(collected) == min(rows, 11)
    assert report.exit_code() == (0 if rows == 10 else 1)
    if rows > 10:
        assert any("row_budget" in result.message for result in report.results)


def test_snapshot_query_uses_same_budget_before_staging(tmp_path):
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(path)) as writer:
        writer.execute("CREATE TABLE readings AS SELECT range AS x FROM range(100000)")
    checked = []
    original = adapters.check_budget

    def observe(frame, budget, what):
        checked.append(frame.height)
        return original(frame, budget, what)

    config = _config(10)
    config["connections"] = {
        "warehouse": {"type": "duckdb", "path": str(path), "snapshot": True},
    }
    with patch("qavf.inputs.adapters.check_budget", side_effect=observe):
        report = validate(
            Query("SELECT * FROM warehouse.readings", connection="warehouse"),
            config=config,
        )
    assert checked and max(checked) == 11
    assert report.exit_code() == 1
    assert any("row_budget" in result.message for result in report.results)
