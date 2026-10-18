"""Consistent multi-table reads: snapshot connections, REC-005, read times."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings
from qavf.sources import DatabaseSource
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "live.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE orders AS SELECT 1 AS version")
        con.execute("CREATE TABLE lines AS SELECT 1 AS version")
    return path


def _settings(path, snapshot: bool) -> QAVFSettings:
    return QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "connections": {
                "live": {
                    "type": "duckdb",
                    "path": str(path),
                    "read_only": False,
                    "snapshot": snapshot,
                }
            },
            "datasets": {
                "orders": {"source": {"type": "database", "connection": "live", "table": "orders"}},
                "lines": {"source": {"type": "database", "connection": "live", "table": "lines"}},
            },
        }
    )


def test_without_a_snapshot_a_concurrent_update_mixes_versions(database, monkeypatch) -> None:
    real_load = DatabaseSource.load

    def load_then_update(self):
        frame = real_load(self)
        if self.params.table == "orders":  # an upstream job commits between the two reads
            with duckdb.connect(str(database)) as writer:
                writer.execute("UPDATE lines SET version = 2")
        return frame

    monkeypatch.setattr(DatabaseSource, "load", load_then_update)
    ctx = DataContext.from_settings(_settings(database, snapshot=False))
    versions = (ctx.get("orders")["version"][0], ctx.get("lines")["version"][0])
    assert versions == (1, 2)  # an inconsistent pair: the failure snapshot mode prevents


def test_a_snapshot_sees_one_coherent_version(database, monkeypatch) -> None:
    real_read = DatabaseSource.read
    read: list[str] = []

    def read_then_update(self, con):
        frame = real_read(self, con)
        read.append(self.params.table)
        if len(read) == 1:
            # After the first table is read, an upstream job commits a new version of the
            # other one (same database instance, its own transaction, committed at once).
            other = "lines" if self.params.table == "orders" else "orders"
            con.cursor().execute(f"UPDATE live.{other} SET version = 2")
        return frame

    monkeypatch.setattr(DatabaseSource, "read", read_then_update)
    ctx = DataContext.from_settings(_settings(database, snapshot=True))
    # Asking for one dataset reads the whole snapshot group.
    versions = (ctx.get("orders")["version"][0], ctx.get("lines")["version"][0])
    assert len(read) == 2 and versions == (1, 1)  # the concurrent commit is not visible
    infos = {info.name: info for info in ctx.describe()}
    assert infos["orders"].loaded_at == infos["lines"].loaded_at
    assert infos["lines"].notes["snapshot"]["datasets"] == ["lines", "orders"]
    with duckdb.connect(str(database)) as check:  # the update did happen
        assert check.execute(
            "SELECT max(version) FROM (FROM orders UNION ALL FROM lines)"
        ).fetchone() == (2,)


def test_one_failing_member_does_not_block_the_others(database) -> None:
    settings = _settings(database, snapshot=True)
    settings.datasets["missing"] = settings.datasets["orders"].model_copy(
        update={"source": {"type": "database", "connection": "live", "table": "nope"}}
    )
    ctx = DataContext.from_settings(settings)
    assert ctx.get("orders").height == 1 and ctx.get("lines").height == 1
    assert not ctx.has("missing") and "nope" in str(ctx.load_error("missing"))


def test_an_unreachable_snapshot_connection_fails_every_member(tmp_path) -> None:
    settings = _settings(tmp_path / "absent.duckdb", snapshot=True)
    settings.connections["live"] = settings.connections["live"].model_copy(
        update={"read_only": True}
    )
    ctx = DataContext.from_settings(settings)
    assert not ctx.has("orders") and not ctx.has("lines")


def test_every_dataset_records_when_it_was_read(database) -> None:
    ctx = DataContext.from_settings(_settings(database, snapshot=False))
    ctx.get("orders")
    [info] = ctx.describe()
    assert info.loaded_at is not None and info.loaded_at.tzinfo is not None


# ------------------------------------------------------------------ REC-005
def _batch(frames):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {name: {} for name in frames},
            "rules": {
                "REC-005": {"params": {"batches": [{"datasets": list(frames), "column": "run"}]}}
            },
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames).get("REC-005")


def test_datasets_from_one_batch_pass() -> None:
    frames = {"a": pl.DataFrame({"run": [7, 7]}), "b": pl.DataFrame({"run": [7]})}
    assert _batch(frames).status is Status.PASS


@pytest.mark.parametrize(
    "frames",
    [
        {"a": pl.DataFrame({"run": [7, 7]}), "b": pl.DataFrame({"run": [8]})},
        {"a": pl.DataFrame({"run": [7, 8]}), "b": pl.DataFrame({"run": [7]})},
        {"a": pl.DataFrame({"run": [7, None]}), "b": pl.DataFrame({"run": [7]})},
    ],
)
def test_mixed_or_missing_batches_fail(frames) -> None:
    assert _batch(frames).status is Status.FAIL
