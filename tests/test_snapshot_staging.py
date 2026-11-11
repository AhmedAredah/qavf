"""Coherent database snapshots stay in memory while they fit and spill to owned files after."""

from __future__ import annotations

import duckdb
import polars as pl
import pytest

from qavf import explain, validate
from qavf.bounded_batches import BoundedBatches
from qavf.duckdb_results import polars_frame
from qavf.engine.context import DataContext
from qavf.engine.resources import ResourceLimitError
from qavf.run_control import CancellationToken
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceError

ROWS = 150_000
TYPED = (
    "SELECT range AS id, (range / 7)::DECIMAL(18, 3) AS amount, "
    "TIMESTAMPTZ '2024-01-01 00:00:00+00' + to_seconds(range) AS zoned, "
    "[range, range + 1] AS pair, {'k': range % 5, 's': range::VARCHAR} AS record, "
    "range::HUGEINT AS huge, (['low', 'mid', 'high'][(range % 3) + 1])::level AS level, "
    "1 AS version FROM range(ROWS)"
).replace("ROWS", str(ROWS))


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "live.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TYPE level AS ENUM ('low', 'mid', 'high')")
        con.execute(f"CREATE TABLE orders AS {TYPED}")
        con.execute(f"CREATE TABLE lines AS {TYPED}")
        con.execute("CREATE TABLE small AS SELECT 1 AS version")
    return path


def _settings(path, **policy) -> QAVFSettings:
    datasets = {
        name: {
            "source": {"type": "database", "connection": "live", "table": name},
            "load_policy": dict(policy),
        }
        for name in ("orders", "lines", "small")
    }
    return QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "connections": {
                "live": {"type": "duckdb", "path": str(path), "read_only": False, "snapshot": True}
            },
            "datasets": datasets,
        }
    )


class Interrupt:
    """Replace the snapshot's row admission: raise ``error`` once past ``after`` rows."""

    def __init__(self, error: BaseException, after: int = BoundedBatches.BATCH_ROWS) -> None:
        self.error, self.after, self.raised = error, after, 0
        self.real = DataContext._admit_snapshot_rows

    def __call__(self, name: str, budget: int | None, rows: int) -> None:
        if rows > self.after and not self.raised:
            self.raised += 1
            raise self.error
        self.real(name, budget, rows)


class CancelOnce:
    """Cancel the run's token once a member has streamed past one batch."""

    def __init__(self, token: CancellationToken) -> None:
        self.token = token
        self.real = DataContext._admit_snapshot_rows

    def __call__(self, name: str, budget: int | None, rows: int) -> None:
        if rows > BoundedBatches.BATCH_ROWS:
            self.token.cancel()
        self.real(name, budget, rows)


class UpdateAfterFirst:
    """Commit a new version of every other member after the first member is staged."""

    def __init__(self, ctx: DataContext) -> None:
        self.staged: list[str] = []
        self.real = ctx._stage_member

    def __call__(self, stage, con, name) -> None:
        self.real(stage, con, name)
        self.staged.append(name)
        if len(self.staged) == 1:
            writer = con.cursor()
            for other in ("orders", "lines", "small"):
                if other != name:
                    writer.execute(f"UPDATE live.{other} SET version = 2")
            writer.close()


class MemoryAllowance:
    """Replace the snapshot's in-memory allowance with a fixed number of bytes."""

    def __init__(self, allowance: int) -> None:
        self.allowance = allowance

    def __call__(self) -> int:
        return self.allowance


@pytest.fixture
def spilling(monkeypatch):
    """Snapshot groups spill from their first batch."""
    monkeypatch.setattr(DataContext, "_snapshot_memory_bytes", MemoryAllowance(0))


def _no_owned_files(ctx: DataContext) -> bool:
    usage = ctx.resource_usage()
    return usage["owned_spill_bytes"] == 0 and usage["owned_spill_files"] == 0


def test_members_stage_to_spill_files_with_their_exact_types(database, spilling):
    with DataContext.from_settings(_settings(database)) as ctx:
        frame = ctx.get("orders")
        usage = ctx.resource_usage()
        # All three members were written before any was read back.
        assert usage["frame_spill_write_count"] == 3
        assert usage["peak_frame_spill_bytes"] > 0
        with duckdb.connect(str(database), read_only=True) as reference:
            reference.execute("SET TimeZone = 'UTC'")
            expected = polars_frame(reference.sql("SELECT * FROM orders"))
        assert frame.schema == expected.schema
        assert frame.equals(expected)
        assert ctx.get("lines").height == ROWS


def test_a_concurrent_writer_between_members_is_not_seen(database, monkeypatch):
    with DataContext.from_settings(_settings(database)) as ctx:
        writer = UpdateAfterFirst(ctx)
        monkeypatch.setattr(ctx, "_stage_member", writer)
        versions = {name: set(ctx.get(name)["version"]) for name in ("orders", "lines", "small")}
    assert len(writer.staged) == 3
    assert versions == {"orders": {1}, "lines": {1}, "small": {1}}
    with duckdb.connect(str(database), read_only=True) as check:
        assert check.execute("SELECT max(version) FROM orders").fetchone() == (2,)


def test_row_budget_is_exact_across_batches(database, spilling):
    settings = _settings(database, row_budget=ROWS)
    with DataContext.from_settings(settings) as ctx:
        assert ctx.get("orders").height == ROWS
    settings = _settings(database, row_budget=ROWS - 1)
    with DataContext.from_settings(settings) as ctx:
        assert not ctx.has("orders") and not ctx.has("lines")
        assert f"more than {ROWS - 1:,} rows" in str(ctx.load_error("orders"))
        assert ctx.get("small").height == 1
        assert ctx.resource_usage()["frame_spill_write_count"] == 1


def test_interrupted_stream_leaves_no_files(database, monkeypatch, spilling):
    stop = Interrupt(ResourceLimitError("validation execution was cancelled"))
    monkeypatch.setattr(DataContext, "_admit_snapshot_rows", staticmethod(stop))
    with DataContext.from_settings(_settings(database)) as ctx:
        assert not ctx.has("orders")
        assert "cancelled" in str(ctx.load_error("lines"))
        assert _no_owned_files(ctx)
        assert ctx.resource_usage()["frame_spill_bytes"] == 0
        assert not any(ctx._frames.directory.iterdir())


def test_a_retry_after_a_mid_stream_failure_restarts_the_group(database, monkeypatch, spilling):
    stop = Interrupt(SourceError("transient", retryable=True))
    monkeypatch.setattr(DataContext, "_admit_snapshot_rows", staticmethod(stop))
    with DataContext.from_settings(_settings(database, retries=1, retry_delay_s=0)) as ctx:
        assert ctx.get("orders").height == ROWS
        assert stop.raised == 1
        usage = ctx.resource_usage()
        # Only the successful attempt's three files remain.
        assert usage["owned_spill_files"] <= 3
        assert usage["frame_spill_write_count"] == 3
        assert ctx.describe()[0].notes["retries_used"] == 1


def test_cancellation_mid_stream_stops_the_run(database, monkeypatch):
    token = CancellationToken()
    monkeypatch.setattr(DataContext, "_admit_snapshot_rows", staticmethod(CancelOnce(token)))
    schema = {"columns": {"version": {"dtype": "int"}}}
    config = _settings(database).model_dump(mode="json", exclude_unset=True)
    for spec in config["datasets"].values():
        spec["schema"] = schema
    report = validate(config=config, backend="polars", cancel=token)
    assert report.metadata.cancelled


def test_explain_describes_adaptive_staging(database):
    config = _settings(database).model_dump(mode="json", exclude_unset=True)
    for spec in config["datasets"].values():
        spec["schema"] = {"columns": {"version": {"dtype": "int"}}}
    config["resources"] = {"max_spill": "1GiB", "target_memory": "512MiB"}
    plan = explain(config=config, backend="polars")
    assert {item.staging for item in plan.datasets} == {"adaptive_snapshot"}
    for item in plan.datasets:
        assert item.staging_limits["max_spill"] == 1024**3
        assert item.staging_limits["target_memory"] == 512 * 1024**2
        assert set(item.staging_limits) == {
            "cache_memory",
            "target_memory",
            "max_spill",
            "temp_dir",
            "batch_rows",
        }


def test_a_plugin_member_without_a_relation_is_spilled_whole(database, monkeypatch, spilling):
    monkeypatch.setattr(
        "qavf.sources.query.SQLSource.snapshot_relation", _no_relation, raising=True
    )
    with DataContext.from_settings(_settings(database)) as ctx:
        assert ctx.get("orders").height == ROWS
        assert ctx.resource_usage()["frame_spill_write_count"] == 3


def _no_relation(self, con):
    return None


def test_polars_frame_matches_for_an_empty_member(database):
    with duckdb.connect(str(database)) as con:
        con.execute("DELETE FROM small")
    with DataContext.from_settings(_settings(database)) as ctx:
        frame = ctx.get("small")
    assert frame.height == 0 and frame.schema == pl.Schema({"version": pl.Int32})


def _reference(database, table: str) -> pl.DataFrame:
    with duckdb.connect(str(database), read_only=True) as reference:
        reference.execute("SET TimeZone = 'UTC'")
        return polars_frame(reference.sql(f"SELECT * FROM {table}"))


def test_a_group_that_fits_in_memory_writes_no_spill(database):
    with DataContext.from_settings(_settings(database)) as ctx:
        frame = ctx.get("orders")
        usage = ctx.resource_usage()
        assert usage["frame_spill_write_count"] == 0
        assert usage["peak_frame_spill_bytes"] == 0
        assert ctx.get("lines").height == ROWS and ctx.get("small").height == 1
    expected = _reference(database, "orders")
    assert frame.schema == expected.schema
    assert frame.equals(expected)


def test_a_group_that_outgrows_memory_spills_the_buffered_members_too(database, monkeypatch):
    # Room for the first member and part of the second: the second overflows mid-stream.
    with DataContext.from_settings(_settings(database)) as ctx:
        member = ctx.get("lines").estimated_size()
    monkeypatch.setattr(DataContext, "_snapshot_memory_bytes", MemoryAllowance(member * 3 // 2))
    with DataContext.from_settings(_settings(database)) as ctx:
        frames = {name: ctx.get(name) for name in ("orders", "lines", "small")}
        assert ctx.resource_usage()["frame_spill_write_count"] == 3
        assert {name: frame.height for name, frame in frames.items()} == {
            "lines": ROWS,
            "orders": ROWS,
            "small": 1,
        }
    for name in ("orders", "lines"):
        expected = _reference(database, name)
        assert frames[name].schema == expected.schema
        assert frames[name].equals(expected)


def test_row_budget_is_exact_in_memory(database):
    settings = _settings(database, row_budget=ROWS - 1)
    with DataContext.from_settings(settings) as ctx:
        assert not ctx.has("orders")
        assert f"more than {ROWS - 1:,} rows" in str(ctx.load_error("orders"))
        assert ctx.resource_usage()["frame_spill_write_count"] == 0


def test_a_retry_in_memory_restarts_the_group(database, monkeypatch):
    stop = Interrupt(SourceError("transient", retryable=True))
    monkeypatch.setattr(DataContext, "_admit_snapshot_rows", staticmethod(stop))
    with DataContext.from_settings(_settings(database, retries=1, retry_delay_s=0)) as ctx:
        assert ctx.get("orders").height == ROWS
        assert stop.raised == 1
        assert ctx.resource_usage()["frame_spill_write_count"] == 0
        assert _no_owned_files(ctx)
