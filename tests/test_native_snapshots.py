"""Snapshot connections on the native backend: one transaction per connection (EP-01).

The members of a ``snapshot: true`` connection are copied into owned temporary
tables inside one ``BEGIN … COMMIT`` on the native connection, under the same
transaction rules as the Polars backend (``SnapshotTransaction``).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import duckdb
import pytest

from qavf import CancellationToken, validate
from qavf.core.snapshots import NativeSnapshotStage
from qavf.core.sources import InputReader
from qavf.sources.base import SourceError

READ_MEMBER = NativeSnapshotStage.read_member
"""The real member read, which the hooks below wrap."""


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "live.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE orders AS SELECT 1 AS version")
        con.execute("CREATE TABLE lines AS SELECT 1 AS version FROM range(3)")
    return path


def _config(path: Path, *, snapshot: bool = True, **policy: Any) -> dict[str, Any]:
    """Two members whose ``version`` must be 1: a newer committed state fails them."""
    schema = {"columns": {"version": {"dtype": "int", "le": 1}}}
    return {
        "connections": {
            "live": {"type": "duckdb", "path": str(path), "read_only": False, "snapshot": snapshot}
        },
        "datasets": {
            name: {
                "source": {"type": "database", "connection": "live", "table": name},
                "schema": schema,
                **({"load_policy": policy} if policy else {}),
            }
            for name in ("orders", "lines")
        },
    }


def _read_then_update(stage: NativeSnapshotStage, prepared: Any) -> None:
    """Read one member, then commit a newer version of the other from another cursor
    of the same database instance (an upstream job writing between the two reads)."""
    READ_MEMBER(stage, prepared)
    if len(stage.loaded) == 1:
        other = "orders" if prepared.name == "lines" else "lines"
        stage.reader.connection.cursor().execute(f"UPDATE live.{other} SET version = 2")


def test_a_concurrent_writer_between_members_is_not_seen(database, monkeypatch):
    monkeypatch.setattr(NativeSnapshotStage, "read_member", _read_then_update)
    report = validate(config=_config(database), backend="duckdb")
    assert report.verdict() == "passed"  # both members come from the first state
    datasets = {dataset.name: dataset for dataset in report.metadata.datasets}
    assert (datasets["orders"].rows, datasets["lines"].rows) == (1, 3)
    assert datasets["orders"].loaded_at == datasets["lines"].loaded_at
    assert datasets["lines"].notes["snapshot"]["datasets"] == ["lines", "orders"]
    with duckdb.connect(str(database), read_only=True) as check:  # the update did happen
        assert check.execute(
            "SELECT max(version) FROM (FROM orders UNION ALL FROM lines)"
        ).fetchone() == (2,)


def test_without_a_snapshot_the_writer_is_seen(database, monkeypatch):
    """The counterpart: without a snapshot the second read sees the newer state."""
    real_load = InputReader.load
    calls: list[str] = []
    monkeypatch.setattr(
        InputReader, "load", functools.partialmethod(_load_then_update, real_load, calls)
    )
    report = validate(config=_config(database, snapshot=False), backend="duckdb")
    assert report.verdict() == "failed"


def _load_then_update(
    reader: InputReader, real_load: Any, calls: list[str], name: str, value: Any, spec: Any
) -> Any:
    loaded = real_load(reader, name, value, spec)
    calls.append(name)
    if len(calls) == 1:
        other = "orders" if name == "lines" else "lines"
        reader.connection.cursor().execute(f"UPDATE live.{other} SET version = 2")
    return loaded


class FailFirstRead:
    """Fails the first member read with ``error``, then reads normally."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.raised = 0

    def __get__(self, stage: NativeSnapshotStage, owner: type) -> functools.partial[None]:
        return functools.partial(self, stage)

    def __call__(self, stage: NativeSnapshotStage, prepared: Any) -> None:
        if not self.raised:
            self.raised += 1
            READ_MEMBER(stage, prepared)  # an owned table to clean up
            raise self.error
        READ_MEMBER(stage, prepared)


def test_a_transient_failure_retries_the_whole_group(database, monkeypatch):
    failing = FailFirstRead(SourceError("transient", retryable=True))
    monkeypatch.setattr(NativeSnapshotStage, "read_member", failing)
    report = validate(config=_config(database, retries=1, retry_delay_s=0), backend="duckdb")
    assert failing.raised == 1 and report.verdict() == "passed"
    notes = [dataset.notes for dataset in report.metadata.datasets]
    assert [note["retries_used"] for note in notes] == [1, 1]


def test_a_final_failure_fails_every_member(database, monkeypatch):
    failing = FailFirstRead(SourceError("transient", retryable=True))
    monkeypatch.setattr(NativeSnapshotStage, "read_member", failing)
    report = validate(config=_config(database), backend="duckdb")
    assert set(report.metadata.unavailable_datasets) == {"orders", "lines"}
    assert all(
        "snapshot 'live' was not assessed" in reason
        for reason in report.metadata.unavailable_datasets.values()
    )


def test_the_row_budget_is_exact_per_member(database):
    config = _config(database, row_budget=1)
    report = validate(config=config, backend="duckdb")
    assert "more than 1 rows" in report.metadata.unavailable_datasets["lines"]
    assert [dataset.name for dataset in report.metadata.datasets] == ["orders"]
    full = validate(config=config, backend="polars")
    assert set(full.metadata.unavailable_datasets) == {"lines"}


def test_cancellation_inside_the_transaction_stops_the_run(database, monkeypatch):
    token = CancellationToken()
    monkeypatch.setattr(NativeSnapshotStage, "read_member", functools.partialmethod(_cancel, token))
    report = validate(config=_config(database), backend="duckdb", cancel=token)
    assert report.metadata.cancelled
    assert not report.metadata.datasets


def _cancel(stage: NativeSnapshotStage, token: CancellationToken, prepared: Any) -> None:
    READ_MEMBER(stage, prepared)
    token.cancel()
