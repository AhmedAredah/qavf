"""Snapshot connections on the native backend: one transaction per connection.

Every dataset read through a connection with ``snapshot: true`` is read in one
``BEGIN … COMMIT`` on the run's owned DuckDB connection, the first time any of them
is asked for. Each member is copied into its own owned temporary table inside the
transaction, so every member reflects the same committed state of the connection.
The transaction rules (whole-group retry, member failures, rollback and cleanup,
cancellation) are :class:`~qavf.engine.snapshot_transaction.SnapshotTransaction`'s,
shared with the Polars backend; the native reader keeps its exact row budget.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import duckdb

from qavf.engine.snapshot_transaction import SnapshotTransaction, snapshot_connection
from qavf.sources.base import SourceError
from qavf.sources.errors import redacted_error
from qavf.sources.execution import RetryPolicy
from qavf.sources.sql import referenced_connections

if TYPE_CHECKING:
    from qavf.core.sources import InputReader, LoadedDataset, PreparedInput


def source_attachments(value: Any, connections: Mapping[str, Any]) -> list[str]:
    """The named connections a configured source reads through, as the native
    reader attaches them (see :class:`~qavf.core.connections.SourceConnections`)."""
    if not isinstance(value, Mapping):
        return []
    kind = value.get("type")
    attached: list[Any] = []
    if kind == "database":
        attached.append(value.get("connection"))
    elif kind == "query":
        named = value.get("connections")
        sql = value.get("sql")
        if named:
            attached.extend(named)
        elif isinstance(sql, str):
            attached.extend(referenced_connections(sql, connections))
        attached.append(value.get("connection"))
    return list(dict.fromkeys(name for name in attached if isinstance(name, str)))


def snapshot_groups(settings: Any, inputs: Mapping[str, Any]) -> dict[str, list[str]]:
    """Each snapshot connection's member datasets, sorted, for configured sources."""
    groups: dict[str, list[str]] = {}
    for name in settings.datasets:
        attached = source_attachments(inputs.get(name), settings.connections)
        connection = snapshot_connection(attached, settings.connections)
        if connection is not None:
            groups.setdefault(connection, []).append(name)
    return {connection: sorted(names) for connection, names in groups.items()}


@dataclass
class NativeSnapshotStage:
    """One attempt's members, each in its own owned temporary table."""

    reader: InputReader
    failures: dict[str, SourceError]
    started_at: datetime = field(default_factory=functools.partial(datetime.now, UTC))
    loaded: dict[str, LoadedDataset] = field(default_factory=dict)
    owned: list[list[tuple[str, str]]] = field(default_factory=list)
    retries_used: int = 0

    def read_member(self, prepared: PreparedInput) -> None:
        """Copy one member into an owned table; a failed read leaves nothing behind."""
        owned: list[tuple[str, str]] = []
        self.owned.append(owned)
        try:
            self.loaded[prepared.name] = self.reader.read(prepared, owned)
        except BaseException:
            self.reader.drop(owned)
            owned.clear()
            raise

    def close(self) -> None:
        """Drop every table the attempt created (a rollback already discarded them)."""
        for owned in reversed(self.owned):
            self.reader.drop(owned)
        self.owned.clear()
        self.loaded.clear()


class NativeSnapshots:
    """Reads the run's snapshot groups on the native owned connection.

    Args:
        reader: The run's input reader (its connection, policies and cancellation).
        settings: The run's settings.
        inputs: Each dataset's input (configured sources are mappings).
    """

    def __init__(self, reader: InputReader, settings: Any, inputs: Mapping[str, Any]) -> None:
        self._reader = reader
        self._settings = settings
        self._inputs = inputs
        self.groups = snapshot_groups(settings, inputs)
        self._connection_of = {
            name: connection for connection, names in self.groups.items() for name in names
        }
        self._results: dict[str, LoadedDataset | BaseException] = {}
        self._taken: set[str] = set()

    def connection(self, name: str) -> str | None:
        """The snapshot connection ``name`` is read through, if any."""
        return self._connection_of.get(name)

    def load(self, name: str) -> LoadedDataset:
        """``name`` from its group's transaction, reading the group on first ask.

        Raises:
            Exception: ``name``'s own failure, or the whole group's.
        """
        connection = self._connection_of[name]
        if name not in self._results:
            self._read_group(connection)
        self._taken.add(name)
        result = self._results[name]
        if isinstance(result, BaseException):
            raise result
        return result

    def release_untaken(self, names: list[str] | tuple[str, ...]) -> None:
        """Drop members of ``names`` read with their group but never asked for."""
        for name in names:
            result = self._results.get(name)
            if (
                name not in self._taken
                and result is not None
                and not isinstance(result, BaseException)
            ):
                self._reader.release(result)
                self._taken.add(name)

    def _read_group(self, connection: str) -> None:
        members = self.groups[connection]
        prepared: dict[str, PreparedInput] = {}
        for name in members:
            try:
                prepared[name] = self._reader.prepare(
                    name, self._inputs[name], self._settings.datasets[name]
                )
            except Exception as exc:
                self._results[name] = exc
        if not prepared:
            return
        policies = [item.retry for item in prepared.values()]
        retry = RetryPolicy(
            retries=min(policy.retries for policy in policies),
            delay_s=max(policy.delay_s for policy in policies),
            safe=all(policy.safe for policy in policies),
        )
        transaction = SnapshotTransaction(
            self._reader.connection,
            functools.partial(NativeSnapshotStage, self._reader),
            functools.partial(self._read_member, prepared),
            self._reader.check_cancelled,
        )
        try:
            stage = self._attempts(list(prepared), transaction, retry)
        except Exception as exc:
            failure = redacted_error(
                SourceError(f"snapshot {connection!r} was not assessed: {exc}")
            )
            for name in prepared:
                self._results[name] = failure
            return
        note = {
            "connection": connection,
            "datasets": list(members),
            "started_at": stage.started_at.isoformat(),
        }
        for name, loaded in stage.loaded.items():
            loaded.metadata.update(
                loaded_at=stage.started_at.isoformat(),
                snapshot=dict(note),
                retries_used=stage.retries_used,
            )
            self._results[name] = loaded
        self._results.update(stage.failures)

    def _attempts(
        self,
        members: list[str],
        transaction: SnapshotTransaction[NativeSnapshotStage],
        retry: RetryPolicy,
    ) -> NativeSnapshotStage:
        """The group's committed stage, retrying the whole transaction under the
        strictest member policy."""
        for attempt in range(retry.retries + 1):
            self._reader.check_cancelled()
            try:
                stage = transaction.read(members)
            except Exception as exc:
                error = redacted_error(exc) if isinstance(exc, duckdb.Error) else exc
                pause = retry.pause(error, attempt)
                if pause is None:
                    raise error from None
                self._reader.pause(pause)
            else:
                stage.retries_used = attempt
                return stage
        raise AssertionError("unreachable retry state")

    @staticmethod
    def _read_member(
        prepared: Mapping[str, PreparedInput], stage: NativeSnapshotStage, con: Any, name: str
    ) -> None:
        stage.read_member(prepared[name])


__all__ = ["NativeSnapshots", "snapshot_groups", "source_attachments"]
