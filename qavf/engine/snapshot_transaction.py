"""The rules of one snapshot transaction, shared by both backends.

A connection with ``snapshot: true`` promises that every dataset read through it
comes from one committed state. :class:`SnapshotTransaction` keeps that promise on
an open DuckDB connection: it reads every member inside one ``BEGIN … COMMIT``.
A transient failure or a changed identity of any member ends the attempt, so the
caller's source policy retries the whole group; a permanent member failure is
recorded for that member and the others read on (the transaction restarts without
it when the failure aborted the transaction before any data was read). A failed
or cancelled attempt rolls back and closes what it staged.

The Polars backend stages members as frames or spill files
(:class:`~qavf.engine.snapshot_coordinator.SnapshotCoordinator`); the native backend
copies them into owned temporary tables (:class:`~qavf.core.snapshots.NativeSnapshots`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import duckdb

from qavf.sources.base import SourceError, SourceIntegrityError, SourceNotFoundError
from qavf.sources.errors import redacted_error

_PLANNING_ERRORS = (duckdb.BinderException, duckdb.CatalogException, duckdb.ParserException)


def transaction_usable(con: Any) -> bool:
    """Whether a snapshot transaction survived a failed member read."""
    try:
        con.execute("SELECT 1").fetchall()
    except duckdb.Error:
        return False
    return True


def failed_before_execution(exc: BaseException) -> bool:
    """Whether a read failed while planning (binding, catalog lookup, parsing),
    so no member data was produced or consumed."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (*_PLANNING_ERRORS, SourceNotFoundError)):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def snapshot_connection(attached: Sequence[str], connections: Mapping[str, Any]) -> str | None:
    """The snapshot connection a source reading through ``attached`` belongs to:
    its only attachment, when that connection has ``snapshot: true``."""
    if len(attached) != 1:
        return None
    spec = connections.get(attached[0])
    return attached[0] if spec is not None and getattr(spec, "snapshot", False) else None


class SnapshotStageLike(Protocol):
    """What one attempt staged: its members' failures, and how to discard it."""

    failures: dict[str, SourceError]

    def close(self) -> None:
        """Discard everything the attempt staged."""
        ...


class SnapshotTransaction[S: SnapshotStageLike]:
    """Reads a snapshot group's members inside one transaction on ``con``.

    Args:
        con: The open DuckDB connection the snapshot connection is attached to.
        open_stage: A fresh, empty stage for one attempt, sharing ``failures``.
        read_member: Reads one member into the stage (``stage, con, name``).
        check_cancelled: Raises once the run is cancelled.
        fatal: Failures that end the whole attempt as they are, never one member's.
    """

    def __init__(
        self,
        con: Any,
        open_stage: Callable[[dict[str, SourceError]], S],
        read_member: Callable[[S, Any, str], None],
        check_cancelled: Callable[[], None],
        fatal: tuple[type[BaseException], ...] = (),
    ) -> None:
        self._con = con
        self._open_stage = open_stage
        self._read_member = read_member
        self._check_cancelled = check_cancelled
        self._fatal = fatal

    def read(self, members: Sequence[str]) -> S:
        """Every member read in one committed transaction, or the attempt's failure.

        A permanent member failure that aborts the transaction (DuckDB 2.0 can abort
        on binder errors) restarts the read without that member, so every member
        read still comes from one transaction.
        """
        failures: dict[str, SourceError] = {}
        pending = list(members)
        while True:
            stage = self._open_stage(failures)
            try:
                self._con.execute("BEGIN TRANSACTION")
                aborted = self._read_pending(stage, pending, failures)
                if not aborted:
                    self._con.execute("COMMIT")
                    return stage
            except BaseException:
                self._discard(stage)
                raise
            self._discard(stage)
            pending = [name for name in pending if name not in failures]

    def _read_pending(self, stage: S, pending: list[str], failures: dict[str, SourceError]) -> bool:
        """Read ``pending`` into ``stage``; whether a member failure aborted the
        transaction (it must restart without that member)."""
        for name in pending:
            self._check_cancelled()
            try:
                self._read_member(stage, self._con, name)
            except self._fatal:
                raise
            except Exception as exc:
                error = exc if isinstance(exc, SourceError) else redacted_error(exc)
                if isinstance(error, SourceIntegrityError) or error.retryable:
                    raise error from None
                failures[name] = redacted_error(
                    error,
                    SourceNotFoundError if isinstance(error, SourceNotFoundError) else SourceError,
                )
                if not transaction_usable(self._con):
                    # Only a failure before execution may restart without the member.
                    if not failed_before_execution(exc):
                        raise failures.pop(name) from None
                    return True
        return False

    def _discard(self, stage: S) -> None:
        with contextlib.suppress(duckdb.Error):
            self._con.rollback()
        stage.close()


__all__ = [
    "SnapshotStageLike",
    "SnapshotTransaction",
    "failed_before_execution",
    "snapshot_connection",
    "transaction_usable",
]
