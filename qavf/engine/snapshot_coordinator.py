"""Snapshot coordination behind the data context: one transaction per snapshot group.

Datasets read through a connection with ``snapshot: true`` are read together, in
one transaction, the first time any of them is asked for. :class:`SnapshotCoordinator`
owns that: it reads every member of the group inside the transaction (staged in
memory, or in owned spill files once the group outgrows the memory allowance,
:class:`~qavf.engine.snapshot_staging.SnapshotStage`), retries a transient failure
of the whole group under the run's source policy, and publishes the members
together, or records each member's failure.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import duckdb

from qavf.bounded_batches import BoundedBatches
from qavf.engine.connections import merge_connection_binding, merge_secret_bindings
from qavf.engine.resources import ResourceLimitError
from qavf.engine.snapshot_staging import MemberPreparationError, SnapshotStage
from qavf.engine.source_execution import LatestAttempt
from qavf.settings import ConnectionSpec, SecretSpec
from qavf.sources.base import SourceError, SourceIntegrityError, SourceNotFoundError
from qavf.sources.collection import collect_result
from qavf.sources.errors import redacted_error
from qavf.sources.native import read_error

if TYPE_CHECKING:
    from qavf.engine.catalog import DatasetCatalog
    from qavf.engine.connections import SqlConnections
    from qavf.engine.resources import RunResources
    from qavf.sources.execution import SourceOperations


def _transaction_usable(con: Any) -> bool:
    """Whether a snapshot transaction survived a failed member read."""
    try:
        con.execute("SELECT 1").fetchall()
    except duckdb.Error:
        return False
    return True


_PLANNING_ERRORS = (duckdb.BinderException, duckdb.CatalogException, duckdb.ParserException)


def _failed_before_execution(exc: BaseException) -> bool:
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


class SnapshotCoordinator:
    """Reads and publishes a run's snapshot groups.

    Args:
        catalog: The run's dataset catalog (members, failures, snapshot metadata).
        resources: The run's resources: its frame cache and memory target.
        operations: The run's source policy (retries, identity, cancellation).
        connections: The run's SQL connections (a temporary one per attempt).
        lock: The catalog lock publications are made under.
        key_lock: Holds the per-key lock of one snapshot group.
        check_cancelled: Raises once the run or invocation is cancelled.
        require_available: Raises for a member released or no longer declared.
    """

    def __init__(
        self,
        catalog: DatasetCatalog,
        resources: RunResources,
        operations: SourceOperations,
        connections: SqlConnections,
        lock: threading.RLock,
        key_lock: Callable[[Any], contextlib.AbstractContextManager[None]],
        check_cancelled: Callable[[], None],
        require_available: Callable[[str], None],
    ) -> None:
        self._catalog = catalog
        self._resources = resources
        self._frames = resources.frames
        self._operations = operations
        self._connections = connections
        self._lock = lock
        self._key_lock = key_lock
        self._check_cancelled = check_cancelled
        self._require_available = require_available

    def members(self, connection: str) -> list[str]:
        """The datasets read through ``connection``'s snapshot, sorted."""
        return sorted(
            name
            for name, source in self._catalog.sources.items()
            if source.snapshot_connection() == connection
        )

    def load(self, connection: str) -> None:
        """Publish only a complete, policy-verified snapshot attempt.

        Permanent unavailable members remain individually unavailable; transient
        failures restart the entire transaction. Retry and identity decisions
        belong to the same coordinator as eager and native source execution.
        A group that fits the memory allowance is published from memory; a larger
        one is staged as owned spill files and published together as spilled cache
        entries, so the group is bounded by spill space, not memory.
        """
        with self._operations.guard(), self._key_lock(("snapshot", connection)):
            members = self.members(connection)
            with self._lock:
                if all(
                    name in self._frames or self._catalog.failure(name) is not None
                    for name in members
                ):
                    return
            attempts = LatestAttempt(functools.partial(self.attempt, connection, members))
            try:
                stage = self._operations.run(
                    [self._catalog.sources[name] for name in members], attempts
                )
                self._check_cancelled()
                publish_metadata = functools.partial(
                    self._publish_metadata, connection, members, stage
                )
                if stage.spilled:
                    self._frames.adopt(stage.spills(), publish_metadata, self._lock)
                    stage.detach()
                else:
                    self._frames.publish(stage.frames(), publish_metadata, self._lock)
            except Exception as exc:
                self._check_cancelled()
                failure = redacted_error(
                    SourceError(f"snapshot {connection!r} was not assessed: {exc}")
                )
                with self._lock:
                    for name in members:
                        self._catalog.record_failure(name, failure)
            finally:
                attempts.close()

    def attempt(self, connection: str, members: list[str]) -> SnapshotStage:
        """Read one transaction into memory or owned spill files without publishing.

        Each member streams in bounded Arrow batches. Members stay in memory while
        the group fits within the cache limit and memory target; past that, the
        attempt holds about one batch in memory at a time. A failed or cancelled
        attempt deletes its files.
        """
        bindings: dict[str, ConnectionSpec] = {}
        secrets: dict[str, SecretSpec] = {}
        for name in members:
            merge_connection_binding(bindings, connection, self._catalog.sources[name].env)
            merge_secret_bindings(secrets, self._catalog.sources[name].env)
        failures: dict[str, SourceError] = {}
        pending = list(members)
        with self._connections.temporary([connection], bindings, secrets) as con:
            # A permanent member failure that aborts the transaction (DuckDB 2.0
            # aborts on binder errors) restarts the read without that member, so
            # every published member still comes from one transaction.
            while True:
                stage = SnapshotStage(
                    self._frames.new_spill,
                    self._catalog.prepare,
                    datetime.now(UTC),
                    self.memory_bytes(),
                )
                stage.failures = failures
                aborted = False
                try:
                    con.execute("BEGIN TRANSACTION")
                    for name in pending:
                        self._check_cancelled()
                        try:
                            self.stage_member(stage, con, name)
                        except (ResourceLimitError, MemberPreparationError):
                            raise
                        except Exception as exc:
                            error = exc if isinstance(exc, SourceError) else redacted_error(exc)
                            if isinstance(error, SourceIntegrityError) or error.retryable:
                                raise error from None
                            failures[name] = redacted_error(
                                error,
                                SourceNotFoundError
                                if isinstance(error, SourceNotFoundError)
                                else SourceError,
                            )
                            if not _transaction_usable(con):
                                # DuckDB 2.0 aborts on binder errors too; only a
                                # failure before execution may restart without it.
                                if not _failed_before_execution(exc):
                                    raise failures.pop(name) from None
                                aborted = True
                                break
                            continue
                    if not aborted:
                        con.execute("COMMIT")
                        return stage
                except BaseException as exc:
                    with contextlib.suppress(duckdb.Error):
                        con.rollback()
                    stage.close()
                    if isinstance(exc, MemberPreparationError):
                        raise exc.error from None
                    raise
                with contextlib.suppress(duckdb.Error):
                    con.rollback()
                stage.close()
                pending = [name for name in pending if name not in failures]

    def stage_member(self, stage: SnapshotStage, con: Any, name: str) -> None:
        """Stream one member inside the snapshot transaction, admitting its row budget."""
        source = self._catalog.sources[name]
        budget = source.params.row_budget
        try:
            relation = source.snapshot_relation(con)
            if relation is None:
                frame = collect_result(source.read(con), budget)
                self.admit_rows(name, budget, frame.height)
                self._check_cancelled()
                stage.stage_frame(name, frame)
                return
            admit = functools.partial(self.admit_rows, name, budget)
            with BoundedBatches(relation, admit, self._check_cancelled) as batches:
                stage.stage_batches(name, batches)
        except duckdb.Error as exc:
            raise read_error(exc) from exc

    def memory_bytes(self) -> int:
        """The decoded bytes a snapshot group may hold in memory before it spills."""
        target = self._resources.memory_bytes
        return self._frames.limit if target is None else min(self._frames.limit, target)

    @staticmethod
    def admit_rows(name: str, budget: int | None, rows: int) -> None:
        """Refuse a member past its row budget (``rows`` read so far)."""
        if budget is not None and rows > budget:
            raise SourceError(f"dataset {name!r} has more than {budget:,} rows (row_budget)")

    def _publish_metadata(self, connection: str, members: list[str], stage: SnapshotStage) -> None:
        """Record a snapshot attempt's member failures and shapes at publication."""
        self._check_cancelled()
        for name in stage.members:
            self._require_available(name)
        for name, failure in stage.failures.items():
            self._catalog.record_failure(name, failure)
        for name, member in stage.members.items():
            self._catalog.record_snapshot(
                name, member.schema, member.rows, connection, members, stage.started_at
            )


__all__ = ["SnapshotCoordinator"]
