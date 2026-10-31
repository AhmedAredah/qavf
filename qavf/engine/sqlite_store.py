"""Disposable indexed work state with a shared disk quota and bounded page cache.

Only indexed query plans are admitted: a temporary sort/group B-tree would escape
the owned database page limit. Tables and indexes should be declared before data
is inserted. This is work state, not a durable transactional database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

if TYPE_CHECKING:
    from qavf.engine.resources import RunResources


class Store:
    """One owned SQLite database charged to ``RunResources.spill_budget``."""

    def __init__(self, resources: RunResources, label: str = "indexed work") -> None:
        from qavf.engine.resources import ResourceLimitError

        self.resources = resources
        self.label = label
        self.path: Path = resources.directory / f"work-{uuid4().hex}.sqlite"
        self._charged = 0
        self._closed = False
        self._connection_closed = False
        if resources.memory_bytes is not None and resources.memory_bytes < 65_536:
            raise ResourceLimitError("indexed work requires at least 65,536 memory target bytes")
        self.connection = sqlite3.connect(self.path, isolation_level=None)
        self.resources.own_cleanup(self)
        self.connection.execute("PRAGMA page_size=4096")
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA automatic_index=OFF")
        memory = resources.memory_bytes
        cache_kib = min(256, max(4, memory // 16 // 1024)) if memory is not None else 256
        self.connection.execute(f"PRAGMA cache_size=-{cache_kib}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _plan(self, sql: str, parameters: Any) -> bool:
        """Ask SQLite which capabilities a prepared statement uses; never parse SQL."""
        from qavf.engine.resources import ResourceLimitError

        readonly = True
        reads = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }

        def authorize(action, _first, _second, _database, _trigger):
            nonlocal readonly
            readonly &= action in reads
            return sqlite3.SQLITE_OK

        self.connection.set_authorizer(authorize)
        try:
            for row in self.connection.execute("EXPLAIN QUERY PLAN " + sql, parameters):
                if "TEMP B-TREE" in str(row[-1]).upper():
                    raise ResourceLimitError(
                        f"{self.label}: query requires an unbounded temporary B-tree; add an index"
                    )
        finally:
            self.connection.set_authorizer(None)
        # Result-bearing writes can outlive their admitted credit while their
        # cursor is being consumed. Work stores permit only completed writes.
        if not readonly and any(
            row[1] == "ResultRow" for row in self.connection.execute("EXPLAIN " + sql, parameters)
        ):
            raise ValueError("indexed work writes must not return result rows")
        return readonly

    def _run(self, sql: str, values: Any, *, many: bool) -> sqlite3.Cursor:
        from qavf.engine.resources import ResourceLimitError

        self.resources.session.check()
        if self._closed:
            raise RuntimeError("indexed work store is closed")
        if not many and self._plan(sql, values):
            return self.connection.execute(sql, values)
        budget = self.resources.spill_budget
        credit = budget.acquire_growth(self.resources.session.check)
        own_transaction = many and not self.connection.in_transaction
        try:
            pages = self.connection.execute("PRAGMA page_count").fetchone()[0]
            allowed = pages + credit // 4096
            if allowed < 1:
                raise ResourceLimitError(f"{self.label}: shared spill budget exhausted")
            self.connection.execute(f"PRAGMA max_page_count={allowed}")
            if many:
                if own_transaction:
                    self.connection.execute("BEGIN")
                cursor = self.connection.executemany(sql, values)
                if own_transaction:
                    self.connection.execute("COMMIT")
            else:
                cursor = self.connection.execute(sql, values)
            return cursor
        except BaseException as exc:
            if own_transaction and self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            if (
                isinstance(exc, sqlite3.OperationalError)
                and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL
            ):
                raise ResourceLimitError(f"{self.label}: shared spill budget exhausted") from exc
            raise
        finally:
            actual = self.path.stat().st_size if self.path.exists() else 0
            budget.settle_growth(credit, actual - self._charged)
            self._charged = actual

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        return self._run(sql, parameters, many=False)

    def executemany(self, sql: str, parameters: Iterable[Any]) -> sqlite3.Cursor:
        # Require bounded callers; sqlite itself consumes the iterable lazily.
        # The insert/update plan is identical for every binding.
        iterator = iter(parameters)
        try:
            first = next(iterator)
        except StopIteration:
            return self.connection.cursor()
        self._plan(sql, first)
        from itertools import chain

        return self._run(sql, chain((first,), iterator), many=True)

    def close(self) -> None:
        """Close and remove the store; a removal that fails stays charged and is
        retried by the next ``close``."""
        self._closed = True
        if not self._connection_closed:
            self.connection.close()
            self._connection_closed = True
        self._remove()
        self.resources.release_cleanup(self)

    def _remove(self) -> None:
        self.path.unlink(missing_ok=True)
        self.resources.spill_budget.release(self._charged)
        self._charged = 0
