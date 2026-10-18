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
        if resources.memory_bytes is not None and resources.memory_bytes < 65_536:
            raise ResourceLimitError("indexed work requires at least 65,536 memory target bytes")
        self.connection = sqlite3.connect(self.path, isolation_level=None)
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

    def _plan(self, sql: str, parameters: Any) -> None:
        from qavf.engine.resources import ResourceLimitError

        for row in self.connection.execute("EXPLAIN QUERY PLAN " + sql, parameters):
            if "TEMP B-TREE" in str(row[-1]).upper():
                raise ResourceLimitError(
                    f"{self.label}: query requires an unbounded temporary B-tree; add an index"
                )

    def _run(self, sql: str, values: Any, *, many: bool) -> sqlite3.Cursor:
        from qavf.engine.resources import ResourceLimitError

        self.resources.session.check()
        if self._closed:
            raise RuntimeError("indexed work store is closed")
        budget = self.resources.spill_budget
        with budget.lock:
            # The allocator lock protects admission through the actual write;
            # another store cannot spend the same available disk bytes.
            pages = self.connection.execute("PRAGMA page_count").fetchone()[0]
            allowed = pages + budget.remaining // 4096
            if allowed < 1:
                raise ResourceLimitError(f"{self.label}: shared spill budget exhausted")
            self.connection.execute(f"PRAGMA max_page_count={allowed}")
            own_transaction = many and not self.connection.in_transaction
            try:
                if many:
                    if own_transaction:
                        self.connection.execute("BEGIN")
                    cursor = self.connection.executemany(sql, values)
                    if own_transaction:
                        self.connection.execute("COMMIT")
                else:
                    self._plan(sql, values)
                    cursor = self.connection.execute(sql, values)
            except BaseException as exc:
                if own_transaction and self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                if (
                    isinstance(exc, sqlite3.OperationalError)
                    and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL
                ):
                    raise ResourceLimitError(
                        f"{self.label}: shared spill budget exhausted"
                    ) from exc
                raise
            finally:
                actual = self.path.stat().st_size if self.path.exists() else 0
                if actual > self._charged:
                    budget.reserve(actual - self._charged)
                    self._charged = actual
        return cursor

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
        if self._closed:
            return
        self._closed = True
        try:
            self.connection.close()
        finally:
            self.path.unlink(missing_ok=True)
            self.resources.spill_budget.release(self._charged)
            self._charged = 0
