"""Worker-owned SQL connections and operation-scoped dataset registrations.

This owner knows connection lifetimes, attachment identity and retained bindings.
Dataset loading, source integrity and rule policy remain with DataContext.
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import Any

import duckdb
import polars as pl
import pyarrow as pa

from qavf.engine.resources import retained_size
from qavf.engine.session import RunSession
from qavf.settings import ConnectionSpec
from qavf.sources.base import SourceEnvironment, SourceError
from qavf.sources.native import OwnedConnection, connect
from qavf.sources.sql import quote_identifier

Connection = duckdb.DuckDBPyConnection | OwnedConnection


def merge_connection_binding(
    bindings: dict[str, ConnectionSpec], name: str, env: SourceEnvironment
) -> None:
    """Resolve one attachment in its source environment; reject conflicting aliases."""
    spec = env.connections.get(name)
    if spec is None:
        raise SourceError(f"unknown connection {name!r} in the source environment")
    if spec.path is not None:
        spec = spec.model_copy(update={"path": str(env.resolve(spec.path))})
    prior = bindings.get(name)
    if prior is not None and prior != spec:
        raise SourceError(f"conflicting connection bindings for {name!r} in one operation")
    bindings[name] = spec


class SqlConnections:
    """Own at most one reusable SQL connection per worker, plus scoped snapshots."""

    def __init__(self, env: SourceEnvironment, session: RunSession) -> None:
        self.env = env
        self.session = session
        self._local = threading.local()
        self._lock = threading.RLock()
        self._open: list[Connection] = []
        self._owners: dict[int, list[Connection]] = {}
        self._registration_bytes: dict[int, int] = {}
        self._peak_registration_bytes = 0
        self._registration_count = 0

    def connection(
        self,
        attach: Iterable[str],
        bindings: Mapping[str, ConnectionSpec] | None = None,
    ) -> Connection:
        """Reuse this worker's connection only for identical normalized attachments."""
        cache: dict[tuple, Connection] | None = getattr(self._local, "connections", None)
        if cache is None:
            cache = self._local.connections = {}
        aliases = tuple(sorted(attach))
        effective = dict(bindings) if bindings is not None else {}
        if bindings is None:
            for name in aliases:
                merge_connection_binding(effective, name, self.env)
        # Hash specifications rather than retaining credential-bearing JSON keys.
        key = tuple(
            (name, hashlib.sha256(effective[name].model_dump_json().encode()).hexdigest())
            for name in aliases
        )
        self.session.check()
        con = cache.get(key)
        if con is None:
            # Attachment combinations must not grow an unbounded connection cache.
            for old in cache.values():
                with contextlib.suppress(duckdb.Error):
                    old.close()
                with self._lock:
                    if old in self._open:
                        self._open.remove(old)
                    owners = self._owners.get(threading.get_ident(), [])
                    if old in owners:
                        owners.remove(old)
            cache.clear()
            self._local.registered = set()
            con = cache[key] = connect(replace(self.env, connections=effective), aliases)
            with self._lock:
                self._open.append(con)
                self._owners.setdefault(threading.get_ident(), []).append(con)
        return con

    @contextlib.contextmanager
    def temporary(
        self, attach: Iterable[str], bindings: Mapping[str, ConnectionSpec]
    ) -> Iterator[Connection]:
        """Keep transaction connections owned until their attempt has actually ended."""
        con = connect(replace(self.env, connections=bindings), attach)
        owner = threading.get_ident()
        with self._lock:
            self._owners.setdefault(owner, []).append(con)
        try:
            yield con
        finally:
            with self._lock:
                owners = self._owners.get(owner, [])
                if con in owners:
                    owners.remove(con)
            con.close()

    @contextlib.contextmanager
    def operation(self) -> Iterator[None]:
        """Release every registration even when dataset binding itself fails."""
        try:
            yield
        finally:
            self.discard(getattr(self._local, "registered", set()).copy())
            self._local.registered = set()
            with self._lock:
                self._registration_bytes.pop(threading.get_ident(), None)

    def remember(self, name: str) -> None:
        """Track a bound relation immediately after its registration succeeds."""
        registered: set[str] = getattr(self._local, "registered", set())
        registered.add(name)
        self._local.registered = registered

    def discard(self, names: Iterable[str]) -> None:
        """Remove only this worker's known registrations from its current connection."""
        registered: set[str] = getattr(self._local, "registered", set())
        discarded = registered.intersection(names)
        connections = getattr(self._local, "connections", {})
        con = next(iter(connections.values()), None)
        if con is not None:
            for name in discarded:
                with contextlib.suppress(duckdb.Error):
                    con.unregister(name)
                with contextlib.suppress(duckdb.Error):
                    con.execute(f"DROP VIEW IF EXISTS {quote_identifier(name)}")
        registered.difference_update(discarded)
        self._local.registered = registered

    def register_frame(self, con: Connection, name: str, frame: pl.DataFrame) -> None:
        """Export Arrow columns before DuckDB binding, avoiding parallel export deadlocks."""
        table = pa.Table.from_arrays([column.to_arrow() for column in frame], names=frame.columns)
        con.register(name, table)
        self.remember(name)
        with self._lock:
            owner = threading.get_ident()
            self._registration_bytes[owner] = self._registration_bytes.get(
                owner, 0
            ) + retained_size(table)
            self._registration_count += 1
            self._peak_registration_bytes = max(
                self._peak_registration_bytes, sum(self._registration_bytes.values())
            )

    def finish_worker(self, owner: int | None = None) -> None:
        """Close connections only after their worker has stopped executing."""
        owner = threading.get_ident() if owner is None else owner
        with self._lock:
            connections = self._owners.pop(owner, [])
            self._open = [con for con in self._open if con not in connections]
        if owner == threading.get_ident():
            self._local.connections = {}
            self._local.registered = set()
        for connection in connections:
            with contextlib.suppress(Exception):
                connection.close()
        self.session.close_worker_connections(owner)

    def close(self) -> None:
        """Close reusable connections after active invocations have completed."""
        with self._lock:
            connections, self._open = self._open, []
        for con in connections:
            with contextlib.suppress(duckdb.Error):
                con.close()

    @property
    def open_count(self) -> int:
        with self._lock:
            return len(self._open)

    def usage(self) -> dict[str, Any]:
        """Return bounded lifetime telemetry without retaining a binding history."""
        with self._lock:
            return {
                "active_sql_registration_bytes_upper_bound": sum(self._registration_bytes.values()),
                "peak_sql_registration_bytes_upper_bound": self._peak_registration_bytes,
                "sql_frame_registrations": self._registration_count,
                "sql_registration_lifetime": "one complete SQL operation, then unregistered",
            }
