"""Invocation-scoped cancellation and ownership of native work."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False, slots=True)
class Invocation:
    """Identity belongs to one call, never to a recyclable operating-system thread ID."""

    cancelled: threading.Event = field(default_factory=threading.Event)
    thread_id: int | None = None


class RunSession:
    """Coordinate invocation lifetimes and every native connection lease in a run."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.closed = False
        self.active: dict[int, Invocation] = {}
        self.connections: dict[object, tuple[int, Any]] = {}

    def check(self) -> None:
        from qavf.engine.resources import ResourceLimitError

        with self.lock:
            invocation = self.active.get(threading.get_ident())
            if self.closed or (invocation is not None and invocation.cancelled.is_set()):
                raise ResourceLimitError(
                    "validation execution was cancelled or its context is closed"
                )

    def begin(self, invocation: Invocation | None = None) -> Invocation:
        invocation = invocation or Invocation()
        with self.lock:
            self.check()
            if invocation.cancelled.is_set():
                from qavf.engine.resources import ResourceLimitError

                raise ResourceLimitError("validation invocation was cancelled before starting")
            invocation.thread_id = threading.get_ident()
            self.active[invocation.thread_id] = invocation
        return invocation

    def end(self) -> bool:
        with self.lock:
            self.active.pop(threading.get_ident(), None)
            return self.closed and not self.active

    def cancel(self, invocation: Invocation) -> None:
        with self.lock:
            invocation.cancelled.set()
            connections = [
                connection
                for owner, connection in self.connections.values()
                if owner == invocation.thread_id and self.active.get(owner) is invocation
            ]
            # DuckDB interrupt is nonblocking. Keep the invocation transition lock
            # until it is requested so a late cancellation cannot hit the next call.
            for connection in connections:
                with contextlib.suppress(Exception):
                    connection.interrupt()

    def register_connection(self, token: object, connection: Any) -> None:
        with self.lock:
            self.connections[token] = (threading.get_ident(), connection)
            # Cancellation may race connection setup. Never admit late uncancelled work.
            self.check()

    def unregister_connection(self, token: object) -> None:
        with self.lock:
            self.connections.pop(token, None)

    def request_close(self) -> bool:
        with self.lock:
            self.closed = True
            return not self.active

    def close_worker_connections(self, owner: int) -> None:
        with self.lock:
            keys = [key for key, value in self.connections.items() if value[0] == owner]
            connections = [self.connections.pop(key)[1] for key in keys]
        for connection in connections:
            with contextlib.suppress(Exception):
                connection.close()

    def close_connections(self) -> None:
        with self.lock:
            connections = [value[1] for value in self.connections.values()]
            self.connections.clear()
        for connection in connections:
            with contextlib.suppress(Exception):
                connection.close()


class KeyedLocks:
    """Keep only in-flight keyed locks, including waiting callers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.entries: dict[Any, tuple[threading.RLock, int]] = {}

    @contextlib.contextmanager
    def hold(
        self, key: Any, *, check_cancelled: Callable[[], None] | None = None
    ) -> Iterator[None]:
        with self.lock:
            lock, users = self.entries.get(key, (threading.RLock(), 0))
            self.entries[key] = (lock, users + 1)
        acquired = False
        try:
            while not lock.acquire(timeout=0.05):
                if check_cancelled is not None:
                    check_cancelled()
            acquired = True
            if check_cancelled is not None:
                check_cancelled()
            yield
        finally:
            if acquired:
                lock.release()
            with self.lock:
                _, users = self.entries[key]
                if users == 1:
                    del self.entries[key]
                else:
                    self.entries[key] = (lock, users - 1)
