"""Bounded retained caches and owned temporary storage.

These limits bound tracked retention and spill, not arbitrary native allocations or
caller-owned frames. Whole-process RSS requires an external process/container limit.
"""

from __future__ import annotations

import contextlib
import functools
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import polars as pl

from qavf.engine.cache_ownership import (
    _snapshot,
    compact_polars_with_size,
    retained_size,
)
from qavf.engine.owned_ipc import project_ipc, reduce_ipc
from qavf.engine.retention import RetainedBudget, RetainedLease
from qavf.engine.session import RunSession
from qavf.engine.spill import SpillBudget, SpillFile, SpillLease


class ResourceLimitError(RuntimeError):
    """A configured retention or spill budget was exceeded."""


def _locked(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return guarded


class ByteCache(MutableMapping):
    """Size-aware LRU for recomputable objects; oversized objects are not retained."""

    def __init__(self, limit: int, budget: RetainedBudget | None = None) -> None:
        self.lock = threading.RLock()
        self.limit = limit
        self.data: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self.bytes = self.peak_bytes = 0
        self.budget = budget or RetainedBudget(limit)

    @_locked
    def __getitem__(self, key: Any) -> Any:
        value, _ = self.data[key]
        self.data.move_to_end(key)
        return _snapshot(value)

    @_locked
    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self.data:
            del self[key]
        # Native frame sizing can require an IPC traversal. Compact and measure it
        # together below instead of serializing the same nested payload twice.
        size = retained_size(key) + (
            int(value.estimated_size())
            if isinstance(value, pl.DataFrame | pl.Series)
            else retained_size(value)
        )
        if size > self.limit:
            return
        try:
            value = _snapshot(value, compact=True)
        except (TypeError, ValueError, RecursionError, NotImplementedError):
            return
        # Measure our detached allocation, not the caller's visible slice.
        size = retained_size(key) + retained_size(value)
        if size > self.limit:
            return
        while self.data and (
            self.bytes + size > self.limit or size > self.budget.remaining
        ):
            _, (_, removed) = self.data.popitem(last=False)
            self.bytes -= removed
            self.budget.release(removed)
        if not self.budget.reserve(size):
            return
        self.data[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    @_locked
    def __delitem__(self, key: Any) -> None:
        _, size = self.data.pop(key)
        self.bytes -= size
        self.budget.release(size)

    @_locked
    def __iter__(self) -> Iterator:
        return iter(tuple(self.data))

    @_locked
    def __len__(self) -> int:
        return len(self.data)

    @_locked
    def __contains__(self, key: object) -> bool:
        return key in self.data

    @_locked
    def clear(self) -> None:
        self.data.clear()
        self.budget.release(self.bytes)
        self.bytes = 0

    @_locked
    def discard(self, key: Any) -> None:
        """Release a value without making the detached copy required by pop()."""
        if key in self.data:
            del self[key]


class FrameStore(MutableMapping[str, pl.DataFrame]):
    """Retain frames within bytes; evict exact immutable snapshots to owned IPC files."""

    def __init__(
        self,
        cache_bytes: int,
        spill_bytes: int,
        directory: Path,
        materialization_bytes: int | None = None,
        spill_budget: SpillBudget | None = None,
        retained_budget: RetainedBudget | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.limit = cache_bytes
        self.spill_limit = spill_bytes
        self.spill_budget = spill_budget or SpillBudget(spill_bytes)
        self.retained_budget = retained_budget or RetainedBudget(cache_bytes)
        self._spill_files: dict[str, SpillFile] = {}
        self.directory = directory / "frames"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.materialization_bytes = materialization_bytes
        self.data: OrderedDict[str, pl.DataFrame] = OrderedDict()
        self._memory_leases: dict[str, RetainedLease] = {}
        self.paths: dict[str, Path] = {}
        self.sizes: dict[str, int] = {}
        self.bytes = self.spill_bytes = self.peak_bytes = self.peak_spill_bytes = 0
        self.spill_read_count = self.spill_read_bytes = self.spill_promotions = 0
        self.spill_write_count = self.spill_write_bytes = 0
        self.spill_project_count = self.spill_reduce_count = 0

    @_locked
    def __getitem__(self, key: str) -> pl.DataFrame:
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key].clone()
        # Read owned bytes: native IPC preserves Int128/Enum, and no file mapping
        # can outlive cleanup. Count the actual bytes read, including IPC metadata.
        with self.paths[key].open("rb") as handle:
            payload = handle.read()
        self.spill_read_count += 1
        self.spill_read_bytes += len(payload)
        frame = pl.read_ipc(payload)
        # Promote only into existing free space. A hit must not trigger eviction,
        # extra writes or ping-pong between two frames larger than the allocation.
        if self.sizes[key] <= self.limit - self.bytes:
            compacted, size = compact_polars_with_size(frame)
            owned = cast(pl.DataFrame, compacted)
            if size <= self.limit - self.bytes and self.retained_budget.reserve(size):
                self.data[key] = owned
                self._memory_leases[key] = RetainedLease(self.retained_budget, size)
                self.sizes[key] = size
                self.bytes += size
                self.peak_bytes = max(self.peak_bytes, self.bytes)
                path = self.paths.pop(key)
                self.spill_bytes -= path.stat().st_size
                self._spill_files.pop(key).close()
                self.spill_promotions += 1
                return owned.clone()
        return frame

    @_locked
    def project(self, key: str, columns: Iterable[str]) -> pl.DataFrame:
        """Read only selected IPC columns; never promote or map an owned spill.

        A native path enables IPC projection. A Python file-like wrapper would
        first buffer the complete file in supported Polars versions. Its physical
        read bytes are therefore deliberately not guessed or added to the exact
        full-buffer byte counter.
        """
        selected = list(columns)
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key].select(selected)
        self.spill_project_count += 1
        return project_ipc(self.paths[key], selected)

    def reduce(self, key: str, build: Callable[[pl.LazyFrame], pl.LazyFrame]) -> pl.DataFrame:
        """Lease input storage, then build and collect without holding the cache lock.

        The callback must return a lazy plan, not retain it. No file-backed plan
        or memory mapping is returned. A lease protects the input from concurrent
        release while telemetry and unrelated cache operations remain available.
        """
        frame, lease = self.observe(key)
        try:
            if frame is not None:
                return build(frame.lazy()).collect(engine="streaming")
            assert isinstance(lease, SpillLease)
            return reduce_ipc(lease.path, build)
        finally:
            if isinstance(lease, RetainedLease):
                lease.release()
            elif lease is not None:
                lease.close()

    @_locked
    def observe(
        self, key: str
    ) -> tuple[pl.DataFrame | None, SpillLease | RetainedLease | None]:
        """Lease immutable buffers/IPC for one scoped lazy reduction operation.

        A spill lease survives eviction, promotion and release of its cache entry.
        Its physical bytes remain charged until the last lease closes.
        """
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key].clone(), self._memory_leases[key].borrow()
        self.spill_reduce_count += 1
        return None, self._spill_files[key].borrow()

    @_locked
    def __setitem__(self, key: str, value: pl.DataFrame) -> None:
        owned, size = compact_polars_with_size(value)
        value = cast(pl.DataFrame, owned)
        if self.materialization_bytes is not None and size > self.materialization_bytes:
            raise ResourceLimitError(
                f"dataset {key!r} materializes {size} bytes; "
                f"memory target is {self.materialization_bytes}"
            )
        if key in self:
            del self[key]
        while self.data and (
            self.bytes + size > self.limit or size > self.retained_budget.remaining
        ):
            old, frame = next(iter(self.data.items()))
            self._spill(old, frame)
            self.data.pop(old)
            self.bytes -= self.sizes[old]
            self._memory_leases.pop(old).release()
        if size > self.limit or not self.retained_budget.reserve(size):
            self._spill(key, value)
        else:
            self.data[key] = value
            self._memory_leases[key] = RetainedLease(self.retained_budget, size)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.sizes[key] = size

    def _spill(self, key: str, frame: pl.DataFrame) -> None:
        if frame.estimated_size() > min(
            self.spill_limit - self.spill_bytes, self.spill_budget.remaining
        ):
            raise ResourceLimitError("frame spill budget exhausted")
        owned = SpillFile(self.directory / f"{uuid4().hex}.arrow", self.spill_budget)
        try:
            with owned.writer() as handle:
                frame.write_ipc(handle, compression="uncompressed")
            self.paths[key] = owned.path
            self._spill_files[key] = owned
            self.spill_bytes += owned.size
            self.peak_spill_bytes = max(self.peak_spill_bytes, self.spill_bytes)
            self.spill_write_count += 1
            self.spill_write_bytes += owned.size
        except BaseException:
            owned.close()
            raise

    @_locked
    def __delitem__(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
            size = self.sizes.pop(key)
            self.bytes -= size
            self._memory_leases.pop(key).release()
        elif key in self.paths:
            path = self.paths.pop(key)
            self.sizes.pop(key)
            self.spill_bytes -= path.stat().st_size
            self._spill_files.pop(key).close()
        else:
            raise KeyError(key)

    @_locked
    def __iter__(self) -> Iterator[str]:
        return iter((*self.data, *self.paths))

    @_locked
    def __len__(self) -> int:
        return len(self.data) + len(self.paths)

    @_locked
    def __contains__(self, key: object) -> bool:
        return key in self.data or key in self.paths

    @_locked
    def clear(self) -> None:
        for owned in self._spill_files.values():
            owned.close()
        self._spill_files.clear()
        for lease in self._memory_leases.values():
            lease.release()
        self._memory_leases.clear()
        self.data.clear()
        self.paths.clear()
        self.sizes.clear()
        self.bytes = self.spill_bytes = 0

    @_locked
    def discard(self, key: str) -> None:
        if key in self:
            del self[key]


class RunResources:
    """Own a run's temporary directory and expose honest resource diagnostics."""

    def __init__(
        self,
        cache_bytes: int,
        spill_bytes: int,
        memory_bytes: int | None,
        temp_directory: Path | None = None,
        connection_slots: int = 16,
    ) -> None:
        if temp_directory is not None:
            Path(temp_directory).mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="qavf-", dir=temp_directory))
        with contextlib.ExitStack() as setup:
            setup.callback(shutil.rmtree, self.directory, ignore_errors=True)
            self.session = RunSession()
            # DuckDB creates its connection-specific leaf on first spill, but
            # does not create missing ancestors of that leaf.
            (self.directory / "duckdb").mkdir()
            self.connection_slots = max(1, connection_slots)
            self.memory_bytes = memory_bytes
            self.reservations: set[str] = set()
            self.peak_connections = 0
            self.connection_spill_bytes = (spill_bytes - spill_bytes // 2) // self.connection_slots
            self.connection_memory_bytes = (
                memory_bytes // 2 // self.connection_slots if memory_bytes is not None else None
            )
            cache_bytes = (
                min(cache_bytes, memory_bytes // 2) if memory_bytes is not None else cache_bytes
            )
            self.retained_budget = RetainedBudget(cache_bytes)
            self.spill_budget = SpillBudget(spill_bytes // 2)
            try:
                self.frames = FrameStore(
                    cache_bytes * 3 // 4,
                    spill_bytes // 2,
                    self.directory,
                    memory_bytes,
                    self.spill_budget,
                    self.retained_budget,
                )
            except BaseException:
                shutil.rmtree(self.directory, ignore_errors=True)
                raise
            self.pandas = ByteCache(cache_bytes // 8, self.retained_budget)
            self.memo = ByteCache(
                cache_bytes - cache_bytes * 3 // 4 - cache_bytes // 8, self.retained_budget
            )
            self.spill_limit = spill_bytes
            self.eager_fallbacks: set[str] = set()
            self.lock = threading.RLock()
            self.closed = False
            self._last_usage: dict[str, Any] = {}
            self.usage()
            setup.pop_all()

    def acquire(self) -> tuple[str, int, int | None]:
        """Reserve a fixed slice; total native limits cannot exceed the run envelope."""
        with self.lock:
            self.session.check()
            if self.closed:
                raise ResourceLimitError("run resources have been closed")
            if self.connection_memory_bytes == 0:
                raise ResourceLimitError(
                    "memory_budget_bytes is too small to reserve one native byte per connection; "
                    "increase the memory target or reduce max_workers"
                )
            if len(self.reservations) >= self.connection_slots:
                raise ResourceLimitError(
                    "run connection capacity exhausted; close unused connections"
                )
            token = uuid4().hex
            self.reservations.add(token)
            self.peak_connections = max(self.peak_connections, len(self.reservations))
            return token, self.connection_spill_bytes, self.connection_memory_bytes

    def release(self, token: object) -> None:
        with self.lock:
            self.reservations.discard(str(token))

    def register_connection(self, token: object, connection: Any) -> None:
        self.session.register_connection(token, connection)

    def unregister_connection(self, token: object) -> None:
        self.session.unregister_connection(token)

    def check_cancelled(self) -> None:
        """Check an owned native operation at its cooperative entry boundary."""
        self.session.check()

    def usage(self, *, blocking: bool = True) -> dict[str, Any]:
        # Cache mutations acquire their own lock before the shared allocator.
        # An overdue worker may be compacting/writing while holding a cache lock.
        # Reporting must be able to use the last complete, explicitly stale
        # snapshot instead of waiting for that worker's native operation.
        with contextlib.ExitStack() as held:
            for lock in (self.frames.lock, self.pandas.lock, self.memo.lock):
                if not lock.acquire(blocking=blocking):
                    return {**self._last_usage, "resource_usage_pending": True}
                held.callback(lock.release)
            cached = self.frames.bytes + self.pandas.bytes + self.memo.bytes
            retained, retained_peak = self.retained_budget.snapshot()
        result = {
            "resource_usage_pending": False,
            "active_connection_reservations": len(self.reservations),
            "peak_connection_reservations": self.peak_connections,
            "reserved_native_spill_bytes": len(self.reservations) * self.connection_spill_bytes,
            "native_spill_budget_bytes": self.connection_slots * self.connection_spill_bytes,
            "native_memory_budget_bytes": self.connection_slots * self.connection_memory_bytes
            if self.connection_memory_bytes is not None
            else None,
            "native_memory_bytes_per_connection": self.connection_memory_bytes,
            "native_memory_budget_scope": (
                "sum of fixed buffer-manager reservation ceilings; an explicit lower "
                "DuckDB memory limit may reduce each actual limit; not process RSS"
            ),
            "retained_cache_bytes": cached,
            "retained_observation_bytes": retained - cached,
            "retained_total_bytes": retained,
            "peak_retained_total_bytes": retained_peak,
            "peak_retained_cache_bytes_upper_bound": self.frames.peak_bytes
            + self.pandas.peak_bytes
            + self.memo.peak_bytes,
            "owned_spill_bytes": self.spill_budget.used,
            "peak_owned_spill_bytes": self.spill_budget.peak,
            "owned_spill_budget_bytes": self.spill_budget.limit,
            "frame_spill_bytes": self.frames.spill_bytes,
            "peak_frame_spill_bytes": self.frames.peak_spill_bytes,
            "frame_spill_read_count": self.frames.spill_read_count,
            "frame_spill_read_bytes": self.frames.spill_read_bytes,
            "frame_spill_read_bytes_scope": (
                "exact full-buffer reads only; excludes native projected/reduced reads"
            ),
            "frame_spill_project_count": self.frames.spill_project_count,
            "frame_spill_reduce_count": self.frames.spill_reduce_count,
            "frame_spill_projected_read_bytes": None,
            "frame_spill_promotions": self.frames.spill_promotions,
            "frame_spill_write_count": self.frames.spill_write_count,
            "frame_spill_write_bytes": self.frames.spill_write_bytes,
            "eager_fallbacks": sorted(self.eager_fallbacks),
            "memory_limit_scope": (
                "tracked retained caches and post-collection target; not whole-process RSS"
            ),
            "caller_owned_inputs": "may already be resident before validation starts",
        }
        self._last_usage = result
        return dict(result)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
        # Native close may call release(); never hold the resource lock while
        # waiting on a handle another thread may already be closing.
        self.session.request_close()
        self.session.close_connections()
        self.frames.clear()
        self.pandas.clear()
        self.memo.clear()
        shutil.rmtree(self.directory, ignore_errors=True)
