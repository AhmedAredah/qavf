"""Shared byte reservations for owned frame, observation and index files."""

from __future__ import annotations

import contextlib
import io
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO, cast

from qavf.engine.cleanup import attempt_cleanup, raise_cleanup_errors


class SpillBudget:
    """Coordinate exact owned-byte growth; native SQL retains its separate reservation."""

    def __init__(self, limit: int, max_files: int = 4096) -> None:
        self.limit = limit
        self.max_files = max_files
        self.peak_files = 0
        self.used = self.peak = 0
        self.lock = threading.RLock()
        self._credits = 0
        self._holders: dict[int, int] = {}
        """Unsettled growth credits by thread: a credit lends out all free space."""
        self._available = threading.Condition(self.lock)
        self._files: set[SpillFile] = set()

    def register(self, owner: SpillFile) -> None:
        from qavf.engine.resources import ResourceLimitError

        with self.lock:
            if len(self._files) >= self.max_files:
                raise ResourceLimitError("owned spill file-count budget exhausted")
            self._files.add(owner)
            self.peak_files = max(self.peak_files, len(self._files))

    def file_usage(self) -> dict[str, int | str]:
        with self.lock:
            return {
                "owned_spill_files": len(self._files),
                "peak_owned_spill_files": self.peak_files,
                "owned_spill_file_limit": self.max_files,
                "owned_spill_byte_scope": "logical file bytes; excludes filesystem allocation",
            }

    def close_files(self) -> None:
        """Retry every owned file, including failed unpublished writes.

        A file with live leases remains registered and charged after its owner
        closes. Its final lease removes it, or a later call retries removal.
        """
        with self.lock:
            owners = tuple(self._files)
        raise_cleanup_errors(
            "spill files could not be released", attempt_cleanup(owner.close for owner in owners)
        )

    @property
    def has_files(self) -> bool:
        """Whether any live or pending file still has an owner, even at zero bytes."""
        with self.lock:
            return bool(self._files)

    @property
    def remaining(self) -> int:
        with self.lock:
            return self.limit - self.used

    def reserve(self, size: int) -> None:
        from qavf.engine.resources import ResourceLimitError

        if size < 0:
            raise ValueError("spill reservation must be nonnegative")
        with self._available:
            # Space lent to another thread's native write returns when it
            # settles: wait for it rather than fail. Never wait on this thread's
            # own credit, which only this thread can settle.
            while size > self.limit - self.used and self._lent_elsewhere():
                self._available.wait(0.05)
            if size > self.limit - self.used:
                raise ResourceLimitError("owned spill budget exhausted")
            self.used += size
            self.peak = max(self.peak, self.used)

    def _lent_elsewhere(self) -> bool:
        return self._credits > self._holders.get(threading.get_ident(), 0)

    def release(self, size: int) -> None:
        with self.lock:
            if size < 0 or size > self.used:
                raise ValueError("invalid spill reservation release")
            self.used -= size
            self._available.notify_all()

    def acquire_growth(self, check_cancelled) -> int:
        """Reserve currently available growth, waiting only for unsettled credit.

        Native execution runs outside the allocator lock. Existing-page reads
        need no credit. No failed SQL statement is replayed to extend a quota.
        """
        with self._available:
            while self.used == self.limit and self._credits:
                check_cancelled()
                self._available.wait(0.05)
            check_cancelled()
            credit = self.limit - self.used
            self.used += credit
            self.peak = max(self.peak, self.used)
            self._credits += 1
            thread = threading.get_ident()
            self._holders[thread] = self._holders.get(thread, 0) + 1
            return credit

    def settle_growth(self, credit: int, growth: int) -> None:
        with self._available:
            try:
                if growth > credit:
                    raise RuntimeError("native file grew beyond its admitted credit")
                self.release(credit - growth)
            finally:
                self._credits -= 1
                thread = threading.get_ident()
                if self._holders.get(thread, 0) > 1:
                    self._holders[thread] -= 1
                else:
                    self._holders.pop(thread, None)
                self._available.notify_all()


class _BudgetWriter(io.RawIOBase):
    """One sequential handle; failures remain latched even if a consumer suppresses them."""

    def __init__(self, owner: SpillFile, handle: BinaryIO) -> None:
        self.owner, self.handle = owner, handle
        self._lock = threading.RLock()

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.handle.tell()

    def _check(self) -> None:
        if self.closed or self.owner._closed:
            raise ValueError("spill writer has ended")
        if self.owner._error is not None:
            raise self.owner._error

    def write(self, data) -> int:
        with self._lock:
            try:
                self._check()
                view = memoryview(data).cast("B")
                size = view.nbytes
                self.owner.budget.reserve(size)
                self.owner.size += size
                before = self.handle.tell()
                try:
                    written = self.handle.write(view)
                except BaseException:
                    # The unbuffered file's position reveals any partial write.
                    actual = max(0, min(size, self.handle.tell() - before))
                    self.owner.budget.release(size - actual)
                    self.owner.size -= size - actual
                    raise
                actual = 0 if written is None else written
                self.owner.budget.release(size - actual)
                self.owner.size -= size - actual
                if actual != size:
                    raise OSError("short spill write")
                return actual
            except BaseException as exc:
                if self.owner._error is None:
                    self.owner._error = exc
                raise

    def flush(self) -> None:
        with self._lock:
            if self.closed or self.handle.closed:
                return
            try:
                self.handle.flush()
            except BaseException as exc:
                if self.owner._error is None:
                    self.owner._error = exc
                raise


class SpillFile:
    """One sequential output with reservation-before-write and exactly-once cleanup.

    Releasing the last reference and removing the file are separate steps: a
    removal that fails (a transient Windows sharing lock) keeps the file pending,
    with its bytes still charged, and any later ``close`` retries it.
    """

    def __init__(self, path: Path, budget: SpillBudget) -> None:
        self.path = path
        self.budget = budget
        self.size = 0
        self._closed = False
        self._write_started = False
        self._references = 1
        self._pending_removal = False
        self._error: BaseException | None = None
        budget.register(self)

    @contextlib.contextmanager
    def writer(self) -> Iterator[BinaryIO]:
        """Acquire the sole writer lease; publication requires complete success."""
        with self.budget.lock:
            if self._closed or self._write_started:
                raise RuntimeError("spill writer is closed or has already been acquired")
            self._write_started = True
            self._references += 1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb", buffering=0) as handle:
                writer = _BudgetWriter(self, handle)
                try:
                    try:
                        yield cast(BinaryIO, writer)
                        writer.flush()
                        if self._closed:
                            raise RuntimeError("spill owner closed during writing")
                    except BaseException:
                        if self._error is not None:
                            raise self._error from None
                        raise
                    if self._error is not None:
                        raise self._error
                finally:
                    writer.close()
        finally:
            self._release_reference()

    def close(self) -> None:
        with self.budget.lock:
            if not self._closed:
                self._closed = True
                self._release_reference()
            else:
                self.remove_pending()

    @property
    def pending_removal(self) -> bool:
        """Whether every reference has ended but the file could not yet be removed."""
        with self.budget.lock:
            return self._pending_removal

    def remove_pending(self) -> None:
        """Retry a removal that failed; release the bytes once the file is gone."""
        with self.budget.lock:
            if not self._pending_removal:
                return
            self.path.unlink(missing_ok=True)
            self._pending_removal = False
            self.budget.release(self.size)
            self.size = 0
            self.budget._files.discard(self)

    def borrow(self) -> SpillLease:
        with self.budget.lock:
            if self._closed:
                raise RuntimeError("owned spill file has ended")
            self._references += 1
            return SpillLease(self)

    def _release_reference(self) -> None:
        with self.budget.lock:
            self._references -= 1
            if self._references == 0:
                self._pending_removal = True
                self.remove_pending()

    def __enter__(self) -> SpillFile:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SpillLease:
    """Keep a spill alive independently of its cache entry, without copying bytes."""

    def __init__(self, owner: SpillFile) -> None:
        self._owner = owner
        self._closed = False

    @property
    def path(self) -> Path:
        return self._owner.path

    def borrow(self) -> SpillLease:
        with self._owner.budget.lock:
            if self._closed:
                raise RuntimeError("spill lease has ended")
            self._owner._references += 1
            return SpillLease(self._owner)

    def close(self) -> None:
        with self._owner.budget.lock:
            if not self._closed:
                self._closed = True
                self._owner._release_reference()
            else:
                self._owner.remove_pending()
