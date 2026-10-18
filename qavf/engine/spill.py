"""Shared byte reservations for owned frame, observation and index files."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


class SpillBudget:
    """Coordinate exact owned-byte growth; native SQL retains its separate reservation."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = self.peak = 0
        self.lock = threading.RLock()

    @property
    def remaining(self) -> int:
        with self.lock:
            return self.limit - self.used

    def reserve(self, size: int) -> None:
        from qavf.engine.resources import ResourceLimitError

        if size < 0:
            raise ValueError("spill reservation must be nonnegative")
        with self.lock:
            if size > self.limit - self.used:
                raise ResourceLimitError("owned spill budget exhausted")
            self.used += size
            self.peak = max(self.peak, self.used)

    def release(self, size: int) -> None:
        with self.lock:
            if size < 0 or size > self.used:
                raise ValueError("invalid spill reservation release")
            self.used -= size


class SpillFile:
    """One sequential output with reservation-before-write and exactly-once cleanup."""

    def __init__(self, path: Path, budget: SpillBudget) -> None:
        self.path = path
        self.budget = budget
        self.size = 0
        self._closed = False
        self._references = 1
        self._error: BaseException | None = None

    @contextlib.contextmanager
    def writer(self) -> Iterator[BinaryIO]:
        owner = self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as handle:

            class Writer:
                def write(self, data):
                    try:
                        owner.budget.reserve(len(data))
                    except BaseException as exc:
                        owner._error = exc
                        raise
                    owner.size += len(data)
                    return handle.write(data)

                def tell(self):
                    return handle.tell()

                def flush(self):
                    return handle.flush()

            from typing import cast

            try:
                yield cast(BinaryIO, Writer())
            except BaseException:
                if self._error is not None:
                    raise self._error from None
                raise

    def close(self) -> None:
        with self.budget.lock:
            if self._closed:
                return
            self._closed = True
            self._release_reference()

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
                self.path.unlink(missing_ok=True)
                self.budget.release(self.size)
                self.size = 0

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

    def close(self) -> None:
        with self._owner.budget.lock:
            if not self._closed:
                self._closed = True
                self._owner._release_reference()
