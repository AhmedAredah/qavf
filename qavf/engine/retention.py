"""Shared admission for retained caches and coherent in-memory observations."""

from __future__ import annotations

import threading


class RetainedBudget:
    """Reserve before retaining; optional values spill or remain uncached on refusal."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = self.peak = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit - self.used

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.used, self.peak

    def reserve(self, size: int) -> bool:
        if size < 0:
            raise ValueError("retained reservation must be nonnegative")
        with self._lock:
            if size > self.limit - self.used:
                return False
            self.used += size
            self.peak = max(self.peak, self.used)
            return True

    def release(self, size: int) -> None:
        with self._lock:
            if size < 0 or size > self.used:
                raise ValueError("invalid retained reservation release")
            self.used -= size


class RetainedLease:
    """Share one already-admitted memory charge until the final observation closes."""

    def __init__(self, budget: RetainedBudget, size: int) -> None:
        self._budget, self._size = budget, size
        self._references = 1
        self._lock = threading.Lock()

    def borrow(self) -> RetainedLease:
        with self._lock:
            if not self._references:
                raise RuntimeError("retained observation has ended")
            self._references += 1
        return self

    def release(self) -> None:
        with self._lock:
            if not self._references:
                raise RuntimeError("retained observation released twice")
            self._references -= 1
            if not self._references:
                self._budget.release(self._size)
