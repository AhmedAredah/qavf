"""Run-local reader decisions, independent of repeated source verification."""

from __future__ import annotations

import threading
from collections.abc import Callable


class ReaderCapabilities:
    """Remember a resolved backend; actual reads still verify their own identity."""

    def __init__(self) -> None:
        self._native: dict[str, bool] = {}
        self._lock = threading.RLock()

    def native(self, name: str, resolve: Callable[[], bool]) -> bool:
        with self._lock:
            if name in self._native:
                return self._native[name]
        # Never hold this metadata lock while acquiring source/operation locks.
        resolved = resolve()
        with self._lock:
            return self._native.setdefault(name, resolved)

    def record_native(self, name: str) -> None:
        with self._lock:
            self._native[name] = True
