"""Bounded, thread-local verification scopes and streaming hash accounting."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from qavf.sources.errors import verify_postcondition


def sha256_file(path: Path, *, record_bytes: Callable[[int], None] | None = None) -> str:
    """Hash a file with a fixed-size buffer, recording bytes actually read."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            if record_bytes is not None:
                record_bytes(len(block))
            digest.update(block)
    return digest.hexdigest()


class IdentitySession:
    """Coalesce nested checks only inside one thread's verified operation.

    An operation is not a file snapshot: each outer entry and exit
    must verify content afresh. Nested checks still inspect metadata. State is
    constant-size per source plus one active flag per currently using thread;
    no file content, historical digest cache, or process-global state is retained.
    """

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._hash_passes = 0
        self._hash_passes_completed = 0
        self._bytes_hashed = 0
        self._coalesced_checks = 0

    @property
    def active(self) -> bool:
        return bool(getattr(self._local, "active", False))

    @contextmanager
    def operation(self, verify: Callable[[], None]) -> Iterator[None]:
        """Verify both boundaries, restoring thread state even on failure."""
        nested = self.active
        verify()
        self._local.active = True
        error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            # The outer exit must not coalesce its own fresh content check,
            # including a failed read before the retry policy classifies it.
            self._local.active = nested
            verify_postcondition(verify, error)

    def coalesced(self) -> None:
        with self._lock:
            self._coalesced_checks += 1

    @contextmanager
    def hashing(self) -> Iterator[Callable[[int], None]]:
        """Count attempted/completed passes, including bytes before a read error."""
        with self._lock:
            self._hash_passes += 1
        yield self._record_bytes
        with self._lock:
            self._hash_passes_completed += 1

    def _record_bytes(self, size: int) -> None:
        with self._lock:
            self._bytes_hashed += size

    def telemetry(self) -> dict[str, int]:
        """Read counters without acquiring a source's potentially slow hash lock."""
        with self._lock:
            return {
                "hash_passes": self._hash_passes,
                "hash_passes_completed": self._hash_passes_completed,
                "bytes_hashed": self._bytes_hashed,
                "coalesced_identity_checks": self._coalesced_checks,
            }
