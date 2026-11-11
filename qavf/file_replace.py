"""Atomic file replacement that waits out a briefly held target on Windows.

Every atomic publication in QAVF (:func:`qavf.io.atomic_stream`, a baseline's
transaction journal and its two visible replacements, an isolated worker's
result) ends with one rename over the destination. On Windows a virus scanner or
the search indexer may hold the previous version open for a moment, and the
rename then fails with a sharing violation (``PermissionError``) although
nothing is wrong. :func:`replace_file` retries that error on Windows for a
bounded time (:data:`REPLACE_TOLERANCE_S`) and then raises it; elsewhere a
``PermissionError`` is real and raised at once.

This module imports nothing from QAVF, so the release scripts load it by path
(``scripts/release_checks/provenance.py``) and share the one implementation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPLACE_TOLERANCE_S = 2.0
"""How long a Windows replace may wait out a scanner or indexer briefly holding the target."""
_FIRST_DELAY_S = 0.01
_LONGEST_DELAY_S = 0.2

# The module's own handles on the system calls, so tests replace them here
# without affecting every other thread of the process.
_rename = os.replace
_clock = time.monotonic
_pause = time.sleep


class HeldTargetRetry:
    """Retries one rename while Windows reports the target briefly held.

    The delay starts short and doubles up to :data:`_LONGEST_DELAY_S`, so a target
    held for a few milliseconds costs little and a longer hold is not polled hard.
    """

    def __init__(self, tolerance_s: float = REPLACE_TOLERANCE_S) -> None:
        self.tolerance_s = tolerance_s if sys.platform == "win32" else 0.0

    def replace(self, source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        """``os.replace(source, target)``, retried while the target is briefly held."""
        deadline = _clock() + self.tolerance_s
        delay = _FIRST_DELAY_S
        while True:
            try:
                _rename(source, target)
                return
            except PermissionError:
                remaining = deadline - _clock()
                if remaining <= 0:
                    raise
                _pause(min(delay, remaining))
                delay = min(delay * 2, _LONGEST_DELAY_S)


def replace_file(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    tolerance_s: float = REPLACE_TOLERANCE_S,
) -> Path:
    """Atomically replace ``target`` by ``source`` (as :func:`os.replace`).

    On Windows a ``PermissionError`` (the target briefly held by another process)
    is retried for up to ``tolerance_s`` seconds before it is raised.

    Returns:
        ``target`` as a path.
    """
    HeldTargetRetry(tolerance_s).replace(source, target)
    return Path(target)


__all__ = ["REPLACE_TOLERANCE_S", "HeldTargetRetry", "replace_file"]
