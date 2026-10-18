"""Automatic native allowances, separate from explicit user overrides.

Process-global pools are initialized before imports. Per-connection query pools
share the CPU allowance between concurrent rule workers instead.
"""

from __future__ import annotations

import os


def startup_threads(explicit: int | None, *, cpu_count: int | None = None) -> int:
    """Size a process-global startup pool without multiplying by rule workers."""
    cpus = cpu_count if cpu_count is not None else os.cpu_count()
    return explicit if explicit is not None else max(1, min(8, cpus or 1))


def connection_threads(
    explicit: int | None, workers: int, *, cpu_count: int | None = None
) -> int:
    """Share automatic query threads across workers; preserve explicit intent."""
    cpus = cpu_count if cpu_count is not None else os.cpu_count()
    return explicit if explicit is not None else max(1, (cpus or 1) // max(1, workers))
