"""Version-adaptive IPC reads with detached results and native column pruning."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import polars as pl

from qavf.engine.cache_ownership import compact_polars

_READ_OPTIONS: dict[str, Any] = (
    {"memory_map": False} if "memory_map" in inspect.signature(pl.read_ipc).parameters else {}
)
_SCAN_OPTIONS: dict[str, Any] = (
    {"memory_map": False} if "memory_map" in inspect.signature(pl.scan_ipc).parameters else {}
)


def project_ipc(path: Path, columns: list[str]) -> pl.DataFrame:
    frame = pl.read_ipc(path, columns=columns, **_READ_OPTIONS)
    # Newer Polars removed the mapping switch. Detach only the selected result
    # before the owner may unlink its file; never buffer the complete source.
    return frame if _READ_OPTIONS else cast(pl.DataFrame, compact_polars(frame))


def reduce_ipc(path: Path, build: Callable[[pl.LazyFrame], pl.LazyFrame]) -> pl.DataFrame:
    frame = build(pl.scan_ipc(path, **_SCAN_OPTIONS)).collect(engine="streaming")
    return frame if _SCAN_OPTIONS else cast(pl.DataFrame, compact_polars(frame))
