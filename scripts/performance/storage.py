"""Bounded fixture writing and atomic JSON evidence persistence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Commit a complete JSON record, leaving prior evidence intact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet_chunks(path: Path, frames: Iterable[Any]) -> int:
    """Write bounded Polars frames without retaining a full fixture in memory."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    try:
        for frame in frames:
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
        if writer is None:
            raise ValueError("At least one typed fixture chunk is required")
        return rows
    finally:
        if writer is not None:
            writer.close()
