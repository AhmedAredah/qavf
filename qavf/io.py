"""Convenience reading and writing of tabular files."""

from __future__ import annotations

import gzip
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from io import IOBase
from pathlib import Path
from typing import BinaryIO, cast

import polars as pl

from qavf.output_format import table_output
from qavf.redaction import redact_text
from qavf.sources.base import SourceEnvironment
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams

logger = logging.getLogger(__name__)


@contextmanager
def atomic_stream(path: str | Path) -> Iterator[BinaryIO]:
    """Stream to a temporary file, committing only after a successful close.

    Peak writer memory is independent of output size. The previous destination
    survives serialization, compression, or fsync errors.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write(path: str | Path, data: bytes | str) -> Path:
    """Atomically write a small bytes/text artifact; use atomic_stream for large data."""
    target = Path(path)
    with atomic_stream(target) as stream:
        if isinstance(data, str):
            for start in range(0, len(data), 65536):
                stream.write(data[start : start + 65536].encode("utf-8"))
        else:
            stream.write(data)
    return target


def read_table(path: str | Path, fmt: str | None = None) -> pl.DataFrame:
    """Read a CSV, Parquet, JSON/NDJSON or Excel file (or glob).

    Parquet logical types use a typed reader when SQL conversion is lossy.

    Raises:
        qavf.sources.SourceNotFoundError: if the file does not exist.
        qavf.sources.SourceError: if it cannot be read.
    """
    params = FileParams.model_validate({"type": "file", "path": str(path), "format": fmt})
    return FileSource(params, SourceEnvironment()).load()


def write_table(frame: pl.DataFrame, path: str | Path, fmt: str | None = None) -> Path:
    """Write ``frame`` in the format its name says, creating parent directories.

    ``.csv`` and ``.txt`` are comma-separated, ``.tsv`` tab-separated;
    ``.parquet``, ``.json`` and ``.ndjson``/``.jsonl`` as named. A ``.gz`` suffix
    on a CSV or JSON name writes gzip-compressed data. Parquet preserves typed
    values; text formats obey their format's representation limits. Baseline
    acceptance verifies an exact round trip before publishing a baseline.

    Raises:
        ValueError: for a name or format that cannot be written as it says (Excel,
            zstd compression, compressed Parquet, an unknown suffix), before
            anything is created.
    """
    target = Path(path)
    output = table_output(target, fmt)

    def write(stream: BinaryIO | gzip.GzipFile) -> None:
        match output.format:
            case "csv":
                frame.write_csv(cast(BinaryIO, stream), separator=output.separator)
            case "parquet":
                frame.write_parquet(cast(BinaryIO, stream))
            case "json":
                frame.write_json(cast(IOBase, stream))
            case _:
                frame.write_ndjson(cast(BinaryIO, stream))

    with atomic_stream(target) as stream:
        if output.compressed:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                write(compressed)
        else:
            write(stream)
    logger.debug("Wrote %d rows to %s", frame.height, redact_text(str(target)))
    return target
