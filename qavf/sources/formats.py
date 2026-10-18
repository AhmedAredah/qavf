"""File format detection and reader option semantics, independent of execution."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from qavf.redaction import redact_text

FileFormat = Literal["csv", "parquet", "json", "ndjson", "excel"]

_SUFFIX_FORMATS: dict[str, FileFormat] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
    ".xlsx": "excel",
}
READERS: dict[FileFormat, str] = {
    "csv": "read_csv",
    "parquet": "read_parquet",
    "json": "read_json",
    "ndjson": "read_json",
    "excel": "read_xlsx",
}
GLOB_PATTERN = re.compile(r"[*?\[]")


#: Compression suffixes DuckDB decompresses transparently for CSV and JSON.
_COMPRESSION_SUFFIXES = frozenset({".gz", ".gzip", ".zst", ".zstd"})
_COMPRESSIBLE: frozenset[FileFormat] = frozenset({"csv", "json", "ndjson"})


def file_name(path: str | Path) -> str:
    """The file name of a local path or URL.

    For a URL the name comes from the decoded path component: the query string
    and fragment are transport details (``data.csv?signature=...``) and
    percent-encoding is undone (``data%2Ecsv``). A local path is taken as is, so
    ``?`` stays a glob wildcard.
    """
    text = str(path)
    if is_remote(text):
        return Path(unquote(urlsplit(text).path)).name
    return Path(text).name


def suffixes_of(path: str | Path) -> tuple[list[str], str | None]:
    """Lower-cased format suffixes of ``path`` and its compression suffix, if any."""
    suffixes = [s.lower() for s in Path(file_name(path)).suffixes]
    if suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        return suffixes[:-1], suffixes[-1]
    return suffixes, None


def infer_format(path: str | Path) -> FileFormat:
    """Infer the data format from the file name.

    A compression suffix is peeled off first and the format is read from the
    suffix beneath it: ``data.csv.gz`` is CSV, ``data.ndjson.gz`` is NDJSON.
    Compression is never mistaken for a format. URLs are read by their path
    (see :func:`file_name`).

    Raises:
        ValueError: if the format cannot be determined, or the file is a
            compressed format DuckDB cannot read compressed (Parquet, Excel).
    """
    name = redact_text(file_name(path))
    suffixes, compression = suffixes_of(path)
    compressed = compression is not None
    fmt = _SUFFIX_FORMATS.get(suffixes[-1]) if suffixes else None
    if fmt is None:
        raise ValueError(
            f"cannot infer the data format of {name!r}; set 'format' ({', '.join(READERS)})"
        )
    if compressed and fmt not in _COMPRESSIBLE:
        raise ValueError(
            f"{name!r}: the {fmt} format cannot be read compressed; decompress it first"
        )
    return fmt


def is_remote(path: str) -> bool:
    return "://" in path


#: Reader options that can drop or skip input rows without failing the load.
_LOSSY_OPTIONS = ("ignore_errors", "store_rejects", "skip", "range", "stop_at_empty")


def lossy_options(options: Mapping[str, Any]) -> list[str]:
    """The reader options in ``options`` that can silently leave rows out."""
    found = []
    for name in _LOSSY_OPTIONS:
        value = options.get(name)
        if name == "skip":
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                found.append(name)
        elif name == "range":
            if value:
                found.append(name)
        elif value is True:
            found.append(name)
    return found
