"""Validate output suffixes once for eager and streamed table writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qavf.sources.formats import infer_format, suffixes_of


@dataclass(frozen=True, slots=True)
class TableOutput:
    format: str
    compressed: bool
    separator: str


def table_output(path: str | Path, format: str | None = None) -> TableOutput:
    target = Path(path)
    try:
        resolved = format or infer_format(target)
    except ValueError as exc:
        raise ValueError(f"cannot write {target.name!r}: {exc}") from exc
    suffixes, compression = suffixes_of(target)
    if resolved not in ("csv", "parquet", "json", "ndjson"):
        raise ValueError(f"cannot write {target.name!r}: writing {resolved} is not supported")
    if compression not in (None, ".gz", ".gzip"):
        raise ValueError(
            f"cannot write {target.name!r}: only gzip (.gz) compression can be written"
        )
    if compression and resolved == "parquet":
        raise ValueError(f"cannot write {target.name!r}: Parquet is compressed internally")
    return TableOutput(
        resolved, compression is not None, "\t" if suffixes[-1:] == [".tsv"] else ","
    )
