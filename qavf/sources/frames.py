"""Conversion of user-supplied tabular objects into Polars frames."""

from __future__ import annotations

import sys
from typing import Any

import polars as pl

from qavf.sources.base import SourceError


def _adapter_kind(frame: object) -> str | None:
    """One non-consuming admission decision shared by detection and conversion."""
    if isinstance(frame, pl.DataFrame):
        return "polars"
    if isinstance(frame, pl.LazyFrame):
        return "lazy"
    if callable(getattr(frame, "read_next_batch", None)):
        return None
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(frame, pandas.DataFrame):
        return "pandas"
    arrow = sys.modules.get("pyarrow")
    if callable(getattr(frame, "__arrow_c_stream__", None)) or (
        arrow is not None and isinstance(frame, arrow.Table | arrow.RecordBatch)
    ):
        return "arrow"
    return None


def supports_frame(frame: object) -> bool:
    """Recognize candidate table representations without consuming them.

    Arrow stream schemas are validated when adapted; a scalar-array stream is
    not a table. Borrowed one-pass readers require an explicit source adapter.
    """
    return _adapter_kind(frame) is not None


def as_polars(frame: Any) -> pl.DataFrame:
    """Convert a Polars/pandas DataFrame or Arrow table to Polars.

    Raises:
        SourceError: for any other type.
    """
    kind = _adapter_kind(frame)
    if kind == "polars":
        return frame
    if kind == "lazy":
        return frame.collect()
    if kind == "pandas":
        return pl.from_pandas(frame, rechunk=False)
    if kind == "arrow":
        # Explicit across supported Polars versions: preserve chunk ownership,
        # avoiding an otherwise unnecessary whole-table contiguous copy.
        result = pl.from_arrow(frame, rechunk=False)
        if isinstance(result, pl.DataFrame):
            return result
        # Polars 2 exposes a generic tabular C stream as a Struct Series;
        # earlier releases return a frame. Unnest only a structural table,
        # without requesting a second (possibly one-shot) stream.
        if isinstance(result, pl.Series) and isinstance(result.dtype, pl.Struct):
            return result.struct.unnest()
    raise SourceError(f"expected a DataFrame or Arrow table, got {type(frame).__name__}")
