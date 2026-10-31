"""Conversion of user-supplied tabular objects into Polars frames."""

from __future__ import annotations

import sys
from typing import Any

import polars as pl

from qavf.sources.base import SourceError


def _optional_types(module: str, *names: str) -> tuple[type, ...]:
    """Do not mistake publication in sys.modules for completed initialization."""
    loaded = sys.modules.get(module)
    return tuple(kind for name in names if isinstance(kind := getattr(loaded, name, None), type))


def _adapter_kind(frame: object) -> str | None:
    """One non-consuming admission decision shared by detection and conversion."""
    if isinstance(frame, pl.DataFrame):
        return "polars"
    if isinstance(frame, pl.LazyFrame):
        return "lazy"
    if callable(getattr(frame, "read_next_batch", None)):
        return None
    if isinstance(frame, _optional_types("pandas", "DataFrame")):
        return "pandas"
    if callable(getattr(frame, "__arrow_c_stream__", None)) or (
        isinstance(frame, _optional_types("pyarrow", "Table", "RecordBatch"))
    ):
        return "arrow"
    if callable(getattr(frame, "__dataframe__", None)):
        return "interchange"
    return None


def supports_frame(frame: object) -> bool:
    """Recognize candidate table representations without consuming them.

    Arrow stream schemas are validated when adapted; a scalar-array stream is
    not a table. Borrowed one-pass readers require an explicit source adapter.
    """
    return _adapter_kind(frame) is not None


def known_rows(frame: object) -> int | None:
    """The row count a pandas frame or Arrow table states, without converting it.

    ``None`` when it is not known without reading: an Arrow C stream is never
    consumed to count it.
    """
    kind = _adapter_kind(frame)
    if kind == "polars":
        return int(frame.height)  # type: ignore[attr-defined]
    if kind == "pandas":
        return len(frame.index)  # type: ignore[attr-defined]
    if kind == "arrow" and isinstance(frame, _optional_types("pyarrow", "Table", "RecordBatch")):
        return int(frame.num_rows)  # type: ignore[attr-defined]
    return None


def admitted_frame(frame: object, row_budget: int | None, label: str) -> pl.DataFrame:
    """Reject cheap known row overflow before an interchange conversion copies data."""
    from qavf.sources.native import check_budget

    rows = known_rows(frame)
    if row_budget is not None and rows is not None and rows > row_budget:
        raise SourceError(f"{label} has more than {row_budget:,} rows (row_budget)")
    return check_budget(as_polars(frame), row_budget, label)


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
    if kind == "interchange":
        from pyarrow.interchange import from_dataframe

        result = pl.from_arrow(from_dataframe(frame), rechunk=False)
        if isinstance(result, pl.DataFrame):
            return result
        raise SourceError("dataframe interchange did not produce a table")
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
