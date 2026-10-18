"""Conversion of user-supplied tabular objects into Polars frames."""

from __future__ import annotations

import sys
from typing import Any

import polars as pl

from qavf.sources.base import SourceError


def supports_frame(frame: object) -> bool:
    """Recognize supported tabular objects without converting or consuming them."""
    if isinstance(frame, pl.DataFrame | pl.LazyFrame):
        return True
    pandas = sys.modules.get("pandas")
    return (
        (pandas is not None and isinstance(frame, pandas.DataFrame))
        or hasattr(frame, "__arrow_c_stream__")
        or type(frame).__module__.startswith("pyarrow")
    )


def as_polars(frame: Any) -> pl.DataFrame:
    """Convert a Polars/pandas DataFrame or Arrow table to Polars.

    Raises:
        SourceError: for any other type.
    """
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    if not supports_frame(frame):
        raise SourceError(f"expected a DataFrame or Arrow table, got {type(frame).__name__}")
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(frame, pandas.DataFrame):
        return pl.from_pandas(frame)
    if hasattr(frame, "__arrow_c_stream__") or type(frame).__module__.startswith("pyarrow"):
        result = pl.from_arrow(frame)
        if isinstance(result, pl.DataFrame):
            return result
    raise SourceError(f"expected a DataFrame or Arrow table, got {type(frame).__name__}")
