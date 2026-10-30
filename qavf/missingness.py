"""Declared missing-value masks and frame normalization, without rule dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

from qavf.value_policy import is_text, members, membership, missing_markers


def missing_expr(frame: pl.DataFrame, column: str, spec: Any = None) -> pl.Expr:
    """Null or NaN, without asking unsupported dtypes to evaluate is_nan."""
    expr = pl.col(column).is_null()
    if frame.schema[column].is_float():
        expr = expr | pl.col(column).is_nan()
    if spec is not None:
        if spec.missing_values:
            dtype = frame.schema[column]
            values, foreign = members(missing_markers(spec, dtype), dtype)
            if foreign:
                raise ValueError(f"missing markers for {column!r} have incompatible types")
            if values:
                expr = expr | membership(values, frame.schema[column])(column).fill_null(False)
        if spec.blank_as_missing and is_text(frame.schema[column]):
            expr = expr | (pl.col(column).cast(pl.String).str.strip_chars() == "").fill_null(False)
    return expr


def missing_count(values: pl.Series) -> int:
    """Null or NaN entries of a series, the same missing values as ``missing_expr``."""
    count = values.null_count()
    if values.dtype.is_float():
        count += int(values.is_nan().sum())
    return count


def missing_rows(frame: pl.DataFrame) -> int:
    """Rows of ``frame`` with a null or NaN in any column."""
    if not frame.columns:
        return 0
    flags = pl.any_horizontal(missing_expr(frame, column) for column in frame.columns)
    return int(frame.select(flags.sum()).item())


def normalize_frame(
    frame: pl.DataFrame,
    specs: Mapping[str, Any],
    columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Apply supplied column contracts to a frame, retaining all original columns."""
    return frame.with_columns(
        pl.when(missing_expr(frame, column, specs.get(column)))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column in (frame.columns if columns is None else columns)
    )
