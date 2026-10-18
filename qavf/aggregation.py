"""Exact accumulator selection and typed reductions shared by execution backends."""

from __future__ import annotations

from decimal import Decimal, localcontext

import polars as pl

from qavf.numeric import exact_number, stable_sum_scalar


def accumulator_dtype(dtype: pl.DataType) -> pl.DataType:
    """Storage for integer/Decimal sums, retaining the input decimal scale."""
    return pl.Decimal(38, dtype.scale or 0) if isinstance(dtype, pl.Decimal) else pl.Int128()


def accumulator_sql(dtype: pl.DataType) -> str:
    """Native SQL storage with the same width and scale as typed accumulation."""
    storage = accumulator_dtype(dtype)
    return f"DECIMAL(38,{storage.scale})" if isinstance(storage, pl.Decimal) else "HUGEINT"


def checked_accumulator_dtype(
    dtype: pl.DataType,
    minimum: object,
    maximum: object,
    count: int,
) -> pl.DataType:
    """Prove a conservative sum bound from scalar metadata, without reading rows.

    Callers supply their existing bound on contributing rows. This guard deliberately
    rejects a potentially overflowing sum even when cancellation could make it fit.
    """
    if not (minimum is None or isinstance(minimum, int | float | Decimal)) or not (
        maximum is None or isinstance(maximum, int | float | Decimal)
    ):
        raise TypeError("numeric aggregation returned a nonnumeric extremum")
    magnitude = max(exact_number(minimum or 0).copy_abs(), exact_number(maximum or 0).copy_abs())
    storage = accumulator_dtype(dtype)
    with localcontext() as context:
        context.prec = 100
        ceiling = (
            Decimal(10) ** (38 - (storage.scale or 0))
            if isinstance(storage, pl.Decimal)
            else 2**127
        )
        if magnitude and magnitude * count >= ceiling:
            raise ValueError("exact sum may exceed accumulator precision; partition the data")
    return storage


def numeric_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    """Preserve exact integers/decimals and normalize floating missing values."""
    value = pl.col(column)
    return value.fill_nan(None) if dtype.is_float() else value


def exact_sum_expr(series: pl.Series) -> pl.Expr:
    """Overflow-checked sum; null-only input remains an undefined measurement."""
    dtype = series.dtype
    value = numeric_expr(series.name, dtype)
    if dtype.is_float():
        return value.map_batches(stable_sum_scalar, return_dtype=pl.Float64, returns_scalar=True)
    if dtype.is_integer() or dtype.is_decimal():
        value = value.cast(
            checked_accumulator_dtype(dtype, series.min(), series.max(), series.len())
        )
    return pl.when(value.count() > 0).then(value.sum()).otherwise(None)


def narrow_integer(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Keep ordinary integer results compact, retaining Int128 when required."""
    if frame.schema[column] != pl.Int128:
        return frame
    lo, hi = frame[column].min(), frame[column].max()
    if lo is None:
        return frame.with_columns(pl.col(column).cast(pl.Int64))
    if not isinstance(lo, int) or not isinstance(hi, int):
        raise TypeError("integer aggregation returned a noninteger extremum")
    if lo >= -(2**63) and hi < 2**63:
        return frame.with_columns(pl.col(column).cast(pl.Int64))
    if lo >= 0 and hi < 2**64:
        return frame.with_columns(pl.col(column).cast(pl.UInt64))
    return frame
