"""Lossless temporal values shared by typed grouping and regression batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import polars as pl


def has_temporal(dtype: pl.DataType) -> bool:
    if isinstance(dtype, pl.Datetime | pl.Duration) or dtype in (pl.Date, pl.Time):
        return True
    if isinstance(dtype, pl.List | pl.Array):
        return has_temporal(cast(pl.DataType, dtype.inner))
    if isinstance(dtype, pl.Struct):
        return any(has_temporal(cast(pl.DataType, field.dtype)) for field in dtype.fields)
    return False


def temporal_expr(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Lossless temporal normalization, including nested values and wide ranges."""
    if not has_temporal(dtype):
        return expr
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            expr = expr.dt.convert_time_zone("UTC")
        # Formatting at the original unit avoids overflow from a cast to ns.
        result = expr.dt.strftime("%Y-%m-%dT%H:%M:%S%.9f")
        return result + pl.lit("Z") if dtype.time_zone is not None else result
    if isinstance(dtype, pl.Duration):
        factor = {"ns": 1, "us": 1000, "ms": 1_000_000}[dtype.time_unit]
        return (expr.cast(pl.Int64).cast(pl.Decimal(38, 0)) * factor).cast(pl.String)
    if dtype == pl.Time:
        return expr.cast(pl.Int64).cast(pl.String)
    if dtype == pl.Date:
        return expr.cast(pl.String)
    if isinstance(dtype, pl.List):
        return expr.list.eval(temporal_expr(pl.element(), cast(pl.DataType, dtype.inner)))
    if isinstance(dtype, pl.Array):
        return expr.arr.to_list().list.eval(
            temporal_expr(pl.element(), cast(pl.DataType, dtype.inner))
        )
    if isinstance(dtype, pl.Struct):
        rebuilt = pl.struct(
            [
                temporal_expr(expr.struct.field(f.name), cast(pl.DataType, f.dtype)).alias(f.name)
                for f in dtype.fields
            ]
        )
        return pl.when(expr.is_null()).then(None).otherwise(rebuilt)
    return expr


def encode_temporal(frame: pl.DataFrame, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Preserve temporal precision/range before any Python evidence conversion."""
    return frame.select(
        [
            temporal_expr(pl.col(name), dtype).alias(name)
            if columns is None or name in columns
            else pl.col(name)
            for name, dtype in frame.schema.items()
        ]
    )
