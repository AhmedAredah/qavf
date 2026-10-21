"""Lossless typed values: temporal normalization, checked casts and exact scaling.

Shared by typed grouping, regression batches and dataset comparison alignment.
No transformation here rounds silently: a value that cannot be represented in
its target is flagged by an accompanying expression for the caller to refuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import cast

import polars as pl

LOGICAL_TYPES: dict[str, pl.DataType] = {
    "int": pl.Int64(),
    "float": pl.Float64(),
    "str": pl.String(),
    "bool": pl.Boolean(),
    "date": pl.Date(),
    "datetime": pl.Datetime("us"),
}
"""Storage for the configuration's logical ``dtype`` names (``decimal`` takes a scale)."""


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


def logical_type(name: str, scale: int | None = None) -> pl.DataType:
    """The Polars storage of a logical ``dtype`` name; decimals need an explicit scale."""
    if name == "decimal":
        if scale is None:
            raise ValueError("a decimal type needs a scale")
        return pl.Decimal(38, scale)
    if scale is not None:
        raise ValueError(f"'scale' applies to decimal, not {name}")
    try:
        return LOGICAL_TYPES[name]
    except KeyError:
        raise ValueError(f"unknown logical type {name!r}") from None


def checked_cast(
    expr: pl.Expr, source: pl.DataType, target: pl.DataType
) -> tuple[pl.Expr, pl.Expr]:
    """Cast ``expr`` and flag each value the target cannot reproduce exactly.

    A value converts losslessly when converting it back yields the original, so
    ``1.5`` to an integer, ``2**53 + 1`` to a float, a dropped time of day, a
    rounded decimal place or text such as ``"01"`` are all flagged. Text parses
    to a date or datetime when the parse succeeds. Null stays null; NaN matches
    NaN. Returns ``(converted, lost)``.
    """
    if source == target:
        return expr, pl.lit(False)
    if source == pl.String and (target == pl.Date or isinstance(target, pl.Datetime)):
        # Text has many spellings of one instant; parsing it is lossless when it succeeds.
        parsed = (
            expr.str.to_date(strict=False)
            if target == pl.Date
            else expr.str.to_datetime(time_unit=target.time_unit, strict=False)  # type: ignore[attr-defined]
        )
        return parsed, expr.is_not_null() & parsed.is_null()
    converted = expr.cast(target, strict=False)
    restored = converted.cast(source, strict=False)
    same = (restored == expr).fill_null(False)
    if source.is_float() and target.is_float():
        same = same | (expr.is_nan() & converted.is_nan()).fill_null(False)
    lost = expr.is_not_null() & (converted.is_null() | ~same)
    return converted, lost


def _decimal_scale(value: Decimal) -> int:
    exponent = value.normalize().as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def exact_scale(expr: pl.Expr, dtype: pl.DataType, factor: int | float | Decimal) -> pl.Expr:
    """Multiply a numeric column by a finite factor without rounding exact values.

    Integers stay integers (Int128) for an integral factor and become decimals
    otherwise; decimals widen their scale to hold the product. Float columns
    multiply in binary64. A float factor denotes its decimal spelling (``0.001``
    is one thousandth), matching configuration files. A product exceeding the
    38-digit decimal storage raises rather than truncating.
    """
    if isinstance(factor, bool) or not isinstance(factor, int | float | Decimal):
        raise ValueError("a scale factor must be a number")
    exact = Decimal(repr(factor)) if isinstance(factor, float) else Decimal(factor)
    if not exact.is_finite() or not exact:
        raise ValueError("a scale factor must be a finite nonzero number")
    if dtype.is_float():
        return expr * float(exact)
    if not (dtype.is_integer() or dtype.is_decimal()):
        raise ValueError(f"only numeric columns can be scaled, not {dtype}")
    factor_scale = _decimal_scale(exact)
    if dtype.is_integer() and not factor_scale:
        return expr.cast(pl.Int128) * pl.lit(int(exact), dtype=pl.Int128)
    scale = (dtype.scale or 0) if isinstance(dtype, pl.Decimal) else 0
    target = scale + factor_scale
    if target > 38:
        raise ValueError(f"scaling needs {target} decimal places, more than 38")
    return expr.cast(pl.Decimal(38, target)) * pl.lit(
        exact.normalize(), dtype=pl.Decimal(38, factor_scale)
    )
