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


def has_object(dtype: pl.DataType) -> bool:
    """Whether any leaf contains opaque Python objects rather than typed buffers."""
    if dtype == pl.Object:
        return True
    if isinstance(dtype, pl.List | pl.Array):
        return has_object(cast(pl.DataType, dtype.inner))
    if isinstance(dtype, pl.Struct):
        return any(has_object(cast(pl.DataType, field.dtype)) for field in dtype.fields)
    return False


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

    Primitive casts must reproduce the original on conversion back, so
    ``1.5`` to an integer, ``2**53 + 1`` to a float, a dropped time of day, a
    rounded decimal place or text such as ``"01"`` are all flagged. Text parses
    to a date or datetime when parsing succeeds at the requested precision.
    Null stays null; NaN matches NaN. Text parsing also refuses discarded
    fractional-second digits.
    Decimal/float conversions must preserve the exact numeric value, not only
    recover it after another rounded cast. Returns ``(converted, lost)``.
    """
    if source == target:
        return expr, pl.lit(False)
    if source == pl.String and (target == pl.Date or isinstance(target, pl.Datetime)):
        # Text has many spellings of one instant; check parse success and precision.
        if target == pl.Date:
            parsed = expr.str.to_date(strict=False)
        else:
            assert isinstance(target, pl.Datetime)
            parsed = expr.str.to_datetime(
                time_unit=target.time_unit,
                time_zone="UTC" if target.time_zone is not None else None,
                strict=False,
                ambiguous="null",
            )
            if target.time_zone is not None:
                # Parse without a local DST transition first. Older supported
                # Polars raises for nonexistent local times even with strict=False.
                # An explicit ISO offset identifies an instant; otherwise the
                # parsed clock belongs to the requested local timezone.
                offset = expr.str.contains(
                    r"(?i)[t ]\d{2}.*(?:z|utc|gmt|[+-]\d{2}(?::?\d{2}){0,2})$"
                )
                parsed = (
                    pl.when(offset)
                    .then(parsed.dt.convert_time_zone(target.time_zone))
                    .otherwise(parsed.dt.replace_time_zone(
                        target.time_zone, ambiguous="null", non_existent="null"
                    ))
                )
        lost = expr.is_not_null() & parsed.is_null()
        if isinstance(target, pl.Datetime):
            # Parsing can succeed after truncating a fractional second. Inspect
            # the source precision without converting wide dates to nanoseconds.
            precision = {"ms": 3, "us": 6, "ns": 9}[target.time_unit]
            fraction = expr.str.extract(r"(?:\d{1,2}:\d{1,2}:\d{1,2}|\d{6})[.,](\d+)", 1)
            lost |= fraction.str.slice(precision).str.contains("[1-9]").fill_null(False)
        return parsed, lost
    converted = expr.cast(target, strict=False)
    restored = converted.cast(source, strict=False)
    same = (restored == expr).fill_null(False)
    if source.is_float() and target.is_float():
        same = same | (expr.is_nan() & converted.is_nan()).fill_null(False)
    # A decimal/float round trip can hide rounding on the return conversion.
    # A decimal c / 10**scale is dyadic exactly when 5**scale divides c.
    # Require this in addition to the round trip, entirely in native integer
    # expressions: no Python observation buffer or per-cell callback is needed.
    decimal = (
        source
        if source.is_decimal() and target.is_float()
        else (target if target.is_decimal() and source.is_float() else None)
    )
    if isinstance(decimal, pl.Decimal):
        decimal_value = expr if source.is_decimal() else converted
        divisor = pl.lit(5 ** (decimal.scale or 0), dtype=pl.Int128)
        same &= (decimal_value.to_physical() % divisor == 0).fill_null(False)
    lost = expr.is_not_null() & (converted.is_null() | ~same)
    return converted, lost


def _decimal_parts(value: Decimal) -> tuple[int, int]:
    """Exact signed coefficient and scale, independent of Decimal's context."""
    sign, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)  # The caller admits only finite factors.
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    while coefficient and coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1
    if exponent > 0:
        coefficient *= 10**exponent
    return -coefficient if sign else coefficient, max(0, -exponent)


def _multiply_coefficients(values: pl.Series, factor: int, minimum: int, maximum: int) -> pl.Series:
    """Check native extrema before a multiplication that would otherwise wrap."""
    low, high = values.min(), values.max()
    if low is None:
        return values
    assert isinstance(low, int) and isinstance(high, int)
    products = (low * factor, high * factor)
    if min(products) < minimum or max(products) > maximum:
        raise ValueError("exact scale product exceeds its integer/decimal storage precision")
    if low == high == 0 or factor == 1:
        return values
    if -(2**127) <= factor < 2**127:
        return values * pl.Series([factor], dtype=pl.Int128)
    # An oversized factor can still have representable products (e.g. -1 * 2**127).
    # Only this exceptional domain needs Python integers; ordinary work is native.
    return pl.Series(
        values.name,
        (None if value is None else value * factor for value in values),
        dtype=pl.Int128,
    )


def _decimal_from_coefficients(values: pl.Series, scale: int) -> pl.Series:
    """Reinterpret already-checked coefficients instead of rescaling their values."""
    import pyarrow as pa

    # Both Arrow decimal types have the same 128-bit coefficient/null buffers.
    # A normal Polars cast to Decimal(scale) would multiply the coefficient again.
    physical = values.cast(pl.Decimal(38, 0)).to_arrow()
    decimal = physical.view(pa.decimal128(38, scale))
    return cast(pl.Series, pl.from_arrow(decimal, rechunk=False)).rename(values.name)


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
    coefficient, factor_scale = _decimal_parts(exact)
    if dtype.is_integer() and not factor_scale:
        return expr.cast(pl.Int128).map_batches(
            lambda values: _multiply_coefficients(values, coefficient, -(2**127), 2**127 - 1),
            return_dtype=pl.Int128,
            is_elementwise=True,
        )
    scale = (dtype.scale or 0) if isinstance(dtype, pl.Decimal) else 0
    target = scale + factor_scale
    if target > 38:
        raise ValueError(f"scaling needs {target} decimal places, more than 38")

    def scale_decimal(values: pl.Series) -> pl.Series:
        physical = values.to_physical() if dtype.is_decimal() else values.cast(pl.Int128)
        product = _multiply_coefficients(physical, coefficient, -(10**38 - 1), 10**38 - 1)
        return _decimal_from_coefficients(product, target)

    return expr.map_batches(
        scale_decimal,
        return_dtype=pl.Decimal(38, target),
        is_elementwise=True,
    )
