"""Exact bounds and logical type decisions shared by bounded profiling."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import polars as pl


def _dtype_name(dtype: pl.DataType) -> str | None:
    if dtype.is_integer():
        return "int"
    if dtype.is_decimal():
        return "decimal"
    if dtype.is_numeric():
        return "float"
    if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
        return "str"
    if dtype == pl.Boolean:
        return "bool"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "datetime"
    return None


def validate_margin(margin: float) -> float:
    """Check a profiling margin: a finite, non-negative fraction of the observed range.

    Raises:
        ValueError: for a negative, infinite or NaN margin.
    """
    if not (isinstance(margin, int | float) and math.isfinite(margin) and margin >= 0):
        raise ValueError(f"margin must be a finite number >= 0, got {margin!r}")
    return float(margin)


def _widen_numbers(
    low: int | float | Decimal, high: int | float | Decimal, margin: float
) -> tuple[Any, Any]:
    """Widen ``[low, high]`` by ``margin`` times its span, keeping integers exact.

    A zero span uses the magnitude of the value (at least 1) as the span.
    Integer bounds are rounded outwards so they never exclude observed data.
    """
    if isinstance(low, Decimal) and isinstance(high, Decimal):
        fraction = Decimal(str(margin))
        numbers = (low, high, fraction)
        precision = (
            sum(
                len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent))
                for value in numbers
            )
            + 16
        )
        with localcontext() as context:
            context.prec = max(38, precision)
            decimal_span = high - low or max(abs(low), abs(high), Decimal(1))
            decimal_widen = fraction * decimal_span
            return low - decimal_widen, high + decimal_widen
    if isinstance(low, int) and isinstance(high, int):
        span = high - low or max(abs(low), abs(high), 1)
        widen = math.ceil(Fraction(margin) * span)
        return low - widen, high + widen
    # A finite final bound need not have a representable Float64 span: opposite
    # extrema near 1e308 are a common counterexample. Work exactly on the stored
    # binary inputs, then round each bound outwards once.
    low_q, high_q = Fraction(float(low)), Fraction(float(high))
    span_q = high_q - low_q or max(abs(low_q), abs(high_q), Fraction(1))
    widen_q = Fraction(margin) * span_q
    result = []
    for exact, direction in ((low_q - widen_q, -math.inf), (high_q + widen_q, math.inf)):
        try:
            rounded = float(exact)
        except OverflowError:
            raise ValueError(
                "widened bounds exceed the finite float range; reduce margin"
            ) from None
        if (direction < 0 and Fraction(rounded) > exact) or (
            direction > 0 and Fraction(rounded) < exact
        ):
            rounded = math.nextafter(rounded, direction)
        if not math.isfinite(rounded):
            raise ValueError("widened bounds exceed the finite float range; reduce margin")
        result.append(rounded)
    return result[0], result[1]


def _widen_dates(low: date, high: date, margin: float) -> tuple[date, date]:
    """Widen a date or datetime range by ``margin`` times its span (a day when it has none).

    Dates widen by whole days, rounded outwards.
    """
    span = high - low
    if not span:
        span = timedelta(days=1)
    if isinstance(low, datetime):
        widen = span * margin
    else:
        widen = timedelta(days=math.ceil(span.total_seconds() / 86_400 * margin))
    return low - widen, high + widen


def _plain(value: Any) -> Any:
    """Write integral floats as integers in YAML (exact below 2**53); keep everything else."""
    if isinstance(value, Decimal):
        # Quoted decimals survive both YAML/JSON and Pydantic parsing exactly.
        return str(value)
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value


def _datetime_bounds(values: pl.Series) -> tuple[datetime, datetime, bool]:
    """Return representable bounds that enclose every observed physical instant.

    Python/configuration datetimes have microsecond precision. Round nanosecond
    extrema outwards before conversion, including negative Unix timestamps.
    """
    dtype = values.dtype
    if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns":
        ticks = values.cast(pl.Int64)
        low, high = int(ticks.min()), int(ticks.max())  # type: ignore[arg-type]
        low_us, high_us = low // 1_000, -(-high // 1_000)
        limits = pl.Series([low_us, high_us], dtype=pl.Int64).cast(
            pl.Datetime("us", time_zone=dtype.time_zone)
        )
        return limits[0], limits[1], low % 1_000 != 0 or high % 1_000 != 0
    return values.min(), values.max(), False  # type: ignore[return-value]
