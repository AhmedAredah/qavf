"""Typed literal, temporal and membership policy shared by rules and SQL admission.

Pure Polars expressions and scalar admission live here without importing the
engine or rule registry. Rule-private imports remain compatibility aliases.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal

import polars as pl

RowTest = Callable[[str], pl.Expr]
Ambiguous = Literal["raise", "earliest", "latest", "null"]
_SHORT_LIST = 32


def zoned(
    value: pl.Expr,
    aware: bool,
    zone: str,
    *,
    ambiguous: Ambiguous = "raise",
    non_existent: Literal["raise", "null"] = "raise",
) -> pl.Expr:
    """Instants of a Datetime expression in ``zone``.

    Aware values are converted (the instant is kept). Naive values are wall-clock
    times in ``zone`` and are localized by Polars, whose ambiguity (DST fall-back)
    and nonexistence (spring-forward gap) handling is chosen by the caller:
    ``"null"`` marks such times instead of guessing.
    """
    if aware:
        return value.dt.convert_time_zone(zone)
    return value.dt.replace_time_zone(zone, ambiguous=ambiguous, non_existent=non_existent)


def temporal_ticks(value: date | datetime, dtype: pl.DataType) -> Fraction:
    """Express a configured instant on a column's physical storage grid.

    Naive bounds on zoned columns are wall-clock times in that zone. Polars
    performs timezone localization, including its ambiguity/nonexistence checks,
    on one microsecond literal before any conversion to the input's unit.
    Fractional ticks are retained for boundaries finer than a Date/ms column.
    """
    stamp = value if isinstance(value, datetime) else datetime.combine(value, time())
    # pandas Timestamp is a datetime subclass, but carries a nanosecond
    # remainder that Polars' Python datetime literal boundary otherwise drops.
    # Avoid importing the optional adapter merely to compare ordinary bounds.
    nanosecond = 0
    timestamp_type = getattr(sys.modules.get("pandas"), "Timestamp", None)
    if isinstance(timestamp_type, type) and isinstance(stamp, timestamp_type):
        scalar: Any = stamp
        nanosecond = scalar.nanosecond
        stamp = scalar.to_pydatetime(warn=False)
    literal = pl.lit(stamp).dt.cast_time_unit("us")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone:
        literal = zoned(literal, stamp.tzinfo is not None, dtype.time_zone)
    microseconds = Fraction(int(pl.select(literal.cast(pl.Int64)).item())) + Fraction(
        nanosecond, 1000
    )
    if dtype == pl.Date:
        return Fraction(microseconds, 86_400_000_000)
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"an instant cannot be compared with {dtype}")
    factors = {"ms": Fraction(1, 1000), "us": Fraction(1), "ns": Fraction(1000)}
    return Fraction(microseconds) * factors[dtype.time_unit]


def is_temporal(dtype: pl.DataType) -> bool:
    """Whether ``dtype`` is a Polars date or datetime type."""
    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


def is_text(dtype: pl.DataType) -> bool:
    """Whether ``dtype`` holds text: String, Categorical or Enum (logical ``str``)."""
    return dtype == pl.String or isinstance(dtype, pl.Categorical | pl.Enum)


def is_nan(value: Any) -> bool:
    """Whether ``value`` is a floating-point NaN."""
    return isinstance(value, float) and math.isnan(value)


def _fits(value: Any, dtype: pl.DataType) -> bool:
    try:
        pl.Series([value], dtype=dtype, strict=True)
    except (OverflowError, TypeError, ValueError, pl.exceptions.PolarsError):
        return False
    return True


def members(values: Sequence[Any], dtype: pl.DataType) -> tuple[list[Any], list[Any]]:
    """Split ``isin``/``notin`` values into those a ``dtype`` column can hold exactly
    (converted to that type) and those of the wrong kind for it.

    A value of the right kind that the column cannot hold exactly (``2.5`` for an
    integer column, ``300`` for ``UInt8``, a huge integer for a float column) can
    never match and is dropped. NaN and null are left out: both count as missing.
    """
    usable: list[Any] = []
    foreign: list[Any] = []
    for value in values:
        if value is None or is_nan(value):
            continue
        number = isinstance(value, int | float | Decimal) and not isinstance(value, bool)
        if dtype.is_integer():
            if not number:
                foreign.append(value)
            elif (
                isinstance(value, int) or (math.isfinite(value) and value == int(value))
            ) and _fits(int(value), dtype):
                usable.append(int(value))
        elif dtype.is_float():
            if not number:
                foreign.append(value)
            else:
                try:
                    converted = pl.Series([value], dtype=dtype, strict=False).item()
                except (OverflowError, TypeError, ValueError, pl.exceptions.PolarsError):
                    continue
                if converted is not None and converted == value:
                    usable.append(converted)
        elif dtype.is_decimal():
            if number or isinstance(value, Decimal):
                requested = (
                    Decimal.from_float(value) if isinstance(value, float) else Decimal(value)
                )
                try:
                    converted = pl.Series([requested], dtype=dtype, strict=True).item()
                except (OverflowError, TypeError, ValueError, pl.exceptions.PolarsError):
                    continue
                if converted == requested:
                    usable.append(converted)
            else:
                foreign.append(value)
        elif is_text(dtype):
            (usable if isinstance(value, str) else foreign).append(value)
        elif dtype == pl.Boolean:
            (usable if isinstance(value, bool) else foreign).append(value)
        elif dtype == pl.Date:
            if isinstance(value, datetime):
                if value.tzinfo is not None:
                    foreign.append(value)
                elif value.time() == time() and temporal_ticks(value, dtype).denominator == 1:
                    usable.append(value.date())
            elif isinstance(value, date):
                usable.append(value)
            else:
                foreign.append(value)
        elif isinstance(dtype, pl.Datetime):
            # A zoned instant cannot identify an unzoned wall-clock value.
            # Share the bounds policy with membership and missing markers.
            if isinstance(value, datetime) and value.tzinfo is not None and not dtype.time_zone:
                foreign.append(value)
            elif isinstance(value, date):
                usable.append(value)
            else:
                foreign.append(value)
        else:
            foreign.append(value)
    return usable, foreign


def membership(values: list[Any], dtype: pl.DataType, negate: bool = False) -> RowTest:
    """Exact, typed membership. Values are already converted to ``dtype``
    (:func:`members`), so no comparison mixes types, which Polars 2 rejects and
    which could otherwise round large integers."""

    def test(column: str) -> pl.Expr:
        expr = _text(column) if is_text(dtype) else pl.col(column)
        if not values:
            inside = pl.lit(False)
        elif is_temporal(dtype):
            physical = [temporal_ticks(value, dtype) for value in values]
            ticks = [int(v) for v in physical if v.denominator == 1 and -(2**63) <= v < 2**63]
            inside = (
                pl.col(column)
                .cast(pl.Int64)
                .is_in(pl.lit(pl.Series(ticks, dtype=pl.Int64)).implode())
                if ticks
                else pl.lit(False)
            )
        elif len(values) <= _SHORT_LIST:
            target = pl.String if is_text(dtype) else dtype
            literals = [pl.lit(v, dtype=target) for v in values]
            inside = pl.any_horizontal([expr == literal for literal in literals])
        else:
            series = pl.Series(values, dtype=pl.String if is_text(dtype) else dtype)
            inside = expr.is_in(pl.lit(series).implode())
        return ~inside if negate else inside

    return test


def _text(column: str) -> pl.Expr:
    # Categorical and Enum values are text too; their string view is checked.
    return pl.col(column).cast(pl.String)
