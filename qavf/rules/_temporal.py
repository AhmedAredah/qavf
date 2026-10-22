"""Lossless temporal decisions; Python datetime is only an evidence representation."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from fractions import Fraction
from typing import Any, Literal

import polars as pl

Ambiguous = Literal["raise", "earliest", "latest", "null"]


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


def wall_clock(value: pl.Expr, dtype: pl.DataType, zone: str | None) -> pl.Expr:
    """Naive local wall-clock times: aware values in ``zone`` (else their own zone).

    Naive values are already wall-clock times and pass through. Two instants of a
    DST fall-back map to the same wall-clock time; keep the instant to tell them apart.
    """
    if not (isinstance(dtype, pl.Datetime) and dtype.time_zone):
        return value
    local = value.dt.convert_time_zone(zone) if zone is not None else value
    return local.dt.replace_time_zone(None)


def utc_wall_clock(instant: pl.Expr) -> pl.Expr:
    """Naive UTC wall-clock time of an aware instant (its offset is local minus this)."""
    return instant.dt.convert_time_zone("UTC").dt.replace_time_zone(None)


def temporal_ticks(value: date | datetime, dtype: pl.DataType) -> Fraction:
    """Express a configured instant on a column's physical storage grid.

    Naive bounds on zoned columns are wall-clock times in that zone. Polars
    performs timezone localization, including its ambiguity/nonexistence checks,
    on one microsecond literal before any conversion to the input's unit.
    Fractional ticks are retained for boundaries finer than a Date/ms column.
    """
    stamp = value if isinstance(value, datetime) else datetime.combine(value, time())
    literal = pl.lit(stamp).dt.cast_time_unit("us")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone:
        literal = zoned(literal, stamp.tzinfo is not None, dtype.time_zone)
    microseconds = int(pl.select(literal.cast(pl.Int64)).item())
    if dtype == pl.Date:
        return Fraction(microseconds, 86_400_000_000)
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"an instant cannot be compared with {dtype}")
    factors = {"ms": Fraction(1, 1000), "us": Fraction(1), "ns": Fraction(1000)}
    return Fraction(microseconds) * factors[dtype.time_unit]


def tick_seconds(dtype: pl.DataType) -> Fraction:
    """Exact duration of one physical tick."""
    if dtype == pl.Date:
        return Fraction(86_400)
    if isinstance(dtype, pl.Datetime | pl.Duration):
        return {
            "ms": Fraction(1, 1000),
            "us": Fraction(1, 1_000_000),
            "ns": Fraction(1, 1_000_000_000),
        }[dtype.time_unit]
    if dtype == pl.Time:
        return Fraction(1, 1_000_000_000)
    raise TypeError(f"{dtype} is not temporal")


def duration_seconds(value: timedelta) -> Fraction:
    return Fraction(value.days * 86_400 + value.seconds) + Fraction(value.microseconds, 1_000_000)


def physical_keys(frame: pl.DataFrame) -> pl.DataFrame:
    """Encode temporal keys without Python's microsecond truncation."""
    return frame.with_columns(
        pl.col(column).cast(pl.Int64)
        for column, dtype in frame.schema.items()
        if dtype.is_temporal()
    )


def temporal_evidence(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep submicrosecond identifiers exact when records become Python objects."""
    expressions = []
    for column, dtype in frame.schema.items():
        value = pl.col(column)
        if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns":
            pattern = "%Y-%m-%dT%H:%M:%S%.9f" + ("%:z" if dtype.time_zone else "")
            expressions.append(value.dt.to_string(pattern).alias(column))
        elif isinstance(dtype, pl.Duration) and dtype.time_unit == "ns":
            expressions.append((value.cast(pl.Int64).cast(pl.String) + "ns").alias(column))
        elif dtype == pl.Time:
            expressions.append(value.dt.to_string("%H:%M:%S%.9f").alias(column))
    return frame.with_columns(expressions) if expressions else frame


def temporal_scalar(ticks: int, dtype: pl.DataType) -> Any:
    frame = pl.Series("value", [ticks], dtype=pl.Int64).cast(dtype).to_frame()
    return temporal_evidence(frame).item()
