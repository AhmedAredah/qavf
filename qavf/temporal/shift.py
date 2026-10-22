"""Calendar offsets of time columns, e.g. to compare this year with last year.

An offset has a calendar part (years, months, weeks, days: local wall-clock
arithmetic, so "one day earlier" keeps the time of day across a DST change)
and an elapsed part (hours, minutes, seconds: physical time). ``same_weekday``
turns years into 52-week steps (364 days), so a Monday is compared with a
Monday. A shift that has no exact counterpart is reported row by row: 29
February one year earlier, a local time skipped by a spring-forward gap, or the
two instants of a fall-back that would collapse into one.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
from pydantic import model_validator

from qavf.engine.registry import RuleParams
from qavf.rules._temporal import wall_clock, zoned


class CalendarOffset(RuleParams):
    """How far one dataset's time is ahead of the other's (``{years: 1}``)."""

    years: int = 0
    months: int = 0
    weeks: int = 0
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    same_weekday: bool = False
    """Count each year as 52 weeks (364 days), keeping the weekday."""

    @model_validator(mode="after")
    def _offset(self) -> CalendarOffset:
        if self.same_weekday and self.months:
            raise ValueError("'same_weekday' counts years in weeks; it cannot shift months")
        if not (self.calendar_months or self.calendar_days or self.elapsed):
            raise ValueError("the offset is zero; set years, months, weeks, days or a time")
        return self

    @property
    def calendar_months(self) -> int:
        return 0 if self.same_weekday else self.years * 12 + self.months

    @property
    def calendar_days(self) -> int:
        return self.weeks * 7 + self.days + (364 * self.years if self.same_weekday else 0)

    @property
    def elapsed(self) -> timedelta:
        return timedelta(hours=self.hours, minutes=self.minutes, seconds=self.seconds)

    def describe(self) -> str:
        parts = [
            f"{value} {name}"
            for name, value in self.model_dump(exclude={"same_weekday"}).items()
            if value
        ]
        return " ".join(parts) + (" (same weekday)" if self.same_weekday else "")


def _calendar(value: pl.Expr, months: int, days: int) -> tuple[pl.Expr, pl.Expr]:
    """Shift a Date or naive Datetime; also the rows whose month shift is not invertible.

    Month arithmetic clamps to the month's last day (29 February a year later is
    28 February), which would merge two days: such rows have no shifted value.
    """
    shifted, clamped = value, pl.lit(False)
    if months:
        shifted = value.dt.offset_by(f"{months}mo")
        clamped = (shifted.dt.offset_by(f"{-months}mo") != value).fill_null(False)
    if days:
        shifted = shifted + pl.duration(days=days)
    return pl.when(~clamped).then(shifted), clamped


def shifted_time(
    column: str, dtype: pl.DataType, offset: CalendarOffset, sign: int, zone: str | None
) -> tuple[pl.Expr, pl.Expr]:
    """``column`` moved by ``sign`` times ``offset``, and the rows that cannot move exactly.

    Dates take calendar offsets only. Naive datetimes without ``zone`` shift as
    naive values. Datetimes in a zone (aware, or naive wall-clock times with
    ``zone``) shift their local wall clock and are localized again, keeping a
    fall-back row's occurrence; the elapsed part then moves the instant. The
    result has the column's own dtype.

    Raises:
        ValueError: for a column type that has no calendar.
    """
    months, days = sign * offset.calendar_months, sign * offset.calendar_days
    elapsed = offset.elapsed * sign
    value = pl.col(column)
    if dtype == pl.Date:
        if elapsed:
            raise ValueError(f"time column {column!r} holds dates; it cannot shift by {elapsed}")
        return _calendar(value, months, days)
    if not isinstance(dtype, pl.Datetime):
        raise ValueError(f"time column {column!r} must be a Date or a Datetime, found {dtype}")
    aware = dtype.time_zone is not None
    zone = zone if zone is not None else dtype.time_zone
    if zone is None:
        shifted, clamped = _calendar(value, months, days)
        return shifted + pl.duration(microseconds=elapsed // timedelta(microseconds=1)), clamped
    local = wall_clock(value, dtype, zone)
    instant = zoned(value, aware, zone, ambiguous="null", non_existent="null")
    unaligned = value.is_not_null() & instant.is_null()
    if months or days:
        earliest = zoned(local, False, zone, ambiguous="earliest", non_existent="null")
        latest = zoned(local, False, zone, ambiguous="latest", non_existent="null")
        second = (earliest != latest) & (instant == latest)
        moved, clamped = _calendar(local, months, days)
        first_after = zoned(moved, False, zone, ambiguous="earliest", non_existent="null")
        second_after = zoned(moved, False, zone, ambiguous="latest", non_existent="null")
        collapsed = second & (first_after == second_after)
        instant = pl.when(second).then(second_after).otherwise(first_after)
        unaligned = unaligned | clamped | collapsed.fill_null(False)
        unaligned = unaligned | (value.is_not_null() & instant.is_null())
    instant = instant + pl.duration(microseconds=elapsed // timedelta(microseconds=1))
    result = (
        instant.dt.convert_time_zone(dtype.time_zone or zone)
        if aware
        else wall_clock(instant, pl.Datetime("us", zone), zone)
    )
    return pl.when(~unaligned).then(result).cast(dtype), unaligned
