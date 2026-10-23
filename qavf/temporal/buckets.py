"""Time buckets: calendar periods and cyclic profiles as exact native labels.

A bucket turns a time column into a *label* per row, computed by Polars
expressions (no per-row Python):

* **Calendar buckets** (``second`` ... ``year``, with multiples such as
  ``15 minutes`` or ``6 hours``). Day and longer buckets are labelled with the
  ``Date`` their local calendar period starts on; shorter buckets with the
  instant they start. Multiples are anchored at the Unix epoch (days, weeks) or
  at year 0 (months, quarters, years), so ``2 months`` starts in January,
  March, ...; sub-day widths must divide a day so they restart at local midnight.
* **Cyclic profiles** (``hour_of_day``, ``day_of_week`` (ISO 1 = Monday),
  ``day_of_year``, ``month_of_year``, ``quarter_of_year``, ``season``) label
  each row with its position in the cycle.

Time zones: naive timestamps are wall-clock times in ``time_zone`` (a naive
time that does not exist there cannot be placed); aware timestamps are
converted to it (else kept in their own zone). Buckets follow the local
calendar, so a spring-forward day has 23 hours and a fall-back day 25. A
sub-day bucket is labelled with its starting *instant*: the two physical
hours of a fall-back each form their own hour bucket, while a bucket that
merely contains a transition (``6 hours`` from midnight) stays one bucket.
Localization uses Polars' DST handling (:mod:`qavf.rules._temporal`); ambiguous
cases are reported as unplaceable rows (null labels), never guessed.

Numeric time columns (a year, a period number) pass through unchanged when no
bucket is set. Labels are exact (dates, integer-tick instants, small integers
or season names): never float time.

A label has two integer positions, which differ only for sub-day buckets
across DST changes; each consumer must pick the one it means:

* :func:`sequence_positions`: spacing in time. Consecutive buckets of the exact
  grid are one apart (both fall-back hours count, the skipped hour does not).
  For rates, slopes, lags and peak shifts.
* :func:`calendar_position`: the wall-clock slot. An hourly bucket's position
  mod 24 is its local hour; both fall-back hours share a slot (two
  observations, never merged into one bucket) and the skipped hour has none.
  For seasons, profiles and anything taken modulo a cycle.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl
from pydantic import Field, field_validator, model_validator

from qavf.engine.registry import RuleParams
from qavf.rules._temporal import temporal_ticks, utc_wall_clock, wall_clock, zoned

BucketUnit = Literal[
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
    "hour_of_day",
    "day_of_week",
    "day_of_year",
    "month_of_year",
    "quarter_of_year",
    "season",
]
Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

WEEKDAYS: tuple[str, ...] = Weekday.__args__  # type: ignore[attr-defined]
SUBDAY_SECONDS = {"second": 1, "minute": 60, "hour": 3600}
CALENDAR_MONTHS = {"month": 1, "quarter": 3, "year": 12}
CYCLES = {
    "hour_of_day": 24,
    "day_of_week": 7,
    "day_of_year": 366,
    "month_of_year": 12,
    "quarter_of_year": 4,
}
_QUARTERS: tuple[tuple[int, ...], ...] = ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
METEOROLOGICAL: dict[str, dict[str, tuple[int, ...]]] = {
    "north": dict(zip(("winter", "spring", "summer", "autumn"), _QUARTERS, strict=True)),
    "south": dict(zip(("summer", "autumn", "winter", "spring"), _QUARTERS, strict=True)),
}
"""Meteorological seasons in calendar order from December, per hemisphere."""
_EPOCH_WEEKDAY = 3
"""1970-01-01 was a Thursday (Monday = 0)."""
_DAY_US = 86_400_000_000

_ALIASES = {
    "s": "second", "sec": "second", "second": "second", "seconds": "second",
    "min": "minute", "minute": "minute", "minutes": "minute",
    "h": "hour", "hr": "hour", "hour": "hour", "hours": "hour",
    "d": "day", "day": "day", "days": "day",
    "w": "week", "week": "week", "weeks": "week",
    "mo": "month", "month": "month", "months": "month",
    "q": "quarter", "quarter": "quarter", "quarters": "quarter",
    "y": "year", "yr": "year", "year": "year", "years": "year",
}  # fmt: skip
_MULTIPLE = re.compile(r"^(\d+)\s*([a-z]+)$")


class Bucket(RuleParams):
    """A calendar bucket (``day``, ``15 minutes``, ``6h``) or a cyclic profile (``season``).

    Written as a unit name, ``<n> <unit>``, or a mapping such as
    ``{unit: week, week_start: sunday}``.
    """

    unit: BucketUnit
    """``second``, ``minute``, ``hour``, ``day``, ``week``, ``month``, ``quarter``,
    ``year``, or a profile: ``hour_of_day``, ``day_of_week``, ``day_of_year``,
    ``month_of_year``, ``quarter_of_year``, ``season``."""
    every: int = Field(default=1, ge=1)
    """Multiple of a calendar unit (``{unit: minute, every: 15}``)."""
    week_start: Weekday = "monday"
    """First day of a ``week`` bucket, and of the ``day_of_week`` order."""
    hemisphere: Literal["north", "south"] = "north"
    """Names of the meteorological seasons (December-February is winter in the north)."""
    seasons: dict[str, tuple[int, ...]] | None = None
    """Custom seasons, ``name: [months]`` in cycle order, covering each month once."""

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if not isinstance(data, str):
            return data
        text = " ".join(data.strip().lower().split())
        if match := _MULTIPLE.match(text.replace(" ", "")):
            count, unit = match.groups()
            if unit in _ALIASES:
                return {"unit": _ALIASES[unit], "every": int(count)}
        unit = text.replace(" ", "_")
        return {"unit": _ALIASES.get(unit, unit)}

    @model_validator(mode="after")
    def _consistent(self) -> Bucket:
        if self.every > 1 and self.cyclic:
            raise ValueError(f"'every' does not apply to the cyclic profile {self.unit!r}")
        if self.subday and _DAY_US % (self.every * SUBDAY_SECONDS[self.unit] * 10**6):
            raise ValueError(
                f"{self.describe()} does not divide a day; sub-day buckets restart at midnight"
            )
        if self.seasons is not None:
            if self.unit != "season":
                raise ValueError("'seasons' only applies to the 'season' profile")
            months = sorted(month for group in self.seasons.values() for month in group)
            if months != list(range(1, 13)) or len(self.seasons) < 2:
                raise ValueError("'seasons' must assign each month 1-12 to exactly one season")
        return self

    @property
    def cyclic(self) -> bool:
        return self.unit in CYCLES or self.unit == "season"

    @property
    def subday(self) -> bool:
        return self.unit in SUBDAY_SECONDS

    @property
    def width_us(self) -> int:
        """Width of a sub-day bucket in microseconds."""
        return self.every * SUBDAY_SECONDS[self.unit] * 10**6

    @property
    def months(self) -> int:
        """Width of a month, quarter or year bucket in months."""
        return self.every * CALENDAR_MONTHS[self.unit]

    @property
    def season_names(self) -> dict[str, tuple[int, ...]]:
        return self.seasons if self.seasons is not None else METEOROLOGICAL[self.hemisphere]

    @property
    def cycle_length(self) -> int | None:
        """Positions in one cycle of a cyclic profile; ``None`` for calendar buckets."""
        if self.unit == "season":
            return len(self.season_names)
        return CYCLES.get(self.unit)

    def describe(self) -> str:
        if self.cyclic:
            return self.unit.replace("_", " ")
        if self.every == 1:
            return self.unit
        return f"{self.every} {self.unit}s"


class TimeBucketing(RuleParams):
    """How a time column is read and bucketed (shared by series rules and levels)."""

    bucket: Bucket | None = None
    """Calendar bucket or cyclic profile of a Date/Datetime time column (``day``,
    ``15 minutes``, ``month``, ``hour_of_day``, ...). Without it, time values are
    used as they are (a numeric period such as a year)."""
    time_zone: str | None = None
    """IANA zone of the local calendar: naive times are wall-clock times there,
    aware times are converted to it. Aware columns default to their own zone."""
    time_format: str | None = None
    """``strftime`` format for parsing a text time column (unparseable values are
    reported, never dropped)."""

    @field_validator("time_zone")
    @classmethod
    def _zone(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, ValueError):
                raise ValueError(f"unknown time zone {value!r}") from None
        return value

    @property
    def bucketing(self) -> TimeBucketing:
        """These settings alone, e.g. to share them with another component."""
        return TimeBucketing(
            bucket=self.bucket, time_zone=self.time_zone, time_format=self.time_format
        )

    def describe(self) -> str:
        """``day in America/Chicago``-style summary for reports."""
        what = self.bucket.describe() if self.bucket is not None else "time values as they are"
        return f"{what} in {self.time_zone}" if self.time_zone else what


# ------------------------------------------------------------------- labels
def _dtype_of(expression: pl.Expr, column: str, dtype: pl.DataType) -> pl.DataType:
    """Output dtype of ``expression`` over a column of ``dtype`` (planned, not run)."""
    plan = pl.LazyFrame(schema={column: dtype}).select(expression.alias(column))
    return plan.collect_schema()[column]


def _aware(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.Datetime) and dtype.time_zone is not None


def _parsed(column: str, dtype: pl.DataType, axis: TimeBucketing) -> tuple[pl.Expr, pl.DataType]:
    """The column as a temporal or numeric value, parsing text with ``time_format``."""
    value = pl.col(column)
    if dtype == pl.String:
        if axis.time_format is None:
            raise ValueError(
                f"time column {column!r} is text; set 'time_format' to parse it as dates or times"
            )
        value = value.str.to_datetime(axis.time_format, strict=False)
        return value, _dtype_of(value, column, dtype)
    if axis.time_format is not None:
        raise ValueError(f"'time_format' applies to text time columns, not {dtype}")
    if dtype.is_numeric() or dtype == pl.Date or isinstance(dtype, pl.Datetime):
        return value, dtype
    raise ValueError(f"time column {column!r} must be numeric, a Date or a Datetime, found {dtype}")


def _zone(dtype: pl.DataType, axis: TimeBucketing) -> str | None:
    if axis.time_zone is not None:
        return axis.time_zone
    return dtype.time_zone if isinstance(dtype, pl.Datetime) else None


def _local(value: pl.Expr, dtype: pl.DataType, zone: str | None):
    """``(wall clock, row offset or None, placeable)`` of a Date/Datetime value.

    The offset (local minus UTC) tells the two instants of a DST fall-back
    apart; it is null for a naive time that is ambiguous in ``zone``. A naive
    time that does not exist in ``zone`` cannot be placed at all.
    """
    if dtype == pl.Date or zone is None:
        return value, None, pl.lit(True)
    if _aware(dtype):
        instant = value.dt.convert_time_zone(zone)
        local = instant.dt.replace_time_zone(None)
        return local, local - utc_wall_clock(instant), pl.lit(True)
    instant = zoned(value, False, zone, ambiguous="null", non_existent="null")
    exists = zoned(value, False, zone, ambiguous="earliest", non_existent="null").is_not_null()
    return value, value - utc_wall_clock(instant), exists | value.is_null()


def _days(local: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    day = local if dtype == pl.Date else local.dt.date()
    return day.cast(pl.Int32).cast(pl.Int64)


def _week_anchor(bucket: Bucket) -> int:
    """Days from the epoch to the first ``week_start`` on or before it."""
    return -((_EPOCH_WEEKDAY - WEEKDAYS.index(bucket.week_start)) % 7)


def _month_index(day: pl.Expr) -> pl.Expr:
    return day.dt.year().cast(pl.Int64) * 12 + day.dt.month().cast(pl.Int64) - 1


def _from_months(index: pl.Expr) -> pl.Expr:
    return pl.date(index // 12, index % 12 + 1, 1)


def _calendar_date(local: pl.Expr, dtype: pl.DataType, bucket: Bucket) -> pl.Expr:
    """The Date a day, week, month, quarter or year bucket starts on."""
    days = _days(local, dtype)
    if bucket.unit == "day":
        return ((days // bucket.every) * bucket.every).cast(pl.Int32).cast(pl.Date)
    if bucket.unit == "week":
        anchor, width = _week_anchor(bucket), 7 * bucket.every
        return (anchor + ((days - anchor) // width) * width).cast(pl.Int32).cast(pl.Date)
    day = local if dtype == pl.Date else local.dt.date()
    return _from_months((_month_index(day) // bucket.months) * bucket.months)


def _season_dtype(bucket: Bucket) -> pl.Enum:
    return pl.Enum(list(bucket.season_names))


def _profile(local: pl.Expr, bucket: Bucket) -> pl.Expr:
    """Position label in a cycle: hour 0-23, ISO weekday 1-7, day 1-366, month, ..."""
    if bucket.unit == "season":
        mapping = {m: name for name, months in bucket.season_names.items() for m in months}
        return local.dt.month().replace_strict(
            list(mapping), list(mapping.values()), return_dtype=_season_dtype(bucket)
        )
    extract = {
        "hour_of_day": lambda v: v.dt.hour(),
        "day_of_week": lambda v: v.dt.weekday(),
        "day_of_year": lambda v: v.dt.ordinal_day(),
        "month_of_year": lambda v: v.dt.month(),
        "quarter_of_year": lambda v: v.dt.quarter(),
    }[bucket.unit]
    return extract(local).cast(pl.Int32)


def _subday_start(local: pl.Expr, dtype: pl.DataType, bucket: Bucket) -> pl.Expr:
    """Naive wall-clock start of a sub-day bucket, exactly, in microseconds."""
    assert isinstance(dtype, pl.Datetime)
    unit = dtype.time_unit
    width = {"ms": bucket.width_us // 1000, "us": bucket.width_us, "ns": bucket.width_us * 1000}
    ticks = local.dt.epoch(unit)
    start = (ticks // width[unit]) * width[unit]
    return pl.from_epoch(start, time_unit=unit).dt.cast_time_unit("us")


def _occurrences(start: pl.Expr, zone: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """``(earliest, latest, gap)`` instants of a naive wall-clock start in ``zone``.

    ``earliest`` and ``latest`` differ for a start repeated by a DST fall-back and
    are null for one skipped by a spring-forward gap. ``gap`` is then the end of
    the gap, the first instant the bucket really has: the start read with the
    offset in effect a day earlier.
    """
    earliest = start.dt.replace_time_zone(zone, ambiguous="earliest", non_existent="null")
    latest = start.dt.replace_time_zone(zone, ambiguous="latest", non_existent="null")
    before = start - pl.duration(days=1)
    previous = before.dt.replace_time_zone(zone, ambiguous="earliest", non_existent="null")
    gap = (start - (before - utc_wall_clock(previous))).dt.replace_time_zone("UTC")
    return earliest, latest, pl.when(earliest.is_null()).then(gap.dt.convert_time_zone(zone))


def _start_instant(start: pl.Expr, offset: pl.Expr, zone: str) -> pl.Expr:
    """The instant a local wall-clock bucket start denotes for a row.

    An ordinary start has one instant and a skipped one its gap's end. A start
    repeated by a DST fall-back has two; the row's own offset chooses (null when
    a naive row is ambiguous, so it cannot be placed).
    """
    earliest, latest, gap = _occurrences(start, zone)
    repeated = earliest != latest
    return (
        pl.when(earliest.is_null())
        .then(gap)
        .when(repeated & offset.is_null())
        .then(None)
        .when(repeated & ((start - offset) == utc_wall_clock(latest)))
        .then(latest)
        .otherwise(earliest)
    )


def bucket_labels(
    column: str, dtype: pl.DataType, axis: TimeBucketing
) -> tuple[pl.Expr, pl.DataType]:
    """The bucket label of each row of ``column`` and the label dtype.

    A label is null where the time is missing or cannot be placed (unparseable
    text, a naive time that is ambiguous for its bucket or does not exist in
    the zone); callers count those rows, they are never dropped silently.

    Raises:
        ValueError: if the bucket cannot apply to the column type (a calendar
            bucket of a numeric column, an hourly bucket of dates, ...).
    """
    value, parsed = _parsed(column, dtype, axis)
    bucket = axis.bucket
    zone = _zone(parsed, axis)
    if bucket is None:
        if isinstance(parsed, pl.Datetime) and axis.time_zone is not None:
            # Exact instants in the zone; an ambiguous or skipped naive time has none.
            value = zoned(
                value, _aware(parsed), axis.time_zone, ambiguous="null", non_existent="null"
            )
        return value, _dtype_of(value, column, dtype)
    if parsed.is_numeric():
        raise ValueError(
            f"time column {column!r} is numeric ({parsed}); a calendar bucket needs a "
            "Date or Datetime column (omit 'bucket' to use the numbers as periods)"
        )
    if parsed == pl.Date and (bucket.subday or bucket.unit == "hour_of_day"):
        raise ValueError(f"time column {column!r} holds dates; it has no {bucket.describe()}")
    local, offset, placeable = _local(value, parsed, zone)
    if bucket.cyclic:
        label = _profile(local, bucket)
    elif not bucket.subday:
        label = _calendar_date(local, parsed, bucket)
    else:
        start = _subday_start(local, parsed, bucket)
        label = start if zone is None else _start_instant(start, offset, zone)
        label = label.cast(pl.Datetime("us", zone))
    label = pl.when(placeable).then(label)
    return label, _dtype_of(label, column, dtype)


def unplaced(column: str, label: pl.Expr) -> pl.Expr:
    """Rows with a time value that has no bucket label."""
    return pl.col(column).is_not_null() & label.is_null()


# ----------------------------------------------------------------- ordinals
def calendar_position(label: pl.Expr, dtype: pl.DataType, bucket: Bucket | None) -> pl.Expr:
    """Exact wall-clock position of a label: the slot it occupies on the local calendar.

    This is the position for anything cyclic or seasonal (``position mod n``,
    profiles): calendar buckets count from the epoch on the *local* clock, so an
    hourly bucket's position mod 24 is its local hour. The two physical hours of a
    DST fall-back share one position (two observations of the same slot, still two
    buckets), and the hour a spring-forward skips has none. Cyclic profiles count
    from the start of the cycle (``week_start`` for weekdays). Without a bucket,
    integer periods are their own position and dates count days.

    For spacing in time (rates, slopes, lags) use :func:`sequence_positions`,
    which differs from this only for sub-day buckets across DST changes.

    Raises:
        ValueError: for labels without an exact position (float or instant periods).
    """
    if bucket is None:
        if dtype.is_integer():
            return label.cast(pl.Int64)
        if dtype == pl.Date:
            return label.cast(pl.Int32).cast(pl.Int64)
        raise ValueError(f"periods of type {dtype} have no exact position; set a bucket")
    if bucket.cyclic:
        if bucket.unit == "season":
            return label.to_physical().cast(pl.Int64)
        if bucket.unit == "day_of_week":
            return (label.cast(pl.Int64) - 1 - WEEKDAYS.index(bucket.week_start)) % 7
        first = 0 if bucket.unit == "hour_of_day" else 1
        return label.cast(pl.Int64) - first
    if bucket.subday:
        local = wall_clock(label, dtype, None)
        return local.dt.epoch("us") // bucket.width_us
    days = label.cast(pl.Int32).cast(pl.Int64)
    if bucket.unit == "day":
        return days // bucket.every
    if bucket.unit == "week":
        return (days - _week_anchor(bucket)) // (7 * bucket.every)
    return _month_index(label) // bucket.months


def _from_ordinal(position: pl.Expr, bucket: Bucket) -> pl.Expr:
    """Date label of a day-or-longer bucket at an ordinal (inverse of the above)."""
    if bucket.unit == "day":
        return (position * bucket.every).cast(pl.Int32).cast(pl.Date)
    if bucket.unit == "week":
        return (_week_anchor(bucket) + position * 7 * bucket.every).cast(pl.Int32).cast(pl.Date)
    return _from_months(position * bucket.months)


def _cycle(bucket: Bucket, dtype: pl.DataType) -> pl.Series:
    if bucket.unit == "season":
        return pl.Series("label", list(bucket.season_names), dtype=dtype)
    if bucket.unit == "day_of_week":
        start = WEEKDAYS.index(bucket.week_start)
        values = [(start + k) % 7 + 1 for k in range(7)]
    elif bucket.unit == "hour_of_day":
        values = list(range(24))
    elif bucket.unit == "day_of_year":
        # Day 366 exists only in leap years; it is never *expected*.
        values = list(range(1, 366))
    else:
        values = list(range(1, (bucket.cycle_length or 0) + 1))
    return pl.Series("label", values, dtype=dtype)


def bucket_grid(
    first: pl.Series, last: pl.Series, bucket: Bucket | None, *, limit: int
) -> pl.Series:
    """Every bucket label from ``first`` to ``last`` (one-element label series), in order.

    The expected grid for completeness: missing buckets are the grid minus the
    labels present. Cyclic profiles expect the whole cycle (except day 366).
    Sub-day grids in a zone with DST follow the local clock: the skipped hour
    is absent and the repeated hour appears twice.

    Raises:
        ValueError: if the grid would exceed ``limit`` labels, or the labels have
            no exact grid (float periods, unbucketed instants).
    """
    dtype = first.dtype
    if bucket is not None and bucket.cyclic:
        cycle = _cycle(bucket, dtype)
        _bounded(len(cycle), limit)
        return cycle
    bounds = pl.DataFrame({"first": first, "last": last})
    if bucket is not None and bucket.subday:
        assert isinstance(dtype, pl.Datetime)
        zone = dtype.time_zone
        local = bounds.select(
            (wall_clock(pl.col(c), dtype, zone) if zone else pl.col(c)).dt.epoch("us")
            for c in ("first", "last")
        ).row(0)
        # A skipped wall-clock start may label its first physical instant (e.g.
        # 02:00 -> 03:00 for a two-hour bucket). Keep the midnight-based phase,
        # rather than using that adjusted instant as the enumeration origin.
        low = (local[0] // bucket.width_us) * bucket.width_us
        high = (local[1] // bucket.width_us) * bucket.width_us
        size = (high - low) // bucket.width_us + 1
        _bounded(size, limit)
        starts = pl.select(
            pl.from_epoch(
                pl.int_range(0, size, dtype=pl.Int64) * bucket.width_us + low,
                time_unit="us",
            ).alias("label")
        )
        if zone is None:
            return starts["label"]
        instants = starts.select(
            pl.concat_list(_occurrences(pl.col("label"), zone)).alias("label")
        ).explode("label")
        grid = instants.drop_nulls().unique().sort("label")["label"].cast(dtype)
        # Compare as Polars instants: Python datetimes of a fall-back hour are equal.
        grid = grid.filter((grid >= first) & (grid <= last))
        _bounded(len(grid), limit)
        return grid
    if bucket is None:
        if not (dtype.is_integer() or dtype == pl.Date):
            raise ValueError(f"periods of type {dtype} have no exact grid; set a bucket")
        positions = bounds.select(calendar_position(pl.col(c), dtype, None) for c in bounds.columns)
        low, high = positions.row(0)
        _bounded(high - low + 1, limit)
        grid = pl.int_range(low, high + 1, dtype=pl.Int64, eager=True)
        return grid.cast(pl.Int32).cast(pl.Date) if dtype == pl.Date else grid.cast(dtype)
    positions = bounds.select(calendar_position(pl.col(c), dtype, bucket) for c in bounds.columns)
    low, high = positions.row(0)
    _bounded(high - low + 1, limit)
    return pl.select(
        _from_ordinal(pl.int_range(low, high + 1, dtype=pl.Int64), bucket).alias("label")
    )["label"]


def sequence_positions(
    labels: pl.Series,
    bucket: Bucket | None,
    *,
    first: pl.Series | None = None,
    last: pl.Series | None = None,
    limit: int = 10_000_000,
) -> pl.Series:
    """Exact sequence position of each label: consecutive buckets differ by exactly one.

    The position for spacing in time: rates per bucket, trend slopes, lags and
    peak shifts. It counts buckets of the exact grid (:func:`bucket_grid`), so the
    two hours of a DST fall-back are consecutive and the hour a spring-forward
    skips is not counted, whatever the zone's offsets and the bucket width. Only
    differences are meaningful: positions count from ``first`` (the earliest
    label by default; pass the same ``first`` to compare series). Day and longer
    buckets, cyclic profiles and unbucketed periods equal their
    :func:`calendar_position`.

    Raises:
        ValueError: for labels without an exact position, or a DST grid beyond
            ``limit``. Naive and fixed-offset labels use exact tick arithmetic
            without materializing intermediate buckets, so their span is not
            restricted by the grid allocation limit.
    """
    frame = labels.alias("label").to_frame()
    if bucket is None or not bucket.subday:
        return frame.select(calendar_position(pl.col("label"), labels.dtype, bucket))["label"]
    present = labels.drop_nulls().sort()
    if present.is_empty():
        return pl.Series("label", [None] * len(labels), dtype=pl.Int64)
    start = present.head(1) if first is None else first
    end = present.tail(1) if last is None else last
    dtype = labels.dtype
    if isinstance(dtype, pl.Datetime) and (
        dtype.time_zone is None or ZoneInfo(dtype.time_zone).utcoffset(None) is not None
    ):
        if start.dtype != dtype or end.dtype != dtype:
            raise ValueError("bucket grid bounds must have the same dtype as the labels")
        if len(start) != 1 or len(end) != 1 or start.null_count() or end.null_count():
            raise ValueError("bucket grid bounds must each contain one non-null label")
        low, high = int(start.cast(pl.Int64).item()), int(end.cast(pl.Int64).item())
        width = bucket.width_us * 1_000 if dtype.time_unit == "ns" else bucket.width_us
        if dtype.time_unit == "ms":
            width //= 1_000
        # Int128 subtraction preserves differences across the Int64 epoch boundary.
        ticks = pl.col("label").cast(pl.Int64).cast(pl.Int128)
        delta = ticks - pl.lit(low, dtype=pl.Int128)
        step = pl.lit(width, dtype=pl.Int128)
        valid = (ticks >= low) & (ticks <= high) & (delta % step == 0)
        computed = frame.select(
            (delta // step).cast(pl.Int64).alias("label"), valid.alias("valid")
        )
        if low > high or not computed["valid"].drop_nulls().all():
            raise ValueError("a label lies outside the bucket grid of the series")
        return computed["label"]
    grid = bucket_grid(start, end, bucket, limit=limit).alias("label").to_frame()
    indexed = grid.with_row_index("position").with_columns(pl.col("position").cast(pl.Int64))
    row = "row"
    positions = (
        frame.with_row_index(row).join(indexed, on="label", how="left").sort(row)["position"]
    )
    if positions.null_count() != labels.null_count():
        raise ValueError("a label lies outside the bucket grid of the series")
    return positions


def _bounded(size: int, limit: int) -> None:
    if size > limit:
        raise ValueError(f"the expected grid has {size:,} buckets, more than the limit {limit:,}")


def bucket_scalar(value: Any, column: str, dtype: pl.DataType, axis: TimeBucketing) -> pl.Series:
    """The label of one configured time (a window bound) as the data's column would get it.

    Dates and datetimes are first placed on the column's storage grid with
    :func:`~qavf.rules._temporal.temporal_ticks` (a naive value on a zoned column
    is wall-clock time there), so a bound and the data are bucketed identically.
    """
    parsed = _parsed(column, dtype, axis)[1]
    temporal = parsed == pl.Date or isinstance(parsed, pl.Datetime)
    if isinstance(value, date | datetime) and temporal:
        grid = parsed
        if isinstance(parsed, pl.Datetime) and axis.time_zone is not None:
            # A naive bound is wall-clock time where the data's naive times are.
            if parsed.time_zone is not None:
                grid = pl.Datetime(parsed.time_unit, axis.time_zone)
            elif isinstance(value, datetime) and value.tzinfo is not None:
                value = value.astimezone(ZoneInfo(axis.time_zone)).replace(tzinfo=None)
        ticks = math.floor(temporal_ticks(value, grid))
        storage = pl.Int32 if parsed == pl.Date else pl.Int64
        frame = pl.Series(column, [ticks], dtype=storage).cast(parsed).to_frame()
    elif isinstance(value, date | datetime) or temporal:
        raise ValueError(f"{value!r} is not comparable with time column {column!r} ({parsed})")
    else:
        frame = pl.Series(column, [value]).cast(parsed, strict=True).to_frame()
    plain = TimeBucketing(bucket=axis.bucket, time_zone=axis.time_zone)
    label, _ = bucket_labels(column, parsed, plain)
    result = frame.select(label.alias("label"))["label"]
    if result.null_count():
        raise ValueError(f"{value!r} cannot be placed in a {plain.describe()} bucket")
    return result
