"""Time buckets against independent oracles: Python's calendar and zoneinfo arithmetic.

Python resolves wall-clock times with PEP 495 folds (an ambiguous time takes the
fold of the row it came from, a skipped time the offset before the gap), which
is the documented bucket semantics, so it is a complete oracle for labels.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qavf.temporal import (
    Bucket,
    CalendarOffset,
    TimeBucketing,
    bucket_grid,
    bucket_labels,
    bucket_scalar,
    calendar_position,
    sequence_positions,
    shifted_time,
)

NY = ZoneInfo("America/New_York")
FALL = datetime(2026, 11, 1, 4, tzinfo=UTC)  # 00:00 EDT on the fall-back day
SPRING = datetime(2026, 3, 8, 5, tzinfo=UTC)  # 00:00 EST on the spring-forward day


def labels(values, unit, dtype=None, **axis):
    frame = pl.DataFrame({"t": pl.Series(values, dtype=dtype)})
    bucketing = TimeBucketing(bucket=Bucket.model_validate(unit), **axis)
    expression, label_type = bucket_labels("t", frame.schema["t"], bucketing)
    return frame.select(expression.alias("label"))["label"], label_type


def ny(values):
    return pl.Series(values).dt.convert_time_zone("America/New_York")


def oracle_start(instant: datetime, hours: int = 0, minutes: int = 0) -> float:
    """The bucket start instant by Python wall-clock arithmetic (keeps the row's fold)."""
    local = instant.astimezone(NY)
    if hours:
        start = local.replace(hour=local.hour // hours * hours, minute=0, second=0, microsecond=0)
    else:
        start = local.replace(minute=local.minute // minutes * minutes, second=0, microsecond=0)
    # The fold is kept: a repeated start takes the row's occurrence, an unambiguous one
    # ignores it, and a skipped one (fold 0) reads with the offset before the gap.
    return start.timestamp()


# ------------------------------------------------------------ shorthand
@pytest.mark.parametrize(
    ("text", "unit", "every"),
    [
        ("day", "day", 1),
        ("15 minutes", "minute", 15),
        ("15min", "minute", 15),
        ("6h", "hour", 6),
        ("2 weeks", "week", 2),
        ("3mo", "month", 3),
        ("hour of day", "hour_of_day", 1),
        ("season", "season", 1),
    ],
)
def test_shorthand(text, unit, every):
    bucket = Bucket.model_validate(text)
    assert (bucket.unit, bucket.every) == (unit, every)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("7h", "does not divide a day"),
        ({"unit": "hour_of_day", "every": 2}, "does not apply"),
        ({"unit": "month", "seasons": {"a": [1]}}, "only applies"),
        ({"unit": "season", "seasons": {"a": [1, 2], "b": [3]}}, "exactly one season"),
        ("fortnight", "unit"),
    ],
)
def test_invalid_buckets_fail_at_configuration(spec, message):
    with pytest.raises(ValueError, match=message):
        Bucket.model_validate(spec)


def test_unknown_time_zone_fails_at_configuration():
    with pytest.raises(ValueError, match="unknown time zone"):
        TimeBucketing(time_zone="Mars/Olympus")


# ------------------------------------------------------- calendar kinds
DATES = [date(2024, 2, 29), date(2023, 12, 31), date(1969, 12, 28), date(2025, 7, 15)]


def _monday_week(day: date, start: int) -> date:
    return day - timedelta(days=(day.weekday() - start) % 7)


@pytest.mark.parametrize(
    ("unit", "oracle"),
    [
        ("day", lambda d: d),
        ("week", lambda d: _monday_week(d, 0)),
        ({"unit": "week", "week_start": "sunday"}, lambda d: _monday_week(d, 6)),
        ("month", lambda d: d.replace(day=1)),
        ("quarter", lambda d: date(d.year, (d.month - 1) // 3 * 3 + 1, 1)),
        ("year", lambda d: date(d.year, 1, 1)),
        ("6 months", lambda d: date(d.year, 1 if d.month <= 6 else 7, 1)),
        ("5 years", lambda d: date(d.year // 5 * 5, 1, 1)),
        (
            "3 days",
            lambda d: date(1970, 1, 1) + timedelta(days=(d - date(1970, 1, 1)).days // 3 * 3),
        ),
    ],
)
def test_calendar_buckets_of_dates_and_datetimes(unit, oracle):
    expected = [oracle(day) for day in DATES]
    by_date, dtype = labels(DATES, unit)
    assert dtype == pl.Date and by_date.to_list() == expected
    stamps = [datetime(d.year, d.month, d.day, 23, 59, 59, 999_999) for d in DATES]
    for unit_name in ("ms", "us", "ns"):
        by_time, _ = labels(stamps, unit, pl.Datetime(unit_name))
        assert by_time.to_list() == expected


@pytest.mark.parametrize(
    ("unit", "minutes"),
    [("second", 1 / 60), ("minute", 1), ("15 minutes", 15), ("hour", 60), ("6 hours", 360)],
)
def test_subday_buckets_of_naive_datetimes(unit, minutes):
    stamps = [
        datetime(2026, 1, 1) + timedelta(seconds=s) for s in (0, 59, 61, 3599, 21_601, 86_399)
    ]
    result, dtype = labels(stamps, unit, pl.Datetime("ns"))
    width = timedelta(minutes=minutes)
    midnight = datetime(2026, 1, 1)
    expected = [midnight + (s - midnight) // width * width for s in stamps]
    assert dtype == pl.Datetime("us") and result.to_list() == expected


@pytest.mark.parametrize(
    ("unit", "oracle"),
    [
        ("hour_of_day", lambda t: t.hour),
        ("day_of_week", lambda t: t.isoweekday()),
        ("day_of_year", lambda t: t.timetuple().tm_yday),
        ("month_of_year", lambda t: t.month),
        ("quarter_of_year", lambda t: (t.month - 1) // 3 + 1),
    ],
)
def test_cyclic_profiles_follow_local_time(unit, oracle):
    instants = [FALL + timedelta(hours=h) for h in range(0, 400, 7)]
    result, _ = labels(ny(instants), unit)
    assert result.to_list() == [oracle(t.astimezone(NY)) for t in instants]


@pytest.mark.parametrize(
    ("hemisphere", "seasons", "expected"),
    [
        ("north", None, ["winter", "spring", "summer", "autumn", "winter"]),
        ("south", None, ["summer", "autumn", "winter", "spring", "summer"]),
        ("north", {"wet": [5, 6, 7, 8, 9, 10], "dry": [11, 12, 1, 2, 3, 4]}, None),
    ],
)
def test_seasons(hemisphere, seasons, expected):
    days = [date(2025, 1, 5), date(2025, 4, 5), date(2025, 7, 5), date(2025, 10, 5)]
    days.append(date(2025, 12, 5))
    spec = {"unit": "season", "hemisphere": hemisphere, "seasons": seasons}
    result, dtype = labels(days, {k: v for k, v in spec.items() if v is not None})
    if expected is None:
        expected = ["dry", "dry", "wet", "wet", "dry"]
    assert result.cast(pl.String).to_list() == expected
    assert isinstance(dtype, pl.Enum)


def test_day_of_week_order_follows_the_week_start():
    days = [date(2026, 1, 4) + timedelta(days=k) for k in range(7)]  # Sunday..Saturday
    bucket = Bucket(unit="day_of_week", week_start="sunday")
    frame = pl.DataFrame({"d": days})
    label, dtype = bucket_labels("d", pl.Date, TimeBucketing(bucket=bucket))
    positions = frame.select(calendar_position(label, dtype, bucket))["d"].to_list()
    assert positions == list(range(7))
    grid = bucket_grid(pl.Series([1]), pl.Series([1]), bucket, limit=10)
    assert grid.to_list() == [7, 1, 2, 3, 4, 5, 6]


# ------------------------------------------------------------ time zones
@pytest.mark.parametrize("unit", ["hour", "30 minutes", "2 hours", "6 hours"])
@pytest.mark.parametrize("anchor", [FALL, SPRING])
def test_subday_buckets_across_dst_match_python_folds(unit, anchor):
    instants = [anchor - timedelta(hours=2) + timedelta(minutes=m) for m in range(0, 600, 20)]
    result, dtype = labels(ny(instants), unit)
    assert dtype == pl.Datetime("us", "America/New_York")
    bucket = Bucket.model_validate(unit)
    hours, minutes = divmod(bucket.width_us // 60_000_000, 60)
    expected = [oracle_start(t, hours=hours, minutes=minutes if not hours else 0) for t in instants]
    assert [label.timestamp() for label in result.to_list()] == expected


def test_fall_back_hours_stay_distinct_and_days_have_25_or_23_hours():
    hourly = [FALL + timedelta(minutes=30 + 60 * h) for h in range(25)]
    hours, _ = labels(ny(hourly), "hour")
    days, _ = labels(ny(hourly), "day")
    assert hours.n_unique() == 25  # the repeated 01:00 hour is two buckets
    assert days.unique().to_list() == [date(2026, 11, 1)]
    spring = [SPRING + timedelta(minutes=30 + 60 * h) for h in range(23)]
    assert labels(ny(spring), "day")[0].unique().to_list() == [date(2026, 3, 8)]
    assert labels(ny(spring), "hour")[0].n_unique() == 23
    # Hourly counts per day: 25 and 23 rows, each in its own hour.
    first, last = hours.head(1), hours.tail(1)
    assert len(bucket_grid(first, last, Bucket(unit="hour"), limit=100)) == 25
    spring_hours = labels(ny(spring), "hour")[0]
    grid = bucket_grid(spring_hours.head(1), spring_hours.tail(1), Bucket(unit="hour"), limit=100)
    assert len(grid) == 23
    assert {g.timestamp() for g in grid.to_list()} == {
        datetime(2026, 3, 8, 5, tzinfo=UTC).timestamp() + 3600 * h for h in range(23)
    }


def _hour_labels(instants, unit="hour", zone="America/New_York"):
    frame = pl.DataFrame({"t": pl.Series(instants).dt.convert_time_zone(zone)})
    bucket = Bucket.model_validate(unit)
    label, dtype = bucket_labels("t", frame.schema["t"], TimeBucketing(bucket=bucket))
    return frame.select(label)["t"], dtype, bucket


@pytest.mark.parametrize("anchor", [FALL, SPRING])
def test_sequence_and_calendar_positions_across_dst(anchor):
    """Sequence: consecutive buckets one apart. Calendar: the local hour's slot."""
    hourly = [anchor - timedelta(hours=3) + timedelta(hours=h, minutes=10) for h in range(30)]
    labels, dtype, bucket = _hour_labels(hourly)
    sequence = sequence_positions(labels, bucket).to_list()
    assert sequence == list(range(30))
    slots = labels.to_frame().select(calendar_position(pl.col("t"), dtype, bucket))["t"]
    # Oracle: Python's local hour of each instant.
    assert (slots % 24).to_list() == [t.astimezone(NY).hour for t in hourly]


def test_fall_back_hours_share_a_calendar_slot_and_the_skipped_hour_has_none():
    fall = [FALL + timedelta(hours=h) for h in range(25)]
    labels, dtype, bucket = _hour_labels(fall)
    slots = labels.to_frame().select(calendar_position(pl.col("t"), dtype, bucket) % 24)["t"]
    assert sorted(slots.to_list()) == sorted([*range(24), 1])  # 01:00 twice, kept apart
    assert labels.n_unique() == 25
    spring = [SPRING + timedelta(hours=h) for h in range(23)]
    labels, dtype, bucket = _hour_labels(spring)
    slots = labels.to_frame().select(calendar_position(pl.col("t"), dtype, bucket) % 24)["t"]
    assert sorted(slots.to_list()) == [h for h in range(24) if h != 2]


@pytest.mark.parametrize("anchor", [datetime(2026, 3, 8, 4), datetime(2026, 11, 1, 3)])
def test_sequence_positions_have_no_gaps_in_half_hour_offset_zones(anchor):
    """Newfoundland (UTC-3:30, DST -2:30) with 2-hour buckets: the bucket containing the
    change is 1 or 3 hours long, yet the next bucket is the next position."""
    instants = [anchor.replace(tzinfo=UTC) + timedelta(minutes=30 * k) for k in range(40)]
    labels, _, bucket = _hour_labels(instants, "2 hours", "America/St_Johns")
    distinct = labels.unique(maintain_order=True)
    positions = sequence_positions(distinct, bucket).to_list()
    assert positions == list(range(len(distinct)))
    # Oracle: every label is a 2-hour local wall-clock start (or the end of the gap).
    local = [t.astimezone(ZoneInfo("America/St_Johns")) for t in distinct.to_list()]
    assert all(t.minute == 0 and (t.hour % 2 == 0 or t.hour == 3) for t in local)


def test_naive_times_are_wall_clock_in_the_configured_zone():
    naive = [
        datetime(2026, 11, 1, 1, 30),  # ambiguous: two instants
        datetime(2026, 3, 8, 2, 30),  # skipped by the gap
        datetime(2026, 7, 1, 12, 45),
        None,
    ]
    hours, _ = labels(naive, "hour", time_zone="America/New_York")
    assert hours[0] is None and hours[1] is None and hours[3] is None
    assert hours[2].timestamp() == datetime(2026, 7, 1, 12, tzinfo=NY).timestamp()
    days, _ = labels(naive, "day", time_zone="America/New_York")
    # The ambiguous time has a calendar day; the nonexistent one cannot be placed.
    assert days.to_list() == [date(2026, 11, 1), None, date(2026, 7, 1), None]
    # The 6-hour bucket of an ambiguous time is unambiguous.
    six, _ = labels(naive, "6 hours", time_zone="America/New_York")
    assert six[0].timestamp() == datetime(2026, 11, 1, 0, tzinfo=NY).timestamp()


def test_aware_times_are_converted_to_the_configured_zone():
    instants = [datetime(2026, 1, 1, 3, tzinfo=UTC), datetime(2026, 1, 1, 6, tzinfo=UTC)]
    days, _ = labels(instants, "day", time_zone="America/New_York")
    assert days.to_list() == [date(2025, 12, 31), date(2026, 1, 1)]
    utc_days, _ = labels(instants, "day")
    assert utc_days.to_list() == [date(2026, 1, 1), date(2026, 1, 1)]
    kolkata, _ = labels(instants, "hour", time_zone="Asia/Kolkata")  # a half-hour offset
    assert [k.astimezone(ZoneInfo("Asia/Kolkata")).minute for k in kolkata] == [0, 0]


# --------------------------------------------------------- column types
def test_numeric_periods_pass_through_and_cannot_take_calendar_buckets():
    frame = pl.DataFrame({"year": [2020, 2021]})
    expression, dtype = bucket_labels("year", pl.Int64, TimeBucketing())
    assert frame.select(expression)["year"].to_list() == [2020, 2021] and dtype == pl.Int64
    with pytest.raises(ValueError, match="numeric"):
        bucket_labels("year", pl.Int64, TimeBucketing(bucket=Bucket(unit="year")))


def test_dates_have_no_hours():
    for unit in ("hour", "hour_of_day"):
        with pytest.raises(ValueError, match="holds dates"):
            bucket_labels("d", pl.Date, TimeBucketing(bucket=Bucket.model_validate(unit)))


def test_text_times_parse_with_a_format_and_report_unparseable_values():
    values = ["2026-01-05 10:15", "2026-01-05 23:59", "not a time", None]
    result, _ = labels(values, "day", time_format="%Y-%m-%d %H:%M")
    assert result.to_list() == [date(2026, 1, 5), date(2026, 1, 5), None, None]
    with pytest.raises(ValueError, match="time_format"):
        labels(values, "day")
    with pytest.raises(ValueError, match="applies to text"):
        labels([date(2026, 1, 1)], "day", time_format="%Y")
    with pytest.raises(ValueError, match="must be numeric, a Date or a Datetime"):
        bucket_labels("b", pl.Boolean, TimeBucketing(bucket=Bucket(unit="day")))


# ------------------------------------------------------------------ grids
@pytest.mark.parametrize(
    ("unit", "first", "last", "expected"),
    [
        ("day", date(2024, 2, 27), date(2024, 3, 2), 5),
        ("week", date(2024, 1, 1), date(2024, 12, 30), 53),
        ("month", date(2023, 11, 1), date(2024, 2, 1), 4),
        ("quarter", date(2020, 1, 1), date(2024, 10, 1), 20),
        ("year", date(2000, 1, 1), date(2024, 1, 1), 25),
    ],
)
def test_calendar_grids(unit, first, last, expected):
    bucket = Bucket.model_validate(unit)
    grid = bucket_grid(pl.Series([first]), pl.Series([last]), bucket, limit=100)
    assert len(grid) == expected and grid[0] == first and grid[-1] == last
    assert grid.is_sorted() and grid.n_unique() == expected
    with pytest.raises(ValueError, match="more than the limit"):
        bucket_grid(pl.Series([first]), pl.Series([last]), bucket, limit=expected - 1)


def test_unbucketed_grids_are_integer_periods_or_days():
    assert bucket_grid(pl.Series([2019]), pl.Series([2022]), None, limit=10).to_list() == [
        2019,
        2020,
        2021,
        2022,
    ]
    with pytest.raises(ValueError, match="no exact grid"):
        bucket_grid(pl.Series([1.5]), pl.Series([3.5]), None, limit=10)


def test_window_bounds_bucket_like_the_column():
    axis = TimeBucketing(bucket=Bucket(unit="hour"), time_zone="America/New_York")
    dtype = pl.Datetime("ns", "UTC")
    bound = bucket_scalar(datetime(2026, 7, 1, 12, 40), "t", dtype, axis)
    assert bound[0].timestamp() == datetime(2026, 7, 1, 12, tzinfo=NY).timestamp()
    assert bucket_scalar(date(2026, 7, 1), "t", pl.Date, TimeBucketing())[0] == date(2026, 7, 1)
    with pytest.raises(ValueError, match="not comparable"):
        bucket_scalar(date(2026, 7, 1), "t", pl.Int64, TimeBucketing())


# -------------------------------------------------------------- offsets
def test_calendar_offsets_shift_local_time_and_report_non_invertible_rows():
    offset = CalendarOffset(years=1)
    frame = pl.DataFrame({"d": [date(2025, 3, 1), date(2024, 2, 29), date(2025, 2, 28)]})
    value, unaligned = shifted_time("d", pl.Date, offset, -1, None)
    out = frame.select(value.alias("v"), unaligned.alias("u"))
    assert out["v"].to_list() == [date(2024, 3, 1), None, date(2024, 2, 28)]
    assert out["u"].to_list() == [False, True, False]
    weekly = CalendarOffset(years=1, same_weekday=True)
    value, unaligned = shifted_time("d", pl.Date, weekly, -1, None)
    out = frame.select(value.alias("v"), unaligned.alias("u"))
    assert all(v.weekday() == d.weekday() for v, d in zip(out["v"], frame["d"], strict=True))
    assert not out["u"].any()


def test_offsets_across_dst_keep_wall_clock_and_flag_gaps_and_folds():
    instants = [
        datetime(2026, 3, 9, 7, 30, tzinfo=UTC),  # 03:30 EDT, a day later
        datetime(2026, 3, 9, 6, 30, tzinfo=UTC),  # 02:30 EDT: a day earlier it did not exist
        datetime(2026, 11, 1, 6, 30, tzinfo=UTC),  # 01:30 EST, second of the fold
        datetime(2026, 11, 1, 5, 30, tzinfo=UTC),  # 01:30 EDT, first of the fold
    ]
    frame = pl.DataFrame({"t": ny(instants)})
    value, unaligned = shifted_time("t", frame.schema["t"], CalendarOffset(days=1), -1, None)
    out = frame.select(value.alias("v"), unaligned.alias("u"))
    assert out["u"].to_list() == [False, True, True, False]
    expected = [
        datetime(2026, 3, 8, 3, 30, tzinfo=NY),
        None,
        None,
        datetime(2026, 10, 31, 1, 30, tzinfo=NY),
    ]
    assert [v.timestamp() if v else None for v in out["v"]] == [
        e.timestamp() if e else None for e in expected
    ]
    hourly, flags = shifted_time("t", frame.schema["t"], CalendarOffset(hours=1), -1, None)
    out = frame.select(hourly.alias("v"), flags.alias("u"))
    assert not out["u"].any()
    assert [v.timestamp() for v in out["v"]] == [t.timestamp() - 3600 for t in instants]


# -------------------------------------------------------------- property
@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, 3 * 366 * 24 * 60), st.integers(-1000, 1000)),
        min_size=1,
        max_size=60,
    ),
    st.sampled_from(["day", "week", "month", "6 hours"]),
)
def test_summing_hourly_buckets_then_rebucketing_equals_bucketing_directly(rows, coarse):
    """For an additive measure, bucket nesting commutes with summation (and DST)."""
    origin = datetime(2024, 1, 1, tzinfo=UTC)
    instants = [origin + timedelta(minutes=m) for m, _ in rows]
    frame = pl.DataFrame({"t": ny(instants), "v": [v for _, v in rows]})
    axis = {"time_zone": "America/Chicago"}
    hourly, _ = bucket_labels(
        "t", frame.schema["t"], TimeBucketing(bucket=Bucket(unit="hour"), **axis)
    )
    by_hour = frame.group_by(hourly.alias("h")).agg(pl.col("v").sum())
    outer = TimeBucketing(bucket=Bucket.model_validate(coarse), **axis)
    nested, _ = bucket_labels("h", by_hour.schema["h"], outer)
    direct, _ = bucket_labels("t", frame.schema["t"], outer)
    two_step = by_hour.group_by(nested.alias("b")).agg(pl.col("v").sum()).sort("b")
    one_step = frame.group_by(direct.alias("b")).agg(pl.col("v").sum()).sort("b")
    assert two_step.equals(one_step)
    # And the direct day labels agree with Python's local calendar.
    if coarse == "day":
        chicago = ZoneInfo("America/Chicago")
        totals: dict[date, int] = {}
        for instant, (_, v) in zip(instants, rows, strict=True):
            day = instant.astimezone(chicago).date()
            totals[day] = totals.get(day, 0) + v
        assert dict(one_step.iter_rows()) == totals


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(-20_000, 40_000), min_size=1, max_size=40),
    st.integers(-3, 3),
    st.integers(-14, 14),
    st.integers(-40, 40),
    st.booleans(),
)
def test_shifting_back_and_forth_is_the_identity_for_aligned_rows(
    days, years, months, extra, weekly
):
    """Every row a shift does not refuse comes back exactly; refused rows are flagged."""
    months = 0 if weekly else months
    extra = extra if years or months or extra else 1
    offset = CalendarOffset(years=years, months=months, days=extra, same_weekday=weekly)
    frame = pl.DataFrame({"d": [date(1970, 1, 1) + timedelta(days=d) for d in days]})
    forward, refused = shifted_time("d", pl.Date, offset, 1, None)
    moved = frame.with_columns(forward.alias("m"), refused.alias("r"))
    back, refused_back = shifted_time("m", pl.Date, offset, -1, None)
    out = moved.with_columns(back.alias("b"), refused_back.alias("rb")).filter(~pl.col("r"))
    assert out["b"].to_list() == out["d"].to_list()
    assert not out["rb"].any()
    # Refused rows are exactly those Python's calendar cannot move (31 April, 29 February).
    for value, flagged in moved.select("d", "r").iter_rows():
        total = value.month - 1 + (0 if weekly else months + 12 * years)
        try:
            date(value.year + total // 12, total % 12 + 1, value.day)
            clamps = False
        except ValueError:
            clamps = True
        assert flagged == clamps
