"""TMP-001..008 on bucketed Date/Datetime series, and unchanged numeric series."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)
NY = ZoneInfo("America/New_York")


def _run(rule, frame, series=None, **params):
    spec = {"dataset": "s", "value": "v", "time_column": "t", **(series or {})}
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {"s": {}},
            "rules": {rule: {"params": {"series": [spec], **params}}},
        }
    )
    return QARunner(settings, REGISTRY).run(frames={"s": frame}).results_for(rule)


def _main(results):
    return results[-1]


def _hourly(daily: dict[date, float], group: str = "a") -> pl.DataFrame:
    """24 rows per day whose values sum to the day's total."""
    rows = [
        (group, datetime(day.year, day.month, day.day, hour, 15), total / 24)
        for day, total in daily.items()
        for hour in range(24)
    ]
    return pl.DataFrame(rows, schema=["g", "t", "v"], orient="row")


DAYS = [date(2026, 1, 1) + timedelta(days=k) for k in range(6)]


# ------------------------------------------------------------------ TMP-001
def test_step_changes_are_per_bucket_of_daily_sums():
    frame = _hourly({day: 240.0 for day in DAYS} | {DAYS[3]: 480.0})
    results = _run("TMP-001", frame, {"bucket": "day"})
    main = _main(results)
    assert main.status is Status.FAIL and main.affected_count == 2  # up, then back down
    assert [r["t"] for r in main.affected_records] == [DAYS[3], DAYS[4]]
    assert main.total_records == 5
    # A rate spans the buckets between two periods: 100 -> 121 over two days is 10 %/day.
    gappy = _hourly({DAYS[0]: 100.0, DAYS[2]: 121.0})
    assert _main(_run("TMP-001", gappy, {"bucket": "day"}, max_change_rate=0.11)).status is (
        Status.PASS
    )
    assert _main(_run("TMP-001", gappy, {"bucket": "day"}, max_change_rate=0.09)).status is (
        Status.FAIL
    )


def test_year_buckets_of_dates_equal_numeric_year_periods():
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    values = [10.0, 11.0, 30.0, 12.0, 0.0, 13.0]
    numeric = pl.DataFrame({"t": years, "v": values})
    dated = pl.DataFrame({"t": [date(y, 7, 1) for y in years], "v": values})
    for rule in ("TMP-001", "TMP-004", "TMP-006"):
        plain = _main(_run(rule, numeric))
        bucketed = _main(_run(rule, dated, {"bucket": "year"}))
        assert (bucketed.status, bucketed.affected_count, bucketed.total_records) == (
            plain.status,
            plain.affected_count,
            plain.total_records,
        )


# ------------------------------------------------------------------ TMP-004
def test_missing_buckets_include_a_day_missing_from_every_group():
    present = [d for d in DAYS if d != DAYS[2]]
    frame = pl.concat(
        [
            _hourly({d: 1.0 for d in present}, "a"),
            _hourly({d: 1.0 for d in present if d != DAYS[4]}, "b"),
        ]
    )
    main = _main(_run("TMP-004", frame, {"bucket": "day", "group_by": ["g"]}))
    assert main.status is Status.FAIL and main.affected_count == 2
    missing = {r["g"]: r["missing_periods"] for r in main.affected_records}
    assert missing == {"a": [DAYS[2]], "b": [DAYS[2], DAYS[4]]}
    # Unbucketed numeric periods keep their old semantics: the observed union.
    numeric = frame.with_columns(pl.col("t").dt.day().alias("t"))
    plain = _main(_run("TMP-004", numeric, {"group_by": ["g"]}))
    assert plain.affected_count == 1


def test_expected_periods_are_dates_for_bucketed_series():
    frame = _hourly({d: 1.0 for d in DAYS[:3]})
    expected = [DAYS[0], DAYS[3]]
    main = _main(_run("TMP-004", frame, {"bucket": "day"}, expected_periods=expected))
    assert main.status is Status.FAIL
    assert main.affected_records[0]["missing_periods"] == [DAYS[3]]
    numeric = frame.with_columns(pl.col("t").dt.day().alias("t"))
    [blocked, *_] = _run("TMP-004", numeric, expected_periods=expected)
    assert blocked.status is Status.ERROR and "bucket" in blocked.message


def test_a_missing_fall_back_hour_is_not_merged_with_its_twin():
    start = datetime(2026, 11, 1, 4, tzinfo=UTC)  # 00:00 EDT
    instants = [start + timedelta(hours=h, minutes=10) for h in range(25)]
    second_one_am = instants.pop(2)  # 01:10 EST, the repeated hour
    frame = pl.DataFrame({"t": instants, "v": [1.0] * len(instants)}).with_columns(
        pl.col("t").dt.convert_time_zone("America/New_York")
    )
    main = _main(_run("TMP-004", frame, {"bucket": "hour"}))
    assert main.status is Status.FAIL
    [missing] = main.affected_records[0]["missing_periods"]
    assert missing.timestamp() == second_one_am.replace(minute=0).timestamp()
    days = _main(_run("TMP-004", frame, {"bucket": "day", "time_zone": "America/New_York"}))
    assert days.status is Status.PASS


# ------------------------------------------------------------- TMP-002/003
def test_trend_rules_use_bucket_positions():
    pytest.importorskip("scipy")
    months = [date(2024, m, 1) for m in range(1, 13)]
    falling = pl.DataFrame({"t": months, "v": [float(100 - m) for m in range(12)]})
    series = {"bucket": "month", "expected_trend": "increasing"}
    assert _main(_run("TMP-002", falling, series)).status is Status.FAIL
    values = [float(10 * m) for m in range(12)]
    values[6] = 400.0
    outlier = _main(_run("TMP-003", pl.DataFrame({"t": months, "v": values}), {"bucket": "month"}))
    assert outlier.status is Status.FAIL
    assert outlier.affected_records[0]["t"] == date(2024, 7, 1)


# --------------------------------------------------------- TMP-005/006/008
def test_stale_cumulative_and_level_shift_rules_on_buckets():
    days = [date(2026, 1, 1) + timedelta(days=k) for k in range(12)]
    stale = pl.DataFrame({"t": days, "v": [1.0, 2.0, 5.0, 5.0, 5.0, 5.0, *range(6)]})
    run = _main(_run("TMP-005", stale, {"bucket": "day"}, copied_periods=False))
    assert run.status is Status.FAIL
    assert (run.affected_records[0]["first_period"], run.affected_records[0]["periods"]) == (
        days[2],
        4,
    )
    falling = pl.DataFrame({"t": days, "v": [float(k) for k in range(12)]}).with_columns(
        pl.when(pl.col("v") == 7).then(1.0).otherwise(pl.col("v")).alias("v")
    )
    cumulative = _main(_run("TMP-006", falling, {"bucket": "day"}))
    assert cumulative.status is Status.FAIL and cumulative.affected_records[0]["period"] == days[7]
    shifted = pl.DataFrame({"t": days, "v": [10.0, 11, 10, 9, 10, 10, 50, 51, 49, 50, 52, 50]})
    level = _main(_run("TMP-008", shifted, {"bucket": "day"}))
    assert level.status is Status.FAIL and level.affected_records[0]["shift_period"] == days[6]


# ------------------------------------------------------------------ TMP-007
def test_monthly_buckets_give_calendar_seasons():
    pattern = [10, 12, 15, 20, 30, 40, 45, 42, 30, 20, 14, 11]
    months = [date(2020 + m // 12, m % 12 + 1, 1) for m in range(60)]
    values = [float(pattern[m % 12]) for m in range(60)]
    values[30] *= 3
    # Daily rows within each month sum to the monthly value.
    rows = [
        (month + timedelta(days=d), value / 4)
        for month, value in zip(months, values, strict=True)
        for d in range(4)
    ]
    frame = pl.DataFrame(rows, schema=["t", "v"], orient="row")
    main = _main(_run("TMP-007", frame, {"bucket": "month"}, season_length=12))
    assert main.status is Status.FAIL
    [record] = main.affected_records
    assert record["period"] == months[30]


# --------------------------------------------------------- fail closed
def test_unplaceable_times_exclude_their_group_with_evidence():
    frame = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "t": [
                datetime(2026, 11, 1, 1, 30),  # ambiguous hour in New York
                datetime(2026, 11, 1, 3, 30),
                datetime(2026, 11, 1, 2, 30),
                datetime(2026, 11, 1, 3, 30),
            ],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    series = {"bucket": "hour", "time_zone": "America/New_York", "group_by": ["g"]}
    results = _run("TMP-006", frame, series)
    blocked = next(r for r in results if r.status is Status.ERROR)
    assert blocked.affected_count == 1 and blocked.affected_records[0]["g"] == "a"
    assert "cannot be placed" in blocked.message
    assert _main(results).status is Status.PASS  # group b is still assessed
    text = pl.DataFrame({"t": ["2026-01-01", "2026-01-02", "garbage"], "v": [1.0, 2.0, 3.0]})
    parsed = _run("TMP-006", text, {"bucket": "day", "time_format": "%Y-%m-%d"})
    assert any(r.status is Status.ERROR and r.affected_count == 1 for r in parsed)
    assert not any(r.status is Status.PASS for r in parsed)


@pytest.mark.parametrize(
    ("frame", "series", "message"),
    [
        (pl.DataFrame({"t": [2020, 2021], "v": [1.0, 2.0]}), {"bucket": "year"}, "numeric"),
        (pl.DataFrame({"t": [date(2020, 1, 1)], "v": [1.0]}), {"bucket": "hour"}, "dates"),
        (pl.DataFrame({"t": ["2020"], "v": [1.0]}), {"bucket": "year"}, "time_format"),
        (pl.DataFrame({"t": [date(2020, 1, 1)], "v": [1.0]}), {}, "numeric"),
    ],
)
def test_unusable_time_columns_are_not_assessed(frame, series, message):
    [result] = _run("TMP-001", frame, series)
    assert result.status is Status.ERROR and message in result.message
