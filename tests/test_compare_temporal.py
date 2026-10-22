"""Temporal comparison levels and calendar alignment against plain-Python oracles."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from qavf import File, ResourceLimits, Status, compare_datasets, validate
from qavf.compare import DatasetComparison
from qavf.execution.os_limits import SUPPORTED_PLATFORMS

NY = ZoneInfo("America/New_York")


def _check(result, level, check):
    return next(c for c in result.level(level).checks if c.check == check)


def _daily(start: date, days: int, sites=("a", "b"), value=lambda site, k: float(k % 5)):
    rows = [
        (
            site,
            datetime(start.year, start.month, start.day) + timedelta(days=k, hours=h),
            value(site, k),
        )
        for site in sites
        for k in range(days)
        for h in (3, 15)
    ]
    return pl.DataFrame(rows, schema=["site", "t", "x"], orient="row")


def _sums(frame: pl.DataFrame) -> dict[tuple[str, date], float]:
    """Oracle: Python sums per site and calendar day."""
    totals: dict[tuple[str, date], float] = defaultdict(float)
    for site, stamp, value in frame.iter_rows():
        totals[(site, stamp.date())] += value
    return dict(totals)


# ---------------------------------------------------------------- temporal
def test_temporal_level_compares_measures_per_bucket_and_group():
    baseline = _daily(date(2025, 3, 1), 20)
    candidate = baseline.with_columns(
        pl.when((pl.col("site") == "b") & (pl.col("t").dt.day() == 7))
        .then(pl.col("x") + 10)
        .otherwise(pl.col("x"))
        .alias("x")
    ).filter(~((pl.col("site") == "a") & (pl.col("t").dt.day() == 12)))
    level = {
        "level": "temporal",
        "time_column": "t",
        "bucket": "day",
        "group_by": ["site"],
        "measures": ["sum(x)"],
    }
    result = compare_datasets(baseline, candidate, levels=[level])
    temporal = result.level("temporal")
    before, after = _sums(baseline), _sums(candidate)
    one_sided = set(before) ^ set(after)
    changed = {key for key in set(before) & set(after) if before[key] != after[key]}
    assert (temporal.only_in_baseline, temporal.only_in_candidate) == (len(one_sided), 0) == (1, 0)
    buckets = _check(result, "temporal", "buckets")
    assert buckets.affected_records == [
        {"site": "a", "bucket": date(2025, 3, 12), "buckets": "only in baseline"}
    ]
    sums = _check(result, "temporal", "sum(x)")
    assert sums.affected_count == len(changed) == 1
    assert sums.affected_records[0]["bucket"] == date(2025, 3, 7)
    assert sums.affected_records[0]["diff"] == 20  # two rows, each +10
    gaps = _check(result, "temporal", "gaps")
    assert gaps.affected_count == 1 and temporal.gaps == {"baseline": 0, "candidate": 1}
    assert temporal.expected_buckets == 20
    assert (temporal.window_start, temporal.window_end) == (date(2025, 3, 1), date(2025, 3, 20))


def test_largest_changes_are_bounded_and_ordered():
    baseline = _daily(date(2025, 1, 1), 30, sites=("a",))
    candidate = baseline.with_columns((pl.col("x") + pl.col("t").dt.day()).alias("x"))
    result = compare_datasets(
        baseline,
        candidate,
        levels=[{"level": "temporal", "time_column": "t", "bucket": "day", "measures": ["sum(x)"]}],
        resources=ResourceLimits(evidence_rows=3),
    )
    sums = _check(result, "temporal", "sum(x)")
    assert sums.affected_count == 30 and sums.truncated
    assert [r["bucket"] for r in sums.affected_records] == [
        date(2025, 1, 30),
        date(2025, 1, 29),
        date(2025, 1, 28),
    ]


def test_fall_back_hours_are_separate_buckets_and_the_day_has_25():
    start = datetime(2026, 11, 1, 4, tzinfo=UTC)  # 00:00 EDT
    hours = [start + timedelta(hours=h, minutes=20) for h in range(25)]
    baseline = pl.DataFrame({"t": hours, "x": [1] * 25}).with_columns(
        pl.col("t").dt.convert_time_zone("America/New_York")
    )
    candidate = baseline.filter(pl.col("t").dt.convert_time_zone("UTC") != hours[2])  # 01:20 EST
    hourly = {"level": "temporal", "time_column": "t", "bucket": "hour", "measures": ["count"]}
    daily = {**hourly, "name": "daily", "bucket": "day"}
    result = compare_datasets(baseline, candidate, levels=[hourly, daily])
    [record] = _check(result, "temporal", "buckets").affected_records
    assert record["bucket"].timestamp() == datetime(2026, 11, 1, 6, tzinfo=UTC).timestamp()
    assert result.level("temporal").baseline_buckets == 25
    [count] = _check(result, "daily", "count").affected_records
    assert (count["bucket"], count["baseline"], count["candidate"]) == (date(2026, 11, 1), 25, 24)


def test_dates_and_datetimes_bucket_alike_and_mismatches_are_not_assessed():
    days = [date(2025, 5, 1) + timedelta(days=k) for k in range(5)]
    dates = pl.DataFrame({"t": days, "x": [1.0] * 5})
    stamps = pl.DataFrame(
        {"t": [datetime(d.year, d.month, d.day, 12) for d in days], "x": [1.0] * 5}
    )
    level = {"level": "temporal", "time_column": "t", "bucket": "day", "measures": ["sum(x)"]}
    assert compare_datasets(dates, stamps, levels=[level]).passed
    numeric = pl.DataFrame({"t": [2025] * 5, "x": [1.0] * 5})
    bad = compare_datasets(dates, numeric, levels=[level])
    [check] = bad.level("temporal").checks
    assert not check.assessed and "numeric" in check.message
    unbucketed = {**level, "bucket": None}
    mixed = compare_datasets(dates, numeric, levels=[unbucketed])
    [check] = mixed.level("temporal").checks
    assert not check.assessed and "different types" in check.message


def test_unparseable_missing_or_ambiguous_times_leave_the_level_not_assessed():
    text = pl.DataFrame({"t": ["2025-01-01", "2025-01-02", "oops"], "x": [1, 2, 3]})
    level = {
        "level": "temporal",
        "time_column": "t",
        "bucket": "day",
        "time_format": "%Y-%m-%d",
    }
    result = compare_datasets(text, text.head(2), levels=[level])
    [check] = result.level("temporal").checks
    assert not check.assessed and "cannot be placed" in check.message
    assert not result.passed
    naive = pl.DataFrame({"t": [datetime(2026, 11, 1, 1, 30), None], "x": [1, 2]})
    zoned = {**level, "time_format": None, "bucket": "hour", "time_zone": "America/New_York"}
    [check] = compare_datasets(naive, naive, levels=[zoned]).level("temporal").checks
    assert not check.assessed
    assert "without a 't' value" in check.message and "cannot be placed" in check.message


def test_cyclic_profiles_are_another_bucket_choice():
    hours = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(24 * 14)]
    baseline = pl.DataFrame({"t": hours, "x": [float(h.hour) for h in hours]})
    candidate = baseline.with_columns(
        pl.when(pl.col("t").dt.hour() == 17).then(pl.col("x") * 2).otherwise(pl.col("x"))
    )
    level = {
        "level": "temporal",
        "time_column": "t",
        "bucket": "hour_of_day",
        "measures": ["mean(x)"],
    }
    result = compare_datasets(baseline, candidate, levels=[level])
    [record] = _check(result, "temporal", "mean(x)").affected_records
    assert (record["bucket"], record["baseline"], record["candidate"]) == (17, 17, 34)
    assert _check(result, "temporal", "gaps").passed  # all 24 hours on both sides


def test_gaps_use_a_configured_window():
    baseline = _daily(date(2025, 1, 5), 5, sites=("a",))
    level = {
        "level": "temporal",
        "time_column": "t",
        "bucket": "day",
        "start": date(2025, 1, 1),
        "end": date(2025, 1, 12),
    }
    result = compare_datasets(baseline, baseline, levels=[level])
    gaps = _check(result, "temporal", "gaps")
    assert gaps.affected_count == 2 * 7  # days 1-4 and 10-12 missing on each side
    assert {r["bucket"] for r in gaps.affected_records} >= {date(2025, 1, 1), date(2025, 1, 12)}
    allowed = compare_datasets(baseline, baseline, levels=[{**level, "allow_gaps": True}])
    assert allowed.passed


# ------------------------------------------------------- calendar shifts
def _year(year: int, bump: date | None = None) -> pl.DataFrame:
    days = [
        date(year, 1, 1) + timedelta(days=k)
        for k in range((date(year + 1, 1, 1) - date(year, 1, 1)).days)
    ]
    values = [float(d.month * 100 + d.day) + (50 if d == bump else 0) for d in days]
    return pl.DataFrame({"t": days, "x": values})


def test_year_over_year_compares_the_candidate_a_year_earlier():
    v1 = _year(2023)
    v2 = _year(2024, bump=date(2024, 6, 15)).filter(pl.col("t") != date(2024, 2, 29))
    level = {"level": "temporal", "time_column": "t", "bucket": "day", "measures": ["sum(x)"]}
    shift = {"step": "shift_time", "column": "t", "offset": {"years": 1}}
    result = compare_datasets(v1, v2, steps=[shift], levels=[level])
    [record] = _check(result, "temporal", "sum(x)").affected_records
    assert (record["bucket"], record["diff"]) == (date(2023, 6, 15), 50)
    assert _check(result, "temporal", "buckets").passed
    # 29 February 2024 has no counterpart a year earlier: refused unless dropped.
    leap = _year(2024)
    refused = compare_datasets(v1, leap, steps=[shift], levels=[level])
    assert not refused.assessed and "no exact counterpart" in refused.error
    dropped = compare_datasets(v1, leap, steps=[{**shift, "unaligned": "drop"}], levels=[level])
    assert dropped.candidate_rows == 365
    assert any("dropped 1 row" in note for note in dropped.alignment)


def test_same_weekday_alignment_compares_mondays_with_mondays():
    weekly = lambda d: float(d.isoweekday())  # noqa: E731
    v1 = pl.DataFrame({"t": [date(2024, 3, 4) + timedelta(days=k) for k in range(28)]})
    v1 = v1.with_columns(pl.col("t").map_elements(weekly, return_dtype=pl.Float64).alias("x"))
    v2 = v1.with_columns(pl.col("t") + timedelta(days=364))
    level = {"level": "temporal", "time_column": "t", "bucket": "day", "measures": ["sum(x)"]}
    same = {"step": "shift_time", "column": "t", "offset": {"years": 1, "same_weekday": True}}
    assert compare_datasets(v1, v2, steps=[same], levels=[level]).passed
    calendar = {**same, "offset": {"years": 1}}
    shifted = compare_datasets(v1, v2, steps=[calendar], levels=[level])
    assert not _check(shifted, "temporal", "sum(x)").passed  # weekdays no longer line up


def test_hourly_shift_aligns_a_lagging_candidate():
    hours = [datetime(2025, 7, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(48)]
    v1 = pl.DataFrame({"t": hours, "x": [float(h % 24) for h in range(48)]})
    v2 = v1.with_columns(pl.col("t") + timedelta(hours=1))
    level = {"level": "temporal", "time_column": "t", "bucket": "hour", "measures": ["sum(x)"]}
    shift = {"step": "shift_time", "column": "t", "offset": {"hours": 1}}
    assert compare_datasets(v1, v2, steps=[shift], levels=[level]).passed
    assert not compare_datasets(v1, v2, levels=[level]).passed


# ------------------------------------------------------------ series shape
def _series(values, group="g1", start=date(2025, 1, 1)):
    days = [start + timedelta(days=k) for k in range(len(values))]
    return pl.DataFrame({"g": [group] * len(values), "t": days, "x": [float(v) for v in values]})


def _shape(baseline, candidate, **options):
    level = {
        "level": "series_shape",
        "time_column": "t",
        "bucket": "day",
        "group_by": ["g"],
        "measure": "sum(x)",
        **options,
    }
    return compare_datasets(baseline, candidate, levels=[level])


SIGNAL = [3, 8, 1, 9, 4, 7, 2, 6, 5, 10, 0, 4, 8, 3, 9, 1]


def test_identical_series_agree_on_every_shape_check():
    result = _shape(_series(SIGNAL), _series(SIGNAL))
    assert result.passed, result.summary()
    [series] = result.level("series_shape").series
    assert series.pearson == 1.0 and series.spearman == 1.0 and series.lag == 0


def test_a_known_lag_is_found():
    lagged = [SIGNAL[0], *SIGNAL[:-1]]  # candidate[t] = baseline[t - 1]
    result = _shape(_series(SIGNAL), _series(lagged))
    lag = _check(result, "series_shape", "lag")
    assert not lag.passed and lag.affected_records[0]["lag"] == 1
    assert lag.affected_records[0]["correlation"] == pytest.approx(1.0)
    allowed = _shape(_series(SIGNAL), _series(lagged), lag={"allowed": 1})
    assert _check(allowed, "series_shape", "lag").passed


def test_a_known_slope_change_is_exact():
    result = _shape(_series([2 * k for k in range(10)]), _series([3 * k + 1 for k in range(10)]))
    trend = _check(result, "series_shape", "trend")
    [record] = trend.affected_records
    assert (record["slope_baseline"], record["slope_candidate"], record["change"]) == (2, 3, 1)
    assert _check(result, "series_shape", "correlation").passed  # same shape, steeper
    gentle = _shape(_series([2 * k for k in range(10)]), _series([2 * k for k in range(10)]))
    assert _check(gentle, "series_shape", "trend").passed


def test_a_known_peak_shift_and_magnitude():
    base = [1, 2, 3, 4, 5, 9, 4, 3, 2, 1, 1, 1]
    later = [1, 2, 3, 4, 5, 4, 3, 2, 9, 1, 1, 1]
    result = _shape(_series(base), _series(later))
    [record] = _check(result, "series_shape", "peak").affected_records
    assert (record["baseline_peak"], record["candidate_peak"], record["shift"]) == (
        date(2025, 1, 6),
        date(2025, 1, 9),
        3,
    )
    taller = _shape(_series(base), _series([*base[:5], 12, *base[6:]]))
    [record] = _check(taller, "series_shape", "peak").affected_records
    assert (record["shift"], record["baseline_value"], record["candidate_value"]) == (0, 9, 12)


def test_a_level_shift_in_one_version_only():
    flat = [10, 11, 10, 9, 10, 10, 11, 10, 9, 10, 11, 10]
    stepped = [*flat[:6], *(v + 40 for v in flat[6:])]
    result = _shape(_series(flat), _series(stepped), correlation=None, trend=None, peak=None)
    [record] = _check(result, "series_shape", "level shift").affected_records
    assert (record["baseline_shift"], record["candidate_shift"]) == (None, date(2025, 1, 7))
    both = _shape(_series(stepped), _series(stepped))
    assert _check(both, "series_shape", "level shift").passed


def test_correlation_thresholds_and_flat_series():
    inverse = _shape(_series(SIGNAL), _series([-v for v in SIGNAL]))
    record = _check(inverse, "series_shape", "correlation").affected_records[0]
    assert record["pearson"] == pytest.approx(-1.0) and record["spearman"] == pytest.approx(-1.0)
    loose = _shape(
        _series(SIGNAL),
        _series([-v for v in SIGNAL]),
        correlation={"min_pearson": -1, "min_spearman": None},
    )
    assert _check(loose, "series_shape", "correlation").passed
    flat = _shape(_series([5] * 10), _series([5] * 10))
    assert flat.passed
    one_flat = _shape(_series([5] * 10), _series(SIGNAL[:10]))
    assert not _check(one_flat, "series_shape", "correlation").passed


def test_profile_difference_by_day_of_week():
    weeks = 4
    base = [10 if k % 7 < 5 else 2 for k in range(7 * weeks)]  # weekdays busy
    flipped = [2 if k % 7 < 5 else 10 for k in range(7 * weeks)]
    profile = {"profile": {"profile": "day_of_week", "max_distance": 0.05}}
    same = _shape(_series(base), _series([2 * v for v in base]), **profile)
    assert _check(same, "series_shape", "profile").passed  # proportional profile
    result = _shape(_series(base), _series(flipped), **profile)
    [record] = _check(result, "series_shape", "profile").affected_records
    # Oracle: shares 10/54 x5 and 2/54 x2 against 2/30 x5 and 10/30 x2.
    expected = (5 * abs(10 / 54 - 2 / 30) + 2 * abs(2 / 54 - 10 / 30)) / 2
    assert record["distance"] == pytest.approx(expected)


def test_short_and_one_sided_series_are_never_passed():
    short = _shape(_series(SIGNAL[:5]), _series(SIGNAL[:5]))
    too_short = _check(short, "series_shape", "too short")
    assert not too_short.assessed and too_short.affected_count == 1
    assert all(not c.assessed for c in short.level("series_shape").checks if c.check != "series")
    assert not short.passed
    # A group whose measure has no value on one side is too short, not skipped.
    missing = _series(SIGNAL).with_columns(pl.lit(None, dtype=pl.Float64).alias("x"))
    empty = _shape(_series(SIGNAL), missing)
    assert _check(empty, "series_shape", "too short").affected_count == 1
    assert not empty.passed
    extra = pl.concat([_series(SIGNAL), _series(SIGNAL, group="g2")])
    one_sided = _shape(_series(SIGNAL), extra)
    series = _check(one_sided, "series_shape", "series")
    assert not series.passed and series.affected_records == [
        {"g": "g2", "series": "only in candidate"}
    ]


def test_shape_lag_on_hourly_buckets_catches_a_one_hour_shift():
    hours = [datetime(2025, 3, 1, tzinfo=UTC) + timedelta(hours=h) for h in range(72)]
    load = [float((h % 24 - 12) ** 2) for h in range(72)]
    v1 = pl.DataFrame({"t": hours, "x": load})
    v2 = v1.with_columns(pl.col("t") + timedelta(hours=1))
    level = {
        "level": "series_shape",
        "time_column": "t",
        "bucket": "hour",
        "measure": "sum(x)",
        "trend": None,
        "level_shift": None,
    }
    result = compare_datasets(v1, v2, levels=[level])
    [record] = _check(result, "series_shape", "lag").affected_records
    assert record["lag"] == 1 and record["correlation"] == pytest.approx(1.0)


@pytest.mark.parametrize("anchor", [datetime(2026, 11, 1, 4), datetime(2026, 3, 8, 5)])
def test_shape_lag_and_peak_count_physical_hours_across_dst(anchor):
    """Through a 25- or 23-hour day, a one-hour shift is lag 1 and the peak moves one
    bucket: sequence positions count physical buckets, not wall-clock slots."""
    start = anchor.replace(tzinfo=UTC) - timedelta(hours=12)
    hours = [start + timedelta(hours=h) for h in range(36)]
    load = [float((h * 7) % 13) for h in range(36)]
    load[18] = 50.0  # the peak, near the change
    v1 = pl.DataFrame({"t": hours, "x": load}).with_columns(
        pl.col("t").dt.convert_time_zone("America/New_York")
    )
    v2 = v1.with_columns(pl.col("t") + timedelta(hours=1))
    level = {
        "level": "series_shape",
        "time_column": "t",
        "bucket": "hour",
        "measure": "sum(x)",
        "trend": None,
        "level_shift": None,
        "correlation": None,
    }
    result = compare_datasets(v1, v2, levels=[level])
    [lag] = _check(result, "series_shape", "lag").affected_records
    assert lag["lag"] == 1 and lag["correlation"] == pytest.approx(1.0)
    [peak] = _check(result, "series_shape", "peak").affected_records
    assert peak["shift"] == 1
    assert peak["candidate_peak"].timestamp() - peak["baseline_peak"].timestamp() == 3600


# --------------------------------------------------- results and CMP-001
def test_results_round_trip_through_json_with_their_types():
    start = datetime(2026, 11, 1, 4, tzinfo=UTC)
    hours = [start + timedelta(hours=h) for h in range(30)]
    frame = pl.DataFrame({"t": hours, "x": [float(h % 5) for h in range(30)]}).with_columns(
        pl.col("t").dt.convert_time_zone("America/New_York")
    )
    levels = [
        {"level": "temporal", "time_column": "t", "bucket": "hour"},
        {"level": "series_shape", "time_column": "t", "bucket": "hour", "measure": "sum(x)"},
    ]
    result = compare_datasets(frame, frame.with_columns(pl.col("x") + 1), levels=levels)
    back = DatasetComparison.model_validate_json(result.model_dump_json())
    assert back.model_dump() == result.model_dump()
    assert type(back.level("temporal")).__name__ == "TemporalResult"
    assert type(back.level("series_shape")).__name__ == "ShapeResult"
    peaks = back.level("series_shape").series[0].peaks
    assert (
        peaks["baseline"].timestamp()
        == result.level("series_shape").series[0].peaks["baseline"].timestamp()
    )


@pytest.mark.parametrize(
    ("level", "message"),
    [
        ({"level": "temporal", "time_column": "t", "bucket": "7h"}, "divide a day"),
        ({"level": "temporal", "time_column": "t", "time_zone": "Nowhere/City"}, "time zone"),
        ({"level": "temporal"}, "time_column"),
        (
            {"level": "series_shape", "time_column": "t", "profile": {"profile": "day"}},
            "cyclic",
        ),
        (
            {"level": "series_shape", "time_column": "t", "profile": {"profile": "hour_of_day"}},
            "calendar 'bucket'",
        ),
    ],
)
def test_configuration_errors_fail_before_reading_data(level, message):
    frame = pl.DataFrame({"t": [date(2025, 1, 1)]})
    with pytest.raises(ValueError, match=message):
        compare_datasets(frame, frame, levels=[level])


def test_cmp001_runs_temporal_levels_from_configuration():
    v1 = _daily(date(2025, 1, 1), 14)
    v2 = v1.with_columns(pl.col("x") * 2)
    config = {
        "datasets": {"v1": {}, "v2": {}},
        "rules": {
            "CMP-001": {
                "params": {
                    "comparisons": [
                        {
                            "name": "daily",
                            "baseline": "v1",
                            "candidate": "v2",
                            "levels": [
                                {
                                    "level": "temporal",
                                    "time_column": "t",
                                    "bucket": "day",
                                    "group_by": ["site"],
                                    "measures": ["sum(x)"],
                                },
                                {
                                    "level": "series_shape",
                                    "severity": "WARNING",
                                    "time_column": "t",
                                    "bucket": "day",
                                    "group_by": ["site"],
                                    "measure": "sum(x)",
                                },
                            ],
                        }
                    ]
                }
            }
        },
    }
    report = validate({"v1": v1, "v2": v2}, config=config)
    results = {r.check: r for r in report.results_for("CMP-001")}
    assert results["daily: temporal: sum(x)"].status is Status.FAIL
    assert results["daily: series_shape: correlation"].status is Status.PASS
    assert results["daily: series_shape: trend"].status is Status.FAIL
    assert results["daily: series_shape: trend"].severity.name == "WARNING"


def test_isolated_comparison_matches_in_process(tmp_path):
    v1 = _year(2023)
    v2 = _year(2024, bump=date(2024, 6, 15))
    v1.write_parquet(tmp_path / "v1.parquet")
    v2.write_parquet(tmp_path / "v2.parquet")
    options = {
        "steps": [
            {"step": "shift_time", "column": "t", "offset": {"years": 1}, "unaligned": "drop"}
        ],
        "levels": [
            {"level": "temporal", "time_column": "t", "bucket": "month", "measures": ["sum(x)"]},
            {"level": "series_shape", "time_column": "t", "bucket": "week", "measure": "sum(x)"},
        ],
    }
    local = compare_datasets(tmp_path / "v1.parquet", tmp_path / "v2.parquet", **options)
    assert local.level("temporal").measures[0].material == 1
    if sys.platform not in SUPPORTED_PLATFORMS:
        return
    isolated = compare_datasets(
        File(tmp_path / "v1.parquet"),
        File(tmp_path / "v2.parquet"),
        resources=ResourceLimits(isolate=True, wall_time=120),
        **options,
    )
    assert isolated.execution_limits["execution_mode"] == "isolated"
    assert isolated.model_dump(exclude={"execution_limits"}) == local.model_dump(
        exclude={"execution_limits"}
    )
