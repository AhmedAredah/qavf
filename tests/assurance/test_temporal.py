"""Physical-time contracts independent of Python's lossy datetime equality."""

from __future__ import annotations

import operator
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qavf.engine.severity import Status

from .support.adapters import BACKENDS, adapted_frames
from .support.contracts import run_checks

_COMPARISONS = {"ge": operator.ge, "gt": operator.gt, "le": operator.le, "lt": operator.lt}
_MICROSECONDS_PER_TICK = {"ms": Fraction(1000), "us": Fraction(1), "ns": Fraction(1, 1000)}
_POLARS_BACKENDS = ("eager", "lazy", "arrow", "partitioned")
_ZONES = ("UTC", "America/New_York", "Asia/Kathmandu")


def _assert_verdict(results, failures):
    assert sum(result.affected_count for result in results) == failures
    assert any(result.status is Status.FAIL for result in results) == bool(failures)


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    microseconds=st.integers(-(10**12), 10**12),
    unit=st.sampled_from(tuple(_MICROSECONDS_PER_TICK)),
    comparison=st.sampled_from(tuple(_COMPARISONS)),
    partition_size=st.integers(1, 3),
)
def test_datetime_limits_preserve_fractional_and_negative_ticks(
    microseconds, unit, backend, comparison, partition_size
):
    bound = datetime(1970, 1, 1) + timedelta(microseconds=microseconds)
    threshold = Fraction(microseconds) / _MICROSECONDS_PER_TICK[unit]
    lower = threshold.numerator // threshold.denominator
    ticks = [lower - 1, lower, lower + 1]
    expected = sum(not _COMPARISONS[comparison](tick, threshold) for tick in ticks)
    frame = pl.DataFrame({"t": pl.Series(ticks, dtype=pl.Int64).cast(pl.Datetime(unit))})
    with adapted_frames({"d": frame}, backend, partition_size=partition_size) as inputs:
        results = run_checks(
            inputs,
            "INP-003",
            datasets={"d": {"schema": {"columns": {"t": {comparison: bound}}}}},
        )
    _assert_verdict(results, expected)


@pytest.mark.parametrize("backend", _POLARS_BACKENDS)
@given(
    microseconds=st.integers(-(10**12), 10**12),
    source_zone=st.sampled_from(_ZONES),
    bound_zone=st.sampled_from(_ZONES),
    comparison=st.sampled_from(tuple(_COMPARISONS)),
)
def test_aware_bounds_compare_instants_across_timezones(
    microseconds, source_zone, bound_zone, backend, comparison
):
    instant = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=microseconds)
    bound = instant.astimezone(ZoneInfo(bound_zone))
    origin = microseconds * 1000
    ticks = [origin - 1, origin, origin + 1]
    expected = sum(not _COMPARISONS[comparison](tick, origin) for tick in ticks)
    frame = pl.DataFrame({"t": pl.Series(ticks, dtype=pl.Datetime("ns", source_zone))})
    with adapted_frames({"d": frame}, backend) as inputs:
        results = run_checks(
            inputs,
            "INP-003",
            datasets={"d": {"schema": {"columns": {"t": {comparison: bound}}}}},
        )
    _assert_verdict(results, expected)


@given(
    year=st.integers(2020, 2035),
    minute=st.integers(0, 59),
    zone=st.sampled_from(("America/New_York", "America/Chicago")),
    order=st.permutations((0, 1, 2)),
    backend=st.sampled_from(_POLARS_BACKENDS),
    partition_size=st.integers(1, 3),
    explicit=st.booleans(),
)
def test_dst_fold_coverage_preserves_both_physical_instants(
    year, minute, zone, order, backend, partition_size, explicit
):
    november = datetime(year, 11, 1)
    sunday = 1 + (6 - november.weekday()) % 7
    early = datetime(year, 11, sunday, 1, minute, tzinfo=ZoneInfo(zone), fold=0)
    late = early.replace(fold=1)
    # Check the independent UTC identity, never local datetime equality.
    assert late.astimezone(UTC) - early.astimezone(UTC) == timedelta(hours=1)
    frame = pl.DataFrame({"entity": [1, 1, 2], "t": [early, late, late]})[list(order)]
    params = {"entity_column": "entity", "time_column": "t", "expected_entities": [1, 2]}
    if explicit:
        params["expected_periods"] = [early, late]
    with adapted_frames({"d": frame}, backend, partition_size=partition_size) as inputs:
        results = run_checks(inputs, "SPA-003", params=params)
    _assert_verdict(results, 1)
    assert results[0].metrics["slices_checked"] == 2
    assert results[0].affected_records[0]["missing_entities"] == [2]
    assert results[0].affected_records[0]["t"].astimezone(UTC) == early.astimezone(UTC)


@given(
    microseconds=st.integers(-(10**12), 10**12),
    backend=st.sampled_from(BACKENDS),
    zone=st.sampled_from(_ZONES),
)
def test_aware_bound_on_unzoned_data_is_explicitly_unassessed(microseconds, backend, zone):
    naive = datetime(1970, 1, 1) + timedelta(microseconds=microseconds)
    bound = naive.replace(tzinfo=UTC).astimezone(ZoneInfo(zone))
    with adapted_frames({"d": pl.DataFrame({"t": [naive]})}, backend) as inputs:
        results = run_checks(
            inputs,
            "INP-003",
            datasets={"d": {"schema": {"columns": {"t": {"le": bound}}}}},
            assessed_only=False,
        )
    blocked = [result for result in results if result.check and ": le " in result.check]
    assert blocked and all(result.status is Status.ERROR for result in blocked)
    assert all("time zone" in result.message for result in blocked)


@pytest.mark.parametrize("bound", [datetime(2026, 3, 8, 2, 30), datetime(2026, 11, 1, 1, 30)])
@pytest.mark.parametrize("backend", _POLARS_BACKENDS)
def test_nonexistent_or_ambiguous_local_bound_is_not_guessed(bound, backend):
    frame = pl.DataFrame({"t": pl.Series([0], dtype=pl.Datetime("us", "America/New_York"))})
    with adapted_frames({"d": frame}, backend) as inputs:
        results = run_checks(
            inputs,
            "INP-003",
            datasets={"d": {"schema": {"columns": {"t": {"le": bound}}}}},
            assessed_only=False,
        )
    blocked = [result for result in results if result.check and ": le " in result.check]
    assert blocked and all(result.status is Status.ERROR for result in blocked)
    assert all(
        any(word in result.message.lower() for word in ("ambiguous", "non-existent"))
        for result in blocked
    )


@given(
    origin=st.integers(-(10**12), 10**12),
    remainder=st.integers(1, 999),
    backend=st.sampled_from(_POLARS_BACKENDS),
)
def test_coverage_rejects_unsupported_submicrosecond_identity(origin, remainder, backend):
    ticks = [origin * 1000, origin * 1000 + remainder]
    frame = pl.DataFrame({"entity": [1, 2], "t": pl.Series(ticks, dtype=pl.Datetime("ns"))})
    with adapted_frames({"d": frame}, backend) as inputs:
        results = run_checks(
            inputs,
            "SPA-003",
            params={"entity_column": "entity", "time_column": "t"},
            assessed_only=False,
        )
    assert {result.status for result in results} == {Status.ERROR}
    assert any("microsecond precision" in result.message for result in results)


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    microseconds=st.integers(-(10**12), 10**12),
    offset=st.sampled_from([-1_000_000_001, -1_000_000_000, -999_999_999, 0, 1]),
)
def test_freshness_keeps_nanoseconds_at_age_and_future_boundaries(microseconds, offset, backend):
    now = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=microseconds)
    frame = pl.DataFrame({"t": pl.Series([microseconds * 1000 + offset], dtype=pl.Datetime("ns"))})
    expected = int(offset < -1_000_000_000 or offset > 0)
    with adapted_frames({"d": frame}, backend) as inputs:
        results = run_checks(
            inputs,
            "INP-008",
            params={"now": now, "checks": [{"dataset": "d", "column": "t", "max_age": "PT1S"}]},
        )
    _assert_verdict(results, expected)
