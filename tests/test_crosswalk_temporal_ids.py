"""Crosswalk ids keep native identity: nanosecond and DST-fold instants stay distinct."""

from datetime import UTC, datetime

import polars as pl
import pytest

from qavf.spatial.hierarchy import SOURCE, TARGET, WEIGHT, aggregate_up, compose, validate_crosswalk


def _nanoseconds(*ticks: int, tz: str | None = None) -> pl.Series:
    return pl.Series(SOURCE, list(ticks), dtype=pl.Int64).cast(pl.Datetime("ns", tz))


def _crosswalk(ids: pl.Series, targets, weights) -> pl.DataFrame:
    return pl.DataFrame({SOURCE: ids, TARGET: targets, WEIGHT: weights})


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_nanosecond_ids_one_tick_apart_are_separate_zones(tz):
    frame = _crosswalk(_nanoseconds(1, 2, tz=tz), ["A", "B"], [1.0, 1.0])
    assert validate_crosswalk(frame, 1e-9, "t") is frame


def test_an_off_nanosecond_zone_is_named_precisely():
    frame = _crosswalk(_nanoseconds(1, 2), ["A", "B"], [1.0, 0.5])
    with pytest.raises(ValueError, match=r"1 zone do not sum to 1.*000000002"):
        validate_crosswalk(frame, 1e-9, "t")


def test_dst_fold_instants_stay_distinct():
    # 01:30 local time occurs twice on 2026-11-01 in Chicago: 06:30 and 07:30 UTC.
    first = datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    second = datetime(2026, 11, 1, 7, 30, tzinfo=UTC)
    ids = pl.Series(SOURCE, [first, second]).dt.convert_time_zone("America/Chicago")
    frame = _crosswalk(ids.cast(pl.Datetime("ns", "America/Chicago")), ["A", "B"], [1.0, 1.0])
    assert validate_crosswalk(frame, 1e-9, "t") is frame


def test_compose_keeps_nanosecond_parents_and_children_distinct():
    children = _crosswalk(_nanoseconds(1, 2), [10, 20], [1.0, 1.0])
    parents = pl.DataFrame(
        {SOURCE: [10, 20], TARGET: _nanoseconds(5, 6).alias(TARGET), WEIGHT: [1.0, 1.0]}
    )
    chained = compose(children, parents)
    assert chained.schema[SOURCE] == pl.Datetime("ns")
    assert chained.sort(SOURCE)[SOURCE].cast(pl.Int64).to_list() == [1, 2]
    assert chained.sort(SOURCE)[TARGET].cast(pl.Int64).to_list() == [5, 6]
    assert validate_crosswalk(chained, 1e-9, "t") is chained


def test_aggregate_up_keeps_nanosecond_parents_distinct():
    crosswalk = pl.DataFrame(
        {
            SOURCE: ["a", "a", "b"],
            TARGET: _nanoseconds(1, 2, 2).alias(TARGET),
            WEIGHT: [0.5, 0.5, 1.0],
        }
    )
    frame = pl.DataFrame({"zone": ["a", "b"], "value": [10, 4]})
    totals = aggregate_up(frame, "zone", ["value"], crosswalk)
    assert totals["zone"].dtype == pl.Datetime("ns")
    assert totals["zone"].cast(pl.Int64).to_list() == [1, 2]
    assert totals["value"].to_list() == [5.0, 9.0]
