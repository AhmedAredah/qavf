"""TMP-005: stale or repeated values (stuck runs and copied periods)."""

from __future__ import annotations

import polars as pl
import pytest

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


def _run(frame, **params):
    series = [{"dataset": "s", "value": "v", "group_by": ["site"]}]
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {"s": {}},
            "rules": {"TMP-005": {"params": {"series": series, **params}}},
        }
    )
    return QARunner(settings, REGISTRY).run(frames={"s": frame}).get("TMP-005")


def _series(values_by_site: dict[str, list[float]]) -> pl.DataFrame:
    rows = [
        {"site": site, "year": 2020 + i, "v": value}
        for site, values in values_by_site.items()
        for i, value in enumerate(values)
    ]
    return pl.DataFrame(rows)


def test_varying_series_pass() -> None:
    frame = _series({"a": [1, 2, 3, 4, 5], "b": [5, 4, 6, 3, 7]})
    assert _run(frame).status is Status.PASS


def test_a_stuck_run_is_flagged_with_its_extent() -> None:
    frame = _series({"a": [1, 7, 7, 7, 2], "b": [5, 4, 6, 3, 7]})
    result = _run(frame)
    assert result.status is Status.FAIL
    [record] = result.affected_records
    assert (
        record["site"],
        record["value"],
        record["first_period"],
        record["last_period"],
        record["periods"],
    ) == ("a", 7.0, 2021, 2023, 3)


@pytest.mark.parametrize(("repeats", "status"), [(3, Status.PASS), (2, Status.FAIL)])
def test_max_repeats_sets_the_run_length(repeats, status) -> None:
    frame = _series({"a": [1, 7, 7, 2, 3], "b": [5, 4, 6, 3, 7]})
    assert _run(frame, max_repeats=repeats, copied_periods=False).status is status


def test_near_identical_values_count_with_atol() -> None:
    frame = _series({"a": [7.0, 7.001, 6.999, 1.0], "b": [5, 4, 6, 3]})
    assert _run(frame).status is Status.PASS
    assert _run(frame, atol=0.01).status is Status.FAIL


def test_legitimately_constant_values_can_be_ignored() -> None:
    frame = _series({"a": [0, 0, 0, 0, 5], "b": [5, 4, 6, 3, 7]})
    assert _run(frame).status is Status.FAIL
    assert _run(frame, ignore_values=[0]).status is Status.PASS


def test_a_period_copied_from_the_previous_one_is_flagged() -> None:
    frame = _series({"a": [1, 2, 2, 5], "b": [9, 3, 3, 1], "c": [4, 6, 6, 8]})
    result = _run(frame)
    [record] = [r for r in result.affected_records if r["kind"] == "copied period"]
    assert (record["period"], record["copied_from"], record["groups"]) == (2022, 2021, 3)
    assert _run(frame, copied_periods=False).status is Status.PASS


def test_one_group_repeating_is_not_a_copied_period() -> None:
    frame = _series({"a": [1, 2, 2, 5], "b": [9, 3, 4, 1]})
    assert _run(frame, max_repeats=3).status is Status.PASS


def test_series_too_short_to_test_are_unassessed() -> None:
    frame = _series({"a": [1, 1], "b": [2, 2]})
    result = _run(frame, copied_periods=False)
    assert result.status is Status.ERROR and result.metrics["too_short"] == 2
