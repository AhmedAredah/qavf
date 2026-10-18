"""TMP-006 cumulative resets, TMP-007 seasonal anomalies, TMP-008 level shifts."""

from __future__ import annotations

import polars as pl
import pytest

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)
SERIES = [{"dataset": "s", "value": "v", "group_by": ["site"], "time_column": "t"}]


def _run(rule, frame, **params):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {"s": {}},
            "rules": {rule: {"params": {"series": SERIES, **params}}},
        }
    )
    return QARunner(settings, REGISTRY).run(frames={"s": frame}).get(rule)


def _frame(values: list[float], site: str = "a") -> pl.DataFrame:
    floats = [float(v) for v in values]
    return pl.DataFrame({"site": [site] * len(floats), "t": list(range(len(floats))), "v": floats})


# ------------------------------------------------------------------ TMP-006
def test_cumulative_series_must_not_decrease() -> None:
    assert _run("TMP-006", _frame([1, 3, 3, 7])).status is Status.PASS
    result = _run("TMP-006", _frame([1, 3, 2, 7]))
    assert result.status is Status.FAIL
    [record] = result.affected_records
    assert (record["period"], record["previous_value"], record["value"]) == (2, 3.0, 2.0)


def test_resets_to_zero_can_be_allowed() -> None:
    frame = _frame([5, 9, 0, 4])
    assert _run("TMP-006", frame).status is Status.FAIL
    assert _run("TMP-006", frame, allow_reset_to_zero=True).status is Status.PASS
    assert _run("TMP-006", _frame([5, 9, 1, 4]), allow_reset_to_zero=True).status is Status.FAIL


# ------------------------------------------------------------------ TMP-007
def _monthly(spike_at: int | None = None, years: int = 5) -> pl.DataFrame:
    pattern = [10, 12, 15, 20, 30, 40, 45, 42, 30, 20, 14, 11]
    values = [float(pattern[m % 12]) * (1 + 0.01 * (m // 12)) for m in range(12 * years)]
    if spike_at is not None:
        values[spike_at] *= 3
    return _frame(values)


def test_a_value_unlike_its_season_is_flagged() -> None:
    assert _run("TMP-007", _monthly(), season_length=12).status is Status.PASS
    result = _run("TMP-007", _monthly(spike_at=12 * 2 + 1), season_length=12)
    assert result.status is Status.FAIL
    [record] = result.affected_records
    assert record["period"] == 25 and record["season"] == 1


def test_seasonal_patterns_are_not_anomalies() -> None:
    # The summer peak would be an outlier against the whole series, not its season.
    result = _run("TMP-007", _monthly(), season_length=12)
    assert result.status is Status.PASS and result.total_records == 60


def test_too_few_cycles_are_unassessed() -> None:
    result = _run("TMP-007", _monthly(years=2), season_length=12)
    assert result.status is Status.ERROR and result.metrics["too_short"] == 12


# ------------------------------------------------------------------ TMP-008
@pytest.mark.parametrize(
    ("values", "status"),
    [
        ([10, 11, 10, 9, 10, 11, 10, 9, 10, 11], Status.PASS),
        ([10, 11, 10, 9, 10, 20, 21, 20, 19, 20], Status.FAIL),
        ([10, 11, 10, 9, 10, 11.5, 12.5, 11.5, 10.5, 11.5], Status.PASS),  # shift too small
    ],
)
def test_level_shifts(values, status) -> None:
    result = _run("TMP-008", _frame(values))
    assert result.status is status
    if status is Status.FAIL:
        [record] = result.affected_records
        assert record["shift_period"] == 5 and record["median_after"] == 20


def test_level_shift_needs_two_full_segments() -> None:
    result = _run("TMP-008", _frame([1, 2, 3, 4, 5]))
    assert result.status is Status.ERROR and result.metrics["too_short"] == 1
