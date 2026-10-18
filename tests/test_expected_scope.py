"""Expected scope, freshness and statistical rule options.

Expected group combinations for TMP-004, strict time columns and unexpected entities
for SPA-003, the INP-008 freshness rule, explicit normaliser join keys, and STA-001
direction, materiality and log transform.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.rules.spatial_checks import SpatialMetric
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


def _run(frames, rules):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False, "parallel": False},
            "datasets": {name: {} for name in frames},
            "rules": rules,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


# ------------------------------------------------------------------ TMP-004 groups
SERIES = pl.DataFrame({"county": [1, 1, 2, 2], "year": [2020, 2021, 2020, 2021], "v": [1.0] * 4})
SPEC = {"dataset": "s", "value": "v", "group_by": ["county"]}


def test_expected_groups_find_a_group_missing_altogether() -> None:
    params = {"series": [SPEC], "expected_groups": [{"county": 1}, {"county": 2}, {"county": 3}]}
    result = _run({"s": SERIES}, {"TMP-004": {"params": params}}).get("TMP-004")
    assert result.status is Status.FAIL and result.total_records == 3
    [record] = result.affected_records
    assert record["county"] == 3 and record["missing_group"] is True
    assert record["missing_periods"] == [2020, 2021]


def test_groups_can_come_from_a_reference_dataset() -> None:
    inventory = pl.DataFrame({"county": [1, 2, 4]})
    params = {"series": [SPEC], "groups_from": "inv"}
    result = _run({"s": SERIES, "inv": inventory}, {"TMP-004": {"params": params}}).get("TMP-004")
    assert result.status is Status.FAIL and [r["county"] for r in result.affected_records] == [4]


@pytest.mark.parametrize(
    "groups",
    [[{"state": 1}], [{"county": "not a number"}]],
)
def test_unusable_expected_groups_are_unassessed(groups) -> None:
    params = {"series": [SPEC], "expected_groups": groups}
    report = _run({"s": SERIES}, {"TMP-004": {"params": params}})
    assert any(
        r.status is Status.ERROR and "expected groups" in (r.check or "") for r in report.results
    )


# ------------------------------------------------------------------ SPA-003
DATA = pl.DataFrame({"site": ["a", "b", "c"], "year": [2020, 2020, 2020]})


def test_unexpected_entities_are_reported_when_asked() -> None:
    params = {"entity_column": "site", "expected_entities": ["a", "b"], "datasets": ["d"]}
    lenient = _run({"d": DATA}, {"SPA-003": {"params": params}}).get("SPA-003")
    assert lenient.status is Status.PASS
    strict = _run({"d": DATA}, {"SPA-003": {"params": {**params, "unexpected_entities": True}}})
    result = strict.get("SPA-003")
    assert result.status is Status.FAIL and result.affected_records[0]["unexpected_entities"] == [
        "c"
    ]


def test_a_required_time_column_cannot_silently_disappear() -> None:
    params = {"entity_column": "site", "time_column": "year", "datasets": ["d"]}
    timeless = DATA.drop("year")
    assert (
        _run({"d": timeless}, {"SPA-003": {"params": params}}).get("SPA-003").status is Status.PASS
    )
    report = _run({"d": timeless}, {"SPA-003": {"params": {**params, "require_time_column": True}}})
    [result] = report.results_for("SPA-003")
    assert result.status is Status.ERROR and "lacks the time column" in result.message


# ------------------------------------------------------------------ INP-008
STAMPS = pl.DataFrame(
    {
        "at": [datetime(2026, 1, 1), datetime(2026, 1, 9, 12)],
        "day": [date(2026, 1, 1), date(2026, 1, 9)],
        "n": [1, 2],
    }
)
NOW = datetime(2026, 1, 10, tzinfo=UTC)


@pytest.mark.parametrize(
    ("column", "max_age", "status"),
    [
        ("at", "P1D", Status.PASS),
        ("at", "PT6H", Status.FAIL),
        ("day", "P1D", Status.PASS),
        ("day", "PT12H", Status.FAIL),
    ],
)
def test_freshness(column, max_age, status) -> None:
    params = {"checks": [{"dataset": "t", "column": column, "max_age": max_age}], "now": NOW}
    result = _run({"t": STAMPS}, {"INP-008": {"params": params}}).get("INP-008")
    assert result.status is status
    if status is Status.FAIL:
        assert result.affected_records[0]["age_hours"] > 6


def test_freshness_needs_a_temporal_column_with_values() -> None:
    params = {"checks": [{"dataset": "t", "column": "n", "max_age": "P1D"}], "now": NOW}
    [result] = _run({"t": STAMPS}, {"INP-008": {"params": params}}).results_for("INP-008")
    assert result.status is Status.ERROR and "date or datetime" in result.message
    empty = STAMPS.clear()
    params = {"checks": [{"dataset": "t", "column": "at", "max_age": "P1D"}], "now": NOW}
    [result] = _run({"t": empty}, {"INP-008": {"params": params}}).results_for("INP-008")
    assert result.status is Status.ERROR and "no values" in result.message


def test_freshness_compares_time_zone_aware_values_as_instants() -> None:
    aware = STAMPS.with_columns(
        pl.col("at").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Chicago")
    )
    params = {"checks": [{"dataset": "t", "column": "at", "max_age": "PT13H"}], "now": NOW}
    assert _run({"t": aware}, {"INP-008": {"params": params}}).get("INP-008").status is Status.PASS


# ------------------------------------------------------------------ normaliser keys
def test_normaliser_join_keys_are_explicit_and_validated() -> None:
    with pytest.raises(ValidationError, match=r"normalize_by.on"):
        SpatialMetric.model_validate(
            {
                "dataset": "m",
                "value": "v",
                "entity_column": "e",
                "normalize_by": {"dataset": "n", "value": "w", "on": ["year"]},
            }
        )
    metrics = pl.DataFrame({"e": list("abcde"), "year": [2020] * 5, "v": [10.0, 11, 9, 10, 10]})
    # The denominator also has 'year' but at another grain: join on the entity only.
    denominators = pl.DataFrame({"e": list("abcde"), "year": [1999] * 5, "w": [1.0] * 5})
    metric = {
        "dataset": "m",
        "value": "v",
        "entity_column": "e",
        "group_by": ["year"],
        "normalize_by": {"dataset": "n", "value": "w", "on": ["e"]},
    }
    report = _run({"m": metrics, "n": denominators}, {"SPA-002": {"params": {"metrics": [metric]}}})
    assert report.get("SPA-002").status is Status.PASS and report.get("SPA-002").total_records == 5


# ------------------------------------------------------------------ STA-001 options
VALUES = pl.DataFrame({"v": [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 30.0, 1.0]})


def _sta(frame, **target):
    params = {
        "targets": [{"dataset": "t", "value": "v", **target}],
        "method": "iqr",
        "min_group_size": 5,
    }
    return _run({"t": frame}, {"STA-001": {"params": params}})


def test_direction_keeps_only_high_or_low_outliers() -> None:
    both = _sta(VALUES).get("STA-001")
    assert sorted(r["v"] for r in both.affected_records) == [1.0, 30.0]
    high = _sta(VALUES, direction="high").get("STA-001")
    assert [r["v"] for r in high.affected_records] == [30.0]
    low = _sta(VALUES, direction="low").get("STA-001")
    assert [r["v"] for r in low.affected_records] == [1.0]


def test_materiality_filters_immaterial_outliers() -> None:
    tight = pl.DataFrame({"v": [100.0, 100.01, 99.99, 100.0, 100.02, 99.98, 100.0, 101.0]})
    assert _sta(tight).get("STA-001").status is Status.FAIL
    result = _sta(tight, min_relative_deviation=0.05).get("STA-001")
    assert result.status is Status.PASS


def test_log_transform_and_non_positive_values() -> None:
    skewed = pl.DataFrame({"v": [1.0, 2.0, 4.0, 8.0, 16.0, 2.0, 4.0, 1.0, 0.0, -1.0]})
    report = _sta(skewed, transform="log")
    blocked = [r for r in report.results if r.status is Status.ERROR]
    assert blocked and "cannot be log-transformed" in blocked[0].message
    assert blocked[0].affected_count == 2
