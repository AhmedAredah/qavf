"""Coverage identities must distinguish physical timestamps across DST folds."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from qavf.engine.severity import Status
from qavf.testing import rule_results


def fold_instants():
    zone = ZoneInfo("America/New_York")
    return [datetime(2026, 11, 1, hour, 30, tzinfo=UTC).astimezone(zone) for hour in [5, 6]]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_fold_periods_retain_two_slices_and_correct_missing_entity(explicit, lazy, order):
    early, late = fold_instants()
    frame = pl.DataFrame({"id": [1, 1, 2], "t": [early, late, late]})[order]
    params = {"entity_column": "id", "time_column": "t", "expected_entities": [1, 2]}
    if explicit:
        params["expected_periods"] = [early, late]
    results = rule_results("SPA-003", {"d": frame.lazy() if lazy else frame}, params)
    assert {r.status for r in results} == {Status.FAIL}
    result = results[0]
    assert result.metrics["slices_checked"] == 2
    assert result.affected_count == 1
    assert result.affected_records[0]["missing_entities"] == [2]
    assert result.affected_records[0]["t"].timestamp() == early.timestamp()


@pytest.mark.parametrize("source", ["explicit", "reference", "union"])
def test_fold_entity_identifiers_are_distinct_and_keep_evidence(source):
    early, late = fold_instants()
    frames = {"d": pl.DataFrame({"id": [early]})}
    params = {"entity_column": "id"}
    if source == "explicit":
        params["expected_entities"] = [early, late]
    else:
        frames["r"] = pl.DataFrame({"id": [early, late]})
        if source == "reference":
            params["reference_dataset"] = "r"
    results = rule_results("SPA-003", frames, params)
    failed = next(r for r in results if r.status is Status.FAIL)
    assert failed.metrics["expected_entities"] == 2
    assert failed.affected_records[0]["missing_entities"][0].timestamp() == late.timestamp()


def test_temporal_and_integer_key_domains_cannot_silently_merge():
    early, _ = fold_instants()
    results = rule_results(
        "SPA-003",
        {
            "a": pl.DataFrame({"id": [early]}),
            "b": pl.DataFrame({"id": [int(early.timestamp() * 1e6)]}),
        },
        {"entity_column": "id"},
    )
    assert any(r.status is Status.ERROR and "cannot be compared" in r.message for r in results)
