"""Public GEO-001 admission controls preserve independent coordinate checks."""

from __future__ import annotations

import polars as pl
import pytest

from qavf import Status, validate


def _within(limits):
    points = pl.DataFrame({"latitude": [0.5, 0.5, 0.75, 3.0], "longitude": [0.5, 0.5, 0.75, 3.0]})
    zones = pl.DataFrame({"zone": ["study"], "geometry": ["POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"]})
    report = validate(
        {"points": points, "zones": zones},
        config={
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "datasets": {"points": {}, "zones": {}},
            "rules": {
                "GEO-001": {
                    "params": {
                        "points": [
                            {
                                "dataset": "points",
                                "within": {"dataset": "zones"},
                                "assignment_limits": limits,
                            }
                        ]
                    }
                }
            },
        },
    )
    return {result.check: result for result in report.results_for("GEO-001")}


@pytest.mark.parametrize("limit", ["max_points", "max_matches", "max_work"])
def test_within_limit_reports_unassessed_without_hiding_valid_coordinates(limit):
    results = _within({limit: 1, "batch_size": 1})
    coordinates = results[None]
    within = results["points(latitude, longitude) within zones"]
    assert coordinates.status is Status.PASS
    assert coordinates.metrics["passed_checks"] == ["points(latitude, longitude)"]
    assert within.status is Status.ERROR
    assert limit in within.message


@pytest.mark.parametrize("batch_size", [1, 2, 16_384])
def test_within_override_keeps_complete_row_counts_and_exact_evidence(batch_size):
    results = _within({"max_points": 3, "max_matches": 2, "max_work": 2, "batch_size": batch_size})
    within = results["points(latitude, longitude) within zones"]
    assert results[None].status is Status.PASS
    assert results[None].metrics["passed_checks"] == ["points(latitude, longitude)"]
    assert within.status is Status.FAIL
    assert (within.affected_count, within.total_records) == (1, 4)
    assert within.affected_records == [{"latitude": 3.0, "longitude": 3.0}]
