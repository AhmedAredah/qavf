"""Capacity cases prove public-path execution and reject plausible wrong counts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.performance import capacity, streaming
from scripts.performance.models import WorkloadOptions


def test_distinct_file_regression_exercises_native_stream_and_rejects_false_counts(tmp_path):
    fixture = tmp_path / "fixture"
    streaming.prepare_fixture(fixture, rows=303, seed=1, chunk_rows=47)
    assert (fixture / "baseline.parquet").read_bytes() != (
        fixture / "candidate.parquet"
    ).read_bytes()
    workload = streaming.build_workload(
        fixture, tmp_path / "operation", WorkloadOptions(backend="file", evidence_rows=30)
    )
    result = workload.run()
    observed = workload.verify(result)
    assert observed == {
        "verified": True,
        "rows_added": 3,
        "rows_removed": 3,
        "rows_changed": 6,
        "values_changed": 6,
        "execution": "native_stream",
    }
    incorrect = deepcopy(result)
    incorrect["values_changed"] = 0
    with pytest.raises(AssertionError, match="values_changed"):
        workload.verify(incorrect)
    incorrect = deepcopy(result)
    incorrect["differences"][0]["key"]["id"] = -1
    with pytest.raises(AssertionError, match="evidence contradicts"):
        workload.verify(incorrect)


@pytest.mark.parametrize("case,expected", [("grouped_statistics", 1), ("scalar_statistics", 5)])
def test_capacity_oracles_and_sample_values(tmp_path, case, expected):
    fixture = tmp_path / "fixture"
    capacity.prepare_fixture(fixture, rows=411, seed=2, chunk_rows=43)
    workload = capacity.build_workload(case, fixture, tmp_path / "operation", WorkloadOptions())
    report = workload.run()
    assert workload.verify(report)["affected_count"] == expected
    bad = report.model_copy(
        update={"results": [report.results[0].model_copy(update={"affected_count": 0})]}
    )
    with pytest.raises(AssertionError, match="oracle mismatch"):
        workload.verify(bad)
