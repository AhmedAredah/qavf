"""Added benchmark families execute real APIs and detect false outcomes."""

from copy import deepcopy

import pytest

from scripts.performance import breadth, catalog
from scripts.performance.models import WorkloadOptions


@pytest.mark.parametrize(
    "case",
    ["dataset_records", "temporal_buckets", "spatial_locations", "dirty_csv", "report_rendering"],
)
def test_breadth_workloads_have_independent_oracles(tmp_path, case):
    fixture = tmp_path / "fixture"
    breadth.prepare_fixture(fixture, rows=303, seed=1, chunk_rows=47)
    options = WorkloadOptions(input_format="csv" if case == "dirty_csv" else "parquet")
    workload = catalog.build_workload(case, fixture, tmp_path / "operation", options)
    result = workload.run()
    assert workload.verify(result)["verified"] is True
    if case == "dirty_csv":
        results = list(result.results)
        index = next(i for i, r in enumerate(results) if r.check and "unreadable" in r.check)
        results[index] = results[index].model_copy(update={"affected_count": 0})
        bad = result.model_copy(update={"results": results})
    else:
        bad = deepcopy(result)
        if case == "report_rendering":
            bad["results"] = 0
        elif case == "dataset_records":
            bad["levels"][0]["rows_changed"] = 0
        elif case == "temporal_buckets":
            bad["levels"][0]["measures"][0]["material"] = 0
        else:
            check = next(c for c in bad["levels"][0]["checks"] if c["check"] == "moved")
            check["affected_count"] = 0
    with pytest.raises(AssertionError, match="oracle mismatch"):
        workload.verify(bad)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_memory_target_is_positive_integral(value):
    with pytest.raises(ValueError, match="memory_budget_bytes must be a positive integer"):
        WorkloadOptions(memory_budget_bytes=value)


def test_zero_cache_and_spill_limits_remain_valid():
    assert WorkloadOptions(memory_budget_bytes=1, cache_max_bytes=0, spill_max_bytes=0)


def test_cli_rejects_zero_memory_before_starting_workers(capsys):
    from scripts.performance.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["--memory-bytes", "0"])
    assert stopped.value.code == 2
    assert "memory_budget_bytes must be a positive integer" in capsys.readouterr().err
