"""Stress benchmark oracles must reject plausible incorrect outcomes."""

from copy import deepcopy

import pytest

from scripts.performance import catalog, stress
from scripts.performance.models import WorkloadOptions


@pytest.mark.parametrize("case", ["many_small_files", "huge_cells", "resampling_inference"])
def test_stress_case_oracles(tmp_path, case):
    fixture = tmp_path / "fixture"
    stress.prepare_fixture(fixture, rows=303, seed=-7)
    workload = catalog.build_workload(case, fixture, tmp_path / "operation", WorkloadOptions())
    result = workload.run()
    assert workload.verify(result)["verified"] is True
    if case == "many_small_files":
        bad = result.model_copy(update={"results": []})
    else:
        bad = deepcopy(result)
        if case == "huge_cells":
            bad["levels"][0]["rows_changed"] = 0
        else:
            bad["ci_low"] += 0.1
    with pytest.raises(AssertionError, match="oracle mismatch"):
        workload.verify(bad)
