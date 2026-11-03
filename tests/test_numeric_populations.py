"""Exact fractions and undefined numeric populations across execution paths."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qavf import validate

CONFIG = {"engine": {"parallel": False, "native_threads": 2}, "plugins": {"entry_points": False}}


def column_results(report):
    return [r for r in report.results if r.rule_id == "INP-003" and r.status.value != "SKIPPED"]


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("forced_eager", [False, True])
def test_missing_fraction_uses_exact_ratio_for_decision(backend, forced_eager):
    check = {"max_null_fraction": 0.3333333333333333}
    if forced_eager:
        # This forces the compatibility eager path; an independent row bound
        # remains valid so only the exact missing fraction supplies the failure.
        check["not_null"] = True
    report = validate(
        pa.table({"x": [None, 1, 2]}),
        backend=backend,
        config=CONFIG,
        schema={"columns": {"x": {"nullable": True, "conditions": [check]}}},
    )
    failures = [r for r in column_results(report) if r.status.value == "FAIL"]
    assert failures and any("fraction" in (r.name + r.message) for r in failures)
    if backend == "polars" and forced_eager:
        assert any("missing fraction " in r.message for r in failures)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("source", ["arrow", "parquet"])
@pytest.mark.parametrize("statistic", ["min", "max", "sum", "mean", "median", "std"])
def test_all_missing_numeric_statistic_is_unassessed(tmp_path, backend, source, statistic):
    data = pa.table({"x": pa.array([None, None], type=pa.float64())})
    if source == "parquet":
        path = tmp_path / "missing.parquet"
        pq.write_table(data, path)
        data = path
    report = validate(
        data,
        backend=backend,
        config=CONFIG,
        schema={
            "columns": {"x": {"dtype": "float", "nullable": True, "stats": {statistic: {"ge": 0}}}},
        },
    )
    results = column_results(report)
    errors = [r for r in results if r.status.value == "ERROR"]
    assert errors and all("undefined" in r.message for r in errors)
    assert not [r for r in results if r.status.value == "FAIL"]
    assert report.incomplete


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_empty_missing_fraction_and_distinct_count_remain_defined(backend):
    report = validate(
        pa.table({"x": ["N/A"]}),
        backend=backend,
        config=CONFIG,
        schema={
            "columns": {
                "x": {
                    "dtype": "float",
                    "na_values": ["N/A"],
                    "nullable": True,
                    "max_null_fraction": 0,
                    "stats": {"distinct_count": {"ge": 0, "le": 0}},
                }
            },
        },
    )
    assert report.verdict() == "passed", [r.message for r in report.results]
