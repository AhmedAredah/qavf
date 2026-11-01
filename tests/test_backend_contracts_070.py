"""Differential contracts across native and optional dataframe execution."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qavf import File, Query, validate


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("kind", ["arrow", "pandas", "parquet", "csv", "query"])
def test_converted_key_contract_across_backends(tmp_path, backend, kind):
    values = ["07", "7", "+7", "7.0"]
    data = pa.table({"k": values})
    if kind == "pandas":
        data = data.to_pandas()
    elif kind == "parquet":
        path = tmp_path / "keys.parquet"
        pq.write_table(data, path)
        data = path
    elif kind == "csv":
        path = tmp_path / "keys.csv"
        path.write_text("k\n" + "\n".join(values) + "\n")
        data = File(path, options={"types": {"k": "VARCHAR"}})
    elif kind == "query":
        data = Query("SELECT * FROM (VALUES ('07'), ('7'), ('+7'), ('7.0')) t(k)")
    report = validate(
        data,
        backend=backend,
        schema={
            "columns": {"k": {"dtype": "int", "number_grammar": {"int_decimal": True}}},
            "primary_key": ["k"],
            "duplicate_rows_per_key": "all",
        },
    )
    errors = [r.message for r in report.results if r.status.value == "ERROR"]
    assert not errors, errors
    failures = [r for r in report.results if r.rule_id == "INP-004" and r.affected_count]
    assert len(failures) == 1
    failure = failures[0]
    assert failure.affected_count == 1 and failure.affected_rows == 4
    assert failure.affected_records[0]["k"] == 7
    assert failure.affected_records[0]["rows"] == [0, 1, 2, 3]
    assert failure.affected_count_exact


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_na_is_not_a_missing_primary_key(tmp_path, backend):
    data = pa.table({"k": ["N/A", "01", "1"], "v": ["N/A", "2", "3"]})
    report = validate(
        data,
        backend=backend,
        schema={
            "columns": {c: {"dtype": "int", "na_values": ["N/A"]} for c in ["k", "v"]},
            "primary_key": ["k"],
            "na_groups": [["k", "v"]],
        },
    )
    failures = [r for r in report.results if r.status.value in ("ERROR", "FAIL")]
    assert len(failures) == 1, [(r.rule_id, r.message) for r in failures]
    assert failures[0].rule_id == "INP-004"
    assert failures[0].affected_count == 1 and failures[0].affected_rows == 2
    exempt = failures[0].metrics["exempt_na_rows"]
    assert (sum(exempt.values()) if isinstance(exempt, dict) else exempt) == 1


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("nulls,expected", [("exempt", 0), ("equal", 1), ("forbid", 2)])
def test_nullable_alternate_keys(tmp_path, backend, nulls, expected):
    data = pa.table({"a": [1, 1], "b": pa.array([None, None], type=pa.int64())})
    report = validate(
        data,
        backend=backend,
        schema={
            "columns": {"a": {"dtype": "int"}, "b": {"dtype": "int", "nullable": True}},
            "unique_keys": [{"columns": ["a", "b"], "nulls": nulls}],
        },
    )
    assert not [r for r in report.results if r.status.value == "ERROR"]
    assert sum(r.affected_count for r in report.results if r.rule_id == "INP-004") == expected


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_numeric_allowed_values_and_exact_domain(backend):
    schema = {
        "columns": {
            "v": {
                "dtype": "int",
                "isin": ["01", "+2", "3.0"],
                "convert_options": True,
                "number_grammar": {"int_decimal": True},
                "each_allowed_once": True,
            }
        }
    }
    assert (
        validate(pa.table({"v": ["1", "02", "+3"]}), schema=schema, backend=backend).verdict()
        == "passed"
    )
    report = validate(pa.table({"v": ["1", "01", "+3"]}), schema=schema, backend=backend)
    assert report.verdict() == "failed"
    assert not [r for r in report.results if r.status.value == "ERROR"]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_structured_identity_survives_json(backend):
    report = validate(
        pa.table({"v": [1, 2]}),
        backend=backend,
        schema={"columns": {"v": {"dtype": "int", "ge": 0, "le": 3}}},
    )
    for result in report.model_dump(mode="json")["results"]:
        assert result["check_id"]
        assert isinstance(result["checks"], list)
        assert isinstance(result["parameters"], dict)
        assert isinstance(result["columns"], list)
    checks = [c for r in report.results for c in [r.identity, *r.checks]]
    assert any(c.columns == ["v"] and c.parameters.get("ge") == 0 for c in checks)
    assert any(c.columns == ["v"] and c.parameters.get("le") == 3 for c in checks)
