"""The core contract is executable without an optional dataframe backend."""

from __future__ import annotations

import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qavf import Dataset, File, Query, ResourceLimits, validate
from qavf.optional import OptionalDependencyError


def test_validation_import_does_not_load_heavy_backends():
    program = (
        "from qavf import validate; import sys; "
        "assert not any(n in sys.modules for n in "
        "['polars', 'pandas', 'pyarrow', 'duckdb', 'numpy'])"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_core_runs_and_writes_reports_when_polars_import_is_blocked(tmp_path):
    program = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'polars' or fullname.startswith('polars.'):
            raise ImportError('Polars deliberately unavailable')
sys.meta_path.insert(0, Block())
from qavf import validate, File, Dataset, Query, ResourceLimits
from qavf.reports.html import write_html
import pyarrow as pa
r = validate(pa.table({'k':['07','7','+7']}),
             schema={'columns': {'k': {'dtype':'int'}}, 'primary_key':['k']})
assert r.verdict() == 'failed'
assert any(x.affected_rows == 3 for x in r.results)
r.write_json(sys.argv[1])
write_html(r, sys.argv[2])
assert 'polars' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "r.json"), str(tmp_path / "r.html")],
        check=True,
    )


@pytest.mark.parametrize("source", ["arrow", "csv", "parquet", "query", "loader"])
def test_core_sources_share_converted_values(source, tmp_path):
    from qavf import Loader

    table = pa.table({"key": ["07", "7", "+7", "7.0"], "value": [1, 2, 3, 4]})
    if source == "arrow":
        data = table
    elif source == "parquet":
        path = tmp_path / "data.parquet"
        pq.write_table(table, path)
        data = path
    elif source == "csv":
        path = tmp_path / "data.csv"
        path.write_text("key,value\n07,1\n7,2\n+7,3\n7.0,4\n")
        data = File(path, options={"all_varchar": True})
    elif source == "query":
        data = Query("select * from (values ('07', 1), ('7', 2), ('+7',3), ('7.0',4)) t(key,value)")
    else:
        data = Loader(lambda: table)
    report = validate(
        data,
        schema={
            "columns": {"key": {"dtype": "int", "number_grammar": {"int_decimal": True}}},
            "primary_key": ["key"],
            "duplicate_rows_per_key": "all",
        },
        backend="duckdb",
    )
    assert report.verdict() == "failed"
    duplicate = next(r for r in report.results if r.affected_rows == 4)
    assert duplicate.affected_count == 1
    assert duplicate.affected_records[0]["rows"] == [0, 1, 2, 3]


def test_core_options_are_exact_and_coercion_is_explicit():
    table = pa.table({"k": ["07"]})
    report = validate(
        table, schema={"columns": {"k": {"dtype": "int", "isin": ["07"]}}}, backend="duckdb"
    )
    assert report.verdict() != "passed"
    report = validate(
        table,
        schema={"columns": {"k": {"dtype": "int", "isin": ["07"], "convert_options": True}}},
        backend="duckdb",
    )
    assert report.verdict() == "passed"


def test_core_constraints_and_exactly_once():
    report = validate(
        pa.table({"k": ["07", "8"]}),
        schema={
            "columns": {
                "k": {
                    "dtype": "int",
                    "isin": ["7", 8],
                    "convert_options": True,
                    "each_allowed_once": True,
                    "ge": 0,
                    "sum": {"ge": 15, "le": 15},
                }
            }
        },
        backend="duckdb",
    )
    assert report.verdict() == "passed", [(r.check, r.message) for r in report.results if r.failed]


def test_core_na_states_and_nullable_key_policy():
    report = validate(
        pa.table({"k": ["N/A", None, None], "b": [1, 2, 2]}),
        schema={
            "columns": {
                "k": {"dtype": "int", "nullable": True, "na_values": ["N/A"]},
                "b": {"dtype": "int"},
            },
            "unique_keys": [{"columns": ["k", "b"], "nulls": "equal"}],
        },
        backend="duckdb",
    )
    failed = [r for r in report.results if r.failed]
    assert len(failed) == 1
    assert failed[0].affected_rows == 2


def test_core_na_group_and_where_severity():
    report = validate(
        pa.table({"a": ["N/A", "4"], "b": ["1", "2"]}),
        schema={
            "columns": {
                "a": {
                    "dtype": "int",
                    "na_values": ["N/A"],
                    "checks": [{"where": "b = 2", "ge": 5, "severity": "WARNING"}],
                },
                "b": {"dtype": "int", "na_values": ["N/A"]},
            },
            "na_groups": [["a", "b"]],
        },
        backend="duckdb",
    )
    assert any(r.rule_id == "INP-009" and r.affected_count == 1 for r in report.results)
    assert any(str(r.severity) == "WARNING" and r.affected_count == 1 for r in report.results)


def test_core_foreign_keys_expressions_and_sql_assertions():
    inputs = {
        "parent": Dataset(pa.table({"id": ["07"]}), schema={"columns": {"id": {"dtype": "int"}}}),
        "child": Dataset(
            pa.table({"id": [7, 8], "n": [1, 2]}),
            foreign_keys=[{"columns": ["id"], "ref_dataset": "parent"}],
            checks=[{"check": "expression", "expr": "n < 2"}],
        ),
    }
    report = validate(
        inputs,
        config={
            "checks": [
                {"check": "sql", "name": "too big", "sql": "select * from child where n = 2"}
            ]
        },
        backend="duckdb",
    )
    assert {r.rule_id for r in report.results if r.failed} == {"INP-005", "INP-006", "INP-007"}


def test_core_explicit_unsupported_work_is_rejected_before_loader_runs():
    from qavf import Loader

    calls = []
    with pytest.raises(OptionalDependencyError, match=r"qavf\[polars\]"):
        validate(
            Loader(lambda: calls.append(True)),
            config={"checks": [{"check": "coverage", "entity": "x"}]},
            backend="duckdb",
        )
    assert calls == []


def test_core_evidence_budget_never_changes_totals():
    report = validate(
        pa.table({"k": [7] * 1000}),
        schema={"primary_key": ["k"], "duplicate_rows_per_key": "all"},
        resources=ResourceLimits(evidence_bytes=100),
        backend="duckdb",
    )
    result = next(r for r in report.results if r.affected_rows == 1000)
    assert result.affected_count == 1
    assert result.affected_count_exact
    from qavf.audit_codec import evidence_json_size

    assert evidence_json_size(result.affected_records) <= 100


def test_core_strict_csv_diagnostic(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text('id,value\n1,"unterminated\n')
    report = validate(path, schema={"min_rows": 0}, backend="duckdb")
    error = next(r for r in report.results if r.failed)
    assert "row 2" in error.message and "line 2" in error.message
    assert "quote" in error.message.lower()


def test_core_optional_csv_column_and_renamed_governed_text(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("raw\n07\n7\n")
    report = validate(
        Dataset(path, rename={"raw": "key"}),
        schema={
            "columns": {
                "key": {"dtype": "int"},
                "absent": {"dtype": "int", "required": False},
            },
            "primary_key": ["key"],
        },
        backend="duckdb",
    )
    failures = [result for result in report.results if result.failed]
    assert len(failures) == 1
    assert failures[0].affected_rows == 2


def test_core_query_file_paths_follow_configuration_directory(tmp_path):
    import json

    (tmp_path / "relative.csv").write_text("value\n7\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"datasets": {"data": {"schema": {"columns": {"value": {"ge": 7, "le": 7}}}}}})
    )
    report = validate(
        Query("SELECT * FROM read_csv('relative.csv')"), config=config, backend="duckdb"
    )
    assert report.verdict() == "passed"
