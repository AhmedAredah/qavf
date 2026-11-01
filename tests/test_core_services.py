"""Dialect, provenance and shared native lifecycle regressions."""

from __future__ import annotations

import time

import duckdb
import pyarrow as pa

from qavf import Query, validate
from qavf.engine.severity import Status


def test_native_regex_declines_non_equivalent_unicode_and_runs_next_check():
    report = validate(
        pa.table({"x": ["é"]}),
        schema={
            "columns": {
                "x": {
                    "checks": [
                        {"regex": r"\w+", "name": "Unicode"},
                        {"min_length": 1, "name": "length"},
                    ]
                }
            }
        },
        backend="duckdb",
    )
    errors = [r for r in report.results if r.status is Status.ERROR]
    assert len(errors) == 1
    assert "shared Rust/RE2 subset" in errors[0].message
    assert any(r.status is Status.PASS and "length" in r.check for r in report.results)
    full = validate(
        pa.table({"x": ["é"]}), schema={"columns": {"x": {"regex": r"\w+"}}}, backend="polars"
    )
    assert full.acceptance.verdict == "passed"


def test_where_admission_does_not_depend_on_optional_package(monkeypatch):
    import qavf.optional

    for installed in (True, False):
        monkeypatch.setattr(qavf.optional, "has_polars", lambda installed=installed: installed)
        report = validate(
            pa.table({"x": [2.0]}),
            schema={
                "columns": {
                    "x": {
                        "checks": [
                            {"where": "isfinite(x)", "ge": 0},
                        ]
                    }
                }
            },
            backend="duckdb",
        )
        assert report.acceptance.verdict == "passed"


def test_projection_and_dependency_component_release(monkeypatch):
    from qavf.core.sources import InputReader

    original = InputReader.load
    retained = []

    def load(self, name, value, spec):
        count = self.connection.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE '__qavf_input_%'"
        ).fetchone()[0]
        assert count == 0
        result = original(self, name, value, spec)
        retained.append(result.columns)
        return result

    monkeypatch.setattr(InputReader, "load", load)
    config = {
        "datasets": {
            name: {"schema": {"columns": {"id": {"dtype": "int"}}, "primary_key": ["id"]}}
            for name in ("a", "b")
        }
    }
    report = validate(
        {name: pa.table({"id": [1, 2], "wide": ["payload" * 1000] * 2}) for name in ("a", "b")},
        config=config,
        backend="duckdb",
    )
    assert report.acceptance.verdict == "passed"
    assert retained == [("id",), ("id",)]
    assert report.metadata.datasets[0].columns == ["id", "wide"]


def test_foreign_key_parent_projection_and_exact_dependencies():
    report = validate(
        {"a": pa.table({"child": [1]}), "b": pa.table({"parent": [1], "payload": ["huge"]})},
        config={
            "datasets": {
                "a": {
                    "foreign_keys": [
                        {"columns": ["child"], "ref_dataset": "b", "ref_columns": ["parent"]}
                    ]
                },
                "b": {},
            }
        },
        backend="duckdb",
    )
    assert report.acceptance.verdict == "passed"
    assert report.metadata.datasets[1].notes["retained_columns"] == ["parent"]


def test_native_metadata_provenance_and_timing(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n2\n")
    report = validate(
        path,
        schema={"columns": {"x": {"dtype": "int", "ge": 0}}},
        config={"report": {"hash_inputs": True}},
        backend="duckdb",
    )
    assert report.acceptance.verdict == "passed"
    metadata = report.metadata
    assert metadata.config_digest and metadata.plan_digest
    assert metadata.environment["duckdb"] == duckdb.__version__
    assert metadata.rules and metadata.resource_usage["stage_timings_s"]
    assert metadata.datasets[0].loaded_at and metadata.datasets[0].source
    assert "sha256" in str(metadata.datasets[0].fingerprint)
    assert any(result.execution_time_s > 0 for result in report.results)


def test_native_require_marks_disabled_and_missing_contracts_incomplete():
    report = validate(
        pa.table({"x": [1]}),
        config={"engine": {"require": ["INP-004"]}, "rules": {"INP-004": {"enabled": False}}},
        backend="duckdb",
    )
    assert report.acceptance.verdict == "incomplete"
    result = next(r for r in report.results if r.check_id == "acceptance.required")
    assert result.status is Status.ERROR and not result.affected_count_exact


def test_cooperative_native_deadline_discards_partial_assessment(monkeypatch):
    from qavf.core.checks import DatasetChecks

    original = DatasetChecks.columns

    def slow(self):
        original(self)
        time.sleep(0.02)

    monkeypatch.setattr(DatasetChecks, "columns", slow)
    report = validate(
        pa.table({"x": [1]}),
        schema={"columns": {"x": {"ge": 0}}},
        config={"engine": {"rule_timeout_s": 0.005}},
        backend="duckdb",
    )
    assert any(
        r.status is Status.ERROR and r.parameters.get("stage") == "timeout" for r in report.results
    )
    assert not any(r.rule_id == "INP-003" and r.status is Status.PASS for r in report.results)


def test_native_isolated_file_preserves_default_resource_policy(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n2\n")
    report = validate(
        path,
        schema={"primary_key": ["x"]},
        config={"engine": {"execution_mode": "isolated", "run_timeout_s": 20}},
        backend="duckdb",
    )
    assert report.acceptance.verdict == "passed"
    assert report.metadata.resource_usage["execution_mode"] == "isolated"


def test_native_configured_database_and_parameterized_query(tmp_path):
    path = tmp_path / "source.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE valueset AS SELECT 7::BIGINT AS id")
    config = {
        "connections": {"warehouse": {"type": "duckdb", "path": str(path)}},
        "datasets": {"d": {"schema": {"primary_key": ["id"]}}},
    }
    report = validate(
        Query(
            "SELECT * FROM warehouse.valueset WHERE id = $n",
            connection="warehouse",
            parameters={"n": 7},
        ),
        config=config,
        backend="duckdb",
    )
    assert report.acceptance.verdict == "passed"
    config["datasets"]["d"]["source"] = {
        "type": "database",
        "connection": "warehouse",
        "table": "valueset",
    }
    assert validate(config=config, backend="duckdb").acceptance.verdict == "passed"


def test_core_shape_only_can_retain_no_value_columns():
    report = validate(pa.table({"payload": ["a", "b"]}), schema={"min_rows": 1}, backend="duckdb")
    assert report.acceptance.verdict == "passed"
    assert report.metadata.datasets[0].notes["retained_columns"] == []


def test_source_query_dependencies_survive_projection_and_reverse_input_order():
    report = validate(
        {"derived": Query("SELECT value FROM original"), "original": pa.table({"value": [7]})},
        config={
            "datasets": {"derived": {"schema": {"columns": {"value": {"ge": 7}}}}, "original": {}}
        },
        backend="duckdb",
    )
    assert report.acceptance.verdict == "passed"
    assert report.metadata.datasets[0].name == "original"


def test_projection_retains_case_insensitive_and_nested_condition_dependencies():
    report = validate(
        pa.table({"x": [1], "Status": ["ready"], "payload": [{"flag": True}]}),
        schema={
            "columns": {"x": {"checks": [{"where": "STATUS = 'ready' AND payload.flag", "ge": 2}]}}
        },
        backend="duckdb",
    )
    failed = [r for r in report.results if r.status is Status.FAIL]
    assert len(failed) == 1 and failed[0].affected_count == 1
    assert failed[0].affected_records[0]["Status"] == "ready"
    assert failed[0].affected_records[0]["payload"] == {"flag": True}


def test_subquery_condition_conservatively_retains_other_sources():
    report = validate(
        {"measurements": pa.table({"x": [1]}), "limits": pa.table({"threshold": [2]})},
        config={
            "datasets": {
                "measurements": {
                    "schema": {
                        "columns": {
                            "x": {
                                "checks": [
                                    {"where": "x < (SELECT max(threshold) FROM limits)", "ge": 2}
                                ]
                            }
                        }
                    }
                },
                "limits": {},
            }
        },
        backend="duckdb",
    )
    failed = [r for r in report.results if r.status is Status.FAIL]
    assert len(failed) == 1 and failed[0].affected_count == 1
