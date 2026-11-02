"""The minimal engine preserves explicit caps and never hides unassessed work."""

import json

import pyarrow as pa
import pytest

from qavf import File, Query, validate
from qavf.audit_codec import evidence_json_size
from qavf.core.results import ResultCollector
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings


@pytest.mark.parametrize("native,expected", [("1GB", "64.0 MiB"), ("16MiB", "16.0 MiB")])
def test_native_and_run_memory_caps_compose_by_taking_the_smaller(native, expected):
    report = validate(
        Query("SELECT current_setting('memory_limit') AS cap"),
        schema={"columns": {"cap": {"isin": [expected]}}},
        config={"engine": {"memory_budget_bytes": 64 * 1024**2, "duckdb_memory_limit": native}},
        backend="duckdb",
    )
    assert report.exit_code() == 0
    assert report.metadata.resource_usage["native_memory_limit"] == expected


def test_relative_temporary_parent_is_configuration_relative_and_owned_files_cleaned(tmp_path):
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"engine": {"temp_directory": "owned", "spill_max_bytes": 0}}))
    report = validate(pa.table({"x": [1]}), config=config, backend="duckdb")
    assert report.exit_code() == 0
    assert report.metadata.resource_usage["native_spill_limit_bytes"] == 0
    assert (tmp_path / "owned").is_dir()
    assert not list((tmp_path / "owned").iterdir())


def test_records_metrics_and_diagnostic_share_evidence_budget():
    settings = QAVFSettings(engine={"max_evidence_bytes": 50, "max_affected_records": 2})
    collector = ResultCollector(None, settings)
    collector.report(
        "INP-003",
        None,
        Status.FAIL,
        "diagnostic " * 100,
        columns=("x",),
        count=5,
        records=[{"x": i} for i in range(5)],
        metrics={"huge": "x" * 1000},
    )
    result = collector.results[0]
    assert result.affected_count == 5 and len(result.affected_records) <= 2
    assert result.evidence_truncated and result.evidence_complete is False
    payload = (
        (evidence_json_size(result.affected_records) if result.affected_records else 0)
        + (evidence_json_size(result.metrics) if result.metrics else 0)
        + len(result.message.encode())
    )
    assert payload <= 50


def test_source_failure_cannot_disappear_when_availability_rule_is_disabled(tmp_path):
    settings = {
        "rules": {"INP-001": {"enabled": False}},
        "datasets": {
            "good": {"schema": {"columns": {"x": {"dtype": "int"}}}},
            "bad": {
                "path": str(tmp_path / "absent.csv"),
                "schema": {"columns": {"x": {"dtype": "int"}}},
            },
        },
    }
    report = validate({"good": pa.table({"x": [1]})}, config=settings, backend="duckdb")
    assert report.exit_code() != 0
    assert any(
        r.rule_id == "INP-002" and r.datasets == ["bad"] and r.status == Status.ERROR
        for r in report.results
    )
    assert all(r.rule_id != "INP-001" for r in report.results)


def test_disabled_assertion_is_not_executed(monkeypatch):
    def forbidden(query):
        raise AssertionError("disabled assertion was evaluated")

    monkeypatch.setattr("qavf.core.runner.validate_query", forbidden)
    report = validate(
        pa.table({"x": [1]}),
        config={
            "rules": {"INP-007": {"enabled": False}},
            "assertions": [{"name": "disabled", "sql": "SELECT * FROM absent"}],
        },
        backend="duckdb",
    )
    assert report.exit_code() == 0
    assert all(r.rule_id != "INP-007" for r in report.results)


def test_key_execution_error_does_not_depend_on_column_rule_being_enabled(monkeypatch):
    def broken(self):
        raise RuntimeError("native key operator failed")

    monkeypatch.setattr("qavf.core.checks.DatasetChecks.keys", broken)
    report = validate(
        pa.table({"x": [1]}),
        schema={"primary_key": ["x"]},
        config={"rules": {"INP-003": {"enabled": False}}},
        backend="duckdb",
    )
    assert report.exit_code() != 0
    assert any(r.rule_id == "INP-004" and r.status == Status.ERROR for r in report.results)
    assert all(r.rule_id != "INP-003" for r in report.results)


def test_only_availability_does_not_compile_disabled_column_contracts(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled check compiler was called")

    monkeypatch.setattr("qavf.core.runner.DatasetChecks", forbidden)
    report = validate(
        pa.table({"x": [1]}),
        schema={"columns": {"x": {"ge": 0}}},
        config={"engine": {"run_unconfigured_rules": False}, "rules": {"INP-001": {}}},
        backend="duckdb",
    )
    assert report.exit_code() == 0
    assert [r.rule_id for r in report.results] == ["INP-001"]


def test_core_report_redacts_credentials_in_configuration_and_parameters():
    report = validate(
        pa.table({"x": [1]}),
        config={"extensions": {"plugin": {"password": "private-value"}}},
        backend="duckdb",
    )
    assert "private-value" not in report.model_dump_json()


@pytest.mark.parametrize("kind", ["string", "path", "file", "source"])
def test_native_file_resource_plan_records_input_and_key_size(tmp_path, kind):
    path = tmp_path / "keys.csv"
    path.write_text("a,b\n1,2\n1,2\n3,4\n3,4\n")
    value = {
        "string": str(path),
        "path": path,
        "file": File(path),
        "source": {"type": "file", "path": str(path)},
    }[kind]
    report = validate(
        {"d": value},
        config={
            "datasets": {
                "d": {
                    "schema": {
                        "columns": {"a": {"dtype": "int"}, "b": {"dtype": "int"}},
                        "primary_key": ["a", "b"],
                    }
                }
            }
        },
        backend="duckdb",
    )
    plan = report.metadata.resource_usage["resource_plan"]
    assert plan["input_bytes"] == path.stat().st_size
    assert plan["estimated_key_state_bytes"] > 0
    assert plan["automatic_memory"] and plan["memory_bytes"] is not None
