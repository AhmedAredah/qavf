"""Public API contracts: unambiguous inputs, lazy boundaries and ownership."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from qavf import Loader, Query, ResourceLimits, Status, validate
from qavf.engine.runner import QARunner
from qavf.inputs.adapters import _QueryInput
from qavf.settings import QAVFSettings

SCHEMA = {"columns": {"x": {"dtype": "int", "ge": 0}}}


@pytest.mark.parametrize("kind", ["polars", "lazy", "pandas", "arrow", "path"])
def test_equivalent_inputs_keep_the_same_contract(kind, tmp_path):
    frame = pl.DataFrame({"x": [-1, 2]})
    match kind:
        case "polars":
            data = frame
        case "lazy":
            data = frame.lazy()
        case "pandas":
            data = pd.DataFrame({"x": [-1, 2]})
        case "arrow":
            data = pa.table({"x": [-1, 2]})
        case "path":
            data = tmp_path / "data.parquet"
            frame.write_parquet(data)
    report = validate(data, schema=SCHEMA, resources=ResourceLimits(max_workers=1))
    failures = [r for r in report.results if r.status is Status.FAIL]
    assert [(r.rule_id, r.affected_count) for r in failures] == [("INP-003", 1)]


def test_normalization_does_not_collect_lazy_input_or_invoke_loader(monkeypatch):
    lazy = pl.DataFrame({"x": [1]}).lazy()
    calls = []
    received = []

    def load():
        calls.append("called")
        return lazy

    def capture(self, *, frames):
        received.append(frames)
        return None

    monkeypatch.setattr(QARunner, "run", capture)
    validate(lazy, schema=SCHEMA)
    validate(Loader(load), schema=SCHEMA)
    assert received[0]["data"] is lazy
    assert calls == []


def test_loader_called_once_and_keeps_returned_lazy_frame():
    calls = []

    def load(*, multiplier):
        calls.append(multiplier)
        return pl.DataFrame({"x": [-multiplier, multiplier]}).lazy()

    report = validate(Loader(load, kwargs={"multiplier": 2}), schema=SCHEMA)
    assert calls == [2]
    assert any(r.status is Status.FAIL and r.affected_count == 1 for r in report.results)


def test_loader_error_is_unassessed_without_retry():
    calls = []

    def load():
        calls.append(1)
        raise RuntimeError("source offline")

    report = validate(Loader(load), schema=SCHEMA)
    assert calls == [1]
    assert any(r.status is Status.ERROR for r in report.results)
    assert report.exit_code() == 1


def test_native_query_preserves_parameters_without_eager_load(monkeypatch):
    def fail_if_loaded(self):
        raise AssertionError("native query should use its ScanPlan")

    monkeypatch.setattr(_QueryInput, "load", fail_if_loaded)
    report = validate(
        Query("SELECT $value AS x", parameters={"value": -1}),
        config={
            "datasets": {"data": {"expressions": [{"expr": "x >= 0"}]}},
            "engine": {"run_unconfigured_rules": False, "parallel": False},
            "rules": {"INP-006": {}},
        },
    )
    assert any(r.status is Status.FAIL and r.affected_count == 1 for r in report.results)
    assert not any(r.status is Status.ERROR for r in report.results)


def test_borrowed_query_connection_stays_open_after_validation():
    con = duckdb.connect()
    try:
        report = validate(Query("SELECT $value AS x", con, {"value": -1}), schema=SCHEMA)
        assert any(r.status is Status.FAIL for r in report.results)
        assert con.execute("SELECT 42").fetchone() == (42,)
    finally:
        con.close()


def test_configured_query_obeys_snapshot_connection(tmp_path):
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE readings AS SELECT -1 AS x")
    report = validate(
        Query("SELECT x FROM warehouse.readings", connection="warehouse"),
        schema=SCHEMA,
        config={
            "connections": {
                "warehouse": {"type": "duckdb", "path": str(path), "snapshot": True}
            }
        },
    )
    assert any(r.rule_id == "INP-003" and r.status is Status.FAIL for r in report.results)
    assert not any(r.status is Status.ERROR for r in report.results)


@pytest.mark.parametrize("sql", ["DELETE FROM data", "SELECT 1; SELECT 2", "CREATE TABLE x(i INT)"])
def test_query_rejects_mutation_or_multiple_statements(sql):
    with pytest.raises(ValueError, match="exactly one SELECT"):
        Query(sql)


def test_query_and_loader_copy_their_option_mappings():
    parameters = {"value": 1}
    query = Query("SELECT $value AS x", parameters=parameters)
    parameters["value"] = -1
    assert query.parameters["value"] == 1
    kwargs = {"value": 1}
    loader = Loader(lambda **kw: pl.DataFrame(kw), kwargs=kwargs)
    kwargs["value"] = -1
    assert loader.kwargs["value"] == 1


def test_unknown_dataset_does_not_bypass_configured_checks():
    with pytest.raises(ValueError, match="not configured"):
        validate({"typo": pl.DataFrame({"x": [-1]})}, config={"datasets": {"data": {}}})


def test_schema_conflict_and_multi_dataset_shortcut_rejected():
    frame = pl.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="conflicts"):
        validate(frame, schema=SCHEMA, config={"datasets": {"data": {"schema": SCHEMA}}})
    with pytest.raises(ValueError, match="one dataset"):
        validate({"left": frame, "right": frame}, schema=SCHEMA)
    with pytest.raises(ValueError, match="ambiguous"):
        validate(frame, config={"datasets": {"left": {}, "right": {}}})


def test_source_override_preserves_schema_without_mutating_configuration(tmp_path):
    settings = QAVFSettings.model_validate(
        {"datasets": {"data": {"path": str(tmp_path / "absent.csv"), "schema": SCHEMA}}}
    )
    before = settings.model_dump()
    report = validate({"data": pl.DataFrame({"x": [-1]})}, config=settings)
    assert settings.model_dump() == before
    assert any(r.rule_id == "INP-003" and r.status is Status.FAIL for r in report.results)


def test_configuration_file_controls_relative_paths(tmp_path, monkeypatch):
    folder = tmp_path / "config"
    folder.mkdir()
    pl.DataFrame({"x": [1]}).write_csv(folder / "values.csv")
    path = folder / "qa.json"
    path.write_text(json.dumps({"datasets": {"values": {"path": "values.csv", "schema": SCHEMA}}}))
    monkeypatch.chdir(tmp_path)
    report = validate(config=path)
    assert report.exit_code() == 0
    assert not (tmp_path / "qa_output").exists()


def test_resource_limits_use_binary_and_decimal_units_and_preserve_settings():
    resources = ResourceLimits(
        target_memory="2GiB", cache_memory="1MB", max_spill=0,
        evidence_rows=0, evidence_bytes="1KiB", report_bytes="2MiB",
        max_workers=1, native_threads=1,
    )
    report = validate(pl.DataFrame({"x": [-1]}), schema=SCHEMA, resources=resources)
    engine = report.metadata.config["engine"]
    assert engine["memory_budget_bytes"] == 2 * 1024**3
    assert engine["cache_max_bytes"] == 1_000_000
    assert engine["spill_max_bytes"] == 0
    assert engine["max_evidence_bytes"] == 1024
    assert engine["max_report_bytes"] == 2 * 1024**2
    assert all(not r.affected_records for r in report.results)


@pytest.mark.parametrize("options", [
    {"target_memory": 0}, {"max_spill": -1}, {"evidence_rows": -1},
    {"max_workers": True}, {"native_threads": 0}, {"cache_memory": "invalid"},
    {"evidence_bytes": "0.1B"}, {"report_bytes": 0},
])
def test_invalid_resource_limits_are_rejected(options):
    with pytest.raises(ValueError):
        ResourceLimits(**options)


def test_borrowed_one_pass_reader_is_not_silently_consumed():
    reader = pa.RecordBatchReader.from_batches(
        pa.schema([("x", pa.int64())]), [pa.record_batch({"x": [1]})]
    )
    with pytest.raises(TypeError, match="one-pass"):
        validate(reader, schema=SCHEMA)
    assert reader.read_next_batch().column(0).to_pylist() == [1]


def test_no_inferred_acceptance_limits_and_no_implicit_reports(tmp_path, monkeypatch):
    frame = pl.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="not inferred"):
        validate(frame)
    monkeypatch.chdir(tmp_path)
    validate(frame, schema=SCHEMA)
    assert list(tmp_path.iterdir()) == []


def test_importing_package_version_does_not_import_heavy_integrations():
    code = (
        "import sys, qavf; assert qavf.__version__; "
        "assert 'pandas' not in sys.modules; "
        "assert 'qavf.regression.compare_outputs' not in sys.modules; "
        "assert 'qavf.reports.html' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1])
