"""Plugin discovery outcomes and file-source guarantees.

Plugin discovery outcomes and required plugins, glob completeness, acknowledged lossy
reader options with counted rejects, disclosure of how data was loaded, atomic writes,
and missing database files treated as absent.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine import registry as registry_module
from qavf.engine.registry import RuleRegistry
from qavf.engine.runner import ConfigurationError, QARunner
from qavf.engine.severity import Status
from qavf.io import atomic_write, write_table
from qavf.settings import QAVFSettings
from qavf.sources.parameters import FileParams
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


def _settings(**config) -> QAVFSettings:
    return QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "engine": {"run_unconfigured_rules": False}, **config}
    )


# ------------------------------------------------------------------ plugin discovery
class _Entry(SimpleNamespace):
    def load(self):
        return self.loader()


def _broken():
    raise ImportError("plugin needs a missing dependency")


def test_plugin_outcomes_are_recorded_and_required_plugins_fail_closed(monkeypatch) -> None:
    from qavf.rules.input_checks import availability

    entries = [_Entry(name="good", loader=lambda: availability), _Entry(name="bad", loader=_broken)]
    monkeypatch.setattr(registry_module, "entry_points", lambda group: entries)
    registry = RuleRegistry()
    registry.discover_entry_points()
    assert {r.rule_id for r in registry} >= {"INP-001"}
    assert registry.plugins["good"] == "loaded" and "missing dependency" in registry.plugins["bad"]

    runner = QARunner(_settings(rules={"INP-001": {}}), registry)
    report = runner.run()
    assert report.metadata.plugins["rules:good"] == "loaded"
    assert report.metadata.plugins["rules:bad"].startswith("error:")

    for required, match in ((["bad"], "missing dependency"), (["absent"], "not installed")):
        settings = _settings(
            rules={"INP-001": {}}, plugins={"entry_points": False, "required": required}
        )
        with pytest.raises(ConfigurationError, match=match):
            QARunner(settings, registry).plan()
    ok = _settings(rules={"INP-001": {}}, plugins={"entry_points": False, "required": ["good"]})
    assert QARunner(ok, registry).plan()


# ------------------------------------------------------------------ glob completeness
def test_min_files_for_globs(tmp_path) -> None:
    for part in (1, 2):
        pl.DataFrame({"a": [part]}).write_parquet(tmp_path / f"part{part}.parquet")
    datasets = {
        "d": {"source": {"type": "file", "path": str(tmp_path / "*.parquet"), "min_files": 3}}
    }
    report = QARunner(_settings(datasets=datasets, rules={"INP-001": {}}), REGISTRY).run()
    result = report.get("INP-001")
    assert result.failed and "matched 2 file(s), expected at least 3" in result.message
    datasets["d"]["source"]["min_files"] = 2
    report = QARunner(_settings(datasets=datasets, rules={"INP-001": {}}), REGISTRY).run()
    [info] = report.metadata.datasets
    assert info.notes["files_matched"] == 2


def test_min_files_needs_a_local_glob() -> None:
    with pytest.raises(ValidationError, match="min_files"):
        FileParams(path="data.parquet", min_files=2)


def test_glob_fingerprints_list_the_files(tmp_path) -> None:
    for part in (1, 2):
        pl.DataFrame({"a": [part]}).write_parquet(tmp_path / f"part{part}.parquet")
    datasets = {"d": {"path": str(tmp_path / "*.parquet")}}
    report = QARunner(_settings(datasets=datasets, rules={"INP-001": {}}), REGISTRY).run()
    fingerprint = report.metadata.datasets[0].fingerprint
    assert fingerprint["files"] == 2 and len(fingerprint["paths"]) == 2
    assert fingerprint["paths_truncated"] is False


# ------------------------------------------------------------------ lossy reader options
@pytest.mark.parametrize(
    "options",
    [{"ignore_errors": True}, {"skip": 2}, {"store_rejects": True}],
)
def test_lossy_reader_options_must_be_acknowledged(options) -> None:
    with pytest.raises(ValidationError, match="acknowledge_lossy"):
        FileParams(path="data.csv", options=options)
    FileParams(path="data.csv", options=options, acknowledge_lossy=True)


def test_harmless_options_need_no_acknowledgement() -> None:
    FileParams(path="data.csv", options={"delim": ";", "skip": 0, "ignore_errors": False})


def test_rejected_csv_rows_are_counted_and_loading_is_disclosed(tmp_path) -> None:
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\n3,x\n5,6\n7\n", encoding="utf-8")
    source = {
        "type": "file",
        "path": str(path),
        "options": {"ignore_errors": True, "columns": {"a": "INTEGER", "b": "INTEGER"}},
        "acknowledge_lossy": True,
    }
    report = QARunner(
        _settings(datasets={"d": {"source": source}}, rules={"INP-001": {}}), REGISTRY
    ).run()
    [info] = report.metadata.datasets
    assert info.rows == 2 and info.notes["rejected_rows"] == 2
    assert info.notes["reader_options"]["ignore_errors"] is True
    # A projection is disclosed too. (DuckDB then never parses the other columns, so a
    # bad value there is not a reject: the projection defines what was validated.)
    projected = {**source, "columns": ["a"]}
    report = QARunner(
        _settings(datasets={"d": {"source": projected}}, rules={"INP-001": {}}), REGISTRY
    ).run()
    [info] = report.metadata.datasets
    assert info.notes["columns_loaded"] == ["a"] and info.notes["rejected_rows"] == 1


# ------------------------------------------------------------------ atomic writes
def test_failed_writes_leave_the_previous_file_intact(tmp_path, monkeypatch) -> None:
    target = tmp_path / "baseline.csv"
    write_table(pl.DataFrame({"a": [1]}), target)
    before = target.read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        write_table(pl.DataFrame({"a": [2, 3]}), target)
    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["baseline.csv"]  # no temp left


def test_atomic_write_creates_parents(tmp_path) -> None:
    path = atomic_write(tmp_path / "a" / "b.txt", "hello")
    assert path.read_text(encoding="utf-8") == "hello"


# ------------------------------------------------------------------ missing database files
def test_a_missing_optional_database_file_is_absent_not_broken(tmp_path) -> None:
    settings = _settings(
        connections={"archive": {"type": "duckdb", "path": str(tmp_path / "missing.duckdb")}},
        datasets={
            "old": {
                "source": {"type": "database", "connection": "archive", "table": "t"},
                "required": False,
            }
        },
        rules={"INP-001": {}},
    )
    report = QARunner(settings, REGISTRY).run()
    assert not report.get("INP-001").failed  # optional and absent: nothing to report
    assert "not found" in report.metadata.unavailable_datasets["old"]


def test_a_missing_required_database_file_fails(tmp_path) -> None:
    db = tmp_path / "present.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE t AS SELECT 1 AS a")
    settings = _settings(
        connections={"archive": {"type": "duckdb", "path": str(tmp_path / "missing.duckdb")}},
        datasets={"old": {"source": {"type": "database", "connection": "archive", "table": "t"}}},
        rules={"INP-001": {}},
    )
    assert QARunner(settings, REGISTRY).run().get("INP-001").status is Status.FAIL
