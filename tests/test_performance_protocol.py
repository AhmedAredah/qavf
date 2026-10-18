"""Exercise process provenance, planning, and durable benchmark failure evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import qavf
from scripts.performance import planning, provenance, suite, worker
from scripts.performance.models import WorkloadOptions

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/benchmark_performance.py"
PRODUCT_ROOT = Path(qavf.__file__).resolve().parents[1]


def _plan():
    return planning.variants(
        ["wide_validation"], [1000], WorkloadOptions(threads=1), ["native"], ["parquet"]
    )


def _protocol():
    return {
        "profile": "test",
        "rows": [1000],
        "seed": 123,
        "threads": 1,
        "repetitions": 1,
        "warmups": 0,
        "sample_interval_seconds": 0.01,
        "report_format": "both",
        "partitions": 2,
        "chunk_rows": 250,
        "execution": "sequential fresh processes; alternating source order",
        "os_page_cache": "uncontrolled",
        "timeout_seconds": 60,
    }


@pytest.fixture(scope="module")
def cli_result(tmp_path_factory):
    directory = tmp_path_factory.mktemp("performance-cli")
    output = directory / "evidence/results.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--candidate",
            str(PRODUCT_ROOT),
            "--cases",
            "wide_validation",
            "--rows",
            "1000",
            "--backends",
            "native",
            "--formats",
            "parquet",
            "--repeat",
            "1",
            "--warmups",
            "0",
            "--threads",
            "1",
            "--partitions",
            "2",
            "--chunk-rows",
            "250",
            "--report-format",
            "both",
            "--timeout",
            "60",
            "--work-dir",
            str(directory / "work"),
            "--output",
            str(output),
        ],
        cwd=directory,
        env=suite.worker_environment(1),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(output.read_text()), output


def test_real_cli_uses_requested_source_and_validates_outcome(cli_result):
    record, _ = cli_result
    assert record["status"] == "passed"
    assert record["identity_unchanged"] is True
    assert len(record["records"]) == 1
    measurement = record["records"][0]
    assert measurement["status"] == "passed"
    assert measurement["returncode"] == 0
    assert measurement["environment"]["qavf_import_path"] == str(PRODUCT_ROOT / "qavf/__init__.py")
    assert measurement["environment"]["polars_threads"] == 1
    assert isinstance(measurement["outcome"], dict) and measurement["outcome"]
    assert measurement["expected_source_sha256"] == provenance.production_digest(PRODUCT_ROOT)
    assert set(measurement["phases"]) == {"setup", "operation", "report"}
    for phase in measurement["phases"].values():
        assert phase["wall_seconds"] >= 0
        assert phase["sampling"]["sample_count"] >= 2
        assert phase["sampling"]["sample_failures"] == 0
        assert phase["sampling"]["monitor_error"] is None


def test_real_cli_measures_reports_then_removes_owned_inputs_and_outputs(cli_result):
    record, output = cli_result
    measurement = record["records"][0]
    assert {file["name"] for file in measurement["report_files"]} == {
        "qa_report.json",
        "qa_report.html",
    }
    report_bytes = sum(file["size_bytes"] for file in measurement["report_files"])
    assert measurement["report_bytes"] == report_bytes > 0
    assert measurement["phases"]["report"]["disk"]["final_logical_bytes"] >= report_bytes
    assert record["work_directory_removed"] is True
    assert not Path(record["work_directory"]).exists()
    assert output.is_file() and output.with_suffix(".html").is_file()
    assert not Path(measurement["operation_dir"]).exists()
    assert record["fixtures"]["validation-1000"]["total_bytes"] > 0


def test_worker_cli_wrong_source_returns_nonzero_and_durable_failure(tmp_path):
    request = {
        "id": "wrong-source",
        "source_root": str(tmp_path / "absent-source"),
        "operation_dir": str(tmp_path / "operation"),
        "options": asdict(WorkloadOptions(threads=1)),
        "expected_source_sha256": "unavailable",
        "interval_s": 0.01,
    }
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request))
    completed = subprocess.run(
        [sys.executable, str(ENTRY), "--worker", str(request_path), "--result", str(result_path)],
        env=suite.worker_environment(1),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    result = json.loads(result_path.read_text())
    assert result["id"] == request["id"]
    assert result["status"] == "failed"
    assert result["error"]["phase"] == "identity"
    assert "No QAVF source package" in result["error"]["message"]


def test_worker_rejects_import_path_different_from_selected_source(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        worker,
        "worker_environment",
        lambda threads: {
            "qavf_import_path": str(tmp_path / "unexpected/qavf/__init__.py"),
            "polars_threads": threads,
        },
    )
    result = worker.run(
        {
            "id": "wrong-import",
            "source_root": str(PRODUCT_ROOT),
            "operation_dir": str(tmp_path / "operation"),
            "options": asdict(WorkloadOptions(threads=1)),
            "expected_source_sha256": provenance.production_digest(PRODUCT_ROOT),
            "interval_s": 0.01,
        }
    )
    assert result["status"] == "failed"
    assert result["error"]["phase"] == "setup"
    assert "different source" in result["error"]["message"]
    assert "setup" in result["phases"]


def test_schedule_alternates_sources_and_keeps_warmups_distinct():
    jobs = planning.schedule(_plan(), ["baseline", "candidate"], repetitions=3, warmups=1)
    assert len(jobs) == len({job["id"] for job in jobs}) == 8
    warmups = [job for job in jobs if job["warmup"]]
    measured = [job for job in jobs if not job["warmup"]]
    assert [job["source_label"] for job in warmups] == ["baseline", "candidate"]
    assert [job["source_label"] for job in measured] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert [job["repetition"] for job in measured] == [1, 1, 2, 2, 3, 3]


def test_empty_schedule_cannot_appear_successful():
    with pytest.raises(ValueError):
        planning.schedule([], ["candidate"], repetitions=1, warmups=0)


@pytest.mark.parametrize("rows", [[], [99], [100, 100]])
def test_invalid_scale_selection_rejected(rows):
    with pytest.raises(ValueError):
        planning.variants(["wide_validation"], rows, WorkloadOptions())


def test_source_digest_tracks_bytes_and_relative_paths_but_ignores_runtime_cache(tmp_path):
    original = tmp_path / "original"
    package = original / "qavf"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    module = package / "logic.py"
    module.write_text("value = 1\n")
    initial = provenance.production_digest(original)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    assert provenance.production_digest(relocated) == initial
    (package / "__pycache__").mkdir()
    (package / "__pycache__/logic.pyc").write_bytes(b"runtime cache")
    assert provenance.production_digest(original) == initial
    module.write_text("value = 2\n")
    changed = provenance.production_digest(original)
    assert changed != initial
    module.rename(package / "renamed.py")
    assert provenance.production_digest(original) != changed


def test_fixture_identity_detects_content_and_path_mutations(tmp_path):
    path = tmp_path / "part-0.parquet"
    path.write_bytes(b"input bytes")
    original = provenance.fixture_identity(tmp_path)
    assert original["total_bytes"] == len(b"input bytes")
    renamed = path.rename(tmp_path / "part-1.parquet")
    moved = provenance.fixture_identity(tmp_path)
    assert moved["sha256"] != original["sha256"]
    assert moved["files"][0]["sha256"] == original["files"][0]["sha256"]
    renamed.write_bytes(b"changed bytes")
    assert provenance.fixture_identity(tmp_path)["sha256"] != moved["sha256"]


def test_fixture_preparation_failure_preserves_durable_suite_evidence(monkeypatch, tmp_path):
    def broken_fixture(*args, **kwargs):
        raise RuntimeError("fixture creation failed")

    monkeypatch.setattr(suite, "_prepare", broken_fixture)
    output = tmp_path / "results.json"
    result = suite.run_suite(
        root=ROOT,
        sources={"candidate": PRODUCT_ROOT},
        plan=_plan(),
        protocol=_protocol(),
        output=output,
        timeout_s=10,
        work_parent=tmp_path / "work",
    )
    assert result["status"] == "failed"
    assert result["error"]["message"] == "fixture creation failed"
    assert result["records"] == []
    assert result["work_directory_removed"] is True
    assert json.loads(output.read_text())["status"] == "failed"
    assert output.with_suffix(".html").is_file()


def test_cleanup_failure_is_recorded_without_losing_preparation_failure(monkeypatch, tmp_path):
    def broken_fixture(*args, **kwargs):
        raise RuntimeError("fixture creation failed")

    def broken_cleanup(*args, **kwargs):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(suite, "_prepare", broken_fixture)
    monkeypatch.setattr(suite.shutil, "rmtree", broken_cleanup)
    output = tmp_path / "results.json"
    result = suite.run_suite(
        root=ROOT,
        sources={"candidate": PRODUCT_ROOT},
        plan=_plan(),
        protocol=_protocol(),
        output=output,
        timeout_s=10,
        work_parent=tmp_path / "work",
    )
    assert result["status"] == "failed"
    assert result["error"]["message"] == "fixture creation failed"
    assert result["cleanup_error"]["type"] == "PermissionError"
    assert result["work_directory_removed"] is False
    assert Path(result["work_directory"]).exists()
    durable = json.loads(output.read_text())
    assert durable["status"] == "failed"
    assert durable["cleanup_error"] == result["cleanup_error"]
    assert output.with_suffix(".html").is_file()


def test_worker_environment_removes_parent_import_overrides(monkeypatch):
    for variable in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        monkeypatch.setenv(variable, "parent-setting")
    environment = suite.worker_environment(3)
    assert not {"PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"} & environment.keys()
    for variable in ("POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        assert environment[variable] == "3"
    assert os.environ["PYTHONPATH"] == "parent-setting"
