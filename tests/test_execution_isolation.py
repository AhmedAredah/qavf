"""Independent public execution-limit contracts using actual child processes."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

import qavf
from qavf import Dataset, File, Loader, Query, ResourceLimits, Status, validate
from qavf.engine.registry import RuleRegistry
from qavf.engine.runner import QARunner
from qavf.execution.os_limits import SUPPORTED_PLATFORMS
from qavf.reports.models import QAReport
from qavf.settings import QAVFSettings

SUPPORT = Path(__file__).with_name("execution_support")
SCHEMA = {"columns": {"x": {"ge": 0}}}
BASE = {"plugins": {"entry_points": False}, "engine": {"parallel": False}}
_SPEC = importlib.util.spec_from_file_location("qavf_test_processes", SUPPORT / "processes.py")
assert _SPEC is not None and _SPEC.loader is not None
processes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(processes)
_running = processes.running
# Controls each platform cannot enforce; isolation must reject them before launch.
UNENFORCEABLE = {"darwin": {"address_space"}, "win32": {"file_size"}}.get(sys.platform, set())


def _supported_or_rejected() -> bool:
    if sys.platform in SUPPORTED_PLATFORMS:
        return True
    with pytest.raises((RuntimeError, ValueError), match=r"Linux|linux|support"):
        validate(
            "unused.parquet", schema=SCHEMA, config=BASE, resources=ResourceLimits(wall_time=1)
        )
    return False


def _observed_pids(directory: Path) -> list[int]:
    result = []
    for name in ("worker.json", "descendant.json"):
        path = directory / name
        if path.exists():
            result.append(json.loads(path.read_text(encoding="utf-8"))["pid"])
    return result


def _config(directory: Path, mode: str) -> dict:
    return {
        "plugins": {
            "packages": [],
            "paths": [str(SUPPORT / "probes.py")],
            "entry_points": False,
        },
        "engine": {"parallel": False, "native_threads": 1},
        "datasets": {"unused": {"source": {"type": "query", "sql": "SELECT 1 AS x"}}},
        "rules": {"ISOLATION-PROBE": {"params": {"mode": mode, "directory": str(directory)}}},
    }


def _probe(
    tmp_path: Path, mode: str, *, config: dict | None = None, **resources
) -> tuple[dict, QAReport | None]:
    observed = tmp_path / "observed"
    observed.mkdir()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    request = tmp_path / "request.json"
    output = tmp_path / "outcome.json"
    request.write_text(
        json.dumps(
            {
                "config": config if config is not None else _config(observed, mode),
                "resources": {"isolate": True, "temp_dir": str(temporary), **resources},
            }
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        PYTHONOPTIMIZE=str(sys.flags.optimize),
        PYTHONPATH=str(Path(qavf.__file__).resolve().parents[1]),
        POLARS_MAX_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                *(["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []),
                str(SUPPORT / "driver.py"),
                str(request),
                str(output),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.is_file(), result.stdout + result.stderr
        record = json.loads(output.read_text(encoding="utf-8"))
        assert record["driver_optimization"] == sys.flags.optimize
        assert record["limits_before"] == record["limits_after"]
        assert list(temporary.iterdir()) == [], "run-owned temporary storage was not reclaimed"
        deadline = time.monotonic() + 1
        while any(_running(pid) for pid in _observed_pids(observed)):
            if time.monotonic() >= deadline:
                pytest.fail("a worker or ordinary descendant remained active after the call")
            time.sleep(0.01)
        report_path = output.with_suffix(".report.json")
        report = (
            QAReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else None
        )
        return record, report
    finally:
        # Independent safety net. This cleanup never substitutes for assertions
        # that the execution owner itself terminated every observed process.
        for pid in _observed_pids(observed):
            if _running(pid):
                processes.kill(pid)


def _resource_rejected(record: dict, report: QAReport | None) -> None:
    if record["outcome"] == "error":
        assert record["error_type"] == "ExecutionLimitError", record
        assert report is None
    else:
        assert report is not None
        assert report.incomplete
        assert report.exit_code() == 1
        assert report.get("ISOLATION-PROBE").status is Status.ERROR


@pytest.mark.parametrize("inherited_optimization", ["0", "1", "2"])
def test_isolated_worker_explicitly_preserves_assertions(
    tmp_path, monkeypatch, inherited_optimization
):
    if not _supported_or_rejected():
        return
    monkeypatch.setenv("PYTHONOPTIMIZE", inherited_optimization)
    report = validate(
        config=_config(tmp_path / "observed", "pass"),
        resources=ResourceLimits(isolate=True, wall_time=15),
    )
    assert report.get("ISOLATION-PROBE").metrics["worker_optimization"] == 0


def test_isolated_success_keeps_exact_audit_values_and_parent_limits(tmp_path):
    if not _supported_or_rejected():
        return
    record, report = _probe(tmp_path, "pass", wall_time=10)
    assert record["outcome"] == "report", record
    assert report is not None and report.verdict() == "passed"
    result = report.get("ISOLATION-PROBE")
    assert result.total_records == 3 and result.affected_count == 0
    assert result.metrics["worker_pid"] != record["driver_pid"]
    assert result.metrics["decimal"] == Decimal("123456789.123456789123456789")
    assert result.metrics["bytes"] == b"\x00\xff\x80"
    assert result.metrics["date"] == date(2024, 2, 29)
    assert result.metrics["timestamp"] == datetime(2024, 2, 29, 12, 3, 4, 567890, tzinfo=UTC)
    assert result.metrics["duration"] == timedelta(days=-2, microseconds=123456)
    assert result.metrics["integer"] == 2**100 + 1
    assert result.metrics["nested"] == {"text": "\u03b1\x00\U0001f69a", "values": [True, None, 3]}


@pytest.mark.parametrize("mode", ["sleep", "thread_shutdown", "descendant_timeout"])
def test_wall_deadline_terminates_noncooperative_work_and_descendants(tmp_path, mode):
    if not _supported_or_rejected():
        return
    # Instrumented interpreter/plugin startup consumes this same whole-worker
    # budget. Allow it to enter the 90-second probe before enforcing the limit.
    deadline_s = 5
    record, report = _probe(tmp_path, mode, wall_time=deadline_s)
    assert record["outcome"] == "error", record
    assert record["error_type"] == "ExecutionLimitError"
    assert report is None
    assert (tmp_path / "observed/worker.json").exists(), "workload never entered its rule"
    assert deadline_s <= record["elapsed"] < 8
    if mode == "descendant_timeout":
        assert (tmp_path / "observed/descendant.json").exists()


def test_success_does_not_leave_an_ordinary_descendant_running(tmp_path):
    if not _supported_or_rejected():
        return
    record, report = _probe(tmp_path, "descendant_success", wall_time=10)
    assert record["outcome"] == "report", record
    assert report is not None and report.verdict() == "passed"
    assert (tmp_path / "observed/descendant.json").exists()


@pytest.mark.parametrize(
    ("mode", "resources"),
    [
        ("memory", {"address_space": "2GiB"}),
        ("cpu", {"cpu_time": 2}),
        ("file", {"file_size": "32KiB"}),
    ],
)
def test_os_limits_reject_real_excess_without_modifying_parent_limits(tmp_path, mode, resources):
    if not _supported_or_rejected():
        return
    record, report = _probe(tmp_path, mode, wall_time=12, **resources)
    if UNENFORCEABLE & resources.keys():
        assert record["outcome"] == "error", record
        assert record["error_type"] == "ExecutionLimitError"
        assert "omit" in record["message"]
        assert not (tmp_path / "observed/worker.json").exists(), "rejection followed launch"
        return
    assert (tmp_path / "observed/worker.json").exists(), "workload failed before the probe"
    _resource_rejected(record, report)
    if mode == "file":
        assert (tmp_path / "observed/oversized.bin").stat().st_size <= 32 << 10
    if mode == "cpu":
        assert record["elapsed"] < 10, "the wall deadline substituted for the CPU allowance"


def test_abrupt_worker_exit_never_returns_a_partial_report(tmp_path):
    if not _supported_or_rejected():
        return
    record, report = _probe(tmp_path, "exit", wall_time=10)
    assert record["outcome"] == "error", record
    assert record["error_type"] == "ExecutionLimitError"
    assert report is None


def test_wall_deadline_includes_plugin_import_before_rule_execution(tmp_path):
    if not _supported_or_rejected():
        return
    marker = tmp_path / "observed/worker.json"
    plugin = tmp_path / "slow_import.py"
    plugin.write_text(
        "import json, os, time\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'pid': os.getpid()}}))\n"
        "time.sleep(90)\n",
        encoding="utf-8",
    )
    config = _config(tmp_path / "observed", "pass")
    config["plugins"]["paths"] = [str(plugin)]
    record, report = _probe(tmp_path, "pass", config=config, wall_time=2)
    assert record["outcome"] == "error", record
    assert record["error_type"] == "ExecutionLimitError"
    assert report is None and marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["pid"] != record["driver_pid"]
    assert record["elapsed"] < 8


def test_report_budget_rejection_does_not_accept_truncated_success(tmp_path):
    if not _supported_or_rejected():
        return
    record, report = _probe(tmp_path, "pass", wall_time=10, report_bytes=128)
    assert record["outcome"] == "error", record
    assert record["error_type"] == "ExecutionLimitError"
    assert report is None


@pytest.mark.parametrize("kind", ["path", "file", "dataset"])
def test_public_file_inputs_preserve_complete_validation_in_isolation(tmp_path, kind):
    if not _supported_or_rejected():
        return
    path = tmp_path / "values.parquet"
    pl.DataFrame({"x": [-1, 0, 2]}).write_parquet(path)
    data = {"path": path, "file": File(path), "dataset": Dataset(File(path))}[kind]
    report = validate(data, schema=SCHEMA, config=BASE, resources=ResourceLimits(wall_time=10))
    result = report.get("INP-003")
    assert result.status is Status.FAIL and result.affected_count == 1
    assert not report.incomplete


def test_parameterized_query_input_keeps_decimal_values_through_isolation():
    if not _supported_or_rejected():
        return
    value = Decimal("-1.000000000000000001")
    report = validate(
        Query("SELECT $x AS x", parameters={"x": value}),
        schema=SCHEMA,
        config=BASE,
        resources=ResourceLimits(wall_time=10),
    )
    result = report.get("INP-003")
    assert result.status is Status.FAIL and result.affected_count == 1
    assert result.affected_records[0]["x"] == value
    assert not report.incomplete


@pytest.mark.parametrize("kind", ["loader", "frame", "lazy", "arrow"])
def test_live_inputs_rejected_before_loading_or_arrow_conversion(monkeypatch, kind):
    calls = []
    values = {
        "loader": Loader(lambda: calls.append("loaded")),
        "frame": pl.DataFrame({"x": [1]}),
        "lazy": pl.DataFrame({"x": [1]}).lazy(),
        "arrow": pa.table({"x": [1]}),
    }

    def forbidden(*args, **kwargs):
        calls.append("converted")
        raise AssertionError("unsupported input was converted")

    monkeypatch.setattr(pl, "from_arrow", forbidden)
    monkeypatch.setattr(pl.LazyFrame, "collect", forbidden)
    with pytest.raises(
        (TypeError, ValueError, RuntimeError), match=r"isolat|process|Linux|support"
    ):
        validate(values[kind], schema=SCHEMA, config=BASE, resources=ResourceLimits(isolate=True))
    assert calls == []


def test_custom_registry_rejected_before_plugin_discovery(tmp_path):
    plugin = tmp_path / "unreachable.py"
    plugin.write_text("raise AssertionError('plugin discovery must not run')\n", encoding="utf-8")
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"packages": [], "paths": [str(plugin)], "entry_points": False},
            "engine": {"execution_mode": "isolated"},
        }
    )
    with pytest.raises((TypeError, ValueError, RuntimeError), match=r"registr|isolat|support"):
        QARunner(settings, RuleRegistry()).run()


def test_standalone_regression_rejects_live_inputs_before_adaptation(monkeypatch):
    from qavf.regression import compare_outputs

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("inputs were adapted before policy rejection")

    monkeypatch.setattr(pl, "from_arrow", forbidden)
    table = pa.table({"id": [1], "x": [2]})
    with pytest.raises((TypeError, ValueError, RuntimeError), match=r"isolat|REG-001|support"):
        compare_outputs(table, table, keys="id", resources=ResourceLimits(wall_time=1))
    assert calls == []


def test_standalone_file_regression_preserves_exact_results_with_hard_policy(tmp_path):
    if not _supported_or_rejected():
        return
    from qavf.regression import compare_outputs

    baseline, candidate = tmp_path / "before.parquet", tmp_path / "after.parquet"
    pl.DataFrame({"id": [1, 2], "x": [2, 3]}).write_parquet(baseline)
    pl.DataFrame({"id": [2, 1], "x": [4, 2]}).write_parquet(candidate)
    comparison = compare_outputs(
        File(baseline),
        File(candidate),
        keys="id",
        rtol=0,
        atol=0,
        resources=ResourceLimits(wall_time=10),
    )
    assert not comparison.equal
    assert comparison.values_changed == 1


def test_cli_resource_failure_exits_two_without_publishing_success_report(tmp_path):
    if not _supported_or_rejected():
        return
    config = _config(tmp_path / "observed", "sleep")
    config["engine"].update(execution_mode="isolated", run_timeout_s=2)
    config["report"] = {"output_dir": str(tmp_path / "reports"), "html": False, "json": True}
    path = tmp_path / "qavf.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "qavf.cli", "run", "-c", str(path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    try:
        assert result.returncode == 2, result.stdout + result.stderr
        assert not (tmp_path / "reports/qa_report.json").exists()
        assert "QA PASS" not in result.stdout
        assert all(not _running(pid) for pid in _observed_pids(tmp_path / "observed"))
    finally:
        for pid in _observed_pids(tmp_path / "observed"):
            if _running(pid):
                processes.kill(pid)
