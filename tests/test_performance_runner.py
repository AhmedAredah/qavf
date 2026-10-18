"""Fresh-process acceptance cannot be inferred from exit code or partial JSON."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.performance import runner

CHILD = """
import argparse
import json
import pathlib
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--worker')
parser.add_argument('--result')
args = parser.parse_args()
request = json.loads(pathlib.Path(args.worker).read_text())
mode = request['case']
result = {'id': request['id'], 'status': 'passed', 'outcome': {'value': 42}}
if mode == 'timeout':
    time.sleep(60)
if mode == 'missing':
    sys.exit(0)
if mode == 'malformed':
    pathlib.Path(args.result).write_text('{')
    sys.exit(0)
if mode == 'wrong_id':
    result['id'] = 'another-request'
if mode == 'forged':
    result.update(source_label='forged', rows=100, process_elapsed_seconds=-1,
                  returncode=999, stderr_tail='forged')
if mode == 'failed':
    result.update(status='failed', error='Wrong findings')
if mode == 'failed_nonzero':
    result.update(status='failed', error={
        'phase': 'oracle', 'type': 'AssertionError', 'message': 'Unexpected findings'
    })
if mode == 'unknown_status':
    result['status'] = 'success'
if mode == 'list':
    result = []
if mode == 'large_logs':
    sys.stderr.write('x' * 100000 + 'STDERR_END')
    sys.stdout.write('y' * 100000 + 'STDOUT_END')
pathlib.Path(args.result).write_text(json.dumps(result))
if mode == 'crash':
    sys.exit(9)
if mode == 'failed_nonzero':
    sys.exit(1)
"""


@pytest.fixture
def run_child(tmp_path):
    child_path = tmp_path / "child.py"
    child_path.write_text(CHILD)

    def execute(case, *, timeout_s=10):
        return runner.execute_worker(
            [sys.executable, str(child_path)],
            {"id": "request-1", "case": case, "source_label": "candidate", "rows": 7},
            tmp_path / "controller",
            timeout_s=timeout_s,
            environment=dict(os.environ),
        )

    return execute


def test_valid_worker_record_and_controller_paths(run_child):
    result = run_child("valid")
    assert result["status"] == "passed"
    assert result["returncode"] == 0
    assert result["outcome"] == {"value": 42}
    assert result["error"] is None
    assert Path(result["request_path"]).is_file()
    assert Path(result["result_path"]).is_file()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["missing", "malformed", "wrong_id", "unknown_status", "list"])
def test_successful_exit_without_valid_result_is_crashed(run_child, case):
    result = run_child(case)
    assert result["status"] == "crashed"
    assert result["returncode"] == 0
    assert result["error_kind"] == "invalid_result"


def test_nonzero_exit_overrides_forged_success_result(run_child):
    result = run_child("crash")
    assert result["status"] == "crashed"
    assert result["returncode"] == 9
    assert result["error_kind"] == "nonzero_exit"


def test_worker_cannot_overwrite_request_identity_or_process_facts(run_child):
    result = run_child("forged")
    assert result["status"] == "passed"
    assert result["source_label"] == "candidate"
    assert result["rows"] == 7
    assert result["returncode"] == 0
    assert result["process_elapsed_seconds"] >= 0
    assert result["stderr_tail"] == ""


def test_worker_validation_failure_remains_failed(run_child):
    result = run_child("failed")
    assert result["status"] == "failed"
    assert result["error_kind"] == "worker_failed"
    assert result["error"] == "Wrong findings"


def test_nonzero_failure_preserves_structured_worker_error(run_child):
    result = run_child("failed_nonzero")
    assert result["status"] == "failed"
    assert result["returncode"] == 1
    assert result["error_kind"] == "worker_failed"
    assert result["worker_error"] == {
        "phase": "oracle",
        "type": "AssertionError",
        "message": "Unexpected findings",
    }


def test_timeout_is_explicit_and_process_is_reaped(run_child):
    result = run_child("timeout", timeout_s=0.05)
    assert result["status"] == "timed_out"
    assert result["error_kind"] == "timeout"
    assert result["returncode"] is not None


def test_log_tails_bounded_without_truncating_retained_logs(run_child):
    result = run_child("large_logs")
    assert result["status"] == "passed"
    for kind in ("stdout", "stderr"):
        assert len(result[f"{kind}_tail"]) <= runner.TAIL_BYTES
        assert result[f"{kind}_tail"].endswith(f"{kind.upper()}_END")
        assert Path(result[f"{kind}_path"]).stat().st_size > runner.TAIL_BYTES


def test_stale_result_is_removed_before_launch(run_child):
    assert run_child("valid")["status"] == "passed"
    assert run_child("missing")["status"] == "crashed"


def test_missing_executable_is_reported_without_claiming_result(tmp_path):
    result = runner.execute_worker(
        [str(tmp_path / "missing-python")],
        {"id": "missing"},
        tmp_path / "run",
        timeout_s=1,
        environment=dict(os.environ),
    )
    assert result["status"] == "crashed"
    assert result["error_kind"] == "launch_failed"
    assert result["returncode"] is None


def test_cancellation_stops_owned_process_and_propagates(monkeypatch, tmp_path):
    stopped = []

    def interrupt_wait(**kwargs):
        raise KeyboardInterrupt

    process = SimpleNamespace(wait=interrupt_wait)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(runner, "_stop_process", stopped.append)
    with pytest.raises(KeyboardInterrupt):
        runner.execute_worker(
            ["python", "driver.py"],
            {"id": "cancelled"},
            tmp_path / "run",
            timeout_s=1,
            environment={},
        )
    assert stopped == [process]


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_stop_targets_platform_owner_and_waits(monkeypatch, platform):
    events = []
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        runner, "os", SimpleNamespace(killpg=lambda pid, sig: events.append((pid, sig)))
    )
    monkeypatch.setattr(runner, "signal", SimpleNamespace(SIGKILL=9))
    process = SimpleNamespace(
        pid=123, wait=lambda: events.append("wait"), kill=lambda: events.append("kill")
    )
    runner._stop_process(process)
    assert events == (["kill", "wait"] if platform == "win32" else [(123, 9), "wait"])


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_rejected_before_launch(tmp_path, timeout):
    with pytest.raises(ValueError):
        runner.execute_worker(
            ["python"], {"id": "bad"}, tmp_path, timeout_s=timeout, environment={}
        )


def test_nonfinite_result_rejected(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text('{"id":"x","status":"passed","value":NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        runner._read_result(result_path, "x")
