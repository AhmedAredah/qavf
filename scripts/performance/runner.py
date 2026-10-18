"""Fresh-process benchmark execution with bounded orchestration memory.

The controller owns request identity, process status, and log paths. Worker
stdout/stderr go directly to files; only bounded tails enter the final record.
On POSIX timeout/cancellation kills the owned process group. Other platforms
can terminate the direct child only using this standard-library implementation.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TAIL_BYTES = 8192
MAX_RESULT_BYTES = 16 * 1024 * 1024
WORKER_STATUSES = {"passed", "failed"}


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - TAIL_BYTES))
            return stream.read(TAIL_BYTES).decode("utf-8", errors="replace")
    except OSError as error:
        return f"Log unavailable: {type(error).__name__}: {error}"[:TAIL_BYTES]


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    """Reap the direct child and, on POSIX, stop its entire owned session group."""
    try:
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _read_result(path: Path, expected_id: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError(f"Worker result exceeds {MAX_RESULT_BYTES} bytes")
    with path.open(encoding="utf-8") as stream:
        record = json.load(stream, parse_constant=_reject_nonfinite)
    if not isinstance(record, dict):
        raise ValueError("Worker result must be a JSON object")
    if record.get("id") != expected_id:
        raise ValueError("Worker result ID does not match the request")
    if not isinstance(record.get("status"), str) or record["status"] not in WORKER_STATUSES:
        raise ValueError("Worker result has no valid passed/failed status")
    return record


def execute_worker(
    command: list[str],
    request: dict[str, Any],
    run_dir: Path,
    *,
    timeout_s: float,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Execute one request, preserving its identity even if the worker disagrees.

    ``command`` contains an interpreter and absolute driver path. ``environment``
    is the complete child environment. ``run_dir`` holds controller files; pass
    a distinct operation directory in the request for owned disk measurements.
    A missing/malformed/stale result or a nonzero exit can never count as passed.
    """
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("Worker command must be a nonempty list of nonempty strings")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Worker timeout must be positive and finite")
    request_id = request.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("Worker request requires a nonempty string ID")
    serialized_request = json.dumps(request, allow_nan=False, indent=2)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.json"
    result_path = run_dir / "result.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    result_path.unlink(missing_ok=True)
    request_path.write_text(serialized_request + "\n", encoding="utf-8")
    argv = [*command, "--worker", str(request_path), "--result", str(result_path)]
    returncode: int | None = None
    worker_result: dict[str, Any] = {}
    status = "crashed"
    error_kind: str | None = None
    error: str | None = None
    started = time.perf_counter()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                argv,
                cwd=run_dir,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=sys.platform != "win32",
            )
        except OSError as launch_error:
            error_kind = "launch_failed"
            error = f"{type(launch_error).__name__}: {launch_error}"
        else:
            try:
                returncode = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _stop_process(process)
                returncode = process.returncode
                status = "timed_out"
                error_kind = "timeout"
                error = f"Worker exceeded its {timeout_s:g}-second timeout"
            except BaseException:
                _stop_process(process)
                raise
    elapsed = time.perf_counter() - started
    if error_kind is None:
        try:
            worker_result = _read_result(result_path, request_id)
        except (OSError, ValueError) as result_error:
            error_kind = "invalid_result"
            error = f"{type(result_error).__name__}: {result_error}"
        if returncode != 0:
            if returncode == 1 and worker_result.get("status") == "failed":
                status = "failed"
                error_kind = "worker_failed"
                error = "Worker validation failed with exit code 1"
            else:
                error_kind = "nonzero_exit"
                error = f"Worker exited with code {returncode}"
        elif error_kind is None:
            status = worker_result["status"]
            if status == "failed":
                error_kind = "worker_failed"
                error = str(worker_result.get("error", "Worker validation failed"))
    # The request is authoritative for identity and configuration. The
    # controller is authoritative for process observations and acceptance.
    return {
        **worker_result,
        **request,
        "status": status,
        "error_kind": error_kind,
        "error": error,
        "worker_error": worker_result.get("error"),
        "returncode": returncode,
        "process_elapsed_seconds": elapsed,
        "timeout_seconds": timeout_s,
        "termination_method": "direct child"
        if sys.platform == "win32"
        else "POSIX process group SIGKILL",
        "request_path": str(request_path),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": _tail(stdout_path),
        "stderr_tail": _tail(stderr_path),
    }
