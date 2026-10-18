"""Bounded parent supervision of one fresh, isolated validation process.

Linux and macOS own the worker through a new session's process group; Windows
owns it through a Job Object that the worker joins before doing any work.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qavf.engine.thread_budget import startup_threads
from qavf.execution import os_limits
from qavf.execution.errors import ExecutionLimitError
from qavf.execution.protocol import REQUEST_LIMIT
from qavf.redaction import redact_text

if TYPE_CHECKING:
    from qavf.execution._windows_job import Job
    from qavf.settings import QAVFSettings

_BOOTSTRAP_PATH = Path(__file__).with_name("bootstrap.py")
_PROC_ROOT = Path("/proc")
_REAP_TIMEOUT_S = 2.0
_CREATE_NO_WINDOW = 0x08000000
_CPU_POLL_S = 0.05
# macOS <sys/proc_info.h>: PROC_PGRP_ONLY, PROC_PIDTBSDINFO and the 136-byte
# struct proc_bsdinfo with pbi_status at offset 4 and pbi_pgid at offset 100.
_PROC_PGRP_ONLY = 2
_PROC_PIDTBSDINFO = 3
_BSDINFO_SIZE = 136
_SZOMB = 5
_ERROR_BYTES = 8192
_NATIVE_ENVIRONMENT = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _read_regular(path: Path, limit: int) -> bytes:
    """Read a bounded regular file without following a replaced result symlink."""
    flags = os.O_RDONLY
    for name in ("O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"):
        flags |= getattr(os, name, 0)
    # Windows has no O_NOFOLLOW; reject a link up front and require the opened
    # file to be the same one that was inspected.
    before = None if hasattr(os, "O_NOFOLLOW") else os.lstat(path)
    if before is not None and stat.S_ISLNK(before.st_mode):
        raise OSError(f"isolated result is a symbolic link: {path}")
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if before is not None and (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
            raise OSError(f"isolated result changed while it was opened: {path}")
        if not stat.S_ISREG(info.st_mode):
            raise ExecutionLimitError("isolated execution produced a nonregular result file")
        if info.st_size > limit:
            raise ExecutionLimitError(f"isolated result exceeds its {limit} byte transport limit")
        result = handle.read(limit + 1)
        if len(result) > limit:
            raise ExecutionLimitError(f"isolated result exceeds its {limit} byte transport limit")
        return result


def _failure(directory: Path, returncode: int) -> str:
    prefix = f"isolated execution failed with exit code {returncode}"
    if returncode < 0:
        with contextlib.suppress(ValueError):
            prefix += f" ({signal.Signals(-returncode).name})"
    elif sys.platform == "win32":
        from qavf.execution._windows_job import CPU_LIMIT_EXIT_CODE

        if returncode == CPU_LIMIT_EXIT_CODE:
            prefix += " (Windows job CPU-time limit)"
        elif returncode >= 0xC0000000:
            prefix += f" (status 0x{returncode:08X})"
    try:
        detail = json.loads(_read_regular(directory / "error.json", _ERROR_BYTES))
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            prefix += f": {redact_text(detail['message'])}"
    except (OSError, ValueError, ExecutionLimitError):
        pass
    return prefix + "; no assessment result was accepted"


def _stop(process: subprocess.Popen[bytes], job: Job | None = None) -> None:
    """Kill the owned process group or job, including children after leader success."""
    deadline = time.monotonic() + _REAP_TIMEOUT_S
    if sys.platform == "win32":
        if job is not None:
            try:
                job.terminate()
            except OSError as exc:
                raise ExecutionLimitError("could not terminate the isolated job") from exc
        # A worker that failed before joining its job is still the direct child.
        with contextlib.suppress(OSError):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            # macOS refuses group signals while the group holds a zombie, for
            # example an exited, unreaped leader; members are killed below.
            if sys.platform != "darwin":
                raise ExecutionLimitError(
                    "could not terminate the isolated process group"
                ) from exc
        except OSError as exc:
            raise ExecutionLimitError("could not terminate the isolated process group") from exc
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise ExecutionLimitError(
            "isolated process did not terminate within the cleanup tolerance"
        ) from exc
    # wait() reaps only the leader. Do not remove the directory while an owned
    # descendant still executes. Zombies have no open files or running work.
    while _members_running(process.pid, job):
        if time.monotonic() >= deadline:
            raise ExecutionLimitError(
                "isolated descendants did not stop within the cleanup tolerance"
            )
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


def _members_running(group: int, job: Job | None) -> bool:
    if sys.platform == "win32":
        if job is None:
            return False
        try:
            return job.active_processes() > 0
        except OSError as exc:
            raise ExecutionLimitError("isolated job membership could not be verified") from exc
    if sys.platform == "darwin":
        live = _darwin_live_members(group)
        for pid in live:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        return bool(live)
    return _group_running(group)


@cache
def _libproc() -> Any:
    import ctypes

    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_listpids.restype = ctypes.c_int
    library.proc_listpids.argtypes = [
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int
    ]
    library.proc_pidinfo.restype = ctypes.c_int
    library.proc_pidinfo.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
    ]
    return library


def _darwin_live_members(group: int) -> list[int]:
    """Return non-zombie members of a macOS process group, which has no procfs."""
    import ctypes

    try:
        library = _libproc()
        size = library.proc_listpids(_PROC_PGRP_ONLY, group, None, 0)
        if size < 0:
            raise OSError(ctypes.get_errno(), "proc_listpids failed")
        width = ctypes.sizeof(ctypes.c_int)
        # Leave room for members that start between the two calls.
        pids = (ctypes.c_int * (size // width + 64))()
        size = library.proc_listpids(_PROC_PGRP_ONLY, group, pids, ctypes.sizeof(pids))
        if size < 0:
            raise OSError(ctypes.get_errno(), "proc_listpids failed")
    except OSError as exc:
        raise ExecutionLimitError("process group membership could not be verified") from exc
    live = []
    info = ctypes.create_string_buffer(_BSDINFO_SIZE)
    for pid in pids[: size // width]:
        if pid <= 0:
            continue
        if library.proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, info, _BSDINFO_SIZE) != _BSDINFO_SIZE:
            continue  # Exited, or a zombie without task information.
        status = struct.unpack_from("I", info, 4)[0]
        pgid = struct.unpack_from("I", info, 100)[0]
        if pgid == group and status != _SZOMB:
            live.append(pid)
    return live


def _wait(
    process: subprocess.Popen[bytes], job: Job | None, started: float, timeout: float | None
) -> int:
    """Wait for the worker, polling a Windows job's CPU allowance meanwhile."""
    while True:
        remaining = (
            max(0.0, timeout - (time.monotonic() - started)) if timeout is not None else None
        )
        if job is None or not job.limits_cpu:
            return process.wait(timeout=remaining)
        try:
            return process.wait(
                timeout=_CPU_POLL_S if remaining is None else min(remaining, _CPU_POLL_S)
            )
        except subprocess.TimeoutExpired:
            if remaining is not None and remaining <= _CPU_POLL_S:
                raise
        try:
            job.enforce_cpu()
        except OSError as exc:
            raise ExecutionLimitError("could not enforce the isolated CPU allowance") from exc


def _interpreter() -> str:
    """A Windows venv python.exe is a launcher that starts the real interpreter as
    its own child. Run the base interpreter directly so the supervised process is
    the worker; the control file restores the caller's complete ``sys.path``.
    """
    if sys.platform == "win32":
        base = getattr(sys, "_base_executable", None)
        if base and os.path.isfile(base):
            return str(base)
    return sys.executable


def _remove(directory: Path) -> None:
    """Remove run storage, tolerating brief Windows scanner/indexer file handles."""
    deadline = time.monotonic() + (_REAP_TIMEOUT_S if sys.platform == "win32" else 0.0)
    while True:
        try:
            shutil.rmtree(directory)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _group_exists(group: int) -> bool:
    if sys.platform == "win32":
        raise ExecutionLimitError("process groups are a POSIX mechanism")
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise ExecutionLimitError("could not inspect the isolated process group") from exc
    return True


def _group_running(group: int) -> bool:
    """Inspect namespace-local group IDs even when procfs exposes host PIDs.

    A mounted host ``/proc`` can have unrelated entries whose numeric PID/group
    equals our child ID. Namespace identity plus the final NSpgid component is
    required; ``/proc/<child pid>/stat`` alone is not a valid ownership test.
    """
    if not _group_exists(group):
        return False
    observed = False
    try:
        namespace = os.readlink(_PROC_ROOT / "self/ns/pid")
        for entry in _PROC_ROOT.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                if os.readlink(entry / "ns/pid") != namespace:
                    continue
                fields = dict(
                    line.split(":", 1)
                    for line in (entry / "status").read_text().splitlines()
                    if ":" in line
                )
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            groups = fields.get("NSpgid", "").split()
            states = fields.get("State", "").split()
            if not groups or not states:
                raise ExecutionLimitError("process namespace membership could not be verified")
            if int(groups[-1]) == group:
                observed = True
                if states[0] not in {"Z", "X"}:
                    return True
    except (OSError, ValueError) as exc:
        raise ExecutionLimitError("process namespace membership could not be verified") from exc
    if observed or not _group_exists(group):
        return False
    raise ExecutionLimitError(
        "isolated process group still exists but its members are not visible in procfs"
    )


def execute(
    settings: QAVFSettings,
    request: bytes,
    *,
    operation: str = "validate",
) -> tuple[bytes, dict[str, object]]:
    """Return a complete bounded result after successful child exit and cleanup.

    The deadline covers process startup through child shutdown. Request preparation
    and parent result decoding are outside it. Process-stop polling allows two
    additional seconds; process-table inspection, filesystem cleanup and OS
    scheduling are outside that allowance, so it is not a total return-time cap.
    Limits apply to trusted code, not a hostile process escaping its group.
    """
    if not isinstance(request, bytes):
        raise TypeError("isolated request must be encoded bytes")
    if len(request) > REQUEST_LIMIT:
        raise ExecutionLimitError("isolated request exceeds the 16 MiB transport limit")
    engine = settings.engine
    limits = {
        "address_space_bytes": engine.process_address_space_bytes,
        "cpu_time_s": engine.process_cpu_time_s,
        "file_bytes": engine.process_file_bytes,
    }
    reason = os_limits.unsupported(limits)
    if reason is not None:
        raise ExecutionLimitError(reason)
    parent = engine.temp_directory
    if parent is not None:
        parent = settings.resolve(parent)
        parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="qavf-isolated-", dir=parent))
    process: subprocess.Popen[bytes] | None = None
    job: Job | None = None
    stopped = False
    try:
        request_path = directory / "request.json"
        result_path = directory / "result.json"
        request_path.write_bytes(request)
        native_threads = startup_threads(engine.native_threads)
        native_environment = {key: str(native_threads) for key in _NATIVE_ENVIRONMENT}
        control = {
            "request_path": str(request_path),
            "result_path": str(result_path),
            "error_path": str(directory / "error.json"),
            "operation": operation,
            "limits": limits,
            "max_report_bytes": engine.max_report_bytes,
            "native_environment": native_environment,
            "effective_native_threads": native_threads,
            "sys_path": [str(Path(path or os.getcwd()).absolute()) for path in sys.path],
            "run_directory": str(directory),
            "deadline_s": engine.run_timeout_s,
            "run_timeout_s": engine.run_timeout_s,
        }
        launch: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32":
            from qavf.execution._windows_job import Job

            job = Job(
                f"Local\\qavf-isolated-{uuid.uuid4().hex}",
                memory_bytes=engine.process_address_space_bytes,
                cpu_seconds=engine.process_cpu_time_s,
            )
            control["job"] = job.name
            launch = {"creationflags": _CREATE_NO_WINDOW}
        control_path = directory / "control.json"
        control_path.write_text(json.dumps(control), encoding="utf-8")
        started = time.monotonic()
        process = subprocess.Popen(
            [_interpreter(), "-S", str(_BOOTSTRAP_PATH), str(control_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, **native_environment, "PYTHONOPTIMIZE": "0"},
            **launch,
        )
        try:
            returncode = _wait(process, job, started, engine.run_timeout_s)
            if (
                engine.run_timeout_s is not None
                and time.monotonic() - started > engine.run_timeout_s
            ):
                raise subprocess.TimeoutExpired(process.args, engine.run_timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ExecutionLimitError(
                f"isolated execution exceeded its {engine.run_timeout_s:g}s wall-clock limit; "
                "no assessment result was accepted"
            ) from exc
        _stop(process, job)
        stopped = True
        if returncode != 0:
            raise ExecutionLimitError(_failure(directory, returncode))
        try:
            result = _read_regular(result_path, engine.max_report_bytes)
        except OSError as exc:
            raise ExecutionLimitError(
                "isolated execution exited without a readable regular result"
            ) from exc
        if not result:
            raise ExecutionLimitError("isolated execution produced an empty result")
        telemetry: dict[str, object] = {
            "execution_mode": "isolated",
            "worker_pid": process.pid,
            "wall_time_s": time.monotonic() - started,
            "deadline_s": engine.run_timeout_s,
            "cleanup_tolerance_s": _REAP_TIMEOUT_S,
            "native_threads": native_threads,
        }
        return result, telemetry
    finally:
        # Never delete files a worker can still be using. If reaping fails, keep
        # the directory available for diagnosis rather than hiding the failure.
        try:
            if process is not None and not stopped:
                _stop(process, job)
        finally:
            if job is not None:
                job.close()
        try:
            _remove(directory)
        except OSError as exc:
            raise ExecutionLimitError("could not clean isolated execution storage") from exc
