"""Real subprocess fault boundaries and operating-system limit contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qavf.execution import os_limits, supervisor
from qavf.execution.errors import ExecutionLimitError
from qavf.settings import QAVFSettings

pytestmark = pytest.mark.skipif(
    sys.platform not in os_limits.SUPPORTED_PLATFORMS, reason="unsupported isolation platform"
)
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux OS enforcement")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX OS enforcement")
_SPEC = importlib.util.spec_from_file_location(
    "qavf_test_processes", Path(__file__).with_name("execution_support") / "processes.py"
)
assert _SPEC is not None and _SPEC.loader is not None
processes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(processes)


def _settings(tmp_path: Path, **overrides: object) -> QAVFSettings:
    return QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {
                "execution_mode": "isolated",
                "temp_directory": str(tmp_path / "owned"),
                "run_timeout_s": 5.0,
                "native_threads": 2,
                **overrides,
            },
        }
    )


def _worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """Use the real bootstrap with a minimal worker to isolate supervisor faults."""
    package = tmp_path / "worker" / "qavf" / "execution"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "worker.py").write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "def run(request_path, result_path, control):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    monkeypatch.syspath_prepend(str(package.parent.parent))


def _assert_clean(tmp_path: Path) -> None:
    assert list((tmp_path / "owned").iterdir()) == []


def _running(pid: int) -> bool:
    if sys.platform != "linux":
        return processes.running(pid)
    namespace = os.readlink("/proc/self/ns/pid")
    for entry in Path("/proc").iterdir():
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
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if fields.get("NSpid", "").split()[-1:] == [str(pid)]:
            return fields["State"].split()[0] not in {"Z", "X"}
    return False


def test_bootstrap_sets_limits_and_native_pools_before_worker_import(tmp_path, monkeypatch):
    _worker(
        tmp_path,
        monkeypatch,
        "Path(result_path).write_text(json.dumps({\n"
        "    'request': Path(request_path).read_text(),\n"
        "    'limits': control['effective_limits'],\n"
        "    'threads': os.environ['POLARS_MAX_THREADS'],\n"
        "    'openblas': os.environ['OPENBLAS_NUM_THREADS'],\n"
        "}))",
    )
    limits: dict[str, object] = {"process_cpu_time_s": 3}
    if sys.platform != "win32":
        limits["process_file_bytes"] = 4096
    if sys.platform != "darwin":
        limits["process_address_space_bytes"] = 4 << 30
    payload, telemetry = supervisor.execute(_settings(tmp_path, **limits), b'{"ok":true}')
    result = json.loads(payload)
    assert result["request"] == '{"ok":true}'
    assert result["threads"] == result["openblas"] == "2"
    assert result["limits"]["cpu_time_s"]["hard"] <= 3
    if sys.platform == "win32":
        assert "file_bytes" not in result["limits"]
    else:
        assert result["limits"]["file_bytes"]["hard"] <= 4096
    if sys.platform == "darwin":
        assert "address_space_bytes" not in result["limits"]
    else:
        assert 0 < result["limits"]["address_space_bytes"]["hard"] <= 4 << 30
    assert result["limits"]["core_bytes"] == {"soft": 0, "hard": 0}
    assert telemetry["execution_mode"] == "isolated"
    assert not _running(telemetry["worker_pid"])
    _assert_clean(tmp_path)


@pytest.mark.parametrize(
    ("platform", "limits", "message"),
    [
        ("darwin", {"process_address_space_bytes": 1 << 30}, "macOS does not enforce"),
        ("win32", {"process_file_bytes": 4096}, "Windows has no per-file size limit"),
    ],
)
def test_unenforceable_platform_limits_fail_before_creating_storage(
    tmp_path, monkeypatch, platform, limits, message
):
    settings = _settings(tmp_path, **limits)
    monkeypatch.setattr(os_limits.sys, "platform", platform)
    with pytest.raises(ExecutionLimitError, match=message):
        supervisor.execute(settings, b"{}")
    assert not (tmp_path / "owned").exists()


def test_deadline_kills_sleeping_worker_and_repeated_runs_leave_no_storage(tmp_path, monkeypatch):
    _worker(tmp_path, monkeypatch, "time.sleep(30)")
    settings = _settings(tmp_path, run_timeout_s=0.2)
    started = time.monotonic()
    for _ in range(2):
        with pytest.raises(ExecutionLimitError, match="wall-clock limit"):
            supervisor.execute(settings, b"{}")
        _assert_clean(tmp_path)
    assert time.monotonic() - started < 4


def test_crash_never_accepts_existing_partial_result(tmp_path, monkeypatch):
    _worker(tmp_path, monkeypatch, "Path(result_path).write_text('{}')\nos._exit(17)")
    with pytest.raises(ExecutionLimitError, match="exit code 17"):
        supervisor.execute(_settings(tmp_path), b"{}")
    _assert_clean(tmp_path)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("pass", "readable regular result"),
        ("Path(result_path).write_bytes(b'')", "empty result"),
        ("Path(result_path).write_bytes(b'x' * 1025)", "transport limit"),
        pytest.param(
            "os.symlink(request_path, result_path)", "readable regular result", marks=posix_only
        ),
        pytest.param("os.mkfifo(result_path)", "nonregular result", marks=posix_only),
    ],
)
def test_incomplete_or_unbounded_results_fail_closed(tmp_path, monkeypatch, body, message):
    _worker(tmp_path, monkeypatch, body)
    with pytest.raises(ExecutionLimitError, match=message):
        supervisor.execute(_settings(tmp_path, max_report_bytes=1024), b"{}")
    _assert_clean(tmp_path)


@pytest.mark.parametrize("finish", ["Path(result_path).write_text('{}')", "time.sleep(30)"])
def test_descendants_stop_on_success_and_timeout(tmp_path, monkeypatch, finish):
    pid_path = tmp_path / "descendant.pid"
    _worker(
        tmp_path,
        monkeypatch,
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(pid_path)!r}).write_text(str(child.pid))\n"
        + finish,
    )
    settings = _settings(tmp_path, run_timeout_s=0.5)
    if "sleep" in finish:
        with pytest.raises(ExecutionLimitError, match="wall-clock limit"):
            supervisor.execute(settings, b"{}")
    else:
        assert supervisor.execute(settings, b"{}")[0] == b"{}"
    assert not _running(int(pid_path.read_text()))
    _assert_clean(tmp_path)


def test_os_cpu_limit_stops_busy_native_independent_work(tmp_path, monkeypatch):
    _worker(tmp_path, monkeypatch, "while True:\n    pass")
    started = time.monotonic()
    message = "CPU-time limit" if sys.platform == "win32" else r"SIGKILL|SIGXCPU"
    with pytest.raises(ExecutionLimitError, match=message):
        supervisor.execute(_settings(tmp_path, process_cpu_time_s=1), b"{}")
    assert time.monotonic() - started < 4
    _assert_clean(tmp_path)


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not enforce address space")
def test_os_address_space_rejects_large_allocation(tmp_path, monkeypatch):
    _worker(
        tmp_path, monkeypatch,
        "allocation = bytearray(256 * 1024 * 1024)\nPath(result_path).write_text('{}')",
    )
    with pytest.raises(ExecutionLimitError, match="MemoryError"):
        supervisor.execute(
            _settings(tmp_path, process_address_space_bytes=128 * 1024 * 1024), b"{}"
        )
    _assert_clean(tmp_path)


@posix_only
def test_os_file_size_rejects_oversized_worker_output(tmp_path, monkeypatch):
    _worker(tmp_path, monkeypatch, "Path(result_path).write_bytes(b'x' * 2048)")
    with pytest.raises(ExecutionLimitError, match=r"OSError|SIGXFSZ"):
        supervisor.execute(_settings(tmp_path, process_file_bytes=512), b"{}")
    _assert_clean(tmp_path)


@posix_only
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_inherited_limits_are_never_relaxed(monkeypatch, platform):
    import resource

    monkeypatch.setattr(os_limits.sys, "platform", platform)
    values = {
        resource.RLIMIT_AS: (90, 100),
        resource.RLIMIT_CPU: (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
        resource.RLIMIT_FSIZE: (50, 60),
        resource.RLIMIT_CORE: (10, 10),
    }
    monkeypatch.setattr(resource, "getrlimit", values.__getitem__)
    monkeypatch.setattr(resource, "setrlimit", values.__setitem__)
    requested = {"cpu_time_s": 5, "file_bytes": 40}
    expected = {
        "cpu_time_s": {"soft": 5, "hard": 5},
        "file_bytes": {"soft": 40, "hard": 40},
        "core_bytes": {"soft": 0, "hard": 0},
    }
    if platform == "linux":
        requested["address_space_bytes"] = 200
        expected["address_space_bytes"] = {"soft": 90, "hard": 100}
    assert os_limits.apply_limits(requested) == expected
    assert values[resource.RLIMIT_AS] == (90, 100)


def test_unsupported_platform_fails_before_creating_storage(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(supervisor.sys, "platform", "aix7")
    with pytest.raises(ExecutionLimitError, match="supports Linux, macOS and Windows"):
        supervisor.execute(settings, b"{}")
    assert not (tmp_path / "owned").exists()


def test_cleanup_failure_is_visible_instead_of_returning_success(tmp_path, monkeypatch):
    _worker(tmp_path, monkeypatch, "Path(result_path).write_text('{}')")
    original = supervisor.shutil.rmtree

    def fail_cleanup(_):
        raise OSError("cleanup failure")

    monkeypatch.setattr(supervisor.shutil, "rmtree", fail_cleanup)
    try:
        with pytest.raises(ExecutionLimitError, match="clean isolated execution storage"):
            supervisor.execute(_settings(tmp_path), b"{}")
    finally:
        for path in (tmp_path / "owned").iterdir():
            original(path)


def test_deadline_includes_worker_shutdown(tmp_path, monkeypatch):
    _worker(
        tmp_path,
        monkeypatch,
        "import threading\n"
        "threading.Thread(target=time.sleep, args=(30,), daemon=False).start()\n"
        "Path(result_path).write_text('{}')",
    )
    with pytest.raises(ExecutionLimitError, match="wall-clock limit"):
        supervisor.execute(_settings(tmp_path, run_timeout_s=0.2), b"{}")
    _assert_clean(tmp_path)


def test_relative_storage_resolves_against_configuration(tmp_path, monkeypatch):
    _worker(tmp_path, monkeypatch, "Path(result_path).write_text('{}')")
    settings = _settings(tmp_path, temp_directory="owned")
    settings.source_path = tmp_path / "qavf.yaml"
    assert supervisor.execute(settings, b"{}")[0] == b"{}"
    _assert_clean(tmp_path)


def test_request_cap_rejects_before_creating_storage(tmp_path):
    with pytest.raises(ExecutionLimitError, match="16 MiB transport limit"):
        supervisor.execute(_settings(tmp_path), b"x" * (supervisor.REQUEST_LIMIT + 1))
    assert not (tmp_path / "owned").exists()


def _proc_member(root: Path, host_pid: int | str, namespace: str, group: str, state: str):
    entry = root / str(host_pid)
    (entry / "ns").mkdir(parents=True)
    (entry / "ns/pid").symlink_to(namespace)
    (entry / "status").write_text(f"State:\t{state}\nNSpgid:\t{group}\n")


def _proc_fixture(tmp_path, monkeypatch):
    root = tmp_path / "proc"
    _proc_member(root, "self", "pid:[42]", "9000 1", "R (running)")
    monkeypatch.setattr(supervisor, "_PROC_ROOT", root)
    monkeypatch.setattr(supervisor, "_group_exists", lambda _: True)
    return root


@linux_only
def test_group_probe_uses_namespace_local_nspgid_not_host_proc_number(tmp_path, monkeypatch):
    root = _proc_fixture(tmp_path, monkeypatch)
    _proc_member(root, 9010, "pid:[42]", "9020 7", "S (sleeping)")
    assert supervisor._group_running(7)


@linux_only
def test_group_probe_ignores_colliding_ids_in_other_pid_namespaces(tmp_path, monkeypatch):
    root = _proc_fixture(tmp_path, monkeypatch)
    _proc_member(root, 7, "pid:[99]", "7", "R (running)")
    _proc_member(root, 9010, "pid:[99]", "9010 7", "S (sleeping)")
    _proc_member(root, 9020, "pid:[42]", "9020 7", "Z (zombie)")
    assert not supervisor._group_running(7)


@linux_only
@pytest.mark.parametrize("state", ["Z (zombie)", "X (dead)"])
def test_group_probe_accepts_only_stopped_namespace_members(tmp_path, monkeypatch, state):
    root = _proc_fixture(tmp_path, monkeypatch)
    _proc_member(root, 9010, "pid:[42]", "9010 7", state)
    assert not supervisor._group_running(7)


@linux_only
def test_group_probe_fails_closed_when_existing_group_is_not_visible(tmp_path, monkeypatch):
    root = _proc_fixture(tmp_path, monkeypatch)
    _proc_member(root, 7, "pid:[99]", "7", "Z (zombie)")
    with pytest.raises(ExecutionLimitError, match="not visible in procfs"):
        supervisor._group_running(7)


@linux_only
def test_group_probe_fails_closed_without_namespace_identity(tmp_path, monkeypatch):
    root = _proc_fixture(tmp_path, monkeypatch)
    (root / "self/ns/pid").unlink()
    with pytest.raises(ExecutionLimitError, match="namespace membership"):
        supervisor._group_running(7)


@linux_only
def test_group_absence_uses_kernel_probe_without_procfs(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "_PROC_ROOT", tmp_path / "missing")

    def missing_group(group, sig):
        assert (group, sig) == (7, 0)
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", missing_group)
    assert not supervisor._group_running(7)


@linux_only
def test_group_exit_during_scan_is_safe(tmp_path, monkeypatch):
    _proc_fixture(tmp_path, monkeypatch)
    existence = iter([True, False])
    monkeypatch.setattr(supervisor, "_group_exists", lambda _: next(existence))
    assert not supervisor._group_running(7)


def test_early_platform_wakeup_never_shortens_the_deadline():
    # Windows rounds wait timeouts down to whole milliseconds and can wake early.
    class EarlyWaker:
        args = ("worker",)
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            raise subprocess.TimeoutExpired(self.args, timeout)

    process = EarlyWaker()
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        supervisor._wait(process, None, started, 0.2)
    assert time.monotonic() - started >= 0.2
    assert process.waits > 1
