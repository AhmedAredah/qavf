"""Metadata-only estimates improve defaults without widening explicit caps."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qavf.engine.thread_budget import connection_threads
from qavf.resource_plan import ResourcePlan, _available_memory
from qavf.settings import QAVFSettings

_AMPLE_DISK = 512 * 1024**3


def _ample_disk(_directory: Path) -> int:
    """A volume with far more free space than any automatic spill plan asks for."""
    return _AMPLE_DISK


@pytest.fixture
def ample_disk(monkeypatch):
    """Plan against a fixed free-disk figure, not this machine's volume."""
    monkeypatch.setattr("qavf.resource_plan._free_disk_bytes", _ample_disk)


def configuration(**engine):
    return QAVFSettings(
        engine=engine,
        datasets={
            "d": {
                "schema": {
                    "columns": {f"k{i}": {"dtype": "int"} for i in range(14)},
                    "primary_key": [f"k{i}" for i in range(7)],
                }
            }
        },
    )


def test_wide_high_cardinality_keys_reduce_parallel_pressure(monkeypatch, ample_disk):
    monkeypatch.setattr("qavf.resource_plan._available_memory", lambda: 8 * 1024**3)
    frame = SimpleNamespace(num_rows=1_890_000, nbytes=14 * 8 * 1_890_000)
    settings = configuration(memory_budget_bytes=1024**3)
    plan = ResourcePlan.build(settings, {"d": frame}, requested_workers=8)
    assert plan.workers == 1
    assert plan.memory_bytes == 1024**3
    assert plan.estimated_key_state_bytes == 1_890_000 * (7 * 8 + 64)
    assert plan.spill_bytes > 1024**3


def test_explicit_spill_and_memory_caps_are_never_enlarged():
    frame = SimpleNamespace(num_rows=10_000_000, nbytes=1024**3)
    plan = ResourcePlan.build(
        configuration(memory_budget_bytes=64 * 1024**2, spill_max_bytes=0),
        {"d": frame},
        requested_workers=8,
    )
    assert plan.memory_bytes == 64 * 1024**2
    assert plan.spill_bytes == 0
    assert not plan.automatic_memory and not plan.automatic_spill


def test_tiny_file_defaults_leave_room_for_native_csv_buffers(tmp_path, monkeypatch):
    monkeypatch.setattr("qavf.resource_plan._available_memory", lambda: 8 * 1024**3)
    path = tmp_path / "tiny.csv"
    path.write_text("x\n7\n")
    plan = ResourcePlan.build(configuration(), {"d": path}, requested_workers=8)
    assert plan.input_bytes == path.stat().st_size
    assert plan.memory_bytes // 2 // plan.connection_slots >= 128 * 1024**2


def test_file_descriptor_does_not_need_optional_adapters(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"x\n7\n")
    plan = ResourcePlan.build(configuration(), {"d": SimpleNamespace(path=path)})
    assert plan.input_bytes == len("x\n7\n")


def test_file_source_mapping_and_custom_pathlike_use_metadata(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n7\n")

    class CustomPath:
        def __fspath__(self):
            return str(path)

    for source in ({"type": "file", "path": path}, CustomPath()):
        plan = ResourcePlan.build(configuration(), {"d": source})
        assert plan.input_bytes == path.stat().st_size


def test_known_frame_rows_size_keys_even_without_physical_byte_count(monkeypatch, ample_disk):
    monkeypatch.setattr("qavf.resource_plan._available_memory", lambda: 8 * 1024**3)
    plan = ResourcePlan.build(configuration(), {"d": SimpleNamespace(num_rows=10_000_000)})
    assert plan.input_bytes == 0  # Unknown byte size is not invented.
    assert plan.estimated_key_state_bytes == 10_000_000 * (7 * 8 + 64)
    assert plan.memory_bytes is not None and plan.spill_bytes > 1024**3


def test_unknown_runtime_source_does_not_measure_replaced_file(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("x\n7\n")
    settings = QAVFSettings(datasets={"d": {"path": str(path)}})
    plan = ResourcePlan.build(settings, {"d": SimpleNamespace(sql="SELECT 1 AS x")})
    assert plan.input_bytes == 0


def test_auto_plan_has_room_for_its_key_estimate_after_native_partition(monkeypatch):
    monkeypatch.setattr("qavf.resource_plan._available_memory", lambda: 16 * 1024**3)
    frame = SimpleNamespace(num_rows=1_890_000, nbytes=14 * 8 * 1_890_000)
    plan = ResourcePlan.build(configuration(), {"d": frame}, requested_workers=1)
    assert plan.memory_bytes // 2 // plan.connection_slots >= plan.estimated_key_state_bytes


def test_runner_builds_one_immutable_plan_and_context_reuses_it(monkeypatch):
    import polars as pl

    from qavf.engine.runner import QARunner

    original = ResourcePlan.build.__func__
    plans = []

    def record(cls, *args, **kwargs):
        plan = original(cls, *args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(ResourcePlan, "build", classmethod(record))
    report = QARunner(configuration()).run(
        frames={"d": pl.DataFrame({f"k{i}": [i] for i in range(14)})}
    )
    assert len(plans) == 1
    assert report.metadata.resource_usage["resource_plan"]["memory_bytes"] == plans[0].memory_bytes


def test_bound_plan_does_not_leak_into_nested_or_later_contexts():
    settings = configuration()
    frames = {"d": SimpleNamespace(num_rows=10, nbytes=100)}
    plan = ResourcePlan.build(settings, frames)
    with plan.bind(settings, frames):
        assert ResourcePlan.for_context(settings, frames) is plan
        assert ResourcePlan.for_context(configuration(), frames) is not plan
    assert ResourcePlan.for_context(settings, frames) is not plan


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc and cgroup files")
def test_cgroup_reclaimable_cache_is_not_treated_as_live_allocations(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 16000000 kB\n"
        if str(path) == "/sys/fs/cgroup/memory.max":
            return str(8 * 1024**3)
        if str(path) == "/sys/fs/cgroup/memory.current":
            return str(7 * 1024**3)
        if str(path) == "/sys/fs/cgroup/memory.stat":
            return f"inactive_file {5 * 1024**3}\n"
        if str(path).startswith("/sys/fs/cgroup/"):
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert _available_memory() == 6 * 1024**3


def test_automatic_native_threads_follow_each_handle_memory_and_explicit_override():
    assert connection_threads(None, 1, cpu_count=16, memory_bytes=128 * 1024**2) == 1
    assert connection_threads(None, 1, cpu_count=16, memory_bytes=512 * 1024**2) == 4
    assert connection_threads(8, 1, cpu_count=16, memory_bytes=128 * 1024**2) == 8
    assert connection_threads(None, 4, cpu_count=16) == 4


def test_real_csv_keeps_reader_headroom_after_worker_reduction(tmp_path, monkeypatch, ample_disk):
    from qavf import validate
    from qavf.engine.severity import Status

    monkeypatch.setattr("qavf.resource_plan._available_memory", lambda: 768 * 1024**2)
    monkeypatch.setattr("qavf.engine.thread_budget.os.cpu_count", lambda: 16)
    path = tmp_path / "wide.csv"
    columns = [f"k{i}" for i in range(14)]
    with path.open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for index in range(4000):
            handle.write(",".join(str(index // 2) for _ in columns) + "\n")
    report = validate(
        path,
        schema={
            "columns": {c: {"dtype": "int", "ge": 0} for c in columns},
            "primary_key": columns[:7],
        },
        backend="polars",
    )
    assert not any(result.status == Status.ERROR for result in report.results), [
        result.message for result in report.results if result.status == Status.ERROR
    ]
    assert report.metadata.resource_usage["duckdb_threads_per_connection"] == 1
    assert report.results_for("INP-004")[0].affected_count == 2000


def _unknown_memory() -> None:
    """A platform that reports no available memory."""


def test_unknown_memory_imposes_no_automatic_cap(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_bytes(b"x\n7\n")
    monkeypatch.setattr("qavf.resource_plan._available_memory", _unknown_memory)
    plan = ResourcePlan.build(configuration(), {"d": SimpleNamespace(path=path)})
    assert plan.memory_bytes is None and plan.automatic_memory


@pytest.mark.skipif(sys.platform != "win32", reason="Windows memory API")
def test_windows_reports_available_memory():
    available = _available_memory()
    assert available is not None and available > 0


def _eight_gib() -> int:
    return 8 * 1024**3


def test_macos_budget_uses_physical_memory(monkeypatch):
    monkeypatch.setattr("qavf.resource_plan.sys.platform", "darwin")
    monkeypatch.setattr("qavf.resource_plan._macos_physical_memory", _eight_gib)
    monkeypatch.setattr("qavf.resource_plan.Path.read_text", _missing_file)
    assert _available_memory() == _eight_gib()


def _missing_file(*_args, **_kwargs) -> str:
    raise FileNotFoundError


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS physical memory")
def test_macos_reports_physical_memory():
    from qavf.resource_plan import _macos_physical_memory

    memory = _macos_physical_memory()
    assert memory is not None and memory > 0


def _scarce_disk(_directory: Path) -> int:
    return 10 * 1024**2


def test_automatic_spill_is_capped_by_free_disk(monkeypatch):
    monkeypatch.setattr("qavf.resource_plan._available_memory", _eight_gib)
    monkeypatch.setattr("qavf.resource_plan._free_disk_bytes", _scarce_disk)
    plan = ResourcePlan.build(configuration(), {"d": SimpleNamespace(num_rows=10_000_000)})
    assert plan.automatic_spill
    assert plan.spill_bytes == _scarce_disk(Path()) * 4 // 5


def test_free_disk_probe_measures_nearest_existing_ancestor(tmp_path):
    from qavf.resource_plan import _free_disk_bytes

    assert _free_disk_bytes(tmp_path / "not" / "created") > 0
    assert not (tmp_path / "not").exists()


def test_plan_reads_its_limits_from_run_limits(ample_disk):
    from dataclasses import replace

    from qavf.run_limits import RunLimits

    settings = configuration()
    frames = {"d": SimpleNamespace(num_rows=10_000_000)}
    limits = RunLimits.of(settings)
    assert not limits.spill_given and RunLimits.of(configuration(spill_max_bytes=1)).spill_given
    given = replace(limits, spill_given=True, spill_max_bytes=123, memory_budget_bytes=1024**3)
    plan = ResourcePlan.build(settings, frames, limits=given)
    assert (plan.spill_bytes, plan.automatic_spill) == (123, False)
    assert (plan.memory_bytes, plan.automatic_memory) == (1024**3, False)
