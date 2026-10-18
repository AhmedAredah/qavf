"""Measurement semantics, ownership, and failure boundaries, without speed gates."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.performance import metrics


@pytest.mark.parametrize(
    ("contents", "expected"),
    [("Name:\tpython\nVmRSS:\t42 kB\n", 42 * 1024), ("Name: python\n", None)],
)
def test_rss_parser_units_and_absence(contents, expected):
    assert metrics.parse_rss_bytes(contents) == expected


@pytest.mark.parametrize("contents", ["VmRSS: 2 MB", "VmRSS: -1 kB", "VmRSS: bad kB"])
def test_rss_parser_rejects_uninterpretable_data(contents):
    with pytest.raises(ValueError):
        metrics.parse_rss_bytes(contents)


def test_io_parser_and_counter_resets_do_not_fabricate_zeroes():
    before = metrics.parse_io_counters(
        "read_bytes: 7\nwrite_bytes: 9\ncancelled_write_bytes: 10\nrchar: 100\n"
    )
    after = metrics.parse_io_counters(
        "read_bytes: 7\nwrite_bytes: 3\ncancelled_write_bytes: 25\nrchar: 120\nwchar: 6\n"
    )
    assert metrics.counter_deltas(before, after) == {
        "read_bytes": 0,
        "write_bytes": None,
        "cancelled_write_bytes": 15,
        "rchar": 20,
        "wchar": None,
    }
    with pytest.raises(ValueError):
        metrics.parse_io_counters("read_bytes: -1")


@pytest.mark.parametrize(
    ("platform", "expected"), [("linux", 7168), ("darwin", 7), ("win32", None)]
)
def test_high_water_is_explicitly_platform_normalized(monkeypatch, platform, expected):
    monkeypatch.setattr(metrics.sys, "platform", platform)
    monkeypatch.setattr(
        metrics,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _who: SimpleNamespace(ru_maxrss=7)),
    )
    assert metrics.high_water_bytes() == expected


def test_missing_capabilities_produce_null(monkeypatch):
    monkeypatch.setattr(metrics.sys, "platform", "win32")
    monkeypatch.setattr(metrics, "resource", None)
    assert metrics.read_rss_bytes() is None
    assert metrics.high_water_bytes() is None
    assert metrics.read_io_counters() == dict.fromkeys(metrics.IO_FIELDS)


def test_disk_scope_counts_each_distinct_file_once(tmp_path):
    # Windows scandir reports zero inode numbers; distinct files must not collapse.
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 3)
    (tmp_path / "nested" / "b.bin").write_bytes(b"x" * 4)
    (tmp_path / "nested" / "alias.bin").hardlink_to(tmp_path / "a.bin")
    result = metrics.disk_footprint(tmp_path)
    assert result.logical_bytes == 7
    assert result.complete


def test_disk_scope_excludes_fixture_symlinks_and_duplicate_hardlinks(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "large.bin").write_bytes(b"x" * 10000)
    owned = tmp_path / "owned"
    owned.mkdir()
    output = owned / "output.bin"
    output.write_bytes(b"small")
    try:
        (owned / "alias.bin").hardlink_to(output)
        (owned / "fixture-link").symlink_to(fixtures, target_is_directory=True)
        (owned / "file-link").symlink_to(fixtures / "large.bin")
    except (OSError, NotImplementedError):
        pytest.skip("Hard links or symbolic links unavailable in this test environment")
    result = metrics.disk_footprint(owned)
    assert result.logical_bytes == 5
    assert result.complete
    if hasattr(output.stat(), "st_blocks"):
        assert result.allocated_bytes == output.stat().st_blocks * 512
    else:
        assert result.allocated_bytes is None
    assert metrics.disk_footprint(owned / "fixture-link").logical_bytes is None


def test_missing_disk_scope_is_unavailable(tmp_path):
    assert metrics.disk_footprint(tmp_path / "absent") == metrics.DiskFootprint(None, None, False)


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_context_stops_monitor_and_preserves_operation_exceptions(tmp_path, error_type):
    meter = metrics.OperationMetrics(tmp_path, interval_s=100)
    with pytest.raises(error_type), meter:
        (tmp_path / "output").write_bytes(b"abc")
        raise error_type("operation failed")
    assert meter._thread is not None and not meter._thread.is_alive()
    result = meter.result()
    assert result["disk"]["start_logical_bytes"] == 0
    assert result["disk"]["final_logical_bytes"] == 3
    assert result["disk"]["sampled_peak_logical_bytes"] == 3
    assert result["sampling"]["sample_count"] == 2
    assert result["rss"]["peak_is_lower_bound"]
    assert (tmp_path / "output").read_bytes() == b"abc"
    json.dumps(result, allow_nan=False)


def test_result_requires_completed_context_and_meter_is_single_use(tmp_path):
    meter = metrics.OperationMetrics(tmp_path)
    with pytest.raises(RuntimeError):
        meter.result()
    with meter, pytest.raises(RuntimeError):
        meter.result()
    with pytest.raises(RuntimeError), meter:
        pass


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_bad_sampling_interval_rejected(tmp_path, interval):
    with pytest.raises(ValueError):
        metrics.OperationMetrics(tmp_path, interval_s=interval)


def test_unavailable_supported_probes_are_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics.sys, "platform", "linux")
    monkeypatch.setattr(metrics, "read_rss_bytes", lambda: None)
    monkeypatch.setattr(metrics, "high_water_bytes", lambda: None)
    monkeypatch.setattr(metrics, "read_io_counters", lambda: dict.fromkeys(metrics.IO_FIELDS))
    with metrics.OperationMetrics(tmp_path, interval_s=100) as meter:
        pass
    result = meter.result()
    assert result["rss"]["sampled_peak_bytes"] is None
    assert result["rss"]["lifetime_high_water_end_bytes"] is None
    assert result["sampling"]["sample_failures"] == 2
    assert all(result["io"][name] is None for name in metrics.IO_FIELDS)


def test_final_probe_failure_is_exposed_without_masking_operation(monkeypatch, tmp_path):
    meter = metrics.OperationMetrics(tmp_path, interval_s=100)

    def fail_probe():
        raise OSError("observer failed")

    with pytest.raises(ValueError, match="operation failure"), meter:
        monkeypatch.setattr(meter, "_sample", fail_probe)
        raise ValueError("operation failure")
    result = meter.result()
    assert result["sampling"]["monitor_error"] == "OSError: observer failed"
    assert result["sampling"]["sample_failures"] == 1
    assert result["disk"]["final_logical_bytes"] is None


def test_background_failure_is_exposed_and_thread_stopped(monkeypatch, tmp_path):
    observed = threading.Event()
    original = metrics.read_rss_bytes

    def fail_in_monitor():
        if threading.current_thread().name == "qavf-metrics":
            observed.set()
            raise RuntimeError("background observer failure")
        return original()

    monkeypatch.setattr(metrics, "read_rss_bytes", fail_in_monitor)
    with metrics.OperationMetrics(tmp_path, interval_s=0.001) as meter:
        assert observed.wait(timeout=5), "Monitor did not execute"
    assert meter._thread is not None and not meter._thread.is_alive()
    result = meter.result()
    assert result["sampling"]["sample_failures"] == 1
    assert result["sampling"]["monitor_error"] == "RuntimeError: background observer failure"


def test_disk_allocated_bytes_unavailable_when_platform_lacks_blocks(monkeypatch, tmp_path):
    original = Path.lstat

    def without_blocks(path, *args, **kwargs):
        metadata = original(path, *args, **kwargs)
        return SimpleNamespace(st_mode=metadata.st_mode)

    monkeypatch.setattr(Path, "lstat", without_blocks)
    assert metrics.disk_footprint(tmp_path).allocated_bytes is None
