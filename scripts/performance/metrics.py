"""Bounded, dependency-free observations around one benchmark operation.

RSS and owned regular-file footprint peaks are sampled lower bounds, not exact
peaks. Linux process I/O counters include the monitor's own procfs reads; cached
reads need not produce ``read_bytes``. Lifetime RSS high-water marks include
imports and fixture preparation. No metric here claims Python-only allocations.

Sources: https://docs.python.org/3/library/time.html,
https://docs.python.org/3/library/resource.html,
https://man7.org/linux/man-pages/man5/proc_pid_io.5.html,
https://man7.org/linux/man-pages/man5/proc_pid_status.5.html.
"""

from __future__ import annotations

import math
import os
import stat
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from .process_tree import sample_tree

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None  # type: ignore[assignment]


IO_FIELDS = ("read_bytes", "write_bytes", "cancelled_write_bytes", "rchar", "wchar")


def parse_rss_bytes(status: str) -> int | None:
    """Read Linux VmRSS's explicitly documented KiB unit, without guessing."""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise ValueError("Unexpected VmRSS units or format")
            value = int(fields[1])
            if value < 0:
                raise ValueError("Negative VmRSS")
            return value * 1024
    return None


def parse_io_counters(contents: str) -> dict[str, int | None]:
    """Return unavailable counters as null rather than fabricated zeroes."""
    result: dict[str, int | None] = dict.fromkeys(IO_FIELDS)
    for line in contents.splitlines():
        name, separator, raw = line.partition(":")
        if separator and name in result:
            value = int(raw.strip())
            if value < 0:
                raise ValueError("Negative process I/O counter")
            result[name] = value
    return result


def counter_deltas(
    before: dict[str, int | None], after: dict[str, int | None]
) -> dict[str, int | None]:
    """Counter resets or missing endpoints invalidate that counter's delta."""
    result: dict[str, int | None] = {}
    for name in IO_FIELDS:
        start, end = before.get(name), after.get(name)
        result[name] = None if start is None or end is None or end < start else end - start
    return result


def read_rss_bytes() -> int | None:
    if sys.platform != "linux":
        return None
    try:
        return parse_rss_bytes(Path("/proc/self/status").read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None


def read_io_counters() -> dict[str, int | None]:
    if sys.platform != "linux":
        return dict.fromkeys(IO_FIELDS)
    try:
        return parse_io_counters(Path("/proc/self/io").read_text(encoding="ascii"))
    except (OSError, ValueError):
        return dict.fromkeys(IO_FIELDS)


def high_water_bytes() -> int | None:
    """Normalize only platforms with known ru_maxrss units; others are null."""
    if resource is None or sys.platform not in {"linux", "darwin"}:
        return None
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    return int(value) * (1024 if sys.platform == "linux" else 1)


@dataclass(frozen=True)
class DiskFootprint:
    logical_bytes: int | None
    allocated_bytes: int | None
    complete: bool


def _file_identity(entry: os.DirEntry[str], metadata: os.stat_result) -> tuple[object, ...]:
    """Hard-link identity. Windows scandir leaves st_dev/st_ino zero, so ask lstat;
    a filesystem without file IDs falls back to the path, counting each name once.
    """
    if metadata.st_ino == 0:
        metadata = os.lstat(entry.path)
        if metadata.st_ino == 0:
            return ("path", entry.path)
    return (metadata.st_dev, metadata.st_ino)


def disk_footprint(directory: Path) -> DiskFootprint:
    """Count distinct regular-file inodes beneath a caller-owned directory.

    Never follow symbolic links or read file contents. Allocation uses st_blocks
    in 512-byte units when exposed by the OS. Directory metadata is excluded.
    Concurrent deletion is accepted as a partial snapshot, explicitly marked.
    """
    logical = 0
    allocated: int | None
    complete = True
    visited: set[tuple[object, ...]] = set()
    pending = [directory]
    try:
        root_metadata = directory.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode):
            return DiskFootprint(None, None, False)
        allocated = 0 if hasattr(root_metadata, "st_blocks") else None
        while pending:
            current = pending.pop()
            try:
                if current.is_symlink():
                    complete = False
                    continue
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            metadata = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            complete = False
                            continue
                        if stat.S_ISDIR(metadata.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(metadata.st_mode):
                            identity = _file_identity(entry, metadata)
                            if identity in visited:
                                continue
                            visited.add(identity)
                            logical += metadata.st_size
                            blocks = getattr(metadata, "st_blocks", None)
                            if blocks is None:
                                allocated = None
                            elif allocated is not None:
                                allocated += blocks * 512
            except FileNotFoundError:
                complete = False
    except OSError:
        return DiskFootprint(None, None, False)
    return DiskFootprint(logical, allocated, complete)


def _maximum(previous: int | None, value: int | None) -> int | None:
    if value is None:
        return previous
    return value if previous is None else max(previous, value)


class OperationMetrics:
    """Measure once; keep only aggregate samples and join the monitor on exit.

    ``work_dir`` must contain only this operation's output and scratch files,
    never shared input fixtures. The monitor adds work: CPU and I/O include that
    interference. Probe wall/thread-CPU totals cover initial, background, and
    final observations, overlap the operation, and must not be subtracted as a
    benchmark correction. Thread startup/shutdown is not part of probe totals.
    """

    def __init__(self, work_dir: Path, interval_s: float = 0.01) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Sampling interval must be positive and finite")
        self.work_dir = Path(work_dir)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._finished = False
        self._samples = 0
        self._sample_failures = 0
        self._partial_disk_samples = 0
        self._probe_wall = 0.0
        self._probe_cpu = 0.0
        self._rss_peak: int | None = None
        self._logical_peak: int | None = None
        self._allocated_peak: int | None = None
        self._monitor_error: str | None = None
        self._tree_peak: int | None = None
        self._tree_process_peak: int | None = None
        self._tree_partial_samples = 0

    def _sample(self) -> tuple[int | None, DiskFootprint]:
        started_wall = time.perf_counter()
        started_cpu = time.thread_time()
        try:
            rss = read_rss_bytes()
            disk = disk_footprint(self.work_dir)
            tree = sample_tree()
            self._tree_peak = _maximum(self._tree_peak, tree.rss_bytes)
            self._tree_process_peak = _maximum(self._tree_process_peak, tree.processes)
            if sys.platform == "linux" and not tree.complete:
                self._tree_partial_samples += 1
            self._samples += 1
            self._rss_peak = _maximum(self._rss_peak, rss)
            self._logical_peak = _maximum(self._logical_peak, disk.logical_bytes)
            self._allocated_peak = _maximum(self._allocated_peak, disk.allocated_bytes)
            if (sys.platform == "linux" and rss is None) or disk.logical_bytes is None:
                self._sample_failures += 1
            if not disk.complete:
                self._partial_disk_samples += 1
            return rss, disk
        finally:
            self._probe_wall += time.perf_counter() - started_wall
            self._probe_cpu += time.thread_time() - started_cpu

    def _monitor(self) -> None:
        try:
            while not self._stop.wait(self.interval_s):
                self._sample()
        except Exception as error:
            # Keep the original operation exception intact and expose monitor
            # failure in the result. Process-control exceptions are not caught.
            self._monitor_error = f"{type(error).__name__}: {error}"
            self._sample_failures += 1

    def __enter__(self) -> OperationMetrics:
        if self._started:
            raise RuntimeError("OperationMetrics instances can measure only one operation")
        self._started = True
        self._rss_start, self._disk_start = self._sample()
        self._high_water_start = high_water_bytes()
        self._io_start = read_io_counters()
        self._cpu_start = time.process_time()
        self._children_cpu_start = _children_cpu()
        self._wall_start = time.perf_counter()
        self._thread = threading.Thread(target=self._monitor, name="qavf-metrics", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._wall_seconds = time.perf_counter() - self._wall_start
        self._cpu_seconds = time.process_time() - self._cpu_start
        children_end = _children_cpu()
        self._waited_tree_cpu = (
            self._cpu_seconds + children_end - self._children_cpu_start
            if children_end is not None and self._children_cpu_start is not None
            else None
        )
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._io_end = read_io_counters()
        self._high_water_end = high_water_bytes()
        try:
            self._rss_end, self._disk_end = self._sample()
        except Exception as error:
            self._rss_end, self._disk_end = None, DiskFootprint(None, None, False)
            self._monitor_error = f"{type(error).__name__}: {error}"
            self._sample_failures += 1
        self._finished = True

    def result(self) -> dict[str, Any]:
        if not self._finished:
            raise RuntimeError("Metrics are available only after leaving the measurement context")
        return {
            "wall_seconds": self._wall_seconds,
            "cpu_seconds": self._cpu_seconds,
            "waited_tree_cpu_seconds": self._waited_tree_cpu,
            "process_tree": {
                "sampled_peak_rss_bytes": self._tree_peak,
                "sampled_peak_processes": self._tree_process_peak,
                "partial_samples": self._tree_partial_samples,
                "supported": sys.platform == "linux",
                "method": "Linux procfs live descendants across all threads; summed RSS, not PSS",
                "peak_is_lower_bound": True,
            },
            "rss": {
                "sampled_start_bytes": self._rss_start,
                "sampled_peak_bytes": self._rss_peak,
                "sampled_end_bytes": self._rss_end,
                "lifetime_high_water_start_bytes": self._high_water_start,
                "lifetime_high_water_end_bytes": self._high_water_end,
                "sample_method": "Linux /proc/self/status VmRSS; null when unavailable",
                "high_water_method": "resource.getrusage(RUSAGE_SELF); Linux KiB/macOS bytes",
                "peak_is_lower_bound": True,
            },
            "io": {
                **counter_deltas(self._io_start, self._io_end),
                "method": "Linux /proc/self/io deltas; null when unavailable",
            },
            "disk": {
                "start_logical_bytes": self._disk_start.logical_bytes,
                "start_allocated_bytes": self._disk_start.allocated_bytes,
                "sampled_peak_logical_bytes": self._logical_peak,
                "sampled_peak_allocated_bytes": self._allocated_peak,
                "final_logical_bytes": self._disk_end.logical_bytes,
                "final_allocated_bytes": self._disk_end.allocated_bytes,
                "method": "Owned distinct regular files: st_size and st_blocks * 512; no symlinks",
                "peak_is_lower_bound": True,
            },
            "sampling": {
                "interval_seconds": self.interval_s,
                "sample_count": self._samples,
                "overhead_wall_seconds": self._probe_wall,
                "overhead_cpu_seconds": self._probe_cpu,
                "sample_failures": self._sample_failures,
                "partial_disk_samples": self._partial_disk_samples,
                "monitor_error": self._monitor_error,
                "method": "Aggregate-only probes; totals include initial/background/final samples",
                "rss_supported_platform": sys.platform == "linux",
            },
            "notes": [
                "Sampled RSS and disk peaks can miss allocations between observations.",
                "Lifetime RSS high-water marks include imports and fixture preparation.",
                "Process CPU includes all process threads, including the monitor.",
                "I/O includes monitor procfs reads; rchar/wchar are not physical disk traffic.",
                "read_bytes/write_bytes reflect kernel accounting, caching and dirtying/writeback.",
                "cancelled_write_bytes is separate; it is not subtracted from write_bytes.",
                "Probe overhead overlaps the operation and is not a correction to subtract.",
                "Disk scope excludes fixtures and directory metadata; it is not total device I/O.",
                "Final sampling and monitor join occur after the operation stopwatch stops.",
                "Tree RSS sums shared pages repeatedly; brief/reparented children can escape.",
                "Tree CPU includes waited children; live/unwaited descendants are not included.",
                "Owned disk includes child scratch only when configured below this work directory.",
            ],
        }


def _children_cpu() -> float | None:
    if resource is None or sys.platform == "win32":
        return None
    try:
        value = resource.getrusage(resource.RUSAGE_CHILDREN)
        return value.ru_utime + value.ru_stime
    except (OSError, ValueError, AttributeError):
        return None
