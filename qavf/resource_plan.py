"""Metadata-only resource planning shared by native and adapter engines.

Estimates influence concurrency and automatic ceilings, never the data checked.
Explicit user caps are never raised. No row scan, materialization or optional
dataframe library is needed to plan a run.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

_MIB = 1024**2
_BOUND_PLAN: ContextVar[Any] = ContextVar("qavf_resource_plan", default=None)


def _available_memory() -> int:
    """Conservative host/container allowance, without importing psutil."""
    candidates: list[int] = []
    try:
        available = next(
            (
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            ),
            None,
        )
        if available is None and sys.platform != "win32":
            available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        if available is not None:
            candidates.append(available)
    except (ValueError, OSError, AttributeError):
        pass
    for maximum, current in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            limit, used = int(Path(maximum).read_text()), int(Path(current).read_text())
            # Linux charges reclaimable filesystem cache to memory.current.
            # Counting it as live allocations starves the next validation after
            # reading a large file, even though the cache can be reclaimed.
            stat = Path(maximum).parent / "memory.stat"
            values = dict(line.split() for line in stat.read_text().splitlines())
            used = max(
                0, used - int(values.get("inactive_file", values.get("total_inactive_file", "0")))
            )
            candidates.append(max(_MIB, limit - used))
        except (OSError, ValueError):
            pass
    return min(candidates) if candidates else 2 * 1024**3


def _column_width(spec: Any) -> int:
    dtype = str(getattr(spec, "dtype", "") or "").lower()
    if dtype in {"bool", "boolean"}:
        return 1
    if dtype.startswith("decimal"):
        return 16
    if dtype in {"int", "integer", "float", "number", "date", "datetime", "time"}:
        return 8
    # Variable width keys need payload storage plus a pointer/hash descriptor.
    return 64


def _input_shape(
    settings: Any, name: str, spec: Any, frames: Mapping[str, Any]
) -> tuple[int, int | None]:
    supplied_path = None
    if name in frames:
        frame = frames[name]
        rows = getattr(frame, "num_rows", getattr(frame, "height", None))
        if rows is None and type(frame).__module__.startswith("pandas"):
            rows = len(frame)
        size = getattr(frame, "nbytes", None)
        if size is None and callable(getattr(frame, "estimated_size", None)):
            size = frame.estimated_size()
        if rows is not None or size is not None:
            return int(size or 0), int(rows) if rows is not None else None
        if isinstance(frame, str | os.PathLike):
            supplied_path = os.fspath(frame)
        elif isinstance(frame, Mapping) and frame.get("type", "file") == "file":
            supplied_path = frame.get("path")
        else:
            supplied_path = getattr(frame, "path", None)
        if supplied_path is None:
            # A runtime query/lazy source replaces its configured location.
            # Its unknown size must not inherit metadata from the old file.
            return 0, None
    source = spec.source or {}
    path = (
        supplied_path or spec.path or (source.get("path") if source.get("type") == "file" else None)
    )
    if not isinstance(path, str | os.PathLike) or "://" in os.fspath(path):
        return 0, None
    absolute = settings.resolve(path)
    paths = [absolute] if absolute.is_file() else (Path(p) for p in glob.iglob(str(absolute)))
    size = 0
    for candidate in paths:
        try:
            if candidate.is_file():
                size += candidate.stat().st_size
        except OSError:
            pass  # The source reports unreadability; planning does not suppress it.
    return size, None


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """A reproducible scheduling decision and its explicitly estimated inputs."""

    memory_bytes: int | None
    spill_bytes: int
    workers: int
    connection_slots: int
    input_bytes: int
    estimated_key_state_bytes: int
    automatic_memory: bool
    automatic_spill: bool

    @classmethod
    def build(
        cls, settings: Any, frames: Mapping[str, Any] | None = None, *, requested_workers: int = 1
    ) -> ResourcePlan:
        engine = settings.engine
        workers = max(1, requested_workers)
        input_bytes = key_state = 0
        for name, spec in settings.datasets.items():
            size, rows = _input_shape(settings, name, spec, frames or {})
            input_bytes += size
            schema = spec.schema_spec
            if schema is None:
                continue
            keys = [schema.primary_key, *schema.unique_keys]
            if not any(keys):
                continue
            # Without a known row count, use a minimum delimited-field width.
            # Physical sizes of compressed formats are only estimates of their
            # decoded population; this is not a proven cardinality bound.
            estimate_rows = rows if rows is not None else size // max(2, 2 * len(schema.columns))
            for key in keys:
                columns = key if isinstance(key, list | tuple) else key.columns
                if columns:
                    width = sum(_column_width(schema.columns.get(c)) for c in columns)
                    key_state = max(key_state, estimate_rows * (width + 64))
        memory = engine.memory_budget_bytes
        auto_memory = memory is None
        if auto_memory and (input_bytes or key_state):
            # CSV decoding and joins need meaningful per-handle headroom even
            # for a tiny input. Do not turn a small file into a 16/32 MiB cap.
            # Half of the target belongs to retained data; two native handles
            # share the other half for each worker. Provision the key estimate
            # after those partitions, rather than shrink it below its own need.
            desired = max(workers * 512 * _MIB, input_bytes * 3 + key_state * 4)
            memory = max(_MIB, min(desired, _available_memory() // 2))
            workers = min(workers, max(1, memory // (4 * 128 * _MIB)))
        if memory is not None and key_state > 32 * _MIB:
            # Two possible handles per worker share half of the target with
            # retained data. Reserve room for the largest key operator rather
            # than divide the budget among workers doing no substantive work.
            workers = min(workers, max(1, memory // (4 * key_state)))
        automatic_spill = "spill_max_bytes" not in engine.model_fields_set
        spill = engine.spill_max_bytes
        if automatic_spill and (input_bytes or key_state):
            desired_spill = max(spill, 2 * (input_bytes + 2 * key_state))
            directory = (
                settings.resolve(engine.temp_directory)
                if engine.temp_directory
                else Path(tempfile.gettempdir())
            )
            try:
                # Do not create a requested directory merely to estimate capacity.
                ancestor = directory
                while not ancestor.exists() and ancestor != ancestor.parent:
                    ancestor = ancestor.parent
                free = shutil.disk_usage(ancestor).free
                spill = max(0, min(desired_spill, free * 4 // 5))
            except OSError:
                spill = desired_spill
        return cls(
            memory,
            spill,
            workers,
            2 * workers,
            input_bytes,
            key_state,
            auto_memory,
            automatic_spill,
        )

    @contextmanager
    def bind(self, settings: Any, frames: Mapping[str, Any] | None) -> Iterator[None]:
        """Publish this immutable decision during synchronous context construction.

        Identity checks prevent nested/unrelated runs from inheriting a plan;
        ContextVar isolation preserves concurrent callers and existing factory
        signatures. The binding ends before rule/plugin execution begins.
        """
        token = _BOUND_PLAN.set((settings, frames, self))
        try:
            yield
        finally:
            _BOUND_PLAN.reset(token)

    @classmethod
    def for_context(
        cls, settings: Any, frames: Mapping[str, Any] | None = None, *, requested_workers: int = 1
    ) -> ResourcePlan:
        """Reuse a runner's decision, or plan once for a directly created context."""
        bound = _BOUND_PLAN.get()
        if bound is not None and bound[0] is settings and bound[1] is frames:
            return cast(ResourcePlan, bound[2])
        return cls.build(settings, frames, requested_workers=requested_workers)

    def diagnostics(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "estimate_scope": (
                "input metadata and declared key widths; not a row scan or RSS guarantee"
            ),
        }
