"""Record executable, input, and host identities without importing QAVF early."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_digest(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def production_digest(root: Path) -> str:
    paths = [
        p
        for p in (root / "qavf").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    if not paths or not (root / "qavf/__init__.py").is_file():
        raise ValueError(f"No QAVF source package at {root}")
    return tree_digest(root, paths)


def harness_digest(root: Path) -> str:
    paths = list((root / "scripts/performance").glob("*.py"))
    paths.append(root / "scripts/benchmark_performance.py")
    return tree_digest(root, paths)


def fixture_identity(directory: Path) -> dict[str, Any]:
    files = [p for p in directory.rglob("*") if p.is_file()]
    inventory = [
        {
            "path": p.relative_to(directory).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": file_digest(p),
        }
        for p in sorted(files)
    ]
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": inventory,
        "total_bytes": sum(item["size_bytes"] for item in inventory),
    }


def _read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def host_environment() -> dict[str, Any]:
    """Host limits are observations, not benchmark-enforced resource ceilings."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "cgroup_memory_max": _read_optional("/sys/fs/cgroup/memory.max"),
        "cgroup_memory_current": _read_optional("/sys/fs/cgroup/memory.current"),
        "cgroup_cpu_max": _read_optional("/sys/fs/cgroup/cpu.max"),
        "cache_condition": (
            "OS page cache uncontrolled; process isolation is not cold-disk isolation"
        ),
    }


def worker_environment(native_threads: int) -> dict[str, Any]:
    import polars as pl

    import qavf

    dependencies: dict[str, str | None] = {}
    for name in ("duckdb", "polars", "pyarrow", "pydantic", "numpy", "jinja2", "scipy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependencies,
        "polars_threads": pl.thread_pool_size(),
        "native_threads": native_threads,
        "executable": sys.executable,
        "qavf_import_path": str(Path(qavf.__file__).resolve()),
    }
