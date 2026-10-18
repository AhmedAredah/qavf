"""Standard-library-only process limits, applied before native imports.

Linux and macOS lower POSIX rlimits in the worker. Windows has no rlimits: the
parent creates a Job Object carrying the limits and the worker joins it here.
None of these settings is a process-tree RSS or aggregate-disk limit.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

SUPPORTED_PLATFORMS = ("linux", "darwin", "win32")
_CONTROLS = ("address_space_bytes", "cpu_time_s", "file_bytes")


def unsupported(requested: Mapping[str, Any], platform: str | None = None) -> str | None:
    """Explain why this platform cannot enforce a requested control, if it cannot."""
    platform = sys.platform if platform is None else platform
    if platform not in SUPPORTED_PLATFORMS:
        return (
            "isolated execution is not supported on this platform; "
            "it supports Linux, macOS and Windows"
        )
    if platform == "darwin" and requested.get("address_space_bytes") is not None:
        # macOS accepts RLIMIT_AS but its kernel does not enforce it.
        return "macOS does not enforce address-space limits; omit address_space for isolation"
    if platform == "win32" and requested.get("file_bytes") is not None:
        return "Windows has no per-file size limit; omit file_size for isolation"
    return None


def _lower(current: int, requested: int, infinity: int) -> int:
    return requested if current == infinity else min(current, requested)


def _validate(requested: Mapping[str, Any]) -> None:
    for name in _CONTROLS:
        value = requested.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer")


def apply_limits(
    requested: Mapping[str, Any], job: str | None = None
) -> dict[str, dict[str, int]]:
    """Lower both soft and hard limits; never relax an inherited restriction.

    On Windows, ``job`` names the parent's Job Object; joining it applies limits.
    """
    reason = unsupported(requested)
    if reason is not None:
        raise RuntimeError(reason)
    _validate(requested)
    if sys.platform == "win32":
        if job is None:
            raise RuntimeError("isolated execution on Windows requires its parent job")
        if __package__:
            from qavf.execution._windows_job import join
        else:  # Executed beside the bootstrap, before the qavf package is importable.
            from _windows_job import join  # type: ignore[no-redef]
        return join(job)
    import resource

    limits = {
        "address_space_bytes": resource.RLIMIT_AS,
        "cpu_time_s": resource.RLIMIT_CPU,
        "file_bytes": resource.RLIMIT_FSIZE,
        "core_bytes": resource.RLIMIT_CORE,
    }
    if sys.platform == "darwin":
        # Reported values would not be enforced; unsupported() rejects requests.
        del limits["address_space_bytes"]
    effective = {}
    for name, identifier in limits.items():
        value = 0 if name == "core_bytes" else requested.get(name)
        if value is not None:
            soft, hard = resource.getrlimit(identifier)
            hard = _lower(hard, value, resource.RLIM_INFINITY)
            soft = _lower(soft, hard, resource.RLIM_INFINITY)
            resource.setrlimit(identifier, (soft, hard))
        soft, hard = resource.getrlimit(identifier)
        effective[name] = {"soft": soft, "hard": hard}
    return effective
