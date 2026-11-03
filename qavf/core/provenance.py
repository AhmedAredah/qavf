"""Small provenance and acceptance services independent of execution backend."""

from __future__ import annotations

import contextlib
import functools
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from qavf._version import __version__


@functools.cache
def _environment():
    result = {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "qavf": __version__,
    }
    for package in ("duckdb", "pyarrow", "pydantic"):
        with contextlib.suppress(PackageNotFoundError):
            result[package] = version(package)
    return tuple(result.items())


def environment():
    """Detached process-stable environment, without repeated package metadata I/O."""
    return dict(_environment())
