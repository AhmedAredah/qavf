"""Explicit optional-backend capabilities, independent of dataframe imports."""

from __future__ import annotations

import importlib.util
from typing import NoReturn


class OptionalDependencyError(ImportError):
    """A requested capability requires an optional backend."""


def has_polars() -> bool:
    try:
        return importlib.util.find_spec("polars") is not None
    except (ImportError, ValueError):
        return False


def require_polars(capability: str):
    try:
        import polars
    except ImportError:
        raise OptionalDependencyError(
            f"{capability} requires the optional Polars backend; install 'qavf[polars]'."
        ) from None
    return polars


BACKENDS = ("auto", "duckdb", "polars")
"""The ``backend=`` values of validation and planning."""


def resolve_backend(backend: str) -> str:
    """The backend a run requested with ``backend`` uses: ``"duckdb"`` or ``"polars"``.

    ``"auto"`` is Polars when installed, otherwise the DuckDB core.

    Raises:
        ValueError: for an unknown backend name.
        OptionalDependencyError: for ``"polars"`` without the optional extra.
    """
    if backend not in BACKENDS:
        raise ValueError("backend must be 'auto', 'duckdb', or 'polars'")
    if backend == "polars":
        require_polars("backend='polars'")
        return "polars"
    return "duckdb" if backend == "duckdb" or not has_polars() else "polars"


def unsupported(capability: str) -> NoReturn:
    action = (
        "select backend='polars' (CLI: --backend polars)"
        if has_polars()
        else "install 'qavf[polars]' and select backend='polars' (CLI: --backend polars)"
    )
    raise OptionalDependencyError(
        f"{capability} requires the optional Polars backend (qavf[polars]); {action}."
    )
