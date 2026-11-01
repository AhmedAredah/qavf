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


def unsupported(capability: str) -> NoReturn:
    action = (
        "select backend='polars' (CLI: --backend polars)"
        if has_polars()
        else "install 'qavf[polars]' and select backend='polars' (CLI: --backend polars)"
    )
    raise OptionalDependencyError(
        f"{capability} requires the optional Polars backend (qavf[polars]); {action}."
    )
