"""Explicit integration prerequisites; unavailable services never become a pass."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from urllib.parse import urlsplit


class IntegrationUnavailable(RuntimeError):
    """A required integration prerequisite could not be satisfied."""


def required_mode(environment: Mapping[str, str]) -> bool:
    mode = environment.get("QAVF_INTEGRATION_MODE", "disabled")
    if mode not in {"disabled", "required"}:
        raise IntegrationUnavailable("QAVF_INTEGRATION_MODE must be disabled or required")
    return mode == "required"


def settings(environment: Mapping[str, str], names: tuple[str, ...]) -> dict[str, str]:
    missing = [name for name in names if not environment.get(name, "").strip()]
    if missing:
        raise IntegrationUnavailable("required integration settings missing: " + ", ".join(missing))
    return {name: environment[name] for name in names}


def dependency(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError:
        raise IntegrationUnavailable(
            f"required integration dependency {name!r} unavailable; install the integration group"
        ) from None


def endpoint_parts(endpoint: str) -> tuple[str, bool]:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise IntegrationUnavailable(
            "QAVF_S3_ENDPOINT must be an HTTP(S) origin without credentials"
        )
    return parsed.netloc, parsed.scheme == "https"


def extension(name: str) -> None:
    """Load a preinstalled extension; installation belongs to environment setup."""
    import duckdb

    if name not in {"httpfs", "postgres"}:
        raise ValueError("unsupported integration extension")
    try:
        with duckdb.connect() as connection:
            connection.execute(f"LOAD {name}")
    except duckdb.Error:
        raise IntegrationUnavailable(
            f"required DuckDB extension {name!r} unavailable; install it before the run"
        ) from None
