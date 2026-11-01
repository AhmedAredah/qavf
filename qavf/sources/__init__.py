"""Pluggable sources, imported on demand without loading optional adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    **dict.fromkeys(
        (
            "DataSource",
            "ScanPlan",
            "SourceEnvironment",
            "SourceError",
            "SourceIntegrityError",
            "SourceNotFoundError",
            "SourceParams",
            "SourceRegistry",
        ),
        "base",
    ),
    "FileSource": "file",
    "infer_format": "formats",
    "as_polars": "frames",
    **dict.fromkeys(("DatabaseParams", "FileParams", "PythonParams", "QueryParams"), "parameters"),
    "PythonSource": "python",
    "DatabaseSource": "query",
    "QuerySource": "query",
    "validate_query": "sql",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"qavf.sources.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
