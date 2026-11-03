"""Recognise configurations written for QAVF 0.6 and earlier.

Those configurations select rules by code (``rules: {TMP-001: ...}``), bound
column statistics with top-level column keys (``mean: {ge: 1}``) and call a
column's conditional limits ``checks``. They are rejected before validation with
an error naming the first old key and the command that converts them, instead of
a list of unrelated schema errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LEGACY_ENGINE_KEYS = (
    "run_unconfigured_rules",
    "include_categories",
    "exclude_categories",
    "fail_on",
    "fail_on_incomplete",
    "require",
)
"""``engine`` settings that moved to ``run`` or were removed."""

MIGRATE_HINT = "qavf migrate"

COLUMN_STATISTICS = ("min", "max", "mean", "median", "std", "sum", "distinct_count")
"""Column keys of QAVF 0.6 that now go under the column's ``stats``."""

_REASONS = {
    "stats": "column statistics now go under 'stats'",
    "conditions": "a column's conditional limits are now 'conditions'",
}


class LegacyConfigurationError(ValueError):
    """A configuration written for QAVF 0.6 or earlier."""

    def __init__(self, key: str, source: str | None = None) -> None:
        self.key = key
        target = source or "<config>"
        reason = _REASONS.get(_column_change(key) or "", "checks are now configured by name")
        super().__init__(
            f"'{key}' is a QAVF 0.6 setting: {reason}. "
            f"Convert this configuration with '{MIGRATE_HINT} {target} -o <new config>'."
        )


def _column_change(key: str) -> str | None:
    """``stats`` or ``conditions`` when ``key`` is an old column key."""
    if ".columns." not in key:
        return None
    return "conditions" if key.rsplit(".", 1)[-1] == "checks" else "stats"


def legacy_column_key(column: Any) -> str | None:
    """The first QAVF 0.6 key of one raw column mapping (``mean`` or ``checks``), if any."""
    if not isinstance(column, Mapping):
        return None
    for name in COLUMN_STATISTICS:
        if name in column:
            return name
    return "checks" if "checks" in column else None


def _legacy_column(datasets: Mapping[str, Any]) -> str | None:
    for name, spec in datasets.items():
        schema = spec.get("schema") if isinstance(spec, Mapping) else None
        columns = schema.get("columns") if isinstance(schema, Mapping) else None
        if not isinstance(columns, Mapping):
            continue
        for column, column_spec in columns.items():
            key = legacy_column_key(column_spec)
            if key is not None:
                return f"datasets.{name}.schema.columns.{column}.{key}"
    return None


def first_legacy_key(raw: Any) -> str | None:
    """Path of the first QAVF 0.6 key in a raw configuration mapping, if any."""
    if not isinstance(raw, Mapping):
        return None
    for key in ("rules", "assertions"):
        if key in raw:
            return key
    datasets = raw.get("datasets")
    if isinstance(datasets, Mapping):
        for name, spec in datasets.items():
            if isinstance(spec, Mapping) and "expressions" in spec:
                return f"datasets.{name}.expressions"
        column = _legacy_column(datasets)
        if column is not None:
            return column
    engine = raw.get("engine")
    if isinstance(engine, Mapping):
        for key in LEGACY_ENGINE_KEYS:
            if key in engine:
                return f"engine.{key}"
    return None


def reject_legacy(raw: Any, source: str | None = None) -> None:
    """Raise :class:`LegacyConfigurationError` for a QAVF 0.6 configuration."""
    key = first_legacy_key(raw)
    if key is not None:
        raise LegacyConfigurationError(key, source)


__all__ = [
    "COLUMN_STATISTICS",
    "LegacyConfigurationError",
    "first_legacy_key",
    "legacy_column_key",
    "reject_legacy",
]
