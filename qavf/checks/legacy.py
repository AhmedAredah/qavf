"""Recognise configurations written for QAVF 0.6 and earlier.

Those configurations select rules by code (``rules: {TMP-001: ...}``). They
are rejected before validation with an error naming the first old key and the
command that converts them, instead of a list of unrelated schema errors.
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


class LegacyConfigurationError(ValueError):
    """A configuration written for QAVF 0.6 or earlier."""

    def __init__(self, key: str, source: str | None = None) -> None:
        self.key = key
        target = source or "<config>"
        super().__init__(
            f"'{key}' is a QAVF 0.6 setting: checks are now configured by name. "
            f"Convert this configuration with '{MIGRATE_HINT} {target} -o <new config>'."
        )


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


__all__ = ["LegacyConfigurationError", "first_legacy_key", "reject_legacy"]
