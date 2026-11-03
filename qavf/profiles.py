"""Explicit, additive check profiles; data never defines its own acceptance limits."""

from __future__ import annotations

from typing import Literal

from qavf.settings import QAVFSettings

ProfileName = Literal["validity", "regression", "change"]
_INPUT_RULES = tuple(f"INP-{i:03d}" for i in range(1, 8))
_ANALYSIS = {"regression": "matches_baseline", "change": "compare"}


def _configured(settings: QAVFSettings, check: str) -> bool:
    """Whether an enabled ``check`` item is configured anywhere."""
    items = [
        *settings.checks,
        *(item for spec in settings.datasets.values() for item in spec.checks),
    ]
    return any(item.check == check and item.enabled for item in items)


def apply_profile(settings: QAVFSettings, name: ProfileName) -> QAVFSettings:
    """Check a named profile's requirements and record it in the configuration.

    All profiles include declared schema/key/reference/expression validity, which
    always runs. Regression/change additionally require a configured
    ``matches_baseline``/``compare`` check; no tolerances, geography limits, or
    accepted baselines are inferred.
    """
    if name not in ("validity", "regression", "change"):
        raise ValueError("profile must be 'validity', 'regression', or 'change'")
    if not any(
        spec.schema_spec is not None or spec.foreign_keys or spec.expressions
        for spec in settings.datasets.values()
    ) and not _configured(settings, "sql"):
        raise ValueError("a validation profile requires an explicit dataset QA contract")
    if name in _ANALYSIS and not _configured(settings, _ANALYSIS[name]):
        raise ValueError(f"profile {name!r} requires a configured {_ANALYSIS[name]} check")
    has_applicability = any(
        spec.schema_spec is not None
        and any(column.na_values for column in spec.schema_spec.columns.values())
        for spec in settings.datasets.values()
    )
    input_rules = (*_INPUT_RULES, "INP-009") if has_applicability else _INPUT_RULES
    # The profile enters the configuration digest; schema contracts always run.
    extensions = dict(settings.extensions)
    provenance = {"name": name, "input_rules": list(input_rules)}
    if "qavf.profile" in extensions and extensions["qavf.profile"] != provenance:
        raise ValueError("extensions['qavf.profile'] conflicts with the selected profile")
    extensions["qavf.profile"] = provenance
    return settings.model_copy(update={"extensions": extensions})
