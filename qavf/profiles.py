"""Explicit, additive check profiles; data never defines its own acceptance limits."""

from __future__ import annotations

from typing import Literal

from qavf.settings import QAVFSettings, RuleSettings

ProfileName = Literal["validity", "regression", "change"]
_INPUT_RULES = tuple(f"INP-{i:03d}" for i in range(1, 8))
_ANALYSIS = {"regression": ("REG-001", "baselines"), "change": ("CMP-001", "comparisons")}


def apply_profile(settings: QAVFSettings, name: ProfileName) -> QAVFSettings:
    """Expand a named profile while retaining every explicitly configured check.

    All profiles include declared schema/key/reference/expression validity.
    Regression/change additionally require explicit baseline/comparison settings;
    no tolerances, geography limits, or accepted baselines are inferred.
    """
    if name not in ("validity", "regression", "change"):
        raise ValueError("profile must be 'validity', 'regression', or 'change'")
    if (
        not any(
            spec.schema_spec is not None or spec.foreign_keys or spec.expressions
            for spec in settings.datasets.values()
        )
        and not settings.assertions
    ):
        raise ValueError("a validation profile requires an explicit dataset QA contract")
    required_categories = {"input"}
    if name in _ANALYSIS:
        rule_id, collection = _ANALYSIS[name]
        override = settings.rules.get(rule_id)
        if override is None or not override.enabled or not override.params.get(collection):
            raise ValueError(f"profile {name!r} requires configured {rule_id}.{collection}")
        required_categories.add("regression" if name == "regression" else "comparison")
    engine = settings.engine
    if required_categories & set(engine.exclude_categories) or (
        engine.include_categories is not None
        and not required_categories <= set(engine.include_categories)
    ):
        raise ValueError("engine category filters exclude checks required by the profile")
    rules = dict(settings.rules)
    for rule_id in _INPUT_RULES:
        if rule_id in rules and not rules[rule_id].enabled:
            raise ValueError(f"profile {name!r} conflicts with disabled {rule_id}")
        rules.setdefault(rule_id, RuleSettings())
    # Expanded checks are the effective configuration and enter the plan digest.
    # Preserve explicit custom checks while avoiding unrelated unconfigured rules.
    extensions = dict(settings.extensions)
    provenance = {"name": name, "input_rules": list(_INPUT_RULES)}
    if "qavf.profile" in extensions and extensions["qavf.profile"] != provenance:
        raise ValueError("extensions['qavf.profile'] conflicts with the selected profile")
    extensions["qavf.profile"] = provenance
    return settings.model_copy(
        update={
            "rules": rules,
            "engine": engine.model_copy(update={"run_unconfigured_rules": False}),
            "extensions": extensions,
        }
    )
