"""Typed check items for the configuration JSON Schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from qavf.checks.catalog import PluginDefinition, definition_for
from qavf.checks.items import COMMON_FIELDS
from qavf.engine.registry import RuleRegistry


def check_item_models(
    registry: RuleRegistry,
) -> tuple[list[type[BaseModel]], list[type[BaseModel]]]:
    """Item models of the installed checks: ``(dataset checks, top-level checks)``.

    Plugin checks may be placed on a dataset or at the top level.
    """
    dataset: list[type[BaseModel]] = []
    top: list[type[BaseModel]] = []
    for rule_cls in registry:
        definition = definition_for(rule_cls)
        if not definition.configurable:
            continue
        model = definition.item_model()
        plugin = isinstance(definition, PluginDefinition)
        if plugin or definition.placement == "dataset":
            dataset.append(model)
        if plugin or definition.placement == "top":
            top.append(model)
    return dataset, top


VAR_REFERENCE = {
    "type": "string",
    "pattern": r"^\$\{vars\.[A-Za-z_][A-Za-z0-9_]*\}$",
    "description": "A variable defined under vars, e.g. ${vars.years}.",
}
VALUE_RANGE = {
    "anyOf": [
        {"type": "string", "pattern": r"^\s*-?\d+\s*\.\.\s*-?\d+\s*$"},
        {
            "type": "object",
            "properties": {
                "from": {"type": "integer"},
                "to": {"type": "integer"},
                "step": {"type": "integer", "exclusiveMinimum": 0},
            },
            "required": ["from", "to"],
            "additionalProperties": False,
        },
    ],
    "description": "Every whole number of a range: 2016..2024 or {from, to, step}.",
}
_ANNOTATIONS = ("title", "description", "default", "examples")


def accept_reuse_values(schema: dict[str, Any]) -> dict[str, Any]:
    """Let every check setting in ``schema`` take ``${vars.name}`` or a range.

    Variables and ranges are resolved before validation, so the editor schema of a
    configuration file must accept them wherever a check setting goes.
    """
    definitions = schema.setdefault("$defs", {})
    definitions["VarReference"] = VAR_REFERENCE
    definitions["ValueRange"] = VALUE_RANGE
    for name, definition in definitions.items():
        if not name.startswith("Check_"):
            continue
        properties = definition.get("properties", {})
        for field, spec in properties.items():
            if field not in COMMON_FIELDS:
                properties[field] = _reusable(spec)
    return schema


def _reusable(spec: dict[str, Any]) -> dict[str, Any]:
    outer = {key: spec[key] for key in _ANNOTATIONS if key in spec}
    inner = {key: value for key, value in spec.items() if key not in _ANNOTATIONS}
    alternatives = [{"$ref": "#/$defs/VarReference"}, {"$ref": "#/$defs/ValueRange"}]
    return {**outer, "anyOf": [inner, *alternatives]}


__all__ = ["VALUE_RANGE", "VAR_REFERENCE", "accept_reuse_values", "check_item_models"]
