"""The reuse values an editor JSON Schema must accept before they are resolved.

``${vars.name}`` is substituted in any value before validation, so every setting
of a configuration file may be written as a variable. A value range
(``2016..2024`` or ``{from, to, step}``) is expanded only in check settings
declared as lists (:class:`qavf.checks.reuse.TypedRanges`), so only those
advertise it. A single string where a check lists ``datasets`` is one selector.

This module imports no rule code: the full and the native (DuckDB) schemas both
use it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import Any

from qavf.checks.items import COMMON_FIELDS

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
DATASET_SELECTOR = {
    "type": "string",
    "minLength": 1,
    "description": "One dataset selector, e.g. counts@* for every copy of counts.",
}
"""A single string where a check lists ``datasets``: a one-element selector."""

REUSE_DEFINITIONS = {
    "VarReference": VAR_REFERENCE,
    "ValueRange": VALUE_RANGE,
    "DatasetSelector": DATASET_SELECTOR,
}
_ANNOTATIONS = ("title", "description", "default", "examples")
_CHECK_ITEMS = ("Check_", "Inline_")
_NATIVE_ITEMS = ("ExpressionItem", "SqlItem")
_ROOT_UNRESOLVED = frozenset({"include", "vars", "$schema"})
"""Top-level settings read before variables are substituted (or naming the schema)."""


class ReuseSchema:
    """Adds the reuse alternatives to every property of a configuration schema."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.definitions: dict[str, Any] = schema.setdefault("$defs", {})
        self.ranged = self._check_settings()

    def apply(self) -> dict[str, Any]:
        for name, definition in list(self.definitions.items()):
            if name not in REUSE_DEFINITIONS:
                self._definition(name, definition)
        self.definitions.update(copy.deepcopy(REUSE_DEFINITIONS))
        properties = self.schema.get("properties", {})
        for field, spec in properties.items():
            if field not in _ROOT_UNRESOLVED and _wrappable(spec):
                properties[field] = _reusable(spec)
        return self.schema

    def _definition(self, name: str, definition: Mapping[str, Any]) -> None:
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            return
        item = _is_check_item(name)
        ranged = name in self.ranged
        for field, spec in properties.items():
            if not _wrappable(spec):
                continue
            ranges = ranged and self._sequence(spec) and not (item and field in COMMON_FIELDS)
            properties[field] = _reusable(
                spec, ranges=ranges, selector=item and field == "datasets"
            )

    def _check_settings(self) -> frozenset[str]:
        """The definitions describing check settings: ranges expand only there."""
        found: set[str] = set()
        pending = [name for name in self.definitions if _is_check_item(name)]
        while pending:
            name = pending.pop()
            if name in found or name not in self.definitions:
                continue
            found.add(name)
            pending.extend(_references(self.definitions[name]))
        return frozenset(found)

    def _sequence(self, spec: Mapping[str, Any]) -> bool:
        """Whether a property takes a list, and no string, mapping or any value."""
        arms = [self._resolved(arm) for arm in spec.get("anyOf", [spec])]
        if not any(arm.get("type") == "array" for arm in arms):
            return False
        return not any(_literal_arm(arm) for arm in arms)

    def _resolved(self, arm: Mapping[str, Any]) -> Mapping[str, Any]:
        reference = arm.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            target = self.definitions.get(reference.removeprefix("#/$defs/"), {})
            return target if isinstance(target, Mapping) else {}
        return arm


def accept_reuse_values(schema: dict[str, Any]) -> dict[str, Any]:
    """Let every setting in ``schema`` take ``${vars.name}``, and check settings
    declared as lists take a value range. Variables and ranges are resolved before
    validation, so the editor schema of a configuration file must accept them."""
    return ReuseSchema(schema).apply()


def _is_check_item(name: str) -> bool:
    return name.startswith(_CHECK_ITEMS) or name in _NATIVE_ITEMS


def _wrappable(spec: Any) -> bool:
    """Discriminating constants (a check's ``check``) stay as they are."""
    return isinstance(spec, dict) and "const" not in spec


def _literal_arm(arm: Mapping[str, Any]) -> bool:
    kind = arm.get("type")
    if not arm or kind in ("string", "object"):
        return True
    return kind is None and ("properties" in arm or "additionalProperties" in arm)


def _references(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            yield reference.removeprefix("#/$defs/")
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _reusable(
    spec: dict[str, Any], *, ranges: bool = False, selector: bool = False
) -> dict[str, Any]:
    outer = {key: spec[key] for key in _ANNOTATIONS if key in spec}
    inner = {key: value for key, value in spec.items() if key not in _ANNOTATIONS}
    alternatives: list[dict[str, Any]] = [{"$ref": "#/$defs/VarReference"}]
    if ranges:
        alternatives.append({"$ref": "#/$defs/ValueRange"})
    if selector:
        # A selector (counts@*) stands for several datasets, so a list needs no minimum.
        inner.pop("minItems", None)
        alternatives.insert(0, {"$ref": "#/$defs/DatasetSelector"})
    return {**outer, "anyOf": [inner, *alternatives]}


__all__ = [
    "DATASET_SELECTOR",
    "REUSE_DEFINITIONS",
    "VALUE_RANGE",
    "VAR_REFERENCE",
    "ReuseSchema",
    "accept_reuse_values",
]
