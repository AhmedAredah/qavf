"""Typed check items for the configuration JSON Schema."""

from __future__ import annotations

from pydantic import BaseModel

from qavf.checks.catalog import PluginDefinition, definition_for
from qavf.engine.registry import RuleRegistry
from qavf.schema_reuse import DATASET_SELECTOR, VALUE_RANGE, VAR_REFERENCE, accept_reuse_values


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


__all__ = [
    "DATASET_SELECTOR",
    "VALUE_RANGE",
    "VAR_REFERENCE",
    "accept_reuse_values",
    "check_item_models",
]
