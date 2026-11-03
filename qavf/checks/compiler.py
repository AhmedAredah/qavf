"""Compile check items into rule parameters, one rule instance per item.

Every item keeps its own severity, name, tags and description: it becomes one
instance of its rule, bound to the item (:class:`qavf.engine.registry.CheckBinding`).
Two items of the same check therefore never share a severity or parameters.
The ``expression`` and ``sql`` checks are the exception: one rule evaluates all
of them, and each expression or query carries its item's severity and name.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qavf.checks.catalog import CATALOG, CheckDefinition, PluginDefinition, definition_for
from qavf.checks.items import CheckItem
from qavf.checks.placement import (
    ROW_CHECKS,
    CheckSettingError,
    PlacedItem,
    placed_items,
    row_check_errors,
)
from qavf.engine.registry import CheckBinding, Rule, RuleRegistry

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings


class CheckCompilationError(ValueError):
    """A check item that cannot become a rule (unknown check, invalid settings)."""


@dataclass(frozen=True, slots=True)
class CompiledCheck:
    """A check item turned into its rule's parameters."""

    definition: CheckDefinition
    placed: PlacedItem
    params: dict[str, Any]

    @property
    def item(self) -> CheckItem:
        return self.placed.item

    @property
    def check(self) -> str:
        return self.definition.name

    @property
    def rule_id(self) -> str:
        return self.definition.rule_id

    @property
    def datasets(self) -> tuple[str, ...]:
        return self.placed.datasets

    @property
    def name(self) -> str:
        """The item's name, or its default ``<check> on <dataset>``."""
        return self.placed.name

    @property
    def enabled(self) -> bool:
        return self.item.enabled

    def binding(self) -> CheckBinding:
        return CheckBinding(
            key=self.placed.path,
            check=self.check,
            name=self.name,
            datasets=self.datasets,
            tags=tuple(self.item.tags),
            description=self.item.description,
        )

    def rule(self) -> Rule:
        """A rule instance bound to this item."""
        rule = self.definition.rule_cls(params=self.params, severity=self.item.severity)
        rule.bind(self.binding())
        return rule


def _definition(check: str, registry: RuleRegistry | None) -> CheckDefinition | None:
    if registry is None:
        return CATALOG.get(check)
    for rule_cls in registry:
        if rule_cls.check_name == check:
            return definition_for(rule_cls)
    return None


def configuration_errors(settings: QAVFSettings) -> list[str]:
    """Problems with built-in check items, names and ``run.require``, as ``path: message``.

    Checks that are not built in may come from plugins, which are not loaded
    while a configuration is read; they are validated when the run is planned
    (:func:`compile_checks` with the run's registry).
    """
    problems = []
    for placed in placed_items(settings):
        definition = CATALOG.get(placed.item.check)
        if definition is None or placed.item.check in ROW_CHECKS:
            continue
        # Settings may name components of installed plugins, which load only when
        # the run is planned; such parameters are validated then.
        deferred = definition.plugin_components and settings.plugins.entry_points
        if placed.item.enabled or placed.item.settings:
            # A check switched off before any settings were given is not validated.
            problems.extend(definition.errors(placed.item, placed.scope, parameters=not deferred))
    return problems + row_check_errors(settings)


def compile_checks(
    settings: QAVFSettings, registry: RuleRegistry | None = None
) -> list[CompiledCheck]:
    """Every check item (enabled or not) with its rule parameters.

    With a ``registry``, checks resolve against the installed rules (plugins
    included) and plugin settings are validated; without one, only built-in
    checks resolve.

    Raises:
        CheckCompilationError: for an unknown check name or invalid settings.
    """
    compiled = []
    problems = []
    for placed in placed_items(settings):
        definition = _definition(placed.item.check, registry)
        if definition is None:
            problems.append(
                f"{placed.path}.check: unknown check {placed.item.check!r}. Run "
                "'qavf list-checks' to see what is available (plugins must be installed)."
            )
            continue
        if not definition.configurable:
            problems.extend(definition.errors(placed.item, placed.scope))
            continue
        configured = placed.item.enabled or placed.item.settings
        if isinstance(definition, PluginDefinition) and configured:
            errors = definition.errors(placed.item, placed.scope)
            if errors:
                problems.extend(errors)
                continue
        if not configured:
            # Switched off before any settings were given: nothing to compile.
            compiled.append(CompiledCheck(definition, placed, {}))
            continue
        try:
            params = definition.compile(placed.item, placed.scope)
        except CheckSettingError as exc:
            problems.append(f"{placed.scope.at(exc.field)}: {exc.message}")
            continue
        compiled.append(CompiledCheck(definition, placed, params))
    if problems:
        raise CheckCompilationError("invalid checks: " + "; ".join(problems))
    return compiled


def item_names(compiled: Iterable[CompiledCheck]) -> dict[str, list[CompiledCheck]]:
    """Compiled checks by (effective) item name."""
    names: dict[str, list[CompiledCheck]] = {}
    for check in compiled:
        names.setdefault(check.name, []).append(check)
    return names


__all__ = [
    "CheckCompilationError",
    "CompiledCheck",
    "compile_checks",
    "configuration_errors",
    "item_names",
]
