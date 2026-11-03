"""What ``qavf list-checks`` shows: every check a configuration can name."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qavf.engine.registry import RuleRegistry

NATIVE_PLACEMENT = {"INP-006": "dataset", "INP-007": "top"}
"""Where the checks the native backend runs from check items go."""


@dataclass(frozen=True, slots=True)
class ListedCheck:
    """One check: its name, rule code, and where it is configured."""

    check_name: str
    rule_id: str
    placement: str
    """``dataset`` (a dataset's ``checks``), ``top`` (top-level ``checks``), ``any``
    (plugins: either) or ``schema`` (runs from the dataset's schema)."""
    category: str
    default_severity: str
    name: str
    description: str
    module: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def listed_checks(registry: RuleRegistry) -> tuple[ListedCheck, ...]:
    """The checks of every installed rule, configurable checks first."""
    # The catalog imports the rule implementations (the full backend).
    from qavf.checks.catalog import PluginDefinition, definition_for

    checks = []
    for rule_cls in registry:
        definition = definition_for(rule_cls)
        if not definition.configurable:
            placement = "schema"
        elif isinstance(definition, PluginDefinition):
            placement = "any"
        else:
            placement = definition.placement
        checks.append(
            ListedCheck(
                check_name=rule_cls.check_name,
                rule_id=rule_cls.rule_id,
                placement=placement,
                category=rule_cls.category,
                default_severity=rule_cls.default_severity.value,
                name=rule_cls.name,
                description=rule_cls.description,
                module=rule_cls.__module__,
            )
        )
    return tuple(sorted(checks, key=_order))


def _order(check: ListedCheck) -> tuple[bool, str]:
    return check.placement == "schema", check.check_name


__all__ = ["NATIVE_PLACEMENT", "ListedCheck", "listed_checks"]
