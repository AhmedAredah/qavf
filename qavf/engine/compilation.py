"""Data-free rule discovery, configuration checks, and atomic plan compilation.

Compilation produces owned, read-only plan collections. Scheduling, deadlines,
invocation cleanup, and result admission remain the runner's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from operator import attrgetter
from types import MappingProxyType

from pydantic import ValidationError

from qavf.checks.selection import NOT_SELECTED
from qavf.engine.registry import Rule, RuleRegistry, default_registry
from qavf.plugins.registry import entry_point_scope
from qavf.redaction import exception_details, redact_text
from qavf.settings import QAVFSettings, validate_config_mapping_keys
from qavf.sources.base import SourceRegistry


class ConfigurationError(ValueError):
    """The configuration refers to unknown rules or carries invalid rule parameters."""


def build_registry(settings: QAVFSettings) -> RuleRegistry:
    """Discover rules as directed by ``settings.plugins``.

    Rules registered on the process-wide default registry via ``@register``
    are included as well.
    """
    # Component plugins may register rules through the public decorator. Load
    # them before copying defaults, using the same visibility policy as the run.
    with entry_point_scope(settings.plugins.entry_points):
        registry = RuleRegistry().discover(
            packages=settings.plugins.packages,
            modules=settings.plugins.modules,
            paths=[settings.resolve(path) for path in settings.plugins.paths],
            use_entry_points=settings.plugins.entry_points,
        )
        for rule_cls in default_registry:
            registry.register(rule_cls)
    return registry


def build_source_registry(settings: QAVFSettings) -> SourceRegistry:
    """Built-in sources plus those from ``plugins.modules`` and entry points."""
    return SourceRegistry().discover(
        modules=tuple(settings.plugins.modules), use_entry_points=settings.plugins.entry_points
    )


@dataclass(frozen=True, slots=True)
class RulePlan:
    """Validated rule instances and structurally immutable planning diagnostics.

    Instances retain each rule's normal execution contract. The container owns
    its mappings and tuple values, so caller mutation cannot change diagnostics.
    """

    rules: tuple[Rule, ...]
    excluded: Mapping[str, str]
    """Disabled check items, by item name, with the reason."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "excluded", MappingProxyType(dict(self.excluded)))


def compile_rules(
    settings: QAVFSettings, registry: RuleRegistry, sources: SourceRegistry
) -> RulePlan:
    """Instantiate the rules that will run, in ``rule_id`` order.

    Comparison components and statistical tests from installed plugins load and
    resolve only when ``plugins.entry_points`` is enabled; rule parameters are
    validated in that scope.

    Raises:
        ConfigurationError: for unknown rule ids or invalid rule parameters.
    """
    with entry_point_scope(settings.plugins.entry_points):
        return _compile_rules(settings, registry, sources)


def _compile_rules(
    settings: QAVFSettings, registry: RuleRegistry, sources: SourceRegistry
) -> RulePlan:
    # Imported here: the check catalog imports the built-in rule implementations.
    from qavf.checks.catalog import CONTRACT_RULES
    from qavf.checks.compiler import CheckCompilationError, compile_checks

    check_config_keys(settings)
    check_required_plugins(settings, registry, sources)
    try:
        compiled = compile_checks(settings, registry)
    except CheckCompilationError as exc:
        raise ConfigurationError(str(exc)) from None

    planned: list[Rule] = []
    excluded: dict[str, str] = {}
    applied = settings.selection
    not_selected = applied.not_selected if applied is not None else frozenset()
    # Schema contracts always run (for the selected datasets); they come from each
    # dataset's schema.
    planned.extend(_contracts(settings, registry, CONTRACT_RULES, excluded))
    list_driven: set[str] = set()
    for check in compiled:
        if not check.enabled:
            excluded[check.name] = (
                NOT_SELECTED
                if check.placed.path in not_selected
                else "disabled in the configuration"
            )
        if check.definition.list_driven:
            if check.enabled:
                list_driven.add(check.rule_id)
            continue
        if not check.enabled and not check.item.settings:
            continue  # switched off before any settings were given
        try:
            rule = check.rule()
        except ValidationError as exc:
            raise ConfigurationError(
                f"invalid settings for {check.placed.path} ({check.check}): "
                f"{exception_details(exc)}"
            ) from None
        if check.enabled:
            planned.append(rule)
    # Row expressions and SQL checks are evaluated together, each with its item's
    # severity and name.
    planned.extend(registry.get(rule_id)() for rule_id in sorted(list_driven))
    planned.sort(key=attrgetter("rule_id"))
    problems = [
        f"{rule.key}: {problem}"
        for rule in planned
        for problem in rule.configuration_errors(settings)
    ]
    if problems:
        raise ConfigurationError("; ".join(problems))
    return RulePlan(tuple(planned), excluded)


def _contracts(
    settings: QAVFSettings,
    registry: RuleRegistry,
    rule_ids: tuple[str, ...],
    excluded: dict[str, str],
) -> list[Rule]:
    """The schema contract rules, restricted by the run's selection."""
    applied = settings.selection
    available = [rule_id for rule_id in rule_ids if rule_id in registry]
    if applied is None:
        return [registry.get(rule_id)() for rule_id in available]
    if not applied.contract_runs(None):
        excluded.update(dict.fromkeys(available, NOT_SELECTED))
        return []
    datasets = applied.contract_datasets()
    params = {} if datasets is None else {"datasets": datasets}
    return [registry.get(rule_id)(params=params) for rule_id in available]


def check_config_keys(settings: QAVFSettings) -> None:
    """Validate the live model before discovery or execution sees mutations."""
    try:
        validate_config_mapping_keys(settings)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from None


def plugin_status(
    registry: RuleRegistry, sources: SourceRegistry, *, comparison_plugins: bool = True
) -> dict[str, str]:
    """Every entry point tried, as ``rules:name`` / ``sources:name`` /
    ``compare.<kind>:name`` / ``inference.tests:name`` -> status.
    ``comparison_plugins`` is the run's ``plugins.entry_points``: a run without
    plugins reports no comparison or test plugin."""
    from qavf.plugins.registry import plugin_status as comparison_status

    status = {f"rules:{name}": state for name, state in getattr(registry, "plugins", {}).items()}
    status |= {f"sources:{name}": state for name, state in getattr(sources, "plugins", {}).items()}
    status |= comparison_status(enabled=comparison_plugins)
    return {key: redact_text(value) for key, value in sorted(status.items())}


def check_required_plugins(
    settings: QAVFSettings, registry: RuleRegistry, sources: SourceRegistry
) -> None:
    required = settings.plugins.required
    if not required:
        return
    if settings.plugins.entry_points:
        from qavf.plugins.registry import load_entry_points

        load_entry_points()  # judge components after discovering them
    status = plugin_status(registry, sources, comparison_plugins=settings.plugins.entry_points)
    problems: dict[str, str] = {}
    for required_name in required:
        qualified = required_name.startswith(("rules:", "sources:", "compare.", "inference."))
        components = {
            name: state
            for name, state in status.items()
            if (name == required_name if qualified else name.split(":", 1)[1] == required_name)
        }
        if not components:
            problems[required_name] = "not installed"
        problems.update({name: state for name, state in components.items() if state != "loaded"})
    if problems:
        detail = "; ".join(f"{name}: {state}" for name, state in sorted(problems.items()))
        raise ConfigurationError(f"required plugin(s) not available: {detail}")
