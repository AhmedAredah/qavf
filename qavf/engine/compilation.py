"""Data-free rule discovery, configuration checks, and atomic plan compilation.

Compilation produces owned, read-only plan collections. Scheduling, deadlines,
invocation cleanup, and result admission remain the runner's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import ValidationError

from qavf.engine.registry import Rule, RuleRegistry, default_registry
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
    needs_configuration: Mapping[str, tuple[str, ...]]
    excluded: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(
            self,
            "needs_configuration",
            MappingProxyType(
                {key: tuple(value) for key, value in self.needs_configuration.items()}
            ),
        )
        object.__setattr__(self, "excluded", MappingProxyType(dict(self.excluded)))


def compile_rules(
    settings: QAVFSettings, registry: RuleRegistry, sources: SourceRegistry
) -> RulePlan:
    """Instantiate the rules that will run, in ``rule_id`` order.

    Comparison components from installed plugins load and resolve only when
    ``plugins.entry_points`` is enabled; rule parameters are validated in that scope.

    Raises:
        ConfigurationError: for unknown rule ids or invalid rule parameters.
    """
    from qavf.plugins.registry import entry_point_scope

    with entry_point_scope(settings.plugins.entry_points):
        return _compile_rules(settings, registry, sources)


def _compile_rules(
    settings: QAVFSettings, registry: RuleRegistry, sources: SourceRegistry
) -> RulePlan:
    check_config_keys(settings)
    engine = settings.engine
    check_required_plugins(settings, registry, sources)
    unknown = sorted(set(settings.rules) - {rule.rule_id for rule in registry})
    if unknown:
        raise ConfigurationError(
            f"configuration refers to unknown rule id(s): {', '.join(unknown)}. "
            f"Run 'qavf list-rules' to see what is available (plugins must be installed)."
        )

    known = set(registry.categories())
    requested = set(engine.include_categories or ()) | set(engine.exclude_categories)
    if requested - known:
        raise ConfigurationError(
            f"unknown categor{'y' if len(requested - known) == 1 else 'ies'} in engine "
            f"filters: {', '.join(sorted(requested - known))}; "
            f"known: {', '.join(sorted(known)) or 'none'}"
        )

    required = {required_rule(entry) for entry in engine.require}
    unknown_required = sorted(required - {rule.rule_id for rule in registry})
    if unknown_required:
        raise ConfigurationError(
            f"engine.require names unknown rule id(s): {', '.join(unknown_required)}"
        )

    planned: list[Rule] = []
    needs_configuration: dict[str, list[str]] = {}
    excluded: dict[str, str] = {}
    for rule_cls in registry:
        override = settings.rules.get(rule_cls.rule_id)
        if override is None and not engine.run_unconfigured_rules:
            excluded[rule_cls.rule_id] = "not configured (run_unconfigured_rules: false)"
            continue
        if override is not None and not override.enabled:
            excluded[rule_cls.rule_id] = "disabled in the configuration"
            # Parameters given for a disabled rule are still checked, so a typo is
            # caught now rather than when the rule is switched on.
            if "params" in override.model_fields_set:
                try:
                    rule_cls.Params.model_validate(override.params)
                except ValidationError as exc:
                    raise ConfigurationError(
                        f"invalid parameters for disabled rule {rule_cls.rule_id}: "
                        f"{exception_details(exc)}"
                    ) from None
            continue
        if engine.include_categories is not None and (
            rule_cls.category not in engine.include_categories
        ):
            excluded[rule_cls.rule_id] = (
                f"category {rule_cls.category!r} not in engine.include_categories"
            )
            continue
        if rule_cls.category in engine.exclude_categories:
            excluded[rule_cls.rule_id] = (
                f"category {rule_cls.category!r} in engine.exclude_categories"
            )
            continue
        try:
            planned.append(
                rule_cls(
                    params=override.params if override else None,
                    severity=override.severity if override else None,
                )
            )
        except ValidationError as exc:
            if override is None:
                # Discovered but never configured, and it has required parameters
                # (typically a plugin rule): it cannot apply until configured. An
                # installed plugin must not break configurations that ignore it.
                needs_configuration[rule_cls.rule_id] = sorted(
                    str(error["loc"][0]) for error in exc.errors() if error["loc"]
                )
                continue
            raise ConfigurationError(
                f"invalid parameters for rule {rule_cls.rule_id}: {exception_details(exc)}"
            ) from None
    problems = [
        f"{rule.rule_id}: {problem}"
        for rule in planned
        for problem in rule.configuration_errors(settings)
    ]
    if problems:
        raise ConfigurationError("; ".join(problems))
    return RulePlan(
        tuple(planned), {key: tuple(value) for key, value in needs_configuration.items()}, excluded
    )


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
    ``compare.<kind>:name`` -> status. ``comparison_plugins`` is the run's
    ``plugins.entry_points``: a run without plugins reports no comparison plugin."""
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
        qualified = required_name.startswith(("rules:", "sources:", "compare."))
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


def required_rule(entry: str) -> str:
    return entry.split("[", 1)[0].strip()


def required_check(entry: str) -> str | None:
    if "[" not in entry or not entry.rstrip().endswith("]"):
        return None
    return entry.split("[", 1)[1].rstrip()[:-1]
