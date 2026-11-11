"""Data-free public explanations of configured checks and unresolved source work."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from qavf.checks.placement import placed_items
from qavf.checks.selection import Selection, narrow
from qavf.engine.compilation import (
    ConfigurationError,
    build_registry,
    build_source_registry,
    compile_rules,
)
from qavf.engine.planning import build_plan
from qavf.engine.registry import RuleRegistry, default_registry
from qavf.inputs.normalize import Configuration, prepare_configuration
from qavf.inputs.resources import ResourceLimits
from qavf.optional import resolve_backend
from qavf.plugins.transactions import visibility
from qavf.profiles import ProfileName, apply_profile
from qavf.redaction import exception_details
from qavf.settings import QAVFSettings, SchemaSpec
from qavf.snapshots import SnapshotModel
from qavf.sources.base import SourceRegistry

if TYPE_CHECKING:
    from qavf._input_types import ValidationInput
    from qavf.core.planning import NativePlan


class DatasetExplanation(SnapshotModel):
    """Declared metadata only; schema and dimensions have not been read."""

    name: str
    input_kind: str
    declared_columns: tuple[str, ...] = ()
    row_budget: int | None = None
    observed_schema: None = None
    source_evaluations: None = None
    notes: tuple[str, ...] = ()
    retained_columns: tuple[str, ...] | None = None
    """The columns the native backend reads (``None``: every column, or a Polars plan)."""
    retention: str | None = None
    """Why the native backend reads every column of this dataset."""
    component: tuple[str, ...] = ()
    """The datasets the native backend keeps loaded together with this one."""
    staging: str | None = None
    """How the input is staged before checks read it, when it is copied in a stream:
    ``bounded_batch_copy`` (a borrowed DuckDB query copied into the native backend's
    owned table one bounded Arrow batch at a time) or ``adaptive_snapshot`` (a
    member of a coherent database snapshot, streamed in bounded batches inside the
    snapshot's transaction and kept in memory while the group fits, else written
    to owned spill files). ``None`` for other inputs."""


class RuleExplanation(SnapshotModel):
    rule_id: str
    check_name: str | None = None
    """The check's name in configurations."""
    item_name: str | None = None
    """The check item this rule instance runs for (``None`` for schema contracts)."""
    name: str
    severity: str
    datasets: tuple[str, ...]
    columns: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    dependency_mode: str
    work: tuple[str, ...]


class ValidationPlan(SnapshotModel):
    """Inspectable planning facts, with unknown costs explicitly left unknown."""

    datasets: tuple[DatasetExplanation, ...]
    rules: tuple[RuleExplanation, ...]
    backend: str = "polars"
    """The backend whose admission rules the plan was checked against."""
    selection: str | None = None
    """The selection the plan runs under (``None``: everything configured)."""
    excluded: dict[str, str] = Field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    projection_fallback: tuple[str, ...] = ()
    """Why the native backend reads every dataset in full as one component."""
    overrides: tuple[str, ...] = ()
    """Each applied ``overrides`` entry, in the order applied: the instance it
    changed, the target as written, the settings it set and its file."""
    resources: dict[str, Any] = Field(default_factory=dict)
    profile: ProfileName | None = None
    data_read: bool = False
    complete: bool = True
    notes: tuple[str, ...] = (
        "No input data, file schemas, queries, or loader functions were evaluated.",
        "Declared dependencies do not estimate native working memory or physical I/O.",
        "Input validity and change from a baseline are distinct checks.",
    )


def _work(rule_id: str) -> tuple[str, ...]:
    if rule_id in {"INP-002", "INP-003"}:
        return ("coherent value reductions where required", "bounded failure evidence")
    if rule_id in {"INP-004", "INP-005", "REG-001", "CMP-001"}:
        return ("key/group alignment or indexing may require spill", "exact typed fallback")
    if rule_id.startswith(("TMP-", "SPT-", "STA-")):
        return ("grouped/ordered or numerical work", "work admission depends on observed size")
    return ("execution depends on the configured contract and observed source schema",)


def _enabled_checks(settings: QAVFSettings) -> set[str]:
    """The checks of the enabled items (those a selection left on)."""
    return {placed.item.check for placed in placed_items(settings) if placed.item.enabled}


def _registered_rule_gaps(settings: QAVFSettings, registry: RuleRegistry) -> list[str]:
    """Disclose configured in-code rules without invoking their configuration hooks.

    The runner merges the process registry even when no plugin discovery was
    requested. Reading class metadata is sufficient to identify that omitted
    work, including registrations that would conflict before rule filtering.
    """
    with visibility(settings.plugins.entry_points):
        registered = tuple(default_registry)
    configured = _enabled_checks(settings)
    gaps = []
    for rule in registered:
        if rule.rule_id in registry:
            builtin = registry.get(rule.rule_id)
            if (rule.__module__, rule.__qualname__) != (builtin.__module__, builtin.__qualname__):
                gaps.append(f"registered rule {rule.rule_id!r} conflicts with a built-in rule")
            continue
        if rule.check_name in configured:
            gaps.append(
                f"registered check {rule.check_name!r} was not planned; "
                "load_plugins=True allows its configuration hooks"
            )
    return gaps


def _known_checks_only(settings: QAVFSettings, known: set[str]) -> QAVFSettings:
    """``settings`` without check items whose check is not in ``known``."""
    datasets = {
        name: spec.model_copy(
            update={"checks": [item for item in spec.checks if item.check in known]}
        )
        for name, spec in settings.datasets.items()
    }
    return settings.model_copy(
        update={
            "datasets": datasets,
            "checks": [item for item in settings.checks if item.check in known],
        }
    )


class InputStaging:
    """Name the streamed staging an input receives, from declarations alone."""

    NOTES: ClassVar[Mapping[str, str]] = {
        "bounded_batch_copy": (
            "The borrowed query is copied into an owned table in bounded Arrow batches; "
            "one batch crosses the connection at a time."
        ),
        "adaptive_snapshot": (
            "Snapshot members are streamed in bounded batches inside one transaction and "
            "published together after it commits. They stay in memory while the group "
            "fits the cache and memory target; a larger group is written to owned spill "
            "files instead, so it needs spill space for the whole group, not memory."
        ),
    }

    def __init__(self, settings: QAVFSettings, backend: str) -> None:
        self.settings = settings
        self.backend = backend

    def describe(self, value: Any, source: Mapping[str, Any]) -> str | None:
        import duckdb

        from qavf.inputs.descriptors import Query

        if isinstance(value, Query) and isinstance(value.connection, duckdb.DuckDBPyConnection):
            return "bounded_batch_copy" if self.backend == "duckdb" else None
        if value is None and self._snapshot_member(source):
            return "adaptive_snapshot"
        return None

    def _snapshot_member(self, source: Mapping[str, Any]) -> bool:
        from qavf.sources.sql import referenced_connections

        connections = self.settings.connections
        kind = source.get("type")
        attached: list[str]
        if kind == "database":
            attached = [str(source.get("connection"))]
        elif kind == "query":
            listed = source.get("connections")
            attached = (
                [str(name) for name in listed]
                if listed is not None
                else referenced_connections(str(source.get("sql") or ""), connections)
            )
        else:
            return False
        if len(attached) != 1:
            return False
        spec = connections.get(attached[0])
        return spec is not None and spec.snapshot


def explain(
    data: ValidationInput = None,
    *,
    schema: SchemaSpec | Mapping[str, Any] | None = None,
    config: Configuration = None,
    resources: ResourceLimits | None = None,
    profile: ProfileName | None = None,
    load_plugins: bool = False,
    backend: str = "auto",
    only: str | Iterable[str] | None = None,
    tags: str | Iterable[str] | None = None,
    skip_tags: str | Iterable[str] | None = None,
    datasets: str | Iterable[str] | None = None,
) -> ValidationPlan:
    """Explain checks without reading data or adapting caller-owned inputs.

    The default does not import configured/installed plugin code. Unknown plugin
    checks are disclosed as unresolved. ``load_plugins=True`` opts into normal
    plugin import and rule-configuration hooks. QAVF does not request dataset
    evaluation, but trusted plugin hooks can perform arbitrary Python I/O.
    No filesystem/source inspection is performed; file-based configuration itself
    is read when ``config`` names a configuration file.

    ``backend``, ``only``, ``tags``, ``skip_tags`` and ``datasets`` mean what they
    mean for :func:`~qavf.validate`: the plan applies the same selection and the
    same backend admission, so a check the chosen backend cannot run raises
    :class:`~qavf.optional.OptionalDependencyError` here, before any data is read.

    Raises:
        pydantic.ValidationError: for an invalid configuration (with field paths).
        SelectionError: for an unknown check name or dataset, or an empty selection.
    """
    if not isinstance(load_plugins, bool):
        raise TypeError("load_plugins must be a boolean")
    selection = Selection.of(only=only, tags=tags, skip_tags=skip_tags, datasets=datasets)
    chosen = resolve_backend(backend)
    settings, inputs = prepare_configuration(
        data, schema=schema, config=config, resources=resources
    )
    if profile is not None:
        settings = apply_profile(settings, profile)
    settings = narrow(settings, selection)
    native: NativePlan | None = None
    if chosen == "duckdb":
        # The native backend's own admission: unsupported work is refused, not dropped.
        from qavf.core.configuration import prepare as native_admission
        from qavf.core.planning import NativePlan
        from qavf.core.results import RuleSwitches

        native_settings, native_inputs = native_admission(
            data, schema, config, resources, profile, selection
        )
        native = NativePlan.build(native_settings, RuleSwitches(native_settings), native_inputs)
    descriptions = []
    for name, spec in settings.datasets.items():
        source = spec.source_config() or {}
        kind = type(inputs[name]).__name__ if name in inputs else str(source.get("type", "unbound"))
        notes = ["Schema and row count require execution; declared columns are not a projection."]
        if kind in {"Loader", "BatchSource", "LazyFrame"}:
            notes.append("A coherent owned observation may require memory or temporary disk.")
        staging = InputStaging(settings, chosen).describe(inputs.get(name), source)
        if staging is not None:
            notes.append(InputStaging.NOTES[staging])
        descriptions.append(
            DatasetExplanation(
                name=name,
                input_kind=kind,
                declared_columns=tuple(spec.schema_spec.columns) if spec.schema_spec else (),
                row_budget=spec.effective_load_policy().row_budget,
                notes=tuple(notes),
                staging=staging,
                **(native.describe(name) if native is not None else {}),
            )
        )
    unresolved: list[str] = []
    if load_plugins:
        registry, sources = build_registry(settings), build_source_registry(settings)
        planned_settings = settings
    else:
        registry = RuleRegistry().discover(packages=("qavf.rules",), use_entry_points=False)
        sources = SourceRegistry().discover(use_entry_points=False)
        known = {rule.check_name for rule in registry}
        unresolved.extend(_registered_rule_gaps(settings, registry))
        unresolved.extend(
            f"plugin check {check!r} was not imported"
            for check in sorted(_enabled_checks(settings) - known)
        )
        plugins = settings.plugins
        if (
            plugins.entry_points
            or plugins.modules
            or plugins.paths
            or (set(plugins.packages) - {"qavf.rules"})
            or plugins.required
        ):
            unresolved.append("Configured or installed plugin discovery was not performed.")
        planned_settings = _known_checks_only(settings, known).model_copy(
            update={
                "plugins": plugins.model_copy(
                    update={
                        "packages": ["qavf.rules"],
                        "modules": [],
                        "paths": [],
                        "entry_points": False,
                        "required": [],
                    }
                ),
            }
        )
    rules: list[RuleExplanation] = []
    excluded: dict[str, str] = {}
    try:
        compiled = compile_rules(planned_settings, registry, sources)
        dependencies = build_plan(compiled.rules, planned_settings, settings.datasets).describe()
        for rule, dependency in zip(compiled.rules, dependencies, strict=True):
            rules.append(
                RuleExplanation(
                    rule_id=rule.rule_id,
                    check_name=rule.check_name,
                    item_name=rule.binding.name if rule.binding else None,
                    name=rule.name,
                    severity=rule.severity.value,
                    datasets=tuple(dependency["datasets"]),
                    columns=dependency["columns"],
                    dependency_mode=dependency["dependency_mode"],
                    work=_work(rule.rule_id),
                )
            )
        excluded = dict(compiled.excluded)
    except ConfigurationError as exc:
        if load_plugins:
            raise
        unresolved.append(f"Plan could not be fully compiled: {exception_details(exc)}")
    limits = settings.engine
    return ValidationPlan(
        datasets=tuple(descriptions),
        rules=tuple(rules),
        backend=chosen,
        selection=selection.describe() if selection.active else None,
        excluded=excluded,
        unresolved=tuple(unresolved),
        projection_fallback=native.fallback if native is not None else (),
        overrides=tuple(applied.describe() for applied in settings.applied_overrides),
        complete=not unresolved,
        profile=profile,
        resources={
            key: getattr(limits, key)
            for key in (
                "memory_budget_bytes",
                "cache_max_bytes",
                "spill_max_bytes",
                "max_workers",
                "native_threads",
                "max_affected_records",
                "max_evidence_bytes",
                "max_report_bytes",
                "execution_mode",
            )
        },
    )


__all__ = ["DatasetExplanation", "RuleExplanation", "ValidationPlan", "explain"]
