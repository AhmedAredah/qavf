"""Data-free native CLI services, independent of the full rule registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import Field, create_model

from qavf.capabilities import CORE_RULE_IDS
from qavf.checks.items import ExpressionItem, SqlItem
from qavf.checks.listing import NATIVE_PLACEMENT, ListedCheck
from qavf.engine.severity import Severity
from qavf.input_rules import INPUT_RULES
from qavf.optional import unsupported
from qavf.settings import DatasetSpec, QAVFSettings
from qavf.sources.parameters import DatabaseParams, FileParams, QueryParams


@dataclass(frozen=True, slots=True)
class CoreRule:
    rule_id: str
    name: str
    description: str
    default_severity: Severity
    category: str = "input"


_RULES = tuple(
    CoreRule(rule.rule_id, rule.name, rule.description, rule.default_severity, rule.category)
    for rule_id, rule in INPUT_RULES.items()
    if rule_id in CORE_RULE_IDS
)


def core_rules(settings: QAVFSettings) -> tuple[CoreRule, ...]:
    """List the native catalog without discovering or importing plugin code."""
    plugins = settings.plugins
    if plugins.modules or plugins.paths or plugins.required or plugins.packages != ["qavf.rules"]:
        unsupported("custom rule/source plugin discovery")
    return _RULES


def core_checks(settings: QAVFSettings) -> tuple[ListedCheck, ...]:
    """The native backend's checks: schema contracts plus ``expression`` and ``sql``."""
    return tuple(
        ListedCheck(
            check_name=rule.check_name,
            rule_id=rule.rule_id,
            placement=NATIVE_PLACEMENT.get(rule.rule_id, "schema"),
            category=rule.category,
            default_severity=rule.default_severity.value,
            name=rule.name,
            description=rule.description,
            module="qavf.core.checks",
        )
        for rule in (INPUT_RULES[core.rule_id] for core in core_rules(settings))
    )


def config_json_schema(settings: QAVFSettings) -> dict[str, Any]:
    """Describe native configuration without constructing or opening sources.

    The catalog only restricts check and source vocabulary: the native backend
    runs the schema contracts plus ``expression`` and ``sql`` checks. Cross-field
    policy, SQL safety and backend admission checks are performed at validation time.
    """
    core_rules(settings)
    dataset_model = create_model(
        "NativeDataset",
        __base__=DatasetSpec,
        source=(FileParams | QueryParams | DatabaseParams | None, Field(default=None)),
        checks=(list[ExpressionItem], Field(default_factory=list)),
    )
    root = create_model(
        "NativeQAVFConfig",
        __base__=QAVFSettings,
        datasets=(dict[str, dataset_model], Field(default_factory=dict)),
        checks=(list[SqlItem], Field(default_factory=list)),
    )
    schema = root.model_json_schema(by_alias=True)
    schema["properties"].pop("source_path", None)
    schema["properties"]["$schema"] = {"type": "string"}
    schema["title"] = "QAVF DuckDB configuration"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["x-qavf-backend"] = "duckdb"
    schema["x-qavf-rule-ids"] = list(CORE_RULE_IDS)
    from qavf.schema_hints import apply_ui_hints
    from qavf.schema_reuse import accept_reuse_values

    return apply_ui_hints(accept_reuse_values(schema))
