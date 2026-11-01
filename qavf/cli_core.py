"""Data-free native CLI services, independent of the full rule registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from qavf.capabilities import CORE_RULE_IDS
from qavf.engine.severity import Severity
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


_RULES = (
    CoreRule("INP-001", "Datasets available", "Configured inputs can be read.", Severity.CRITICAL),
    CoreRule(
        "INP-002", "Schema conformance", "Inputs match their declared schema.", Severity.CRITICAL
    ),
    CoreRule("INP-003", "Column QA limits", "Column values meet declared limits.", Severity.ERROR),
    CoreRule(
        "INP-004",
        "Unique primary keys",
        "Primary, alternate and row keys meet uniqueness contracts.",
        Severity.ERROR,
    ),
    CoreRule(
        "INP-005",
        "Referential integrity",
        "Foreign keys refer to valid parent keys.",
        Severity.ERROR,
    ),
    CoreRule("INP-006", "Row expressions", "Declared row expressions hold.", Severity.ERROR),
    CoreRule(
        "INP-007", "SQL assertions", "Assertion queries return no offending rows.", Severity.ERROR
    ),
    CoreRule(
        "INP-009",
        "Not-applicable values",
        "Column groups meet applicability contracts.",
        Severity.ERROR,
    ),
)


def core_rules(settings: QAVFSettings) -> tuple[CoreRule, ...]:
    """List the native catalog without discovering or importing plugin code."""
    plugins = settings.plugins
    if plugins.modules or plugins.paths or plugins.required or plugins.packages != ["qavf.rules"]:
        unsupported("custom rule/source plugin discovery")
    return _RULES


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _DatasetParams(_NoParams):
    datasets: list[str] | None = None


def config_json_schema(settings: QAVFSettings) -> dict[str, Any]:
    """Describe native configuration without constructing or opening sources.

    The catalog only restricts rule and source vocabulary. Cross-field policy,
    SQL safety and backend admission checks are performed at validation time.
    """
    rules = core_rules(settings)
    strict = ConfigDict(extra="forbid", populate_by_name=True)
    fields: dict[str, Any] = {}
    for rule in rules:
        params = _NoParams if rule.rule_id in {"INP-001", "INP-007"} else _DatasetParams
        model_name = "NativeRule_" + rule.rule_id.replace("-", "_")
        model = create_model(
            model_name,
            __config__=strict,
            enabled=(bool, True),
            severity=(Severity | None, None),
            params=(params, Field(default_factory=params)),
        )
        fields[rule.rule_id.replace("-", "_")] = (
            model,
            Field(default_factory=model, alias=rule.rule_id, description=rule.description),
        )
    rule_model = create_model("NativeRules", __config__=strict, **fields)
    dataset_model = create_model(
        "NativeDataset",
        __base__=DatasetSpec,
        source=(FileParams | QueryParams | DatabaseParams | None, Field(default=None)),
    )
    root = create_model(
        "NativeQAVFConfig",
        __base__=QAVFSettings,
        datasets=(dict[str, dataset_model], Field(default_factory=dict)),
        rules=(rule_model, Field(default_factory=rule_model)),
    )
    schema = root.model_json_schema(by_alias=True)
    schema["properties"].pop("source_path", None)
    schema["properties"]["$schema"] = {"type": "string"}
    schema["title"] = "QAVF DuckDB configuration"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["x-qavf-backend"] = "duckdb"
    schema["x-qavf-rule-ids"] = list(CORE_RULE_IDS)
    return schema
