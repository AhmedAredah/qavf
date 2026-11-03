"""Availability and schema conformance rules (INP-001 and INP-002)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.input_rules import (
    INPUT_RULES,
    dataset_unavailable,
    datasets_accessible,
    row_count_problem,
    schema_violations,
    schemas_conform,
)
from qavf.rules._predicates import (
    is_temporal,
)
from qavf.rules.helpers import (
    absent_column_hint,
    outcome,
)
from qavf.settings import (
    DType,
)
from qavf.sources.base import SourceNotFoundError

from ._conformance import DatasetView, SourceChecks, column_conformance
from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _schemas,
    _SelectsDatasets,
)


def _matches(dtype: pl.DataType, expected: DType) -> bool:
    """Whether a Polars dtype satisfies a logical type."""
    match expected:
        case "int":
            return dtype.is_integer()
        case "float":
            return dtype.is_numeric()
        case "decimal":
            return dtype.is_decimal()
        case "str":
            return dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum)
        case "bool":
            return dtype == pl.Boolean
        case "date":
            return dtype == pl.Date
        case "datetime":
            return is_temporal(dtype)


class RequiredDatasetsRule(_SelectsDatasets):
    """Required datasets are accessible; optional datasets may be absent.

    Native sources are bound and their schemas inspected without a full scan.
    This checks availability, not whether every source record can be parsed.
    """

    rule_id: ClassVar[str] = "INP-001"
    check_name: ClassVar[str] = INPUT_RULES["INP-001"].check_name
    name: ClassVar[str] = INPUT_RULES["INP-001"].name
    description: ClassVar[str] = INPUT_RULES["INP-001"].description
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = INPUT_RULES["INP-001"].default_severity
    recommendation: ClassVar[str] = INPUT_RULES["INP-001"].recommendation
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        loaded: list[str] = []
        selection = self.params.datasets
        # Every declared dataset, including one with no source whose in-memory frame
        # the caller forgot to pass: that must fail, not vanish from the check.
        for name in ctx.specs:
            if selection is not None and name not in selection:
                continue
            if ctx.has(name):
                loaded.append(name)
                continue
            error = ctx.load_error(name)
            required = ctx.specs[name].required
            absent = error is None or isinstance(error, SourceNotFoundError)
            if not required and absent:
                continue
            failures.append(
                Finding.fail(
                    dataset_unavailable(name, required, ctx.unavailable_reason(name)),
                    check=name,
                    severity=None if required else Severity.WARNING,
                    affected=[{"dataset": name, "error": str(error)}],
                    unit="datasets",
                    datasets=(name,),
                    check_id="dataset.available",
                    parameters={"required": required},
                )
            )
        return outcome(
            failures,
            loaded,
            datasets_accessible(len(loaded)),
            INPUT_RULES[self.rule_id].unconfigured,
            check_id="dataset.available.summary",
            checks=tuple(
                {
                    "check_id": "dataset.available",
                    "datasets": (name,),
                    "parameters": {"required": ctx.specs[name].required},
                }
                for name in loaded
            ),
        )


class SchemaConformanceRule(_SelectsDatasets):
    """Columns, logical types and row counts match the configured schema."""

    rule_id: ClassVar[str] = "INP-002"
    check_name: ClassVar[str] = INPUT_RULES["INP-002"].check_name
    name: ClassVar[str] = INPUT_RULES["INP-002"].name
    description: ClassVar[str] = INPUT_RULES["INP-002"].description
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = INPUT_RULES["INP-002"].default_severity
    recommendation: ClassVar[str] = INPUT_RULES["INP-002"].recommendation
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        schemas, blocked = _schemas(ctx, self.params.datasets, "schema check")
        failures: list[Finding] = list(blocked)
        failing: set[str] = {dataset for finding in blocked for dataset in finding.datasets}
        for dataset, schema in schemas.items():
            try:
                with DatasetView.open(ctx, dataset, schema) as view:
                    found = self._dataset(view)
                failures.extend(
                    finding
                    if finding.check_id
                    else replace(
                        finding,
                        check_id="schema.contract",
                        parameters=schema.model_dump(mode="python", exclude_defaults=True),
                    )
                    for finding in found
                )
                if found:
                    failing.add(dataset)
            except Exception as exc:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: {dataset}: {type(exc).__name__}: {exc}.",
                        check=dataset,
                        datasets=(dataset,),
                    )
                )
                failing.add(dataset)
        failing.update(finding.check for finding in failures if finding.check)
        passed = [dataset for dataset in schemas if dataset not in failing]
        return outcome(
            failures,
            passed,
            schemas_conform(len(passed)),
            INPUT_RULES[self.rule_id].unconfigured,
            check_id="schema.summary",
            checks=tuple(
                {
                    "check_id": "schema.contract",
                    "datasets": (dataset,),
                    "parameters": schemas[dataset].model_dump(mode="python", exclude_defaults=True),
                }
                for dataset in passed
            ),
        )

    def _dataset(self, view: DatasetView) -> list[Finding]:
        """Assess independent contracts against the same scoped population."""
        ctx, dataset, schema = view.ctx, view.name, view.schema
        actual_schema, row_count = view.actual, view.rows
        failures: list[Finding] = []
        problems: list[dict[str, Any]] = []
        for column, spec in schema.columns.items():
            if column not in actual_schema:
                if spec.required:
                    problem = {"column": column, "problem": "missing required column"}
                    hint = absent_column_hint(column, actual_schema)
                    if hint:
                        problem["hint"] = hint.removeprefix("; ")
                    problems.append(problem)
                continue
            actual = actual_schema[column]
            conformance = column_conformance(view, column)
            if conformance is not None:
                finding = conformance.finding()
                if finding is not None:
                    failures.append(finding)
                continue
            if spec.dtype is None or _matches(actual, spec.dtype):
                continue
            problems.append({"column": column, "problem": f"expected {spec.dtype}, found {actual}"})
        source = ctx.source(dataset)
        if source is not None:
            rename = ctx.specs[dataset].rename if dataset in ctx.specs else {}
            checks = SourceChecks(view, source, rename)
            found = checks.findings()
            failures.extend(found)
            problems.extend(checks.problems())
        if schema.strict:
            for column in actual_schema:
                if column not in schema.columns:
                    problems.append({"column": column, "problem": "undeclared column"})
        if row_count < schema.min_rows:
            problems.append(
                {
                    "column": None,
                    "problem": row_count_problem(row_count, schema.min_rows, at_least=True),
                }
            )
        if schema.max_rows is not None and row_count > schema.max_rows:
            problems.append(
                {
                    "column": None,
                    "problem": row_count_problem(row_count, schema.max_rows, at_least=False),
                }
            )
        if problems:
            failures.append(
                Finding.fail(
                    schema_violations(len(problems), dataset),
                    check=dataset,
                    affected=problems,
                    unit="problems",
                    datasets=(dataset,),
                )
            )
        return failures
