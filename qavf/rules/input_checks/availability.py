"""Availability and schema conformance rules (INP-001 and INP-002)."""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.rules._declared_types import convert, converts_text, described
from qavf.rules._predicates import (
    is_temporal,
)
from qavf.rules.helpers import (
    absent_column_hint,
    outcome,
    plural,
)
from qavf.settings import (
    ColumnSpec,
    DType,
    QAVFSettings,
    SchemaSpec,
)
from qavf.sources.base import SourceNotFoundError

from . import _column_values as values
from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _row_count,
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


class RequiredDatasetsRule(Rule):
    """Required datasets are accessible; optional datasets may be absent.

    Native sources are bound and their schemas inspected without a full scan.
    This checks availability, not whether every source record can be parsed.
    """

    rule_id: ClassVar[str] = "INP-001"
    name: ClassVar[str] = "Datasets available"
    description: ClassVar[str] = (
        "Required datasets are accessible and their schemas can be inspected; optional "
        "datasets may be absent. Native sources are not fully scanned by this availability "
        "check, so it does not certify that every source record can be parsed."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.CRITICAL
    recommendation: ClassVar[str] = (
        "Check the dataset's source settings (paths, queries, connection details) and confirm "
        "the upstream step that produces it completed. Mark a dataset 'required: false' if it "
        "is genuinely optional."
    )

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        return DataRequirements(frozenset(settings.datasets))

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        loaded: list[str] = []
        # Every declared dataset, including one with no source whose in-memory frame
        # the caller forgot to pass: that must fail, not vanish from the check.
        for name in ctx.specs:
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
                    f"{'Required' if required else 'Optional'} dataset {name!r} is not "
                    f"available: {ctx.unavailable_reason(name)}",
                    check=name,
                    severity=None if required else Severity.WARNING,
                    affected=[{"dataset": name, "error": str(error)}],
                    unit="datasets",
                    datasets=(name,),
                )
            )
        return outcome(
            failures,
            loaded,
            f"{plural(len(loaded), 'dataset')} accessible.",
            "No datasets are configured.",
        )


class SchemaConformanceRule(_SelectsDatasets):
    """Columns, logical types and row counts match the configured schema."""

    rule_id: ClassVar[str] = "INP-002"
    name: ClassVar[str] = "Schema conformance"
    description: ClassVar[str] = (
        "Each dataset has its required columns with the declared types, no undeclared "
        "columns when strict, and a row count within the configured limits."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.CRITICAL
    recommendation: ClassVar[str] = (
        "Fix the step that produced the dataset so column names and types match the schema, "
        "or update the schema if the layout changed intentionally. A column read as text "
        "when another type is declared usually holds a few stray values: the listed rows "
        "show them. Declare placeholders such as 'N/A' in the column's missing_values. A "
        "code declared str but read as a number has lost its leading zeros: read it as "
        "text at the source ('text_columns' for CSV files)."
    )
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        schemas, blocked = _schemas(ctx, self.params.datasets, "schema check")
        failures: list[Finding] = list(blocked)
        failing: set[str] = {dataset for finding in blocked for dataset in finding.datasets}
        for dataset, schema in schemas.items():
            try:
                actual_schema = ctx.schema(dataset)
                row_count = _row_count(ctx, dataset)
            except Exception as exc:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: {dataset}: {type(exc).__name__}: {exc}.",
                        check=dataset,
                        datasets=(dataset,),
                    )
                )
                continue
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
                if spec.dtype is None or _matches(actual, spec.dtype):
                    continue
                if converts_text(spec, actual):
                    finding = _invalid_values(ctx, dataset, schema, actual_schema, column, spec)
                    if finding is not None:
                        failures.append(finding)
                        failing.add(dataset)
                    continue
                problems.append(
                    {"column": column, "problem": f"expected {spec.dtype}, found {actual}"}
                )
            if schema.strict:
                for column in actual_schema:
                    if column not in schema.columns:
                        problems.append({"column": column, "problem": "undeclared column"})
            if row_count < schema.min_rows:
                problems.append(
                    {
                        "column": None,
                        "problem": f"{row_count} rows, expected at least {schema.min_rows}",
                    }
                )
            if schema.max_rows is not None and row_count > schema.max_rows:
                problems.append(
                    {
                        "column": None,
                        "problem": f"{row_count} rows, expected at most {schema.max_rows}",
                    }
                )
            if problems:
                failing.add(dataset)
                failures.append(
                    Finding.fail(
                        f"{plural(len(problems), 'schema violation')} in {dataset}.",
                        check=dataset,
                        affected=problems,
                        unit="problems",
                        datasets=(dataset,),
                    )
                )
        failing.update(finding.check for finding in failures if finding.check)
        passed = [dataset for dataset in schemas if dataset not in failing]
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'dataset')} conform to their schemas.",
            "No available dataset has a schema configured.",
        )


def _invalid_values(
    ctx: DataContext,
    dataset: str,
    schema: SchemaSpec,
    actual_schema: dict[str, pl.DataType],
    column: str,
    spec: ColumnSpec,
) -> Finding | None:
    """The cells of a text column that are neither missing nor of its declared type.

    A column whose every value is of the declared type conforms: it is text only
    because of declared missing markers, and INP-003 checks the converted values.
    """
    assert spec.dtype is not None
    check = f"{dataset}.{column}: dtype {spec.dtype}"
    actual = actual_schema[column]
    context = [c for c in schema.primary_key or [] if c != column and c in actual_schema]
    try:
        frame = ctx.project(dataset, [*context, column])
        invalid = convert(frame[column], spec).invalid
        count = int(invalid.sum())
    except Exception as exc:
        return Finding.unassessed(
            f"Not assessed: {check}: {type(exc).__name__}: {exc}.",
            check=check,
            datasets=(dataset,),
        )
    if not count:
        return None
    kind = described(spec.dtype)
    return Finding.fail(
        f"{plural(count, 'value')} of {frame.height:,} in {dataset}.{column} "
        f"{'is' if count == 1 else 'are'} not {kind}, so the column was read as {actual}.",
        check=check,
        affected=values._rows(frame, invalid, f"not {kind}", ctx.evidence_limit),
        affected_count=count,
        total_records=frame.height,
        unit="rows",
        datasets=(dataset,),
    )
