"""Column rule orchestration, fused plans and projected exact fallbacks (INP-003)."""

from __future__ import annotations

from typing import ClassVar

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.rules._column_plan import column_checks, column_plans
from qavf.rules._execution import (
    projected_frame,
)
from qavf.rules.helpers import (
    outcome,
    plural,
)
from qavf.settings import (
    ColumnSpec,
    SchemaSpec,
)

from . import _column_values as values
from ._column_eager import evaluate_eager
from ._column_simple import evaluate_simple
from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _schemas,
    _SelectsDatasets,
)


class ColumnCheckRule(_SelectsDatasets):
    """Every configured column limit holds; each check is graded at its own severity."""

    rule_id: ClassVar[str] = "INP-003"
    name: ClassVar[str] = "Column QA limits"
    description: ClassVar[str] = (
        "Column values respect their configured QA limits: nulls, bounds, allowed and "
        "forbidden values, patterns, lengths, uniqueness and aggregate statistics."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Inspect the listed rows. Out-of-range values usually come from unit mix-ups, sign "
        "errors, sentinel values such as -999, or failed joins that leave nulls. If a value "
        "is legitimate, widen the limit or downgrade that check's severity."
    )
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        schemas, blocked = _schemas(ctx, self.params.datasets, "column QA limits")
        failures += blocked
        for dataset, schema in schemas.items():
            try:
                dataset_failures, dataset_passed = self._dataset(ctx, dataset, schema)
            except Exception as exc:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: {dataset}: {type(exc).__name__}: {exc}.",
                        check=dataset,
                        datasets=(dataset,),
                    )
                )
                continue
            failures += dataset_failures
            passed += dataset_passed
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'column check')} passed.",
            "No column QA limits are configured.",
        )

    def _dataset(
        self, ctx: DataContext, dataset: str, schema: SchemaSpec
    ) -> tuple[list[Finding], list[str]]:
        kind = ctx.execution_kind(dataset)
        if kind not in ("lazy", "native_sql"):
            return evaluate_eager(ctx, dataset, schema)
        # Preserve fused reductions when every contract fits the simple backend;
        # an unsupported sibling now selects per-check plans, never a full load.
        fused = evaluate_simple(ctx, dataset, schema)
        if fused is not None:
            return fused
        actual = ctx.schema(dataset)
        failures: list[Finding] = []
        passed: list[str] = []
        for column, spec in schema.columns.items():
            if column not in actual and spec.required:
                failures.append(values._missing_column(dataset, column))
        items = column_plans(schema, actual)
        assessed = column_checks(
            ctx,
            dataset,
            schema,
            actual,
            items,
            lambda item: values._check_name(item.check) if item.check is not None else item.kind,
            values._statistic,
            values._within,
            values._show,
        )
        for item, finding in zip(items, assessed, strict=True):
            label = values._check_name(item.check) if item.check is not None else item.kind
            try:
                if isinstance(finding, Exception):
                    raise finding
                if finding is not NotImplemented:
                    if finding is None:
                        passed.append(f"{dataset}.{item.column}: {label}")
                    else:
                        failures.append(finding)
                    continue
                single = item.single_schema()
                # Unsupported exact operations project only this contract's
                # dependencies; siblings retain their source-side reductions.
                dependencies = list(item.dependencies)
                projected = projected_frame(ctx, dataset, dependencies)
                # The where condition sees each dependency's missing policy.
                contract = schema.model_copy(
                    update={
                        "columns": {
                            **{
                                c: schema.columns[c].model_copy(update={"checks": []})
                                for c in dependencies
                                if c in schema.columns and c != item.column
                            },
                            **single.columns,
                        }
                    }
                )
                # Only the target has limits; dependency specs carry markers.
                for c in list(contract.columns):
                    if c == item.column:
                        continue
                    dependency = contract.columns[c]
                    contract.columns[c] = ColumnSpec(
                        nullable=True,
                        allow_infinite=True,
                        missing_values=dependency.missing_values,
                        blank_as_missing=dependency.blank_as_missing,
                    )
                found, good = evaluate_eager(ctx, dataset, contract, projected)
                failures.extend(found)
                passed.extend(good)
            except Exception as exc:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: '{label}' on {dataset}.{item.column}: "
                        f"{type(exc).__name__}: {exc}.",
                        check=f"{dataset}.{item.column}: {label}",
                        severity=item.check.severity if item.check else None,
                        datasets=(dataset,),
                    )
                )
        return failures, passed
