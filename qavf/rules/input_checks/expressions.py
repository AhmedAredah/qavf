"""Row expressions and cross-dataset SQL assertions (INP-006 and INP-007)."""

from __future__ import annotations

from typing import ClassVar

import duckdb

from qavf.engine.context import DataContext, DatasetNotAvailableError
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.rules.helpers import (
    outcome,
    plural,
)
from qavf.settings import (
    QAVFSettings,
)
from qavf.sources.base import SourceError
from qavf.sources.sql import quote_identifier

from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _must_assess,
    _row_count,
    _SelectsDatasets,
)


class ExpressionRule(_SelectsDatasets):
    """Row-level SQL expressions hold for every row."""

    rule_id: ClassVar[str] = "INP-006"
    name: ClassVar[str] = "Row expressions"
    description: ClassVar[str] = (
        "Every row satisfies the dataset's configured SQL expressions, e.g. "
        "'pm25 <= pm10' or 'end_date >= start_date'."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "An expression failure means a relationship between columns that must always hold "
        "does not. Inspect the listed rows for mismatched units or rows combined from "
        "different sources."
    )
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        return None  # Expressions may query any supplied frame via subqueries.

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for dataset, spec in ctx.specs.items():
            selection = self.params.datasets
            if (selection is not None and dataset not in selection) or not spec.expressions:
                continue
            if not ctx.has(dataset):
                if _must_assess(ctx, dataset, selection):
                    failures += [
                        Finding.unassessed(
                            f"Not assessed: {ctx.unavailable_reason(dataset)}.",
                            check=f"{dataset}: {expression.name or expression.expr}",
                            severity=expression.severity,
                            datasets=(dataset,),
                        )
                        for expression in spec.expressions
                    ]
                continue
            rows = _row_count(ctx, dataset)
            table = quote_identifier(dataset)
            for expression in spec.expressions:
                label = expression.name or expression.expr
                fallback = "true" if expression.null_as == "pass" else "false"
                query = f"SELECT * FROM {table} WHERE NOT coalesce(({expression.expr}), {fallback})"
                try:
                    # Dependencies come from parsing the whole query, so names inside
                    # literals or comments are not mistaken for datasets. Offending rows
                    # are counted in full but only a sample is materialised.
                    bad, count = ctx.sql_sample(query, ctx.evidence_limit, normalize_missing=True)
                except (duckdb.Error, DatasetNotAvailableError, SourceError) as exc:
                    # Each expression stands alone: one that cannot run never hides the
                    # results of the others.
                    failures.append(
                        Finding.unassessed(
                            f"Not assessed: expression '{label}' on {dataset} could not be "
                            f"evaluated: {_reason(exc)}",
                            check=f"{dataset}: {label}",
                            severity=expression.severity,
                            datasets=(dataset,),
                        )
                    )
                    continue
                if not count:
                    passed.append(f"{dataset}: {label}")
                    continue
                failures.append(
                    Finding.fail(
                        f"{plural(count, 'row')} of {rows:,} in {dataset} "
                        f"{'violates' if count == 1 else 'violate'} '{expression.expr}'.",
                        check=f"{dataset}: {label}",
                        severity=expression.severity,
                        affected=bad,
                        affected_count=count,
                        total_records=rows,
                        unit="rows",
                        datasets=(dataset,),
                    )
                )
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'row expression')} hold.",
            "No row expressions are configured.",
        )


def _reason(exc: Exception) -> str:
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


class SqlAssertionRule(Rule):
    """Configured SQL assertions return no rows."""

    rule_id: ClassVar[str] = "INP-007"
    name: ClassVar[str] = "SQL assertions"
    description: ClassVar[str] = (
        "Each configured SQL assertion, a query that selects offending rows across any "
        "datasets, returns nothing."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Each returned row is a counter-example to the assertion. Run the assertion's query "
        "yourself to explore them."
    )

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        # SQL may reference supplied runtime frames absent from settings.
        return None

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for assertion in ctx.assertions:
            name, sql, severity = assertion.name, assertion.sql, assertion.severity
            try:
                # Counted in full; the kept sample is the first rows by every column,
                # so it is the same on every run (SQL result order is not guaranteed).
                bad, count = ctx.sql_sample(
                    sql, ctx.evidence_limit, ordered=True, normalize_missing=True
                )
            except (duckdb.Error, DatasetNotAvailableError, SourceError) as exc:
                # e.g. a database connection that cannot be attached: this assertion only.
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: assertion '{name}' could not be evaluated: {_reason(exc)}",
                        check=name,
                        severity=severity,
                    )
                )
                continue
            if not count:
                passed.append(name)
                continue
            failures.append(
                Finding.fail(
                    f"Assertion '{name}' returned {plural(count, 'offending row')}.",
                    check=name,
                    severity=severity,
                    affected=bad,
                    affected_count=count,
                    unit="rows",
                    datasets=tuple(ctx.referenced(sql)),
                )
            )
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'SQL assertion')} hold.",
            "No SQL assertions are configured.",
        )
