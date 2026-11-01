"""Row expressions and cross-dataset SQL assertions (INP-006 and INP-007)."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from typing import Any, ClassVar

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.observation_sql import sample_observed
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.redaction import exception_details, redact_text
from qavf.rules.helpers import (
    fresh_internal,
    outcome,
    plural,
    present,
)
from qavf.settings import (
    QAVFSettings,
)
from qavf.sources.sql import quote_identifier

from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _must_assess,
    _SelectsDatasets,
)

logger = logging.getLogger(__name__)


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
        identities: list[dict[str, Any]] = []
        known_unavailable = ctx.unavailable()
        for dataset, spec in ctx.specs.items():
            selection = self.params.datasets
            if (selection is not None and dataset not in selection) or not spec.expressions:
                continue
            # Availability probing can evaluate a parameter-bound native query
            # on older DuckDB versions. Consult known failures without reading;
            # the owned observation below performs the first actual read.
            if (reason := known_unavailable.get(dataset)) is not None:
                if _must_assess(ctx, dataset, selection):
                    failures += [
                        Finding.unassessed(
                            f"Not assessed: {reason}.",
                            check=f"{dataset}: {expression.name or expression.expr}",
                            severity=expression.severity,
                            **_expression_identity(dataset, expression),
                        )
                        for expression in spec.expressions
                    ]
                continue
            with contextlib.ExitStack() as stack:
                try:
                    observed = stack.enter_context(ctx.observe(dataset))
                    rows = int(observed.reduce(lambda frame: frame.select(pl.len())).item())
                except Exception as exc:
                    if not _must_assess(ctx, dataset, selection):
                        continue
                    failures.extend(
                        Finding.unassessed(
                            f"Not assessed: expression '{expression.name or expression.expr}' "
                            f"on {dataset} could not read its input: {_reason(exc)}",
                            check=f"{dataset}: {expression.name or expression.expr}",
                            severity=expression.severity,
                            **_expression_identity(dataset, expression),
                        )
                        for expression in spec.expressions
                    )
                    continue
                found, successful, assessed = self._evaluate_observed(ctx, dataset, rows, observed)
                failures.extend(found)
                passed.extend(successful)
                identities.extend(assessed)
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'row expression')} hold.",
            "No row expressions are configured.",
            check_id="row.expression.summary",
            checks=tuple(identities),
        )

    @staticmethod
    def _evaluate_observed(
        ctx: DataContext, dataset: str, rows: int, observed: Observation
    ) -> tuple[list[Finding], list[str], list[dict[str, Any]]]:
        failures: list[Finding] = []
        passed: list[str] = []
        identities: list[dict[str, Any]] = []
        table = quote_identifier(dataset)
        located = ViolatingRows(ctx, dataset, observed)
        for expression in ctx.specs[dataset].expressions:
            identity = _expression_identity(dataset, expression)
            label = expression.name or expression.expr
            fallback = "true" if expression.null_as == "pass" else "false"
            condition = f"NOT coalesce(({expression.expr}), {fallback})"
            query = f"SELECT * FROM {table} WHERE {condition}"
            try:
                bad, count = sample_observed(
                    ctx, query, {dataset: observed}, ctx.evidence_limit, normalize_missing=True
                )
            except Exception as exc:
                # Each expression stands alone: one that cannot run never hides the
                # results of the others.
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: expression '{label}' on {dataset} could not be "
                        f"evaluated: {_reason(exc)}",
                        check=f"{dataset}: {label}",
                        severity=expression.severity,
                        **identity,
                    )
                )
                continue
            if not count:
                passed.append(f"{dataset}: {label}")
                identities.append(identity)
                continue
            bad = located.sample(condition, bad)
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
                    **identity,
                )
            )
        return failures, passed, identities


def _expression_identity(dataset: str, expression: Any) -> dict[str, Any]:
    """Keep the configured contract with its result; labels need not be unique."""
    return {
        "check_id": "row.expression",
        "datasets": (dataset,),
        "parameters": expression.model_dump(mode="python"),
    }


class ViolatingRows:
    """The first rows violating an expression, with their ``_row`` positions.

    Positions need every row numbered, which holds the scan, so they are looked up
    only after an expression has failed, on the same observation. A lookup that
    fails leaves the evidence without positions.
    """

    def __init__(self, ctx: DataContext, dataset: str, observed: Observation) -> None:
        self.ctx = ctx
        self.dataset = dataset
        self.observed = observed

    def sample(self, condition: str, unlocated: pl.DataFrame) -> pl.DataFrame:
        try:
            return self._located(condition)
        except Exception as exc:
            logger.warning(
                "%s",
                redact_text(
                    f"Row positions of expression failures in {self.dataset}: "
                    f"{exception_details(exc)}"
                ),
            )
            return unlocated

    def _located(self, condition: str) -> pl.DataFrame:
        q = quote_identifier
        table = q(self.dataset)
        row = fresh_internal("row", [name.casefold() for name in self.observed.schema()])
        limit = self.ctx.evidence_limit
        # The numbered rows keep the dataset's name, so the expression reads unchanged.
        query = (
            f"SELECT * FROM (SELECT *, row_number() OVER () - 1 AS {q(row)} FROM {table}) "
            f"AS {table} WHERE {condition} ORDER BY {q(row)} LIMIT {limit}"
        )
        sample, _ = sample_observed(
            self.ctx, query, {self.dataset: self.observed}, limit, normalize_missing=True
        )
        return present(sample, {row: "_row"})


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
            except Exception as exc:
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
            [
                replace(
                    finding,
                    check_id="sql.assertion",
                    parameters=next(
                        assertion.model_dump(mode="python")
                        for assertion in ctx.assertions
                        if assertion.name == finding.check
                    ),
                )
                for finding in failures
            ],
            passed,
            f"{plural(len(passed), 'SQL assertion')} hold.",
            "No SQL assertions are configured.",
            check_id="sql.assertion.summary",
            checks=tuple(
                {
                    "check_id": "sql.assertion",
                    "scope": "run",
                    "parameters": assertion.model_dump(mode="python"),
                }
                for assertion in ctx.assertions
                if assertion.name in passed
            ),
        )
