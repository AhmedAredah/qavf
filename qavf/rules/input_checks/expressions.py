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
from qavf.input_rules import (
    INPUT_RULES,
    assertion_not_evaluated,
    assertion_rows,
    assertions_hold,
    expression_input_unavailable,
    expression_not_evaluated,
    expression_violations,
    expressions_hold,
)
from qavf.redaction import exception_details, redact_text
from qavf.rules.helpers import (
    fresh_internal,
    outcome,
    present,
)
from qavf.settings import ExpressionCheck, QAVFSettings, SqlAssertion
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
    check_name: ClassVar[str] = INPUT_RULES["INP-006"].check_name
    name: ClassVar[str] = INPUT_RULES["INP-006"].name
    description: ClassVar[str] = INPUT_RULES["INP-006"].description
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = INPUT_RULES["INP-006"].default_severity
    recommendation: ClassVar[str] = INPUT_RULES["INP-006"].recommendation
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        return None  # Expressions may query any supplied frame via subqueries.

    def item_name(self, finding: Finding | None) -> str | None:
        """The ``expression`` item a finding reports on (``None`` for the summary)."""
        if finding is None or finding.check_id != "row.expression" or not finding.datasets:
            return None
        return finding.parameters.get("name") or f"{self.check_name} on {finding.datasets[0]}"

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        identities: list[dict[str, Any]] = []
        passed_ids: list[str] = []
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
                            expression_input_unavailable(
                                expression.name or expression.expr, dataset, reason
                            ),
                            check=f"{dataset}: {expression.name or expression.expr}",
                            severity=expression.severity,
                            item_ids=_ids(expression),
                            **_expression_identity(dataset, expression),
                        )
                        for expression in spec.expressions_of(dataset)
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
                            expression_input_unavailable(
                                expression.name or expression.expr, dataset, _reason(exc)
                            ),
                            check=f"{dataset}: {expression.name or expression.expr}",
                            severity=expression.severity,
                            item_ids=_ids(expression),
                            **_expression_identity(dataset, expression),
                        )
                        for expression in spec.expressions_of(dataset)
                    )
                    continue
                found, successful, assessed, ids = self._evaluate_observed(
                    ctx, dataset, rows, observed
                )
                failures.extend(found)
                passed.extend(successful)
                identities.extend(assessed)
                passed_ids.extend(ids)
        return outcome(
            failures,
            passed,
            expressions_hold(len(passed)),
            INPUT_RULES[self.rule_id].unconfigured,
            check_id="row.expression.summary",
            checks=tuple(identities),
            item_ids=tuple(passed_ids),
        )

    @staticmethod
    def _evaluate_observed(
        ctx: DataContext, dataset: str, rows: int, observed: Observation
    ) -> tuple[list[Finding], list[str], list[dict[str, Any]], list[str]]:
        """Failures, then the labels, identities and instance ids of the passes."""
        failures: list[Finding] = []
        passed: list[str] = []
        identities: list[dict[str, Any]] = []
        passed_ids: list[str] = []
        table = quote_identifier(dataset)
        located = ViolatingRows(ctx, dataset, observed)
        for expression in ctx.specs[dataset].expressions_of(dataset):
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
                        expression_not_evaluated(label, dataset, _reason(exc)),
                        check=f"{dataset}: {label}",
                        severity=expression.severity,
                        item_ids=_ids(expression),
                        **identity,
                    )
                )
                continue
            if not count:
                passed.append(f"{dataset}: {label}")
                identities.append(identity)
                passed_ids.extend(_ids(expression))
                continue
            bad = located.sample(condition, bad)
            failures.append(
                Finding.fail(
                    expression_violations(count, rows, dataset, expression.expr),
                    check=f"{dataset}: {label}",
                    severity=expression.severity,
                    affected=bad,
                    affected_count=count,
                    total_records=rows,
                    unit="rows",
                    item_ids=_ids(expression),
                    **identity,
                )
            )
        return failures, passed, identities, passed_ids


def _ids(check: ExpressionCheck | SqlAssertion) -> tuple[str, ...]:
    """The instance id of an expression's or assertion's check item, as ``item_ids``."""
    return (check.item_id,) if check.item_id is not None else ()


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
    check_name: ClassVar[str] = INPUT_RULES["INP-007"].check_name
    name: ClassVar[str] = INPUT_RULES["INP-007"].name
    description: ClassVar[str] = INPUT_RULES["INP-007"].description
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = INPUT_RULES["INP-007"].default_severity
    recommendation: ClassVar[str] = INPUT_RULES["INP-007"].recommendation

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        # SQL may reference supplied runtime frames absent from settings.
        return None

    def item_name(self, finding: Finding | None) -> str | None:
        """The ``sql`` item a finding reports on (``None`` for the summary)."""
        if finding is None or finding.check_id != "sql.assertion":
            return None
        return finding.check

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
                        assertion_not_evaluated(name, _reason(exc)),
                        check=name,
                        severity=severity,
                        item_ids=_ids(assertion),
                    )
                )
                continue
            if not count:
                passed.append(name)
                continue
            failures.append(
                Finding.fail(
                    assertion_rows(name, count),
                    check=name,
                    severity=severity,
                    affected=bad,
                    affected_count=count,
                    unit="rows",
                    datasets=tuple(ctx.referenced(sql)),
                    item_ids=_ids(assertion),
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
            assertions_hold(len(passed)),
            INPUT_RULES[self.rule_id].unconfigured,
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
            item_ids=tuple(
                item_id
                for assertion in ctx.assertions
                if assertion.name in passed
                for item_id in _ids(assertion)
            ),
        )
