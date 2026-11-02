"""Applicability adapter around the existing independently planned checks.

Filtering belongs before type conversion: a token is a valid N/A state, not an
invalid value that happens to convert to null. Observations bound memory and keep
volatile inputs coherent; only configured N/A columns enter this adapter.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import polars as pl

from qavf.applicability import NotApplicablePolicy
from qavf.engine.registry import Finding
from qavf.input_rules import all_not_applicable
from qavf.rules._declared_types import ConvertedPlan, TextConversion, converts_text, typed_spec
from qavf.rules._evidence_columns import EvidenceColumns
from qavf.rules.helpers import fresh_internal
from qavf.settings import ColumnSpec, SchemaSpec


def _population_counts(
    policy: NotApplicablePolicy, column: str, dtype: Any, frame: pl.LazyFrame
) -> pl.LazyFrame:
    """Count all rows and the column's N/A rows in one reduction."""
    return frame.select(pl.len().alias("rows"), policy.polars(column, dtype).sum().alias("na"))


def _applicable_rows(
    policy: NotApplicablePolicy,
    column: str,
    dtype: Any,
    original_row: str,
    plan: Any,
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """Number source rows, drop N/A values and apply any declared conversion."""
    selected = frame.with_row_index(original_row).filter(~policy.polars(column, dtype))
    return plan(selected) if plan else selected


class ApplicableColumns:
    """Assess each target on its own applicable population, retaining dependencies."""

    def __init__(self, rule: Any, ctx: Any, dataset: str, schema: SchemaSpec, columns: list[str]):
        self.rule, self.ctx, self.dataset = rule, ctx, dataset
        self.schema, self.columns = schema, columns

    def assess(self) -> tuple[list[Finding], list[str]]:
        findings: list[Finding] = []
        passed: list[str] = []
        for column in self.columns:
            found, good = self._column(column)
            findings.extend(found)
            passed.extend(good)
        return findings, passed

    def _column(self, column: str) -> tuple[list[Finding], list[str]]:
        actual = self.ctx.schema(self.dataset)
        spec = self.schema.columns[column]
        policy = NotApplicablePolicy.from_spec(spec)
        original_row = fresh_internal("applicability_row", {name.casefold() for name in actual})
        needed = [column]
        if spec.unit_column:
            needed.append(spec.unit_column)
        for check in spec.all_checks():
            if check.where:
                needed.extend(pl.sql_expr(check.where).meta.root_names())
        if self.ctx.evidence_limit and self.ctx.evidence_bytes:
            needed.extend(EvidenceColumns(self.schema, tuple(actual)).of(needed))
        with self.ctx.observe(self.dataset, list(dict.fromkeys(needed))) as observed:
            totals = observed.reduce(partial(_population_counts, policy, column, actual[column]))
            rows, na_count = int(totals["rows"][0]), int(totals["na"][0] or 0)
            clean = spec.model_copy(update={"na_values": []})
            conversion = (
                TextConversion(column, clean, actual[column])
                if converts_text(spec, actual[column])
                else None
            )
            plan = ConvertedPlan.resolved([conversion], observed.reduce) if conversion else None
            applicable = partial(
                _applicable_rows, policy, column, actual[column], original_row, plan
            )
            columns = {
                name: ColumnSpec(
                    nullable=True,
                    allow_infinite=True,
                    missing_values=self.schema.columns[name].missing_values,
                    blank_as_missing=self.schema.columns[name].blank_as_missing,
                )
                for name in needed
                if name in self.schema.columns and name != column
            }
            columns[column] = typed_spec(clean) if conversion else clean
            contract = self.schema.model_copy(
                update={"columns": columns, "primary_key": [*self.schema.primary_key, original_row]}
            )
            with observed.derived(applicable) as scoped:
                found, passed = self.rule._planned_dataset(
                    self.ctx, self.dataset, contract, scoped.schema(), scoped
                )
            found = [self._source_positions(finding, original_row, actual) for finding in found]
            if rows and na_count == rows:
                # A statistic over an all-N/A population is undefined, including
                # sum; reporting zero would fabricate an observed numeric value.
                from qavf.input_rules import check_label

                aggregates = {"min", "max", "mean", "median", "std", "sum"}
                aggregate_labels = {
                    f"{self.dataset}.{column}: {check_label(check)}"
                    for check in clean.all_checks()
                    if aggregates.intersection(check.constraints())
                }
                found = [
                    replace(
                        finding,
                        assessed=False,
                        message=all_not_applicable(str(finding.check)),
                    )
                    if finding.check in aggregate_labels
                    else finding
                    for finding in found
                ]
        metrics = {
            "na_count": na_count,
            "applicable_count": rows - na_count,
            "population_count": rows,
        }
        return [
            replace(finding, metrics={**finding.metrics, **metrics}) for finding in found
        ], passed

    @staticmethod
    def _source_positions(finding: Finding, original_row: str, actual: dict) -> Finding:
        """Filtering N/A must never renumber the rows shown as source evidence."""
        affected = finding.affected
        if not isinstance(affected, pl.DataFrame) or original_row not in affected.columns:
            return finding
        row_column = "_row"
        while row_column in actual:
            row_column = "qavf_" + row_column
        affected = affected.with_columns(pl.col(original_row).alias(row_column)).drop(original_row)
        return replace(finding, affected=affected, row_column=row_column)
