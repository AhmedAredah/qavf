"""Explicit not-applicable populations and all-or-none groups (INP-009)."""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar

import polars as pl

from qavf.applicability import AllOrNoneApplicability, NotApplicablePolicy
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.input_rules import (
    INPUT_RULES,
    na_group_absent,
    na_group_label,
    na_group_satisfied,
    na_group_violations,
    not_applicable_population,
)
from qavf.rules._row_summary import row_aliases, summarize_rows
from qavf.rules.helpers import present
from qavf.settings import SchemaSpec
from qavf.sources.sql import quote_identifier, sql_literal

from ._datasets import CATEGORY, DatasetSelection, _schemas, _SelectsDatasets


def _select_projections(projections: list[Any], frame: pl.LazyFrame) -> pl.LazyFrame:
    """Reduce a frame to the given aggregate projections."""
    return frame.select(projections)


def _classify_rows(
    shown: list[str],
    row: str,
    label: str,
    problem: str,
    policy: AllOrNoneApplicability,
    actual: Any,
    frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """Number rows and label those that are only partly not applicable."""
    return (
        frame.select(shown)
        .with_row_index(row)
        .with_columns(pl.when(policy.polars(actual)).then(pl.lit(label)).alias(problem))
    )


class ApplicabilityRule(_SelectsDatasets):
    """Count valid N/A values separately and check configured row groups."""

    rule_id: ClassVar[str] = "INP-009"
    check_name: ClassVar[str] = INPUT_RULES["INP-009"].check_name
    name: ClassVar[str] = INPUT_RULES["INP-009"].name
    description: ClassVar[str] = INPUT_RULES["INP-009"].description
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = INPUT_RULES["INP-009"].default_severity
    recommendation: ClassVar[str] = INPUT_RULES["INP-009"].recommendation
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def evaluate(self, ctx) -> list[Finding]:
        schemas, findings = _schemas(ctx, self.params.datasets, "not-applicable values")
        findings = [
            finding
            for finding in findings
            if any(
                ctx.specs[name].schema_spec is not None
                and any(spec.na_values for spec in ctx.specs[name].schema_spec.columns.values())
                for name in finding.datasets
            )
        ]
        for dataset, schema in schemas.items():
            columns = [name for name, spec in schema.columns.items() if spec.na_values]
            if not columns:
                continue
            try:
                findings.extend(self._counts(ctx, dataset, schema, columns))
                for group in schema.na_groups:
                    findings.append(self._group(ctx, dataset, schema, group))
            except Exception as exc:
                findings.append(
                    Finding.unassessed(
                        f"Not assessed: N/A applicability on {dataset}: "
                        f"{type(exc).__name__}: {exc}.",
                        check=dataset,
                        datasets=(dataset,),
                    )
                )
        return findings or [Finding.skip(INPUT_RULES[self.rule_id].unconfigured)]

    def _counts(self, ctx, dataset: str, schema: SchemaSpec, columns: list[str]) -> list[Finding]:
        actual = ctx.schema(dataset)
        present_columns = [name for name in columns if name in actual]
        if not present_columns:
            return []
        policies = [NotApplicablePolicy.from_spec(schema.columns[name]) for name in present_columns]
        if ctx.execution_kind(dataset) == "native_sql":
            projections = ["count(*) AS rows"] + [
                f"count(*) FILTER (WHERE {policy.sql(quote_identifier(name))}) AS n{i}"
                for i, (name, policy) in enumerate(zip(present_columns, policies, strict=True))
            ]
            result = ctx.sql(
                f"SELECT {', '.join(projections)} FROM {quote_identifier(dataset)}",
                datasets=(dataset,),
            )
        else:
            projections = [pl.len().alias("rows")] + [
                policy.polars(name, actual[name]).sum().alias(f"n{i}")
                for i, (name, policy) in enumerate(zip(present_columns, policies, strict=True))
            ]
            result = ctx.reduce(dataset, partial(_select_projections, projections))
        rows = int(result["rows"][0])
        return [
            Finding.ok(
                not_applicable_population(dataset, name, int(result[f"n{i}"][0] or 0), rows),
                check=f"{dataset}.{name}: N/A population",
                check_id="applicability.population",
                columns=(name,),
                parameters={
                    "na_values": list(policy.tokens),
                    "trim_whitespace": policy.trim_whitespace,
                },
                total_records=rows,
                affected_count=0,
                unit="rows",
                datasets=(dataset,),
                metrics={
                    "na_count": int(result[f"n{i}"][0] or 0),
                    "applicable_count": rows - int(result[f"n{i}"][0] or 0),
                },
            )
            for i, (name, policy) in enumerate(zip(present_columns, policies, strict=True))
        ]

    def _group(self, ctx, dataset: str, schema: SchemaSpec, columns: list[str]) -> Finding:
        actual = ctx.schema(dataset)
        identity = dict(
            check=na_group_label(dataset, columns),
            check_id="applicability.all_or_none",
            columns=tuple(columns),
            parameters={"na_values": {name: schema.columns[name].na_values for name in columns}},
            datasets=(dataset,),
        )
        absent = [column for column in columns if column not in actual]
        if absent:
            return Finding.unassessed(na_group_absent(absent), **identity)
        policy = AllOrNoneApplicability.from_schema(columns, schema)
        row, problem = row_aliases(tuple(actual))
        label = "partly not applicable"
        shown = list(dict.fromkeys([*schema.primary_key, *columns]))
        shown = [name for name in shown if name in actual]
        q = quote_identifier
        if ctx.execution_kind(dataset) == "native_sql":
            classification = (
                f"SELECT {', '.join(q(name) for name in shown)}, "
                f"row_number() OVER () - 1 AS {q(row)}, "
                f"CASE WHEN {policy.sql()} THEN {sql_literal(label)} END AS {q(problem)} "
                f"FROM {q(dataset)}"
            )
            summary = summarize_rows(
                ctx,
                dataset,
                problem=problem,
                labels=(label,),
                sample_columns=[*shown, row, problem],
                sample_order=row,
                sql=classification,
            )
        else:
            classify = partial(_classify_rows, shown, row, label, problem, policy, actual)
            summary = summarize_rows(
                ctx,
                dataset,
                problem=problem,
                labels=(label,),
                sample_columns=[*shown, row, problem],
                build=classify,
            )
        kwargs = dict(
            total_records=summary.rows, affected_count=summary.invalid, unit="rows", **identity
        )
        if not summary.invalid:
            return Finding.ok(
                na_group_satisfied(summary.rows, dataset),
                **kwargs,
            )
        return Finding.fail(
            na_group_violations(summary.invalid, summary.rows, dataset),
            affected=present(summary.sample, {row: "_row", problem: "violates"}),
            **kwargs,
        )
