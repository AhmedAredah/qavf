"""Fused SQL and lazy reductions for simple column contracts."""

from __future__ import annotations

import math
from decimal import Decimal

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.engine.severity import Severity
from qavf.rules._predicates import (
    decimal_bound,
    float_bound,
    format_value,
    row_tests,
)
from qavf.rules.helpers import (
    fresh_internal,
    missing_expr,
    plural,
    present,
)
from qavf.settings import (
    SchemaSpec,
)
from qavf.sources.sql import quote_identifier

from . import _column_values as values


def evaluate_simple(
    ctx: DataContext,
    dataset: str,
    schema: SchemaSpec,
) -> tuple[list[Finding], list[str]] | None:
    """Native SQL reductions for numeric bounds, missingness and finiteness.

    Complex predicates retain the Polars path, subject to the context's
    materialization budget. Eligibility is decided before reading any rows.
    """
    kind = ctx.execution_kind(dataset)
    if kind not in ("native_sql", "lazy"):
        return None
    actual = ctx.schema(dataset)
    for column, spec in schema.columns.items():
        if column not in actual:
            continue
        if spec.unit_column or spec.missing_values or spec.blank_as_missing:
            return None
        for check in spec.all_checks():
            if check.where or set(check.constraints()) - {"ge", "le", "gt", "lt"}:
                return None
            if not actual[column].is_numeric():
                return None
            for bound in check.constraints().values():
                if not isinstance(bound, int | float | Decimal):
                    return None
                # DuckDB DECIMAL literals have at most 38 digits. Do not
                # silently fall back to a rounded double for larger bounds.
                decimal = Decimal(str(bound))
                exponent = decimal.as_tuple().exponent
                if not isinstance(exponent, int):
                    return None
                if max(0, decimal.adjusted() + 1) + max(0, -exponent) > 38:
                    return None
    if kind == "lazy":
        return _lazy_dataset(ctx, dataset, schema, actual)
    table = quote_identifier(dataset)
    rows = 0
    jobs: list[tuple[str, str, list[tuple[str, str]], Severity | None]] = []
    failures: list[Finding] = []
    passed: list[str] = []
    row_name = fresh_internal("row", actual)
    reason_name = fresh_internal("violates", [*actual, row_name])
    base = f"(SELECT row_number() OVER () - 1 AS {quote_identifier(row_name)}, * FROM {table})"

    def record(
        column: str,
        label: str,
        conditions: list[tuple[str, str]],
        severity: Severity | None = None,
    ) -> None:
        jobs.append((column, label, conditions, severity))

    def emit(
        column: str,
        label: str,
        conditions: list[tuple[str, str]],
        severity: Severity | None,
        count: int,
    ) -> None:
        if not count:
            passed.append(f"{dataset}.{column}: {label}")
            return
        bad = " OR ".join(f"({predicate})" for _, predicate in conditions)
        labels = ", ".join(
            "CASE WHEN (" + predicate + ") THEN '" + name.replace("'", "''") + "' END"
            for name, predicate in conditions
        )
        sql = (
            f"SELECT *, concat_ws('; ', {labels}) AS {quote_identifier(reason_name)} "
            f"FROM {base} WHERE {bad}"
        )
        try:
            sample = (
                ctx.sql(f"{sql} LIMIT {ctx.evidence_limit}")
                if ctx.evidence_limit
                else pl.DataFrame(schema=actual)
            )
        except Exception as exc:
            failures.append(
                Finding.unassessed(
                    f"Not assessed: '{label}' on {dataset}.{column}: {type(exc).__name__}: {exc}.",
                    check=f"{dataset}.{column}: {label}",
                    severity=severity,
                    datasets=(dataset,),
                )
            )
            return
        if count:
            failures.append(
                _row_failure(
                    dataset,
                    column,
                    label,
                    severity,
                    present(sample, {row_name: "_row", reason_name: "violates"}),
                    count,
                    rows,
                )
            )
        else:
            passed.append(f"{dataset}.{column}: {label}")

    operators = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<"}
    for column, spec in schema.columns.items():
        if column not in actual:
            if spec.required:
                failures.append(values._missing_column(dataset, column))
            continue
        quoted = quote_identifier(column)
        floating = actual[column].is_float()
        missing = f"({quoted} IS NULL" + (f" OR isnan({quoted})" if floating else "") + ")"
        if not spec.nullable:
            record(column, "not null", [("missing (null or NaN)", missing)])
        if floating and not spec.allow_infinite:
            record(column, "finite", [("infinite", f"isinf({quoted})")])
        for check in spec.all_checks():
            conditions = []
            for operator, bound in check.constraints().items():
                if actual[column].is_integer():
                    bound = math.ceil(bound) if operator in ("ge", "lt") else math.floor(bound)
                comparison = quoted
                comparison_operator = operator
                if floating:
                    comparison_operator, bound = float_bound(operator, bound)
                    comparison = f"CAST({quoted} AS DOUBLE)"
                    literal = f"'{bound}'::DOUBLE" if math.isinf(bound) else repr(bound)
                elif actual[column].is_decimal():
                    comparison_operator, bound = decimal_bound(operator, bound, actual[column])
                    literal = format(bound, "f") if not isinstance(bound, bool) else "0"
                else:
                    literal = format(Decimal(str(bound)), "f")
                predicate = (
                    f"NOT {missing} AND NOT "
                    f"({comparison} {operators[comparison_operator]} {literal})"
                )
                if isinstance(bound, bool):
                    predicate = f"NOT {missing} AND {'FALSE' if bound else 'TRUE'}"
                conditions.append(
                    (f"{operator} {format_value(check.constraints()[operator])}", predicate)
                )
            record(column, values._check_name(check), conditions, check.severity)
    for offset in range(0, len(jobs), 64):
        batch = jobs[offset : offset + 64]
        reductions = ["count(*) AS rows"]
        for index, (_, _, conditions, _) in enumerate(batch):
            bad = " OR ".join(f"({predicate})" for _, predicate in conditions)
            reductions.append(f"coalesce(sum(CASE WHEN {bad} THEN 1 ELSE 0 END), 0) AS c{index}")
        counts = ctx.sql(f"SELECT {', '.join(reductions)} FROM {table}").row(0)
        rows = int(counts[0])
        for job, count in zip(batch, counts[1:], strict=True):
            emit(*job, int(count))
    return failures, passed


def _lazy_dataset(
    ctx: DataContext, dataset: str, schema: SchemaSpec, actual: dict[str, pl.DataType]
) -> tuple[list[Finding], list[str]]:
    """Simple built-ins remain lazy for caller-provided Polars plans."""
    plan = ctx.scan(dataset)
    rows = 0
    jobs: list[tuple[str, str, list[tuple[str, pl.Expr]], Severity | None]] = []
    failures: list[Finding] = []
    passed: list[str] = []
    row = fresh_internal("row", actual)
    reason = fresh_internal("violates", [*actual, row])
    typed = pl.DataFrame(schema=actual)

    def record(
        column: str,
        label: str,
        predicates: list[tuple[str, pl.Expr]],
        severity: Severity | None = None,
    ) -> None:
        jobs.append((column, label, predicates, severity))

    def emit(
        column: str,
        label: str,
        predicates: list[tuple[str, pl.Expr]],
        severity: Severity | None,
        count: int,
    ) -> None:
        try:
            bad = pl.any_horizontal([predicate for _, predicate in predicates])
            if not count:
                passed.append(f"{dataset}.{column}: {label}")
                return
            reasons = pl.concat_str(
                [
                    pl.when(predicate).then(pl.lit(name)).otherwise(None)
                    for name, predicate in predicates
                ],
                separator="; ",
                ignore_nulls=True,
            )
            sample = (
                ctx.collect(
                    dataset,
                    plan.with_row_index(row)
                    .filter(bad)
                    .with_columns(reasons.alias(reason))
                    .limit(ctx.evidence_limit),
                )
                if ctx.evidence_limit
                else pl.DataFrame(schema=actual)
            )
            failures.append(
                _row_failure(
                    dataset,
                    column,
                    label,
                    severity,
                    present(sample, {row: "_row", reason: "violates"}),
                    count,
                    rows,
                )
            )
        except Exception as exc:
            failures.append(
                Finding.unassessed(
                    f"Not assessed: '{label}' on {dataset}.{column}: {type(exc).__name__}: {exc}.",
                    check=f"{dataset}.{column}: {label}",
                    severity=severity,
                    datasets=(dataset,),
                )
            )

    for column, spec in schema.columns.items():
        if column not in actual:
            if spec.required:
                failures.append(values._missing_column(dataset, column))
            continue
        missing = missing_expr(typed, column)
        if not spec.nullable:
            record(column, "not null", [("missing (null or NaN)", missing)])
        if actual[column].is_float() and not spec.allow_infinite:
            record(column, "finite", [("infinite", pl.col(column).is_infinite())])
        for check in spec.all_checks():
            try:
                predicates = [
                    (label, ~missing & ~test(column).fill_null(True))
                    for label, test in row_tests(check.constraints(), actual[column]).values()
                ]
                record(column, values._check_name(check), predicates, check.severity)
            except Exception as exc:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: '{values._check_name(check)}' on {dataset}.{column}: "
                        f"{type(exc).__name__}: {exc}.",
                        check=f"{dataset}.{column}: {values._check_name(check)}",
                        severity=check.severity,
                        datasets=(dataset,),
                    )
                )
    for offset in range(0, len(jobs), 64):
        batch = jobs[offset : offset + 64]
        counts = ctx.collect(
            dataset,
            plan.select(
                pl.len().alias("rows"),
                *[
                    pl.any_horizontal([p for _, p in predicates]).sum().alias(f"c{i}")
                    for i, (_, _, predicates, _) in enumerate(batch)
                ],
            ),
        ).row(0)
        rows = int(counts[0])
        for job, count in zip(batch, counts[1:], strict=True):
            emit(*job, int(count or 0))
    return failures, passed


def _row_failure(
    dataset: str,
    column: str,
    label: str,
    severity: Severity | None,
    sample: pl.DataFrame,
    count: int,
    rows: int,
) -> Finding:
    """Fused backends share the same row-count and bounded-evidence result contract."""
    return Finding.fail(
        f"{plural(count, 'value')} of {rows:,} in {dataset}.{column} violate '{label}'.",
        check=f"{dataset}.{column}: {label}",
        severity=severity,
        affected=sample,
        affected_count=count,
        total_records=rows,
        unit="rows",
        datasets=(dataset,),
    )
