"""Vectorized Polars reductions for supported normalized column checks."""

from __future__ import annotations

from collections.abc import Callable
from types import NotImplementedType
from typing import Any

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.input_rules import outside_bounds
from qavf.rules._column_batch import (
    ColumnInputs,
    ColumnReduction,
    exceeds_missing_fraction,
    independent_scope,
)
from qavf.rules._column_contract import (
    ColumnOutcome,
    ColumnPlan,
    _unsafe_unit_cast,
    require_statistic_population,
)
from qavf.rules._evidence_columns import EvidenceColumns
from qavf.rules._predicates import compatibility_error, is_nan, row_tests
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import present
from qavf.settings import COLUMN_CONSTRAINTS, ROW_CONSTRAINTS, SchemaSpec
from qavf.value_policy import normalized_text


def _lazy_column_reduction(
    ctx: DataContext,
    dataset: str,
    schema: SchemaSpec,
    actual: dict[str, pl.DataType],
    item: ColumnPlan,
    label: str,
    statistic: Callable[[pl.Series, str], Any],
    within: Callable[[Any, Any], bool],
    show: Callable[[Any], str],
    inputs: ColumnInputs,
    capture_columns: tuple[str, ...] | None = None,
) -> ColumnReduction[pl.Expr]:
    """Assess one lazy contract with scalar reductions and a bounded row sample."""
    column, check = item.column, item.check
    outcome = ColumnOutcome(dataset, column, label, check)
    dependencies = item.dependencies
    inputs.validate(dependencies)
    row, reason, saved = inputs.row, inputs.reason, inputs.saved_for(dependencies)
    shown = EvidenceColumns(inputs.schema, tuple(actual)).of(dependencies)
    scope = pl.sql_expr(check.where).fill_null(False) if check and check.where else pl.lit(True)
    value = pl.col(column).filter(scope)
    predicates: list[tuple[str, pl.Expr]] = []
    reducers: dict[str, pl.Expr] = {"applicable": scope.sum()}
    scalar_statistics: dict[str, Any] = {}
    # Literal True is scalar; select length must remain the actual row count.
    if not (check and check.where):
        reducers["applicable"] = pl.len()
    if item.kind == "not null":
        predicates.append(("missing", pl.col(column).is_null()))
    elif item.kind == "finite":
        predicates.append(("infinite", pl.col(column).is_infinite()))
    elif item.kind == "units":
        allowed = list(item.spec.allowed_units or [str(item.spec.unit)])
        units = pl.col(str(item.spec.unit_column)).cast(pl.String)
        predicates.append(
            ("unit", pl.col(column).is_not_null() & (units.is_null() | ~units.is_in(allowed)))
        )
    else:
        assert check is not None
        constraints = check.constraints()
        for name, raw in constraints.items():
            if name not in ROW_CONSTRAINTS:
                continue
            error = compatibility_error(name, raw, actual[column])
            if error:
                raise ValueError(error)
            tests = row_tests({name: raw}, actual[column], normalize=check.normalize)
            row_label, test = tests[name]
            predicates.append(
                (row_label, scope & pl.col(column).is_not_null() & ~test(column).fill_null(True))
            )
        if actual[column].is_float() and any(is_nan(v) for v in constraints.get("notin") or []):
            predicates.append(("notin NaN", scope & pl.col(saved[column]).is_nan()))
        if check.max_null_fraction is not None:
            reducers["missing"] = value.is_null().sum()
        if check.unique:
            distinct = normalized_text(value, check.normalize, actual[column])
            reducers["duplicates"] = distinct.drop_nulls().len() - distinct.drop_nulls().n_unique()
        for name in COLUMN_CONSTRAINTS:
            if name not in constraints or name in ("unique", "max_null_fraction"):
                continue
            if name != "distinct_count" and not actual[column].is_numeric():
                raise ValueError(f"{name} requires a numeric column, found {actual[column]}")
            bound = constraints[name]
            if name != "distinct_count":
                reducers["usable"] = value.count()
            if name in ("min", "max", "distinct_count"):
                scalar_statistics[name] = bound
                reducers["value_" + name] = (
                    value.drop_nulls().n_unique()
                    if name == "distinct_count"
                    else value.min()
                    if name == "min"
                    else value.max()
                )
                continue

            # Preserve the eager kernel's exact rational/integer/Decimal decision.
            # Only a boolean and compact diagnostic cross the materialization boundary.
            def reduce(values: pl.Series, kind: str = name, bounds: Any = bound) -> str | None:
                result = statistic(values, kind)
                return None if within(result, bounds) else outside_bounds(kind, result, bounds)

            reducers["stat_" + name] = value.map_batches(
                reduce, return_dtype=pl.String, returns_scalar=True
            )
    bad = (
        pl.any_horizontal([bad for _, bad in predicates]).fill_null(False)
        if predicates
        else pl.lit(False)
    )
    reducers["bad"] = bad.sum()
    missing = scope & pl.col(column).is_null()
    if check and check.max_null_fraction is not None and capture_columns is not None:
        reducers["bad_missing"] = (bad & missing).sum()
    # Count a constant scope in the same reduction; scalar literals do not broadcast
    # when summed by select, unlike filter's row-wise application.
    if check and check.where and not pl.sql_expr(check.where).meta.root_names():
        reducers["applicable"] = pl.when(scope).then(pl.len()).otherwise(pl.lit(0))

    if capture_columns is not None and ctx.evidence_limit:
        reasons = (
            pl.concat_str(
                [pl.when(p).then(pl.lit(name)).otherwise(None) for name, p in predicates],
                separator="; ",
                ignore_nulls=True,
            )
            if predicates
            else pl.lit("")
        )
        reducers["sample"] = (
            pl.struct(
                *(pl.col(inputs.saved[c]).alias(c) for c in capture_columns),
                pl.col(row),
                reasons.alias(reason),
            )
            .filter(bad)
            .head(ctx.evidence_limit)
            .implode()
        )
        if check and check.max_null_fraction is not None:
            missing_reason = (
                pl.when(reasons != "")
                .then(pl.concat_str(reasons, pl.lit("missing"), separator="; "))
                .otherwise(pl.lit("missing"))
            )
            reducers["missing_sample"] = (
                pl.struct(
                    *(pl.col(inputs.saved[c]).alias(c) for c in capture_columns),
                    pl.col(row),
                    missing_reason.alias(reason),
                )
                .filter(missing)
                .head(ctx.evidence_limit)
                .implode()
            )

    def finish(summary: dict[str, Any]) -> Finding | NotImplementedType | None:
        applicable = int(summary["applicable"] or 0)
        outcome.require_applicable(applicable)
        if check is not None and "usable" in summary:
            require_statistic_population(
                set(check.constraints())
                & (set(COLUMN_CONSTRAINTS) - {"unique", "max_null_fraction"}),
                int(summary["usable"] or 0),
            )
        problems = []
        for name in COLUMN_CONSTRAINTS:
            if name in scalar_statistics:
                bounds = scalar_statistics[name]
                result = summary["value_" + name]
                if not within(result, bounds):
                    problems.append(outside_bounds(name, result, bounds))
            elif summary.get("stat_" + name) is not None:
                problems.append(summary["stat_" + name])
        missing_count = int(summary.get("missing") or 0)
        missing_failed = exceeds_missing_fraction(check, missing_count, applicable)
        if missing_failed:
            if capture_columns is None:
                return NotImplemented
            assert check is not None and check.max_null_fraction is not None
            problems.append(
                f"missing fraction {missing_count / applicable:.4g} (null or NaN) exceeds "
                f"{check.max_null_fraction:g}"
            )
        if summary.get("duplicates", 0):
            # Duplicates are rare and need their rows: the exact kernel lists them.
            return NotImplemented
        count = int(summary["bad"] or 0)
        if missing_failed:
            count += missing_count - int(summary.get("bad_missing") or 0)
        if not count:
            return outcome.finding(count, applicable, problems)
        sample = None
        if ctx.evidence_limit:
            reasons = (
                pl.concat_str(
                    [pl.when(p).then(pl.lit(name)).otherwise(None) for name, p in predicates],
                    separator="; ",
                    ignore_nulls=True,
                )
                if predicates
                else pl.lit("")
            )
            if capture_columns is not None:
                sample = summary["sample"]
                if missing_failed:
                    sample = (
                        pl.concat([sample, summary["missing_sample"]])
                        .unique(subset=[row], keep="last", maintain_order=True)
                        .sort(row)
                        .head(ctx.evidence_limit)
                    )
            else:
                # Raw values of the dependencies, the other shown columns as read.
                sample = inputs.collect(
                    inputs.lazy(dependencies)
                    .filter(bad)
                    .with_columns(reasons.alias(reason))
                    .select(*(saved.get(name, name) for name in shown), row, reason)
                    .limit(ctx.evidence_limit),
                )
                sample = sample.rename({saved[name]: name for name in shown if name in saved})
            sample = temporal_evidence(
                present(
                    sample.select(*(name for name in shown if name in sample.columns), row, reason),
                    {row: "_row", reason: "violates"},
                )
            )
        return outcome.finding(count, applicable, problems, sample)

    return ColumnReduction(
        dependencies,
        reducers,
        finish,
        _unsafe_unit_cast(item, actual)
        or independent_scope(check.where if check else None, actual)
        or any(name.startswith("stat_") for name in reducers),
    )
