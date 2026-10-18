"""Per-check column plans: reductions and bounded evidence before materialization.

A check's dependencies and execution needs are independent of sibling checks.
Polars plans keep all predicates lazy; SQL plans use native reductions for common
contracts and explicitly project only a check's dependencies for exact fallbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import NotImplementedType
from typing import Any

import polars as pl

from qavf.aggregation import accumulator_sql
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.rules._column_batch import (
    ColumnInputs,
    ColumnReduction,
    independent_scope,
    reduce_columns,
)
from qavf.rules._column_rows import native_row_check
from qavf.rules._column_stream import native_float_statistics
from qavf.rules._predicates import compatibility_error, is_nan, row_tests
from qavf.rules._regex import sql_compatible
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import present
from qavf.settings import COLUMN_CONSTRAINTS, ROW_CONSTRAINTS, ColumnCheck, ColumnSpec, SchemaSpec


@dataclass(frozen=True, slots=True)
class ColumnPlan:
    """One independently assessed column contract and its input dependencies."""

    column: str
    spec: ColumnSpec
    kind: str
    check: ColumnCheck | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        columns = [self.column]
        if self.kind == "units" and self.spec.unit_column:
            columns.append(self.spec.unit_column)
        if self.check is not None and self.check.where:
            columns.extend(pl.sql_expr(self.check.where).meta.root_names())
        return tuple(dict.fromkeys(columns))

    def single_schema(self) -> SchemaSpec:
        """Only this contract, for projected fallback through the eager kernel."""
        raw = self.spec.model_dump()
        defaults = ColumnSpec().model_dump()
        for name in (*ROW_CONSTRAINTS, *COLUMN_CONSTRAINTS):
            raw[name] = defaults[name]
        raw.update(
            checks=[],
            nullable=True,
            allow_infinite=True,
            unit_column=None,
            allowed_units=None,
            unit=None,
        )
        if self.kind == "not null":
            raw["nullable"] = False
        elif self.kind == "finite":
            raw["allow_infinite"] = False
        elif self.kind == "units":
            raw.update(
                unit=self.spec.unit,
                unit_column=self.spec.unit_column,
                allowed_units=self.spec.allowed_units,
            )
        else:
            raw["checks"] = [self.check]
        return SchemaSpec(columns={self.column: ColumnSpec.model_validate(raw)})


def column_plans(schema: SchemaSpec, actual: dict[str, pl.DataType]) -> list[ColumnPlan]:
    plans = []
    for column, spec in schema.columns.items():
        if column not in actual:
            continue
        if not spec.nullable:
            plans.append(ColumnPlan(column, spec, "not null"))
        if actual[column].is_float() and not spec.allow_infinite:
            plans.append(ColumnPlan(column, spec, "finite"))
        if spec.unit_column is not None:
            plans.append(ColumnPlan(column, spec, "units"))
        plans.extend(ColumnPlan(column, spec, "check", check) for check in spec.all_checks())
    return plans


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
) -> ColumnReduction[pl.Expr]:
    """Assess one lazy contract with scalar reductions and a bounded row sample."""
    column, check = item.column, item.check
    dependencies = item.dependencies
    inputs.validate(dependencies)
    row, reason, saved = inputs.row, inputs.reason, inputs.saved_for(dependencies)
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
            row_label, test = row_tests({name: raw}, actual[column])[name]
            predicates.append(
                (row_label, scope & pl.col(column).is_not_null() & ~test(column).fill_null(True))
            )
        if actual[column].is_float() and any(is_nan(v) for v in constraints.get("notin") or []):
            predicates.append(("notin NaN", scope & pl.col(saved[column]).is_nan()))
        if check.max_null_fraction is not None:
            reducers["missing"] = value.is_null().sum()
        if check.unique:
            reducers["duplicates"] = value.drop_nulls().len() - value.drop_nulls().n_unique()
        for name in COLUMN_CONSTRAINTS:
            if name not in constraints or name in ("unique", "max_null_fraction"):
                continue
            if name != "distinct_count" and not actual[column].is_numeric():
                raise ValueError(f"{name} requires a numeric column, found {actual[column]}")
            bound = constraints[name]
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
                return (
                    None
                    if within(result, bounds)
                    else f"{kind} {show(result)} outside [{show(bounds.ge)}, {show(bounds.le)}]"
                )

            reducers["stat_" + name] = value.map_batches(
                reduce, return_dtype=pl.String, returns_scalar=True
            )
    bad = (
        pl.any_horizontal([bad for _, bad in predicates]).fill_null(False)
        if predicates
        else pl.lit(False)
    )
    reducers["bad"] = bad.sum()
    # Count a constant scope in the same reduction; scalar literals do not broadcast
    # when summed by select, unlike filter's row-wise application.
    if check and check.where and not pl.sql_expr(check.where).meta.root_names():
        reducers["applicable"] = pl.when(scope).then(pl.len()).otherwise(pl.lit(0))

    def finish(summary: dict[str, Any]) -> Finding | None:
        applicable = int(summary["applicable"] or 0)
        if check and applicable < check.min_applicable_rows:
            raise ValueError(
                f"only {applicable} row(s) match 'where: {check.where}', "
                f"fewer than min_applicable_rows {check.min_applicable_rows}"
            )
        problems = []
        for name in COLUMN_CONSTRAINTS:
            if name in scalar_statistics:
                bounds = scalar_statistics[name]
                result = summary["value_" + name]
                if not within(result, bounds):
                    problems.append(
                        f"{name} {show(result)} outside [{show(bounds.ge)}, {show(bounds.le)}]"
                    )
            elif summary.get("stat_" + name) is not None:
                problems.append(summary["stat_" + name])
        if check and check.max_null_fraction is not None:
            fraction = int(summary["missing"] or 0) / applicable if applicable else 0.0
            if fraction > check.max_null_fraction:
                problems.append(
                    f"missing fraction {fraction:.4g} exceeds {check.max_null_fraction:g}"
                )
        if summary.get("duplicates", 0):
            problems.append("values are not unique")
        count = int(summary["bad"] or 0)
        if not count and not problems:
            return None
        severity = check.severity if check else None
        if not count:
            return Finding.fail(
                f"{dataset}.{column} fails '{label}': {'; '.join(problems)}.",
                check=f"{dataset}.{column}: {label}",
                severity=severity,
                affected=[{"dataset": dataset, "column": column, "problem": p} for p in problems],
                unit="problems",
                datasets=(dataset,),
            )
        sample = None
        if ctx.evidence_limit:
            reasons = pl.concat_str(
                [pl.when(p).then(pl.lit(name)).otherwise(None) for name, p in predicates],
                separator="; ",
                ignore_nulls=True,
            )
            sample = ctx.collect(
                dataset,
                inputs.lazy(dependencies)
                .filter(bad)
                .with_columns(reasons.alias(reason))
                .limit(ctx.evidence_limit),
            )
            sample = sample.drop(list(saved)).rename({v: k for k, v in saved.items()})
            sample = temporal_evidence(
                present(sample.select(*actual, row, reason), {row: "_row", reason: "violates"})
            )
        return Finding.fail(
            f"{count:,} value{'s' if count != 1 else ''} of {applicable:,} "
            f"in {dataset}.{column} violate '{label}'."
            + (f" Column-level: {'; '.join(problems)}." if problems else ""),
            check=f"{dataset}.{column}: {label}",
            severity=severity,
            affected=sample,
            affected_count=count,
            total_records=applicable,
            unit="rows",
            metrics={"column_problems": problems} if problems else {},
            datasets=(dataset,),
        )

    return ColumnReduction(
        dependencies,
        reducers,
        finish,
        independent_scope(check.where if check else None, actual)
        or any(name.startswith("stat_") for name in reducers),
    )


def _native_column_reduction(
    ctx: DataContext,
    dataset: str,
    actual: dict[str, pl.DataType],
    item: ColumnPlan,
    label: str,
    within: Callable[[Any, Any], bool],
    show: Callable[[Any], str],
    inputs: ColumnInputs,
) -> ColumnReduction[str] | NotImplementedType:
    """Native count/sample for common SQL-safe contracts; exact fallback is explicit."""
    import math
    from fractions import Fraction

    from qavf.rules._predicates import (
        decimal_bound,
        float_bound,
        format_value,
        is_temporal,
        members,
    )
    from qavf.sources.sql import quote_identifier, sql_literal

    column, check = item.column, item.check
    dtype = actual[column]
    constraints = check.constraints() if check else {}
    if check is not None and check.regex is not None and not sql_compatible(check.regex):
        return NotImplemented
    statistics = {
        k: v
        for k, v in constraints.items()
        if k in COLUMN_CONSTRAINTS and k not in ("unique", "max_null_fraction")
    }
    # Float statistics and order statistics retain the exact reference reducer.
    if any(
        k in ("median", "std") or (dtype.is_float() and k in ("sum", "mean")) for k in statistics
    ):
        return NotImplemented
    if is_temporal(dtype) and any(k in ROW_CONSTRAINTS for k in constraints):
        return NotImplemented
    if dtype.is_float() and any(k in constraints for k in ("integral", "max_decimals")):
        return NotImplemented
    dependencies = item.dependencies
    inputs.validate(dependencies)
    q = quote_identifier
    row, reason, saved = inputs.row, inputs.reason, inputs.saved_for(dependencies)
    scope = f"coalesce(({check.where}), FALSE)" if check and check.where else "TRUE"
    value = q(column)
    predicates: list[tuple[str, str]] = []
    if item.kind == "not null":
        predicates.append(("missing", f"{value} IS NULL"))
    elif item.kind == "finite":
        predicates.append(("infinite", f"isinf({value})"))
    elif item.kind == "units":
        units = f"CAST({q(str(item.spec.unit_column))} AS VARCHAR)"
        allowed_units = ", ".join(
            sql_literal(v) for v in item.spec.allowed_units or [str(item.spec.unit)]
        )
        predicates.append(
            (
                "unit",
                f"{value} IS NOT NULL AND ({units} IS NULL OR {units} NOT IN ({allowed_units}))",
            )
        )
    else:
        operators = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<"}
        for name, raw in constraints.items():
            if name not in ROW_CONSTRAINTS:
                continue
            error = compatibility_error(name, raw, dtype)
            if error:
                raise ValueError(error)
            valid = None
            if name in operators:
                bound, operator, lhs = raw, name, value
                if dtype.is_integer():
                    bound = math.ceil(bound) if name in ("ge", "lt") else math.floor(bound)
                    # DuckDB cannot represent a bigger exact integer literal.
                    if not -(2**127) <= bound < 2**127:
                        return NotImplemented
                elif dtype.is_float():
                    operator, bound = float_bound(name, raw)
                    lhs = f"CAST({value} AS DOUBLE)"
                elif dtype.is_decimal():
                    operator, bound = decimal_bound(name, raw, dtype)
                if isinstance(bound, bool):
                    valid = "TRUE" if bound else "FALSE"
                else:
                    literal = (
                        f"CAST({sql_literal(str(bound))} AS DOUBLE)"
                        if isinstance(bound, float) and math.isinf(bound)
                        else sql_literal(bound)
                    )
                    valid = f"{lhs} {operators[operator]} {literal}"
            elif name in ("isin", "notin"):
                allowed, _ = members(raw, dtype)
                literals = []
                for v in allowed:
                    literals.append(
                        f"CAST({sql_literal(str(v))} AS DOUBLE)"
                        if isinstance(v, float) and not math.isfinite(v)
                        else sql_literal(v)
                    )
                valid = f"{value} IN ({', '.join(literals)})" if literals else "FALSE"
                if name == "notin":
                    valid = f"NOT ({valid})"
                    if dtype.is_float() and any(is_nan(v) for v in raw):
                        predicates.append(("notin NaN", f"({scope}) AND isnan({q(saved[column])})"))
            elif name == "regex":
                valid = f"regexp_full_match(CAST({value} AS VARCHAR), {sql_literal(raw)})"
            elif name in ("min_length", "max_length"):
                operator = ">=" if name == "min_length" else "<="
                valid = f"length(CAST({value} AS VARCHAR)) {operator} {raw}"
            elif name in ("integral", "max_decimals") and dtype.is_integer():
                valid = "TRUE"
            else:
                return NotImplemented
            predicates.append(
                (
                    f"{name} {format_value(raw)}",
                    f"({scope}) AND {value} IS NOT NULL AND NOT ({valid})",
                )
            )
    bad = " OR ".join(f"({predicate})" for _, predicate in predicates) or "FALSE"
    reducers = {
        "applicable": f"count(*) FILTER (WHERE {scope})",
        "bad": f"count(*) FILTER (WHERE {bad})",
    }
    if check and check.max_null_fraction is not None:
        reducers["missing"] = f"count(*) FILTER (WHERE ({scope}) AND {value} IS NULL)"
    if check and check.unique:
        reducers["duplicates"] = (
            f"count({value}) FILTER (WHERE {scope}) - "
            f"count(DISTINCT {value}) FILTER (WHERE {scope})"
        )
    for name in statistics:
        if name != "distinct_count" and not dtype.is_numeric():
            raise ValueError(f"{name} requires a numeric column, found {dtype}")
        if name == "distinct_count":
            expression = f"count(DISTINCT {value})"
        elif name in ("min", "max"):
            expression = f"{name}({value})"
        else:
            storage = accumulator_sql(dtype)
            expression = f"sum(CAST({value} AS {storage}))"
        reducers["stat_" + name] = f"{expression} FILTER (WHERE {scope})"
        if name == "mean":
            reducers["mean_n"] = f"count({value}) FILTER (WHERE {scope})"

    def finish(summary: dict[str, Any]) -> Finding | None:
        applicable = int(summary["applicable"])
        if check and applicable < check.min_applicable_rows:
            raise ValueError(
                f"only {applicable} row(s) match 'where: {check.where}', "
                f"fewer than min_applicable_rows {check.min_applicable_rows}"
            )
        problems = []
        for name, bound in statistics.items():
            result = summary["stat_" + name]
            if name == "mean" and result is not None:
                result = Fraction(result) / int(summary["mean_n"])
            if not within(result, bound):
                problems.append(
                    f"{name} {show(result)} outside [{show(bound.ge)}, {show(bound.le)}]"
                )
        if check and check.max_null_fraction is not None:
            fraction = int(summary["missing"]) / applicable if applicable else 0.0
            if fraction > check.max_null_fraction:
                problems.append(
                    f"missing fraction {fraction:.4g} exceeds {check.max_null_fraction:g}"
                )
        if summary.get("duplicates", 0):
            problems.append("values are not unique")
        count = int(summary["bad"])
        if not count and not problems:
            return None
        severity = check.severity if check else None
        if not count:
            return Finding.fail(
                f"{dataset}.{column} fails '{label}': {'; '.join(problems)}.",
                check=f"{dataset}.{column}: {label}",
                severity=severity,
                affected=[{"dataset": dataset, "column": column, "problem": p} for p in problems],
                unit="problems",
                datasets=(dataset,),
            )
        sample = None
        if ctx.evidence_limit:
            reasons = ", ".join(
                f"CASE WHEN ({p}) THEN {sql_literal(name)} END" for name, p in predicates
            )
            sample = ctx.sql(
                f"SELECT *, concat_ws('; ', {reasons}) AS {q(reason)} "
                f"FROM {inputs.native(dependencies)} WHERE {bad} LIMIT {ctx.evidence_limit}"
            )
            sample = sample.drop(list(saved)).rename({v: k for k, v in saved.items()})
            sample = temporal_evidence(
                present(sample.select(*actual, row, reason), {row: "_row", reason: "violates"})
            )
        return Finding.fail(
            f"{count:,} value{'s' if count != 1 else ''} of {applicable:,} "
            f"in {dataset}.{column} violate '{label}'."
            + (f" Column-level: {'; '.join(problems)}." if problems else ""),
            check=f"{dataset}.{column}: {label}",
            severity=severity,
            affected=sample,
            affected_count=count,
            total_records=applicable,
            unit="rows",
            metrics={"column_problems": problems} if problems else {},
            datasets=(dataset,),
        )

    return ColumnReduction(
        dependencies,
        reducers,
        finish,
        independent_scope(check.where if check else None, actual)
        or any(name in statistics for name in ("sum", "mean")),
    )


def column_checks(
    ctx: DataContext,
    dataset: str,
    schema: SchemaSpec,
    actual: dict[str, pl.DataType],
    items: list[ColumnPlan],
    label: Callable[[ColumnPlan], str],
    statistic: Callable[[pl.Series, str], Any],
    within: Callable[[Any, Any], bool],
    show: Callable[[Any], str],
) -> list[Finding | Exception | NotImplementedType | None]:
    """Reduce compatible contracts together; preserve independent failures and fallbacks."""
    inputs = ColumnInputs(ctx, dataset, schema, actual)
    lazy = ctx.execution_kind(dataset) == "lazy"
    jobs: dict[int, ColumnReduction[Any]] = {}
    results: list[Finding | Exception | NotImplementedType | None] = [None] * len(items)
    for index, item in enumerate(items):
        try:
            prepared = (
                _lazy_column_reduction(
                    ctx, dataset, schema, actual, item, label(item), statistic, within, show, inputs
                )
                if lazy
                else _native_column_reduction(
                    ctx, dataset, actual, item, label(item), within, show, inputs
                )
            )
            if prepared is NotImplemented:
                streamed = native_row_check(
                    inputs, item.column, item.dependencies, item.check, label(item)
                )
                if streamed is NotImplemented:
                    streamed = native_float_statistics(
                        inputs,
                        item.column,
                        item.dependencies,
                        item.check,
                        label(item),
                        within,
                        show,
                    )
                results[index] = streamed
            else:
                jobs[index] = prepared
        except Exception as exc:
            results[index] = exc
    results_by_index = reduce_columns(inputs, jobs, lazy=lazy)
    for index, result in results_by_index.items():
        results[index] = result
    return results
