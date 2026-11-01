"""Native SQL reductions with explicit exact-fallback decisions."""

from __future__ import annotations

from collections.abc import Callable
from types import NotImplementedType
from typing import Any

import polars as pl

from qavf.aggregation import accumulator_sql
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
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
from qavf.rules._predicates import compatibility_error, is_nan
from qavf.rules._regex import sql_compatible
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import present
from qavf.settings import COLUMN_CONSTRAINTS, ROW_CONSTRAINTS


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
    outcome = ColumnOutcome(dataset, column, label, check)
    dtype = actual[column]
    constraints = check.constraints() if check else {}
    # DuckDB's non-text VARCHAR casts can differ from the typed reference
    # kernel (notably Binary escapes versus UTF-8 decoding). Preserve parity.
    if _unsafe_unit_cast(item, actual):
        return NotImplemented
    if check is not None and check.regex is not None and not sql_compatible(check.regex):
        return NotImplemented
    if check is not None and check.normalize:
        return NotImplemented  # SQL folds case and trims differently from the reference
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
    shown = EvidenceColumns(inputs.schema, tuple(actual)).of(dependencies)
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
        if name != "distinct_count":
            reducers["usable"] = f"count({value}) FILTER (WHERE {scope})"
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

    def finish(summary: dict[str, Any]) -> Finding | NotImplementedType | None:
        applicable = int(summary["applicable"])
        outcome.require_applicable(applicable)
        require_statistic_population(statistics, int(summary.get("usable") or 0))
        problems = []
        for name, bound in statistics.items():
            result = summary["stat_" + name]
            if name == "mean" and result is not None:
                result = Fraction(result) / int(summary["mean_n"])
            if not within(result, bound):
                problems.append(
                    f"{name} {show(result)} outside [{show(bound.ge)}, {show(bound.le)}]"
                )
        if exceeds_missing_fraction(check, int(summary.get("missing") or 0), applicable):
            return NotImplemented
        if summary.get("duplicates", 0):
            # Duplicates are rare and need their rows: the exact kernel lists them.
            return NotImplemented
        count = int(summary["bad"])
        if not count:
            return outcome.finding(count, applicable, problems)
        sample = None
        if ctx.evidence_limit:
            reasons = ", ".join(
                f"CASE WHEN ({p}) THEN {sql_literal(name)} END" for name, p in predicates
            )
            columns = ", ".join(q(saved.get(name, name)) for name in (*shown, row))
            sample = ctx.sql(
                f"SELECT {columns}, concat_ws('; ', {reasons}) AS {q(reason)} "
                f"FROM {inputs.native(dependencies)} WHERE {bad} LIMIT {ctx.evidence_limit}"
            )
            sample = sample.rename({saved[name]: name for name in shown if name in saved})
            sample = temporal_evidence(
                present(sample.select(*shown, row, reason), {row: "_row", reason: "violates"})
            )
        return outcome.finding(count, applicable, problems, sample)

    return ColumnReduction(
        dependencies,
        reducers,
        finish,
        independent_scope(check.where if check else None, actual)
        or any(name in statistics for name in ("sum", "mean")),
    )
