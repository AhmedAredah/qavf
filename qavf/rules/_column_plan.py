"""Per-check column plans: reductions and bounded evidence before materialization.

A check's dependencies and execution needs are independent of sibling checks.
Polars plans keep all predicates lazy; SQL plans use native reductions for common
contracts and explicitly project only a check's dependencies for exact fallbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from types import NotImplementedType
from typing import Any

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.registry import Finding
from qavf.rules._column_batch import (
    BATCH_SIZE,
    ColumnInputs,
    ColumnReduction,
    independent_scope,
    reduce_captured_columns,
    reduce_columns,
)
from qavf.rules._column_contract import ColumnPlan as ColumnPlan
from qavf.rules._column_contract import _unsafe_unit_cast
from qavf.rules._column_contract import column_plans as column_plans
from qavf.rules._column_lazy import _lazy_column_reduction
from qavf.rules._column_native import _native_column_reduction
from qavf.rules._column_rows import native_row_check
from qavf.rules._column_stream import native_float_statistics
from qavf.rules._evidence_columns import EvidenceColumns
from qavf.settings import ROW_CONSTRAINTS, SchemaSpec


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
    observation: Observation | None = None,
) -> list[Finding | Exception | NotImplementedType | None]:
    """Reduce compatible contracts together; preserve independent failures and fallbacks."""
    inputs = ColumnInputs(ctx, dataset, schema, actual, observation)
    lazy = observation is not None or ctx.execution_kind(dataset) == "lazy"
    jobs: dict[int, ColumnReduction[Any]] = {}
    results: list[Finding | Exception | NotImplementedType | None] = [None] * len(items)
    for index, item in enumerate(items):
        if item.exact_only:
            results[index] = NotImplemented
            continue
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


def coherent_column_checks(
    ctx: DataContext,
    dataset: str,
    schema: SchemaSpec,
    actual: dict[str, pl.DataType],
    items: list[ColumnPlan],
    label: Callable[[ColumnPlan], str],
    statistic: Callable[[pl.Series, str], Any],
    within: Callable[[Any, Any], bool],
    show: Callable[[Any], str],
) -> list[Finding | Exception | NotImplementedType | None] | None:
    """Read eligible lazy row contracts once, retaining only counts and samples.

    Global statistics, arbitrary scope expressions and large contract sets keep
    the owned observation path. Preparation never evaluates the user's source.
    """
    if len(items) > BATCH_SIZE:
        return None
    for item in items:
        if _unsafe_unit_cast(item, actual):
            return None
        check = item.check
        if check and (
            independent_scope(check.where, actual)
            or set(check.constraints()) - set(ROW_CONSTRAINTS) - {"max_null_fraction"}
        ):
            return None
    inputs = ColumnInputs(ctx, dataset, schema, actual)
    needed = set(schema.primary_key or []) if ctx.evidence_limit else set()
    if ctx.evidence_limit and EvidenceColumns(schema, tuple(actual)).whole_rows:
        needed.update(actual)
    for item in items:
        try:
            needed.update(item.dependencies)
        except Exception:
            # The check-specific preparation below records invalid syntax.
            continue
    captured = tuple(name for name in actual if name in needed) if ctx.evidence_limit else ()
    jobs: dict[int, ColumnReduction[pl.Expr]] = {}
    results: list[Finding | Exception | NotImplementedType | None] = [None] * len(items)
    for index, item in enumerate(items):
        try:
            jobs[index] = _lazy_column_reduction(
                ctx,
                dataset,
                schema,
                actual,
                item,
                label(item),
                statistic,
                within,
                show,
                inputs,
                captured,
            )
        except Exception as exc:
            results[index] = exc
    for index, result in reduce_captured_columns(inputs, jobs, captured).items():
        results[index] = result
    return results
