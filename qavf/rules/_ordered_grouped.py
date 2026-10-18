"""Exact ordered group selection through scoped relational extrema."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack

import polars as pl

from qavf.aggregation import numeric_expr
from qavf.engine.observation import Observation
from qavf.ordering import OrderBy, ordering_columns, usable_order
from qavf.rules.helpers import check_aggregation, fresh_internal


def ordered_grouped(
    observation: Observation,
    plan: pl.LazyFrame,
    keys: Sequence[str],
    value: str,
    alias: str,
    how: str,
    order_by: OrderBy,
) -> pl.DataFrame:
    """Select first/last exactly without collecting all input members.

    Each lexicographic component narrows candidates by a native per-group
    extremum and join. Intermediate candidates use managed spill; only the
    previous and next stages coexist. Native group/join state and the G-row
    result scale with group cardinality, not a constant-memory promise.
    """
    if how not in {"first", "last"}:
        raise ValueError("ordered_grouped requires first or last")
    check_aggregation(how, None, order_by)
    schema = dict(plan.collect_schema())
    dtype = schema[value]
    if not dtype.is_numeric() and dtype != pl.Null:
        raise ValueError(f"aggregation requires numeric values, found {dtype}")
    orders = ordering_columns(order_by)
    used = [*schema, alias]
    candidate_value = fresh_internal("ordered_value", used)
    used.append(candidate_value)
    group_keys = list(keys)
    if not group_keys:
        group = fresh_internal("ordered_group", used)
        used.append(group)
        group_keys = [group]
        plan = plan.with_columns(pl.lit(0, dtype=pl.UInt8).alias(group))
        groups = pl.DataFrame({group: [0]}, schema={group: pl.UInt8}).lazy()
    else:
        groups = plan.select(group_keys).unique(maintain_order=True)
    measured = numeric_expr(value, dtype)
    candidates = (
        plan.filter(usable_order(measured, orders, schema))
        .with_columns(measured.alias(candidate_value))
        .select(list(dict.fromkeys([*group_keys, *orders, candidate_value])))
    )
    if not observation.collect(candidates.select(pl.len())).item():
        # An empty native self-join can panic on supported Polars versions.
        # Groups with no usable candidate still represent an undefined value.
        return observation.collect(
            groups.with_columns(pl.lit(None, dtype=dtype).alias(alias)).select([*keys, alias])
        )
    stage = ExitStack()
    try:
        for column in orders:
            extreme_name = fresh_internal("ordered_extreme", used)
            order = pl.col(column)
            extreme = order.max() if how == "last" else order.min()
            extrema = candidates.group_by(group_keys).agg(extreme.alias(extreme_name))
            narrowed = (
                candidates.join(extrema, on=group_keys, how="inner", nulls_equal=True)
                .filter(pl.col(column) == pl.col(extreme_name))
                .drop(extreme_name)
            )
            successor = ExitStack()

            def selected_plan(_: pl.LazyFrame, selected: pl.LazyFrame = narrowed) -> pl.LazyFrame:
                return selected

            try:
                captured = successor.enter_context(observation.derived(selected_plan))
            except BaseException:
                successor.close()
                raise
            stage.close()
            stage = successor
            candidates = captured.scan()
        maximum = fresh_internal("ordered_maximum", used)
        conflict = fresh_internal("ordered_conflict", [*used, maximum])
        reduced = candidates.group_by(group_keys).agg(
            pl.col(candidate_value).min().alias(alias),
            pl.col(candidate_value).max().alias(maximum),
        )
        result = captured.collect(
            groups.join(reduced, on=group_keys, how="left", nulls_equal=True, maintain_order="left")
            .with_columns((pl.col(alias) != pl.col(maximum)).fill_null(False).alias(conflict))
            .select([*keys, alias, conflict])
        )
        if result[conflict].any():
            raise ValueError(
                "ordered aggregation has conflicting values tied at the selected order_by key; "
                "provide additional order_by columns to break the tie explicitly"
            )
        return result.drop(conflict)
    finally:
        stage.close()
