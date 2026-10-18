"""Projected spatial metrics with source reductions and explicit exact fallbacks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from qavf.engine.context import DataContext
from qavf.rules._exact_grouped import exact_grouped, supports_exact
from qavf.rules._execution import projected_frame
from qavf.rules._grouped import (
    duplicate_groups,
    grouped_aggregate,
    normalized_relation,
    scoped_relation,
    supports_grouped,
)
from qavf.rules._ordered_grouped import ordered_grouped
from qavf.rules.helpers import (
    Aggregate,
    OrderBy,
    aggregate,
    duplicated_groups,
    normalize_missing,
    ordering_columns,
)
from qavf.sources.sql import quote_identifier


@dataclass(frozen=True, slots=True)
class SpatialAggregate:
    values: pl.DataFrame | None
    missing_keys: int
    missing_sample: pl.DataFrame
    repeated_groups: int
    error: Exception | None = None


def spatial_aggregate(
    ctx: DataContext,
    dataset: str,
    keys: Sequence[str],
    value: str,
    alias: str,
    how: Aggregate,
    weight: str | None,
    order_by: OrderBy,
    *,
    exclude_missing_keys: bool = False,
) -> SpatialAggregate:
    """Keep metric and denominator preparation consistent across execution backends."""
    columns = (*keys, value, *([weight] if weight is not None else []), *ordering_columns(order_by))
    if how in {"first", "last"} or supports_exact(ctx.schema(dataset)[value], how):
        with scoped_relation(ctx, dataset, columns) as scoped:
            missing_count, missing_sample = 0, pl.DataFrame()
            missing = pl.any_horizontal(pl.col(key).is_null() for key in keys)
            if exclude_missing_keys:
                missing_count, missing_sample = scoped.filter(missing, "TRUE").count_sample(
                    ctx.evidence_limit
                )
                scoped = scoped.filter(~missing, "TRUE")
            assert scoped.observation is not None and scoped.lazy is not None
            try:
                exact_values = (
                    ordered_grouped(
                        scoped.observation, scoped.lazy, keys, value, alias, how, order_by
                    )
                    if how in {"first", "last"}
                    else exact_grouped(
                        ctx, scoped.observation, scoped.lazy, keys, value, alias, how, weight=weight
                    )
                )
                return SpatialAggregate(exact_values, missing_count, missing_sample, 0)
            except Exception as exc:
                return SpatialAggregate(None, missing_count, missing_sample, 0, exc)
    relation = normalized_relation(ctx, dataset, columns)
    missing_count = repeated = 0
    missing_sample = pl.DataFrame()
    missing = pl.any_horizontal(pl.col(key).is_null() for key in keys)
    error: Exception | None = None
    try:
        supported = relation is not None and supports_grouped(relation, value, how)
    except ValueError as exc:
        # Key diagnostics remain assessable even when a value's type is invalid.
        supported, error = relation is not None, exc
    if relation is not None and supported:
        if exclude_missing_keys:
            missing_sql = " OR ".join(f"{quote_identifier(key)} IS NULL" for key in keys)
            missing_count, missing_sample = relation.filter(missing, missing_sql).count_sample(
                ctx.evidence_limit
            )
            relation = relation.filter(~missing, f"NOT ({missing_sql})")
        if how == "none":
            repeated = duplicate_groups(relation, keys)
        try:
            values = (
                None if repeated or error else grouped_aggregate(relation, keys, value, alias, how)
            )
        except Exception as exc:
            values, error = None, exc
    else:
        # Unsupported typed kernels retain the exact projected reference path;
        # a supported sibling never inherits this materialization.
        source = normalize_missing(ctx, dataset, projected_frame(ctx, dataset, columns))
        if exclude_missing_keys:
            missing_count = int(source.select(missing.sum()).item())
            if missing_count:
                missing_sample = source.lazy().filter(missing).head(ctx.evidence_limit).collect()
                source = source.filter(~missing)
        if how == "none":
            repeated = duplicated_groups(source, keys)
        try:
            values = (
                None if repeated else aggregate(source, keys, value, alias, how, weight, order_by)
            )
        except Exception as exc:
            values, error = None, exc
    return SpatialAggregate(values, missing_count, missing_sample, repeated, error)
