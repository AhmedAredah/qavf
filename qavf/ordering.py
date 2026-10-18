"""Deterministic ordered aggregation without depending on physical row order."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

OrderBy = str | tuple[str, ...] | None


def ordering_columns(order_by: OrderBy) -> tuple[str, ...]:
    """Normalize one ordering column or explicit lexicographic tie-break columns."""
    if order_by is None:
        return ()
    columns: tuple[str, ...]
    if isinstance(order_by, str):
        columns = (order_by,)
    elif isinstance(order_by, tuple | list):
        columns = tuple(order_by)
    else:
        raise ValueError("order_by must name a column or a nonempty sequence of columns")
    if not columns or any(not isinstance(name, str) or not name for name in columns):
        raise ValueError("order_by must name nonempty columns")
    if len(set(columns)) != len(columns):
        raise ValueError("order_by columns must be distinct")
    return columns


def usable_order(
    values: pl.Expr, columns: tuple[str, ...], schema: Mapping[str, pl.DataType]
) -> pl.Expr:
    """Select rows whose value and every ordering component are usable."""
    selected = values.is_not_null()
    for name in columns:
        order = pl.col(name)
        usable = order.is_not_null()
        if schema[name].is_float():
            usable &= ~order.is_nan()
        selected &= usable
    return selected


def ordered_candidates(
    values: pl.Expr, columns: tuple[str, ...], schema: Mapping[str, pl.DataType], *, last: bool
) -> pl.Expr:
    """Retain usable values at the extreme lexicographic key within each group.

    Each ordering component narrows the candidates by a min/max reduction. No
    full-row sort or physical-row tie breaker is required. The caller must reject
    conflicting values among the remaining ties before publishing a result.
    """
    selected = usable_order(values, columns, schema)
    for name in columns:
        order = pl.col(name)
        candidates = order.filter(selected)
        extreme = candidates.max() if last else candidates.min()
        selected = selected & (order == extreme)
    return values.filter(selected)
