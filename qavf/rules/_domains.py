"""Lossless lazy key-domain operations with bounded scalar/evidence collection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import polars as pl

from qavf.keys import key_casts
from qavf.rules.helpers import fresh_internal

Collector = Callable[[pl.LazyFrame], pl.DataFrame]


def align_domains(
    left: pl.LazyFrame, right: pl.LazyFrame, columns: Sequence[str], collect: Collector
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Prove casts using scalar reductions; never collect an identifier domain."""
    left_schema, right_schema = left.collect_schema(), right.collect_schema()
    a_types, b_types = {}, {}
    for name in columns:

        def exact_float(column: str = name) -> bool:
            for plan, schema in ((left, left_schema), (right, right_schema)):
                value = pl.col(column)
                valid = value.is_between(-(2**53), 2**53)
                if schema[column].is_float():
                    valid &= value.is_finite() & (value == value.floor())
                if not collect(plan.select(valid.all())).item():
                    return False
            return True

        a, b = key_casts(name, left_schema[name], right_schema[name], exact_float=exact_float)
        if a is not None:
            a_types[name] = a
        if b is not None:
            b_types[name] = b
    return (
        left.with_columns(pl.col(name).cast(dtype) for name, dtype in a_types.items()),
        right.with_columns(pl.col(name).cast(dtype) for name, dtype in b_types.items()),
    )


def union_domains(plans: Iterable[pl.LazyFrame], column: str, collect: Collector) -> pl.LazyFrame:
    """Combine distinct domains after checking every semantic dtype boundary."""
    iterator = iter(plans)
    try:
        result = next(iterator).select(column).drop_nulls()
    except StopIteration:
        return pl.DataFrame(schema={column: pl.Null}).lazy()
    for plan in iterator:
        result, other = align_domains(result, plan.select(column).drop_nulls(), [column], collect)
        result = pl.concat([result, other])
    return result.unique()


def count(plan: pl.LazyFrame, collect: Collector) -> int:
    return int(collect(plan.select(pl.len())).item())


def count_sample(
    plan: pl.LazyFrame, limit: int, collect: Collector, *, order: Sequence[str] = ()
) -> tuple[int, pl.DataFrame]:
    """Count every row while collecting only the requested evidence prefix."""
    schema = dict(plan.collect_schema())
    total = fresh_internal("domain_count", schema)
    sample = fresh_internal("domain_sample", [*schema, total])
    reducers = [pl.len().alias(total)]
    if limit:
        selected = pl.struct(list(schema))
        if order:
            selected = selected.sort_by(list(order))
        reducers.append(selected.head(limit).implode().alias(sample))
    reduced = collect(plan.select(reducers))
    values = reduced[sample][0] if limit else None
    frame = values.struct.unnest() if values is not None else pl.DataFrame(schema=schema)
    return int(reduced[total][0]), frame
