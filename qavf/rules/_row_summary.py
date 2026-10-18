"""Exact row classifications with bounded original-row evidence in one evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

from qavf.engine.context import DataContext
from qavf.rules.helpers import fresh_internal
from qavf.sources.sql import quote_identifier, sql_literal


@dataclass(frozen=True, slots=True)
class RowSummary:
    rows: int
    counts: dict[str, int]
    sample: pl.DataFrame

    @property
    def invalid(self) -> int:
        return sum(self.counts.values())


def row_aliases(columns: Sequence[str]) -> tuple[str, str]:
    """Keep internal identifiers distinct even on case-insensitive SQL backends."""
    used = {name.casefold() for name in columns}
    row = fresh_internal("row", used)
    problem = fresh_internal("problem", [*used, row])
    return row, problem


def summarize_rows(
    ctx: DataContext,
    dataset: str,
    *,
    problem: str,
    labels: Sequence[str],
    sample_columns: Sequence[str],
    sample_order: str | None = None,
    build: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    sql: str | None = None,
) -> RowSummary:
    """Reduce a labelled plan; every nonnull problem belongs to exactly one label."""
    reserved = {name.casefold() for name in (*sample_columns, dataset, problem)}
    total = fresh_internal("row_total", reserved)
    sample = fresh_internal("row_sample", [*reserved, total])
    aliases = [
        fresh_internal(f"problem_{i}", [*reserved, total, sample]) for i in range(len(labels))
    ]
    limit = ctx.evidence_limit
    if sql is not None:
        q = quote_identifier
        reducers = [f"count(*) AS {q(total)}"]
        reducers.extend(
            f"count(*) FILTER (WHERE {q(problem)} = {sql_literal(label)}) AS {q(alias)}"
            for label, alias in zip(labels, aliases, strict=True)
        )
        if limit:
            if sample_order is None:
                raise ValueError("native bounded evidence requires an input row position")
            fields = ", ".join(f"{q(column)} := {q(column)}" for column in sample_columns)
            reducers.append(
                f"arg_min(struct_pack({fields}), {q(sample_order)}, {limit}) "
                f"FILTER (WHERE {q(problem)} IS NOT NULL) AS {q(sample)}"
            )
        # Bounded arg_min retains only the first K original rows. A materialized
        # labelled CTE would unnecessarily retain all wide rows for count + sample.
        reduced = ctx.sql(
            f"SELECT {', '.join(reducers)} FROM ({sql}) AS classified",
            datasets=(dataset,),
        )
    else:
        if build is None:
            raise ValueError("a row summary requires a lazy or SQL classification plan")
        reducers_lazy = [pl.len().alias(total)]
        reducers_lazy.extend(
            (pl.col(problem) == label).sum().alias(alias)
            for label, alias in zip(labels, aliases, strict=True)
        )
        if limit:
            reducers_lazy.append(
                pl.struct(sample_columns)
                .filter(pl.col(problem).is_not_null())
                .head(limit)
                .implode()
                .alias(sample)
            )
        reduced = ctx.reduce(dataset, lambda source: build(source).select(reducers_lazy))
    values = reduced[sample][0] if limit else None
    evidence = values.struct.unnest() if values is not None else pl.DataFrame()
    return RowSummary(
        int(reduced[total][0]),
        {label: int(reduced[alias][0] or 0) for label, alias in zip(labels, aliases, strict=True)},
        evidence,
    )
