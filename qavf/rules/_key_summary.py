"""Bounded scalar summaries sharing one grouped-key evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from qavf.rules.helpers import fresh_internal
from qavf.sources.sql import quote_identifier

if TYPE_CHECKING:
    from qavf.rules._execution import KeyRelation


@dataclass(frozen=True, slots=True)
class KeySelection:
    """Equivalent backend predicates and the maximum retained sample rows."""

    sql: str = "true"
    predicate: pl.Expr = field(default_factory=lambda: pl.lit(True))
    limit: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("sample limit must be non-negative")


@dataclass(frozen=True, slots=True)
class KeySummary:
    """Exact key/row counts with only a bounded sample transferred to Python."""

    keys: int
    rows: int
    sample: pl.DataFrame


def summarize_keys(
    relation: KeyRelation, selections: Mapping[str, KeySelection]
) -> dict[str, KeySummary]:
    """Fuse related summaries while bounding query width and retained samples."""
    items = list(selections.items())
    summaries = {}
    for start in range(0, len(items), 16):
        batch = items[start : start + 16]
        aliases = []
        # SQL identifiers are case-insensitive, including quoted identifiers.
        reserved = [name.lower() for name in relation.schema]
        for _ in batch:
            names = []
            for role in ("keys", "rows", "sample"):
                name = fresh_internal("summary_" + role, reserved)
                reserved.append(name)
                names.append(name)
            aliases.append(tuple(names))
        result = (
            _native_summary(relation, batch, aliases)
            if relation.sql is not None
            else _lazy_summary(relation, batch, aliases)
        )
        for (label, _), (keys, rows, sample) in zip(batch, aliases, strict=True):
            values = result[sample][0]
            frame = (
                values.struct.unnest()
                if values is not None
                else pl.DataFrame(schema=relation.schema)
            )
            summaries[label] = KeySummary(int(result[keys][0]), int(result[rows][0]), frame)
    return summaries


def _native_summary(
    relation: KeyRelation,
    batch: list[tuple[str, KeySelection]],
    aliases: list[tuple[str, ...]],
) -> pl.DataFrame:
    frequency = quote_identifier(relation.frequency)
    padding = quote_identifier(
        fresh_internal("summary_padding", (name.lower() for name in relation.schema))
    )
    table = quote_identifier(
        fresh_internal(
            "key_rows", (name.lower() for name in (*relation.datasets, *relation.schema))
        )
    )
    selected = ", ".join(quote_identifier(name) for name in relation.schema)
    reducers: list[str] = []
    for (_, selection), names in zip(batch, aliases, strict=True):
        keys, rows, sample = map(quote_identifier, names)
        condition = f"NOT {padding} AND ({selection.sql})"
        reducers.extend(
            (
                f"count(*) FILTER (WHERE {condition})::UBIGINT AS {keys}",
                f"coalesce(sum({frequency}) FILTER (WHERE {condition}), 0)::UBIGINT AS {rows}",
                f"(SELECT list(s) FROM (SELECT {selected} FROM {table} "
                f"WHERE {condition} LIMIT {selection.limit}) s) AS {sample}",
            )
        )
    # A marked padding row preserves scalar results for provably empty relations
    # on DuckDB 1.2. Every count, weight, and bounded sample excludes that row.
    return relation.ctx.sql(
        f"WITH {table} AS MATERIALIZED ("
        f"SELECT grouped.*, false AS {padding} FROM ({relation.sql}) AS grouped "
        f"UNION ALL BY NAME SELECT true AS {padding}) "
        f"SELECT {', '.join(reducers)} FROM {table}",
        datasets=relation.datasets,
        normalize_missing=relation.normalize_missing,
    )


def _lazy_summary(
    relation: KeyRelation,
    batch: list[tuple[str, KeySelection]],
    aliases: list[tuple[str, ...]],
) -> pl.DataFrame:
    if relation.lazy is None:
        raise ValueError("a key relation requires an execution plan")
    reducers: list[pl.Expr] = []
    for (_, selection), (keys, rows, sample) in zip(batch, aliases, strict=True):
        predicate = selection.predicate.fill_null(False)
        frequency = pl.col(relation.frequency).filter(predicate)
        reducers.extend(
            (
                frequency.len().cast(pl.UInt64).alias(keys),
                frequency.sum().cast(pl.UInt64).alias(rows),
                pl.struct(pl.all()).filter(predicate).head(selection.limit).implode().alias(sample),
            )
        )
    return relation.collect(relation.lazy.select(reducers))
