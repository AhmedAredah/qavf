"""Canonical SQL key scopes and original row evidence for conditional keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from qavf.engine.context import DataContext
from qavf.rules._duplicate_rows import NativeDuplicateRows
from qavf.rules._execution import KeyRelation
from qavf.rules._key_values import CanonicalKeyValues
from qavf.rules.helpers import fresh_internal
from qavf.sources.sql import quote_identifier as qi


@dataclass(frozen=True)
class ConditionalKeySource:
    """A key's selected, canonical rows; positions precede all filtering."""

    ctx: DataContext
    dataset: str
    columns: tuple[str, ...]
    sql: str
    schema: dict[str, pl.DataType]
    row: str
    excluded_sql: str

    @classmethod
    def build(cls, ctx: DataContext, dataset: str, columns: list[str], where: str):
        actual = ctx.schema(dataset)
        row = fresh_internal("row", actual)
        na = fresh_internal("na_key", [*actual, row])
        values = CanonicalKeyValues(ctx, dataset, columns)
        source = (
            f"SELECT *, row_number() OVER () - 1 AS {qi(row)}, "
            f"({values.na_sql()}) AS {qi(na)} FROM {qi(dataset)}"
        )
        # The condition sees each declared column's converted value, including
        # non-key dependencies, while the N/A flag retains its raw meaning.
        canonical, schema = CanonicalKeyValues(ctx, dataset, actual).sql(
            source_sql=source, preserve=(row, na)
        )
        selected = (
            f"SELECT * FROM ({canonical}) canonical WHERE NOT {qi(na)} "
            f"AND coalesce(({where}), FALSE)"
        )
        excluded = (
            f"SELECT count(*) FROM ({canonical}) canonical WHERE {qi(na)} "
            f"AND coalesce(({where}), FALSE)"
        )
        return cls(ctx, dataset, tuple(columns), selected, schema, row, excluded)

    def exempt_na_rows(self) -> int:
        """N/A exemptions use the same canonical condition as the key population."""
        return int(self.ctx.sql(self.excluded_sql, datasets=(self.dataset,)).item())

    def relation(self, frequency: str) -> KeyRelation:
        selected = ", ".join(qi(c) for c in self.columns)
        return KeyRelation(
            self.ctx,
            (self.dataset,),
            self.columns,
            frequency,
            {**{c: self.schema[c] for c in self.columns}, frequency: pl.UInt64()},
            sql=f"SELECT {selected}, count(*)::UBIGINT AS {qi(frequency)} "
            f"FROM ({self.sql}) selected GROUP BY {selected}",
            normalize_missing=False,
        )

    def positions(self, name: str, limit: int | Literal["all"]) -> NativeDuplicateRows:
        return ConditionalDuplicateRows(self, name, limit)


class ConditionalDuplicateRows(NativeDuplicateRows):
    """Use exactly the condition used for counts when looking up row evidence."""

    def __init__(self, source: ConditionalKeySource, name: str, limit: int | Literal["all"]):
        super().__init__(source.ctx, source.dataset, source.columns, name, limit)
        self.source = source

    def key_source(self, row_name: str):
        fields = ", ".join(qi(c) for c in self.columns)
        return (
            f"SELECT {fields}, {qi(self.source.row)} AS {qi(row_name)} "
            f"FROM ({self.source.sql}) scoped_keys",
            self.source.schema,
        )
