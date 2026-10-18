"""Bounded scalar batches over one normalized source, without execution retries."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import duckdb
import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.engine.sql import missing_sql
from qavf.rules.helpers import fresh_internal, missing_expr
from qavf.settings import SchemaSpec
from qavf.sources.sql import quote_identifier

BATCH_SIZE = 64

# Arbitrary SQL expressions can raise only for particular input values (casts,
# division, scalar subqueries, UDFs). Such checks retain independent evaluation.
_IDENTIFIER = r'(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z_0-9]*)'
_LITERAL = (
    r"(?:[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|'(?:[^']|'')*'|TRUE|FALSE|NULL)"
)
_SIMPLE_SCOPE = re.compile(
    rf"(?P<column>{_IDENTIFIER})\s*"
    rf"(?:(?P<operator>[<>=!]+)\s*(?P<literal>{_LITERAL})|IS\s+(?:NOT\s+)?NULL)",
    re.IGNORECASE,
)


def independent_scope(where: str | None, actual: dict[str, pl.DataType]) -> bool:
    """Keep data-dependent SQL conversions/functions isolated from siblings."""
    if where is None or where.strip().upper() in ("TRUE", "FALSE", "NULL"):
        return False
    matched = _SIMPLE_SCOPE.fullmatch(where.strip())
    if matched is None:
        return True
    literal = matched["literal"]
    if literal is None or literal.upper() == "NULL":
        return False
    column = matched["column"]
    if column.startswith('"'):
        column = column[1:-1].replace('""', '"')
    dtype = actual.get(column)
    if dtype is None:
        return True
    if literal.startswith("'"):
        return not (dtype == pl.String or isinstance(dtype, pl.Categorical | pl.Enum))
    if literal.upper() in ("TRUE", "FALSE"):
        return dtype != pl.Boolean
    return not dtype.is_numeric()


@dataclass(slots=True)
class ColumnInputs:
    """Shared alias layout; build only the dependencies of the current batch."""

    ctx: DataContext
    dataset: str
    schema: SchemaSpec
    actual: dict[str, pl.DataType]
    row: str = field(init=False)
    reason: str = field(init=False)
    saved: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        # DuckDB treats even quoted identifiers case-insensitively.
        used = set(self.actual) | {column.casefold() for column in self.actual}
        self.row = fresh_internal("row", used)
        used.add(self.row.casefold())
        self.reason = fresh_internal("violates", used)
        used.add(self.reason.casefold())
        self.saved = {}
        for column in self.actual:
            alias = fresh_internal("original_" + column, used)
            while alias.casefold() in used:
                alias += "_"
            self.saved[column] = alias
            used.add(alias.casefold())

    def validate(self, dependencies: Iterable[str]) -> None:
        for column in dependencies:
            if column not in self.actual:
                raise ValueError(f"required column {column!r} does not exist")

    def saved_for(self, dependencies: Iterable[str]) -> dict[str, str]:
        return {column: self.saved[column] for column in dependencies}

    def lazy(self, dependencies: Iterable[str], *, empty: bool = False) -> pl.LazyFrame:
        saved = self.saved_for(dependencies)
        typed = pl.DataFrame(schema=self.actual)
        plan = typed.lazy() if empty else self.ctx.scan(self.dataset)
        return (
            plan.with_row_index(self.row)
            .with_columns(pl.col(column).alias(alias) for column, alias in saved.items())
            .with_columns(
                pl.when(missing_expr(typed, column, self.schema.columns.get(column)))
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
                for column in saved
            )
        )

    def native(self, dependencies: Iterable[str]) -> str:
        q = quote_identifier
        saved = self.saved_for(dependencies)
        projections = [
            f"{q(column)} AS {q(saved[column])}" if column in saved else q(column)
            for column in self.actual
        ]
        for column in saved:
            missing = missing_sql(column, self.actual[column], self.schema.columns.get(column))
            projections.append(f"CASE WHEN {missing} THEN NULL ELSE {q(column)} END AS {q(column)}")
        return (
            f"(SELECT row_number() OVER () - 1 AS {q(self.row)}, "
            f"{', '.join(projections)} FROM {q(self.dataset)})"
        )


@dataclass(slots=True)
class ColumnReduction[T]:
    dependencies: tuple[str, ...]
    reducers: dict[str, T]
    finish: Callable[[dict[str, Any]], Finding | None]
    independent: bool = False


def reduce_columns(
    inputs: ColumnInputs,
    jobs: dict[int, ColumnReduction[Any]],
    *,
    lazy: bool,
) -> dict[int, Finding | Exception | None]:
    """Fuse scalar reductions, with bounded plan width and no failure-driven replay.

    Compilation errors are isolated before execution. A failed source read or
    resource limit invalidates its shared batch; retry decisions stay exclusively
    with the source's configured policy. Evidence samples remain per-check and
    bounded, so no all-failures table or source-sized mask is retained.
    """
    results: dict[int, Finding | Exception | None] = {}
    batches: list[list[tuple[int, ColumnReduction[Any]]]] = []
    shared: list[tuple[int, ColumnReduction[Any]]] = []
    for index, job in jobs.items():
        if job.independent:
            batches.append([(index, job)])
        else:
            shared.append((index, job))
            if len(shared) == BATCH_SIZE:
                batches.append(shared)
                shared = []
    if shared:
        batches.append(shared)

    def query(
        batch: list[tuple[int, ColumnReduction[Any]]],
    ) -> tuple[tuple[str, ...], list[tuple[int, str, str]]]:
        dependencies = tuple(dict.fromkeys(c for _, job in batch for c in job.dependencies))
        aliases = [
            (index, name, f"c{index}_{name}") for index, job in batch for name in job.reducers
        ]
        return dependencies, aliases

    def sql(batch: list[tuple[int, ColumnReduction[Any]]]) -> str:
        dependencies, aliases = query(batch)
        reducers = ", ".join(
            f"{jobs[index].reducers[name]} AS {quote_identifier(alias)}"
            for index, name, alias in aliases
        )
        return f"SELECT {reducers} FROM {inputs.native(dependencies)}"

    # Validate against owned zero-row data, never against the user's source. Only
    # these disposable validation plans may be split after an error. DuckDB uses
    # its own binder/regex engine rather than guessing compatibility from Polars.
    def preflight(
        batch: list[tuple[int, ColumnReduction[Any]]], connection: Any = None
    ) -> list[tuple[int, ColumnReduction[Any]]]:
        try:
            if lazy:
                dependencies, aliases = query(batch)
                inputs.lazy(dependencies, empty=True).select(
                    jobs[index].reducers[name].alias(alias) for index, name, alias in aliases
                ).collect()
            else:
                connection.execute(sql(batch)).fetchall()
            return batch
        except Exception as exc:
            if len(batch) == 1:
                results[batch[0][0]] = exc
                return []
            middle = len(batch) // 2
            return preflight(batch[:middle], connection) + preflight(batch[middle:], connection)

    # Isolated checks keep their original execution behavior, including dependent
    # scalar SQL. Shared checks are validated with no I/O and no user UDF execution.
    if lazy:
        batches = [batch if batch[0][1].independent else preflight(batch) for batch in batches]
    elif any(not batch[0][1].independent for batch in batches):
        with duckdb.connect(config={"threads": 1, "memory_limit": "16MB"}) as connection:
            connection.register(inputs.dataset, pl.DataFrame(schema=inputs.actual).to_arrow())
            batches = [
                batch if batch[0][1].independent else preflight(batch, connection)
                for batch in batches
            ]
    for batch in batches:
        if not batch:
            continue
        dependencies, aliases = query(batch)
        try:
            if lazy:
                reduced = inputs.ctx.collect(
                    inputs.dataset,
                    inputs.lazy(dependencies).select(
                        jobs[index].reducers[name].alias(alias) for index, name, alias in aliases
                    ),
                )
            else:
                reduced = inputs.ctx.sql(sql(batch))
            summary = reduced.row(0, named=True)
        except Exception as exc:
            results.update((index, exc) for index, _ in batch)
            continue
        for index, job in batch:
            try:
                results[index] = job.finish(
                    {name: summary[f"c{index}_{name}"] for name in job.reducers}
                )
            except Exception as exc:
                results[index] = exc
    return results
