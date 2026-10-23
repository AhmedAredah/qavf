"""Bounded scalar batches over one normalized source, without execution retries."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import duckdb
import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
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
    observation: Observation | None = None
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

    def lazy(
        self,
        dependencies: Iterable[str],
        *,
        empty: bool = False,
        raw_columns: Iterable[str] = (),
    ) -> pl.LazyFrame:
        # Evidence-only columns retain raw values without applying their policies.
        # Otherwise an invalid sibling policy can poison an independent check.
        saved = self.saved_for(dependencies)
        originals = {**saved, **self.saved_for(raw_columns)}
        typed = pl.DataFrame(schema=self.actual)
        plan = (
            typed.lazy()
            if empty
            else self.observation.scan()
            if self.observation is not None
            else self.ctx.scan(self.dataset)
        )
        return (
            plan.with_row_index(self.row)
            .with_columns(pl.col(column).alias(alias) for column, alias in originals.items())
            .with_columns(
                pl.when(missing_expr(typed, column, self.schema.columns.get(column)))
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
                for column in saved
            )
        )

    def collect(self, plan: pl.LazyFrame) -> pl.DataFrame:
        """Collect from the owned observation when source reevaluation is unsafe."""
        return (
            self.observation.collect(plan)
            if self.observation is not None
            else self.ctx.collect(self.dataset, plan)
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
                reduced = inputs.collect(
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


def reduce_captured_columns(
    inputs: ColumnInputs,
    jobs: dict[int, ColumnReduction[pl.Expr]],
    capture_columns: tuple[str, ...],
) -> dict[int, Finding | Exception | None]:
    """Merge additive row counts and bounded typed samples from one evaluation.

    Each source batch drops its full rows after reducing. Only at-most-K evidence
    structs per check per batch proceed to the final merge. Polars controls input
    batch sizes and upstream buffering; this is not a process-RSS ceiling.
    """
    results: dict[int, Finding | Exception | None] = {}

    def expressions(batch: list[int]) -> list[pl.Expr]:
        return [
            value.alias(f"c{index}_{name}")
            for index in batch
            for name, value in jobs[index].reducers.items()
        ]

    def dependencies(batch: list[int]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c for index in batch for c in jobs[index].dependencies))

    def preflight(batch: list[int]) -> list[int]:
        if not batch:
            return []
        try:
            inputs.lazy(dependencies(batch), empty=True, raw_columns=capture_columns).select(
                expressions(batch)
            ).collect()
            return batch
        except Exception as exc:
            if len(batch) == 1:
                results[batch[0]] = exc
                return []
            middle = len(batch) // 2
            return preflight(batch[:middle]) + preflight(batch[middle:])

    batch = preflight(list(jobs))
    if not batch:
        return results
    needed = dependencies(batch)
    reductions = expressions(batch)
    typed = (
        inputs.lazy(needed, empty=True, raw_columns=capture_columns)
        .select(reductions)
        .collect_schema()
    )

    def reduce_batch(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(reductions)

    partials = (
        inputs.lazy(needed, raw_columns=capture_columns)
        .select(
            inputs.row,
            *needed,
            *(inputs.saved[c] for c in dict.fromkeys((*needed, *capture_columns))),
        )
        .map_batches(
            reduce_batch,
            schema=typed,
            streamable=True,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
        )
    )
    merged = []
    for index in batch:
        for name in jobs[index].reducers:
            column = f"c{index}_{name}"
            values = pl.col(column)
            if name == "sample":
                records = values.explode().drop_nulls()
                merged.append(
                    records.sort_by(records.struct.field(inputs.row))
                    .head(inputs.ctx.evidence_limit)
                    .implode()
                    .alias(column)
                )
            else:
                # Widen before adding partial counts; a partition count may be UInt32.
                merged.append(values.cast(pl.UInt64).sum().alias(column))
    try:
        reduced = inputs.collect(partials.select(merged))
    except Exception as exc:
        results.update((index, exc) for index in batch)
        return results
    for index in batch:
        try:
            summary = {}
            for name in jobs[index].reducers:
                series = reduced[f"c{index}_{name}"]
                if name == "sample":
                    # Keep Arrow/Polars types intact, including nanosecond times.
                    size = int(series.list.len().item())
                    summary[name] = series.explode().struct.unnest().head(size)
                else:
                    summary[name] = series.item()
            results[index] = jobs[index].finish(summary)
        except Exception as exc:
            results[index] = exc
    return results
