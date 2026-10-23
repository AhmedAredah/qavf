"""Bounded SQL samples over already owned, coherent typed observations."""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

import polars as pl
import pyarrow as pa

from qavf.engine.sql import check_sql_columns, sample_result, sample_statement
from qavf.missingness import normalize_frame
from qavf.sources.base import SourceError
from qavf.sources.native import drop_temporary_relation
from qavf.sources.sql import quote_identifier, resolve_sql_paths
from qavf.sources.typed import needs_typed_sql_binding

if TYPE_CHECKING:
    from qavf.engine.context import DataContext
    from qavf.engine.observation import Observation


def sample_observed(
    ctx: DataContext,
    query: str,
    observations: Mapping[str, Observation],
    limit: int,
    *,
    normalize_missing: bool = False,
) -> tuple[pl.DataFrame, int]:
    """Bind bounded readers without reloading their original sources.

    Other referenced datasets retain normal binding, integrity and retry policy.
    Readers and registrations end before returning; observation owners remain
    with the caller. DuckDB owns the sampled CTE under native memory/spill limits.
    """
    names = ctx.referenced(query)
    others = [name for name in names if name not in observations]
    statement = sample_statement(
        resolve_sql_paths(query, ctx.env.base_dir, known_tables=ctx.declared()), limit
    )

    def attempt() -> tuple[pl.DataFrame, int]:
        with (
            contextlib.ExitStack() as readers,
            ctx._sql_lease(query, others, normalize_missing=normalize_missing) as con,
            contextlib.ExitStack() as tables,
        ):
            for name in names:
                if name not in observations:
                    continue
                observed = observations[name]
                schema = observed.schema()
                check_sql_columns(name, schema)
                unsupported = [
                    key for key, dtype in schema.items() if needs_typed_sql_binding(dtype)
                ]
                if unsupported:
                    raise SourceError(
                        f"dataset {name!r} has logical types that this SQL binding cannot "
                        f"preserve: {', '.join(unsupported)}. Use typed checks instead."
                    )
                spec = ctx.specs.get(name)
                columns = spec.schema_spec.columns if spec and spec.schema_spec else {}

                def batches(
                    observed=observed, columns=columns
                ) -> Generator[pa.RecordBatch, None, None]:
                    for frame in observed.batches():
                        if normalize_missing:
                            frame = normalize_frame(frame, columns)
                        table = pa.Table.from_arrays(
                            [column.to_arrow() for column in frame], names=frame.columns
                        )
                        yield from table.to_batches()

                stream = batches()
                readers.callback(stream.close)
                reader = pa.RecordBatchReader.from_batches(
                    pl.DataFrame(schema=schema).to_arrow().schema, stream
                )
                readers.callback(reader.close)
                alias = "__qavf_observation_" + uuid4().hex
                con.register(alias, reader)
                ctx._sql_connections.remember(alias)
                # A query may scan its input more than once (self-subqueries,
                # main-qualified references). Arrow readers are one-shot, so
                # stage once in DuckDB's bounded, spillable temporary storage.
                table_name = quote_identifier(name)
                con.execute(
                    f"CREATE TEMP TABLE {table_name} AS SELECT * FROM {quote_identifier(alias)}"
                )
                tables.callback(drop_temporary_relation, con, name, kind="TABLE")
            result = con.execute(statement)
            fetch = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table
            table = fetch()
        ctx._check_cancelled()
        sample, count = sample_result(table)
        ctx._check_materialization("SQL observation sample", sample)
        return sample, count

    return ctx._operations.run(ctx._native_participants(query, others), attempt)
