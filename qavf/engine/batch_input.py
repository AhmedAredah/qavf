"""Coherent admission of owned source batches into the existing observation store."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

import polars as pl

from qavf.engine.observation import Observation, _ObservationBuilder
from qavf.sources.base import DataSource, SourceError
from qavf.sources.capabilities import BatchReader, SourceBatchReader

if TYPE_CHECKING:
    from qavf.engine.context import DataContext


def observe_batches(
    ctx: DataContext, source: DataSource, rename: Mapping[str, str] | None = None
) -> Observation:
    """Own exactly one complete stream attempt, with no population published early.

    The caller's source-operation scope owns retry/identity policy. A failed
    attempt releases its builder before a retry can begin. Native table buffers
    are detached or spilled by the same builder as other observations.
    """
    reader = SourceBatchReader.adapt(source)
    builder = _ObservationBuilder(ctx)
    try:
        schema = consume_batches(
            reader,
            builder,
            ctx.observation_storage.check_cancelled,
            ctx.check_materialization,
            rename,
        )
        return builder.finish(schema)
    finally:
        builder.close()


class BatchSink(Protocol):
    def append(self, frame: pl.DataFrame) -> None: ...


def consume_batches(
    reader: BatchReader,
    sink: BatchSink,
    check_cancelled: Callable[[], None],
    check_materialization: Callable[[str, pl.DataFrame], None],
    rename: Mapping[str, str] | None = None,
) -> dict[str, pl.DataType]:
    """Assess one complete owned stream without depending on an engine facade."""
    schema = reader.schema
    names = dict(rename or {})
    output_schema = {names.get(name, name): dtype for name, dtype in schema.items()}
    rows = 0
    read_failure: BaseException | None = None
    check_cancelled()
    with reader.open() as batches:
        try:
            for batch in batches:
                check_cancelled()
                if not isinstance(batch, pl.DataFrame) or list(batch.schema.items()) != list(
                    schema.items()
                ):
                    raise SourceError("source batch schema/order differs from its declaration")
                rows += batch.height
                budget = reader.row_budget
                if budget is not None and rows > budget:
                    raise SourceError(f"batch source has more than {budget:,} rows (row_budget)")
                check_materialization("source batch", batch)
                sink.append(batch.rename(names) if names else batch)
        except BaseException as exc:
            # A plugin cannot suppress an admission failure to publish partial data.
            read_failure = exc
            raise
    if read_failure is not None:
        raise read_failure
    check_cancelled()
    reader.record_rows(rows)
    return output_schema
