"""Read DuckDB results the same way on DuckDB 1.x and 2.x.

DuckDB 2.0 removed ``pl()``, ``arrow()`` and ``fetch_arrow_table()`` from the result
of ``execute`` and ``fetchone()`` from relations; ``to_arrow_table()`` replaces the
first three and is absent before 1.4. These helpers pick whichever exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa


def arrow_table(result: Any) -> pa.Table:
    """The complete result as an Arrow table."""
    fetch = getattr(result, "to_arrow_table", None) or result.fetch_arrow_table
    return fetch()


def record_batch_reader(result: Any, batch_rows: int) -> pa.RecordBatchReader:
    """The result as a stream of Arrow batches of at most ``batch_rows`` rows."""
    open_reader = getattr(result, "to_arrow_reader", None) or result.fetch_record_batch
    return open_reader(batch_rows)


def polars_frame(result: Any) -> pl.DataFrame:
    """The complete result as a Polars frame (requires the ``polars`` extra)."""
    convert = getattr(result, "pl", None)
    if convert is not None:
        return convert()
    import polars as pl

    table = arrow_table(result)
    frame = pl.from_arrow(table.rename_columns(unique_names(table.column_names)))
    assert isinstance(frame, pl.DataFrame)
    return frame


def unique_names(names: list[str]) -> list[str]:
    """Repeated column names suffixed ``_1``, ``_2``, ... as DuckDB's ``pl()`` does."""
    seen: set[str] = set()
    unique = []
    for name in names:
        candidate, number = name, 0
        while candidate in seen:
            number += 1
            candidate = f"{name}_{number}"
        seen.add(candidate)
        unique.append(candidate)
    return unique


def first_row(relation: Any) -> tuple[Any, ...] | None:
    """The first row of a relation, or ``None`` when it has none."""
    fetch = getattr(relation, "fetchone", None)
    if fetch is not None:
        return fetch()
    rows = relation.limit(1).fetchall()
    return tuple(rows[0]) if rows else None


def settle(connection: Any) -> None:
    """Finish a partly read result before registering or unregistering a relation.

    DuckDB 2.0 refuses such work while a result is unfinished, even after
    ``fetchone()`` read its only row; 1.x has no ``complete()`` and needs none.
    Finishing a result is cleanup, so an owned connection's cancellation check
    is bypassed: release after a cancelled rule must still succeed.
    """
    native = getattr(connection, "_connection", connection)
    complete = getattr(native, "complete", None)
    if complete is not None:
        complete()


def bind_relation(connection: Any, relation: Any, name: str) -> None:
    """Make ``relation`` queryable as ``name`` on ``connection``.

    A view is preferred. DuckDB 2.0 refuses a view over rows held in this
    process (such as a materialized volatile query), so those rows are registered
    under the name instead; the existing unregistration releases either.
    """
    import duckdb

    try:
        relation.create_view(name, replace=True)
    except duckdb.InvalidInputException as exc:
        if "held in this process" not in str(exc):
            raise
        settle(connection)
        connection.register(name, arrow_table(relation))
