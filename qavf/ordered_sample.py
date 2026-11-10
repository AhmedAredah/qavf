"""Count a query's rows exactly while keeping only its first rows by every column.

SQL evidence must be the same on every run, so a sample is the first ``limit``
rows in ``ORDER BY ALL`` order. Staging the whole result to sort it costs as
much temporary storage as the result itself. Instead one evaluation feeds two
aggregates: ``count(*)`` and ``min(row, limit)``, a bounded top-N over whole-row
structs. DuckDB compares structs field by field with the same ordering (and
NULLs last) as ``ORDER BY ALL``, so the kept rows are the ones a full sort would
keep, and only ``limit`` of them are ever held.

A result whose rows cannot be aggregated this way fails to bind; callers then
keep their staged form, which evaluates the query once into temporary storage.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import duckdb

from qavf.sources.sql import quote_identifier

COUNT = "__count"
SAMPLE = "__sample"


class OrderedSample:
    """One evaluation of a query: its exact row count and first ``limit`` rows."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("sample limit must be non-negative")
        self.limit = limit

    def statement(self, query: str) -> str:
        """A one-row ``(__count, __sample)`` result; ``__sample`` is a list of rows.

        ``query`` must be a single query. A zero limit keeps a typed empty sample.
        """
        # The row alias names the whole row; a fresh name cannot be a result column.
        row = quote_identifier("__qavf_row_" + uuid4().hex)
        keep = max(self.limit, 1)
        empty = "" if self.limit else " FILTER (WHERE false)"
        return (
            f"SELECT count(*) AS {COUNT}, min({row}, {keep}){empty} AS {SAMPLE} "
            f"FROM ({query}\n) AS {row}"
        )

    def bound(self, connection: Any, query: str) -> str | None:
        """:meth:`statement` when it binds on ``connection`` (binding does not run it).

        ``None`` when the result's rows cannot be ordered as whole-row values; the
        caller then stages the result instead.
        """
        statement = self.statement(query)
        try:
            connection.sql(statement)
        except (duckdb.BinderException, duckdb.NotImplementedException):
            return None
        return statement

    @staticmethod
    def rows(table: str) -> str:
        """The kept rows of a table holding a :meth:`statement` result, in order."""
        return (
            f"SELECT unnest(__qavf_kept) FROM (SELECT unnest({SAMPLE}) AS __qavf_kept "
            f"FROM {quote_identifier(table)}) ORDER BY ALL"
        )


__all__ = ["OrderedSample"]
