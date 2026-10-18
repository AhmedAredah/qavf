"""SQL and named-database sources sharing result and connection policy."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import polars as pl

from qavf.sources.base import DataSource, ScanPlan, SourceParams
from qavf.sources.native import budgeted, check_budget, connect, fetch
from qavf.sources.parameters import DatabaseParams, QueryParams
from qavf.sources.sql import (
    quote_identifier,
    referenced_connections,
    resolve_sql_paths,
    validate_query,
)


class SQLSource[P: SourceParams](DataSource[P]):
    """Common bounded reads for sources defined by SQL and attachments.

    Concrete sources own query construction; this contract owns result budgets,
    connection lifetime, and snapshot selection. A borrowed snapshot connection
    remains open, while a load always closes its own native connection.
    """

    @abstractmethod
    def _sql(self) -> str:
        """Return the complete query with source-relative paths resolved."""

    @abstractmethod
    def _attachments(self) -> list[str]:
        """Return the configured database names required by the query."""

    def _budget_description(self) -> str:
        return self.describe()

    def scan(self) -> ScanPlan:
        return ScanPlan(validate_query(self._sql()), connections=tuple(self._attachments()))

    def load(self) -> pl.DataFrame:
        # Validate before opening a connection so a rejected query cannot leak it.
        sql = budgeted(self._sql(), self.params.row_budget)
        frame = fetch(connect(self.env, self._attachments()), sql)
        return check_budget(frame, self.params.row_budget, self._budget_description())

    def read(self, con: Any) -> pl.DataFrame:
        sql = budgeted(self._sql(), self.params.row_budget)
        frame = fetch(con, sql, keep_open=True)
        return check_budget(frame, self.params.row_budget, self._budget_description())

    def snapshot_connection(self) -> str | None:
        attached = self._attachments()
        if len(attached) != 1:
            return None
        spec = self.env.connections.get(attached[0])
        return attached[0] if spec is not None and spec.snapshot else None


class QuerySource(SQLSource[QueryParams]):
    """The result of a DuckDB SQL query."""

    kind: ClassVar[str] = "query"
    Params: ClassVar[type[SourceParams]] = QueryParams

    def _attachments(self) -> list[str]:
        if self.params.connections is not None:
            return list(self.params.connections)
        return referenced_connections(self.params.sql, self.env.connections)

    def _sql(self) -> str:
        return resolve_sql_paths(self.params.sql, self.env.base_dir)

    def _budget_description(self) -> str:
        return "the query"

    def describe(self) -> str:
        first_line = self.params.sql.strip().splitlines()[0]
        return f"query: {first_line[:80]}"


class DatabaseSource(SQLSource[DatabaseParams]):
    """A table or query from a DuckDB, SQLite, PostgreSQL or MySQL/MariaDB database."""

    kind: ClassVar[str] = "database"
    Params: ClassVar[type[SourceParams]] = DatabaseParams

    def _attachments(self) -> list[str]:
        return [self.params.connection]

    def _sql(self) -> str:
        p = self.params
        known_tables: tuple[str, ...] = ()
        if p.query is not None:
            sql = p.query
        else:
            assert p.table is not None
            parts = [p.connection, *([p.db_schema] if p.db_schema else []), p.table]
            sql = "SELECT * FROM " + ".".join(quote_identifier(part) for part in parts)
            known_tables = (".".join(parts),)
        return resolve_sql_paths(sql, self.env.base_dir, known_tables=known_tables)

    def describe(self) -> str:
        p = self.params
        target = p.table if p.table else "query"
        return f"{p.connection}:{p.db_schema + '.' if p.db_schema else ''}{target}"
