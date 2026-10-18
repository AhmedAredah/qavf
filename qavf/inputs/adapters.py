"""Run-local source adapters for explicit Query and Loader descriptions."""

from __future__ import annotations

import threading
from typing import Any, ClassVar

import duckdb
import polars as pl

from qavf.settings import SourcePolicy
from qavf.sources.base import DataSource, ScanPlan, SourceEnvironment, SourceError, SourceParams
from qavf.sources.frames import as_polars
from qavf.sources.native import budgeted, check_budget, connect
from qavf.sources.sql import normalize_sql_paths, referenced_connections

from .descriptors import Loader, Query


class _QueryInput(DataSource[SourceParams]):
    kind: ClassVar[str] = "query_input"
    Params: ClassVar[type[SourceParams]] = SourceParams

    def __init__(
        self,
        query: Query,
        env: SourceEnvironment,
        lock: threading.Lock,
        policy: SourcePolicy | None = None,
    ) -> None:
        super().__init__(
            SourceParams(
                type=self.kind, **(policy.model_dump(exclude_unset=True) if policy else {})
            ),
            env,
        )
        self.query = query
        self._query_lock = lock
        if isinstance(query.connection, str) and query.connection not in env.connections:
            raise ValueError(f"unknown configured connection {query.connection!r}")
        if isinstance(query.connection, duckdb.DuckDBPyConnection):
            self.load_notes["execution"] = "borrowed connection: eager load, never closed by QAVF"

    def bind(self, env: SourceEnvironment) -> _QueryInput:
        """Create run-local state while retaining borrowed connection ownership."""
        policy = SourcePolicy.model_validate(self.params.model_dump(exclude={"type"}))
        return type(self)(self.query, self.bound_environment(env), self._query_lock, policy)

    def _attachments(self) -> tuple[str, ...]:
        if isinstance(self.query.connection, str):
            return (self.query.connection,)
        return tuple(referenced_connections(self.query.sql, self.env.connections))

    def scan(self) -> ScanPlan | None:
        if isinstance(self.query.connection, duckdb.DuckDBPyConnection):
            return None
        sql, parameters = normalize_sql_paths(
            self.query.sql, self.env.base_dir, self.query.parameters
        )
        return ScanPlan(
            sql=sql,
            parameters=parameters,
            connections=self._attachments(),
        )

    def snapshot_connection(self) -> str | None:
        if isinstance(self.query.connection, duckdb.DuckDBPyConnection):
            return None
        attached = self._attachments()
        if len(attached) == 1:
            spec = self.env.connections.get(attached[0])
            if spec is not None and spec.snapshot:
                return attached[0]
        return None

    def read(self, con: Any) -> pl.DataFrame:
        with self._query_lock:
            sql, parameters = normalize_sql_paths(
                self.query.sql, self.env.base_dir, self.query.parameters
            )
            frame = con.execute(budgeted(sql, self.params.row_budget), parameters).pl()
            return check_budget(frame, self.params.row_budget, self.describe())

    def load(self) -> pl.DataFrame:
        if isinstance(self.query.connection, duckdb.DuckDBPyConnection):
            return self.read(self.query.connection)
        con = connect(self.env, self._attachments())
        try:
            return self.read(con)
        finally:
            con.close()

    def describe(self) -> str:
        borrowed = isinstance(self.query.connection, duckdb.DuckDBPyConnection)
        return "parameterized query" + (" (borrowed connection)" if borrowed else "")


class _LoaderInput(DataSource[SourceParams]):
    kind: ClassVar[str] = "loader_input"
    Params: ClassVar[type[SourceParams]] = SourceParams

    def __init__(
        self,
        loader: Loader,
        env: SourceEnvironment,
        policy: SourcePolicy | None = None,
    ) -> None:
        super().__init__(
            SourceParams(
                type=self.kind,
                **(policy.model_dump(exclude_unset=True) if policy else {}),
            ),
            env,
        )
        self.loader = loader
        self.retry_safe = loader.retry_safe and loader.replayable
        self.load_notes.update(replayable=loader.replayable, retry_safe=loader.retry_safe)

    def bind(self, env: SourceEnvironment) -> _LoaderInput:
        policy = SourcePolicy.model_validate(self.params.model_dump(exclude={"type"}))
        return type(self)(self.loader, self.bound_environment(env), policy)

    def load(self) -> pl.DataFrame | pl.LazyFrame:
        value = self.loader.function(**self.loader.kwargs)
        if callable(getattr(value, "read_next_batch", None)):
            raise SourceError("Loader must return a frame/table, not a borrowed one-pass reader")
        if isinstance(value, pl.LazyFrame):
            # Keep one overflow row so the context rejects excess input instead
            # of accepting a truncated dataset, without materializing all rows.
            budget = self.params.row_budget
            return value.limit(budget + 1) if budget is not None else value
        return check_budget(as_polars(value), self.params.row_budget, self.describe())

    def describe(self) -> str:
        function = self.loader.function
        return f"loader: {getattr(function, '__qualname__', type(function).__name__)}"
