"""Run-local source adapters for explicit Query and Loader descriptions."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import duckdb
import polars as pl

from qavf.duckdb_results import polars_frame
from qavf.settings import SourcePolicy
from qavf.sources.base import DataSource, ScanPlan, SourceEnvironment, SourceError, SourceParams
from qavf.sources.frames import admitted_frame
from qavf.sources.native import budgeted, check_budget, connect
from qavf.sources.sql import normalize_sql_paths, referenced_connections

from .descriptors import BatchSource, Loader, Query


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
            frame = polars_frame(con.execute(budgeted(sql, self.params.row_budget), parameters))
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


class _FrameInput(DataSource[SourceParams]):
    """A borrowed pandas frame or Arrow table under a row budget, converted on load.

    A frame whose stated row count exceeds the budget is refused before any
    conversion allocates a copy of it.
    """

    kind: ClassVar[str] = "frame_input"
    Params: ClassVar[type[SourceParams]] = SourceParams

    def __init__(
        self, frame: object, env: SourceEnvironment, policy: SourcePolicy | None = None
    ) -> None:
        super().__init__(
            SourceParams(
                type=self.kind, **(policy.model_dump(exclude_unset=True) if policy else {})
            ),
            env,
        )
        self.frame = frame

    def bind(self, env: SourceEnvironment) -> _FrameInput:
        policy = SourcePolicy.model_validate(self.params.model_dump(exclude={"type"}))
        return type(self)(self.frame, self.bound_environment(env), policy)

    def load(self) -> pl.DataFrame:
        return admitted_frame(self.frame, self.params.row_budget, self.describe())

    def describe(self) -> str:
        return f"in-memory {type(self.frame).__name__}"

    def scan_polars(self) -> pl.LazyFrame | None:
        return self.frame.clone() if isinstance(self.frame, pl.LazyFrame) else None


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
        return admitted_frame(value, self.params.row_budget, self.describe())

    def describe(self) -> str:
        function = self.loader.function
        return f"loader: {getattr(function, '__qualname__', type(function).__name__)}"


class _BatchInput(DataSource[SourceParams]):
    """The run-local lifecycle of a validated owned batch factory."""

    kind: ClassVar[str] = "batch_input"
    Params: ClassVar[type[SourceParams]] = SourceParams

    def __init__(
        self, batch: BatchSource, env: SourceEnvironment, policy: SourcePolicy | None = None
    ) -> None:
        super().__init__(
            SourceParams(
                type=self.kind, **(policy.model_dump(exclude_unset=True) if policy else {})
            ),
            env,
        )
        self.batch = batch
        self.retry_safe = batch.replayable and batch.retry_safe
        self._opened = False
        self.load_notes.update(
            replayable=batch.replayable,
            retry_safe=batch.retry_safe,
            execution="owned_batch_snapshot",
        )

    def bind(self, env: SourceEnvironment) -> _BatchInput:
        policy = SourcePolicy.model_validate(self.params.model_dump(exclude={"type"}))
        return type(self)(self.batch, self.bound_environment(env), policy)

    def batch_schema(self) -> dict[str, pl.DataType]:
        import pyarrow as pa

        if isinstance(self.batch.schema, pa.Schema):
            frame = pl.from_arrow(pa.Table.from_batches([], schema=self.batch.schema))
            assert isinstance(frame, pl.DataFrame)
            return dict(frame.schema)
        return dict(pl.Schema(self.batch.schema))

    @contextmanager
    def open_batches(self) -> Iterator[Iterator[pl.DataFrame]]:
        with self._state_lock:
            if self._opened and not self.retry_safe:
                raise SourceError("BatchSource stream was already opened and is not retry-safe")
            self._opened = True
        manager = self.batch.factory(**self.batch.kwargs)
        if not callable(getattr(manager, "__enter__", None)) or not callable(
            getattr(manager, "__exit__", None)
        ):
            # A factory returning a raw reader did not establish the promised
            # ownership contract. Close it if possible rather than leak it.
            close = getattr(manager, "close", None)
            if callable(close):
                close()
            raise SourceError("BatchSource.factory must return an owned context manager")
        with manager as batches:
            yield self._converted(batches)

    def _converted(self, batches: Iterable[object]) -> Iterator[pl.DataFrame]:
        """Admit each produced batch as an eager frame matching the declared schema/order."""
        for batch in batches:
            if isinstance(batch, pl.LazyFrame):
                raise SourceError("BatchSource batches must be eager typed tables")
            frame = admitted_frame(batch, self.params.row_budget, self.describe())
            if dict(frame.schema) != self.batch_schema():
                raise SourceError("BatchSource batch schema/order differs from its declaration")
            # Dict equality ignores order; column order is part of this contract.
            if frame.columns != list(self.batch_schema()):
                raise SourceError("BatchSource batch schema/order differs from its declaration")
            yield frame

    def load(self) -> pl.DataFrame:
        raise SourceError("BatchSource requires QAVF's managed observation engine")

    def describe(self) -> str:
        factory = self.batch.factory
        return f"batch source: {getattr(factory, '__qualname__', type(factory).__name__)}"
