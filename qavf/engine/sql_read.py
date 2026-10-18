"""Operation-scoped native SQL readers with resettable, bounded batch consumers.

A consumer owns no connection: every attempt creates a fresh consumer and closes
all readers and its reserved connection before source postconditions can publish
its result. Native capability probing never materializes unsupported inputs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.connections import Connection, merge_connection_binding
from qavf.engine.projection import repeatable_file_source
from qavf.engine.relations import bind_scan
from qavf.engine.sql import check_sql_columns
from qavf.settings import ConnectionSpec, check_sql_names
from qavf.sources.base import DataSource, SourceError, SourceRegistry
from qavf.sources.sql import normalize_single_query, quote_identifier, resolve_sql_paths

if TYPE_CHECKING:
    from qavf.engine.context import DataContext


class NativeReadUnavailable(RuntimeError):
    """A participant requires a typed, eager or snapshot reader instead of SQL."""


class SqlReadSession:
    """Read-only queries valid only inside ``DataContext.consume_sql``.

    ``batches`` yields Polars frames and must be exhausted within the callback.
    At most one batch reader may be active. Per-batch memory targets are checked
    after decoding; callers must keep aggregate state bounded themselves.
    """

    def __init__(
        self, ctx: DataContext, connection: Connection, sources: Mapping[str, DataSource]
    ) -> None:
        self._ctx = ctx
        self._connection = connection
        self._sources = dict(sources)
        self._readers: dict[int, Any] = {}
        self._incomplete = False
        self._closed = False

    def _statement(self, query: str) -> str:
        self._ctx._check_cancelled()
        if self._closed:
            raise RuntimeError("SQL read session has ended")
        if self._readers:
            raise RuntimeError("exhaust the active batch reader before another query")
        return normalize_single_query(
            resolve_sql_paths(
                query,
                self._ctx.env.base_dir,
                known_tables=set(self._ctx.declared()) | set(self._sources),
            )
        )

    def schema(self, query: str) -> dict[str, pl.DataType]:
        """Resolve output types without collecting data rows."""
        return dict(self._connection.sql(self._statement(query)).limit(0).pl().schema)

    def frame(self, query: str) -> pl.DataFrame:
        """Collect a small metadata/reduction query within the materialization target."""
        result = self._connection.execute(self._statement(query)).pl()
        self._ctx._check_materialization("SQL read result", result)
        self._ctx._check_cancelled()
        return result

    def batches(self, query: str, batch_rows: int = 65536) -> Iterator[pl.DataFrame]:
        """Decode bounded Arrow batches without retaining prior result batches."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        statement = self._statement(query)
        query_result = self._connection.execute(statement)
        reader = (
            query_result.to_arrow_reader(batch_rows)
            if hasattr(query_result, "to_arrow_reader")
            else query_result.fetch_record_batch(batch_rows)
        )
        self._readers[id(reader)] = reader
        exhausted = False
        try:
            for batch in reader:
                self._ctx._check_cancelled()
                frame = pl.from_arrow(batch)
                assert isinstance(frame, pl.DataFrame)
                self._ctx._check_materialization("SQL result batch", frame)
                yield frame
            exhausted = True
            self._ctx._check_cancelled()
        finally:
            reader.close()
            self._readers.pop(id(reader), None)
            if not exhausted:
                self._incomplete = True

    def repeatable_scan(self, alias: str) -> bool:
        """Whether this identity-verified participant supports separate read passes.

        This checks the same bound source that supplies the session's data and
        provenance, including ad-hoc aliases. It does not guarantee row ordering.
        """
        self._ctx._check_cancelled()
        if self._closed:
            raise RuntimeError("SQL read session has ended")
        return repeatable_file_source(self._sources[alias])

    def fingerprint(self, alias: str, *, hash_contents: bool = True) -> dict[str, Any] | None:
        """Fingerprint the same bound source instance used by this session."""
        self._ctx._check_cancelled()
        if self._closed:
            raise RuntimeError("SQL read session has ended")
        return self._sources[alias].fingerprint(hash_contents=hash_contents)

    def close(self, *, require_exhausted: bool = False) -> None:
        pending = bool(self._readers)
        try:
            for reader in tuple(self._readers.values()):
                reader.close()
        finally:
            self._readers.clear()
            self._closed = True
        if require_exhausted and (pending or self._incomplete):
            raise RuntimeError("batch consumers must exhaust each query before returning a result")


def consume_native[T](
    ctx: DataContext,
    consumer_factory: Callable[[], Callable[[SqlReadSession], T]],
    *,
    datasets: Iterable[str] = (),
    sources: Mapping[str, Mapping[str, Any] | DataSource] | None = None,
    native_only: bool = True,
) -> T:
    """Bind native sources locally and run a fresh whole consumer on each retry."""
    if not native_only:
        raise ValueError(
            "consume_sql supports native_only=True; use project/reduce for typed inputs"
        )
    ctx._check_cancelled()
    names = tuple(dict.fromkeys(datasets))
    for name in names:
        ctx._require_available(name)
        if name in ctx._frames or name in ctx._lazy or name not in ctx._catalog.sources:
            raise NativeReadUnavailable(f"dataset {name!r} requires its typed/materialized reader")
    bindings_sources = {name: ctx._catalog.sources[name] for name in names}
    registry = ctx._source_registry
    for name, config in (sources or {}).items():
        if name.lower() in {item.lower() for item in ctx.declared()}:
            raise ValueError(f"ad-hoc source alias {name!r} collides with a declared dataset")
        if isinstance(config, DataSource):
            bindings_sources[name] = (
                config if config.env.connection_budget is ctx._resources else config.bind(ctx.env)
            )
        else:
            if registry is None:
                registry = SourceRegistry().discover()
            bindings_sources[name] = registry.build(config, ctx.env)
    check_sql_names(bindings_sources, "SQL source")
    observations: dict[str, tuple[dict[str, pl.DataType], float]] = {}
    fingerprints: dict[str, dict[str, Any] | None] = {}

    def attempt() -> T:
        plans = {}
        attachments: set[str] = set()
        bindings: dict[str, ConnectionSpec] = {}
        for name, source in bindings_sources.items():
            source.prepare_identity(hash_contents=ctx._hash_inputs or name not in names)
            if source.snapshot_connection() is not None or source.scan_polars() is not None:
                raise NativeReadUnavailable(f"source {name!r} requires its typed/snapshot reader")
            plan = source.scan()
            if plan is None:
                raise NativeReadUnavailable(f"source {name!r} has no native SQL reader")
            plans[name] = plan
            attachments.update(plan.connections)
            for attachment in plan.connections:
                merge_connection_binding(bindings, attachment, source.env)
        with ctx._sql_connections.temporary(attachments, bindings) as connection:
            for name, plan in plans.items():
                binding_started = time.perf_counter()
                relation = bind_scan(connection, bindings_sources[name], plan)
                budget = bindings_sources[name].params.row_budget
                if budget is not None:
                    row = relation.limit(budget + 1).aggregate("count(*)").fetchone()
                    if row is not None and row[0] > budget:
                        raise SourceError(
                            f"dataset {name!r} has more than {budget:,} rows (row_budget)"
                        )
                empty = relation.limit(0).pl()
                prepared = ctx._catalog.prepare(name, empty) if name in names else empty
                check_sql_columns(name, prepared.columns)
                if prepared.columns != empty.columns:
                    relation = relation.project(
                        ", ".join(
                            f"{quote_identifier(old)} AS {quote_identifier(new)}"
                            for old, new in zip(empty.columns, prepared.columns, strict=True)
                        )
                    )
                relation.create_view(name, replace=True)
                observations[name] = (dict(prepared.schema), time.perf_counter() - binding_started)
            session = SqlReadSession(ctx, connection, bindings_sources)
            try:
                result = consumer_factory()(session)
                session.close(require_exhausted=True)
                for name in names:
                    if not ctx._catalog.has_fingerprint(name):
                        fingerprints[name] = ctx._fingerprint(name, bindings_sources[name])
            finally:
                session.close()
            return result

    result = ctx._operations.run(bindings_sources.values(), attempt)
    with ctx._lock:
        ctx._catalog.mark_identity_used(names)
        for name in names:
            ctx._capabilities.record_native(name)
            ctx._catalog.record_native(name, *observations[name])
            ctx._catalog.record_native_plan(name)
            if name in fingerprints:
                ctx._catalog.record_fingerprint(name, fingerprints[name])
    return result
