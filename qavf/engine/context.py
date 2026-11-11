"""The read-only data context handed to every rule."""

from __future__ import annotations

import contextlib
import functools
import logging
import operator
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import duckdb
import polars as pl

from qavf.bounded_batches import BoundedBatches
from qavf.duckdb_results import arrow_table, bind_relation, first_row, query_frame
from qavf.engine.capabilities import ReaderCapabilities
from qavf.engine.catalog import DatasetCatalog
from qavf.engine.cleanup import attempt_cleanup, raise_cleanup_errors
from qavf.engine.connections import (
    SqlConnections,
    merge_connection_binding,
    merge_secret_bindings,
)
from qavf.engine.relations import bind_scan
from qavf.engine.resources import ResourceLimitError, RunResources, retained_size
from qavf.engine.session import Invocation, KeyedLocks
from qavf.engine.snapshot_staging import MemberPreparationError, SnapshotStage
from qavf.engine.sql import (
    BoundedSample,
    check_sql_columns,
    missing_projection,
    sample_result,
    sample_statement,
    sql_table_names,
)
from qavf.ordered_sample import OrderedSample
from qavf.redaction import exception_details, redact_text
from qavf.reports.models import DatasetInfo
from qavf.settings import (
    ConnectionSpec,
    DatasetSpec,
    QAVFSettings,
    SecretSpec,
    SqlAssertion,
    check_sql_names,
)
from qavf.sources.base import (
    DataSource,
    SourceEnvironment,
    SourceError,
    SourceIntegrityError,
    SourceNotFoundError,
    SourceRegistry,
)
from qavf.sources.collection import collect_result
from qavf.sources.errors import redacted_error
from qavf.sources.execution import SourceOperations
from qavf.sources.frames import as_polars
from qavf.sources.native import OwnedConnection, read_error
from qavf.sources.sql import (
    normalize_single_query,
    quote_identifier,
    referenced_connections,
    resolve_sql_paths,
)
from qavf.sources.typed import needs_typed_sql_binding

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

    from qavf.engine.observation import Observation, Observations
    from qavf.engine.observation_plan import ObservationPlan
    from qavf.engine.observation_store import ObservationStorage
    from qavf.engine.source_read import SourceReadServices, SourceReadView
    from qavf.engine.sql_read import NativeReadServices, SqlReadSession
    from qavf.engine.sqlite_store import Store


class DatasetNotAvailableError(KeyError):
    """Raised when a rule asks for a dataset the context cannot provide."""


class _Closeable(Protocol):
    def close(self) -> None: ...


class _LatestAttempt[T: _Closeable]:
    """Own one result per source attempt (an observation, a snapshot stage), closing
    every earlier attempt first."""

    def __init__(self, open_attempt: Callable[[], T]) -> None:
        self._open_attempt = open_attempt
        self._attempts: list[T] = []

    def __call__(self) -> T:
        """Discard earlier attempts and own a fresh one."""
        self.close()
        current = self._open_attempt()
        self._attempts.append(current)
        return current

    def close(self) -> None:
        """Release every attempt still owned."""
        for previous in self._attempts:
            previous.close()
        self._attempts.clear()


class _NativeStatement:
    """Consumer factory that runs one statement through a native session reader."""

    def __init__(self, reader: str, statement: str) -> None:
        self._reader = reader
        self._statement = statement

    def __call__(self) -> Callable[[SqlReadSession], Any]:
        """Return a fresh consumer for one native read attempt."""
        return operator.methodcaller(self._reader, self._statement)


def _transaction_usable(con: Any) -> bool:
    """Whether a snapshot transaction survived a failed member read."""
    try:
        con.execute("SELECT 1").fetchall()
    except duckdb.Error:
        return False
    return True


_PLANNING_ERRORS = (duckdb.BinderException, duckdb.CatalogException, duckdb.ParserException)


def _failed_before_execution(exc: BaseException) -> bool:
    """Whether a read failed while planning (binding, catalog lookup, parsing),
    so no member data was produced or consumed."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (*_PLANNING_ERRORS, SourceNotFoundError)):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class DataContext:
    """Named datasets, loaded lazily from their sources, plus run-wide settings.

    Datasets are held as Polars DataFrames. A cached pandas view is available
    through :meth:`pandas` for integrations that need one, and
    :meth:`sql` runs DuckDB SQL over any datasets by name. All accessors are
    thread-safe; rules must treat frames as immutable because they are shared
    between concurrently running rules.

    Args:
        frames: In-memory datasets keyed by name (Polars, pandas or Arrow).
        specs: Dataset specifications (schemas, foreign keys, expressions).
        sources: Data sources keyed by dataset name.
        env: Base directory and connections, for :meth:`sql` and path resolution.
        extensions: Plugin settings keyed by plugin name.
        assertions: SQL assertions for the assertion rule.
        source_registry: Used by :meth:`load_source`; built-ins only when omitted.
        canonical_columns: Column spellings to normalise to case-insensitively.
        hash_inputs: Record a SHA-256 digest of local input files in provenance.
        evidence_limit: How many offending records a rule needs to keep per result
            (``engine.max_affected_records``); rules count the rest without keeping them.
        concurrent_loads: Let different datasets load at the same time (each one is
            still loaded only once). When false, loads run one at a time.

    Raises:
        ValueError: for canonical spellings that differ only in case, or an in-memory
            frame whose columns cannot be normalised unambiguously.
    """

    def __init__(
        self,
        frames: Mapping[str, Any] | None = None,
        specs: Mapping[str, DatasetSpec] | None = None,
        sources: Mapping[str, DataSource] | None = None,
        env: SourceEnvironment | None = None,
        extensions: Mapping[str, Mapping[str, Any]] | None = None,
        canonical_columns: Iterable[str] = (),
        assertions: Iterable[SqlAssertion] = (),
        source_registry: SourceRegistry | None = None,
        hash_inputs: bool = False,
        evidence_limit: int = 50,
        evidence_bytes: int = 1048576,
        concurrent_loads: bool = True,
        cache_max_bytes: int = 268435456,
        spill_max_bytes: int = 1073741824,
        memory_budget_bytes: int | None = None,
        temp_directory: Path | None = None,
        reference_time: datetime | None = None,
        connection_slots: int = 2,
        max_spill_files: int = 4096,
    ) -> None:
        specifications = dict(specs or {})
        provided_names = frozenset(frames or {})
        self.env = env or SourceEnvironment()
        direct_sources = {
            name: value for name, value in (frames or {}).items() if isinstance(value, DataSource)
        }
        supplied_sources = {**dict(sources or {}), **direct_sources}
        if not self.env.text_functions:
            from qavf.text_policies import requires_sql_functions, text_type_requires_sql_functions

            needs_text_functions = requires_sql_functions(specifications) or any(
                text_type_requires_sql_functions(column)
                for source in supplied_sources.values()
                for column in getattr(source.params, "declared_columns", {}).values()
            )
            if needs_text_functions:
                self.env = replace(self.env, text_functions=True)
        frames = {
            name: value
            for name, value in (frames or {}).items()
            if not isinstance(value, DataSource)
        }
        self.reference_time = reference_time or datetime.now(UTC)
        self._resources = RunResources(
            cache_max_bytes,
            spill_max_bytes,
            memory_budget_bytes,
            temp_directory,
            connection_slots,
            max_spill_files,
        )
        with contextlib.ExitStack() as setup:
            setup.callback(self._resources.close)
            self.env = replace(
                self.env,
                temp_directory=self._resources.directory / "duckdb",
                connection_budget=self._resources,
            )
            bound_sources = {
                name: source.bind(self.env) for name, source in supplied_sources.items()
            }
            for name, source in bound_sources.items():
                spec = specifications.get(name)
                if spec is not None:
                    policy = spec.effective_load_policy().model_dump(exclude_unset=True)
                    source.params = type(source.params).model_validate(
                        {
                            **source.params.model_dump(by_alias=True),
                            **policy,
                        }
                    )
            self._lock = threading.RLock()
            self._catalog = DatasetCatalog(
                specifications, bound_sources, provided_names, canonical_columns, self._lock
            )
            self._lazy: dict[str, pl.LazyFrame] = {}
            self._typed_plans: dict[str, pl.LazyFrame] = {}
            self._typed_checked: set[str] = set()
            self._capabilities = ReaderCapabilities()
            self._budget_observations: dict[str, Observation] = {}
            self._closed = False
            self._released: set[str] = set()
            self._release_requested: set[str] = set()
            self._execution_plan: Any = None
            self._session = self._resources.session
            self._operations = SourceOperations(
                concurrent=concurrent_loads, check_cancelled=self._check_cancelled
            )
            self._sql_connections = SqlConnections(self.env, self._session)

            self._extensions = {k: dict(v) for k, v in (extensions or {}).items()}
            self.assertions: list[SqlAssertion] = list(assertions)
            self._source_registry = source_registry
            self._hash_inputs = hash_inputs
            self.evidence_limit = evidence_limit
            self.evidence_bytes = evidence_bytes
            self._frames = self._resources.frames
            self._pandas = self._resources.pandas
            self._memo = self._resources.memo
            # ``_lock`` publishes cache entries and catalog observations together;
            # per-key locks make each dataset load (and each memo computation) happen once
            # while unrelated keys proceed. SourceOperations owns backend serialization.
            self._key_locks = KeyedLocks()
            self._local = threading.local()
            check_sql_names(
                set(self.specs) | set(frames or {}) | set(self._catalog.sources), "dataset"
            )
            for name, frame in (frames or {}).items():
                if isinstance(frame, pl.LazyFrame):
                    self._lazy[name] = frame
                else:
                    prepared = self._catalog.prepare(name, as_polars(frame))
                    try:
                        self._check_row_budget(name, prepared)
                    except SourceError as exc:
                        self._catalog.record_failure(name, redacted_error(exc))
                        continue
                    self._frames[name] = prepared
                    self._catalog.record_frame(name, prepared)
            setup.pop_all()

    @classmethod
    def from_settings(
        cls,
        settings: QAVFSettings,
        frames: Mapping[str, Any] | None = None,
        source_registry: SourceRegistry | None = None,
        threads: int | None = None,
        *,
        execution_workers: int | None = None,
    ) -> DataContext:
        """Build a context from configuration.

        ``frames`` supplies or overrides datasets in memory, which is how an
        application validates its outputs before writing them anywhere.

        Raises:
            KeyError, ValueError: for an invalid source specification.
        """
        from qavf.execution.policy import check_policy
        from qavf.text_policies import requires_sql_functions

        check_policy(settings.engine)
        if settings.engine.execution_mode == "isolated":
            raise ValueError(
                "isolated execution requires validate(), QARunner.run(), or compare_outputs()"
            )
        registry = source_registry or SourceRegistry().discover(
            modules=tuple(settings.plugins.modules),
            use_entry_points=settings.plugins.entry_points,
        )
        env = SourceEnvironment(
            base_dir=settings.base_dir,
            connections=settings.connections,
            secrets=settings.secrets,
            threads=threads,
            text_functions=requires_sql_functions(settings),
            memory_limit=settings.engine.duckdb_memory_limit,
            temp_directory=(
                settings.resolve(settings.engine.temp_directory)
                if settings.engine.temp_directory is not None
                else None
            ),
            spill_limit_bytes=settings.engine.spill_max_bytes
            // (2 * max(1, settings.engine.max_workers or 8)),
        )
        provided = set(frames or {})
        workers = execution_workers or (
            max(1, settings.engine.max_workers or min(8, os.cpu_count() or 1))
            if settings.engine.parallel
            else 1
        )
        from qavf.engine.thread_budget import memory_workers

        workers = memory_workers(workers, settings.engine.memory_budget_bytes)
        from qavf.resource_plan import ResourcePlan

        resource_plan = ResourcePlan.for_context(settings, frames, requested_workers=workers)
        workers = resource_plan.workers
        if threads is None:
            from qavf.engine.thread_budget import connection_threads

            per_connection = (
                resource_plan.memory_bytes // 2 // resource_plan.connection_slots
                if resource_plan.memory_bytes is not None
                else None
            )
            env = replace(
                env,
                threads=connection_threads(
                    settings.engine.native_threads, workers, memory_bytes=per_connection
                ),
            )
        context = cls(
            frames=frames,
            specs=settings.datasets,
            env=env,
            extensions=settings.extensions,
            canonical_columns=settings.engine.resolved_canonical_columns(),
            assertions=settings.assertions,
            source_registry=registry,
            hash_inputs=settings.report.hash_inputs,
            evidence_limit=settings.engine.max_affected_records,
            evidence_bytes=settings.engine.max_evidence_bytes,
            concurrent_loads=settings.engine.concurrent_loads,
            cache_max_bytes=settings.engine.cache_max_bytes,
            spill_max_bytes=resource_plan.spill_bytes,
            max_spill_files=settings.engine.max_spill_files,
            memory_budget_bytes=resource_plan.memory_bytes,
            temp_directory=(
                settings.resolve(settings.engine.temp_directory)
                if settings.engine.temp_directory is not None
                else None
            ),
            connection_slots=2 * workers,
        )
        context._resources.plan = {
            **resource_plan.diagnostics(),
            "native_threads_per_connection": env.threads,
        }

        try:
            for name, spec in settings.datasets.items():
                config = spec.source_config()
                if config is not None and name not in provided:
                    try:
                        context._catalog.sources[name] = registry.build(config, context.env)
                    except (KeyError, ValueError) as exc:
                        message = exc.args[0] if exc.args else exc
                        raise ValueError(f"dataset {name!r}: {message}") from None
            context._attach_views(settings)
            return context
        except BaseException:
            context.close()
            raise

    def _attach_views(self, settings: QAVFSettings) -> None:
        """Give every dataset view a source that slices its parent as this run loads it,
        and every derived dataset (consensus, pooled members) one that builds it."""
        from qavf.engine.derived import DerivedSource
        from qavf.engine.views import ViewSource

        for name, (parent, spec) in settings.views().items():
            self._attach_view(name, ViewSource.of(parent, spec, self.env, self))
        for name, (parent, spec, _) in settings.where_views().items():
            # The rows a filtered check item's ``where`` selects; read only by its rule.
            self._attach_view(name, ViewSource.of(parent, spec, self.env, self))
        for name, params in settings.derived_datasets().items():
            self._attach_view(name, DerivedSource.of(params, self.env, self))

    def _attach_view(self, name: str, source: DataSource) -> None:
        """Register a view or derived dataset; it never replaces another source."""
        if name in self._catalog.sources:
            raise ValueError(f"view {name!r} would replace the dataset of that name")
        self._catalog.sources[name] = source
        self._catalog.views.add(name)

    @property
    def observation_storage(self) -> ObservationStorage:
        """Narrow observation admission/writing capability owned by this run."""
        return self._resources.observations

    @property
    def specs(self) -> dict[str, DatasetSpec]:
        """Dataset declarations owned by this context's catalog."""
        return self._catalog.specs

    def contract_spec(self, name: str) -> DatasetSpec | None:
        """The declaration whose contract ``name``'s values follow: its own, or for a
        view (``roads.active``) its parent dataset's, as views keep the parent's contract."""
        spec = self._catalog.specs.get(name)
        if spec is not None or name not in self._catalog.views:
            return spec
        from qavf.engine.views import ViewSource

        source = self._catalog.sources.get(name)
        if not isinstance(source, ViewSource):
            return None
        return self._catalog.specs.get(source.params.dataset)

    # ------------------------------------------------------------------ access
    def source(self, name: str) -> DataSource | None:
        """The configured source of ``name``; ``None`` for in-memory inputs."""
        return self._catalog.sources.get(name)

    def names(self) -> list[str]:
        """Every dataset that has data or a source, loaded or not."""
        return self._catalog.names(self._frames, self._lazy)

    def declared(self) -> list[str]:
        """Every dataset named in the configuration or supplied in memory."""
        return self._catalog.declared(self.names())

    def has(self, name: str) -> bool:
        """Whether the dataset is accessible, using schema-only native binding when possible."""
        if self._catalog.failure(name) is not None:
            return False
        try:
            self._check_cancelled()
            if name in self._released:
                return False
            if name in self._frames:
                return True
            if self.execution_kind(name) in ("lazy", "native_sql"):
                self.schema(name)
            else:
                self.get(name)
            return True
        except Exception as exc:
            self._check_cancelled()
            with self._lock:
                self._catalog.record_failure(
                    name,
                    redacted_error(
                        exc,
                        SourceNotFoundError
                        if isinstance(exc, SourceNotFoundError)
                        else SourceError,
                    ),
                    replace=False,
                )
            return False

    def unavailable_reason(self, name: str) -> str | None:
        """Why ``name`` could not be loaded, or ``None`` if it is available."""
        if self.has(name):
            return None
        failure = self._catalog.failure(name)
        if failure:
            return str(failure)
        if name in self.specs:
            return (
                f"dataset {name!r} is declared but has no source, and no in-memory frame "
                f"was supplied"
            )
        return f"dataset {name!r} is not configured"

    def load_error(self, name: str) -> SourceError | None:
        """The loading error for ``name`` (after an attempt), if any."""
        self.has(name)
        return self._catalog.failure(name)

    def get(self, name: str) -> pl.DataFrame:
        """Return dataset ``name`` as a Polars DataFrame, loading it if needed.

        Raises:
            DatasetNotAvailableError: if the dataset is not configured or failed to load.
        """
        self._check_cancelled()
        if name in self._released:
            raise DatasetNotAvailableError(
                f"dataset {name!r} was released after its last planned consumer"
            )
        cached = self._cached(name)
        if cached is not None:
            return cached
        # Always acquire the operation guard before a load key: a native query
        # may reenter get() for an eager participant while holding that guard.
        with self._operations.guard(), self._key_lock(("load", name)):
            # Another thread may have loaded (or failed to load) it meanwhile.
            cached = self._cached(name)
            if cached is not None:
                return cached
            if self._is_lazy(name):
                self._resources.eager_fallbacks.add(name)
                self._check_lazy_budget(name)
                frame = self.collect(name, self._lazy_snapshot(name))
                if name not in self._budget_observations:
                    frame = self._catalog.prepare(name, frame)
                self._check_row_budget(name, frame)
                self._check_cancelled()
                self._frames.publish(
                    {name: frame},
                    functools.partial(self._publish_lazy_metadata, name, frame),
                    self._lock,
                )
                return frame.clone()
            source = self._catalog.sources.get(name)
            if source is None:
                raise DatasetNotAvailableError(f"dataset {name!r} is not configured")
            if (connection := source.snapshot_connection()) is not None:
                self._load_snapshot(connection)
                cached = self._cached(name)
                assert cached is not None  # the snapshot recorded a frame or a failure
                return cached
            started = time.perf_counter()
            failure: SourceError | None = None
            try:
                frame = self._catalog.prepare(name, self._load_with_retries(name, source))
            except Exception as exc:
                self._check_cancelled()
                # Every loading problem, expected or not, becomes a recorded failure of
                # this dataset: the run carries on and INP-001 reports it.
                # The failure is recorded, logged and reported, so it is stored as a new,
                # unchained error with credentials masked (a chained original could still
                # carry them into a traceback).
                if isinstance(exc, SourceError):
                    try:
                        failure = redacted_error(exc, type(exc))
                    except TypeError:  # a plugin error class with another signature
                        failure = redacted_error(exc)
                else:
                    failure = redacted_error(
                        SourceError(f"loading failed: {type(exc).__name__}: {exc}")
                    )
                with self._lock:
                    self._catalog.record_failure(name, failure)
                level = (
                    logging.INFO if isinstance(failure, SourceNotFoundError) else logging.WARNING
                )
                logger.log(level, "Dataset %r unavailable: %s", name, failure)
            if failure is not None:
                # Raised outside the handler, so the original error is not even attached
                # as context (it may hold credentials).
                raise DatasetNotAvailableError(str(failure))
            self._add_thread_load_time(time.perf_counter() - started)
            notes = dict(getattr(source, "load_notes", {}) or {})
            self._check_cancelled()
            self._frames.publish(
                {name: frame},
                functools.partial(self._publish_load_metadata, name, frame, notes, started),
                self._lock,
            )
            logger.info(
                "Loaded dataset %r: %d rows from %s",
                name,
                frame.height,
                redact_text(source.describe()),
            )
            return frame.clone()

    def _publish_lazy_metadata(self, name: str, frame: pl.DataFrame) -> None:
        """Record a collected lazy dataset's frame metadata at publication."""
        self._check_cancelled()
        self._require_available(name)
        self._catalog.record_frame(name, frame)

    def _publish_load_metadata(
        self, name: str, frame: pl.DataFrame, notes: dict[str, Any], started: float
    ) -> None:
        """Record an eager load's frame metadata, notes and duration at publication."""
        self._check_cancelled()
        self._require_available(name)
        self._catalog.record_frame(name, frame)
        self._catalog.record_load(name, notes, time.perf_counter() - started)

    def pandas(self, name: str) -> pd.DataFrame:
        """Return a detached pandas view; cached state cannot be mutated through the result."""
        with self._operations.guard(), self._key_lock(("pandas", name)):
            with self._lock:
                if name in self._pandas:
                    return self._pandas[name]
            converted = self.get(name).to_pandas()
            with self._lock:
                self._pandas[name] = converted
            return converted

    def execution_kind(self, name: str) -> str:
        """Return materialized, lazy, native_sql, or eager for dispatch decisions."""
        if name in self._frames:
            return "materialized"
        if self._is_lazy(name):
            return "lazy"
        source = self._catalog.sources.get(name)
        if (
            source is not None
            and source.snapshot_connection() is None
            and self._capabilities.native(
                name, lambda: self._source_operation([name], lambda: source.scan() is not None)
            )
        ):
            return "native_sql"
        return "eager"

    def _require_available(self, name: str) -> None:
        if name in self._released:
            raise DatasetNotAvailableError(f"dataset {name!r} has been released")
        if name not in self.declared():
            raise DatasetNotAvailableError(f"dataset {name!r} is not declared")

    def repeatable_scan(self, name: str) -> bool:
        """Whether separate reduction passes have a known repeatable file reader.

        Only the built-in local FileSource qualifies, after establishing its pinned
        identity. Supplied lazy plans, queries, remote files and plugins may change
        on evaluation even when their schemas are unchanged. This capability does
        not promise row ordering; each subsequent read must still use this context's
        verified accessors. Owned snapshots use project/reduce directly instead.
        """
        from qavf.engine.projection import repeatable_scan

        return repeatable_scan(self, name)

    def project(self, name: str, columns: Iterable[str]) -> pl.DataFrame:
        """Read selected raw columns with source policy and exact logical types.

        Lazy/native sources prune columns before collection; owned IPC spills are
        read without full deserialization or cache promotion. Opaque eager and
        transaction snapshot sources retain their complete-load semantics.
        """
        from qavf.engine.projection import project

        return project(self, name, columns)

    def reduce(self, name: str, build: Callable[[pl.LazyFrame], pl.LazyFrame]) -> pl.DataFrame:
        """Collect a single-source lazy reduction under its storage lifetime.

        ``build`` returns a plan and must not retain it. Owned spill files remain
        protected until collection ends; source policies cover complete inputs.
        SQL-native callers should use SQL reductions to retain native pushdown.
        """
        from qavf.engine.projection import reduce_frame

        return reduce_frame(self, name, build)

    def is_materialized(self, name: str) -> bool:
        return name in self._frames

    def observe(
        self, name: str, columns: Iterable[str] | None = None
    ) -> contextlib.AbstractContextManager[Observation]:
        """Own one coherent projection; batch/reduce it without rereading the source."""
        from qavf.engine.observation import observe

        return observe(self, name, columns)

    def projected_observations(
        self, selections: Mapping[str, Iterable[str] | None]
    ) -> contextlib.AbstractContextManager[Observations]:
        """Own stable projections for a complete multi-input operation."""
        from qavf.engine.observation import projected_observations

        return projected_observations(self, selections)

    def observation_plan(
        self, contracts: Sequence[Mapping[str, Iterable[str] | None]]
    ) -> ObservationPlan:
        """Reuse validated projections until each input's final scheduled use."""
        from qavf.engine.observation_plan import ObservationPlan

        return ObservationPlan(self, contracts)

    def work_store(self, label: str) -> Store:
        """Create a bounded owned index using the run's shared spill reservations."""
        from qavf.engine.sqlite_store import Store

        return Store(self._resources, label)

    def spill_file(self, path: Path):
        """Own a caller-selected staging file under this run's shared disk quota.

        Close the returned owner (preferably with a ``with`` block). The caller
        owns publication of the completed payload, e.g. an atomic baseline move.
        """
        from qavf.engine.spill import SpillFile

        self._check_cancelled()
        return SpillFile(Path(path), self._resources.spill_budget)

    def check_materialization(self, name: str, frame: pl.DataFrame) -> None:
        """Apply the configured target to a completed typed result."""
        self._check_materialization(name, frame)

    @property
    def memory_budget_bytes(self) -> int | None:
        """The run's memory target for materialized results (``None``: none set)."""
        return self._resources.memory_bytes

    def check_retained_bytes(self, name: str, size: int) -> None:
        """Admit measured aggregate state without exposing the resource owner."""
        self._check_retained_bytes(name, size)

    def schema(self, name: str) -> dict[str, pl.DataType]:
        """Inspect a dataset's schema without materializing native scan sources."""
        self._check_cancelled()
        if name in self._released:
            raise DatasetNotAvailableError(f"dataset {name!r} has been released")
        known = self._catalog.schema(name)
        if known is not None:
            return known
        source = self._catalog.sources.get(name)
        if source is not None and (schema := source.batch_schema()) is not None:
            # Declared batch metadata does not open a factory or consume data.
            return dict(self._catalog.prepare(name, pl.DataFrame(schema=schema)).schema)
        if self._is_lazy(name):
            self._check_lazy_budget(name)
            schema = self._source_operation(
                [name],
                lambda: dict(self._prepare_lazy(name, self._lazy_snapshot(name)).collect_schema()),
            )
            self._remember_lazy_dataset(name, schema)
            return schema
        frame = self.sql(f'SELECT * FROM "{name.replace(chr(34), chr(34) * 2)}" LIMIT 0')
        return dict(frame.schema)

    def scan(self, name: str) -> pl.LazyFrame:
        """Return a Polars lazy plan; supplied LazyFrames preserve pushdown.

        SQL-native sources use :meth:`sql` for pushdown. This Polars accessor
        records its explicit materializing fallback for other source types.
        """
        self._check_cancelled()
        if self._is_lazy(name):
            self._check_lazy_budget(name)
            return self._source_operation(
                [name], lambda: self._prepare_lazy(name, self._lazy_snapshot(name))
            )
        self._resources.eager_fallbacks.add(name)
        return self.get(name).lazy()

    def _lazy_snapshot(self, name: str) -> pl.LazyFrame:
        # Older supported Polars versions reject concurrent borrows of one Python
        # LazyFrame wrapper. Each caller gets its own wrapper over the shared plan.
        source = self._catalog.sources.get(name)
        if source is not None and source.batch_schema() is not None:
            self._batch_observation(name, source)
        with self._lock:
            if name in self._budget_observations:
                return self._budget_observations[name].scan()
            plan = self._lazy.get(name)
            if plan is None:
                plan = self._typed_plans[name]
            return plan.clone()

    def _is_lazy(self, name: str) -> bool:
        if name in self._lazy or name in self._typed_plans:
            return True
        source = self._catalog.sources.get(name)
        if source is None or source.snapshot_connection() is not None:
            return False
        if source.batch_schema() is not None:
            return True
        with self._operations.guard(), self._key_lock(("typed_plan", name)):
            if name in self._typed_checked:
                return name in self._typed_plans
            if name not in self._typed_plans:
                plan = self._source_operation([name], source.scan_polars)
                self._typed_checked.add(name)
                if plan is None:
                    return False
                with self._lock:
                    self._typed_plans[name] = plan
        return True

    def _batch_observation(self, name: str, source: DataSource) -> Observation:
        """Publish one complete owned stream for every consumer in this run."""
        from qavf.engine.batch_input import observe_batches

        with self._operations.guard(), self._key_lock(("batch_snapshot", name)):
            if name in self._budget_observations:
                return self._budget_observations[name]
            schema = source.batch_schema()
            assert schema is not None
            empty = pl.DataFrame(schema=schema)
            prepared = self._catalog.prepare(name, empty)
            rename = dict(zip(empty.columns, prepared.columns, strict=True))
            admit = _LatestAttempt(functools.partial(observe_batches, self, source, rename))
            try:
                observed = self._source_operation([name], admit)
            except BaseException:
                admit.close()
                raise
            try:
                self._check_cancelled()
                with self._lock:
                    self._budget_observations[name] = observed
                    self._catalog.record_lazy(name, prepared.schema)
                return observed
            except BaseException:
                observed.close()
                raise

    def collect(self, name: str, plan: pl.LazyFrame) -> pl.DataFrame:
        """Collect a projected plan under the dataset's complete source policy.

        Use this with :meth:`scan` so typed source execution verifies identity
        after reading, honors serialized access and never bypasses row budgets.
        """
        return self.collect_many([name], plan)

    def collect_many(self, names: Iterable[str], plan: pl.LazyFrame) -> pl.DataFrame:
        """Collect one combined plan under every participating source's policy.

        Joins and key reductions may use several datasets. Verify all identities
        around the same evaluation and enforce each input's complete row budget.
        """
        self._check_cancelled()
        participants = tuple(dict.fromkeys(names))
        for name in participants:
            if name in self._released:
                raise DatasetNotAvailableError(f"dataset {name!r} has been released")
            self._check_lazy_budget(name)

        def collect() -> pl.DataFrame:
            frame = plan.collect(engine="streaming")
            self._check_materialization(", ".join(participants) or "lazy result", frame)
            return frame

        result = self._source_operation(participants, collect)
        for name in participants:
            if name in self._typed_plans or name in self._lazy:
                self._remember_lazy_dataset(name)
        return result

    def _source_operation[T](self, names: Iterable[str], operation: Callable[[], T]) -> T:
        participants = {
            name: self._catalog.sources[name] for name in names if name in self._catalog.sources
        }
        fingerprints: dict[str, dict[str, Any] | None] = {}

        def verified() -> T:
            # Establish content identity before the first read and share it with
            # provenance. The enclosing scope verifies it again before return.
            for source in participants.values():
                source.prepare_identity(hash_contents=self._hash_inputs)
            value = operation()
            for name, source in participants.items():
                if value is not None and not self._catalog.has_fingerprint(name):
                    fingerprints[name] = self._fingerprint(name, source)
            return value

        result = self._operations.run(participants.values(), verified)
        with self._lock:
            self._catalog.mark_identity_used(participants)
            for name, fingerprint in fingerprints.items():
                self._catalog.record_fingerprint(name, fingerprint)
        return result

    def _remember_lazy_dataset(
        self, name: str, schema: Mapping[str, pl.DataType] | None = None
    ) -> None:
        source = self._catalog.sources.get(name)
        if schema is None:
            schema = self._catalog.schema(name)
            if schema is None:
                schema = self._prepare_lazy(name, self._lazy_snapshot(name)).collect_schema()
        with self._lock:
            self._catalog.record_lazy(name, schema)
            if source is not None and not self._catalog.has_fingerprint(name):
                self._catalog.record_fingerprint(name, self._fingerprint(name, source))

    def _prepare_lazy(self, name: str, frame: pl.LazyFrame) -> pl.LazyFrame:
        if name in self._budget_observations:
            return frame  # Admission already normalized names exactly once.
        empty = pl.DataFrame(schema=frame.collect_schema())
        prepared = self._catalog.prepare(name, empty)
        return frame.rename(dict(zip(empty.columns, prepared.columns, strict=True)))

    def sql_missing_projection(self, name: str, schema: Mapping[str, pl.DataType]) -> str:
        """Apply exactly the typed row predicates' missing-marker semantics."""
        dataset = self.specs.get(name)
        columns = dataset.schema_spec.columns if dataset and dataset.schema_spec else {}
        return missing_projection(schema, columns)

    def _sql_connection(
        self,
        query: str,
        datasets: Iterable[str] | None = None,
        *,
        normalize_missing: bool = False,
    ):
        self._check_cancelled()
        names = list(datasets) if datasets is not None else self.referenced(query)
        plans = {}
        attachments = set(referenced_connections(query, self.env.connections))
        bindings: dict[str, ConnectionSpec] = {}
        secrets: dict[str, SecretSpec] = {}
        if attachments:
            merge_secret_bindings(secrets, self.env)
        for connection in attachments:
            merge_connection_binding(bindings, connection, self.env)
        for name in names:
            source = self._catalog.sources.get(name)
            if (
                source is not None
                and name not in self._frames
                and source.snapshot_connection() is None
            ):
                source.prepare_identity(hash_contents=self._hash_inputs)
                plan = source.scan()
                if plan is not None:
                    plans[name] = plan
                    merge_secret_bindings(secrets, source.env)
                    attachments.update(plan.connections)
                    for connection in plan.connections:
                        merge_connection_binding(bindings, connection, source.env)
        if not plans:
            merge_secret_bindings(secrets, self.env)
        con = self._connection(attachments, bindings, secrets)
        self._sql_connections.discard(self._released)
        for name in names:
            if name in self._released:
                raise DatasetNotAvailableError(
                    f"dataset {name!r} was released after its last planned consumer"
                )
            if name in plans:
                self._capabilities.record_native(name)
                binding_started = time.perf_counter()
                plan = plans[name]
                relation = bind_scan(con, self._catalog.sources[name], plan)
                budget = getattr(self._catalog.sources[name].params, "row_budget", None)
                if budget is not None:
                    count_row = first_row(relation.limit(budget + 1).aggregate("count(*)"))
                    if count_row is not None and count_row[0] > budget:
                        raise SourceError(
                            f"dataset {name!r} has more than {budget:,} rows (row_budget)"
                        )
                empty = relation.limit(0).pl()
                prepared = self._catalog.prepare(name, empty)
                check_sql_columns(name, prepared.columns)
                if empty.columns != prepared.columns:
                    projection = ", ".join(
                        '"' + old.replace('"', '""') + '" AS "' + new.replace('"', '""') + '"'
                        for old, new in zip(empty.columns, prepared.columns, strict=True)
                    )
                    relation = relation.project(projection)
                if normalize_missing and prepared.width:
                    relation = relation.project(self.sql_missing_projection(name, prepared.schema))
                bind_relation(con, relation, name)
                self._sql_connections.remember(name)
                with self._lock:
                    self._catalog.record_native(
                        name, prepared.schema, time.perf_counter() - binding_started
                    )
                    if not self._catalog.has_fingerprint(name):
                        self._catalog.record_fingerprint(
                            name, self._fingerprint(name, self._catalog.sources[name])
                        )
                    self._catalog.record_native_plan(name)
            else:
                if name not in self._frames:
                    self._resources.eager_fallbacks.add(name)
                frame = self.get(name)
                check_sql_columns(name, frame.columns)
                if normalize_missing:
                    from qavf.missingness import normalize_frame

                    spec = self.specs.get(name)
                    columns = (
                        spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
                    )
                    frame = normalize_frame(frame, columns)
                unsupported = [
                    column
                    for column, dtype in frame.schema.items()
                    if needs_typed_sql_binding(dtype)
                ]
                if unsupported:
                    raise SourceError(
                        f"dataset {name!r} has logical types that this SQL binding cannot "
                        f"preserve: {', '.join(unsupported)}. Use scan()/collect() for typed "
                        "checks, or explicitly project supported columns before SQL validation."
                    )
                self._sql_connections.register_frame(con, name, frame)
        return con

    def release(self, names: Iterable[str]) -> None:
        """Release exact cached data after the planner's last possible consumer.

        Snapshot members release as a group. Metadata survives. SQL registrations
        are removed by their owning worker before its next query or final close.
        """
        with self._lock:
            self._release_requested.update(names)
            eligible = set(self._release_requested)
            for name in tuple(eligible):
                source = self._catalog.sources.get(name)
                connection = source.snapshot_connection() if source is not None else None
                if connection is not None:
                    group = {
                        key
                        for key, item in self._catalog.sources.items()
                        if item.snapshot_connection() == connection
                    }
                    if not group <= self._release_requested:
                        eligible -= group
            observations = {
                name: self._budget_observations[name]
                for name in eligible
                if name in self._budget_observations
            }
            for name in eligible:
                self._lazy.pop(name, None)
                self._typed_plans.pop(name, None)
            # Stop admission under the catalog lock, then release ownership
            # outside it. Cache writers commit under this same lock and must
            # never be awaited while holding it.
            self._released.update(eligible)
            self._release_requested -= eligible
        failures = []
        for name in eligible:
            failures.extend(
                attempt_cleanup(
                    [
                        functools.partial(self._frames.discard, name),
                        functools.partial(self._pandas.discard, name),
                    ]
                )
            )
        for name, observed in observations.items():
            try:
                observed.close()
            except Exception as exc:
                failures.append(exc)
            else:
                with self._lock:
                    if self._budget_observations.get(name) is observed:
                        self._budget_observations.pop(name)
        if eligible:
            failures.extend(attempt_cleanup([self._memo.clear]))
        raise_cleanup_errors("released datasets could not be fully cleaned", failures)

    def _native_participants(self, query: str, datasets: Iterable[str] | None = None):
        names = list(datasets) if datasets is not None else self.referenced(query)
        return [
            source
            for name in names
            if (source := self._catalog.sources.get(name)) is not None
            and name not in self._frames
            and source.snapshot_connection() is None
        ]

    def sql(
        self,
        query: str,
        datasets: Iterable[str] | None = None,
        *,
        normalize_missing: bool = False,
    ) -> pl.DataFrame:
        """Execute binding and query as one verified, retryable operation."""
        names = list(datasets) if datasets is not None else None
        return self._operations.run(
            self._native_participants(query, names),
            lambda: self._sql_once(query, names, normalize_missing=normalize_missing),
        )

    def consume_sql[T](
        self,
        consumer_factory: Callable[[], Callable[[SqlReadSession], T]],
        *,
        datasets: Iterable[str] = (),
        sources: Mapping[str, Mapping[str, Any]] | None = None,
        native_only: bool = True,
    ) -> T:
        """Consume native SQL batches inside one verified, resettable operation.

        A fresh callback is created for every retry. It receives a scoped session
        supporting schema(), frame(), batches() and fingerprint(); no reader may
        escape. Unsupported typed/eager/snapshot inputs raise NativeReadUnavailable
        before eager materialization. Ad-hoc sources share this run's resources.
        """
        from qavf.engine.sql_read import consume_native

        return consume_native(
            self, consumer_factory, datasets=datasets, sources=sources, native_only=native_only
        )

    def sql_sample(
        self, query: str, limit: int, *, ordered: bool = False, normalize_missing: bool = False
    ) -> tuple[pl.DataFrame, int]:
        """Count and sample one evaluation under the same source identity/retry policy."""
        return self._operations.run(
            self._native_participants(query),
            lambda: self._sql_sample_once(
                query, limit, ordered=ordered, normalize_missing=normalize_missing
            ),
        )

    def _sql_once(
        self,
        query: str,
        datasets: Iterable[str] | None = None,
        *,
        normalize_missing: bool = False,
    ) -> pl.DataFrame:
        """Execute SQL, retaining native source scans until result collection."""
        with self._sql_lease(query, datasets, normalize_missing=normalize_missing) as con:
            result = query_frame(
                con, resolve_sql_paths(query, self.env.base_dir, known_tables=self.declared())
            )
            if result is None:
                raise ValueError("the SQL statement returned no result set")
        self._check_materialization("SQL result", result)
        self._check_cancelled()
        return result

    def _sql_sample_once(
        self, query: str, limit: int, *, ordered: bool = False, normalize_missing: bool = False
    ) -> tuple[pl.DataFrame, int]:
        """Evaluate once; count all rows and collect only a bounded evidence sample.

        An unordered sample streams the result once (:class:`BoundedSample`); an
        ordered one keeps a bounded top-N (:class:`~qavf.ordered_sample.OrderedSample`),
        or, for rows it cannot order, stages the result in a materialized CTE owned
        by DuckDB, subject to its memory/spill budget. Non-query statements are
        rejected, never executed.
        """
        resolved = resolve_sql_paths(query, self.env.base_dir, known_tables=self.declared())
        with self._sql_lease(query, normalize_missing=normalize_missing) as con:
            if not ordered:
                sample, count = BoundedSample(limit).read(con, resolved, self._check_cancelled)
            else:
                # A bounded top-N over whole rows; rows it cannot order are staged.
                body = normalize_single_query(resolved)
                statement = OrderedSample(limit).bound(con, body)
                if statement is None:
                    statement = sample_statement(resolved, limit, ordered=True)
                sample, count = sample_result(arrow_table(con.sql(statement)))
        self._check_cancelled()
        self._check_materialization("SQL evidence sample", sample)
        return sample, count

    @contextlib.contextmanager
    def _sql_lease(
        self,
        query: str,
        datasets: Iterable[str] | None = None,
        *,
        normalize_missing: bool = False,
    ):
        """A query owns its registrations; the reusable connection owns no input frames."""
        with self._sql_connections.operation():
            yield self._sql_connection(query, datasets, normalize_missing=normalize_missing)

    def _check_materialization(self, name: str, frame: pl.DataFrame) -> None:
        if self._resources.memory_bytes is None:
            return
        self._check_retained_bytes(name, retained_size(frame))

    def _check_retained_bytes(self, name: str, size: int) -> None:
        """Admit a measured retained result, including incrementally built output."""
        target = self._resources.memory_bytes
        if target is None:
            return
        if size > target:
            raise ResourceLimitError(
                f"{name} materialized {size:,} bytes, exceeding memory_budget_bytes={target:,}; "
                "use a projected scan or increase the target"
            )

    @property
    def _active(self) -> set[int]:
        with self._session.lock:
            return set(self._session.active)

    def begin_rule(self, invocation: Invocation | None = None) -> Invocation:
        return self._session.begin(invocation)

    def end_rule(self) -> None:
        if self._session.end():
            self._close_owned()

    def cancel_rule(self, invocation: Invocation) -> None:
        self._session.cancel(invocation)

    def cancel_active_rules(self) -> None:
        """Interrupt every rule running now; used when the caller cancels the run."""
        self._session.cancel_active()

    def _check_cancelled(self) -> None:
        self._session.check()

    def close(self) -> None:
        """Close after actual worker completion, without waiting for arbitrary work."""
        self._closed = True
        if self._session.request_close():
            self._close_owned()

    def finish_worker(self, owner: int | None = None) -> None:
        """Release connections of a worker that has actually stopped executing."""
        self._sql_connections.finish_worker(owner)

    def finish_inactive_workers(self) -> None:
        """Clean up executor threads after shutdown; retain this caller's SQL session."""
        with self._session.lock:
            owners = {owner for owner, _ in self._session.connections.values()}
            owners -= set(self._session.active)
        owners.discard(threading.get_ident())
        for owner in owners:
            self.finish_worker(owner)

    def _close_owned(self) -> None:
        with self._lock:
            self._usage_at_close = self._resources.usage()
            observations = tuple(self._budget_observations.items())
            self._lazy.clear()
            self._typed_plans.clear()
        failures = []
        for name, observed in observations:
            try:
                observed.close()
            except Exception as exc:
                failures.append(exc)
            else:
                with self._lock:
                    if self._budget_observations.get(name) is observed:
                        self._budget_observations.pop(name)
        failures.extend(attempt_cleanup([self._sql_connections.close, self._resources.close]))
        raise_cleanup_errors("context could not be fully released", failures)

    def resource_usage(self) -> dict[str, Any]:
        snapshot = getattr(self, "_usage_at_close", None)
        usage = dict(
            self._resources.usage(blocking=not self._active) if snapshot is None else snapshot
        )
        usage["active_workers_at_report"] = len(self._active)
        usage["polars_process_threads"] = pl.thread_pool_size()
        usage["duckdb_threads_per_connection"] = self.env.threads
        usage.update(self._sql_connections.usage())
        return usage

    def reporting_snapshot(self) -> dict[str, Any]:
        """Describe completed metadata without joining an overdue admission.

        The runner records a snapshot before scheduling. If an active invocation
        owns the catalog lock during cache admission, keep that prior complete
        snapshot and explicitly mark metadata/identity as pending.
        """
        active = bool(self._active)
        catalog_acquired = self._lock.acquire(blocking=not active)
        frames_acquired = catalog_acquired and self._frames.lock.acquire(blocking=not active)
        acquired = catalog_acquired and frames_acquired
        snapshot: dict[str, Any]
        try:
            if acquired:
                snapshot = {
                    "datasets": self.describe(),
                    "unused_datasets": self.unused(),
                    "unavailable_datasets": self.unavailable(),
                    "query_plans": self.query_plans(),
                }
                self._reporting_snapshot = snapshot
            else:
                snapshot = dict(
                    getattr(
                        self,
                        "_reporting_snapshot",
                        {
                            "datasets": [],
                            "unused_datasets": [],
                            "unavailable_datasets": {},
                            "query_plans": [],
                        },
                    )
                )
                snapshot["datasets"] = [
                    info.model_copy(
                        update={
                            "notes": {
                                **info.notes,
                                "identity_verified": False,
                                "identity_verification": "pending: timed-out work remains active",
                            }
                        }
                    )
                    for info in snapshot["datasets"]
                ]
        finally:
            if frames_acquired:
                self._frames.lock.release()
            if catalog_acquired:
                self._lock.release()
        return {
            **snapshot,
            "resource_usage": {
                **self.resource_usage(),
                "metadata_snapshot_pending": not acquired,
            },
        }

    def query_plans(self) -> list[dict[str, Any]]:
        return self._catalog.query_plans()

    def __enter__(self):
        self._check_cancelled()
        return self

    def __exit__(self, *exc):
        self.close()

    def referenced(self, query: str) -> list[str]:
        """Datasets ``query`` reads as tables, matched case-insensitively like SQL.

        Table references come from DuckDB's parser, so names that only occur in
        string literals, comments or as column names are not dependencies.
        """
        lookup = {name.lower(): name for name in self.declared()}
        return sorted(
            {lookup[table.lower()] for table in sql_table_names(query) if table.lower() in lookup}
        )

    def build_source(self, config: Mapping[str, Any]) -> DataSource:
        """Bind an ad-hoc description to this run's registry and environment."""
        return (self._source_registry or SourceRegistry().discover()).build(config, self.env)

    def run_source_operation[T](
        self, sources: Iterable[DataSource], operation: Callable[[], T]
    ) -> T:
        return self._operations.run(sources, operation)

    def native_read_services(self) -> NativeReadServices:
        """Wire native reads without exposing context internals to session code."""
        from qavf.engine.sql_read import NativeReadServices, SqlReadPolicy

        return NativeReadServices(
            SqlReadPolicy(
                self.env.base_dir, self.declared, self._check_cancelled, self.check_materialization
            ),
            self._native_sources,
            self._sql_connections,
            self._operations,
            self._catalog.prepare,
            self._native_fingerprint,
            self._publish_native,
            self._hash_inputs,
        )

    def _native_sources(
        self, names: tuple[str, ...], extra: Mapping[str, Mapping[str, Any] | DataSource]
    ) -> dict[str, DataSource]:
        """Bind declared native participants and ad-hoc aliases for one native read."""
        from qavf.engine.sql_read import NativeReadUnavailable

        for name in names:
            self._require_available(name)
            if name in self._frames or name in self._lazy or name not in self._catalog.sources:
                raise NativeReadUnavailable(
                    f"dataset {name!r} requires its typed/materialized reader"
                )
        bound = {name: self._catalog.sources[name] for name in names}
        declared = {name.lower() for name in self.declared()}
        for name, config in extra.items():
            if name.lower() in declared:
                raise ValueError(f"ad-hoc source alias {name!r} collides with a declared dataset")
            bound[name] = (
                (
                    config
                    if config.env.connection_budget is self._resources
                    else config.bind(self.env)
                )
                if isinstance(config, DataSource)
                else self.build_source(config)
            )
        return bound

    def _native_fingerprint(self, name: str, source: DataSource) -> dict[str, Any] | None:
        """Fingerprint a native participant unless one is already recorded."""
        if not self._catalog.has_fingerprint(name):
            return self._fingerprint(name, source)
        return None

    def _publish_native(
        self,
        names: tuple[str, ...],
        observations: dict[str, tuple[dict[str, pl.DataType], float]],
        fingerprints: dict[str, dict[str, Any] | None],
    ) -> None:
        """Record a completed native read's capabilities, schemas and fingerprints."""
        with self._lock:
            self._catalog.mark_identity_used(names)
            for name in names:
                self._capabilities.record_native(name)
                self._catalog.record_native(name, *observations[name])
                self._catalog.record_native_plan(name)
                if not self._catalog.has_fingerprint(name):
                    self._catalog.record_fingerprint(name, fingerprints[name])

    def source_read_services(self) -> SourceReadServices:
        """Compose scoped-reader dependencies at the engine boundary."""
        from qavf.engine.batch_input import observe_batches
        from qavf.engine.observation import observe_source
        from qavf.engine.source_read import SourceReadServices

        return SourceReadServices(
            self._check_cancelled,
            self.declared,
            self.check_materialization,
            functools.partial(observe_batches, self),
            self._lazy_observation,
            self._load_ad_hoc,
            self._native_ad_hoc_schema,
            self._native_ad_hoc_frame,
            functools.partial(observe_source, self),
            self.work_store,
        )

    def _lazy_observation(self, plan: pl.LazyFrame, budget: int) -> Observation:
        """Observe an ad-hoc lazy plan in full under its row budget."""
        from qavf.engine.observation import _lazy

        return _lazy(self, plan, None, budget)

    def _load_ad_hoc(self, source: DataSource) -> pl.DataFrame:
        """Load an ad-hoc source eagerly, recording the fallback."""
        self._resources.eager_fallbacks.add("<ad-hoc>")
        return self._load_with_retries("<ad-hoc>", source)

    def _native_ad_hoc_schema(self, source: DataSource, alias: str) -> dict[str, pl.DataType]:
        """Read an ad-hoc source's schema through a native session."""
        from qavf.engine.sql_read import consume_native

        return consume_native(
            self,
            _NativeStatement("schema", f"SELECT * FROM {quote_identifier(alias)}"),
            sources={alias: source},
        )

    def _native_ad_hoc_frame(
        self, source: DataSource, alias: str, columns: list[str]
    ) -> pl.DataFrame:
        """Read selected columns of an ad-hoc source through a native session."""
        from qavf.engine.sql_read import consume_native

        projection = ", ".join(quote_identifier(column) for column in columns)
        return consume_native(
            self,
            _NativeStatement("frame", f"SELECT {projection} FROM {quote_identifier(alias)}"),
            sources={alias: source},
        )

    def read_source[T](
        self, config: Mapping[str, Any], consumer: Callable[[SourceReadView], T]
    ) -> T:
        """Read an ad-hoc source through scoped schema/projection/provenance methods.

        The source is bound once to this run's actual policy/resource owner. Every
        retry gets a new view; initialize aggregate state inside the callback.
        Complete-source row budgets and final identity checks precede publication.
        """
        from qavf.engine.source_read import read_source

        return read_source(self, config, consumer)

    def load_source(self, config: Mapping[str, Any]) -> pl.DataFrame:
        """Load an ad-hoc source specification (e.g. a baseline), uncached.

        Raises:
            KeyError, ValueError: for an invalid specification.
            SourceError: if loading fails.
        """
        registry = self._source_registry or SourceRegistry().discover()
        source = registry.build(config, self.env)
        return self._load_with_retries("<ad-hoc>", source)

    def source_fingerprint(self, config: Mapping[str, Any]) -> dict[str, Any] | None:
        """The fingerprint of an ad-hoc source, with a content digest where possible
        (baselines are what a comparison is judged against, so their identity matters)."""
        registry = self._source_registry or SourceRegistry().discover()
        try:
            return registry.build(config, self.env).fingerprint(hash_contents=True)
        except Exception as exc:
            logger.warning(
                "Could not fingerprint source %s: %s",
                redact_text(str(config)),
                exception_details(exc),
            )
            return None

    def memo[T](self, key: Any, compute: Callable[[], T]) -> T:
        """Compute ``compute()`` once per ``key`` and share it between rules.

        Computations for different keys run concurrently; ``compute`` may itself
        load datasets or use other memo keys. Supported values are detached on
        insertion/retrieval; oversized or opaque values are returned but not cached.
        """
        with self._operations.guard(), self._key_lock(("memo", key)):
            with self._lock:
                if key in self._memo:
                    hit: T = self._memo[key]
                    return hit
            value = compute()
            with self._lock:
                self._memo[key] = value
            return value

    def extension(self, name: str) -> dict[str, Any]:
        """Plugin settings from the configuration's ``extensions.<name>`` (empty if absent)."""
        return dict(self._extensions.get(name, {}))

    def available(self) -> list[str]:
        """Names of datasets that loaded successfully (loads them)."""
        return [name for name in self.names() if self.has(name)]

    def describe(self) -> list[DatasetInfo]:
        """Describe loaded frames without reviving evicted data or native scans."""
        return self._catalog.describe()

    def unused(self) -> list[str]:
        """Datasets with a source that nothing has asked for yet (not loaded)."""
        return self._catalog.unused()

    def unavailable(self) -> dict[str, str]:
        """Known loading failures and declarations without inputs; never loads data."""
        return self._catalog.unavailable(self._frames)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve ``path`` against the configuration directory unless it is absolute."""
        return self.env.resolve(path)

    def thread_load_seconds(self) -> float:
        """Seconds this thread has spent loading datasets (for per-rule telemetry)."""
        return float(getattr(self._local, "load_seconds", 0.0))

    # ---------------------------------------------------------------- internal
    def _add_thread_load_time(self, seconds: float) -> None:
        self._local.load_seconds = self.thread_load_seconds() + seconds

    def _load_with_retries(self, name: str, source: DataSource) -> pl.DataFrame:
        """Eager and native paths use the same identity, retry and cancellation policy."""
        record_fingerprint = name in self._catalog.sources and not self._catalog.has_fingerprint(
            name
        )
        fingerprint: dict[str, Any] | None = None

        def collect() -> pl.DataFrame:
            nonlocal fingerprint
            source.prepare_identity(hash_contents=self._hash_inputs)
            self._resources.eager_fallbacks.add(name)
            result = collect_result(source.load(), source.params.row_budget)
            self._check_materialization(name, result)
            budget = getattr(source.params, "row_budget", None)
            if budget is not None and result.height > budget:
                raise SourceError(f"dataset {name!r} has more than {budget:,} rows (row_budget)")
            if record_fingerprint:
                fingerprint = self._fingerprint(name, source)
            return result

        result = self._operations.run([source], collect)
        with self._lock:
            if name in self._catalog.sources:
                self._catalog.mark_identity_used([name])
                if record_fingerprint:
                    self._catalog.record_fingerprint(name, fingerprint)
        return result

    def _check_row_budget(self, name: str, frame: pl.DataFrame) -> None:
        spec = self.specs.get(name)
        budget = spec.effective_load_policy().row_budget if spec is not None else None
        if budget is not None and frame.height > budget:
            raise SourceError(f"dataset {name!r} has more than {budget:,} rows (row_budget)")

    def _check_lazy_budget(self, name: str) -> None:
        if name in self._frames:
            # Eager materialization already checked the complete source result.
            # scan() can legitimately return that frame's lazy wrapper.
            return
        spec = self.specs.get(name)
        source = self._catalog.sources.get(name)
        if source is not None and source.batch_schema() is not None:
            self._batch_observation(name, source)
            return
        budget = (
            source.params.row_budget
            if source is not None
            else (spec.effective_load_policy().row_budget if spec is not None else None)
        )
        if budget is None or name in self._budget_observations:
            return
        with self._operations.guard(), self._key_lock(("row_budget", name)):
            if name in self._budget_observations:
                return
            from qavf.engine.observation import _lazy

            attempts: list[Observation] = []

            def admit() -> Observation:
                for previous in attempts:
                    previous.close()
                attempts.clear()
                plan = self._prepare_lazy(name, self._lazy_snapshot(name))
                observed = _lazy(self, plan, None, budget)
                attempts.append(observed)
                return observed

            try:
                observed = self._source_operation([name], admit)
            except BaseException:
                for previous in attempts:
                    previous.close()
                raise
            with self._lock:
                self._budget_observations[name] = observed

    def verify_inputs(self) -> dict[str, str]:
        """Final consistency check for every source used by any execution mode."""
        failures: dict[str, str] = {}
        if self._active:
            # An overdue worker may hold a source's identity lock while hashing or
            # reading. Finalization must never wait on that worker's source state.
            # Admission also publishes catalog/cache state under this lock and
            # may be blocked in native compaction. Reporting marks its retained
            # metadata snapshot pending if the lock cannot be acquired now.
            if self._lock.acquire(blocking=False):
                try:
                    for name in self._catalog.identity_names():
                        self._catalog.identity_pending(name)
                finally:
                    self._lock.release()
            return failures
        for name in self._catalog.identity_names():
            try:
                with self._operations.guard():
                    self._catalog.sources[name].verify_identity()
            except Exception as exc:
                failure = redacted_error(exc)
                self._catalog.identity_failed(name, failure)
                failures[name] = str(failure)
        return failures

    def _load_snapshot(self, connection: str) -> None:
        """Publish only a complete, policy-verified snapshot attempt.

        Permanent unavailable members remain individually unavailable; transient
        failures restart the entire transaction. Retry and identity decisions
        belong to the same coordinator as eager and native source execution.
        A group that fits the memory allowance is published from memory; a larger
        one is staged as owned spill files and published together as spilled cache
        entries, so the group is bounded by spill space, not memory.
        """
        with self._operations.guard(), self._key_lock(("snapshot", connection)):
            members = sorted(
                name
                for name, source in self._catalog.sources.items()
                if source.snapshot_connection() == connection
            )
            with self._lock:
                if all(
                    name in self._frames or self._catalog.failure(name) is not None
                    for name in members
                ):
                    return
            attempts = _LatestAttempt(
                functools.partial(self._snapshot_attempt, connection, members)
            )
            try:
                stage = self._operations.run(
                    [self._catalog.sources[name] for name in members], attempts
                )
                self._check_cancelled()
                publish_metadata = functools.partial(
                    self._publish_snapshot_metadata, connection, members, stage
                )
                if stage.spilled:
                    self._frames.adopt(stage.spills(), publish_metadata, self._lock)
                    stage.detach()
                else:
                    self._frames.publish(stage.frames(), publish_metadata, self._lock)
            except Exception as exc:
                self._check_cancelled()
                failure = redacted_error(
                    SourceError(f"snapshot {connection!r} was not assessed: {exc}")
                )
                with self._lock:
                    for name in members:
                        self._catalog.record_failure(name, failure)
            finally:
                attempts.close()

    def _publish_snapshot_metadata(
        self, connection: str, members: list[str], stage: SnapshotStage
    ) -> None:
        """Record a snapshot attempt's member failures and shapes at publication."""
        self._check_cancelled()
        for name in stage.members:
            self._require_available(name)
        for name, failure in stage.failures.items():
            self._catalog.record_failure(name, failure)
        for name, member in stage.members.items():
            self._catalog.record_snapshot(
                name, member.schema, member.rows, connection, members, stage.started_at
            )

    def _snapshot_attempt(self, connection: str, members: list[str]) -> SnapshotStage:
        """Read one transaction into memory or owned spill files without publishing.

        Each member streams in bounded Arrow batches. Members stay in memory while
        the group fits within the cache limit and memory target; past that, the
        attempt holds about one batch in memory at a time. A failed or cancelled
        attempt deletes its files.
        """
        bindings: dict[str, ConnectionSpec] = {}
        secrets: dict[str, SecretSpec] = {}
        for name in members:
            merge_connection_binding(bindings, connection, self._catalog.sources[name].env)
            merge_secret_bindings(secrets, self._catalog.sources[name].env)
        failures: dict[str, SourceError] = {}
        pending = list(members)
        with self._sql_connections.temporary([connection], bindings, secrets) as con:
            # A permanent member failure that aborts the transaction (DuckDB 2.0
            # aborts on binder errors) restarts the read without that member, so
            # every published member still comes from one transaction.
            while True:
                stage = SnapshotStage(
                    self._frames.new_spill,
                    self._catalog.prepare,
                    datetime.now(UTC),
                    self._snapshot_memory_bytes(),
                )
                stage.failures = failures
                aborted = False
                try:
                    con.execute("BEGIN TRANSACTION")
                    for name in pending:
                        self._check_cancelled()
                        try:
                            self._stage_member(stage, con, name)
                        except (ResourceLimitError, MemberPreparationError):
                            raise
                        except Exception as exc:
                            error = exc if isinstance(exc, SourceError) else redacted_error(exc)
                            if isinstance(error, SourceIntegrityError) or error.retryable:
                                raise error from None
                            failures[name] = redacted_error(
                                error,
                                SourceNotFoundError
                                if isinstance(error, SourceNotFoundError)
                                else SourceError,
                            )
                            if not _transaction_usable(con):
                                # DuckDB 2.0 aborts on binder errors too; only a
                                # failure before execution may restart without it.
                                if not _failed_before_execution(exc):
                                    raise failures.pop(name) from None
                                aborted = True
                                break
                            continue
                    if not aborted:
                        con.execute("COMMIT")
                        return stage
                except BaseException as exc:
                    with contextlib.suppress(duckdb.Error):
                        con.rollback()
                    stage.close()
                    if isinstance(exc, MemberPreparationError):
                        raise exc.error from None
                    raise
                with contextlib.suppress(duckdb.Error):
                    con.rollback()
                stage.close()
                pending = [name for name in pending if name not in failures]

    def _stage_member(self, stage: SnapshotStage, con: Any, name: str) -> None:
        """Stream one member inside the snapshot transaction, admitting its row budget."""
        source = self._catalog.sources[name]
        budget = source.params.row_budget
        try:
            relation = source.snapshot_relation(con)
            if relation is None:
                frame = collect_result(source.read(con), budget)
                self._admit_snapshot_rows(name, budget, frame.height)
                self._check_cancelled()
                stage.stage_frame(name, frame)
                return
            admit = functools.partial(self._admit_snapshot_rows, name, budget)
            with BoundedBatches(relation, admit, self._check_cancelled) as batches:
                stage.stage_batches(name, batches)
        except duckdb.Error as exc:
            raise read_error(exc) from exc

    def _snapshot_memory_bytes(self) -> int:
        """The decoded bytes a snapshot group may hold in memory before it spills."""
        target = self._resources.memory_bytes
        return self._frames.limit if target is None else min(self._frames.limit, target)

    @staticmethod
    def _admit_snapshot_rows(name: str, budget: int | None, rows: int) -> None:
        if budget is not None and rows > budget:
            raise SourceError(f"dataset {name!r} has more than {budget:,} rows (row_budget)")

    def _cached(self, name: str) -> pl.DataFrame | None:
        """The loaded frame (as a clone), ``None`` if not loaded; raises for a recorded failure."""
        with self._lock:
            if name not in self._frames:
                if self._catalog.failure(name) is not None:
                    raise DatasetNotAvailableError(str(self._catalog.failure(name)))
                return None
        # Read outside the metadata lock: a spilled frame's read must not stall
        # unrelated context operations. FrameStore returns a distinct wrapper over
        # its owned buffers, including on promotion; no second clone is necessary.
        try:
            return self._frames[name]
        except KeyError:
            return None  # released meanwhile: loaded again by the caller

    def _connection(
        self,
        attach: Iterable[str],
        bindings: Mapping[str, ConnectionSpec] | None = None,
        secrets: Mapping[str, SecretSpec] | None = None,
    ) -> duckdb.DuckDBPyConnection | OwnedConnection:
        """This thread's reusable connection with exactly ``attach`` attached."""
        return self._sql_connections.connection(attach, bindings, secrets)

    def _key_lock(self, key: Any) -> contextlib.AbstractContextManager[None]:
        return self._key_locks.hold(key, check_cancelled=self._check_cancelled)

    def _fingerprint(self, name: str, source: DataSource) -> dict[str, Any] | None:
        """The source's fingerprint; provenance must never break a run."""
        try:
            return source.fingerprint(hash_contents=self._hash_inputs)
        except SourceIntegrityError:
            raise
        except Exception as exc:
            logger.warning(
                "Could not fingerprint dataset %r: %s", redact_text(name), exception_details(exc)
            )
            return None


def stable_order(frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` sorted by every column that can be sorted (nested columns are not keys)."""
    keys = [
        name
        for name, dtype in frame.schema.items()
        if not isinstance(dtype, pl.List | pl.Array | pl.Struct | pl.Object)
    ]
    if not keys or frame.height < 2:
        return frame
    return frame.sort(keys, nulls_last=True, maintain_order=True)


__all__ = ["DataContext", "DatasetNotAvailableError", "stable_order"]
