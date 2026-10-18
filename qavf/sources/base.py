"""Data-source contract and registry.

A *source* turns a configuration block into a Polars DataFrame. Rules never see
sources: they ask the :class:`~qavf.engine.context.DataContext` for a dataset by
name, so any rule works with any source.

Writing a source::

    class HttpCsvParams(SourceParams):
        type: Literal["http_csv"] = "http_csv"
        url: str

    class HttpCsvSource(DataSource):
        kind: ClassVar[str] = "http_csv"
        Params: ClassVar[type[SourceParams]] = HttpCsvParams

        def load(self) -> pl.DataFrame | pl.LazyFrame:
            return pl.read_csv(self.params.url)

Register it by listing its module under ``plugins.modules`` or by advertising it
in the ``qavf.sources`` entry-point group. Configuration then selects it with
``source: {type: http_csv, url: ...}``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from importlib.metadata import entry_points
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from qavf.redaction import exception_details, redact_text

if TYPE_CHECKING:
    from qavf.settings import ConnectionSpec, SecretSpec

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "qavf.sources"


class SourceError(RuntimeError):
    """A load failed; only explicitly transient errors are eligible for retry."""

    def __init__(self, *args: object, retryable: bool = False) -> None:
        super().__init__(*args)
        self.retryable = retryable


class SourceNotFoundError(SourceError):
    """The data a source points at does not exist (as opposed to failing to load)."""


class SourceIntegrityError(SourceError):
    """The input changed after its run identity was established; never optional metadata."""


class SourceParams(BaseModel):
    """Base class for source parameter models. Unknown keys are rejected."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, use_attribute_docstrings=True, hide_input_in_errors=True
    )

    type: str
    retries: int = Field(default=0, ge=0, le=10)
    """Retry a failed load this many times (not when the data is simply not found),
    for transient failures of remote files and databases."""
    retry_delay_s: float = Field(default=1.0, ge=0, le=300)
    """Delay before the first retry; doubled before each further one."""
    row_budget: int | None = Field(default=None, ge=1)
    """Maximum result rows. Exceeding this fails the load. Query aggregates, sorts
    and joins may still scan more input rows; this is not a memory or scan budget."""


class ConnectionBudget(Protocol):
    """Run-wide reservation owner; sources release each acquired token on close."""

    def acquire(self) -> tuple[object, int | None, int | None]: ...
    def release(self, token: object) -> None: ...
    def register_connection(self, token: object, connection: Any) -> None: ...
    def unregister_connection(self, token: object) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceEnvironment:
    """What a source may need beyond its own parameters."""

    base_dir: Path = field(default_factory=Path.cwd)
    """Directory that relative paths resolve against (the configuration's directory)."""
    connections: Mapping[str, ConnectionSpec] = field(default_factory=dict)
    """Named database connections from the configuration."""
    secrets: Mapping[str, SecretSpec] = field(default_factory=dict)
    """Cloud storage secrets created on every connection and used for remote reads."""
    threads: int | None = None
    """DuckDB worker threads per connection; ``None`` for DuckDB's default (all cores).
    Set when rules run in parallel so the two levels of parallelism share the CPUs."""
    memory_limit: str | None = None
    """DuckDB memory limit per connection (e.g. ``"4GB"``); larger work spills to disk."""

    temp_directory: Path | None = None
    """Context-owned directory for connection-local temporary spill storage."""
    spill_limit_bytes: int | None = None
    """Maximum temporary storage for each connection, allocated by the engine."""

    connection_budget: ConnectionBudget | None = None
    """Shared run-wide allocator for native connection memory and spill capacity."""

    def resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` against :attr:`base_dir` unless it is absolute."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.base_dir / candidate).resolve()


@dataclass(frozen=True, slots=True)
class ScanPlan:
    """A native SQL scan; the caller owns connection, transaction and result lifecycle.

    The SQL describes the entire source, before user projections or aggregations.
    Result-row limits and source loading policy remain the context's responsibility.
    No query is executed while this descriptor is constructed.
    """

    sql: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    connections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "connections", tuple(self.connections))


class DataSource[P: SourceParams](ABC):
    """Loads one dataset. Instances are created per dataset from configuration."""

    retry_safe: bool = True
    """Whether an explicitly configured retry may safely invoke the source again."""
    kind: ClassVar[str]
    """Identifier used as ``source.type`` in configuration."""
    Params: ClassVar[type[SourceParams]]

    def __init__(self, params: P, env: SourceEnvironment) -> None:
        self.params: P = params
        self.env = env
        self._state_lock = threading.RLock()
        self.load_notes: dict[str, Any] = {}
        """Facts about the last load a report should disclose (reader options in
        effect, rows the reader rejected, files matched); set by :meth:`load`."""

    def bound_environment(self, env: SourceEnvironment) -> SourceEnvironment:
        """Keep the source's location/credentials while borrowing a run's resource policy."""
        return replace(
            env,
            base_dir=self.env.base_dir,
            connections=MappingProxyType(dict(self.env.connections)),
        )

    def bind(self, env: SourceEnvironment) -> DataSource[P]:
        """Create an independent run session without mutating this source definition.

        Parameter-only sources inherit this implementation. A source with a custom
        constructor or stateful external handle must implement its ownership policy
        explicitly here; arbitrary plugin state is never copied implicitly.
        """
        if type(self).__init__ is not DataSource.__init__:
            raise SourceError(
                f"source {self.kind!r} has a custom constructor; implement bind(env) "
                "to create a run-local source session"
            )
        params = type(self.params).model_validate(self.params.model_dump(by_alias=True))
        bound = type(self)(params, self.bound_environment(env))
        bound.retry_safe = self.retry_safe
        return bound

    def verify_identity(self) -> None:
        """Verify the established input identity before/after a complete operation.

        Sources with mutable backing data override this and raise
        :class:`SourceIntegrityError` on a detected change. Such failures must mark
        the source unassessed instead of silently discarding its provenance.
        """
        return None

    @contextmanager
    def verified_operation(self) -> Iterator[None]:
        """Verify a complete operation; specialized sources may coalesce nested checks.

        The default preserves every plugin's existing identity hooks. A source
        overriding this must verify again before successfully returning a result.
        """
        from qavf.sources.errors import verify_postcondition

        self.verify_identity()
        error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            verify_postcondition(self.verify_identity, error)

    def identity_telemetry(self) -> dict[str, int]:
        """Bounded identity-work counters, when the source can measure them."""
        return {}

    def prepare_identity(self, hash_contents: bool = False) -> None:
        """Establish an optional read identity before an operation starts reading.

        Unlike fingerprint(), this hook must not depend on a completed load.
        Mutable file sources use it to anchor their first content-verified read.
        """
        return None

    @abstractmethod
    def load(self) -> pl.DataFrame | pl.LazyFrame:
        """Return the dataset.

        Raises:
            SourceNotFoundError: if the underlying data does not exist.
            SourceError: for any other loading failure.
        """

    def scan(self) -> ScanPlan | None:
        """Describe a native scan when supported; otherwise use the bounded load path."""
        return None

    def scan_polars(self) -> pl.LazyFrame | None:
        """Describe the complete source with exact logical types when SQL cannot.

        The context applies row budgets and coordinates identity verification,
        retries and cancellation around execution of the returned plan. Merely
        returning a plan does not assess or materialize its records.
        """
        return None

    def snapshot_connection(self) -> str | None:
        """The configured connection this source reads through, when it can take part in
        a snapshot (see :meth:`read`); ``None`` otherwise."""
        return None

    def read(self, con: Any) -> pl.DataFrame | pl.LazyFrame:
        """Read the data on an open DuckDB connection (with the snapshot connection
        attached, inside the caller's transaction). Sources that support snapshots
        override it together with :meth:`snapshot_connection`."""
        raise NotImplementedError

    def describe(self) -> str:
        """Short human-readable description of where the data comes from.

        Reports pass it through :func:`qavf.redaction.redact_text`, but avoid
        including secrets in the first place.
        """
        return self.kind

    def fingerprint(self, hash_contents: bool = False) -> dict[str, Any] | None:
        """Identity of the underlying data, recorded in report provenance after loading.

        Override it to let a report show which version of the data was checked
        (file size and modification time, a table's last-modified stamp, a
        snapshot id). The default records nothing.

        Args:
            hash_contents: Also compute a content digest, when that is possible.
        """
        return None


class SourceRegistry:
    """Maps ``source.type`` values to :class:`DataSource` classes."""

    def __init__(self) -> None:
        self._sources: dict[str, type[DataSource]] = {}
        self.plugins: dict[str, str] = {}
        """Entry points tried by :meth:`discover`: ``"loaded"`` or the error."""

    def register[S: DataSource](self, source_cls: type[S]) -> type[S]:
        """Register a source class. Usable as a decorator.

        Raises:
            ValueError: if a different class already uses the same ``type``.
        """
        existing = self._sources.get(source_cls.kind)
        if existing is not None and existing is not source_cls:
            same = (existing.__module__, existing.__qualname__) == (
                source_cls.__module__,
                source_cls.__qualname__,
            )
            if not same:
                raise ValueError(f"duplicate source type {source_cls.kind!r}")
        self._sources[source_cls.kind] = source_cls
        return source_cls

    def register_module(self, module: ModuleType) -> list[type[DataSource]]:
        """Register every concrete source class defined in ``module``."""
        staged = SourceRegistry()
        staged._sources = dict(self._sources)
        found = []
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, DataSource)
                and obj.__module__ == module.__name__
                and not inspect.isabstract(obj)
                and getattr(obj, "kind", None)
            ):
                found.append(staged.register(obj))
        self._sources = staged._sources
        return found

    def discover(
        self, modules: tuple[str, ...] = (), use_entry_points: bool = True
    ) -> SourceRegistry:
        """Register built-in sources, then sources from ``modules`` and entry points."""
        from qavf.sources.builtin import BUILTIN_SOURCES

        with self._transaction():
            for source_cls in BUILTIN_SOURCES:
                self.register(source_cls)
        for name in modules:
            try:
                with self._transaction():
                    self.register_module(importlib.import_module(name))
            except Exception as exc:
                raise ImportError(
                    f"could not load source module {redact_text(name)!r}: {exception_details(exc)}"
                ) from None
        if use_entry_points:
            entries = list(entry_points(group=ENTRY_POINT_GROUP))
            counts = Counter(entry.name for entry in entries)
            for entry in entries:
                if counts[entry.name] > 1:
                    self.plugins[entry.name] = "error: duplicate entry-point name in sources group"
                    logger.error("Duplicate source entry point %r", redact_text(entry.name))
                    continue
                try:
                    with self._transaction():
                        loaded = entry.load()
                        if inspect.ismodule(loaded):
                            self.register_module(loaded)
                        elif inspect.isclass(loaded) and issubclass(loaded, DataSource):
                            self.register(loaded)
                        else:
                            raise TypeError("it is neither a DataSource nor a module")
                except Exception as exc:
                    message = exception_details(exc)
                    logger.error(
                        "Failed to load source entry point %r: %s", redact_text(entry.name), message
                    )
                    self.plugins[entry.name] = f"error: {message}"
                    continue
                self.plugins[entry.name] = "loaded"
        return self

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        # A source module can use the public rule decorator too. Its imports
        # participate in the same QAVF-owned registration transaction.
        from qavf.engine.registry import registration_transaction

        before = dict(self._sources)
        try:
            with registration_transaction():
                yield
        except BaseException:
            self._sources = before
            raise

    def get(self, source_type: str) -> type[DataSource]:
        """Return the class for ``source_type``.

        Raises:
            KeyError: for an unknown type.
        """
        try:
            return self._sources[source_type]
        except KeyError:
            available = ", ".join(sorted(self._sources))
            raise KeyError(
                redact_text(f"unknown source type {source_type!r}; available: {available}")
            ) from None

    def build(self, config: Mapping[str, Any], env: SourceEnvironment) -> DataSource:
        """Validate ``config`` against the source's parameters and instantiate it.

        Raises:
            KeyError: for an unknown ``type``.
            ValueError: if ``config`` has no ``type`` or invalid parameters.
        """
        source_type = config.get("type")
        if not isinstance(source_type, str):
            raise ValueError("a source needs a 'type', e.g. {type: file, path: data.csv}")
        source_cls = self.get(source_type)
        try:
            params = source_cls.Params.model_validate(dict(config))
        except ValidationError as exc:
            raise ValueError(
                f"invalid {redact_text(source_type)!r} source: {exception_details(exc)}"
            ) from None
        try:
            return source_cls(params, env)
        except Exception as exc:
            raise ValueError(
                f"could not construct {redact_text(source_type)!r} source: {exception_details(exc)}"
            ) from None

    def __iter__(self) -> Iterator[type[DataSource]]:
        return iter(sorted(self._sources.values(), key=lambda s: s.kind))

    def __contains__(self, source_type: object) -> bool:
        return source_type in self._sources

    def __len__(self) -> int:
        return len(self._sources)
