"""Explicit file, dataset, query and callable input descriptions."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from typing import Any

import duckdb

from qavf.settings import DatasetSpec, ExpressionCheck, ForeignKeySpec, SchemaSpec, SourcePolicy
from qavf.sources.base import SourceError
from qavf.sources.formats import FileFormat
from qavf.sources.parameters import FileParams
from qavf.sources.sql import validate_query


@dataclass(frozen=True, slots=True)
class File:
    """A file input with explicit reader settings, independent of an old source.

    Use a plain path to retain configured reader settings. ``File`` replaces
    those settings, while dataset contracts and loading safeguards remain.
    Construction validates settings without discovering or reading files.
    """

    path: str | os.PathLike[str]
    _: KW_ONLY
    format: FileFormat | None = None
    options: Mapping[str, Any] | None = None
    columns: Sequence[str] | None = None
    min_files: int | None = None
    acknowledge_lossy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.path, str | os.PathLike):
            raise TypeError("File.path must be a string or path")
        path = os.fspath(self.path)
        if not isinstance(path, str):
            raise TypeError("File.path must resolve to a string, not bytes")
        params = FileParams.model_validate(
            {
                "path": path,
                "format": self.format,
                "options": {} if self.options is None else self.options,
                "columns": self.columns,
                "min_files": self.min_files,
                "acknowledge_lossy": self.acknowledge_lossy,
            }
        )
        object.__setattr__(self, "path", params.path)
        object.__setattr__(self, "format", params.format)
        object.__setattr__(self, "options", MappingProxyType(deepcopy(params.options)))
        object.__setattr__(self, "columns", tuple(params.columns) if params.columns else None)
        object.__setattr__(self, "min_files", params.min_files)
        object.__setattr__(self, "acknowledge_lossy", params.acknowledge_lossy)

    def source_config(self) -> dict[str, Any]:
        """Return a fresh validated specification, without source-policy defaults."""
        params = FileParams.model_validate(
            {
                "path": self.path,
                "format": self.format,
                "options": self.options,
                "columns": self.columns,
                "min_files": self.min_files,
                "acknowledge_lossy": self.acknowledge_lossy,
            }
        )
        return params.model_dump(exclude={"row_budget", "retries", "retry_delay_s"})


@dataclass(frozen=True, slots=True)
class Dataset:
    """An input and its inline QA contract; omit fields to retain configuration.

    Contract declarations are snapshotted without copying ``data``. Duplicate
    contract fields in configuration are rejected. Explicit loading policies
    may add safeguards, but cannot override already-declared policy values.
    """

    data: object
    _: KW_ONLY
    schema: SchemaSpec | Mapping[str, Any] | None = None
    rename: Mapping[str, str] | None = None
    foreign_keys: Sequence[ForeignKeySpec | Mapping[str, Any]] | None = None
    expressions: Sequence[ExpressionCheck | Mapping[str, Any]] | None = None
    load_policy: SourcePolicy | Mapping[str, Any] | None = None
    required: bool | None = None
    description: str | None = None
    _spec: DatasetSpec = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.data is None or isinstance(self.data, Dataset):
            raise TypeError("Dataset.data must be an input, not None or another Dataset")
        raw = {
            name: getattr(self, name)
            for name in (
                "schema",
                "rename",
                "foreign_keys",
                "expressions",
                "load_policy",
                "required",
                "description",
            )
            if getattr(self, name) is not None
        }
        spec = DatasetSpec.model_validate(raw).model_copy(deep=True)
        object.__setattr__(self, "_spec", spec)
        for name in raw:
            value = getattr(spec, "schema_spec" if name == "schema" else name)
            if name == "rename":
                value = MappingProxyType(value)
            elif name in ("foreign_keys", "expressions"):
                value = tuple(value)
            object.__setattr__(self, name, value)

    def contract(self) -> DatasetSpec:
        """Revalidate a separate contract while retaining omitted fields."""
        supplied = self._spec.model_fields_set
        raw = self._spec.model_dump(by_alias=True)
        return DatasetSpec.model_validate(
            {
                "schema" if name == "schema_spec" else name: raw[
                    "schema" if name == "schema_spec" else name
                ]
                for name in supplied
            }
        )


@dataclass(frozen=True, slots=True)
class Query:
    """A parameterized read query, distinct from a string denoting a file path.

    ``connection`` is a configured connection name or a borrowed DuckDB
    connection. Borrowed connections remain open and are materialized once at
    the explicit load boundary; they cannot participate in native source scans.
    Queries without borrowed connections retain a native scan descriptor.
    Parameter bindings are copied and frozen; nested values remain caller-owned
    and must not be mutated during validation.
    """

    sql: str
    connection: str | duckdb.DuckDBPyConnection | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sql, str) or not self.sql.strip():
            raise ValueError("Query.sql must be a nonempty SQL query")
        try:
            object.__setattr__(self, "sql", validate_query(self.sql))
        except SourceError as exc:
            raise ValueError(f"Query requires exactly one SELECT query: {exc}") from None
        if self.connection is not None and not isinstance(
            self.connection, str | duckdb.DuckDBPyConnection
        ):
            raise TypeError("Query.connection must be a configured name or DuckDB connection")
        if isinstance(self.connection, str) and not self.connection.strip():
            raise ValueError("Query.connection must be a nonempty configured name")
        if not isinstance(self.parameters, Mapping) or any(
            not isinstance(key, str) for key in self.parameters
        ):
            raise TypeError("Query.parameters must be a mapping with string keys")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class Loader:
    """A function invoked when its dataset is first required in a run.

    The function may return a supported frame, Arrow table or Polars lazy frame.
    By default it is invoked once. Retries require an explicit loading policy,
    ``replayable=True``, ``retry_safe=True`` and a retryable SourceError. A borrowed
    one-pass iterator is not an accepted return value: wrap a function that
    produces a fresh supported dataset or use an explicit source plugin.
    Keyword bindings are copied and frozen, but nested values and external
    handles are not copied. Keep their state stable while validation is active.
    """

    function: Callable[..., object]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    replayable: bool = False
    retry_safe: bool = False

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("Loader.function must be callable")
        if not isinstance(self.kwargs, Mapping) or any(
            not isinstance(key, str) for key in self.kwargs
        ):
            raise TypeError("Loader.kwargs must be a mapping with string keys")
        if not isinstance(self.replayable, bool) or not isinstance(self.retry_safe, bool):
            raise TypeError("Loader replayable and retry_safe must be booleans")
        if self.retry_safe and not self.replayable:
            raise ValueError("a retry-safe Loader must also be replayable")
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))
