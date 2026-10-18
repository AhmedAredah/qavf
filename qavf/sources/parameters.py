"""Validated configuration models for the built-in source families."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from qavf.sources.base import SourceParams
from qavf.sources.formats import GLOB_PATTERN, FileFormat, infer_format, is_remote, lossy_options

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""A string that must contain something other than whitespace."""


class FileParams(SourceParams):
    """Parameters of :class:`qavf.sources.FileSource`."""

    type: Literal["file"] = "file"
    path: str
    """Exact file path, glob or URL; relative to the configuration file.

    Leading and trailing whitespace is part of the location, never stripped.
    """
    format: FileFormat | None = None
    """Inferred from the suffix when omitted."""
    options: dict[str, Any] = Field(default_factory=dict)
    """Extra arguments for the DuckDB reader, e.g. ``{delim: ';', header: true}`` for
    CSV or ``{sheet: Counts}`` for Excel."""
    columns: list[str] | None = Field(default=None, min_length=1)
    """Only load these columns (the report discloses the projection)."""
    min_files: int | None = Field(default=None, ge=1)
    """With a local glob: fewer matching files than this fails the load (a glob alone
    only proves that some files matched)."""
    acknowledge_lossy: bool = False
    """Allow reader options that can leave rows out (``ignore_errors``, ``skip``,
    ``store_rejects``, Excel ``range``/``stop_at_empty``). Rejected CSV rows are then
    counted and reported."""

    @field_validator("path")
    @classmethod
    def _exact_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must contain a non-whitespace character")
        return value

    @field_validator("options", mode="before")
    @classmethod
    def _typed_options(cls, options: Any) -> dict[str, Any]:
        if not isinstance(options, Mapping):
            raise ValueError("reader options must be a mapping")
        result: dict[str, Any] = {}
        boolean = {
            "ignore_errors",
            "store_rejects",
            "stop_at_empty",
            "header",
            "all_varchar",
            "null_padding",
            "union_by_name",
            "auto_detect",
            "parallel",
            "strict_mode",
        }
        integer = {"skip", "rejects_limit"}
        for original, value in options.items():
            if not isinstance(original, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z_0-9]*", original
            ):
                raise ValueError("reader option names must be SQL identifiers")
            name = original.lower()
            if name in result:
                raise ValueError(f"reader option {name!r} is repeated with different casing")
            if name in boolean and type(value) is not bool:
                raise ValueError(
                    f"reader option {name!r} must be a Boolean, not {type(value).__name__}"
                )
            if name in integer and (type(value) is not int or value < 0):
                raise ValueError(f"reader option {name!r} must be a non-negative integer")
            if name in ("rejects_table", "rejects_scan") and (
                not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value)
            ):
                raise ValueError(f"reader option {name!r} must be a simple table identifier")
            result[name] = value
        return result

    @model_validator(mode="after")
    def _known_format(self) -> FileParams:
        # An unknown suffix is a configuration error, reported before any loading.
        if self.format is None:
            infer_format(self.path)
        if self.min_files is not None and (
            is_remote(self.path) or not GLOB_PATTERN.search(self.path)
        ):
            raise ValueError("'min_files' needs a local glob path (e.g. data/*.parquet)")
        lossy = lossy_options(self.options)
        if lossy and not self.acknowledge_lossy:
            raise ValueError(
                f"reader option(s) {', '.join(lossy)} can leave rows out silently; set "
                f"acknowledge_lossy: true to accept that (rejected CSV rows are counted)"
            )
        return self


class QueryParams(SourceParams):
    """Parameters of :class:`qavf.sources.QuerySource`."""

    type: Literal["query"] = "query"
    sql: NonBlank
    """Any DuckDB query: read_csv/read_parquet/read_json over files, joins, filters,
    and tables of attached connections as ``<connection>.<table>``."""
    connections: list[str] | None = None
    """Connections to attach; by default every configured connection the query reads from
    as a database (``<name>.<table>``, quoted or in any case), found by DuckDB's parser."""

    @field_validator("connections")
    @classmethod
    def _unique_connections(cls, names: list[str] | None) -> list[str] | None:
        if names is not None and len({name.lower() for name in names}) != len(names):
            raise ValueError("connections must not repeat names, including case variants")
        return names


class DatabaseParams(SourceParams):
    """Parameters of :class:`qavf.sources.DatabaseSource`."""

    type: Literal["database"] = "database"
    connection: NonBlank
    """Name of a connection under ``connections`` in the configuration."""
    table: NonBlank | None = None
    db_schema: str | None = Field(default=None, alias="schema")
    query: NonBlank | None = None
    """SQL run against the attached database; reference its tables as
    ``<connection>.<table>``. Mutually exclusive with ``table``."""

    @model_validator(mode="after")
    def _table_or_query(self) -> DatabaseParams:
        if (self.table is None) == (self.query is None):
            raise ValueError("set exactly one of 'table' or 'query'")
        return self


class PythonParams(SourceParams):
    """Parameters of :class:`qavf.sources.PythonSource`."""

    type: Literal["python"] = "python"
    callable: NonBlank
    """``package.module:function`` returning a Polars or pandas DataFrame or a
    PyArrow table."""
    kwargs: dict[str, Any] = Field(default_factory=dict)
