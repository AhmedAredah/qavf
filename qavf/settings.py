"""Pydantic configuration models and YAML/JSON loading.

A configuration file describes *what* to validate (datasets, their sources and
their column-level QA limits), *how* (per-rule enablement, severity and
parameters) and *where* results go. ``qavf init`` writes a complete starting
configuration and ``qavf schema`` a JSON Schema for editor autocompletion.

String values may reference environment variables as ``${NAME}`` or
``${NAME:-default}``, which keeps credentials out of configuration files.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    field_serializer,
    model_serializer,
    model_validator,
)

from qavf.engine.severity import Severity

if TYPE_CHECKING:
    from qavf.sources.base import SourceEnvironment

logger = logging.getLogger(__name__)

DType = Literal["int", "float", "decimal", "str", "bool", "date", "datetime"]


def _preserve_numeric_type(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """Validate bounds without changing the meaning of an existing numeric value.

    Older Pydantic versions prefer Decimal over float in this union and convert a
    binary float through its decimal display. That changes an inclusive boundary.
    String bounds still follow normal parsing, including exact quoted Decimals.
    """
    if isinstance(value, bool):
        raise ValueError("numeric bounds must not be boolean")
    parsed = handler(value)
    return value if type(value) in (int, float, Decimal) else parsed


NumericBound = Annotated[int | Decimal | float, WrapValidator(_preserve_numeric_type)]
Bound = Annotated[
    int | Decimal | float | datetime | date, WrapValidator(_preserve_numeric_type)
]
"""A numeric or temporal bound. Integers stay integers (exact beyond 2**53); YAML
dates (``2020-01-01``) parse as dates."""

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Strict(BaseModel):
    """Base model that rejects unknown keys so typos in configuration fail loudly."""

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, use_attribute_docstrings=True,
        hide_input_in_errors=True,
    )


# ------------------------------------------------------------------ QA limits
def _require_finite(name: str, value: Any) -> None:
    if (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, Decimal) and not value.is_finite()
    ):
        raise ValueError(f"'{name}' must be a finite number, got {value}; omit it for no limit")


def _comparable(value: Any) -> Any:
    """Dates compare as midnight datetimes, so date and datetime bounds can be ordered."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    return value


def _check_range(lower: list[tuple[str, Any, bool]], upper: list[tuple[str, Any, bool]]) -> None:
    """Reject bounds no value can satisfy. Entries are ``(name, value, inclusive)``."""
    for low_name, low, low_inclusive in lower:
        for high_name, high, high_inclusive in upper:
            try:
                a, b = _comparable(low), _comparable(high)
                empty = a > b or (a == b and not (low_inclusive and high_inclusive))
            except TypeError as exc:
                raise ValueError(
                    f"'{low_name}' and '{high_name}' cannot be compared: {exc}"
                ) from exc
            if empty:
                raise ValueError(
                    f"'{low_name}: {low}' and '{high_name}: {high}' leave no allowed value"
                )


class Bounds(_Strict):
    """Inclusive bounds for an aggregate statistic; either side may be omitted."""

    ge: NumericBound | None = None
    le: NumericBound | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Bounds:
        if self.ge is None and self.le is None:
            raise ValueError("set 'ge', 'le' or both")
        _require_finite("ge", self.ge)
        _require_finite("le", self.le)
        if self.ge is not None and self.le is not None and self.ge > self.le:
            raise ValueError("'ge' must not exceed 'le'")
        return self


#: Row-level constraint fields shared by :class:`ColumnConstraints` subclasses.
ROW_CONSTRAINTS = (
    "ge",
    "le",
    "gt",
    "lt",
    "isin",
    "notin",
    "regex",
    "min_length",
    "max_length",
    "integral",
    "max_decimals",
)
#: Column-level (aggregate) constraint fields.
COLUMN_CONSTRAINTS = (
    "max_null_fraction",
    "unique",
    "min",
    "max",
    "mean",
    "median",
    "std",
    "sum",
    "distinct_count",
)


class ColumnConstraints(_Strict):
    """QA limits for a column. Every limit that is set must hold.

    Row-level limits flag individual values; column-level limits test the
    column as a whole.
    """

    # row level
    ge: Bound | None = None
    le: Bound | None = None
    gt: Bound | None = None
    lt: Bound | None = None
    isin: list[Any] | None = None
    """Allowed values."""
    notin: list[Any] | None = None
    """Forbidden values (sentinels such as -999)."""
    regex: str | None = None
    """Pattern every text value must match in full."""
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    integral: bool = False
    """Every value is a whole number (for float columns; integers always are)."""
    max_decimals: int | None = Field(default=None, ge=0)
    """At most this many decimal places; exact for decimals, bounded rounding noise for floats."""
    # column level
    max_null_fraction: float | None = Field(default=None, ge=0, le=1)
    """Largest acceptable share of missing values (requires ``nullable: true``
    on the column to be meaningful; non-nullable columns allow none)."""
    unique: bool = False
    min: Bounds | None = None
    """Bounds on the column minimum."""
    max: Bounds | None = None
    """Bounds on the column maximum."""
    mean: Bounds | None = None
    median: Bounds | None = None
    std: Bounds | None = None
    sum: Bounds | None = None
    distinct_count: Bounds | None = None

    def constraints(self) -> dict[str, Any]:
        """The limits that are set, by field name."""
        names = (*ROW_CONSTRAINTS, *COLUMN_CONSTRAINTS)
        values = {name: getattr(self, name) for name in names}
        return {k: v for k, v in values.items() if v is not None and v is not False}

    @model_validator(mode="after")
    def _satisfiable(self) -> ColumnConstraints:
        """Reject limits that are invalid, or that no value could ever satisfy.

        Boundary cases that admit a value (``ge == le``) are valid.
        """
        bounds = {name: getattr(self, name) for name in ("ge", "le", "gt", "lt")}
        bounds = {name: value for name, value in bounds.items() if value is not None}
        for name, value in bounds.items():
            _require_finite(name, value)
        kinds = {"date" if isinstance(value, date) else "number" for value in bounds.values()}
        if len(kinds) > 1:
            raise ValueError(
                f"bounds mix numbers and dates ({', '.join(sorted(bounds))}); use one kind"
            )
        _check_range(
            [(n, bounds[n], n == "ge") for n in ("ge", "gt") if n in bounds],
            [(n, bounds[n], n == "le") for n in ("le", "lt") if n in bounds],
        )
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError(
                f"'min_length: {self.min_length}' exceeds 'max_length: {self.max_length}'"
            )
        if self.isin is not None and not self.isin:
            raise ValueError("'isin: []' allows no value at all; list the allowed values")
        if self.regex is not None:
            _check_regex(self.regex)
        for name in ("std", "distinct_count"):
            limit = getattr(self, name)
            if limit is not None and limit.le is not None and limit.le < 0:
                raise ValueError(f"'{name}' can never be negative, but 'le' is {limit.le}")
        return self


def _check_where(condition: str) -> None:
    """Compile a ``where`` condition with the engine that evaluates it (Polars SQL)."""
    import polars as pl

    try:
        pl.sql_expr(condition)
    except Exception as exc:
        raise ValueError(f"invalid where condition {condition!r}: {exc}") from exc


def _check_regex(pattern: str) -> None:
    """Compile ``pattern`` with the engine that evaluates it (Polars' Rust regex)."""
    import polars as pl

    try:
        pl.Series([""], dtype=pl.String).str.contains(f"^(?:{pattern})$")
    except Exception as exc:
        reason = str(exc).strip().splitlines()
        raise ValueError(
            f"invalid regex {pattern!r}: {reason[-1] if reason else type(exc).__name__}"
        ) from exc


class ColumnCheck(ColumnConstraints):
    """A named group of QA limits with its own severity."""

    name: str | None = None
    """Label shown in reports; generated from the limits when omitted."""
    severity: Severity | None = None
    """Severity when this check fails; the column-check rule's severity when omitted."""
    description: str | None = None
    where: str | None = None
    """Apply the check only to rows matching this SQL condition (Polars SQL syntax),
    e.g. ``"status = 'valid'"``; other rows are outside its scope."""
    min_applicable_rows: int = Field(default=0, ge=0)
    """With ``where``: fewer matching rows than this makes the check not assessed."""

    @model_validator(mode="after")
    def _not_empty(self) -> ColumnCheck:
        if not self.constraints():
            raise ValueError("a check must set at least one limit (ge, le, isin, regex, …)")
        if self.where is not None:
            _check_where(self.where)
        elif self.min_applicable_rows:
            raise ValueError("'min_applicable_rows' goes with 'where'")
        return self


class ColumnSpec(ColumnConstraints):
    """Expectations for one column.

    Limits set directly on the column form an implicit check at the default
    severity; ``checks`` adds further checks with their own severities::

        vmt:
          dtype: float
          ge: 0                         # ERROR (rule default)
          checks:
            - {le: 1.0e9, severity: WARNING, name: implausibly large}
    """

    dtype: DType | None = None
    """Logical type. ``float`` also accepts integers; ``datetime`` also accepts dates."""
    required: bool = True
    nullable: bool = False
    """Allow missing values. NaN counts as missing, exactly like null."""
    allow_infinite: bool = False
    """Allow +/-infinity in a floating-point column (rejected by default)."""
    missing_values: list[Any] = Field(default_factory=list)
    """Values that mean "missing" (sentinels such as -999): counted as missing for
    ``nullable`` and ``max_null_fraction``. They are normalized to null for every
    QA consumer, including statistics, reconciliation, keys and expressions;
    each check then applies its documented null policy."""
    blank_as_missing: bool = False
    """Normalize empty and whitespace-only text to missing for every QA consumer."""
    unit: str | None = None
    """The column's unit (``kg``, ``vehicle-miles``). Recorded in reports; ``REC-001``
    refuses to reconcile columns declared in different units."""
    unit_column: str | None = None
    """A column holding each row's unit; with ``allowed_units`` (or ``unit``), rows in
    any other unit are violations."""
    allowed_units: list[str] | None = None
    description: str | None = None
    checks: list[ColumnCheck] = Field(default_factory=list)

    def all_checks(self) -> list[ColumnCheck]:
        """The implicit check (if any limits are set inline) followed by ``checks``."""
        inline = self.constraints()
        implicit = [ColumnCheck.model_validate(inline)] if inline else []
        return implicit + list(self.checks)

    @model_validator(mode="after")
    def _compatible_with_dtype(self) -> ColumnSpec:
        """With a declared ``dtype``, reject limits that cannot apply to it."""
        if self.blank_as_missing and self.dtype not in (None, "str"):
            raise ValueError(f"'blank_as_missing' applies to str columns, not {self.dtype}")
        if self.allowed_units is not None and not self.allowed_units:
            raise ValueError("'allowed_units' must not be empty")
        if self.unit_column is not None and self.unit is None and self.allowed_units is None:
            raise ValueError("'unit_column' needs 'unit' or 'allowed_units'")
        if self.allowed_units is not None and self.unit_column is None:
            raise ValueError("'allowed_units' goes with 'unit_column'")
        if self.dtype is None:
            return self
        numeric = self.dtype in ("int", "float", "decimal")
        temporal = self.dtype in ("date", "datetime")
        for check in self.all_checks():
            limits = check.constraints()
            for name in ("ge", "le", "gt", "lt"):
                if name not in limits:
                    continue
                is_date = isinstance(limits[name], date)
                if not (temporal if is_date else numeric):
                    kind = "date" if is_date else "numeric"
                    raise ValueError(
                        f"a {kind} bound '{name}' cannot apply to a {self.dtype} column"
                    )
            for name in ("regex", "min_length", "max_length"):
                if name in limits and self.dtype != "str":
                    raise ValueError(f"'{name}' applies to str columns, not {self.dtype}")
            for name in ("integral", "max_decimals"):
                if name in limits and not numeric:
                    raise ValueError(f"'{name}' applies to numeric columns, not {self.dtype}")
            for name in ("min", "max", "mean", "median", "std", "sum"):
                if name in limits and not numeric:
                    raise ValueError(
                        f"the '{name}' statistic needs a numeric column, not {self.dtype}"
                    )
        return self


class SchemaSpec(_Strict):
    """Expectations for a whole dataset."""

    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    primary_key: list[str] = Field(default_factory=list)
    """Columns that together must identify every row uniquely; none may be null."""
    strict: bool = False
    """Reject columns that are not declared."""
    min_rows: int = Field(default=1, ge=0)
    max_rows: int | None = Field(default=None, ge=0)
    unique_keys: list[list[str]] = Field(default_factory=list)
    """Further column sets that must each be unique (alternate keys). As in SQL, a row
    with a null in the set is exempt (and counted)."""
    unique_rows: bool = False
    """No two rows may be identical in every column."""

    @model_validator(mode="after")
    def _consistent_key(self) -> SchemaSpec:
        seen_keys: set[tuple[str, ...]] = set()
        for columns in self.unique_keys:
            if not columns:
                raise ValueError("unique_keys entries must name at least one column")
            if len(set(columns)) != len(columns):
                raise ValueError(f"unique key {columns} repeats a column")
            if tuple(columns) in seen_keys:
                raise ValueError(f"unique key {columns} is listed twice")
            seen_keys.add(tuple(columns))
        if self.max_rows is not None and self.min_rows > self.max_rows:
            hint = (
                "" if "min_rows" in self.model_fields_set else " (the default; set it explicitly)"
            )
            raise ValueError(
                f"'min_rows: {self.min_rows}'{hint} exceeds 'max_rows: {self.max_rows}', so no "
                f"dataset could pass"
            )
        repeated = sorted({c for c in self.primary_key if self.primary_key.count(c) > 1})
        if repeated:
            raise ValueError(f"primary_key repeats column(s): {', '.join(repeated)}")
        for column in self.primary_key:
            spec = self.columns.get(column)
            if spec is None:
                continue
            if spec.nullable:
                raise ValueError(
                    f"primary-key column {column!r} cannot be nullable; use 'unique' on the "
                    f"column for a nullable unique constraint"
                )
            if not spec.required:
                raise ValueError(f"primary-key column {column!r} must be required")
        return self


class ForeignKeySpec(_Strict):
    """Every distinct value of ``columns`` must exist in ``ref_dataset.ref_columns``."""

    columns: list[str] = Field(min_length=1)
    ref_dataset: str
    ref_columns: list[str] | None = Field(default=None, min_length=1)
    """Defaults to ``columns`` when the names match on both sides."""
    severity: Severity | None = None
    match: Literal["simple", "full"] = "simple"
    """How rows with null key components count. ``simple`` (SQL MATCH SIMPLE, the
    default): a row with any null component is exempt. ``full``: a partly null key
    is a violation; only all-null keys are exempt. Exempt rows are always counted."""
    cardinality: Literal["any", "many_to_one", "one_to_one"] = "any"
    """``many_to_one``: referenced keys must be unique in the parent (a join cannot
    multiply rows). ``one_to_one``: child keys must be unique too."""

    @property
    def target_columns(self) -> list[str]:
        """Referenced column names, resolved against the default."""
        return self.ref_columns or self.columns

    @model_validator(mode="after")
    def _same_arity(self) -> ForeignKeySpec:
        if self.ref_columns is not None and len(self.ref_columns) != len(self.columns):
            raise ValueError("ref_columns must have the same length as columns")
        for name, columns in (("columns", self.columns), ("ref_columns", self.ref_columns or [])):
            repeated = sorted({c for c in columns if columns.count(c) > 1})
            if repeated:
                raise ValueError(f"{name} repeats column(s): {', '.join(repeated)}")
        return self


class ExpressionCheck(_Strict):
    """A row-level SQL condition every row must satisfy, e.g. ``pm25 <= pm10``.

    Evaluated by DuckDB; the expression may use any SQL function and may
    reference other datasets by name in subqueries.
    """

    expr: str
    name: str | None = None
    severity: Severity | None = None
    description: str | None = None
    null_as: Literal["pass", "fail"] = "pass"
    """How a row whose expression evaluates to NULL counts (SQL CHECK semantics: pass)."""


class SqlAssertion(_Strict):
    """A DuckDB query that returns offending rows; it passes when it returns none.

    Every dataset is available as a table under its name::

        sql: SELECT county FROM inventory GROUP BY county HAVING sum(nox) = 0
    """

    name: str
    sql: str
    severity: Severity | None = None
    description: str | None = None


class SourcePolicy(_Strict):
    """Loading safeguards independent of the input's location or representation.

    Only explicitly supplied fields override a source's own policy. Changing a
    file into an in-memory frame keeps its row budget; retries apply only where
    a source declares that retrying is safe.
    """

    retries: int = Field(default=0, ge=0, le=10)
    """Extra attempts for explicitly retryable failures of a retry-safe source."""
    retry_delay_s: float = Field(default=1.0, ge=0, le=300)
    """Delay before the first retry, doubled on each additional attempt."""
    row_budget: int | None = Field(default=None, ge=1)
    """Maximum input result rows, preserved across representations. This does
    not authorize sampling and does not bound rows scanned by a source query."""

    @model_validator(mode="after")
    def _finite_delay(self) -> SourcePolicy:
        _require_finite("retry_delay_s", self.retry_delay_s)
        return self


class DatasetSpec(_Strict):
    """Where a dataset comes from and what it must look like."""

    path: str | None = None
    """Shorthand for ``source: {type: file, path: ...}``."""
    format: Literal["csv", "parquet", "json", "ndjson", "excel"] | None = None
    """With ``path``: the file format, inferred from the suffix when omitted."""
    source: dict[str, Any] | None = None
    """Full source specification, e.g. ``{type: query, sql: ...}``. See ``qavf schema``."""
    load_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    """Location-independent safeguards. Explicit values override policy fields
    inside ``source`` and are preserved when the public API replaces an input."""
    required: bool = True
    description: str | None = None
    rename: dict[str, str] = Field(default_factory=dict)
    """Column renames applied after loading (``old: new``)."""
    schema_spec: SchemaSpec | None = Field(default=None, alias="schema")
    foreign_keys: list[ForeignKeySpec] = Field(default_factory=list)
    expressions: list[ExpressionCheck] = Field(default_factory=list)

    @field_serializer("load_policy")
    def _serialize_load_policy(self, value: SourcePolicy) -> dict[str, Any]:
        # Preserve which policy fields override the source across settings round-trips.
        return value.model_dump(exclude_unset=True)

    @model_validator(mode="after")
    def _one_source(self) -> DatasetSpec:
        if self.path is not None and self.source is not None:
            raise ValueError("set either 'path' or 'source', not both")
        targets: dict[str, str] = {}
        for old, new in self.rename.items():
            other = targets.setdefault(new, old)
            if other != old:
                raise ValueError(
                    f"rename maps both {other!r} and {old!r} to {new!r}; each column needs "
                    f"its own name"
                )
        if self.format is not None and self.path is None:
            raise ValueError("'format' goes with 'path'; put it inside 'source' otherwise")
        return self

    def source_config(self) -> dict[str, Any] | None:
        """The effective source specification, expanding the ``path`` shorthand."""
        if self.path is not None:
            config: dict[str, Any] = {"type": "file", "path": self.path}
            if self.format is not None:
                config["format"] = self.format
        elif self.source is not None:
            config = dict(self.source)
        else:
            return None
        config.update(self.effective_load_policy().model_dump(exclude_unset=True))
        return config

    def effective_load_policy(self) -> SourcePolicy:
        """Combine legacy source policy with explicit dataset-level safeguards."""
        source = self.source or {}
        inherited = {name: source[name] for name in SourcePolicy.model_fields if name in source}
        return SourcePolicy.model_validate(
            {**inherited, **self.load_policy.model_dump(exclude_unset=True)}
        )


class ConnectionSpec(_Strict):
    """A database DuckDB attaches for ``database`` and ``query`` sources."""

    type: Literal["duckdb", "sqlite", "postgres", "mysql"]
    path: str | None = None
    """Database file for ``duckdb`` and ``sqlite``; relative to the configuration file."""
    uri: str | None = None
    """libpq/MySQL connection string for ``postgres`` and ``mysql``, e.g.
    ``host=localhost user=qa database=movesdb`` (use ``${ENV}`` for passwords)."""
    read_only: bool = True
    snapshot: bool = False
    """Read every dataset that comes from this connection together, on one connection
    in one transaction, so related tables reflect a single moment. Guaranteed for DuckDB
    files (MVCC); for other databases it is what DuckDB's extension transaction gives
    (see the documentation), so add a batch check (``REC-005``) where it matters."""

    @property
    def file_based(self) -> bool:
        """Whether the database is a local file (``path``) rather than a server (``uri``)."""
        return self.type in ("duckdb", "sqlite")

    @model_validator(mode="after")
    def _target(self) -> ConnectionSpec:
        if self.file_based:
            if not self.path:
                raise ValueError(f"a {self.type} connection needs 'path'")
            if self.uri is not None:
                raise ValueError(f"a {self.type} connection takes 'path', not 'uri'")
        else:
            if not self.uri:
                raise ValueError(f"a {self.type} connection needs 'uri'")
            if self.path is not None:
                raise ValueError(f"a {self.type} connection takes 'uri', not 'path'")
        return self

    def attach_sql(self, name: str, env: SourceEnvironment) -> str:
        """The DuckDB ``ATTACH`` statement for this connection."""
        from qavf.sources.sql import quote_identifier, sql_literal

        # The target is chosen by type, never by which field happens to be set.
        target = str(env.resolve(str(self.path))) if self.file_based else str(self.uri)
        options = [] if self.type == "duckdb" else [f"TYPE {self.type}"]
        if self.read_only:
            options.append("READ_ONLY")
        suffix = f" ({', '.join(options)})" if options else ""
        return f"ATTACH {sql_literal(target)} AS {quote_identifier(name)}{suffix}"


class SecretSpec(_Strict):
    """Cloud storage credentials for remote files (``s3://``, ``az://``) and SQL reads.

    Each secret becomes a DuckDB secret on every QAVF-owned connection and supplies
    the matching ``storage_options`` for remote Parquet read through Polars. Use
    ``${ENV}`` for credential values; reports and logs mask them.
    """

    type: Literal["s3", "azure"]
    """``s3`` for AWS S3 and S3-compatible stores (Cloudflare R2, MinIO, Wasabi, GCS
    interoperability); ``azure`` for Azure Blob Storage and ADLS Gen2."""
    provider: Literal["config", "credential_chain"] = "config"
    """``config`` uses the values given here; ``credential_chain`` uses the platform's
    own credentials (environment, profile, instance or workload identity, CLI login)."""
    scope: list[str] = Field(default_factory=list)
    """URL prefixes this secret applies to (``s3://bucket/path``). Unscoped secrets
    apply to every URL of their type; at most one per type may be unscoped."""
    key_id: str | None = None
    """S3 access key id (``config`` provider)."""
    secret: str | None = None
    """S3 secret access key (``config`` provider)."""
    session_token: str | None = None
    """Optional S3 session token for temporary credentials."""
    region: str | None = None
    """S3 region, e.g. ``us-east-1`` (``auto`` for Cloudflare R2)."""
    endpoint: str | None = None
    """S3-compatible endpoint as ``host[:port]`` without a scheme, e.g.
    ``<account>.r2.cloudflarestorage.com`` or ``storage.googleapis.com``."""
    url_style: Literal["vhost", "path"] | None = None
    """S3 addressing; most S3-compatible servers need ``path``."""
    use_ssl: bool = True
    """Use HTTPS for a custom S3 endpoint."""
    connection_string: str | None = None
    """Azure storage connection string (``config`` provider)."""
    account_name: str | None = None
    """Azure storage account name (``credential_chain`` provider)."""
    chain: str | None = None
    """Credential sources to try for ``credential_chain``, e.g. ``env;config`` (S3) or
    ``cli;managed_identity`` (Azure); omitted means the platform default order."""

    @model_validator(mode="after")
    def _complete(self) -> SecretSpec:
        s3_only = ("key_id", "secret", "session_token", "region", "endpoint", "url_style")
        azure_only = ("connection_string", "account_name")
        # Judge by values, not by which fields were set: serialized settings (for
        # example an isolated-worker request) state every field, defaults included.
        foreign = sorted(
            name
            for name in (azure_only if self.type == "s3" else s3_only)
            if getattr(self, name) is not None
        )
        if self.type == "azure" and not self.use_ssl:
            foreign.append("use_ssl")
        if foreign:
            raise ValueError(f"a {self.type} secret does not take {', '.join(foreign)}")
        chained = self.provider == "credential_chain"
        if self.chain is not None and not chained:
            raise ValueError("'chain' applies only to provider 'credential_chain'")
        if self.type == "s3":
            if chained and (self.key_id or self.secret or self.session_token):
                raise ValueError("credential_chain secrets take no key_id, secret or session_token")
            if not chained and not (self.key_id and self.secret):
                raise ValueError("an s3 secret with provider 'config' needs key_id and secret")
            if self.endpoint is not None and ("://" in self.endpoint or "/" in self.endpoint):
                raise ValueError("endpoint is host[:port] without a scheme or path")
        elif chained:
            if self.connection_string:
                raise ValueError(
                    "credential_chain secrets take account_name, not connection_string"
                )
            if not self.account_name:
                raise ValueError("an azure credential_chain secret needs account_name")
        elif not self.connection_string:
            raise ValueError("an azure secret with provider 'config' needs connection_string")
        for prefix in self.scope:
            if "://" not in prefix:
                raise ValueError(f"secret scope {prefix!r} must be a URL prefix such as s3://bucket")
        return self


# --------------------------------------------------------------------- engine
class RuleSettings(_Strict):
    """Per-rule overrides."""

    enabled: bool = True
    severity: Severity | None = None
    """Overrides the rule's default severity."""
    params: dict[str, Any] = Field(default_factory=dict)
    """Validated against the rule's own ``Params`` model at plan time."""

    @model_serializer(mode="wrap")
    def _preserve_parameter_presence(self, handler: SerializerFunctionWrapHandler) -> Any:
        """Omitted parameters and an explicitly empty mapping have distinct meaning.

        Disabled rules may omit required parameters. Keep that omission through
        normalization, while retaining explicit emptiness and in-place additions.
        """
        result = handler(self)
        if "params" not in self.model_fields_set and not self.params:
            result.pop("params", None)
        return result


class EngineSettings(_Strict):
    """Execution behaviour."""

    execution_mode: Literal["in_process", "isolated"] = "in_process"
    """Use a fresh supervised process (Linux, macOS, Windows) for transportable inputs."""
    run_timeout_s: float | None = Field(default=None, gt=0, strict=True)
    """Whole isolated worker wall time, including imports, discovery and serialization."""
    process_address_space_bytes: int | None = Field(default=None, ge=1, strict=True)
    """OS virtual address-space ceiling per isolated process; not a resident-memory cap."""
    process_cpu_time_s: int | None = Field(default=None, ge=1, strict=True)
    """OS CPU seconds per isolated process, summed over its threads."""
    process_file_bytes: int | None = Field(default=None, ge=0, strict=True)
    """OS maximum size of any individual file written by the isolated worker."""
    parallel: bool = True
    max_workers: int | None = Field(default=None, ge=1)
    """Thread count; defaults to ``min(8, cpu_count)``."""
    run_unconfigured_rules: bool = True
    """Run discovered rules that have no entry under ``rules`` with their defaults."""
    include_categories: list[str] | None = None
    exclude_categories: list[str] = Field(default_factory=list)
    max_affected_records: int = Field(default=50, ge=0)
    """Cap on the offending-record sample kept per result."""
    fail_on: Severity = Severity.ERROR
    """Lowest failure severity that makes the run (and CLI exit code) fail."""
    fail_on_incomplete: bool = True
    """Fail the run when it is incomplete: a configured check could not be assessed
    (missing data, invalid SQL, nothing testable), or no check assessed anything."""
    rule_timeout_s: float | None = Field(default=None, gt=0)
    """Cooperative deadline: overdue work is unassessed and cancellation is requested.
    Arbitrary Python/native plugin calls cannot be forcibly terminated by a thread;
    overdue calls remain part of the active-worker budget until they stop."""
    duckdb_memory_limit: str | None = None
    """Buffer-manager limit per DuckDB connection, not a process RAM limit. Some
    operators allocate outside it or cannot spill; resource exhaustion is an error."""
    memory_budget_bytes: int | None = Field(default=None, ge=1, strict=True)
    """Run target for tracked materializations and cache state, not an OS-enforced
    process RSS cap. Caller-owned inputs and opaque plugin allocations may exceed it."""
    cache_max_bytes: int = Field(default=268_435_456, ge=0, strict=True)
    """Maximum tracked retained cache bytes; zero disables optional caching."""
    spill_max_bytes: int = Field(default=1_073_741_824, ge=0, strict=True)
    """Run temporary-storage budget. Zero disables spilling; insufficient storage
    makes affected checks unassessed rather than reducing validation coverage."""
    temp_directory: Path | None = None
    """Parent of the run-owned temporary directory; relative to the configuration."""
    max_report_bytes: int = Field(default=16_777_216, ge=1, strict=True)
    """Maximum encoded report size; essential outcomes exceeding it raise a resource error."""
    max_evidence_bytes: int = Field(default=1_048_576, ge=0, strict=True)
    """Maximum serialized evidence/diagnostic payload bytes per result. Truncation
    is disclosed and never changes exact violation counts or assessment status."""
    native_threads: int | None = Field(default=None, ge=1, strict=True)
    """Native execution thread allowance used where the backend supports it.
    Polars process-global settings must be configured before its pool initializes."""
    timeout_grace_s: float = Field(default=0.1, ge=0)
    """Bounded grace after cancellation; this is not a hard process-kill deadline."""
    concurrent_loads: bool = True
    """Load independent datasets concurrently when rules run in parallel. Turn off
    when a backend cannot take several queries at once or loads must not overlap."""
    require: list[str] = Field(default_factory=list)
    """Checks that must actually be assessed, as rule ids (``INP-004``) or single
    checks (``INP-003[counts.aadt: ge 0]``). A required check that is disabled,
    filtered out, skipped or not assessed makes the run incomplete."""
    canonical_columns: list[str] | str = Field(default_factory=list)
    """Column spellings to normalise to case-insensitively on load (``sourcetypeid``
    becomes ``sourceTypeID``). Either a list or an import reference such as
    ``my_plugin:COLUMNS`` to a list defined by a plugin."""

    @model_validator(mode="after")
    def _distinct_canonical(self) -> EngineSettings:
        for name in ("rule_timeout_s", "timeout_grace_s", "run_timeout_s"):
            _require_finite(name, getattr(self, name))
        if any(
            value is not None
            for value in (
                self.run_timeout_s, self.process_address_space_bytes,
                self.process_cpu_time_s, self.process_file_bytes,
            )
        ):
            if "execution_mode" in self.model_fields_set and self.execution_mode != "isolated":
                raise ValueError("process execution limits require execution_mode='isolated'")
            self.execution_mode = "isolated"
        if isinstance(self.canonical_columns, list):
            check_canonical_columns(self.canonical_columns)
        return self

    def resolved_canonical_columns(self) -> list[str]:
        """:attr:`canonical_columns` with an import reference resolved."""
        if isinstance(self.canonical_columns, list):
            return self.canonical_columns
        module, _, attribute = self.canonical_columns.partition(":")
        if not attribute:
            raise ValueError("canonical_columns must be a list or 'module:attribute'")
        return list(getattr(importlib.import_module(module), attribute))


class PluginSettings(_Strict):
    """Where to discover rules and sources."""

    packages: list[str] = Field(default_factory=lambda: ["qavf.rules"])
    """Importable packages whose modules are scanned for rules."""
    modules: list[str] = Field(default_factory=list)
    """Additional importable modules scanned for rules and sources."""
    paths: list[Path] = Field(default_factory=list)
    """Stand-alone ``.py`` files (or directories of them) scanned for rules."""
    entry_points: bool = True
    """Also load plugins advertised under the ``qavf.rules`` / ``qavf.sources`` groups."""
    required: list[str] = Field(default_factory=list)
    """Required entry-point names, or qualified names such as ``rules:moves``.
    Every advertised component of an unqualified name must load; at least one
    must exist. Missing or broken components stop the run with a configuration error."""


class ReportSettings(_Strict):
    """Report outputs."""

    title: str = "QA Report"
    output_dir: Path = Path("qa_output")
    html: bool = True
    html_max_cells: int = Field(default=10_000, ge=1, strict=True)
    html_max_columns: int = Field(default=64, ge=1, strict=True)
    html_max_bytes: int = Field(default=16_777_216, ge=1, strict=True)
    json_: bool = Field(default=True, alias="json")
    hash_inputs: bool = False
    """Record a SHA-256 digest of each local input file in the report's provenance
    (size and modification time are always recorded). Whole-file content checks
    surround each complete read operation and run again at finalization. Nested
    helpers share their operation's checks; separate operations always rehash.
    This verifies boundary consistency, not an immutable snapshot. Per-source
    provenance notes disclose actual hash passes and bytes read."""


class QAVFSettings(_Strict):
    """Root configuration object."""

    project: str = "QA run"
    datasets: dict[str, DatasetSpec] = Field(default_factory=dict)
    connections: dict[str, ConnectionSpec] = Field(default_factory=dict)
    secrets: dict[str, SecretSpec] = Field(default_factory=dict)
    """Cloud storage credentials, keyed by a SQL-identifier name."""
    assertions: list[SqlAssertion] = Field(default_factory=list)
    rules: dict[str, RuleSettings] = Field(default_factory=dict)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Free-form settings for plugins, keyed by plugin name (e.g. ``moves: {version: MOVES4}``)."""

    source_path: Path | None = Field(default=None, exclude=True)
    """File this configuration was loaded from; set by :func:`load_settings`."""

    @model_validator(mode="before")
    @classmethod
    def _string_mapping_keys(cls, data: Any) -> Any:
        validate_config_mapping_keys(data)
        return data

    @model_validator(mode="after")
    def _references_resolve(self) -> QAVFSettings:
        check_sql_names(self.datasets, "dataset")
        check_sql_names(self.connections, "connection")
        check_sql_names(self.secrets, "secret")
        unscoped: dict[str, str] = {}
        for name, secret in self.secrets.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"secret name {name!r} must be a plain SQL identifier")
            if not secret.scope:
                other = unscoped.setdefault(secret.type, name)
                if other != name:
                    raise ValueError(
                        f"secrets {other!r} and {name!r} are both unscoped {secret.type} "
                        "secrets; give at least one a scope so each URL has one credential"
                    )
        for name, spec in self.datasets.items():
            for fk in spec.foreign_keys:
                if fk.ref_dataset not in self.datasets:
                    raise ValueError(
                        f"dataset {name!r}: foreign key references unknown dataset "
                        f"{fk.ref_dataset!r}"
                    )
            source = spec.source or {}
            connection = source.get("connection")
            if isinstance(connection, str) and connection not in self.connections:
                raise ValueError(f"dataset {name!r}: unknown connection {connection!r}")
            attached = source.get("connections")
            if isinstance(attached, list):
                if not all(isinstance(item, str) for item in attached):
                    raise ValueError(f"dataset {name!r}: source connections must be names")
                if len(attached) != len(set(attached)):
                    raise ValueError(f"dataset {name!r}: duplicate source connections")
                unknown = set(attached) - self.connections.keys()
                if unknown:
                    raise ValueError(f"dataset {name!r}: unknown connections {sorted(unknown)!r}")
        names = [assertion.name for assertion in self.assertions]
        if len(names) != len(set(names)):
            raise ValueError("assertion names must be unique")
        return self

    @property
    def base_dir(self) -> Path:
        """Directory that relative paths in the configuration resolve against."""
        return self.source_path.parent if self.source_path else Path.cwd()

    def resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` against :attr:`base_dir` unless it is already absolute."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.base_dir / candidate).resolve()


def validate_config_mapping_keys(value: Any) -> None:
    """Reject key coercions that could give distinct runtime settings one digest.

    Traverse raw mappings and existing models without serialization: serialization
    may already have coerced a key. Locations use entry indexes below known model
    fields so diagnostics never reveal arbitrary plugin keys or their values.
    """
    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if not isinstance(item, (BaseModel, Mapping, list, tuple)) or id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, BaseModel):
            for name in type(item).model_fields:
                visit(getattr(item, name), f"{path}.{name}")
        elif isinstance(item, Mapping):
            for index, (key, child) in enumerate(item.items()):
                if not isinstance(key, str):
                    raise ValueError(
                        "configuration mapping keys must be strings "
                        f"at {path} (key entry {index})"
                    )
                child_path = (
                    f"{path}.{key}"
                    if path == "configuration" and key in QAVFSettings.model_fields
                    else f"{path}[value {index}]"
                )
                visit(child, child_path)
        else:
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "configuration")


def check_canonical_columns(names: Iterable[str]) -> dict[str, str]:
    """Map each lower-cased canonical spelling to the spelling itself.

    Raises:
        ValueError: if two spellings differ only in case, which would make the
            target of a case-insensitive match ambiguous.
    """
    canonical: dict[str, str] = {}
    for name in names:
        other = canonical.setdefault(name.lower(), name)
        if other != name:
            raise ValueError(
                f"canonical_columns lists both {other!r} and {name!r}, which differ only in "
                f"case; keep one spelling"
            )
    return canonical


def check_sql_names(names: Iterable[str], kind: str) -> None:
    """Reject names that SQL would treat as one identifier (DuckDB is case-insensitive).

    Raises:
        ValueError: naming the colliding names.
    """
    seen: dict[str, str] = {}
    for name in names:
        other = seen.setdefault(name.lower(), name)
        if other != name:
            raise ValueError(
                f"{kind} names {other!r} and {name!r} differ only in case; SQL treats them as "
                f"the same table, so one would silently replace the other. Rename one."
            )


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that rejects duplicate mapping keys instead of keeping the last."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self.validated_nodes: set[int] = set()


def _validate_yaml_keys(loader: _UniqueKeyLoader, node: yaml.Node) -> None:
    """Check raw mappings before merge flattening, including referenced anchors.

    An explicit override of an inherited merge value is legal YAML; two explicit
    occurrences in the same mapping are not. Checking before flattening preserves
    that distinction, including when an anchor is consumed before its own mapping.
    """
    if id(node) in loader.validated_nodes:
        return
    loader.validated_nodes.add(id(node))
    if isinstance(node, yaml.MappingNode):
        first: dict[Any, tuple[int, int]] = {}
        for key_node, value_node in node.value:
            key = (
                "<<"
                if key_node.tag == "tag:yaml.org,2002:merge"
                else loader.construct_object(key_node, deep=False)
            )
            location = (key_node.start_mark.line + 1, key_node.start_mark.column + 1)
            try:
                if key in first:
                    earlier = first[key]
                    raise ValueError(
                        f"duplicate key {key!r} at line {location[0]}, column {location[1]} "
                        f"(first defined at line {earlier[0]}, column {earlier[1]}); "
                        "a repeated key would silently replace the earlier entry"
                    )
                first[key] = location
            except TypeError:
                pass  # PyYAML supplies the precise invalid-mapping-key error.
            _validate_yaml_keys(loader, value_node)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _validate_yaml_keys(loader, child)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> Any:
    _validate_yaml_keys(loader, node)
    loader.flatten_mapping(node)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"duplicate key {key!r}; a repeated key would silently replace the earlier entry"
            )
        result[key] = value
    return result


def _describe_yaml_error(exc: yaml.YAMLError) -> str:
    """One-line description of a YAML error with its 1-based line and column."""
    problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    mark = getattr(exc, "problem_mark", None)
    if mark is None:
        return str(problem)
    return f"{problem} (line {mark.line + 1}, column {mark.column + 1})"


def expand_env(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    """Recursively replace ``${NAME}`` / ``${NAME:-default}`` in strings.

    Raises:
        ValueError: for a reference to an unset variable without a default.
    """
    env = os.environ if environ is None else environ
    if isinstance(value, str):

        def substitute(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ValueError(f"environment variable {name!r} is not set")

        return _ENV_REFERENCE.sub(substitute, value)
    if isinstance(value, dict):
        return {key: expand_env(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item, env) for item in value]
    return value


def _read_config_mapping(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Parse one configuration document before callers assemble inline contracts."""
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8")
    match source.suffix.lower():
        case ".yaml" | ".yml":
            try:
                raw = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
            except yaml.YAMLError as exc:
                raise ValueError(f"{source}: invalid YAML: {_describe_yaml_error(exc)}") from exc
        case ".json":
            try:
                raw = json.loads(text, object_pairs_hook=_unique_json_object)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}: invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
                ) from exc
        case other:
            raise ValueError(f"unsupported configuration format {other!r}; use YAML or JSON")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top level of the configuration must be a mapping")
    raw.pop("$schema", None)
    return expand_env(raw), source


def load_settings(path: str | Path) -> QAVFSettings:
    """Load and validate a YAML (``.yaml``/``.yml``) or JSON configuration file.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: for an unknown suffix, a non-mapping document or an unset
            environment variable.
        pydantic.ValidationError: if the content violates the schema.
    """
    raw, source = _read_config_mapping(path)
    settings = QAVFSettings.model_validate(raw)
    settings.source_path = source
    logger.info("Loaded configuration from %s", source)
    return settings
