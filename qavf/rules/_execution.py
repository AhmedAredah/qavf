"""Small execution adapters for projected rule inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.sql import missing_sql
from qavf.keys import KeyTypeError, key_casts
from qavf.rules._key_summary import KeySelection, KeySummary, summarize_keys
from qavf.rules.helpers import missing_expr
from qavf.sources.sql import quote_identifier

type KeyBackend = Literal["native_sql", "polars"]


def paired_key_backend(ctx: DataContext, left: str, right: str) -> KeyBackend | None:
    """Select one backend for a paired operation, or its mixed-input fallback.

    Another rule may materialize an input after this decision. Both SQL and
    Polars plans can still read that materialized snapshot, so callers retain
    this choice for both relations instead of reclassifying each independently.
    """
    left_native = ctx.execution_kind(left) == "native_sql"
    right_native = ctx.execution_kind(right) == "native_sql"
    if left_native != right_native:
        return None
    return "native_sql" if left_native else "polars"


def _native_key_type(dtype: pl.DataType) -> str:
    """SQL casts used for exact alignment, including an empty Null-typed side."""
    native: dict[pl.DataType, str] = {
        pl.Int8(): "TINYINT",
        pl.Int16(): "SMALLINT",
        pl.Int32(): "INTEGER",
        pl.Int64(): "BIGINT",
        pl.Int128(): "HUGEINT",
        pl.UInt8(): "UTINYINT",
        pl.UInt16(): "USMALLINT",
        pl.UInt32(): "UINTEGER",
        pl.UInt64(): "UBIGINT",
        pl.Float32(): "FLOAT",
        pl.Float64(): "DOUBLE",
        pl.String(): "VARCHAR",
        pl.Boolean(): "BOOLEAN",
        pl.Binary(): "BLOB",
        pl.Date(): "DATE",
        pl.Time(): "TIME",
    }
    if dtype in native:
        return native[dtype]
    if isinstance(dtype, pl.Decimal):
        return f"DECIMAL({dtype.precision or 38}, {dtype.scale})"
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        return {"ns": "TIMESTAMP_NS", "us": "TIMESTAMP", "ms": "TIMESTAMP_MS"}[dtype.time_unit]
    raise KeyTypeError(f"SQL cannot align key type {dtype} losslessly; use a typed source")


@dataclass(frozen=True)
class KeyRelation:
    """Grouped keys retained in their execution backend until summary collection."""

    ctx: DataContext
    datasets: tuple[str, ...]
    columns: tuple[str, ...]
    frequency: str
    schema: Mapping[str, pl.DataType]
    sql: str | None = None
    lazy: pl.LazyFrame | None = None
    normalize_missing: bool = True
    collector: Callable[[pl.LazyFrame], pl.DataFrame] | None = None

    def collect(self, plan: pl.LazyFrame) -> pl.DataFrame:
        return (
            self.collector(plan)
            if self.collector is not None
            else self.ctx.collect_many(self.datasets, plan)
        )

    def filter(self, sql: str, predicate: pl.Expr) -> KeyRelation:
        if self.sql is not None:
            return replace(self, sql=f"SELECT * FROM ({self.sql}) AS keys WHERE {sql}")
        assert self.lazy is not None
        return replace(self, lazy=self.lazy.filter(predicate))

    def summarize(self, limit: int) -> KeySummary:
        """Count, weight and sample one coherent evaluation, including empty input."""
        return self.summarize_many({"all": KeySelection(limit=limit)})["all"]

    def summarize_many(self, selections: Mapping[str, KeySelection]) -> dict[str, KeySummary]:
        """Share grouped-key work between related counts and bounded samples."""
        return summarize_keys(self, selections)

    def rename(self, mapping: Mapping[str, str]) -> KeyRelation:
        schema = {mapping.get(name, name): dtype for name, dtype in self.schema.items()}
        columns = tuple(mapping.get(name, name) for name in self.columns)
        frequency = mapping.get(self.frequency, self.frequency)
        if self.sql is not None:
            selected = ", ".join(
                f"{quote_identifier(name)} AS {quote_identifier(mapping.get(name, name))}"
                for name in self.schema
            )
            return replace(
                self,
                columns=columns,
                frequency=frequency,
                schema=schema,
                sql=f"SELECT {selected} FROM ({self.sql}) AS keys",
            )
        assert self.lazy is not None
        return replace(
            self,
            columns=columns,
            frequency=frequency,
            schema=schema,
            lazy=self.lazy.rename(mapping),
        )

    def cast(self, types: Mapping[str, pl.DataType]) -> KeyRelation:
        schema = {**self.schema, **types}
        if self.sql is not None:
            selected = ", ".join(
                f"CAST({quote_identifier(name)} AS {_native_key_type(types[name])}) "
                f"AS {quote_identifier(name)}"
                if name in types
                else quote_identifier(name)
                for name in self.schema
            )
            return replace(self, schema=schema, sql=f"SELECT {selected} FROM ({self.sql}) AS keys")
        assert self.lazy is not None
        return replace(
            self,
            schema=schema,
            lazy=self.lazy.with_columns(pl.col(name).cast(dtype) for name, dtype in types.items()),
        )

    def anti(self, other: KeyRelation) -> KeyRelation:
        if (self.sql is None) != (other.sql is None):
            raise ValueError("key relations use different execution backends")
        datasets = tuple(dict.fromkeys((*self.datasets, *other.datasets)))
        if self.sql is not None:
            on = " AND ".join(
                f"l.{quote_identifier(name)} IS NOT DISTINCT FROM r.{quote_identifier(name)}"
                for name in self.columns
            )
            return replace(
                self,
                datasets=datasets,
                sql=(f"SELECT l.* FROM ({self.sql}) AS l ANTI JOIN ({other.sql}) AS r ON {on}"),
            )
        assert self.lazy is not None and other.lazy is not None
        return replace(
            self,
            datasets=datasets,
            lazy=self.lazy.join(
                other.lazy.select(other.columns),
                on=self.columns,
                how="anti",
                nulls_equal=True,
            ),
        )

    def exact_float_keys(self, name: str) -> bool:
        """Prove cross integer/float alignment using one scalar aggregate."""
        if self.sql is not None:
            column_sql = quote_identifier(name)
            valid_sql = f"abs({column_sql}) <= 9007199254740992"
            if self.schema[name].is_float():
                valid_sql += f" AND isfinite({column_sql}) AND floor({column_sql}) = {column_sql}"
            return bool(
                self.ctx.sql(
                    f"SELECT coalesce(bool_and({valid_sql}), true) FROM ({self.sql}) AS keys",
                    datasets=self.datasets,
                    normalize_missing=self.normalize_missing,
                ).item()
            )
        assert self.lazy is not None
        column = pl.col(name)
        valid = (column >= -(2**53)) & (column <= 2**53)
        if self.schema[name].is_float():
            valid &= column.is_finite() & (column == column.round(0))
        return bool(self.collect(self.lazy.select(valid.all())).item())


def align_key_relations(left: KeyRelation, right: KeyRelation) -> tuple[KeyRelation, KeyRelation]:
    """Lossless key alignment without collecting high-cardinality key tables."""
    a_types: dict[str, pl.DataType] = {}
    b_types: dict[str, pl.DataType] = {}
    for name in left.columns:

        def exact_float(key: str = name) -> bool:
            return left.exact_float_keys(key) and right.exact_float_keys(key)

        a, b = key_casts(
            name,
            left.schema[name],
            right.schema[name],
            exact_float=exact_float,
        )
        if a is not None:
            a_types[name] = a
        if b is not None:
            b_types[name] = b
    return left.cast(a_types) if a_types else left, right.cast(b_types) if b_types else right


def key_relation(
    ctx: DataContext,
    dataset: str,
    columns: Iterable[str],
    frequency: str,
    *,
    normalize_missing: bool = True,
    backend: KeyBackend | None = None,
    _snapshot: pl.DataFrame | None = None,
) -> KeyRelation:
    """Group keys at the source using a fixed paired backend when supplied."""
    keys = tuple(columns)
    schema = {name: ctx.schema(dataset)[name] for name in keys}
    result_schema = {**schema, frequency: pl.UInt64()}
    selected_backend = (
        "polars"
        if _snapshot is not None
        else backend or ("native_sql" if ctx.execution_kind(dataset) == "native_sql" else "polars")
    )
    if selected_backend == "native_sql":
        selected = ", ".join(quote_identifier(name) for name in keys)
        source = quote_identifier(dataset)
        if normalize_missing:
            spec = ctx.specs.get(dataset)
            contracts = spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
            normalized = ", ".join(
                f"CASE WHEN {missing_sql(name, schema[name], contracts.get(name))} THEN NULL "
                f"ELSE {quote_identifier(name)} END AS {quote_identifier(name)}"
                for name in keys
            )
            source = f"(SELECT {normalized} FROM {source}) AS normalized_keys"
        return KeyRelation(
            ctx,
            (dataset,),
            keys,
            frequency,
            result_schema,
            sql=(
                f"SELECT {selected}, count(*)::UBIGINT AS {quote_identifier(frequency)} "
                f"FROM {source} GROUP BY {selected}"
            ),
            # Only this key's columns are normalized above. A malformed missing
            # policy on an unrelated field must not poison this contract.
            normalize_missing=False,
        )
    plan = (_snapshot.lazy() if _snapshot is not None else ctx.scan(dataset)).select(keys)
    if normalize_missing:
        spec = ctx.specs.get(dataset)
        columns_spec = spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
        empty = pl.DataFrame(schema=schema)
        plan = plan.with_columns(
            pl.when(missing_expr(empty, name, columns_spec.get(name)))
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
            for name in keys
        )
    return KeyRelation(
        ctx,
        (dataset,),
        keys,
        frequency,
        result_schema,
        lazy=(
            plan.group_by(keys, maintain_order=True).agg(pl.len().cast(pl.UInt64).alias(frequency))
        ),
        normalize_missing=normalize_missing,
    )


def projected_frame(ctx: DataContext, dataset: str, columns: Iterable[str]) -> pl.DataFrame:
    """Materialize only required columns while retaining context execution policy."""
    return ctx.project(dataset, columns)


def normalized_plan(
    ctx: DataContext, dataset: str, plan: pl.LazyFrame, *, declared_only: bool = False
) -> pl.LazyFrame:
    """Apply a source's missing-value contract to an already scoped observation.

    ``declared_only`` normalizes only columns declaring ``missing_values`` or
    ``blank_as_missing``; other NaN values then remain observable values, as an
    exact value comparison needs.
    """
    schema = dict(plan.collect_schema())
    spec = ctx.specs.get(dataset)
    contracts = spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
    empty = pl.DataFrame(schema=schema)
    selected = [
        name
        for name in schema
        if not declared_only
        or (
            name in contracts
            and (bool(contracts[name].missing_values) or contracts[name].blank_as_missing)
        )
    ]
    return plan.with_columns(
        pl.when(missing_expr(empty, name, contracts.get(name)))
        .then(None)
        .otherwise(pl.col(name))
        .alias(name)
        for name in selected
    )


def observed_keys(
    ctx: DataContext,
    dataset: str,
    plan: pl.LazyFrame,
    columns: Iterable[str],
    frequency: str,
    *,
    collector: Callable[[pl.LazyFrame], pl.DataFrame],
) -> KeyRelation:
    """Group one coherent projected observation without collecting its key domain."""
    keys = tuple(columns)
    plan = normalized_plan(ctx, dataset, plan.select(keys))
    schema = dict(plan.collect_schema())
    return KeyRelation(
        ctx,
        (dataset,),
        keys,
        frequency,
        {**schema, frequency: pl.UInt64()},
        lazy=plan.group_by(keys, maintain_order=True).agg(
            pl.len().cast(pl.UInt64).alias(frequency)
        ),
        collector=collector,
    )


def key_frequencies(
    ctx: DataContext, dataset: str, columns: Iterable[str], frequency: str
) -> pl.DataFrame:
    """Reduce keys at the source; retain missing keys for the caller's own policy."""
    keys = list(columns)
    kind = ctx.execution_kind(dataset)
    if kind == "native_sql":
        selected = ", ".join(quote_identifier(column) for column in keys)
        return ctx.sql(
            f"SELECT {selected}, count(*)::UBIGINT AS {quote_identifier(frequency)} "
            f"FROM {quote_identifier(dataset)} GROUP BY {selected}"
        )
    return ctx.reduce(
        dataset,
        lambda frame: (
            frame.select(keys)
            .group_by(keys, maintain_order=True)
            .agg(pl.len().cast(pl.UInt64).alias(frequency))
        ),
    )
