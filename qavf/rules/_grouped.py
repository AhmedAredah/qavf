"""Projected, missing-normalized relations and exact grouped reductions.

Only reduced results and bounded samples cross the context collection boundary.
Unsupported numerical kernels return ``None`` so callers retain their exact
projected fallback rather than substitute an approximate backend reduction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction

import polars as pl

from qavf.aggregation import checked_accumulator_dtype, narrow_integer, numeric_expr
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.rules._exact_grouped import exact_grouped
from qavf.rules._execution import normalized_plan
from qavf.rules.helpers import Aggregate, fresh_internal, missing_expr
from qavf.sources.sql import quote_identifier
from qavf.statistics.ordered import linear_quantile, quantile_ranks


@dataclass(frozen=True)
class ProjectedRelation:
    ctx: DataContext
    dataset: str
    schema: dict[str, pl.DataType]
    lazy: pl.LazyFrame | None = None
    sql: str | None = None
    observation: Observation | None = None
    collector: Callable[[pl.LazyFrame], pl.DataFrame] | None = None

    def collect_plan(self, plan: pl.LazyFrame) -> pl.DataFrame:
        return (
            self.collector(plan)
            if self.collector is not None
            else self.ctx.collect(self.dataset, plan)
        )

    def collect(self) -> pl.DataFrame:
        if self.sql is not None:
            return self.ctx.sql(self.sql, datasets=(self.dataset,))
        assert self.lazy is not None
        return self.collect_plan(self.lazy)

    def filter(self, predicate: pl.Expr, sql: str) -> ProjectedRelation:
        if self.sql is not None:
            return replace(self, sql=f"SELECT * FROM ({self.sql}) AS selected WHERE {sql}")
        assert self.lazy is not None
        return replace(self, lazy=self.lazy.filter(predicate))

    def without_groups(
        self, predicate: pl.Expr, sql: str, keys: Sequence[str]
    ) -> ProjectedRelation:
        """Exclude every group containing a selected row, with null-equal keys."""
        if self.sql is not None:
            joined = (
                " AND ".join(
                    f"good.{quote_identifier(key)} IS NOT DISTINCT FROM bad.{quote_identifier(key)}"
                    for key in keys
                )
                or "TRUE"
            )
            return replace(
                self,
                sql=(
                    f"WITH input AS MATERIALIZED ({self.sql}), "
                    f"bad AS (SELECT * FROM input WHERE {sql}) "
                    f"SELECT good.* FROM input AS good WHERE NOT EXISTS "
                    f"(SELECT 1 FROM bad WHERE {joined})"
                ),
            )
        assert self.lazy is not None
        if not keys:
            return replace(self, lazy=self.lazy.filter(~predicate.any()))
        return replace(
            self,
            lazy=self.lazy.join(
                self.lazy.filter(predicate).select(keys).unique(),
                on=list(keys),
                how="anti",
                nulls_equal=True,
            ),
        )

    def count_sample(self, limit: int) -> tuple[int, pl.DataFrame]:
        """Count every selected row while collecting at most ``limit`` rows."""
        if limit <= 0:
            return self.count(), pl.DataFrame(schema=self.schema)
        if self.sql is not None:
            frame, count = self.ctx.sql_sample(self.sql, limit, ordered=True)
            return count, frame
        assert self.lazy is not None
        name = fresh_internal("sample_count", self.schema)
        frame = self.collect_plan(self.lazy.with_columns(pl.len().alias(name)).head(limit))
        count = int(frame[name][0]) if frame.height else 0
        return count, frame.drop(name)

    def count(self) -> int:
        if self.sql is not None:
            return int(
                self.ctx.sql(
                    f"SELECT count(*) FROM ({self.sql}) AS selected", datasets=(self.dataset,)
                ).item()
            )
        assert self.lazy is not None
        return int(self.collect_plan(self.lazy.select(pl.len())).item())

    def summary(self, expressions: Iterable[pl.Expr], sql: str) -> pl.DataFrame:
        """Collect caller-specific scalar metadata under the same source policy."""
        if self.sql is not None:
            return self.ctx.sql(
                f"SELECT {sql} FROM ({self.sql}) AS selected", datasets=(self.dataset,)
            )
        assert self.lazy is not None
        return self.collect_plan(self.lazy.select(expressions))

    def distinct(self, columns: Sequence[str]) -> ProjectedRelation:
        schema = {name: self.schema[name] for name in columns}
        if self.sql is not None:
            projection = ", ".join(map(quote_identifier, columns))
            return replace(
                self,
                schema=schema,
                sql=f"SELECT DISTINCT {projection} FROM ({self.sql}) AS selected",
            )
        assert self.lazy is not None
        return replace(
            self, schema=schema, lazy=self.lazy.select(columns).unique(maintain_order=True)
        )


@contextmanager
def scoped_relation(
    ctx: DataContext, dataset: str, columns: Iterable[str]
) -> Iterator[ProjectedRelation]:
    """A coherent, normalized projected relation for exact multipass kernels."""
    with ctx.observe(dataset, columns) as observation:
        plan = normalized_plan(ctx, dataset, observation.scan())
        yield ProjectedRelation(
            ctx,
            dataset,
            dict(plan.collect_schema()),
            lazy=plan,
            observation=observation,
            collector=observation.collect,
        )


def normalized_relation(
    ctx: DataContext, dataset: str, columns: Iterable[str], *, require_repeatable: bool = True
) -> ProjectedRelation | None:
    """Build a selected relation, guarding multi-pass work against changing observations.

    Arbitrary lazy/query/plugin plans retain their coherent projected fallback.
    A caller performing exactly one reduction may disable the repeatability gate.
    """
    kind = ctx.execution_kind(dataset)
    if kind not in {"lazy", "native_sql"}:
        return None
    if require_repeatable and not ctx.repeatable_scan(dataset):
        return None
    selected = list(dict.fromkeys(columns))
    available = ctx.schema(dataset)
    schema = {name: available[name] for name in selected}
    if not selected:
        marker = fresh_internal("counted_row", available)
        schema = {marker: pl.UInt8()}
        if kind == "native_sql":
            return ProjectedRelation(
                ctx,
                dataset,
                schema,
                sql=(
                    f"SELECT 1::UTINYINT AS {quote_identifier(marker)} "
                    f"FROM {quote_identifier(dataset)}"
                ),
            )
        return ProjectedRelation(
            ctx,
            dataset,
            schema,
            lazy=ctx.scan(dataset)
            .with_columns(pl.lit(1, dtype=pl.UInt8).alias(marker))
            .select(marker),
        )
    if kind == "native_sql":
        projection = ctx.sql_missing_projection(dataset, schema)
        return ProjectedRelation(
            ctx, dataset, schema, sql=f"SELECT {projection} FROM {quote_identifier(dataset)}"
        )
    spec = ctx.specs.get(dataset)
    contracts = spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
    empty = pl.DataFrame(schema=schema)
    plan = (
        ctx.scan(dataset)
        .select(selected)
        .with_columns(
            pl.when(missing_expr(empty, name, contracts.get(name)))
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
            for name in selected
        )
    )
    return ProjectedRelation(ctx, dataset, schema, lazy=plan)


def duplicate_groups(relation: ProjectedRelation, keys: Sequence[str]) -> int:
    """Count repeated groups without collecting their keys or member rows."""
    if relation.sql is not None:
        groups = ", ".join(map(quote_identifier, keys))
        grouping = f" GROUP BY {groups}" if keys else ""
        query = (
            "SELECT count(*) FROM (SELECT count(*) AS n "
            f"FROM ({relation.sql}) AS selected{grouping} HAVING count(*) > 1) AS repeated"
        )
        return int(relation.ctx.sql(query, datasets=(relation.dataset,)).item())
    assert relation.lazy is not None
    if not keys:
        plan = relation.lazy.select((pl.len() > 1).cast(pl.UInt64))
    else:
        name = fresh_internal("group_count", relation.schema)
        plan = (
            relation.lazy.group_by(list(keys))
            .agg(pl.len().alias(name))
            .filter(pl.col(name) > 1)
            .select(pl.len())
        )
    return int(relation.collect_plan(plan).item())


def supports_grouped(relation: ProjectedRelation, value: str | None, how: Aggregate) -> bool:
    """Whether the established numerical semantics have a reduced backend path."""
    dtype = relation.schema[value] if value is not None else None
    return how in {"sum", "count", "min", "max", "none"} and not (
        how == "sum" and dtype is not None and dtype.is_float()
    )


def grouped_aggregate(
    relation: ProjectedRelation,
    keys: Sequence[str],
    value: str | None,
    alias: str,
    how: Aggregate = "sum",
    *,
    presence_alias: str | None = None,
    missing_alias: str | None = None,
    bounds: tuple[object, object, int] | None = None,
) -> pl.DataFrame | None:
    """Reduce safe numeric kernels, retaining null-only and overflow semantics.

    ``none`` requires the caller to reject duplicate groups first. Float sums,
    means, weighted means, and ordered first/last use separate scoped reducers
    at their callers. This helper does not approximate them.
    """
    dtype = relation.schema[value] if value is not None else None
    if value is None and how != "count":
        raise ValueError("an aggregate without a value must count rows")
    if dtype is not None and not dtype.is_numeric() and dtype != pl.Null:
        raise ValueError(f"aggregation requires numeric values, found {dtype}")
    if not supports_grouped(relation, value, how):
        return None
    column = pl.col(value) if value is not None else pl.lit(1)
    quoted = quote_identifier(value) if value is not None else "*"
    if how == "sum" and dtype is not None and (dtype.is_integer() or dtype.is_decimal()):
        if bounds is not None:
            minimum, maximum, count = bounds
        elif relation.sql is not None:
            minimum, maximum, count = relation.ctx.sql(
                f"SELECT min({quoted}), max({quoted}), count(*) FROM ({relation.sql}) AS selected",
                datasets=(relation.dataset,),
            ).row(0)
        else:
            assert relation.lazy is not None
            minimum, maximum, count = relation.collect_plan(
                relation.lazy.select(
                    column.min().alias("minimum"), column.max().alias("maximum"), pl.len()
                ),
            ).row(0)
        storage = checked_accumulator_dtype(dtype, minimum, maximum, int(count))
        column = column.cast(storage)
        sql_type = f"DECIMAL(38,{storage.scale})" if isinstance(storage, pl.Decimal) else "HUGEINT"
        quoted = f"CAST({quoted} AS {sql_type})"
    expressions = {
        "sum": pl.when(column.count() > 0).then(column.sum()).otherwise(None),
        "count": (column.count() if value is not None else pl.len()).cast(pl.UInt64),
        "min": column.min(),
        "max": column.max(),
        "none": column.first(),
    }
    if relation.sql is not None:
        reductions = {
            "sum": f"CASE WHEN count({quoted}) > 0 THEN sum({quoted}) ELSE NULL END",
            "count": f"count({quoted})::UBIGINT",
            "min": f"min({quoted})",
            "max": f"max({quoted})",
            "none": f"first({quoted})",
        }
        selected = ", ".join(map(quote_identifier, keys))
        projection = f"{selected}, " if keys else ""
        grouping = f" GROUP BY {selected}" if keys else ""
        presence = (
            f", count(*)::UBIGINT AS {quote_identifier(presence_alias)}" if presence_alias else ""
        )
        if missing_alias:
            presence += (
                f", (count(*) - count({quoted}))::UBIGINT AS {quote_identifier(missing_alias)}"
            )
        result = relation.ctx.sql(
            f"SELECT {projection}{reductions[how]} AS {quote_identifier(alias)}{presence} "
            f"FROM ({relation.sql}) AS selected{grouping}",
            datasets=(relation.dataset,),
        )
    else:
        assert relation.lazy is not None
        expression = [expressions[how].alias(alias)]
        if presence_alias:
            expression.append(pl.len().cast(pl.UInt64).alias(presence_alias))
        if missing_alias:
            expression.append(column.is_null().sum().cast(pl.UInt64).alias(missing_alias))
        plan = (
            relation.lazy.group_by(list(keys), maintain_order=True).agg(expression)
            if keys
            else relation.lazy.select(expression)
        )
        result = relation.collect_plan(plan)
    if how == "sum" and dtype is not None and dtype.is_integer() and relation.sql is not None:
        # DuckDB exposes HUGEINT through Arrow Decimal(38, 0); restore the
        # established integer result before choosing its compact storage.
        result = result.with_columns(pl.col(alias).cast(pl.Int128))
    return (
        narrow_integer(result, alias)
        if how == "sum" and dtype is not None and dtype.is_integer()
        else result
    )


def keyed_values(
    frame: pl.DataFrame, keys: Sequence[str], alias: str, values: Sequence[str | None]
) -> pl.DataFrame:
    """Group keys of ``frame`` with one exact rational-string value per group."""
    column = pl.Series(alias, list(values), dtype=pl.String)
    return frame.select(*keys).with_columns(column) if keys else column.to_frame()


def grouped_reduce(
    relation: ProjectedRelation, keys: Sequence[str], expressions: Mapping[str, pl.Expr]
) -> pl.DataFrame:
    """Collect native per-group reductions whose backend result is already exact.

    Counts, distinct counts, extrema and other dtype-preserving reductions need
    no exact fallback. Only the G-row result crosses the collection boundary.
    Polars plans only: SQL relations keep their own reductions.
    """
    if relation.lazy is None:
        raise ValueError("grouped reductions need a Polars relation")
    named = [expression.alias(alias) for alias, expression in expressions.items()]
    plan = (
        relation.lazy.group_by(list(keys), maintain_order=True).agg(named)
        if keys
        else relation.lazy.select(named)
    )
    return relation.collect_plan(plan)


def grouped_quantile(
    relation: ProjectedRelation,
    keys: Sequence[str],
    value: str,
    alias: str,
    probability: float | Decimal,
) -> pl.DataFrame:
    """Exact NumPy-linear quantiles per group as canonical rational strings.

    The backend selects the two bracketing order statistics in their stored
    dtype; interpolation is then exact (:func:`~qavf.statistics.ordered.linear_quantile`),
    so integers beyond 2**53 and decimals are never rounded. Null and NaN are
    missing; a group without usable values has no quantile.
    """
    dtype = relation.schema[value]
    if not dtype.is_numeric() and dtype != pl.Null:
        raise ValueError(f"quantiles require numeric values, found {dtype}")
    exact = Fraction(str(probability))
    if not 0 <= exact <= 1:
        raise ValueError("quantile probabilities must lie in [0, 1]")
    taken = [*relation.schema, alias]
    count, low, high = (fresh_internal(name, taken) for name in ("q_count", "q_low", "q_high"))
    measured = numeric_expr(value, dtype)
    ordered = measured.drop_nulls().sort()
    size = measured.count()
    # A count fits UInt64. With a numerator below 2**63, their product
    # therefore fits signed Int128, regardless of the actual group size.
    # Decimal probabilities can exceed that proven domain; only their scalar
    # group counts cross the arbitrary-precision fallback, never member rows.
    if exact.numerator < 2**63 and exact.denominator < 2**127:
        rank = (size.cast(pl.Int128) - 1).clip(lower_bound=0) * pl.lit(
            exact.numerator, dtype=pl.Int128
        )
        denominator = pl.lit(exact.denominator, dtype=pl.Int128)
        quotient = rank // denominator
        lower = quotient.cast(pl.Int64)
        upper = (quotient + (rank % denominator != 0).cast(pl.Int128)).cast(pl.Int64)
    else:
        def rank_at(count: int, *, upper: bool = False) -> int:
            quotient, remainder = divmod(max(count - 1, 0) * exact.numerator, exact.denominator)
            return quotient + int(upper and remainder != 0)

        lower = size.map_elements(rank_at, return_dtype=pl.Int64)
        upper = size.map_elements(lambda count: rank_at(count, upper=True), return_dtype=pl.Int64)
    reduced = grouped_reduce(
        relation,
        keys,
        {
            count: measured.count().cast(pl.UInt64),
            low: ordered.slice(lower, 1).first(),
            high: ordered.slice(upper, 1).first(),
        },
    )
    results: list[str | None] = []
    for size, a, b in reduced.select(count, low, high).iter_rows():
        if not size:
            results.append(None)
            continue
        results.append(str(linear_quantile(a, b, quantile_ranks(size, probability)[2])))
    return keyed_values(reduced, keys, alias, results)


def grouped_mean(
    relation: ProjectedRelation, keys: Sequence[str], value: str, alias: str
) -> pl.DataFrame:
    """Exact means per group as canonical rational strings.

    Integer and decimal sums use the overflow-checked native accumulator;
    stored floats use the exact sorted-batch reducer. Division is exact. A group
    without usable values has no mean.
    """
    dtype = relation.schema[value]
    if dtype.is_float():
        if relation.observation is None or relation.lazy is None:
            raise ValueError("float means need a scoped observation")
        return exact_grouped(
            relation.ctx,
            relation.observation,
            relation.lazy,
            keys,
            value,
            alias,
            "mean",
            rational=True,
        )
    taken = [*relation.schema, alias]
    rows, missing = fresh_internal("mean_rows", taken), fresh_internal("mean_missing", taken)
    sums = grouped_aggregate(
        relation, keys, value, alias, "sum", presence_alias=rows, missing_alias=missing
    )
    assert sums is not None
    results: list[str | None] = []
    for total, size, absent in sums.select(alias, rows, missing).iter_rows():
        usable = size - absent
        results.append(None if total is None or not usable else str(Fraction(total) / usable))
    return keyed_values(sums, keys, alias, results)
