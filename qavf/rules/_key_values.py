"""Canonical key values shared by grouping and row-evidence lookup.

The declared logical type, not an adapter's inferred storage type, defines key
equality. Conversion uses the same strict text policy as column validation;
invalid values become null and are handled by the caller's null contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial

import polars as pl

from qavf.applicability import NotApplicablePolicy
from qavf.engine.context import DataContext
from qavf.engine.sql import missing_sql
from qavf.rules._declared_types import ConvertedPlan, TextConversion, converts_text, typed_spec
from qavf.rules.helpers import missing_expr
from qavf.settings import ColumnSpec
from qavf.sources.sql import quote_identifier, sql_literal
from qavf.text_types import sql_value


def _collect_dataset(ctx: DataContext, dataset: str, selected: pl.LazyFrame) -> pl.DataFrame:
    """Collect a plan under the dataset's own execution context."""
    return ctx.collect(dataset, selected)


def _collect_reduced(
    collect: Callable[[pl.LazyFrame], pl.DataFrame],
    plan: pl.LazyFrame,
    reduce: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> pl.DataFrame:
    """Collect one reduction of a fixed plan."""
    return collect(reduce(plan))


class CanonicalKeyValues:
    """Compile equivalent SQL and lazy projections for one dataset's keys."""

    def __init__(self, ctx: DataContext, dataset: str, columns: Iterable[str]) -> None:
        self.ctx = ctx
        self.dataset = dataset
        self.columns = tuple(columns)
        spec = ctx.specs.get(dataset)
        self.contracts = spec.schema_spec.columns if spec and spec.schema_spec else {}

    @property
    def na_policies(self) -> dict[str, NotApplicablePolicy]:
        return {
            name: NotApplicablePolicy.from_spec(self.contracts[name])
            for name in self.columns
            if name in self.contracts and self.contracts[name].na_values
        }

    def na_sql(self) -> str:
        return (
            " OR ".join(
                policy.sql(quote_identifier(name)) for name, policy in self.na_policies.items()
            )
            or "false"
        )

    def na_expr(self) -> pl.Expr:
        expressions = [policy.polars(name) for name, policy in self.na_policies.items()]
        return pl.any_horizontal(expressions) if expressions else pl.lit(False)

    def lazy(
        self,
        plan: pl.LazyFrame,
        collector: Callable[[pl.LazyFrame], pl.DataFrame] | None = None,
        *,
        exclude_na: bool = False,
        preserve: Iterable[str] = (),
    ) -> pl.LazyFrame:
        plan = plan.select([*self.columns, *preserve])
        if exclude_na and self.na_policies:
            plan = plan.filter(~self.na_expr())
        schema = dict(plan.collect_schema())
        conversions = [
            TextConversion(name, self.contracts[name], schema[name])
            for name in self.columns
            if name in self.contracts and converts_text(self.contracts[name], schema[name])
        ]
        converted = {conversion.column for conversion in conversions}
        if conversions:
            collect = collector or partial(_collect_dataset, self.ctx, self.dataset)
            conversion = ConvertedPlan.resolved(
                conversions, partial(_collect_reduced, collect, plan)
            )
            plan = conversion(plan)
        empty = pl.DataFrame(schema=plan.collect_schema())
        return plan.with_columns(
            pl.when(
                missing_expr(
                    empty,
                    name,
                    typed_spec(self.contracts[name])
                    if name in converted
                    else self.contracts.get(name),
                )
            )
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
            for name in self.columns
        )

    def sql(
        self,
        *,
        exclude_na: bool = False,
        row_column: str | None = None,
        source_sql: str | None = None,
        preserve: Iterable[str] = (),
    ) -> tuple[str, dict[str, pl.DataType]]:
        """Source query and resulting key schema, with no materialization."""
        q = quote_identifier
        actual = self.ctx.schema(self.dataset)
        schema = {name: actual[name] for name in self.columns}
        projected = []
        normalized = []
        for name in self.columns:
            spec = self.contracts.get(name)
            value = q(name)
            if spec is not None and converts_text(spec, schema[name]):
                typed = spec.text_type()
                if typed is not None:
                    actual_datetime = schema[name]
                    if isinstance(actual_datetime, pl.Datetime):
                        from qavf.core.policy import ColumnValue

                        physical = (
                            "TIMESTAMP WITH TIME ZONE"
                            if actual_datetime.time_zone
                            else {"ms": "TIMESTAMP_MS", "us": "TIMESTAMP", "ns": "TIMESTAMP_NS"}[
                                actual_datetime.time_unit
                            ]
                        )
                        value = ColumnValue.compile(name, physical, spec).value
                    else:
                        value = sql_value(q(name), typed)[0]
                    schema[name] = {
                        "int": pl.Int64(),
                        "float": pl.Float64(),
                        "bool": pl.Boolean(),
                        "date": pl.Date(),
                        "datetime": pl.Datetime(typed.datetime_unit),
                    }[typed.dtype]
                else:
                    value, schema[name] = self._decimal(name, spec)
                spec = typed_spec(spec)
            projected.append(f"{value} AS {q(name)}")
            normalized.append(
                f"CASE WHEN {missing_sql(name, schema[name], spec)} THEN NULL "
                f"ELSE {q(name)} END AS {q(name)}"
            )
        source = f"({source_sql}) AS key_source" if source_sql is not None else q(self.dataset)
        for name in preserve:
            projected.append(q(name))
            normalized.append(q(name))
        if row_column is not None:
            source = (
                f"(SELECT *, row_number() OVER () - 1 AS {q(row_column)} FROM {source}) AS numbered"
            )
            projected.append(q(row_column))
            normalized.append(q(row_column))
        where = f" WHERE NOT ({self.na_sql()})" if exclude_na and self.na_policies else ""
        return (
            f"SELECT {', '.join(normalized)} FROM "
            f"(SELECT {', '.join(projected)} FROM {source}{where}) AS converted_keys",
            schema,
        )

    def _decimal(self, name: str, spec: ColumnSpec) -> tuple[str, pl.DataType]:
        """Use the same exact decimal admission and scale policy as native checks."""
        from qavf.core.policy import ColumnValue, DecimalTextSQL

        actual = self.ctx.schema(self.dataset)[name]
        if isinstance(actual, pl.Decimal) and spec.decimal_precision is not None:
            assert spec.decimal_scale is not None
            physical = f"DECIMAL({actual.precision or 38},{actual.scale or 0})"
            value = ColumnValue.compile(name, physical, spec)
            return value.value, pl.Decimal(spec.decimal_precision, spec.decimal_scale)

        policy = DecimalTextSQL(quote_identifier(name), spec)
        scale = spec.decimal_scale
        if scale is None:
            exempt = [NotApplicablePolicy.from_spec(spec).sql(quote_identifier(name))]
            markers = [v for v in spec.missing_values if isinstance(v, str)]
            if markers:
                exempt.append(
                    f"{quote_identifier(name)} IN ({', '.join(sql_literal(v) for v in markers)})"
                )
            scope = "NOT (" + " OR ".join(f"({item})" for item in exempt) + ")"
            scope += f" AND {policy.integer} + {policy.fraction} <= 38"
            reduction = (
                f"SELECT greatest(0, least(coalesce(max({policy.fraction}) "
                f"FILTER (WHERE {scope}), 0), "
                f"38 - coalesce(max({policy.integer}) FILTER (WHERE {scope}), 0))) "
                f"FROM {quote_identifier(self.dataset)}"
            )
            scale = int(
                self.ctx.sql(reduction, datasets=(self.dataset,), normalize_missing=False).item()
            )
        return policy.parsed(scale), pl.Decimal(spec.decimal_precision or 38, scale)
