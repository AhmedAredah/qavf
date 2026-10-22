"""Grouped measures: one value per group on each side of a comparison.

A measure reduces one side's aligned relation to a G-row frame of group keys
and one value column. Built-ins reuse the package's exact reductions: integer
and decimal sums use the overflow-checked native accumulator, float sums and
means the exact sorted-batch reducer, quantiles exact order statistics. Exact
non-integer results are canonical rational strings (``"7/2"``), which
comparators read without rounding. Only ``std`` is computed in binary64.

Each measure also names its default significance test (``test``) and what a
test may read from it: the sample statistic a resampling test compares
(``statistic``), whether its values are event counts (``counts``), and the
successes and trials of a share (``proportions``). A measure without a sound
test (``min``, ``max``, ``distinct_count``) is judged by materiality alone.

Writing a measure::

    @register_measure
    class Range(Measure):
        name = "range"

        def compute(self, data):
            value = pl.col(data.column)
            return grouped_reduce(data.relation, data.keys, {data.alias: value.max() - value.min()})
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

import polars as pl
from pydantic import Field

from qavf.aggregation import numeric_expr
from qavf.compare.registry import Component, ComponentRegistry
from qavf.compare.spec import Side
from qavf.engine.registry import RuleParams
from qavf.rules._exact_grouped import exact_grouped
from qavf.rules._grouped import (
    ProjectedRelation,
    grouped_aggregate,
    grouped_mean,
    grouped_quantile,
    grouped_reduce,
    keyed_values,
)
from qavf.rules.helpers import fresh_internal, missing_expr


@dataclass(frozen=True, slots=True)
class MeasureInput:
    """One side's aligned relation and the measure's place in it."""

    relation: ProjectedRelation
    keys: tuple[str, ...]
    """Group columns (empty: one group, the whole side)."""
    column: str | None
    alias: str
    """Name of the value column the result must have."""
    side: Side


class Measure(Component):
    """A per-group reduction; ``compute`` returns ``keys`` plus ``alias``."""

    requires_column: ClassVar[bool] = True
    numeric: ClassVar[bool] = True
    """Whether the column must be numeric."""
    test: ClassVar[str | None] = None
    """Default significance test (``None``: no sound test, judged by materiality)."""
    statistic: ClassVar[str | None] = None
    """What a resampling test compares: ``mean``, ``median``, ``sum``, ``std`` or
    ``quantile`` of the group's values (``None``: no sample statistic)."""
    counts: ClassVar[bool] = False
    """Whether the measure's values are event counts (for the Poisson test)."""

    @abstractmethod
    def compute(self, data: MeasureInput) -> pl.DataFrame:
        """Reduce ``data.relation`` per group; values may be numbers or rational strings."""

    def columns(self, column: str | None) -> tuple[str, ...]:
        """Columns read, for availability checks (the weight of a weighted mean, ...)."""
        return (column,) if column is not None else ()

    def default_test(self, dtype: pl.DataType | None) -> str | None:
        """The test used unless the measure configures ``test`` (``dtype``: the column's)."""
        return self.test

    def probability(self) -> float | None:
        """The quantile probability of a ``quantile`` statistic."""
        return None

    def proportions(self, data: MeasureInput) -> tuple[pl.Expr, pl.Expr | None] | None:
        """Successes and trials per group for a two-proportion test, as reductions.

        ``None`` trials mean the side's total of successes (a share of the side);
        ``None`` altogether: the measure is not a proportion.
        """
        return None

    def check(self, data: MeasureInput) -> None:
        """Refuse a column the measure cannot reduce (not assessed, never a pass)."""
        if data.column is None:
            if self.requires_column:
                raise ValueError(f"measure {self.name!r} needs a column")
            return
        dtype = data.relation.schema[data.column]
        if self.numeric and not dtype.is_numeric() and dtype != pl.Null:
            raise ValueError(f"measure {self.name!r} needs a numeric column, not {dtype}")


MEASURES: ComponentRegistry[Measure] = ComponentRegistry(
    "measure", "qavf.compare.measures", Measure, builtins=(__name__,)
)
register_measure = MEASURES.register


def _usable(data: MeasureInput) -> pl.Expr:
    assert data.column is not None
    return numeric_expr(data.column, data.relation.schema[data.column])


def _sum(data: MeasureInput) -> pl.DataFrame:
    """Exact sums: native for integers/decimals, the exact reducer for floats."""
    assert data.column is not None
    relation = data.relation
    if relation.schema[data.column].is_float():
        assert relation.observation is not None and relation.lazy is not None
        return exact_grouped(
            relation.ctx,
            relation.observation,
            relation.lazy,
            data.keys,
            data.column,
            data.alias,
            "sum",
            rational=True,
        )
    result = grouped_aggregate(relation, data.keys, data.column, data.alias, "sum")
    assert result is not None
    return result


@register_measure
class Count(Measure):
    """Rows per group, or non-missing values of ``column`` when one is given."""

    name: ClassVar[str] = "count"
    requires_column: ClassVar[bool] = False
    numeric: ClassVar[bool] = False
    test: ClassVar[str | None] = "poisson"
    counts: ClassVar[bool] = True

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        value = pl.len() if data.column is None else _usable(data).count()
        return grouped_reduce(data.relation, data.keys, {data.alias: value.cast(pl.UInt64)})


@register_measure
class NullCount(Measure):
    """Missing values (null or NaN) per group."""

    name: ClassVar[str] = "null_count"
    numeric: ClassVar[bool] = False
    test: ClassVar[str | None] = "two_proportion"
    counts: ClassVar[bool] = True

    def _missing(self, data: MeasureInput) -> pl.Expr:
        assert data.column is not None
        empty = pl.DataFrame(schema=data.relation.schema)
        return missing_expr(empty, data.column).sum().cast(pl.UInt64)

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        return grouped_reduce(data.relation, data.keys, {data.alias: self._missing(data)})

    def proportions(self, data: MeasureInput) -> tuple[pl.Expr, pl.Expr | None]:
        """Missing values out of the group's rows."""
        return self._missing(data), pl.len().cast(pl.UInt64)


@register_measure
class DistinctCount(Measure):
    """Distinct non-missing values per group."""

    name: ClassVar[str] = "distinct_count"
    numeric: ClassVar[bool] = False

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        distinct = _usable(data).drop_nulls().n_unique().cast(pl.UInt64)
        return grouped_reduce(data.relation, data.keys, {data.alias: distinct})


@register_measure
class Minimum(Measure):
    """Smallest non-missing value (any ordered type)."""

    name: ClassVar[str] = "min"
    numeric: ClassVar[bool] = False

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        return grouped_reduce(data.relation, data.keys, {data.alias: _usable(data).min()})


@register_measure
class Maximum(Measure):
    """Largest non-missing value (any ordered type)."""

    name: ClassVar[str] = "max"
    numeric: ClassVar[bool] = False

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        return grouped_reduce(data.relation, data.keys, {data.alias: _usable(data).max()})


@register_measure
class Sum(Measure):
    """Exact sum of non-missing values; a group with none has no sum."""

    name: ClassVar[str] = "sum"
    statistic: ClassVar[str | None] = "sum"
    counts: ClassVar[bool] = True

    def default_test(self, dtype: pl.DataType | None) -> str | None:
        """Poisson for integer columns (sums of counts), a bootstrap of the sum otherwise."""
        return "poisson" if dtype is not None and dtype.is_integer() else "bootstrap"

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        return _sum(data)


@register_measure
class Mean(Measure):
    """Exact mean of non-missing values."""

    name: ClassVar[str] = "mean"
    test: ClassVar[str | None] = "welch_t"
    statistic: ClassVar[str | None] = "mean"

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        assert data.column is not None
        return grouped_mean(data.relation, data.keys, data.column, data.alias)


@register_measure
class Median(Measure):
    """Exact median (linear interpolation between the middle values)."""

    name: ClassVar[str] = "median"
    test: ClassVar[str | None] = "mann_whitney"
    statistic: ClassVar[str | None] = "median"

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        assert data.column is not None
        return grouped_quantile(data.relation, data.keys, data.column, data.alias, 0.5)


class QuantileParams(RuleParams):
    p: float = Field(ge=0, le=1)
    """Probability, e.g. 0.9 for the 90th percentile (NumPy's linear method)."""


@register_measure
class Quantile(Measure):
    """Exact quantile ``p`` (NumPy's linear interpolation)."""

    name: ClassVar[str] = "quantile"
    Params: ClassVar[type[RuleParams]] = QuantileParams
    test: ClassVar[str | None] = "bootstrap"
    statistic: ClassVar[str | None] = "quantile"

    def probability(self) -> float:
        return float(self.params.p)  # type: ignore[attr-defined]

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        assert data.column is not None
        probability = self.params.p  # type: ignore[attr-defined]
        return grouped_quantile(data.relation, data.keys, data.column, data.alias, probability)


@register_measure
class StandardDeviation(Measure):
    """Sample standard deviation (binary64, as computed by the backend)."""

    name: ClassVar[str] = "std"
    test: ClassVar[str | None] = "brown_forsythe"
    statistic: ClassVar[str | None] = "std"

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        value = _usable(data).cast(pl.Float64).std()
        return grouped_reduce(data.relation, data.keys, {data.alias: value})


class WeightedMeanParams(RuleParams):
    weight: str
    """Weight column; rows with a missing weight are left out."""


@register_measure
class WeightedMean(Measure):
    """Exact weighted mean; a group with a negative weight has none."""

    name: ClassVar[str] = "weighted_mean"
    Params: ClassVar[type[RuleParams]] = WeightedMeanParams

    def columns(self, column: str | None) -> tuple[str, ...]:
        return (*super().columns(column), self.params.weight)  # type: ignore[attr-defined]

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        assert data.column is not None
        relation = data.relation
        assert relation.observation is not None and relation.lazy is not None
        return exact_grouped(
            relation.ctx,
            relation.observation,
            relation.lazy,
            data.keys,
            data.column,
            data.alias,
            "weighted_mean",
            weight=self.params.weight,  # type: ignore[attr-defined]
            rational=True,
        )


@register_measure
class Share(Measure):
    """Each group's exact share of the side's total (of ``column``, else of rows)."""

    name: ClassVar[str] = "share"
    requires_column: ClassVar[bool] = False
    test: ClassVar[str | None] = "two_proportion"

    def proportions(self, data: MeasureInput) -> tuple[pl.Expr, pl.Expr | None]:
        """The group's rows (or integer sum of ``column``) out of the side's total."""
        if data.column is None:
            return pl.len().cast(pl.UInt64), None
        return _usable(data).sum(), None

    def compute(self, data: MeasureInput) -> pl.DataFrame:
        parts = Count().compute(data) if data.column is None else _sum(data)
        values = [None if value is None else Fraction(value) for value in parts[data.alias]]
        total = sum((value for value in values if value is not None), Fraction(0))
        shares = [None if value is None or not total else str(value / total) for value in values]
        return keyed_values(parts, data.keys, data.alias, shares)


def measure_alias(index: int, taken: list[str]) -> str:
    """A value column name that cannot shadow a group column."""
    return fresh_internal(f"measure_{index}", taken)
