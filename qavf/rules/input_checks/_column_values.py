"""Exact column statistics, contract labels and bounded row evidence."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast

import polars as pl

from qavf.aggregation import exact_sum_expr
from qavf.engine.registry import Finding
from qavf.numeric import stable_float_statistic
from qavf.rules._predicates import (
    format_value,
)
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import (
    fresh_internal,
    present,
)
from qavf.settings import (
    Bounds,
    ColumnCheck,
)


def _check_name(check: ColumnCheck) -> str:
    if check.name:
        return check.name
    parts = []
    for name, value in check.constraints().items():
        if isinstance(value, Bounds):
            sides = [
                f"≥ {value.ge:g}" if value.ge is not None else "",
                f"≤ {value.le:g}" if value.le is not None else "",
            ]
            parts.append(f"{name} {' and '.join(s for s in sides if s)}")
        elif value is True:
            parts.append(name)
        else:
            parts.append(f"{name} {format_value(value)}")
    return ", ".join(parts)


def _statistic(series: pl.Series, name: str) -> int | float | Decimal | Fraction | None:
    """Aggregate ``name`` of a column, ignoring missing values (null and NaN).

    Integer and Decimal reductions retain exact values; float sums, means and
    medians use rational arithmetic. Standard deviations are rounded once.
    """
    result: Any
    values = series.drop_nulls()
    if name == "distinct_count":
        return values.n_unique()
    if values.is_empty():
        return None
    if values.dtype.is_float() and name in ("sum", "mean", "median", "std"):
        return stable_float_statistic(values, name)
    if name in ("min", "max"):
        result = values.min() if name == "min" else values.max()
        return result if isinstance(result, int | float | Decimal) else None
    if name == "sum" and (values.dtype.is_integer() or values.dtype.is_decimal()):
        result = values.to_frame().select(exact_sum_expr(values)).item()
        return int(result) if values.dtype.is_integer() and result is not None else result
    if name in ("mean", "median") and not values.dtype.is_float():
        if name == "mean":
            total = values.to_frame().select(exact_sum_expr(values)).item()
            return Fraction(total) / len(values)
        ordered = values.sort()
        middle = len(ordered) // 2
        return (
            Fraction(ordered[middle])
            if len(ordered) % 2
            else (Fraction(ordered[middle - 1]) + Fraction(ordered[middle])) / 2
        )
    if name == "std":
        if len(values) < 2:
            return None
        if values.dtype.is_integer():
            origin_int = int(values[0])
            minimum, maximum = values.min(), values.max()
            if not isinstance(minimum, int) or not isinstance(maximum, int):
                raise TypeError("integer statistic returned noninteger extrema")
            if max(abs(minimum - origin_int), abs(maximum - origin_int)) <= 2**53:
                # The common path stays native and O(N) without Python objects.
                # Int128 subtraction is safe whenever the centered range fits.
                centered_values = (
                    values.to_frame()
                    .select(
                        (
                            pl.col(values.name).cast(pl.Int128)
                            - pl.lit(origin_int, dtype=pl.Int128)
                        ).cast(pl.Float64)
                    )
                    .to_series()
                )
                return cast(float, centered_values.std())
        # Subtract in exact arithmetic before converting, so an additive
        # offset cannot erase adjacent large integers or Decimal variation.
        origin = Fraction(values[0])
        centered = [Fraction(v) - origin for v in values]
        scale = max(abs(v) for v in centered)
        if not scale:
            return 0.0
        normalized = pl.Series([float(v / scale) for v in centered])
        deviation = normalized.std()
        result = float(Fraction.from_float(cast(float, deviation)) * scale)
    else:
        raise ValueError(f"unsupported statistic {name!r} for {values.dtype}")
    if result is None:
        return None
    if not math.isfinite(result):
        raise ValueError(f"{name} cannot be represented as a finite statistic")
    return float(result)


def _within(value: int | float | Decimal | Fraction | None, bounds: Bounds) -> bool:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    return (bounds.ge is None or value >= bounds.ge) and (bounds.le is None or value <= bounds.le)


def _show(value: int | float | Decimal | Fraction | None) -> str:
    if value is None:
        return "undefined"
    return str(value) if isinstance(value, int | Decimal | Fraction) else f"{value:.6g}"


def _show_bound(value: int | float | Decimal | Fraction | None) -> str:
    return "" if value is None else _show(value)


def _row_limits(
    original: pl.DataFrame,
    prepared: pl.DataFrame,
    predicates: list[tuple[str, pl.Expr]],
    limit: int,
) -> tuple[int, pl.DataFrame | None]:
    """Count a union of row failures, then build reasons only for its sample."""
    if not predicates:
        return 0, None
    bad = pl.any_horizontal([predicate for _, predicate in predicates])
    count = int(prepared.select(bad.sum()).item())
    if not count or not limit:
        return count, None
    row = fresh_internal("row", original.columns)
    violates = fresh_internal("violates", [*original.columns, row])
    positions = (
        prepared.lazy().with_row_index(row).filter(bad).select(row).limit(limit).collect()[row]
    )
    sample = original[positions]
    # Evaluate at most one boolean mask per constraint, then gather only
    # sampled positions; never construct the all-failure reason table.
    labels = []
    for name, predicate in predicates:
        selected = prepared.select(predicate.alias("bad"))["bad"][positions]
        labels.append(pl.when(pl.lit(selected)).then(pl.lit(name)).otherwise(None))
    reasons = sample.select(
        pl.concat_str(labels, separator="; ", ignore_nulls=True).alias(violates)
    )
    return count, temporal_evidence(
        present(
            sample.with_columns(positions, reasons[violates]),
            {row: "_row", violates: "violates"},
        )
    )


def _rows(
    frame: pl.DataFrame, mask: pl.Expr | pl.Series, violates: str, limit: int
) -> pl.DataFrame:
    """The first ``limit`` rows matching ``mask`` (callers pass the exact count)."""
    row = fresh_internal("row", frame.columns)
    reason = fresh_internal("violates", [*frame.columns, row])
    selected = frame.lazy().with_row_index(row).filter(mask).limit(limit).collect()
    return temporal_evidence(
        present(
            selected.with_columns(pl.lit(violates).alias(reason)),
            {row: "_row", reason: "violates"},
        )
    )


def _missing_column(dataset: str, column: str) -> Finding:
    """An absent required column leaves its limits unassessed on every backend."""
    return Finding.unassessed(
        f"Not assessed: required column {dataset}.{column} is absent.",
        check=f"{dataset}.{column}",
        datasets=(dataset,),
    )
