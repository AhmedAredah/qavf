"""Zone hierarchies and crosswalks: child zones mapped to parents, possibly split.

A :class:`Crosswalk` maps each ``from`` zone to one or more ``to`` zones: a link
to its traffic zone, a zone to its county, a 2010 zone to the 2020 zones that
replaced it. Many-to-one is the common case (weight 1). One-to-many splits a
zone with ``weight`` shares that must sum to 1 within ``tolerance`` for every
``from`` zone; that, unique pairs, present ids and finite weights in (0, 1]
are validated before anything uses it (:func:`crosswalk_frame`).

Hierarchies (link -> zone -> county -> state) are crosswalks composed with
:func:`compose` (weights multiply exactly). :func:`aggregate_up` aggregates
per-zone values to the parents exactly: unweighted sums with the exact native
accumulators of :mod:`qavf.aggregation`, split sums as exact rationals of the
binary values and weights, rounded once.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import Field, model_validator

from qavf.aggregation import exact_sum_expr, narrow_integer
from qavf.engine.registry import RuleParams
from qavf.numeric import exact_rational, report_number
from qavf.rules.helpers import plural

if TYPE_CHECKING:
    from qavf.engine.context import DataContext

SOURCE, TARGET, WEIGHT = "from", "to", "weight"
"""Columns of a validated crosswalk frame."""


class Crosswalk(RuleParams):
    """Zone ``from`` -> zone ``to`` (with a split ``weight``), from a dataset, file or list."""

    dataset: str | None = None
    """A configured dataset holding the pairs."""
    path: str | None = None
    """A local CSV or Parquet file holding the pairs."""
    pairs: tuple[tuple[Any, Any] | tuple[Any, Any, float], ...] | None = None
    """Inline ``[from, to]`` or ``[from, to, weight]`` pairs."""
    from_column: str = "from"
    """Column of the child (or old) zone ids."""
    to_column: str = "to"
    """Column of the parent (or new) zone ids."""
    weight_column: str | None = None
    """Split shares of each ``from`` zone; weight 1 for every pair when omitted."""
    tolerance: float = Field(default=1e-9, ge=0)
    """How far a ``from`` zone's weights may sum from 1."""

    @model_validator(mode="after")
    def _one_source(self) -> Crosswalk:
        if sum(value is not None for value in (self.dataset, self.path, self.pairs)) != 1:
            raise ValueError("a crosswalk needs exactly one of 'dataset', 'path' or 'pairs'")
        if self.pairs is not None and not self.pairs:
            raise ValueError("'pairs' must list at least one pair")
        return self

    def label(self) -> str:
        return self.dataset or self.path or "inline crosswalk"

    def datasets(self) -> frozenset[str]:
        return frozenset({self.dataset}) if self.dataset is not None else frozenset()


def pairs_frame(ctx: DataContext | None, crosswalk: Crosswalk) -> pl.DataFrame:
    """The configured ``from``/``to``/``weight`` pairs as read, before any validation."""
    if crosswalk.pairs is not None:
        weighted = any(len(pair) == 3 for pair in crosswalk.pairs)
        if weighted and not all(len(pair) == 3 for pair in crosswalk.pairs):
            raise ValueError("crosswalk pairs must all have a weight, or none")
        rows = [list(pair) if weighted else [*pair, 1.0] for pair in crosswalk.pairs]
        return pl.DataFrame(rows, schema=[SOURCE, TARGET, WEIGHT], orient="row", strict=False)
    columns = [crosswalk.from_column, crosswalk.to_column]
    if crosswalk.weight_column is not None:
        columns.append(crosswalk.weight_column)
    if crosswalk.dataset is not None:
        if ctx is None:
            raise ValueError(f"crosswalk dataset {crosswalk.dataset!r} needs a data context")
        frame = ctx.project(crosswalk.dataset, columns)
    else:
        assert crosswalk.path is not None
        path = ctx.resolve_path(crosswalk.path) if ctx is not None else Path(crosswalk.path)
        parquet = path.suffix.lower() in (".parquet", ".pq")
        frame = pl.read_parquet(path, columns=columns) if parquet else pl.read_csv(path)
        if absent := [c for c in columns if c not in frame.columns]:
            raise ValueError(f"crosswalk {path.name!r} lacks column(s) {', '.join(absent)}")
        frame = frame.select(columns)
    weight = pl.col(crosswalk.weight_column) if crosswalk.weight_column else pl.lit(1.0)
    return frame.select(
        pl.col(crosswalk.from_column).alias(SOURCE),
        pl.col(crosswalk.to_column).alias(TARGET),
        weight.cast(pl.Float64).alias(WEIGHT),
    )


def validate_crosswalk(frame: pl.DataFrame, tolerance: float, label: str) -> pl.DataFrame:
    """``frame`` (``from``, ``to``, ``weight``) once every split is proven to sum to 1.

    Raises:
        ValueError: for missing ids or weights, repeated pairs, weights outside
            (0, 1], or a ``from`` zone whose weights do not sum to 1.
    """
    if missing := frame.select(pl.any_horizontal(pl.all().is_null()).sum()).item():
        raise ValueError(f"crosswalk {label!r}: {plural(missing, 'pair')} with a missing value")
    if repeated := frame.height - frame.select(SOURCE, TARGET).n_unique():
        raise ValueError(f"crosswalk {label!r}: {plural(repeated, 'pair')} repeated")
    weight = pl.col(WEIGHT)
    if bad := frame.filter(~weight.is_finite() | (weight <= 0) | (weight > 1)).height:
        raise ValueError(f"crosswalk {label!r}: {plural(bad, 'weight')} outside (0, 1]")
    # Exact sums of the binary weights, compared with the tolerance exactly.
    totals: dict[Any, Fraction] = {}
    for source, value in frame.select(SOURCE, WEIGHT).iter_rows():
        totals[source] = totals.get(source, Fraction(0)) + Fraction(value)
    limit = Fraction(tolerance)
    off = sorted(
        ((source, total) for source, total in totals.items() if abs(total - 1) > limit),
        key=lambda item: str(item[0]),
    )
    if off:
        sample = ", ".join(f"{source!r}: {float(total):.12g}" for source, total in off[:5])
        raise ValueError(
            f"crosswalk {label!r}: the weights of {plural(len(off), 'zone')} do not sum to 1 "
            f"within {tolerance:g} ({sample})"
        )
    return frame


def crosswalk_frame(ctx: DataContext | None, crosswalk: Crosswalk) -> pl.DataFrame:
    """The crosswalk's validated ``from``/``to``/``weight`` pairs.

    Raises:
        ValueError: as :func:`validate_crosswalk`, or for an unreadable source.
    """
    return validate_crosswalk(pairs_frame(ctx, crosswalk), crosswalk.tolerance, crosswalk.label())


def split(frame: pl.DataFrame) -> bool:
    """Whether any ``from`` zone maps to several ``to`` zones."""
    return frame[SOURCE].n_unique() < frame.height


def compose(*frames: pl.DataFrame) -> pl.DataFrame:
    """Crosswalks chained child to parent (link -> zone -> county); weights multiply.

    A child whose parent has no next level is dropped by the chain; validate the
    result with :func:`validate_crosswalk` to prove every split still sums to 1.
    """
    if not frames:
        raise ValueError("compose needs at least one crosswalk")
    chained = frames[0]
    for following in frames[1:]:
        step = following.rename({SOURCE: TARGET, TARGET: "next", WEIGHT: "next_weight"})
        joined = chained.join(step, on=TARGET, how="inner")
        # Products, and sums over paths reaching one parent, are exact; rounded once.
        weights: dict[tuple[Any, Any], Fraction] = {}
        for source, parent, a, b in joined.select(
            SOURCE, "next", WEIGHT, "next_weight"
        ).iter_rows():
            pair = (source, parent)
            weights[pair] = weights.get(pair, Fraction(0)) + Fraction(a) * Fraction(b)
        chained = pl.DataFrame(
            {
                SOURCE: pl.Series([p[0] for p in weights], dtype=joined.schema[SOURCE]),
                TARGET: pl.Series([p[1] for p in weights], dtype=joined.schema["next"]),
                WEIGHT: pl.Series([float(w) for w in weights.values()], dtype=pl.Float64),
            }
        )
    return chained


def apply_crosswalk(
    plan: pl.LazyFrame, column: str, frame: pl.DataFrame, apportion: Sequence[str], taken: str
) -> pl.LazyFrame:
    """``plan`` with ``column`` replaced by its ``to`` zone (null where unmapped).

    A split zone's rows repeat once per ``to`` zone, with the ``apportion``
    columns multiplied by the split weight (binary64). ``taken`` is a free column
    name for the weight while joining.

    Raises:
        ValueError: when the crosswalk ids cannot take the column's type.
    """
    dtype = plan.collect_schema()[column]
    try:
        source = frame[SOURCE].cast(dtype, strict=True)
    except Exception:
        raise ValueError(f"crosswalk ids cannot all be represented as {dtype}") from None
    target = f"{taken}_to"
    mapping = pl.DataFrame({column: source, target: frame[TARGET], taken: frame[WEIGHT]})
    joined = plan.join(mapping.lazy(), on=column, how="left")
    if split(frame):
        joined = joined.with_columns(pl.col(c) * pl.col(taken) for c in apportion)
    return joined.with_columns(pl.col(target).alias(column)).drop(target, taken)


def aggregate_up(
    frame: pl.DataFrame, zone: str, values: Sequence[str], crosswalk: pl.DataFrame
) -> pl.DataFrame:
    """Per-parent totals of ``values`` from one row per ``zone`` (a validated crosswalk).

    Without splits each parent's total is the exact native sum of its zones'
    values (integers and decimals stay exact, floats use the exact float
    accumulator). With splits each zone contributes ``value * weight``, summed as
    exact rationals and rounded once. Zones without a crosswalk entry refuse.

    Raises:
        ValueError: for zones the crosswalk lacks or repeated zones.
    """
    if repeated := frame.height - frame[zone].n_unique():
        raise ValueError(f"{plural(repeated, 'zone')} appear more than once")
    keyed = frame.join(crosswalk, left_on=zone, right_on=SOURCE, how="left")
    if unmapped := keyed.filter(pl.col(TARGET).is_null()).height:
        raise ValueError(f"{plural(unmapped, 'zone')} have no crosswalk entry")
    if not split(crosswalk.join(frame.select(pl.col(zone).alias(SOURCE)), on=SOURCE)):
        parents = keyed.group_by(TARGET, maintain_order=True).agg(
            exact_sum_expr(frame[column]).alias(column) for column in values
        )
        for column in values:
            parents = narrow_integer(parents, column)
        return parents.sort(TARGET).rename({TARGET: zone})
    totals: dict[Any, list[Fraction | None]] = {}
    for row in keyed.select(TARGET, WEIGHT, *values).iter_rows():
        parent, weight, numbers = row[0], Fraction(row[1]), row[2:]
        sums = totals.setdefault(parent, [None] * len(values))
        for index, number in enumerate(numbers):
            exact = exact_rational(number)
            if exact is not None:
                sums[index] = (sums[index] or Fraction(0)) + exact * weight
    order = sorted(totals, key=lambda parent: (str(type(parent)), parent))
    columns = {
        column: pl.Series(
            column, [report_number(totals[p][index]) for p in order], dtype=pl.Float64
        )
        for index, column in enumerate(values)
    }
    return pl.DataFrame({zone: pl.Series(zone, order, dtype=crosswalk.schema[TARGET]), **columns})
