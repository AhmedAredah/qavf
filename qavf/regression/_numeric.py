"""Exact scalar tolerances and conservative columnar equality decisions."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import polars as pl

from qavf.numeric import tolerance_fraction

Number = int | float | Decimal


def _float_tolerance(value: Number | Fraction) -> float:
    """The documented math.isclose control for a validated nonnegative tolerance."""
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _is_nan(value: Number) -> bool:
    return (
        value.is_nan()
        if isinstance(value, Decimal)
        else (isinstance(value, float) and math.isnan(value))
    )


def _finite(value: Number) -> bool:
    if isinstance(value, int):
        return True
    return value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)


def numbers_close(a: Number, b: Number, rtol: float, atol: float) -> bool:
    """Whether two numbers are equal within tolerance, without rounding large integers
    or decimals."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=_float_tolerance(rtol), abs_tol=_float_tolerance(atol))
    if _is_nan(a) or _is_nan(b):
        return _is_nan(a) and _is_nan(b)
    if a == b:  # Python compares int, float and Decimal exactly
        return True
    if not (_finite(a) and _finite(b)):
        return False
    exact_a, exact_b = Fraction(a), Fraction(b)
    gap = abs(exact_a - exact_b)
    # Configuration tolerances denote their decimal spelling, as in the shared
    # exact numeric policy. Float pairs above retain documented math.isclose.
    return gap <= tolerance_fraction(atol) or gap <= tolerance_fraction(rtol) * max(
        abs(exact_a), abs(exact_b)
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | Decimal) and not isinstance(value, bool)


def _difference_size(a: Any, b: Any) -> tuple[float | None, float | None]:
    """Absolute and relative difference of two numbers (exact before the final rounding)."""
    if not (_is_number(a) and _is_number(b)):
        return None, None
    if not (_finite(a) and _finite(b)):
        return None, None
    gap = abs(Fraction(a) - Fraction(b))

    def represent(value: Fraction) -> float:
        try:
            return float(value)
        except OverflowError:
            return math.inf

    relative = represent(gap / abs(Fraction(a))) if a else None
    return represent(gap), relative


def _nonfinite(value: Any) -> bool:
    return _is_number(value) and not _finite(value)


def _integer_difference_maxima(cells: pl.DataFrame) -> tuple[float | None, float | None]:
    """Native exact metrics for ordinary integers; arbitrary-width fallback otherwise.

    When every numerator/denominator fits the exact binary64 integer domain,
    one native division has the same rounding as float(Fraction(gap, abs(a))).
    Wider values use Python integer subtraction/division without rounding inputs
    or building several Fractions per cell. Missing pairs have no difference size.
    """
    finite = cells.filter(pl.col("a").is_not_null() & pl.col("b").is_not_null())
    if finite.is_empty():
        return None, None
    amin, amax, bmin, bmax = finite.select(
        pl.col("a").min().alias("amin"),
        pl.col("a").max().alias("amax"),
        pl.col("b").min().alias("bmin"),
        pl.col("b").max().alias("bmax"),
    ).row(0)
    gap_bound = max(abs(amin - bmax), abs(amax - bmin))
    if max(abs(amin), abs(amax), gap_bound) <= 2**53:
        a, b = pl.col("a").cast(pl.Int128), pl.col("b").cast(pl.Int128)
        gap = (a - b).abs()
        absolute, relative = finite.select(
            gap.max().alias("absolute"),
            pl.when(a != 0)
            .then(gap.cast(pl.Float64) / a.abs().cast(pl.Float64))
            .max()
            .alias("relative"),
        ).row(0)
        return float(absolute), relative
    maximum = 0
    relative_max: float | None = None
    for before, after in finite.iter_rows():
        gap = abs(before - after)
        maximum = max(maximum, gap)
        if before:
            relative = gap / abs(before)
            relative_max = relative if relative_max is None else max(relative_max, relative)
    return float(maximum), relative_max


def _settled(
    a: pl.Expr, b: pl.Expr, da: pl.DataType, db: pl.DataType, rtol: float, atol: float
) -> pl.Expr:
    """True where two cells certainly compare equal under the comparison's own rules.

    Float pairs are promoted to binary64 before applying ``math.isclose``'s formula;
    NaN matches NaN. Other pairs of one dtype are settled
    only when exactly equal, leaving every other decision (integers within
    tolerance, decimals, mixed types) to the exact path. Native nested equality
    can settle identical cells; nested tolerance is handled separately.
    """
    both_null = a.is_null() & b.is_null()
    if da != db or isinstance(da, pl.Object):
        return both_null
    if da.is_float():
        rtol, atol = _float_tolerance(rtol), _float_tolerance(atol)
        a, b = a.cast(pl.Float64), b.cast(pl.Float64)
        gap = (a - b).abs()
        close = (a == b) | (
            a.is_finite()
            & b.is_finite()
            & ((gap <= rtol * b.abs()) | (gap <= rtol * a.abs()) | (gap <= atol))
        )
        return both_null | (a.is_nan() & b.is_nan()) | close.fill_null(False)
    return both_null | (a == b).fill_null(False)
