"""Exact scalar tolerances and conservative columnar equality decisions."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import polars as pl

Number = int | float | Decimal


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
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    if _is_nan(a) or _is_nan(b):
        return _is_nan(a) and _is_nan(b)
    if a == b:  # Python compares int, float and Decimal exactly
        return True
    if not (_finite(a) and _finite(b)):
        return False
    exact_a, exact_b = Fraction(a), Fraction(b)
    gap = abs(exact_a - exact_b)
    return gap <= Fraction(atol) or gap <= Fraction(rtol) * max(abs(exact_a), abs(exact_b))


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
        a, b = a.cast(pl.Float64), b.cast(pl.Float64)
        gap = (a - b).abs()
        close = (a == b) | (
            a.is_finite()
            & b.is_finite()
            & ((gap <= rtol * b.abs()) | (gap <= rtol * a.abs()) | (gap <= atol))
        )
        return both_null | (a.is_nan() & b.is_nan()) | close.fill_null(False)
    return both_null | (a == b).fill_null(False)
