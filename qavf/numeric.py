"""Exact scalar comparisons shared by QA kernels and backend adapters.

Float inputs denote their stored binary value. Tolerances denote their decimal
spelling, matching configuration files. No operation depends on Decimal's
process/thread context.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from decimal import Decimal
from fractions import Fraction
from typing import Any

from qavf._float_accumulator import ExactFloatAccumulator


def exact_number(value: int | float | Decimal) -> Decimal:
    """Represent a supplied value without an intermediate float or rounding."""
    return Decimal.from_float(value) if isinstance(value, float) else Decimal(value)


def tolerance_fraction(value: int | float | Decimal | Fraction) -> Fraction:
    """An exact tolerance from configuration spelling, reusing precomputed rationals."""
    return value if isinstance(value, Fraction) else Fraction(str(value))


def numeric_close(a: Any, b: Any, rtol: float = 0, atol: float = 0) -> bool:
    """Finite-only symmetric tolerance using exact rational intermediates."""
    if a is None or b is None:
        return False
    try:
        x, y = Fraction(a), Fraction(b)
        absolute, relative = tolerance_fraction(atol), tolerance_fraction(rtol)
    except (ValueError, OverflowError, TypeError, ZeroDivisionError):
        return False
    return abs(x - y) <= max(absolute, relative * max(abs(x), abs(y)))


def exact_rational(value: Any) -> Fraction | None:
    """The exact rational of a finite number or canonical rational string.

    Rational strings (``"7/2"``) are how exact reducers return non-integer
    results; ``None`` stands for a missing, non-finite or non-numeric value.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Decimal) and not value.is_finite():
        return None
    if isinstance(value, int | float | Decimal | Fraction):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def report_number(value: Fraction | float | None) -> int | float | None:
    """A rational for reports: integers stay exact, other values round once."""
    if value is None or isinstance(value, float):
        return value
    if value.denominator == 1:
        return value.numerator
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def checked_float(value: int | float | Decimal) -> float:
    """Convert finite exact data only when its stored value is representable."""
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("value cannot be represented exactly as Float64") from None
    if not math.isfinite(result) or Fraction(result) != Fraction(value):
        raise ValueError("value cannot be represented exactly as Float64")
    return result


def exact_float_sum(values: Any) -> Fraction | None:
    """Sum finite binary floats exactly with bounded accumulator storage.

    Each Float64 is an integer multiple of 2**-1074. The integer accumulator
    therefore grows only logarithmically with the number of observations; no
    Python list of the data or order-dependent floating partials is retained.
    Null and NaN are missing. Infinity is not a measurement.
    """
    accumulator = ExactFloatAccumulator()
    accumulator.update(values)
    return accumulator.total()


def finite_float(value: Fraction) -> float:
    """Round an exact result once, refusing overflow or complete underflow."""
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("statistic exceeds Float64 range") from exc
    if not math.isfinite(result) or (result == 0.0 and value):
        raise ValueError("statistic cannot be represented as a nonzero finite Float64")
    if abs(value) >= 2**52 and Fraction(result) != value:
        raise ValueError("aggregated value cannot preserve exact large-value precision")
    return result


def stable_float_statistic(values: Any, name: str) -> Fraction | float | None:
    """Exact sum/mean/median and offset-safe sample deviation of stored floats.

    Standard deviation uses Python's single-pass exact-ratio variance
    algorithm, followed by one square-root rounding. It avoids both overflow
    in squares and cancellation in moments, without a second observation pass.
    """
    import statistics

    if name == "std":
        varies = False
        origin = None

        def finite_values() -> Iterator[float]:
            nonlocal varies, origin
            for value in values:
                if value is None or math.isnan(value):
                    continue
                if not math.isfinite(value):
                    raise ValueError("non-finite observations cannot be aggregated")
                if origin is None:
                    origin = value
                elif value != origin:
                    varies = True
                yield float(value)

        try:
            result = statistics.stdev(finite_values())
        except statistics.StatisticsError:
            return None
        if not math.isfinite(result):
            raise ValueError("standard deviation exceeds Float64 range")
        if result == 0.0 and varies:
            raise ValueError("standard deviation underflows Float64 range")
        return result
    if name == "median":
        ordered = sorted(float(v) for v in values if v is not None and not math.isnan(v))
        if not ordered:
            return None
        middle = len(ordered) // 2
        return (
            Fraction(ordered[middle])
            if len(ordered) % 2
            else (Fraction(ordered[middle - 1]) + Fraction(ordered[middle])) / 2
        )
    accumulator = ExactFloatAccumulator()
    accumulator.update(values)
    total = accumulator.total()
    if total is None or name == "sum":
        return total
    if name != "mean":
        raise ValueError(f"unsupported statistic {name!r}")
    return accumulator.mean()


def stable_sum_scalar(values: Any) -> float | None:
    """Float64 expression adapter for the shared exact accumulator."""
    result = exact_float_sum(values)
    return None if result is None else finite_float(result)


def stable_mean_scalar(values: Any) -> float | None:
    """Float64 expression adapter; division precedes the only rounding."""
    result = stable_float_statistic(values, "mean")
    return None if result is None else finite_float(Fraction(result))


def stable_weighted_mean(values: Any, weights: Any) -> float | None:
    """Accumulate products and weights exactly, with one final rounding."""
    numerator, denominator = Fraction(), Fraction()
    for value, weight in zip(values, weights, strict=True):
        if value is None or weight is None or math.isnan(value) or math.isnan(weight):
            continue
        if not math.isfinite(value) or not math.isfinite(weight):
            raise ValueError("weighted means require finite observations and weights")
        if weight < 0:
            return None
        exact_weight = Fraction(weight)
        numerator += Fraction(value) * exact_weight
        denominator += exact_weight
    return finite_float(numerator / denominator) if denominator else None
