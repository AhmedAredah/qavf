"""Exact statistics of short ordered series: least squares, correlation, lags, splits.

Series here are already reduced (one value per period), so kernels take Python
sequences. Integers and exact rationals stay exact: least-squares slopes are
rationals, and a correlation is kept as an exact numerator and squared
denominator, so thresholds and lag comparisons are decided without rounding.
Only the reported value is rounded, once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import total_ordering

import numpy as np

from qavf.stats import center_scale_observations

type Exact = int | Fraction


@dataclass(frozen=True, slots=True)
class NormalEquations[T: (int, Fraction)]:
    """Least-squares terms of ``y`` on ``x`` without division.

    ``centred[i] = count * x[i] - sum(x)``, ``spread = count**2 * Sxx`` and
    ``cross = count * Sxy``, all exact for integer or rational inputs.
    """

    count: int
    sum_y: T
    centred: list[T]
    spread: T
    cross: T

    @property
    def slope(self) -> Fraction | None:
        """``Sxy / Sxx``; ``None`` when every ``x`` is equal."""
        return Fraction(self.cross * self.count) / self.spread if self.spread else None


def normal_equations[T: (int, Fraction)](x: Sequence[T], y: Sequence[T]) -> NormalEquations[T]:
    """Exact least-squares terms of ``y`` regressed on ``x``."""
    if len(x) != len(y) or not x:
        raise ValueError("least squares needs paired, nonempty observations")
    count = len(x)
    sum_x, sum_y = sum(x), sum(y)
    centred = [count * value - sum_x for value in x]
    spread = sum(c * c for c in centred)
    cross = sum(c * value for c, value in zip(centred, y, strict=True))
    return NormalEquations(count, sum_y, centred, spread, cross)  # type: ignore[arg-type]


def common_scale(values: Sequence[Exact]) -> tuple[list[int], int]:
    """Rationals as integers over their least common denominator."""
    denominator = math.lcm(*(Fraction(value).denominator for value in values)) if values else 1
    return [int(Fraction(value) * denominator) for value in values], denominator


@total_ordering
@dataclass(frozen=True, slots=True)
class Correlation:
    """A Pearson correlation ``numerator / sqrt(squared_denominator)``, held exactly."""

    numerator: int
    squared_denominator: int
    pairs: int

    @property
    def value(self) -> float:
        """The correlation, rounded once."""
        squared = Fraction(self.numerator * self.numerator, self.squared_denominator)
        return math.copysign(math.sqrt(squared), self.numerator)

    def at_least(self, threshold: float) -> bool:
        """``value >= threshold`` decided exactly (the threshold's decimal spelling)."""
        limit = Fraction(str(threshold))
        square = self.numerator * self.numerator * limit.denominator**2
        bound = limit.numerator**2 * self.squared_denominator
        if limit >= 0:
            return self.numerator >= 0 and square >= bound
        return self.numerator >= 0 or square <= bound

    def _key(self) -> tuple[int, Fraction]:
        sign = (self.numerator > 0) - (self.numerator < 0)
        return sign, sign * Fraction(self.numerator * self.numerator, self.squared_denominator)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Correlation):
            return NotImplemented
        return self._key() < other._key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Correlation):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


def pearson(x: Sequence[Exact], y: Sequence[Exact]) -> Correlation | None:
    """Exact Pearson correlation; ``None`` when either side is constant (undefined)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation needs at least two paired observations")
    a, _ = common_scale(x)
    b, _ = common_scale(y)
    count = len(a)
    sum_a, sum_b = sum(a), sum(b)
    covariance = count * sum(p * q for p, q in zip(a, b, strict=True)) - sum_a * sum_b
    variance_a = count * sum(p * p for p in a) - sum_a * sum_a
    variance_b = count * sum(q * q for q in b) - sum_b * sum_b
    if not variance_a or not variance_b:
        return None
    return Correlation(covariance, variance_a * variance_b, count)


def average_ranks(values: Sequence[Exact]) -> list[Fraction]:
    """1-based ranks, ties sharing their average rank (exact)."""
    order = sorted(range(len(values)), key=lambda index: Fraction(values[index]))
    ranks: list[Fraction] = [Fraction(0)] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = Fraction(start + end + 2, 2)
        for position in range(start, end + 1):
            ranks[order[position]] = rank
        start = end + 1
    return ranks


def spearman(x: Sequence[Exact], y: Sequence[Exact]) -> Correlation | None:
    """Spearman's rank correlation: Pearson of average ranks, exactly."""
    return pearson(average_ranks(x), average_ranks(y))


def lagged_pairs(
    baseline: Mapping[int, Exact], candidate: Mapping[int, Exact], lag: int, period: int | None
) -> tuple[list[Exact], list[Exact]]:
    """Pairs ``(baseline[t], candidate[t + lag])`` over positions present in both.

    With a ``period`` (a cyclic profile) positions wrap around the cycle.
    """
    left, right = [], []
    for position in sorted(baseline):
        other = position + lag
        if period is not None:
            other %= period
        if other in candidate:
            left.append(baseline[position])
            right.append(candidate[other])
    return left, right


@dataclass(frozen=True, slots=True)
class Lag:
    """The best-correlated lag: ``candidate[t + lag]`` follows ``baseline[t]``."""

    lag: int
    correlation: Correlation
    at_zero: Correlation | None


def best_lag(
    baseline: Mapping[int, Exact],
    candidate: Mapping[int, Exact],
    max_lag: int,
    min_pairs: int,
    period: int | None = None,
) -> Lag | None:
    """Cross-correlation over ``-max_lag..max_lag``; ``None`` when no lag is defined.

    Lags with fewer than ``min_pairs`` pairs or a constant side are skipped. Ties
    prefer the smallest absolute lag, then the negative one, so the choice is
    deterministic and a shift is reported only when it correlates strictly better.
    """
    best: Lag | None = None
    zero: Correlation | None = None
    limit = max_lag if period is None else min(max_lag, period // 2)
    for lag in sorted(range(-limit, limit + 1), key=lambda k: (abs(k), k)):
        left, right = lagged_pairs(baseline, candidate, lag, period)
        if len(left) < max(min_pairs, 2):
            continue
        correlation = pearson(left, right)
        if correlation is None:
            continue
        if lag == 0:
            zero = correlation
        if best is None or correlation > best.correlation:
            best = Lag(lag, correlation, None)
    return None if best is None else Lag(best.lag, best.correlation, zero)


def best_split(
    values: np.ndarray, min_segment: int, min_relative_shift: float = 0.0
) -> tuple[int, float, float, float] | None:
    """The split maximising the gap between segment medians over the pooled robust spread.

    Returns ``(split, median_before, median_after, score)``: the score is the gap
    in units of 1.4826 x the median absolute residual around the segment medians.
    A robust two-segment screen for a level shift, not a formal change-point test.
    """
    best: tuple[int, float, float, float] | None = None
    if not len(values) or not bool(np.any(values)):
        return None
    values, unit, origin = center_scale_observations(values)
    exact_unit, exact_origin = Fraction(unit), Fraction(origin)
    required_relative = Fraction(str(min_relative_shift))
    for split in range(min_segment, len(values) - min_segment + 1):
        left, right = values[:split], values[split:]
        left_median, right_median = float(np.median(left)), float(np.median(right))
        residuals = np.concatenate([left - left_median, right - right_median])
        spread = 1.4826 * float(np.median(np.abs(residuals)))
        gap = abs(right_median - left_median)
        before = Fraction(left_median) * exact_unit + exact_origin
        after = Fraction(right_median) * exact_unit + exact_origin
        if gap == 0 or abs(after - before) < required_relative * abs(before):
            continue
        score = math.inf if spread == 0 else gap / spread
        if best is None or score > best[3]:
            best = (split, float(before), float(after), score)
    return best
