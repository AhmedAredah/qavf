"""Exact order statistics from repeatable, sorted value-frequency streams.

No kernel retains all observations or distinct values. Sorting and lifecycle
belong to the caller's spill-backed store; this module owns only mathematics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from qavf._float_accumulator import ExactFloatAccumulator
from qavf.stats import DEFAULT_THRESHOLDS, OutlierMethod, check_threshold, minimum_samples

type CountStream = Iterable[tuple[float, int]]
type CountFactory = Callable[[], CountStream]


def quantiles(rows: CountStream, size: int, probabilities: Sequence[float]) -> np.ndarray:
    """NumPy linear quantiles using at most twice the requested quantile count."""
    if size < 1:
        raise ValueError("quantiles require a nonempty sample")
    q = np.asarray(probabilities, dtype=np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("quantile probabilities must lie in [0, 1]")
    ranks = (size - 1) * q
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    wanted = sorted(set(lower.tolist()) | set(upper.tolist()))
    selected: dict[int, float] = {}
    offset = 0
    next_rank = 0
    for value, frequency in rows:
        if frequency < 1:
            continue
        end = offset + frequency
        while next_rank < len(wanted) and wanted[next_rank] < end:
            selected[wanted[next_rank]] = value
            next_rank += 1
        offset = end
        if next_rank == len(wanted):
            break
    if next_rank != len(wanted):
        raise ValueError("ordered observation count changed during quantile evaluation")
    a = np.array([selected[int(i)] for i in lower])
    b = np.array([selected[int(i)] for i in upper])
    fraction = ranks - lower
    # Match NumPy's stable linear interpolation branch; the caller scales
    # extreme values before this operation so subtraction cannot overflow.
    distance = b - a
    return np.where(fraction >= 0.5, b - distance * (1 - fraction), a + distance * fraction)


def power_of_two(magnitude: float) -> float:
    return math.ldexp(1.0, math.frexp(magnitude)[1] - 1) if magnitude else 1.0


def scaled(value: float, origin: float, unit: float) -> float:
    shifted = value - origin
    exponent = math.frexp(unit)[1] - 1
    half = -exponent // 2
    result = shifted * math.ldexp(1.0, half) * math.ldexp(1.0, -exponent - half)
    if result * unit != shifted or shifted + origin != value:
        raise ValueError("observation dynamic range cannot be rescaled without precision loss")
    return result


def counted_mean(rows: CountStream) -> float:
    total = ExactFloatAccumulator()
    total.update_counted(rows)
    value = total.mean()
    if value is None:
        raise ValueError("mean requires a nonempty sample")
    return float(value)


@dataclass(frozen=True, slots=True)
class OutlierModel:
    """Constant-size model fitted to one complete group."""

    method: OutlierMethod
    threshold: float
    size: int
    origin: float
    unit: float
    centre: float
    spread: float
    lower: float
    upper: float
    median: float
    original_median: float
    testable: bool

    def score(self, value: float) -> tuple[float, bool]:
        y = scaled(value, self.origin, self.unit)
        if self.method == OutlierMethod.IQR:
            if y > self.upper:
                return ((y - self.upper) / self.spread if self.spread else math.inf), True
            if y < self.lower:
                return ((y - self.lower) / self.spread if self.spread else -math.inf), True
            return 0.0, False
        score = (y - self.centre) / self.spread if self.spread else 0.0
        return score, abs(score) > self.threshold

    def data_units(self, value: float) -> float:
        return value * self.unit + self.origin


def fit_outliers(
    rows: CountFactory,
    *,
    size: int,
    minimum: float,
    maximum: float,
    method: OutlierMethod | str,
    threshold: float | None,
    min_group_size: int,
    original_median: float,
    deviations: Callable[[float, float, float], CountStream],
) -> OutlierModel:
    """Fit without observation arrays; MAD order is provided by an external index."""
    method = OutlierMethod(method)
    limit = check_threshold(DEFAULT_THRESHOLDS[method] if threshold is None else threshold)
    needed = max(min_group_size, minimum_samples(method, limit))
    origin = minimum if minimum >= 0 else maximum if maximum <= 0 else 0.0
    if any((value - origin) + origin != value for value, _ in rows()):
        origin = 0.0
    unit = power_of_two(max(abs(minimum - origin), abs(maximum - origin)))

    def transformed():
        for value, count in rows():
            yield scaled(value, origin, unit), count

    def model(centre=0.0, spread=0.0, lower=0.0, upper=0.0, median=0.0, testable=False):
        return OutlierModel(
            method,
            limit,
            size,
            origin,
            unit,
            centre,
            spread,
            lower,
            upper,
            median,
            original_median,
            testable,
        )

    if size < needed:
        return model()
    try:
        # Validate every stored observation, including ones after the last rank.
        for _ in transformed():
            pass
        q1, median, q3 = quantiles(transformed(), size, (0.25, 0.5, 0.75))
        centre = float(median)
        if method == OutlierMethod.ZSCORE:
            centre = counted_mean(transformed())
            squared = ExactFloatAccumulator()
            squared.update_counted(((v - centre) ** 2, n) for v, n in transformed())
            total = squared.total()
            if total is None:
                raise ValueError("variance requires a nonempty sample")
            spread = math.sqrt(float(total / (size - 1)))
        elif method == OutlierMethod.MODIFIED_ZSCORE:
            mad = float(quantiles(deviations(centre, origin, unit), size, (0.5,))[0])
            spread = (
                mad / 0.6745
                if mad
                else 1.253314 * counted_mean((abs(v - centre), n) for v, n in transformed())
            )
        else:
            spread = float(q3 - q1)
        lower = (
            float(q1) - limit * spread if method == OutlierMethod.IQR else centre - limit * spread
        )
        upper = (
            float(q3) + limit * spread if method == OutlierMethod.IQR else centre + limit * spread
        )
        valid = all(math.isfinite(v) for v in (spread, lower, upper))
        return model(centre, spread, lower, upper, float(median), valid)
    except ValueError:
        # Grouped API marks unsupported dynamic ranges unassessed; peers remain
        # assessable. No approximated scores or silent clean result is produced.
        return model()
