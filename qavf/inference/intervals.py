"""Confidence intervals: analytic (t, normal) and bootstrap (percentile, BCa).

Analytic intervals take an estimate and its standard error. Bootstrap
intervals take the estimate, its bootstrap distribution and, for BCa, the
jackknife estimates of each sample (Efron's bias-corrected and accelerated
interval, with the per-sample acceleration SciPy uses for several samples).
Quantiles of a bootstrap distribution use NumPy's linear interpolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from qavf.inference._support import Unassessable, check_confidence, common_scale, require_scipy

type Interval = tuple[float, float]


def linear_quantiles(values: np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
    """Linear quantiles with exact interpolation when finite endpoints overflow.

    Keep NumPy's ordinary path and rank convention. Only a nonfinite interpolant
    needs rational arithmetic, so even opposite Float64 extremes stay bounded.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("quantiles need nonempty finite one-dimensional values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(np.quantile(values, probabilities))
    if np.isfinite(result).all():
        return result
    ordered = np.sort(values)
    result = result.copy()
    for index in np.flatnonzero(~np.isfinite(result)):
        rank = (len(ordered) - 1) * probabilities[int(index)]
        lower, upper = math.floor(rank), math.ceil(rank)
        weight = Fraction(float(rank - lower))
        a, b = Fraction(float(ordered[lower])), Fraction(float(ordered[upper]))
        result[index] = float(a + (b - a) * weight)
    return result


def t_interval(estimate: float, se: float, df: float, confidence: float = 0.95) -> Interval:
    """``estimate -/+ t(df) * se``; a zero ``se`` gives the point itself."""
    stats = require_scipy()
    level = check_confidence(confidence)
    if se == 0:
        return estimate, estimate
    half = float(stats.t.ppf(0.5 + level / 2, df)) * se
    return estimate - half, estimate + half


def normal_interval(estimate: float, se: float, confidence: float = 0.95) -> Interval:
    """``estimate -/+ z * se``."""
    stats = require_scipy()
    level = check_confidence(confidence)
    if se == 0:
        return estimate, estimate
    half = float(stats.norm.ppf(0.5 + level / 2)) * se
    return estimate - half, estimate + half


def percentile_interval(distribution: np.ndarray, confidence: float = 0.95) -> Interval:
    """The central ``confidence`` share of a bootstrap distribution."""
    level = check_confidence(confidence)
    tail = (1 - level) / 2
    low, high = linear_quantiles(distribution, [tail, 1 - tail])
    return float(low), float(high)


def bca_interval(
    estimate: float,
    distribution: np.ndarray,
    jackknife: Sequence[np.ndarray],
    confidence: float = 0.95,
) -> Interval:
    """Bias-corrected and accelerated bootstrap interval.

    ``jackknife`` holds, for each sample, the statistic with each of its
    observations left out in turn.

    Raises:
        Unassessable: when the bias correction is undefined (every bootstrap
            estimate on one side of the estimate).
    """
    stats = require_scipy()
    level = check_confidence(confidence)
    below = np.count_nonzero(distribution < estimate) + np.count_nonzero(distribution <= estimate)
    share = below / (2 * len(distribution))
    if not 0 < share < 1:
        raise Unassessable("BCa interval undefined: the bootstrap distribution lies on one side")
    z0 = float(stats.norm.ppf(share))
    numerator = denominator = 0.0
    scaled, _ = common_scale(*jackknife)
    for values in scaled:
        # U_ji / n_j, where U_ji=(n_j-1)*(theta_dot_j-theta_ji).
        # Per-sample factors cannot cancel when the sample sizes differ.
        influence = (1 - 1 / len(values)) * (values.mean() - values)
        numerator += float(np.sum(influence**3))
        denominator += float(np.sum(influence**2))
    acceleration = numerator / (6 * denominator**1.5) if denominator > 0 else 0.0
    tail = (1 - level) / 2
    levels = []
    for z in (float(stats.norm.ppf(tail)), float(stats.norm.ppf(1 - tail))):
        shifted = z0 + z
        levels.append(float(stats.norm.cdf(z0 + shifted / (1 - acceleration * shifted))))
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in levels):
        raise Unassessable("BCa interval undefined: the acceleration is too large")
    low, high = linear_quantiles(distribution, levels)
    return float(low), float(high)
