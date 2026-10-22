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

import numpy as np

from qavf.inference._support import Unassessable, check_confidence, require_scipy

type Interval = tuple[float, float]


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
    low, high = np.quantile(distribution, [tail, 1 - tail])
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
    for values in jackknife:
        centred = values.mean() - values
        numerator += float(np.sum(centred**3)) / len(values) ** 3
        denominator += float(np.sum(centred**2)) / len(values) ** 2
    acceleration = numerator / (6 * denominator**1.5) if denominator > 0 else 0.0
    tail = (1 - level) / 2
    levels = []
    for z in (float(stats.norm.ppf(tail)), float(stats.norm.ppf(1 - tail))):
        shifted = z0 + z
        levels.append(float(stats.norm.cdf(z0 + shifted / (1 - acceleration * shifted))))
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in levels):
        raise Unassessable("BCa interval undefined: the acceleration is too large")
    low, high = np.quantile(distribution, levels)
    return float(low), float(high)
