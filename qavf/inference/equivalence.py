"""Equivalence: show that a difference lies within a margin (two one-sided tests).

TOST rejects "the difference is at most ``low``" and "the difference is at
least ``high``" with one-sided t-tests; its p-value is the larger of the two,
so a small p-value means equivalence within ``(low, high)`` is shown. A
symmetric ``margin`` ``m`` means ``(-m, m)``. The interval reported is the
``confidence`` interval (``1 - 2 alpha``, 0.90 by default) that TOST implies:
equivalence at level alpha holds exactly when it lies inside the margin.

A difference without spread (constant differences, constant samples) is
equivalent when it lies strictly inside the margin (``p = 0``) and not
otherwise (``p = 1``).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from qavf.inference._support import (
    Unassessable,
    assessed,
    at_least,
    check_confidence,
    common_scale,
    require_scipy,
)
from qavf.inference.independent import (
    Moments,
    _two,
    mean_statistic,
    resampled_difference,
    welch_terms,
)
from qavf.inference.intervals import percentile_interval, t_interval
from qavf.inference.paired import differences
from qavf.inference.resampling import (
    DEFAULT_MAX_WORK,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Statistic,
)
from qavf.inference.results import TestResult

type Margin = float | tuple[float, float]


def bounds(margin: Margin) -> tuple[float, float]:
    """``(low, high)`` of a margin; a single number ``m`` means ``(-m, m)``."""
    low, high = (-margin, margin) if isinstance(margin, int | float) else margin
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"an equivalence margin needs finite low <= high, got {margin!r}")
    return float(low), float(high)


def _tost(
    method: str,
    difference: float,
    se: float,
    df: float,
    margin: Margin,
    unit: float,
    confidence: float,
    n: dict[str, int],
    assumptions: list[str],
) -> TestResult:
    low, high = bounds(margin)
    low, high = low / unit, high / unit
    notes = []
    if se == 0:
        notes.append("the difference has no spread")
        inside = low < difference < high
        p, statistic = (0.0 if inside else 1.0), math.inf if inside else -math.inf
    else:
        stats = require_scipy()
        lower = (difference - low) / se
        upper = (difference - high) / se
        p_lower, p_upper = float(stats.t.sf(lower, df)), float(stats.t.cdf(upper, df))
        p, statistic = (p_lower, lower) if p_lower >= p_upper else (p_upper, upper)
    ci_low, ci_high = t_interval(difference, se, df, confidence)
    return TestResult(
        method=method,
        statistic=statistic,
        statistic_name="t",
        p_value=p,
        effect=difference * unit,
        effect_name="mean_difference",
        ci_low=ci_low * unit,
        ci_high=ci_high * unit,
        confidence=confidence,
        n=n,
        assumptions=[*assumptions, f"equivalence margin ({low * unit:g}, {high * unit:g})"],
        notes=notes,
    )


@assessed("tost_paired")
def tost_paired(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, margin: Margin, *, confidence: float = 0.90
) -> TestResult:
    """Paired TOST: the mean of ``candidate - baseline`` lies within the margin (2+ pairs)."""
    check_confidence(confidence)
    bounds(margin)
    d, unit = differences(baseline, candidate)
    at_least(len(d), 2, "pairs", n={"pairs": len(d)})
    se = float(d.std(ddof=1)) / math.sqrt(len(d))
    return _tost(
        "tost_paired", float(d.mean()), se, len(d) - 1, margin, unit, confidence,
        {"pairs": len(d)}, ["independent pairs", "approximately normal mean difference"],
    )  # fmt: skip


def _independent(b: Moments, c: Moments, margin: Margin, unit: float, confidence: float):
    n = {"baseline": b.n, "candidate": c.n}
    if b.n < 2 or c.n < 2:
        raise Unassessable("TOST needs at least 2 values on each side", n=n)
    se, df = welch_terms(b, c)
    return _tost(
        "tost_welch", c.mean - b.mean, se, df, margin, unit, confidence, n,
        ["independent observations", "approximately normal means"],
    )  # fmt: skip


@assessed("tost_welch")
def tost_independent(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, margin: Margin, *, confidence: float = 0.90
) -> TestResult:
    """Welch TOST: the difference of means lies within the margin (2+ values each)."""
    check_confidence(confidence)
    bounds(margin)
    b, c = _two(baseline, candidate, 2)
    (sb, sc), unit = common_scale(b, c)
    return _independent(Moments.of(sb), Moments.of(sc), margin, unit, confidence)


@assessed("tost_welch")
def tost_from_moments(
    baseline: Moments, candidate: Moments, margin: Margin, *, confidence: float = 0.90
) -> TestResult:
    """Welch TOST from each side's size, mean and sample variance."""
    check_confidence(confidence)
    bounds(margin)
    values = (baseline.mean, candidate.mean, baseline.variance, candidate.variance)
    if not np.isfinite(values).all():
        raise Unassessable("the sample moments are not finite")
    return _independent(baseline, candidate, margin, 1.0, confidence)


@assessed("bootstrap_equivalence")
def bootstrap_equivalence(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    margin: Margin,
    *,
    statistic: Statistic = mean_statistic,
    name: str = "mean",
    confidence: float = 0.90,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
    minimum: int = 2,
) -> TestResult:
    """Bootstrap TOST of ``statistic(candidate) - statistic(baseline)`` within the margin.

    Each one-sided p-value is the bootstrap share beyond its bound,
    ``(#(d* <= low) + 1) / (B + 1)`` and ``(#(d* >= high) + 1) / (B + 1)``; the
    larger is the p-value, so equivalence at alpha holds when the ``1 - 2 alpha``
    percentile interval lies inside the margin.
    """
    check_confidence(confidence)
    low, high = bounds(margin)
    b, c = _two(baseline, candidate, minimum)
    observed, distribution = resampled_difference(b, c, statistic, resamples, seed, max_work)
    p_low = (np.count_nonzero(distribution <= low) + 1) / (resamples + 1)
    p_high = (np.count_nonzero(distribution >= high) + 1) / (resamples + 1)
    ci_low, ci_high = percentile_interval(distribution, confidence)
    return TestResult(
        method="bootstrap_equivalence",
        statistic=observed,
        statistic_name=f"{name}_difference",
        p_value=float(max(p_low, p_high)),
        effect=observed,
        effect_name=f"{name}_difference",
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n={"baseline": len(b), "candidate": len(c)},
        assumptions=["independent observations", f"equivalence margin ({low:g}, {high:g})"],
        seed=seed,
        resamples=resamples,
    )
