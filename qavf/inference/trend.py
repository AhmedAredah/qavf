"""Trends and series: monotone trends, slopes, correlations and lags.

* :func:`kendall_tau`: rank correlation of a value with time (TMP-002);
* :func:`mann_kendall`: the Mann-Kendall trend test with Sen's slope, and
  :func:`seasonal_kendall` for seasonal data (Hirsch, Slack and Smith 1982);
* :func:`slope_difference`: do two series on the same positions trend
  differently? The slope of their difference, with Newey-West (HAC) or
  block-bootstrap uncertainty for dependent buckets;
* :func:`correlation_test`: Fisher's z test of a correlation against a value,
  with Bartlett's effective sample size for autocorrelated series,
  :func:`rank_correlation_test` for Spearman's rank correlation (zones ranked
  alike by two versions), and :func:`compare_correlations` for two independent
  correlations;
* :func:`lag_test`: does a lagged pairing correlate better than lag 0? A
  circular block bootstrap of the correlation gain.

Series are paired by position (exact bucket ordinals). Constant series have
explicit outcomes: no trend (``p = 1``); a correlation with a constant series
is undefined, so not assessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal

import numpy as np
import numpy.typing as npt

from qavf.inference._support import (
    Unassessable,
    assessed,
    at_least,
    check_confidence,
    observations,
    require_scipy,
)
from qavf.inference.intervals import percentile_interval, t_interval
from qavf.inference.pvalues import normal_p_value, t_p_value
from qavf.inference.resampling import (
    DEFAULT_MAX_WORK,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    batches,
    block_indices,
    check_resamples,
    check_work,
    default_block_length,
    default_hac_lags,
    generator,
    interval_p_value,
    newey_west_slope,
    ols_slope,
)
from qavf.inference.results import Alternative, Dependence, TestResult
from qavf.stats import center_scale_observations


@assessed("kendall_tau")
def kendall_tau(
    x: npt.ArrayLike, y: npt.ArrayLike, *, alternative: Alternative = "two-sided"
) -> TestResult:
    """Kendall's tau-b of ``y`` against ``x`` with SciPy's p-value (``auto`` method).

    Dense integer ranks are formed without float conversion, preserving large
    integer and fine-resolution time order before SciPy evaluates the test.
    An undefined tau (a constant or missing value) is not assessed.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"unknown alternative {alternative!r}")
    stats = require_scipy()
    try:
        a, b = np.asarray(x), np.asarray(y)
    except (ValueError, TypeError):
        raise Unassessable("Kendall's tau needs paired one-dimensional observations") from None
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise Unassessable("Kendall's tau needs paired one-dimensional observations")
    for sample in (a, b):
        if sample.dtype.kind in "fc" and not np.isfinite(sample).all():
            raise Unassessable("Kendall's tau needs finite observations", n={"pairs": len(a)})
        if sample.dtype.kind in "mM" and np.isnat(sample).any():
            raise Unassessable("Kendall's tau needs nonmissing times", n={"pairs": len(a)})
        if sample.dtype.kind == "O":
            for value in sample:
                missing = (
                    value is None
                    or (isinstance(value, float | np.floating) and not math.isfinite(value))
                    or (isinstance(value, Decimal) and not value.is_finite())
                    or (isinstance(value, np.datetime64 | np.timedelta64) and np.isnat(value))
                )
                if missing:
                    raise Unassessable(
                        "Kendall's tau needs finite nonmissing observations", n={"pairs": len(a)}
                    )
    try:
        # Some SciPy releases coerce integer/time observations to floats before
        # ranking. Dense integer ranks preserve exact ties and order first.
        ranked_a = np.unique(a, return_inverse=True)[1]
        ranked_b = np.unique(b, return_inverse=True)[1]
        result = stats.kendalltau(ranked_a, ranked_b, alternative=alternative)
    except (ValueError, TypeError, ArithmeticError):
        raise Unassessable("Kendall's tau observations cannot be ordered") from None
    tau, p = float(result.statistic), float(result.pvalue)
    n = {"pairs": len(a)}
    if math.isnan(tau) or math.isnan(p):
        raise Unassessable("Kendall's tau is undefined (constant or missing values)", n=n)
    return TestResult(
        method="kendall_tau",
        statistic=tau,
        statistic_name="tau_b",
        p_value=p,
        alternative=alternative,
        effect=tau,
        effect_name="kendall_tau",
        n=n,
        assumptions=["independent observations"],
    )


def _ordered(
    values: npt.ArrayLike, positions: npt.ArrayLike | None
) -> tuple[np.ndarray, np.ndarray]:
    y = observations(values, "values")
    x = (
        np.arange(len(y), dtype=float)
        if positions is None
        else observations(positions, "positions")
    )
    if len(x) != len(y):
        raise Unassessable("positions and values differ in length")
    if len(np.unique(x)) < len(x):
        raise Unassessable("two values share one position")
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _s_and_variance(y: np.ndarray) -> tuple[int, float]:
    """Mann-Kendall ``S`` of an ordered series and its tie-corrected null variance."""
    s = 0
    for i in range(len(y) - 1):
        later = y[i + 1 :]
        s += int(np.count_nonzero(later > y[i])) - int(np.count_nonzero(later < y[i]))
    n = len(y)
    _, ties = np.unique(y, return_counts=True)
    variance = (n * (n - 1) * (2 * n + 5) - float((ties * (ties - 1) * (2 * ties + 5)).sum())) / 18
    return s, variance


def _z(s: int, variance: float) -> float:
    if s == 0 or variance == 0:
        return 0.0
    return (s - math.copysign(1, s)) / math.sqrt(variance)


def _slope_coordinates(x: np.ndarray, y: np.ndarray):
    """Preserve ranks and pairwise slopes without raw signed-range subtraction."""
    try:
        sx, ux, _ = center_scale_observations(x)
        sy, uy, _ = center_scale_observations(y)
    except ValueError as exc:
        raise Unassessable(str(exc)) from None
    return sx, sy, ux, uy


def _slope_units(slope: float, y_unit: float, x_unit: float) -> float:
    """Apply the ratio of exact binary units without an overflowing intermediate."""
    try:
        value = math.ldexp(float(slope), math.frexp(y_unit)[1] - math.frexp(x_unit)[1])
    except OverflowError:
        raise Unassessable("the slope cannot be represented in data units") from None
    if not math.isfinite(value):
        raise Unassessable("the slope cannot be represented in data units")
    return value


@assessed("mann_kendall")
def mann_kendall(
    values: npt.ArrayLike,
    positions: npt.ArrayLike | None = None,
    *,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
    max_work: int = DEFAULT_MAX_WORK,
) -> TestResult:
    """Mann-Kendall test of a monotone trend, with Sen's slope as the effect.

    ``S`` sums the signs of every later-minus-earlier difference; its variance
    is corrected for tied values and ``Z`` uses the continuity correction. The
    slope (per unit of ``positions``, per step without them) is the median of
    pairwise slopes, with its confidence interval (SciPy's ``theilslopes``). At
    least 3 values; ``n^2`` work.
    """
    check_confidence(confidence)
    x, y = _ordered(values, positions)
    n = {"values": len(y)}
    at_least(len(y), 3, "values", n=n)
    check_work(len(y) ** 2, max_work, "Mann-Kendall")
    stats = require_scipy()
    s, variance = _s_and_variance(y)
    z = _z(s, variance)
    sx, sy, x_unit, y_unit = _slope_coordinates(x, y)
    slope, _, low, high = stats.theilslopes(sy, sx, alpha=confidence)
    return TestResult(
        method="mann_kendall",
        statistic=float(s),
        statistic_name="S",
        p_value=normal_p_value(z, alternative) if variance else 1.0,
        alternative=alternative,
        effect=_slope_units(slope, y_unit, x_unit),
        effect_name="sen_slope",
        ci_low=_slope_units(low, y_unit, x_unit),
        ci_high=_slope_units(high, y_unit, x_unit),
        confidence=confidence,
        n=n,
        assumptions=["independent observations (no serial correlation)", "monotone trend"],
        notes=[f"Z = {z:.6g}, tau = {s / (len(y) * (len(y) - 1) / 2):.6g}"],
    )


@assessed("seasonal_kendall")
def seasonal_kendall(
    values: npt.ArrayLike,
    seasons: Sequence[object],
    positions: npt.ArrayLike | None = None,
    *,
    alternative: Alternative = "two-sided",
    max_work: int = DEFAULT_MAX_WORK,
) -> TestResult:
    """Seasonal Kendall test: Mann-Kendall within each season, summed.

    ``S`` and its tie-corrected variance add over seasons (seasons treated as
    independent); the effect is the seasonal Sen slope, the median of
    within-season pairwise slopes. Seasons with one value contribute nothing.
    """
    y = observations(values, "values")
    x = (
        np.arange(len(y), dtype=float)
        if positions is None
        else observations(positions, "positions")
    )
    if not len(seasons) == len(x) == len(y):
        raise Unassessable("one position and one season label are needed per value")
    if len(np.unique(x)) < len(x):
        raise Unassessable("two values share one position")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    labels = [seasons[i] for i in order]
    groups: dict[object, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    n = {"values": len(y), "seasons": len(groups)}
    pairs = sum(len(g) * (len(g) - 1) // 2 for g in groups.values())
    if not pairs:
        raise Unassessable("no season has two values", n=n)
    check_work(pairs, max_work, "seasonal Kendall")
    x, y, x_unit, y_unit = _slope_coordinates(x, y)
    s, variance = 0, 0.0
    slopes = []
    for members in groups.values():
        if len(members) < 2:
            continue
        xs, ys = x[members], y[members]
        part, var = _s_and_variance(ys)
        s += part
        variance += var
        for i in range(len(xs) - 1):
            slopes.append((ys[i + 1 :] - ys[i]) / (xs[i + 1 :] - xs[i]))
    z = _z(s, variance)
    return TestResult(
        method="seasonal_kendall",
        statistic=float(s),
        statistic_name="S",
        p_value=normal_p_value(z, alternative) if variance else 1.0,
        alternative=alternative,
        effect=_slope_units(float(np.median(np.concatenate(slopes))), y_unit, x_unit),
        effect_name="seasonal_sen_slope",
        n=n,
        assumptions=["independent seasons", "no serial correlation within a season"],
        notes=[f"Z = {z:.6g}"],
    )


@assessed("slope_difference")
def slope_difference(
    positions: npt.ArrayLike,
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    dependence: Dependence = "block",
    lags: int | None = None,
    block: int | None = None,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
) -> TestResult:
    """Difference of the least-squares slopes of two series sharing ``positions``.

    With common positions the slope difference is exactly the slope of
    ``candidate - baseline``, so its uncertainty comes from one regression:
    ordinary least squares (``dependence='none'``), Newey-West HAC standard
    errors (``'hac'``; Student t with ``n - 2`` degrees of freedom), or a
    null-imposed circular block bootstrap (``'block'``, the default): blocks of
    the centred difference, at the fixed positions, give the slope's
    distribution without a trend. Variances from block lengths ``l`` and
    ``min(2*l,n)`` are extrapolated as ``max(2*v_2l-v_l,v_2l)``. Its square root
    is a standard error, used with Student t(``n-2``) for both the p-value and
    interval; this is not a permutation-tail p-value or percentile interval.
    Positions are sorted before dependence calculations; lag/block adjacency
    refers to consecutive observed positions. HAC errors are known to be liberal
    for short autocorrelated series; the block bootstrap holds its level
    better. At least 4 positions.
    """
    check_confidence(confidence)
    if dependence not in ("none", "hac", "block"):
        raise ValueError(f"dependence must be 'none', 'hac' or 'block', got {dependence!r}")
    x = observations(positions, "positions")
    b, c = observations(baseline, "baseline"), observations(candidate, "candidate")
    if not len(x) == len(b) == len(c):
        raise Unassessable("positions and both series must have one value each per position")
    n = {"positions": len(x)}
    at_least(len(x), 4, "positions", n=n)
    if len(np.unique(x)) < len(x):
        raise Unassessable("two values share one position", n=n)
    # Dependence follows the declared chronology, not the caller's row order.
    order = np.argsort(x, kind="stable")
    x, b, c = x[order], b[order], c[order]
    try:
        sx, x_unit, _ = center_scale_observations(x)
        joined, unit, _ = center_scale_observations(np.concatenate([b, c]))
    except ValueError as exc:
        raise Unassessable(str(exc), n=n) from None
    d = joined[len(b) :] - joined[: len(b)]
    slope, _, _, residual = ols_slope(sx, d)
    common = {
        "method": "slope_difference",
        "statistic_name": "t",
        "alternative": alternative,
        "effect": _slope_units(slope, unit, x_unit),
        "effect_name": "slope_difference",
        "confidence": confidence,
        "n": n,
        "dependence": dependence,
    }
    df = len(x) - 2
    if not np.any(residual):
        t = 0.0 if slope == 0 else math.copysign(math.inf, slope)
        point = _slope_units(slope, unit, x_unit)
        return TestResult(
            statistic=t, p_value=1.0 if slope == 0 else t_p_value(t, df, alternative),
            ci_low=point, ci_high=point, notes=["the difference is exactly linear"], **common,
        )  # fmt: skip
    resampled: dict[str, int] = {}
    if dependence == "block":
        length = min(len(x), default_block_length(len(x)) if block is None else block)
        longer = min(len(x), 2 * length)
        check_resamples(resamples)
        check_work(2 * resamples * len(x), max_work, "block bootstrap")
        # Null-imposed: blocks of the centred difference (no trend) at the fixed
        # positions give the slope's spread without a trend. (Residuals around the
        # fitted line would lose the low-frequency variation the slope absorbed.)
        rng = generator(seed)
        xc = sx - sx.mean()
        centred = d - d.mean()
        variances = []
        for size in (length, longer):
            slopes = np.empty(resamples)
            done = 0
            for step in batches(resamples, len(x)):
                errors = centred[block_indices(rng, len(x), step, size)]
                slopes[done : done + step] = (
                    (errors - errors.mean(axis=1, keepdims=True)) @ xc / float(xc @ xc)
                )
                done += step
            variances.append(float(slopes.var(ddof=1)))
        # A block bootstrap variance is biased low by O(1/l); extrapolating from
        # blocks of l and 2l removes the leading term (Richardson extrapolation).
        se = math.sqrt(max(2 * variances[1] - variances[0], variances[1]))
        assumption = (
            f"null-imposed circular block bootstrap standard error, blocks of {length} "
            f"and {longer} extrapolated"
        )
        resampled = {"seed": seed, "resamples": resamples}
    elif dependence == "hac":
        _, se = newey_west_slope(sx, d, lags)
        used = default_hac_lags(len(x)) if lags is None else lags
        assumption = f"Newey-West HAC errors, {used} lag(s)"
    else:
        se = math.sqrt(float(residual @ residual) / df / float(((sx - sx.mean()) ** 2).sum()))
        assumption = "independent residuals"
    t = slope / se if se else (math.copysign(math.inf, slope) if slope else 0.0)
    low, high = t_interval(slope, se, df, confidence)
    return TestResult(
        statistic=t,
        p_value=t_p_value(t, df, alternative),
        ci_low=_slope_units(low, unit, x_unit),
        ci_high=_slope_units(high, unit, x_unit),
        assumptions=[assumption, "linear trends"],
        **common,
        **resampled,
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    xc, yc = x - x.mean(), y - y.mean()
    spread = float(math.sqrt((xc @ xc) * (yc @ yc)))
    return None if spread == 0 else float(xc @ yc) / spread


def effective_size(
    x: np.ndarray, y: np.ndarray, lags: int | None = None, *, max_work: int = DEFAULT_MAX_WORK
) -> float:
    """Bartlett's effective sample size of a correlation of two autocorrelated series:
    ``n / (1 + 2 * sum_k r_x(k) r_y(k))`` over lags ``1..lags`` (``n // 5`` by
    default), never above ``n``."""
    n = len(x)
    lags = max(1, n // 5) if lags is None else lags
    if isinstance(lags, bool) or not isinstance(lags, int | np.integer) or lags < 0:
        raise ValueError("lags must be a non-negative integer")
    used = min(lags, max(0, n - 2))
    # Two autocorrelations each touch both overlapping slices at every lag.
    check_work(
        4 * (used * n - used * (used + 1) // 2), max_work, "effective sample size lag correlations"
    )
    total = 0.0
    for k in range(1, min(lags, n - 2) + 1):
        rx, ry = _pearson(x[:-k], x[k:]), _pearson(y[:-k], y[k:])
        if rx is not None and ry is not None:
            total += rx * ry
    return min(float(n), n / max(1 + 2 * total, 1e-12))


def _fisher(r: float) -> float:
    return math.copysign(math.inf, r) if abs(r) >= 1 else math.atanh(r)


@assessed("fisher_z")
def correlation_test(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    rho: float = 0.0,
    alternative: Alternative = "two-sided",
    dependence: Dependence = "none",
    confidence: float = 0.95,
    max_work: int = DEFAULT_MAX_WORK,
) -> TestResult:
    """Fisher's z test of Pearson's correlation against ``rho``.

    ``z = (atanh(r) - atanh(rho)) * sqrt(n - 3)``; with
    ``dependence='effective_n'`` the sample size is Bartlett's effective size
    (autocorrelated series carry less information than their length). ``less``
    tests whether the correlation is below ``rho``. At least 4 pairs; an
    effective size of 3 or less, or a constant series, is not assessed.
    """
    check_confidence(confidence)
    if not -1 < rho < 1:
        raise ValueError(f"rho must lie in (-1, 1), got {rho!r}")
    if dependence not in ("none", "effective_n"):
        raise ValueError(f"dependence must be 'none' or 'effective_n', got {dependence!r}")
    a, b = observations(x, "x"), observations(y, "y")
    if len(a) != len(b):
        raise Unassessable("a correlation needs paired values")
    n: dict[str, int] = {"pairs": len(a)}
    at_least(len(a), 4, "pairs", n=n)
    try:
        sa, sb = center_scale_observations(a)[0], center_scale_observations(b)[0]
    except ValueError as exc:
        raise Unassessable(str(exc), n=n) from None
    r = _pearson(sa, sb)
    if r is None:
        raise Unassessable("a series is constant; the correlation is undefined", n=n)
    size = (
        effective_size(sa, sb, max_work=max_work) if dependence == "effective_n" else float(len(a))
    )
    if size <= 3:
        raise Unassessable(f"the effective sample size {size:.3g} is too small", n=n)
    root = math.sqrt(size - 3)
    gap = _fisher(r) - math.atanh(rho)
    z = gap * root if math.isfinite(gap) else gap
    half = float(require_scipy().norm.ppf(0.5 + confidence / 2)) / root
    centre = _fisher(r)
    notes = [f"effective sample size {size:.4g}"] if dependence == "effective_n" else []
    return TestResult(
        method="fisher_z",
        statistic=z,
        statistic_name="z",
        p_value=normal_p_value(z, alternative),
        alternative=alternative,
        effect=r,
        effect_name="pearson_r",
        ci_low=math.tanh(centre - half),
        ci_high=math.tanh(centre + half),
        confidence=confidence,
        n=n,
        assumptions=["approximately bivariate normal", f"null correlation {rho:g}"],
        dependence=dependence,
        notes=notes,
    )


FIELLER_VARIANCE = 1.06
"""Variance factor of Fisher's z of Spearman's correlation (Fieller, Hartley and
Pearson 1957): ``var(atanh(r_s)) ~ 1.06 / (n - 3)``."""


@assessed("spearman")
def rank_correlation_test(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    rho: float = 0.0,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TestResult:
    """Spearman's rank correlation tested against ``rho`` by Fisher's z.

    ``z = (atanh(r_s) - atanh(rho)) * sqrt((n - 3) / 1.06)``, the Fieller,
    Hartley and Pearson variance of a rank correlation (ties take average
    ranks). ``less`` tests whether the correlation is below ``rho``. At least 4
    pairs; a constant side is not assessed.
    """
    check_confidence(confidence)
    if not -1 < rho < 1:
        raise ValueError(f"rho must lie in (-1, 1), got {rho!r}")
    a, b = observations(x, "x"), observations(y, "y")
    if len(a) != len(b):
        raise Unassessable("a correlation needs paired values")
    n: dict[str, int] = {"pairs": len(a)}
    at_least(len(a), 4, "pairs", n=n)
    stats = require_scipy()
    r = _pearson(stats.rankdata(a), stats.rankdata(b))
    if r is None:
        raise Unassessable("a side is constant; the rank correlation is undefined", n=n)
    root = math.sqrt((len(a) - 3) / FIELLER_VARIANCE)
    gap = _fisher(r) - math.atanh(rho)
    z = gap * root if math.isfinite(gap) else gap
    half = float(stats.norm.ppf(0.5 + confidence / 2)) / root
    centre = _fisher(r)
    return TestResult(
        method="spearman",
        statistic=z,
        statistic_name="z",
        p_value=normal_p_value(z, alternative),
        alternative=alternative,
        effect=r,
        effect_name="spearman_rho",
        ci_low=math.tanh(centre - half),
        ci_high=math.tanh(centre + half),
        confidence=confidence,
        n=n,
        assumptions=[
            "independent pairs",
            f"Fieller-Hartley-Pearson variance {FIELLER_VARIANCE}/(n - 3)",
            f"null correlation {rho:g}",
        ],
    )


@assessed("fisher_z")
def compare_correlations(
    baseline: tuple[float, int],
    candidate: tuple[float, int],
    *,
    alternative: Alternative = "two-sided",
    variance: float = 1.0,
) -> TestResult:
    """Fisher's z test of two independent correlations ``(r, n)``: does the
    candidate's differ from the baseline's? At least 4 pairs each.

    ``variance`` scales ``var(atanh(r)) = variance / (n - 3)``: 1 for Pearson's
    correlation, :data:`FIELLER_VARIANCE` (1.06) for Spearman's."""
    (rb, nb), (rc, nc) = baseline, candidate
    n = {"baseline": nb, "candidate": nc}
    if nb < 4 or nc < 4:
        raise Unassessable("each correlation needs at least 4 pairs", n=n)
    if not (-1 <= rb <= 1 and -1 <= rc <= 1):
        raise ValueError("correlations must lie in [-1, 1]")
    gap = _fisher(rc) - _fisher(rb)
    if variance <= 0:
        raise ValueError("the variance factor must be positive")
    se = math.sqrt(variance * (1 / (nb - 3) + 1 / (nc - 3)))
    z = gap / se if math.isfinite(gap) else (0.0 if math.isnan(gap) else gap)
    return TestResult(
        method="fisher_z",
        statistic=z,
        statistic_name="z",
        p_value=normal_p_value(z, alternative),
        alternative=alternative,
        effect=rc - rb,
        effect_name="correlation_difference",
        n=n,
        assumptions=[
            "independent samples",
            "approximately bivariate normal"
            if variance == 1
            else f"Fieller-Hartley-Pearson variance {variance:g}/(n - 3)",
        ],
    )


@assessed("lag")
def lag_test(
    baseline: Mapping[int, float],
    candidate: Mapping[int, float],
    lag: int,
    *,
    period: int | None = None,
    block: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
    minimum: int = 4,
) -> TestResult:
    """Does pairing ``baseline[t]`` with ``candidate[t + lag]`` correlate better than lag 0?

    On positions where all three values exist, the gain ``r(lag) - r(0)`` is
    resampled by a circular block bootstrap of those positions; the one-sided
    p-value is ``(#(gain* <= 0) + 1) / (B + 1)``. Positions wrap modulo
    ``period`` for cyclic profiles. At least ``minimum`` common positions.
    """
    if lag == 0:
        raise ValueError("lag 0 is the reference; test a nonzero lag")
    positions = []
    for t in sorted(baseline):
        other = (t + lag) % period if period is not None else t + lag
        if t in candidate and other in candidate:
            positions.append((baseline[t], candidate[t], candidate[other]))
    n = {"positions": len(positions)}
    at_least(len(positions), minimum, "common positions", n=n)
    triples = np.array(positions, dtype=float)
    try:
        scaled = np.column_stack([center_scale_observations(column)[0] for column in triples.T])
    except ValueError as exc:
        raise Unassessable(str(exc), n=n) from None
    x, y0, yk = scaled.T
    at_zero, at_lag = _pearson(x, y0), _pearson(x, yk)
    if at_zero is None or at_lag is None:
        raise Unassessable("a series is constant; correlations are undefined", n=n)
    observed = at_lag - at_zero
    length = default_block_length(len(x)) if block is None else block
    check_resamples(resamples)
    check_work(resamples * len(x), max_work, "lag block bootstrap")

    def correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        spread = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (ac * bc).sum(axis=1) / spread

    gain = np.empty(resamples)
    rng = generator(seed)
    done = 0
    for step in batches(resamples, len(x)):
        indices = block_indices(rng, len(x), step, length)
        gain[done : done + step] = correlations(x[indices], yk[indices]) - correlations(
            x[indices], y0[indices]
        )
        done += step
    gain = np.where(np.isnan(gain), 0.0, gain)  # an undefined resample counts as no gain
    low, high = percentile_interval(gain, 0.95)
    return TestResult(
        method="lag",
        statistic=observed,
        statistic_name="correlation_gain",
        p_value=interval_p_value(gain, "greater"),
        alternative="greater",
        effect=observed,
        effect_name="correlation_gain",
        ci_low=low,
        ci_high=high,
        confidence=0.95,
        n=n,
        assumptions=[f"circular block bootstrap, block length {length}"],
        dependence="block",
        seed=seed,
        resamples=resamples,
        notes=[f"correlation {at_lag:.6g} at lag {lag}, {at_zero:.6g} at lag 0"],
    )
