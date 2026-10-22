"""Two independent samples: a baseline and a candidate sample of the same quantity.

* :func:`welch_t`: difference of means without assuming equal variances (also
  from summary :class:`Moments`, so means of large groups need no raw values);
* :func:`mann_whitney`: whether one sample tends to be larger (rank-sum);
* :func:`ks_two_sample` and :func:`anderson_darling`: any difference of
  distribution;
* :func:`brown_forsythe`: difference of spread (Levene's test around medians);
* :func:`permutation` and :func:`bootstrap_difference`: difference of any
  statistic by resampling, seeded and bounded by the work budget.

Effects are oriented candidate minus baseline. Constant samples have explicit
outcomes: two equal constants do not differ (``p = 1``), two different
constants differ with certainty (``p = 0``) for the location tests.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from qavf.inference._support import (
    Unassessable,
    assessed,
    at_least,
    check_confidence,
    common_scale,
    common_unit,
    observations,
    require_scipy,
    sizes,
)
from qavf.inference.effects import rank_biserial
from qavf.inference.intervals import bca_interval, percentile_interval, t_interval
from qavf.inference.pvalues import t_p_value
from qavf.inference.resampling import (
    DEFAULT_MAX_WORK,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Statistic,
    batches,
    bootstrap_distribution,
    check_work,
    generator,
    interval_p_value,
    permutation_p_value,
)
from qavf.inference.results import Alternative, TestResult


@dataclass(frozen=True, slots=True)
class Moments:
    """Size, mean and sample variance (``ddof=1``) of one sample."""

    n: int
    mean: float
    variance: float

    @classmethod
    def of(cls, values: np.ndarray) -> Moments:
        return cls(len(values), float(values.mean()), float(values.var(ddof=1)))


def _two(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, minimum: int
) -> tuple[np.ndarray, np.ndarray]:
    b, c = observations(baseline, "baseline"), observations(candidate, "candidate")
    n = sizes(baseline=b, candidate=c)
    at_least(len(b), minimum, "baseline", n=n)
    at_least(len(c), minimum, "candidate", n=n)
    return b, c


def welch_terms(b: Moments, c: Moments) -> tuple[float, float]:
    """Standard error of the difference of means and Welch-Satterthwaite degrees of
    freedom (``n_b + n_c - 2`` when both samples are constant)."""
    se2 = b.variance / b.n + c.variance / c.n
    if se2 == 0:
        return 0.0, float(b.n + c.n - 2)
    df = se2**2 / ((b.variance / b.n) ** 2 / (b.n - 1) + (c.variance / c.n) ** 2 / (c.n - 1))
    return math.sqrt(se2), df


def _welch(
    b: Moments, c: Moments, unit: float, alternative: Alternative, confidence: float, method: str
) -> TestResult:
    n = {"baseline": b.n, "candidate": c.n}
    if b.n < 2 or c.n < 2:
        raise Unassessable("Welch's t needs at least 2 values on each side", n=n)
    if not all(math.isfinite(v) for v in (b.mean, c.mean, b.variance, c.variance)):
        raise Unassessable("the sample moments are not finite", n=n)
    difference = c.mean - b.mean
    se, df = welch_terms(b, c)
    notes = []
    if se == 0:
        notes.append("both samples are constant")
        t = 0.0 if difference == 0 else math.copysign(math.inf, difference)
        p = 1.0 if difference == 0 else t_p_value(t, df, alternative)
    else:
        stats = require_scipy()
        result = stats.ttest_ind_from_stats(
            c.mean, math.sqrt(c.variance), c.n, b.mean, math.sqrt(b.variance), b.n,
            equal_var=False, alternative=alternative,
        )  # fmt: skip
        t, p = float(result.statistic), float(result.pvalue)
    low, high = t_interval(difference, se, df, confidence)
    return TestResult(
        method=method,
        statistic=t,
        statistic_name="t",
        p_value=p,
        alternative=alternative,
        effect=difference * unit,
        effect_name="mean_difference",
        ci_low=low * unit,
        ci_high=high * unit,
        confidence=confidence,
        n=n,
        assumptions=[
            "independent observations",
            "approximately normal means (central limit theorem for large samples)",
        ],
        notes=notes,
    )


@assessed("welch_t")
def welch_t(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TestResult:
    """Welch's unequal-variance t-test of the difference of means (at least 2 values each)."""
    check_confidence(confidence)
    b, c = _two(baseline, candidate, 2)
    (sb, sc), unit = common_scale(b, c)
    return _welch(Moments.of(sb), Moments.of(sc), unit, alternative, confidence, "welch_t")


@assessed("welch_t")
def welch_t_from_moments(
    baseline: Moments,
    candidate: Moments,
    *,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TestResult:
    """Welch's t-test from each side's size, mean and sample variance."""
    check_confidence(confidence)
    return _welch(baseline, candidate, 1.0, alternative, confidence, "welch_t")


@assessed("mann_whitney")
def mann_whitney(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, *, alternative: Alternative = "two-sided"
) -> TestResult:
    """Mann-Whitney U (Wilcoxon rank-sum): does the candidate tend to be larger or smaller?

    Exact when a side has at most 8 values and there are no ties, otherwise the
    normal approximation with tie and continuity corrections (SciPy's rules,
    chosen explicitly so results do not depend on the SciPy version). At least 2
    values on each side; all values tied gives ``p = 1``.
    """
    b, c = _two(baseline, candidate, 2)
    stats = require_scipy()
    n = sizes(baseline=b, candidate=c)
    effect = rank_biserial(b, c)
    common = {
        "method": "mann_whitney",
        "statistic_name": "U",
        "alternative": alternative,
        "effect": effect,
        "effect_name": "rank_biserial",
        "n": n,
        "assumptions": ["independent observations", "ordinal values"],
    }
    joined = np.concatenate([b, c])
    if (joined == joined[0]).all():
        return TestResult(
            statistic=len(b) * len(c) / 2, p_value=1.0, notes=["all values are tied"], **common
        )
    ties = len(np.unique(joined)) < len(joined)
    method = "exact" if min(len(b), len(c)) <= 8 and not ties else "asymptotic"
    result = stats.mannwhitneyu(c, b, alternative=alternative, method=method)
    return TestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        notes=[f"{method} p-value" + (" with tie correction" if ties else "")],
        **common,
    )


def ks_asymptotic_p_value(statistic: float, sizes: tuple[int, int]) -> float:
    """Two-sided two-sample KS p-value from the exact null distribution of the
    one-sample statistic at ``n = round(m * n / (m + n))``, as SciPy's
    ``ks_2samp`` does for samples too large for its exact method."""
    stats = require_scipy()
    m, n = sorted(map(float, sizes), reverse=True)
    probability = float(stats.kstwo.sf(statistic, np.round(m * n / (m + n))))
    return min(1.0, max(0.0, probability))


@assessed("ks")
def ks_two_sample(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> TestResult:
    """Two-sample Kolmogorov-Smirnov test (two-sided; SciPy's ``auto`` method).

    The statistic, the largest gap between the empirical distribution
    functions, is also the effect. At least 2 values on each side.
    """
    b, c = _two(baseline, candidate, 2)
    stats = require_scipy()
    result = stats.ks_2samp(b, c, method="auto")
    return TestResult(
        method="ks",
        statistic=float(result.statistic),
        statistic_name="D",
        p_value=float(result.pvalue),
        effect=float(result.statistic),
        effect_name="ks_distance",
        n=sizes(baseline=b, candidate=c),
        assumptions=[
            "independent observations",
            "continuous distributions (ties are conservative)",
        ],
    )


@assessed("anderson_darling")
def anderson_darling(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> TestResult:
    """Anderson-Darling k-sample test of a common distribution (two samples).

    More sensitive in the tails than KS. SciPy interpolates the p-value from
    tabulated critical values and caps it to [0.001, 0.25]; a capped p-value is
    noted. At least 2 values on each side; all values tied gives ``p = 1``.
    """
    b, c = _two(baseline, candidate, 2)
    stats = require_scipy()
    n = sizes(baseline=b, candidate=c)
    joined = np.concatenate([b, c])
    if (joined == joined[0]).all():
        return TestResult(
            method="anderson_darling", statistic=0.0, statistic_name="A2",
            p_value=1.0, n=n, notes=["all values are tied"],
        )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.anderson_ksamp([b, c])
    p = float(result.pvalue)
    notes = ["p-value capped by SciPy's table range [0.001, 0.25]"] if p in (0.001, 0.25) else []
    return TestResult(
        method="anderson_darling",
        statistic=float(result.statistic),
        statistic_name="A2",
        p_value=p,
        n=n,
        assumptions=["independent observations"],
        notes=notes,
    )


@assessed("brown_forsythe")
def brown_forsythe(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> TestResult:
    """Brown-Forsythe test of equal spread (Levene's test around the medians).

    The effect is the ratio of standard deviations (candidate over baseline).
    At least 2 values on each side. Absolute deviations that are all equal
    within each side but differ between sides give ``p = 0``; no spread at all
    gives ``p = 1``.
    """
    b, c = _two(baseline, candidate, 2)
    stats = require_scipy()
    (sb, sc), _ = common_scale(b, c)
    deviations = [np.abs(s - np.median(s)) for s in (sb, sc)]
    within = sum(float(((d - d.mean()) ** 2).sum()) for d in deviations)
    spread_b, spread_c = float(sb.std(ddof=1)), float(sc.std(ddof=1))
    common = {
        "method": "brown_forsythe",
        "statistic_name": "W",
        "effect": spread_c / spread_b if spread_b else None,
        "effect_name": "sd_ratio",
        "n": sizes(baseline=b, candidate=c),
        "assumptions": ["independent observations"],
    }
    if within == 0:
        same = deviations[0].mean() == deviations[1].mean()
        return TestResult(
            statistic=0.0 if same else math.inf,
            p_value=1.0 if same else 0.0,
            notes=["absolute deviations are constant within each side"],
            **common,
        )
    result = stats.levene(sb, sc, center="median")
    return TestResult(statistic=float(result.statistic), p_value=float(result.pvalue), **common)


def mean_statistic(values: np.ndarray) -> np.ndarray:
    """The mean along the last axis."""
    return values.mean(axis=-1)


@assessed("permutation")
def permutation(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    statistic: Statistic = mean_statistic,
    alternative: Alternative = "two-sided",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
) -> TestResult:
    """Permutation test of ``statistic(candidate) - statistic(baseline)`` (means by default).

    Exact over every relabelling when there are no more of them than
    ``resamples``; otherwise ``resamples`` seeded random relabellings with the
    ``(k + 1) / (B + 1)`` p-value. At least 2 values on each side.
    """
    b, c = _two(baseline, candidate, 2)
    (sb, sc), unit = common_unit(b, c)
    joined = np.concatenate([sb, sc])
    n = sizes(baseline=b, candidate=c)
    observed = float(statistic(sc) - statistic(sb))
    labellings = math.comb(len(joined), len(sc))
    exact = labellings <= resamples
    check_work((labellings if exact else resamples) * len(joined), max_work, "permutation")
    if exact:
        picks = np.array(list(combinations(range(len(joined)), len(sc))))
        mask = np.zeros((len(picks), len(joined)), dtype=bool)
        np.put_along_axis(mask, picks, True, axis=1)
        values = np.broadcast_to(joined, mask.shape)
        drawn_c = values[mask].reshape(len(picks), len(sc))
        drawn_b = values[~mask].reshape(len(picks), len(sb))
        distribution = statistic(drawn_c) - statistic(drawn_b)
        p = permutation_p_value(distribution, observed, alternative, exact=True)
    else:
        rng = generator(seed)
        distribution = np.empty(resamples)
        done = 0
        for step in batches(resamples, len(joined)):
            order = rng.permuted(np.broadcast_to(joined, (step, len(joined))), axis=1)
            distribution[done : done + step] = statistic(order[:, len(sb) :]) - statistic(
                order[:, : len(sb)]
            )
            done += step
        p = permutation_p_value(distribution, observed, alternative)
    return TestResult(
        method="permutation",
        statistic=observed * unit,
        statistic_name="difference",
        p_value=p,
        alternative=alternative,
        effect=observed * unit,
        effect_name="difference",
        n=n,
        assumptions=["exchangeable observations under no difference"],
        seed=None if exact else seed,
        resamples=None if exact else resamples,
        notes=["exact over all relabellings" if exact else "Monte Carlo relabellings"],
    )


def resampled_difference(
    b: np.ndarray,
    c: np.ndarray,
    statistic: Statistic,
    resamples: int,
    seed: int,
    max_work: int,
    extra_work: int = 0,
) -> tuple[float, np.ndarray]:
    """The observed difference ``statistic(c) - statistic(b)`` and its bootstrap distribution.

    Raises:
        Unassessable: beyond the work budget, or for a statistic that is not finite.
    """
    check_work(resamples * (len(b) + len(c)) + extra_work, max_work, "bootstrap")
    observed = float(statistic(c) - statistic(b))
    distribution = bootstrap_distribution(
        generator(seed), (b, c), lambda xb, xc: statistic(xc) - statistic(xb), resamples
    )
    if not (math.isfinite(observed) and np.isfinite(distribution).all()):
        raise Unassessable("the statistic is not finite in the bootstrap resamples")
    return observed, distribution


def _jackknife(sample: np.ndarray, statistic: Statistic) -> np.ndarray:
    keep = ~np.eye(len(sample), dtype=bool)
    return statistic(np.broadcast_to(sample, keep.shape)[keep].reshape(len(sample), -1))


@assessed("bootstrap")
def bootstrap_difference(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    statistic: Statistic = mean_statistic,
    name: str = "mean",
    alternative: Alternative = "two-sided",
    interval: str = "percentile",
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
    minimum: int = 2,
) -> TestResult:
    """Bootstrap of ``statistic(candidate) - statistic(baseline)`` (two i.i.d. resamples).

    The p-value inverts the percentile interval: the smallest ``1 - level``
    whose interval excludes zero, ``2 * min(#(d* <= 0) + 1, #(d* >= 0) + 1) /
    (B + 1)`` capped at 1 (one tail for a one-sided alternative). The interval
    is the percentile or the BCa interval (``interval='bca'``, whose jackknife
    costs ``n^2`` work). ``statistic`` works along the last axis in data units,
    so sums and quantiles are reported as they are.
    """
    check_confidence(confidence)
    if interval not in ("percentile", "bca"):
        raise ValueError(f"interval must be 'percentile' or 'bca', got {interval!r}")
    b, c = _two(baseline, candidate, minimum)
    extra = len(b) ** 2 + len(c) ** 2 if interval == "bca" else 0
    observed, distribution = resampled_difference(b, c, statistic, resamples, seed, max_work, extra)
    p = interval_p_value(distribution, alternative)
    if interval == "bca":
        jackknife = [
            _jackknife(b, statistic) * -1 + float(statistic(c)),
            float(statistic(b)) * -1 + _jackknife(c, statistic),
        ]
        low, high = bca_interval(observed, distribution, jackknife, confidence)
    else:
        low, high = percentile_interval(distribution, confidence)
    return TestResult(
        method="bootstrap",
        statistic=observed,
        statistic_name=f"{name}_difference",
        p_value=float(p),
        alternative=alternative,
        effect=observed,
        effect_name=f"{name}_difference",
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n=sizes(baseline=b, candidate=c),
        assumptions=["independent observations", f"{interval} bootstrap interval"],
        seed=seed,
        resamples=resamples,
    )


def statistic_for(name: str, probability: float | None = None) -> Statistic:
    """A named statistic along the last axis: mean, median, sum, std or quantile ``p``."""
    functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "mean": mean_statistic,
        "median": lambda a: np.median(a, axis=-1),
        "sum": lambda a: a.sum(axis=-1),
        "std": lambda a: a.std(axis=-1, ddof=1),
    }
    if name == "quantile":
        if probability is None or not 0 <= probability <= 1:
            raise ValueError("a quantile statistic needs a probability in [0, 1]")
        return lambda a: np.quantile(a, probability, axis=-1)
    try:
        return functions[name]
    except KeyError:
        raise ValueError(f"unknown statistic {name!r}") from None
