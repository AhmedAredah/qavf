"""Paired samples: values matched by key, compared through ``candidate - baseline``.

* :func:`paired_t`: mean difference (t-test on the differences);
* :func:`wilcoxon_signed_rank`: whether differences tend to be positive or
  negative (zero differences are dropped, Wilcoxon's convention);
* :func:`sign_test`: the share of positive differences (exact binomial);
* :func:`paired_permutation`: sign-flip permutation of the mean difference;
* :func:`paired_bootstrap`: bootstrap of a statistic of the differences.

No differences at all is an explicit outcome (``p = 1``); differences that are
all one nonzero constant differ with certainty for the t-test (``p = 0``).
"""

from __future__ import annotations

import math
from itertools import islice, product

import numpy as np
import numpy.typing as npt

from qavf.inference._support import (
    Unassessable,
    assessed,
    at_least,
    check_confidence,
    common_unit,
    observations,
    require_scipy,
)
from qavf.inference.independent import mean_statistic
from qavf.inference.intervals import bca_interval, percentile_interval, t_interval
from qavf.inference.pvalues import t_p_value
from qavf.inference.resampling import (
    DEFAULT_MAX_WORK,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Statistic,
    batches,
    bootstrap_distribution,
    check_resamples,
    check_work,
    checked_statistics,
    evaluate,
    generator,
    interval_p_value,
    jackknife,
    permutation_p_value,
)
from qavf.inference.results import Alternative, TestResult


def differences(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> tuple[np.ndarray, float]:
    """``candidate - baseline`` divided by a shared power of two, and that unit.

    Raises:
        Unassessable: for unpaired or non-finite values.
    """
    b, c = observations(baseline, "baseline"), observations(candidate, "candidate")
    if len(b) != len(c):
        raise Unassessable(f"paired samples differ in length ({len(b)} and {len(c)})")
    (sb, sc), unit = common_unit(b, c)
    return sc - sb, unit


@assessed("paired_t")
def paired_t(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TestResult:
    """Paired t-test of the mean difference (at least 2 pairs)."""
    check_confidence(confidence)
    d, unit = differences(baseline, candidate)
    n = {"pairs": len(d)}
    at_least(len(d), 2, "pairs", n=n)
    mean, df = float(d.mean()), len(d) - 1
    se = float(d.std(ddof=1)) / math.sqrt(len(d))
    notes = []
    if se == 0:
        notes.append("every difference is the same")
        t = 0.0 if mean == 0 else math.copysign(math.inf, mean)
        p = 1.0 if mean == 0 else t_p_value(t, df, alternative)
    else:
        result = require_scipy().ttest_rel(d, np.zeros(len(d)), alternative=alternative)
        t, p = float(result.statistic), float(result.pvalue)
    low, high = t_interval(mean, se, df, confidence)
    return TestResult(
        method="paired_t",
        statistic=t,
        statistic_name="t",
        p_value=p,
        alternative=alternative,
        effect=mean * unit,
        effect_name="mean_difference",
        ci_low=low * unit,
        ci_high=high * unit,
        confidence=confidence,
        n=n,
        assumptions=["independent pairs", "approximately normal mean difference"],
        notes=notes,
    )


@assessed("wilcoxon")
def wilcoxon_signed_rank(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, *, alternative: Alternative = "two-sided"
) -> TestResult:
    """Wilcoxon signed-rank test of the differences' symmetry about zero.

    Zero differences are dropped (and counted in the notes). Exact for at most
    50 nonzero differences without ties, otherwise the normal approximation
    with the tie correction, chosen explicitly. The effect is the matched-pairs
    rank-biserial correlation ``(R+ - R-) / (R+ + R-)``. No nonzero difference
    gives ``p = 1``.
    """
    d, _ = differences(baseline, candidate)
    at_least(len(d), 1, "pairs", n={"pairs": len(d)})
    nonzero = d[d != 0]
    n = {"pairs": len(d), "nonzero": len(nonzero)}
    notes = [f"{len(d) - len(nonzero)} zero difference(s) dropped"] if len(nonzero) < len(d) else []
    common = {
        "method": "wilcoxon",
        "statistic_name": "W+",
        "alternative": alternative,
        "effect_name": "rank_biserial",
        "n": n,
        "assumptions": ["independent pairs", "differences symmetric about their median"],
    }
    if not len(nonzero):
        return TestResult(
            statistic=0.0, p_value=1.0, effect=0.0, notes=[*notes, "no change"], **common
        )
    stats = require_scipy()
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    ties = len(np.unique(np.abs(nonzero))) < len(nonzero)
    method = "exact" if len(nonzero) <= 50 and not ties else "approx"
    result = stats.wilcoxon(nonzero, alternative=alternative, method=method)
    p = float(result.pvalue)
    return TestResult(
        statistic=positive,
        p_value=p,
        effect=(positive - negative) / (positive + negative),
        notes=[*notes, f"{method} p-value"],
        **common,
    )


@assessed("sign")
def sign_test(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, *, alternative: Alternative = "two-sided"
) -> TestResult:
    """Exact sign test: are positive and negative differences equally likely?

    Zero differences are dropped. The effect is the share of positive
    differences among the nonzero ones; none gives ``p = 1``.
    """
    d, _ = differences(baseline, candidate)
    at_least(len(d), 1, "pairs", n={"pairs": len(d)})
    positive, nonzero = int((d > 0).sum()), int((d != 0).sum())
    n = {"pairs": len(d), "nonzero": nonzero}
    if not nonzero:
        return TestResult(
            method="sign", statistic=0.0, p_value=1.0, effect=None, effect_name="share_positive",
            n=n, notes=["no change"],
        )  # fmt: skip
    result = require_scipy().binomtest(positive, nonzero, 0.5, alternative=alternative)
    return TestResult(
        method="sign",
        statistic=float(positive),
        statistic_name="positive",
        p_value=float(result.pvalue),
        alternative=alternative,
        effect=positive / nonzero,
        effect_name="share_positive",
        n=n,
        assumptions=["independent pairs"],
    )


@assessed("paired_permutation")
def paired_permutation(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    alternative: Alternative = "two-sided",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
) -> TestResult:
    """Sign-flip permutation test of the mean difference (at least 2 pairs).

    Exact over all ``2^n`` sign assignments when there are no more of them than
    ``resamples``; otherwise seeded random flips with ``(k + 1) / (B + 1)``.
    """
    d, unit = differences(baseline, candidate)
    at_least(len(d), 2, "pairs", n={"pairs": len(d)})
    check_resamples(resamples)
    observed = float(d.mean())
    exact = len(d) < 63 and 2 ** len(d) <= resamples
    check_work((2 ** len(d) if exact else resamples) * len(d), max_work, "paired permutation")
    if exact:
        assignments = product((1.0, -1.0), repeat=len(d))
        distribution = np.empty(2 ** len(d))
        done = 0
        for step in batches(len(distribution), len(d)):
            signs = np.array(list(islice(assignments, step)))
            distribution[done : done + step] = mean_statistic(signs * d)
            done += step
        p = permutation_p_value(distribution, observed, alternative, exact=True)
    else:
        rng = generator(seed)
        distribution = np.empty(resamples)
        done = 0
        for step in batches(resamples, len(d)):
            signs = 1.0 - 2.0 * rng.integers(0, 2, size=(step, len(d)))
            distribution[done : done + step] = mean_statistic(signs * d)
            done += step
        p = permutation_p_value(distribution, observed, alternative)
    return TestResult(
        method="paired_permutation",
        statistic=observed * unit,
        statistic_name="mean_difference",
        p_value=float(p),
        alternative=alternative,
        effect=observed * unit,
        effect_name="mean_difference",
        n={"pairs": len(d)},
        assumptions=["independent pairs", "differences symmetric about zero under no change"],
        seed=None if exact else seed,
        resamples=None if exact else resamples,
        notes=["exact over all sign assignments" if exact else "Monte Carlo sign flips"],
    )


@assessed("paired_bootstrap")
def paired_bootstrap(
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
) -> TestResult:
    """Bootstrap of ``statistic`` of the differences (the mean by default).

    Pairs are resampled together; the p-value inverts the percentile interval
    as in :func:`~qavf.inference.independent.bootstrap_difference`.
    """
    check_confidence(confidence)
    check_resamples(resamples)
    if interval not in ("percentile", "bca"):
        raise ValueError(f"interval must be 'percentile' or 'bca', got {interval!r}")
    if statistic is mean_statistic:
        d, unit = differences(baseline, candidate)
    else:
        b, c = observations(baseline, "baseline"), observations(candidate, "candidate")
        if len(b) != len(c):
            raise Unassessable("paired samples differ in length")
        with np.errstate(over="ignore", invalid="ignore"):
            d = checked_statistics(c - b, (len(b),))
        unit = 1.0
    at_least(len(d), 2, "pairs", n={"pairs": len(d)})
    check_work(
        resamples * len(d) + (len(d) ** 2 if interval == "bca" else 0), max_work, "bootstrap"
    )
    observed = float(evaluate(statistic, d))
    distribution = bootstrap_distribution(
        generator(seed), (d,), lambda values: evaluate(statistic, values), resamples
    )
    p = interval_p_value(distribution, alternative)
    if interval == "bca":
        omitted = jackknife(d, statistic)
        low, high = bca_interval(observed, distribution, [omitted], confidence)
    else:
        low, high = percentile_interval(distribution, confidence)
    return TestResult(
        method="paired_bootstrap",
        statistic=observed * unit,
        statistic_name=f"{name}_difference",
        p_value=float(p),
        alternative=alternative,
        effect=observed * unit,
        effect_name=f"{name}_difference",
        ci_low=low * unit,
        ci_high=high * unit,
        confidence=confidence,
        n={"pairs": len(d)},
        assumptions=["independent pairs", f"{interval} bootstrap interval"],
        seed=seed,
        resamples=resamples,
    )
