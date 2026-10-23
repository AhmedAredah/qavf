"""Counts and rates: shares of trials, and event counts per exposure.

* :func:`two_proportion`: do two shares (``successes / trials``) differ? The
  pooled two-proportion z-test;
* :func:`poisson_rate`: do two event rates (``count / exposure``, e.g.
  crashes per vehicle-mile) differ? The exact conditional test: given the
  total, the candidate's count is binomial with probability
  ``exposure_c / (exposure_b + exposure_c)`` when the rates are equal.

Effects are the difference of shares and the rate ratio (candidate over
baseline). No events (or shares of exactly 0 or 1 on both sides) means no
difference: ``p = 1``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from qavf.inference._support import Unassessable, assessed, check_confidence, require_scipy
from qavf.inference.effects import rate_ratio
from qavf.inference.intervals import normal_interval
from qavf.inference.pvalues import normal_p_value
from qavf.inference.results import Alternative, TestResult


def _whole(value: float, role: str) -> int:
    if value < 0 or not math.isfinite(value) or value != int(value):
        raise Unassessable(f"{role} must be a whole non-negative count, got {value!r}")
    return int(value)


@assessed("two_proportion")
def two_proportion(
    successes: tuple[float, float],
    trials: tuple[float, float],
    *,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TestResult:
    """Pooled two-proportion z-test; ``(baseline, candidate)`` pairs of counts.

    The interval of the difference of shares is the unpooled Wald interval.
    """
    check_confidence(confidence)
    xb, xc = (_whole(v, "successes") for v in successes)
    nb, nc = (_whole(v, "trials") for v in trials)
    n = {"baseline": nb, "candidate": nc}
    if not nb or not nc:
        raise Unassessable("a side has no trials", n=n)
    if xb > nb or xc > nc:
        raise Unassessable("successes exceed trials", n=n)
    pb, pc = xb / nb, xc / nc
    qb, qc = (nb - xb) / nb, (nc - xc) / nc
    difference = float(Fraction(xc, nc) - Fraction(xb, nb))
    pooled = (xb + xc) / (nb + nc)
    complement = (nb - xb + nc - xc) / (nb + nc)
    common = {
        "method": "two_proportion",
        "statistic_name": "z",
        "alternative": alternative,
        "effect": difference,
        "effect_name": "share_difference",
        "n": n,
        "assumptions": [
            "independent trials",
            "normal approximation (enough successes and failures)",
        ],
    }
    se_unpooled = math.hypot(
        math.sqrt(pb) * math.sqrt(qb) / math.sqrt(nb),
        math.sqrt(pc) * math.sqrt(qc) / math.sqrt(nc),
    )
    low, high = normal_interval(difference, se_unpooled, confidence)
    if xb + xc in (0, nb + nc):
        return TestResult(
            statistic=0.0, p_value=1.0, ci_low=low, ci_high=high, confidence=confidence,
            notes=["every trial has the same outcome"], **common,
        )  # fmt: skip
    # Preserve the rare outcome even when its complementary share rounds to 1.
    se = math.sqrt(pooled) * math.sqrt(complement) * math.hypot(
        1 / math.sqrt(nb), 1 / math.sqrt(nc)
    )
    if se == 0:
        raise Unassessable("the proportion standard error is below the supported float range", n=n)
    z = difference / se
    p = normal_p_value(z, alternative)
    return TestResult(
        statistic=z, p_value=p, ci_low=low, ci_high=high, confidence=confidence, **common
    )


@assessed("poisson")
def poisson_rate(
    counts: tuple[float, float],
    exposure: tuple[float, float] = (1.0, 1.0),
    *,
    alternative: Alternative = "two-sided",
    confidence: float = 0.95,
) -> TestResult:
    """Exact conditional test of two Poisson rates; ``(baseline, candidate)`` pairs.

    The candidate's count given the total is tested against the binomial
    probability its exposure share implies (SciPy's exact binomial test, whose
    two-sided p-value sums outcomes no more likely than the one observed). The
    interval of the rate ratio maps the Clopper-Pearson interval of that
    proportion.
    """
    check_confidence(confidence)
    kb, kc = (_whole(v, "counts") for v in counts)
    eb, ec = (float(v) for v in exposure)
    n = {"baseline": kb, "candidate": kc}
    if not (eb > 0 and ec > 0 and math.isfinite(eb) and math.isfinite(ec)):
        raise Unassessable("exposures must be positive and finite", n=n)
    exposure_unit = max(eb, ec)
    eb, ec = eb / exposure_unit, ec / exposure_unit
    if not eb or not ec:
        raise Unassessable("exposure ratio cannot be represented without underflow", n=n)
    common = {
        "method": "poisson",
        "statistic_name": "candidate_count",
        "alternative": alternative,
        "effect": rate_ratio(kb, eb, kc, ec),
        "effect_name": "rate_ratio",
        "n": n,
        "assumptions": ["Poisson counts (independent events at a constant rate)"],
    }
    total = kb + kc
    if not total:
        return TestResult(statistic=0.0, p_value=1.0, notes=["no events"], **common)
    stats = require_scipy()
    share = ec / (eb + ec)
    result = stats.binomtest(kc, total, share, alternative=alternative)
    interval = result.proportion_ci(confidence_level=confidence, method="exact")

    def ratio(p: float) -> float:
        if p >= 1:
            return math.inf
        value = rate_ratio(1 - p, eb, p, ec)
        return math.inf if value is None else value

    return TestResult(
        statistic=float(kc),
        p_value=float(result.pvalue),
        ci_low=ratio(float(interval.low)),
        ci_high=ratio(float(interval.high)),
        confidence=confidence,
        **common,
    )
