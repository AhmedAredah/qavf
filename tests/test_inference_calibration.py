"""Calibration by simulation (fixed seeds): under the null, tests reject at about alpha.

A rate must lie within 3.5 binomial standard deviations of alpha (exact and
discrete tests, which are conservative, only above). On AR(1) series the
dependence-aware methods hold their level while their i.i.d. counterparts
visibly over-reject. Benjamini-Hochberg controls the false discovery rate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qavf.inference import (
    adjust,
    anderson_darling,
    bootstrap_difference,
    brown_forsythe,
    correlation_test,
    ks_two_sample,
    lag_test,
    mann_kendall,
    mann_whitney,
    paired_t,
    permutation,
    poisson_rate,
    sign_test,
    slope_difference,
    two_proportion,
    welch_t,
    wilcoxon_signed_rank,
)

pytest.importorskip("scipy")

ALPHA = 0.05


def _tolerance(reps: int) -> float:
    return 3.5 * math.sqrt(ALPHA * (1 - ALPHA) / reps)


def _rate(test, reps: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    return sum(test(rng, rep).p_value < ALPHA for rep in range(reps)) / reps


def _calibrated(rate: float, reps: int, conservative: bool = False) -> None:
    tolerance = _tolerance(reps)
    assert rate <= ALPHA + tolerance, f"rejection rate {rate:.3f} above {ALPHA}"
    if not conservative:
        assert rate >= ALPHA - tolerance, f"rejection rate {rate:.3f} below {ALPHA}"


NULL_TESTS = {
    "welch_t": (lambda r, i: welch_t(r.normal(5, 1, 20), r.normal(5, 3, 35)), False),
    "mann_whitney": (lambda r, i: mann_whitney(r.exponential(1, 25), r.exponential(1, 30)), False),
    "ks": (lambda r, i: ks_two_sample(r.normal(size=40), r.normal(size=50)), True),
    "anderson_darling": (
        lambda r, i: anderson_darling(r.normal(size=40), r.normal(size=50)),
        False,
    ),
    "brown_forsythe": (lambda r, i: brown_forsythe(r.normal(size=30), r.normal(size=30)), False),
    "paired_t": (lambda r, i: paired_t(*_pairs(r, 25)), False),
    "wilcoxon": (lambda r, i: wilcoxon_signed_rank(*_pairs(r, 25)), False),
    "sign": (lambda r, i: sign_test(*_pairs(r, 25)), True),
    "poisson": (lambda r, i: poisson_rate((r.poisson(40), r.poisson(60)), (2.0, 3.0)), True),
    "two_proportion": (
        lambda r, i: two_proportion((r.binomial(200, 0.3), r.binomial(150, 0.3)), (200, 150)),
        False,
    ),
    "permutation": (
        lambda r, i: permutation(r.normal(size=20), r.normal(size=25), resamples=499, seed=i),
        False,
    ),
    "bootstrap": (
        lambda r, i: bootstrap_difference(
            r.normal(size=40), r.normal(size=40), resamples=499, seed=i
        ),
        False,
    ),
    "mann_kendall": (lambda r, i: mann_kendall(r.normal(size=30)), False),
    "fisher_z": (lambda r, i: correlation_test(r.normal(size=30), r.normal(size=30)), False),
}


def _pairs(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    base = rng.normal(10, 3, n)
    return base, base + rng.normal(0, 1, n)


@pytest.mark.parametrize("name", sorted(NULL_TESTS))
def test_null_rejection_rate_is_alpha(name):
    test, conservative = NULL_TESTS[name]
    reps = 400
    _calibrated(_rate(test, reps, seed=sum(map(ord, name))), reps, conservative)


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    shocks = rng.normal(size=n + 50)
    values = np.zeros(n + 50)
    for t in range(1, n + 50):
        values[t] = phi * values[t - 1] + shocks[t]
    return values[50:]


def test_block_bootstrap_holds_its_level_on_ar1_where_iid_over_rejects():
    reps, n = 300, 120
    positions = np.arange(n, dtype=float)
    rng = np.random.default_rng(11)
    rejected = {"block": 0, "none": 0}
    for rep in range(reps):
        difference = _ar1(rng, n, 0.5)
        for dependence in rejected:
            result = slope_difference(
                positions, np.zeros(n), difference, dependence=dependence, resamples=999, seed=rep
            )
            rejected[dependence] += result.p_value < ALPHA
    _calibrated(rejected["block"] / reps, reps)
    assert rejected["none"] / reps > ALPHA + 2 * _tolerance(reps)  # i.i.d. errors: over-rejects


def test_effective_sample_size_holds_the_correlation_level_on_ar1():
    reps, n = 400, 60
    rng = np.random.default_rng(12)
    rejected = {"effective_n": 0, "none": 0}
    for _ in range(reps):
        x, y = _ar1(rng, n, 0.7), _ar1(rng, n, 0.7)
        for dependence in rejected:
            rejected[dependence] += correlation_test(x, y, dependence=dependence).p_value < ALPHA
    _calibrated(rejected["effective_n"] / reps, reps)
    assert rejected["none"] / reps > ALPHA + 2 * _tolerance(reps)


def test_lag_test_holds_its_level_without_a_lag():
    reps, n = 300, 80
    rng = np.random.default_rng(13)
    rejected = 0
    for rep in range(reps):
        x, y = _ar1(rng, n, 0.5), _ar1(rng, n, 0.5)
        result = lag_test(dict(enumerate(x)), dict(enumerate(y)), 1, resamples=499, seed=rep)
        rejected += result.p_value < ALPHA
    _calibrated(rejected / reps, reps, conservative=True)


def test_benjamini_hochberg_controls_the_false_discovery_rate():
    rng = np.random.default_rng(14)
    reps, m, real = 500, 20, 5
    proportions = []
    for _ in range(reps):
        shifts = np.r_[np.full(real, 0.8), np.zeros(m - real)]
        p = [welch_t(rng.normal(0, 1, 30), rng.normal(s, 1, 30)).p_value for s in shifts]
        rejected = adjust(p, "bh") < ALPHA
        false = int(rejected[real:].sum())
        proportions.append(false / max(int(rejected.sum()), 1))
    fdr = float(np.mean(proportions))
    assert fdr <= ALPHA + 3.5 * float(np.std(proportions)) / math.sqrt(reps), fdr
