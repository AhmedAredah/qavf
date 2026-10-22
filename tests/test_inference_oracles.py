"""Every inference test against an independent reference: SciPy, statsmodels, or a
textbook example worked by hand (never values produced by the code under test)."""

from __future__ import annotations

import math
import warnings
from itertools import combinations

import numpy as np
import pytest

from qavf.inference import (
    Moments,
    anderson_darling,
    bootstrap_difference,
    brown_forsythe,
    chi_square,
    compare_correlations,
    correlation_test,
    fisher_exact,
    g_test,
    kendall_tau,
    ks_two_sample,
    mann_kendall,
    mann_whitney,
    mcnemar,
    morans_i,
    paired_bootstrap,
    paired_permutation,
    paired_t,
    permutation,
    poisson_rate,
    seasonal_kendall,
    sign_test,
    slope_difference,
    stuart_maxwell,
    tost_from_moments,
    tost_independent,
    tost_paired,
    two_proportion,
    welch_t,
    welch_t_from_moments,
    wilcoxon_signed_rank,
)
from qavf.inference.intervals import bca_interval, percentile_interval
from qavf.inference.resampling import newey_west_slope

stats = pytest.importorskip("scipy.stats")
sm = pytest.importorskip("statsmodels.api")

RNG = np.random.default_rng(20251021)
B = RNG.normal(10.0, 2.0, 25)
C = RNG.normal(11.5, 3.0, 31)
PAIRED = B + RNG.normal(0.6, 1.0, 25)


# ------------------------------------------------------------ independent
def test_welch_t_matches_scipy_and_statsmodels():
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW, ttest_ind

    result = welch_t(B, C)
    reference = stats.ttest_ind(C, B, equal_var=False)
    assert result.statistic == pytest.approx(reference.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-10)
    _, p, _ = ttest_ind(C, B, usevar="unequal")
    assert result.p_value == pytest.approx(p, rel=1e-10)
    low, high = CompareMeans(DescrStatsW(C), DescrStatsW(B)).tconfint_diff(usevar="unequal")
    assert (result.ci_low, result.ci_high) == pytest.approx((low, high), rel=1e-10)
    assert result.effect == pytest.approx(C.mean() - B.mean(), rel=1e-12)
    for alternative in ("greater", "less"):
        one = welch_t(B, C, alternative=alternative)
        expected = stats.ttest_ind(C, B, equal_var=False, alternative=alternative).pvalue
        assert one.p_value == pytest.approx(expected, rel=1e-10)


def test_welch_t_from_moments_equals_the_array_test():
    moments = Moments(len(B), B.mean(), B.var(ddof=1)), Moments(len(C), C.mean(), C.var(ddof=1))
    assert welch_t_from_moments(*moments).p_value == pytest.approx(welch_t(B, C).p_value, rel=1e-9)


def test_welch_t_is_invariant_to_a_large_offset():
    offset = 1e12
    shifted = welch_t(B + offset, C + offset)
    assert shifted.p_value == pytest.approx(welch_t(B, C).p_value, rel=1e-3)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_mann_whitney_matches_scipy(alternative):
    result = mann_whitney(B, C, alternative=alternative)
    reference = stats.mannwhitneyu(C, B, alternative=alternative, method="asymptotic")
    assert result.statistic == reference.statistic
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-12)
    small = mann_whitney([1.0, 4.0, 2.5], [3.0, 5.0, 6.0, 7.0], alternative=alternative)
    exact = stats.mannwhitneyu(
        [3.0, 5.0, 6.0, 7.0], [1.0, 4.0, 2.5], alternative=alternative, method="exact"
    )
    assert small.p_value == pytest.approx(exact.pvalue, rel=1e-12)


def test_mann_whitney_rank_biserial_by_hand():
    # 3 baseline, 2 candidate: pairs (c > b) = 5 of 6, (c < b) = 1 -> r = 4/6.
    assert mann_whitney([1, 2, 5], [3, 6]).effect == pytest.approx(4 / 6)


def test_distribution_tests_match_scipy():
    ks = ks_two_sample(B, C)
    reference = stats.ks_2samp(B, C)
    assert (ks.statistic, ks.p_value) == pytest.approx((reference.statistic, reference.pvalue))
    ad = anderson_darling(B, C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = stats.anderson_ksamp([B, C])
    assert ad.statistic == pytest.approx(expected.statistic)
    assert ad.p_value == pytest.approx(expected.pvalue)
    bf = brown_forsythe(B, C)
    levene = stats.levene(B, C, center="median")
    assert bf.statistic == pytest.approx(levene.statistic, rel=1e-10)
    assert bf.p_value == pytest.approx(levene.pvalue, rel=1e-10)
    assert bf.effect == pytest.approx(C.std(ddof=1) / B.std(ddof=1))


def test_exact_permutation_matches_scipy_exact():
    b, c = [1.2, 3.4, 2.2, 5.1, 0.7], [4.4, 6.1, 3.9, 5.5]
    result = permutation(b, c, resamples=10_000)
    reference = stats.permutation_test(
        (c, b),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        permutation_type="independent",
        n_resamples=np.inf,
        vectorized=True,
    )
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-12)
    assert "exact" in result.notes[0] and result.seed is None


def test_monte_carlo_permutation_is_close_to_the_exact_p():
    b, c = B[:12], C[:12]
    exact = stats.permutation_test(
        (c, b),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        permutation_type="independent",
        n_resamples=np.inf,
        vectorized=True,
    ).pvalue
    result = permutation(b, c, resamples=20_000)
    assert result.p_value == pytest.approx(exact, abs=4 * math.sqrt(exact * (1 - exact) / 20_000))


def test_bootstrap_intervals_against_scipy():
    result = bootstrap_difference(B, C, resamples=20_000, seed=3)
    reference = stats.bootstrap(
        (C, B),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        method="percentile",
        n_resamples=20_000,
        random_state=np.random.default_rng(4),
        vectorized=True,
    ).confidence_interval
    spread = (reference.high - reference.low) * 0.03
    assert result.ci_low == pytest.approx(reference.low, abs=spread)
    assert result.ci_high == pytest.approx(reference.high, abs=spread)
    bca = bootstrap_difference(B, C, interval="bca", resamples=20_000, seed=3)
    reference = stats.bootstrap(
        (C, B),
        lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis),
        method="BCa",
        n_resamples=20_000,
        random_state=np.random.default_rng(4),
        vectorized=True,
    ).confidence_interval
    assert bca.ci_low == pytest.approx(reference.low, abs=spread)
    assert bca.ci_high == pytest.approx(reference.high, abs=spread)


def test_bca_interval_textbook_formula():
    # Efron & Tibshirani (1993) section 14.3, written out independently: z0 from
    # the share below the estimate, acceleration from the jackknife skewness.
    rng = np.random.default_rng(1)
    distribution = rng.normal(1.0, 0.5, 4000) ** 2
    jackknife = rng.normal(1.0, 0.1, 30) ** 3
    estimate = 1.1
    z0 = stats.norm.ppf((np.sum(distribution < estimate) + np.sum(distribution <= estimate)) / 8000)
    u = jackknife.mean() - jackknife
    a = (u**3).sum() / (6 * ((u**2).sum()) ** 1.5)
    expected = []
    for z in stats.norm.ppf([0.05, 0.95]):
        expected.append(
            np.quantile(distribution, stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
        )
    assert bca_interval(estimate, distribution, [jackknife], 0.90) == pytest.approx(expected)
    assert percentile_interval(distribution, 0.9) == pytest.approx(
        tuple(np.quantile(distribution, [0.05, 0.95]))
    )


# ----------------------------------------------------------------- paired
def test_paired_tests_match_references():
    from statsmodels.stats.descriptivestats import sign_test as sm_sign

    t = paired_t(B, PAIRED)
    reference = stats.ttest_rel(PAIRED, B)
    assert (t.statistic, t.p_value) == pytest.approx((reference.statistic, reference.pvalue))
    interval = reference.confidence_interval()
    assert (t.ci_low, t.ci_high) == pytest.approx((interval.low, interval.high))
    w = wilcoxon_signed_rank(B, PAIRED)
    assert w.p_value == pytest.approx(stats.wilcoxon(PAIRED - B, method="exact").pvalue)
    s = sign_test(B, PAIRED)
    _, p = sm_sign(PAIRED - B, 0)
    assert s.p_value == pytest.approx(p)


def test_wilcoxon_with_ties_and_zeros_uses_the_normal_approximation():
    before = np.array([5, 7, 3, 9, 4, 6, 8, 2, 5, 7], dtype=float)
    after = before + np.array([1, 2, 0, -1, 2, 1, 3, 0, 2, 1], dtype=float)
    result = wilcoxon_signed_rank(before, after)
    nonzero = (after - before)[after != before]
    reference = stats.wilcoxon(nonzero, method="approx")
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.n == {"pairs": 10, "nonzero": 8} and "2 zero" in result.notes[0]


def test_paired_permutation_exact_matches_scipy():
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    c = b + np.array([0.5, -0.2, 0.9, 0.4, 1.1, -0.1, 0.7])
    result = paired_permutation(b, c, resamples=1000)
    reference = stats.permutation_test(
        (c - b,), lambda d, axis: np.mean(d, axis=axis), permutation_type="samples",
        n_resamples=np.inf, vectorized=True,
    )  # fmt: skip
    assert result.p_value == pytest.approx(reference.pvalue)


def test_paired_bootstrap_interval_is_near_the_t_interval():
    result = paired_bootstrap(B, PAIRED, resamples=20_000)
    t = paired_t(B, PAIRED)
    width = t.ci_high - t.ci_low
    assert result.ci_low == pytest.approx(t.ci_low, abs=0.1 * width)
    assert result.ci_high == pytest.approx(t.ci_high, abs=0.1 * width)


# ------------------------------------------------------------ equivalence
def test_tost_matches_statsmodels():
    from statsmodels.stats.weightstats import ttost_ind, ttost_paired

    p, _, _ = ttost_paired(PAIRED, B, -1.0, 1.0)
    assert tost_paired(B, PAIRED, 1.0).p_value == pytest.approx(p)
    p, _, _ = ttost_ind(C, B, -2.0, 3.0, usevar="unequal")
    assert tost_independent(B, C, (-2.0, 3.0)).p_value == pytest.approx(p)
    moments = Moments(len(B), B.mean(), B.var(ddof=1)), Moments(len(C), C.mean(), C.var(ddof=1))
    assert tost_from_moments(*moments, (-2.0, 3.0)).p_value == pytest.approx(p)


# ------------------------------------------------------------ categorical
def test_contingency_tests_match_scipy():
    table = np.array([[12, 30, 18], [25, 20, 15]])
    chi = chi_square(table)
    reference = stats.chi2_contingency(table, correction=False)
    assert (chi.statistic, chi.p_value) == pytest.approx((reference.statistic, reference.pvalue))
    n = table.sum()
    assert chi.effect == pytest.approx(math.sqrt(reference.statistic / (n * 1)))
    g = g_test(table)
    reference = stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")
    assert (g.statistic, g.p_value) == pytest.approx((reference.statistic, reference.pvalue))


def test_fisher_exact_tea_tasting():
    # Fisher's lady tasting tea: 3 of 4 cups right; one-sided p = 17/70.
    result = fisher_exact([[1, 3], [3, 1]], alternative="greater")
    assert result.p_value == pytest.approx(17 / 70)
    assert result.effect == pytest.approx(9.0)
    two = fisher_exact([[1, 3], [3, 1]])
    assert two.p_value == pytest.approx(stats.fisher_exact([[1, 3], [3, 1]]).pvalue)


@pytest.mark.parametrize("exact", [True, False])
def test_mcnemar_matches_statsmodels(exact):
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    table = np.array([[59, 6], [16, 80]])
    reference = sm_mcnemar(table, exact=exact, correction=True)
    result = mcnemar(table, exact=exact)
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.statistic == pytest.approx(reference.statistic)


def test_stuart_maxwell_matches_statsmodels():
    from statsmodels.stats.contingency_tables import SquareTable

    table = np.array([[20, 5, 3, 1], [4, 30, 6, 2], [2, 8, 22, 4], [1, 2, 6, 15]])
    reference = SquareTable(table, shift_zeros=False).homogeneity(method="stuart_maxwell")
    result = stuart_maxwell(table)
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)


# ------------------------------------------------------------------ counts
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_two_proportion_matches_statsmodels(alternative):
    from statsmodels.stats.proportion import proportions_ztest

    sm_alternative = {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}
    z, p = proportions_ztest([58, 40], [200, 180], alternative=sm_alternative[alternative])
    result = two_proportion((40, 58), (180, 200), alternative=alternative)
    assert (result.statistic, result.p_value) == pytest.approx((z, p))


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_poisson_rate_matches_statsmodels_exact_conditional(alternative):
    from statsmodels.stats.rates import test_poisson_2indep

    reference = test_poisson_2indep(31, 2.5, 18, 2.0, method="exact-cond", alternative=alternative)
    ours = {"two-sided": "two-sided", "larger": "greater", "smaller": "less"}[alternative]
    result = poisson_rate((18, 31), (2.0, 2.5), alternative=ours)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-9)
    assert result.effect == pytest.approx((31 / 2.5) / (18 / 2.0))


def test_poisson_rate_interval_maps_clopper_pearson():
    result = poisson_rate((10, 20), (1.0, 1.0))
    ci = stats.binomtest(20, 30, 0.5).proportion_ci(method="exact")
    assert result.ci_low == pytest.approx(ci.low / (1 - ci.low))
    assert result.ci_high == pytest.approx(ci.high / (1 - ci.high))


# ---------------------------------------------------------- trend, series
def test_kendall_tau_is_scipy():
    x, y = np.arange(9), np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0])
    result = kendall_tau(x, y)
    reference = stats.kendalltau(x, y)
    assert (result.statistic, result.p_value) == (reference.statistic, reference.pvalue)


def _mann_kendall_by_hand(values):
    n = len(values)
    s = sum(np.sign(values[j] - values[i]) for i in range(n) for j in range(i + 1, n))
    groups = [list(values).count(v) for v in set(values)]
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in groups)) / 18
    z = (s - np.sign(s)) / math.sqrt(var) if s else 0.0
    return s, 2 * (1 - stats.norm.cdf(abs(z)))


def test_mann_kendall_and_sen_slope_by_hand():
    values = np.array([2.1, 2.4, 2.4, 3.0, 2.8, 3.6, 3.3, 4.0, 4.1, 3.9, 4.6])
    result = mann_kendall(values)
    s, p = _mann_kendall_by_hand(values)
    assert result.statistic == s and result.p_value == pytest.approx(p)
    slopes = [(values[j] - values[i]) / (j - i) for i, j in combinations(range(len(values)), 2)]
    assert result.effect == pytest.approx(np.median(slopes))


def test_seasonal_kendall_by_hand():
    rng = np.random.default_rng(5)
    seasons = [i % 4 for i in range(24)]
    values = np.arange(24) * 0.1 + np.array([0, 3, 1, 2] * 6) + rng.normal(0, 0.3, 24)
    result = seasonal_kendall(values, seasons)
    s = var = 0.0
    slopes = []
    for season in range(4):
        idx = [i for i in range(24) if seasons[i] == season]
        part, _ = _mann_kendall_by_hand(values[idx])
        n = len(idx)
        s += part
        var += n * (n - 1) * (2 * n + 5) / 18
        slopes += [(values[j] - values[i]) / (j - i) for i, j in combinations(idx, 2)]
    z = (s - np.sign(s)) / math.sqrt(var)
    assert result.statistic == s
    assert result.p_value == pytest.approx(2 * (1 - stats.norm.cdf(abs(z))))
    assert result.effect == pytest.approx(np.median(slopes))


def test_slope_difference_matches_statsmodels_ols_and_hac():
    x = np.arange(30, dtype=float)
    rng = np.random.default_rng(7)
    b = 5 + 0.2 * x + rng.normal(0, 1, 30)
    c = 4 + 0.35 * x + rng.normal(0, 1, 30)
    design = sm.add_constant(x)
    ols = sm.OLS(c - b, design).fit()
    plain = slope_difference(x, b, c, dependence="none")
    assert plain.effect == pytest.approx(ols.params[1])
    assert plain.p_value == pytest.approx(ols.pvalues[1])
    hac = sm.OLS(c - b, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True}, use_t=True
    )
    robust = slope_difference(x, b, c, dependence="hac", lags=3)
    assert robust.statistic == pytest.approx(hac.tvalues[1])
    assert robust.p_value == pytest.approx(hac.pvalues[1])
    assert (robust.ci_low, robust.ci_high) == pytest.approx(tuple(hac.conf_int()[1]))
    assert newey_west_slope(x, c - b, 3)[1] == pytest.approx(hac.bse[1])


def test_fisher_z_textbook_examples():
    # Two independent correlations, r = 0.5 (n = 50) and 0.3 (n = 50), worked by hand:
    # z = (0.549306 - 0.309520) / sqrt(2/47) = 0.239786 / 0.206284 = 1.16241,
    # two-sided p = 2 * (1 - Phi(1.16241)) = 0.24507.
    result = compare_correlations((0.3, 50), (0.5, 50))
    assert result.statistic == pytest.approx(1.16241, abs=1e-5)
    assert result.p_value == pytest.approx(0.24507, abs=1e-5)
    x = RNG.normal(size=40)
    y = x + RNG.normal(size=40)
    test = correlation_test(x, y)
    reference = stats.pearsonr(x, y)
    assert test.effect == pytest.approx(reference.statistic)
    interval = reference.confidence_interval(0.95)
    assert (test.ci_low, test.ci_high) == pytest.approx((interval.low, interval.high))
    expected = math.atanh(reference.statistic) * math.sqrt(37)
    assert test.statistic == pytest.approx(expected)


# ----------------------------------------------------------------- spatial
def test_morans_i_by_hand_on_a_grid():
    # A 3 x 3 rook grid with a gradient: I by the definition, written out.
    values = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0])
    weights = np.zeros((9, 9))
    for i in range(9):
        r, c = divmod(i, 3)
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if 0 <= r + dr < 3 and 0 <= c + dc < 3:
                weights[i, (r + dr) * 3 + c + dc] = 1
    z = values - values.mean()
    expected = (
        9
        / weights.sum()
        * sum(weights[i, j] * z[i] * z[j] for i in range(9) for j in range(9))
        / (z @ z)
    )
    result = morans_i(values, weights, permutations=4999, seed=1, alternative="greater")
    assert result.statistic == pytest.approx(expected)
    assert result.p_value < 0.01
    adjacency = {i: [j for j in range(9) if weights[i, j]] for i in range(9)}
    assert morans_i(values, adjacency).statistic == pytest.approx(expected)
    from scipy import sparse

    assert morans_i(values, sparse.csr_matrix(weights)).statistic == pytest.approx(expected)


def test_morans_i_detects_alternating_values_as_negative_autocorrelation():
    values = np.array([1.0, -1.0] * 20) + RNG.normal(0, 0.1, 40)
    ring = {i: [(i + 1) % 40, (i - 1) % 40] for i in range(40)}
    result = morans_i(values, ring, alternative="less", permutations=999, seed=2)
    z = values - values.mean()
    expected = 40 / 80 * sum(z[i] * z[j] for i in ring for j in ring[i]) / (z @ z)
    assert result.statistic == pytest.approx(expected) and expected < -0.8
    assert result.p_value == pytest.approx(1 / 1000)
    note = next(n for n in result.notes if n.startswith("expected"))
    assert float(note.split()[2]) == pytest.approx(-1 / 39, rel=1e-5)
