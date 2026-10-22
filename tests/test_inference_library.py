"""Inference library behaviour: corrections, effects, decisions, edge cases and plugins."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qavf.inference import (
    TESTS,
    InferenceTest,
    Moments,
    Significance,
    TestInput,
    TestResult,
    adjust,
    anderson_darling,
    bootstrap_difference,
    brown_forsythe,
    chi_square,
    correct,
    correlation_test,
    decide,
    effects,
    lag_test,
    mann_kendall,
    mann_whitney,
    mcnemar,
    morans_i,
    paired_permutation,
    paired_t,
    permutation,
    poisson_rate,
    register_test,
    sign_test,
    slope_difference,
    stuart_maxwell,
    tost_paired,
    two_proportion,
    welch_t,
    welch_t_from_moments,
    wilcoxon_signed_rank,
)
from qavf.inference.multiple import CORRECTIONS
from qavf.plugins.registry import entry_point_scope

pytest.importorskip("scipy")


# ---------------------------------------------------------------- multiple
@pytest.mark.parametrize(
    "method,reference", [("bonferroni", "bonferroni"), ("holm", "holm"), ("bh", "fdr_bh"),
                         ("by", "fdr_by")],
)  # fmt: skip
def test_corrections_match_statsmodels(method, reference):
    from statsmodels.stats.multitest import multipletests

    p = np.random.default_rng(1).uniform(0, 0.2, 25) ** 2
    p[[3, 7]] = p[5]  # ties
    expected = multipletests(p, method=reference)[1]
    np.testing.assert_allclose(adjust(p, method), expected, rtol=1e-12)


def test_untested_family_members_count_as_one():
    from statsmodels.stats.multitest import multipletests

    p = [0.01, 0.02, 0.2]
    for method, reference in (("bh", "fdr_bh"), ("holm", "holm"), ("bonferroni", "bonferroni")):
        expected = multipletests([*p, 1.0, 1.0], method=reference)[1][:3]
        np.testing.assert_allclose(adjust(p, method, family_size=5), expected, rtol=1e-12)
    with pytest.raises(ValueError, match="smaller than"):
        adjust(p, "bh", family_size=2)
    with pytest.raises(ValueError, match="unknown correction"):
        adjust(p, "sidak")  # type: ignore[arg-type]


def test_correct_records_the_family_on_assessed_results_only():
    rng = np.random.default_rng(2)
    results = [welch_t(rng.normal(size=20), rng.normal(0.8 * k, 1, 20)) for k in range(4)]
    results.append(welch_t([1.0], [2.0]))  # not assessed: too few values
    corrected = correct(results, "holm", 0.05)
    assert [r.family_size for r in corrected] == [4, 4, 4, 4, None]
    assert all(r.correction == "holm" for r in corrected[:4]) and corrected[4].correction is None
    assert corrected[4].significant is None and not corrected[4].assessed
    for raw, adjusted in zip(results[:4], corrected[:4], strict=True):
        assert adjusted.adjusted_p >= raw.p_value
        assert adjusted.significant == (adjusted.adjusted_p < 0.05)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=30), st.sampled_from(CORRECTIONS))
def test_adjusted_p_values_are_valid_monotone_and_at_least_raw(p, method):
    raw = np.array(p)
    adjusted = adjust(raw, method)
    assert ((adjusted >= 0) & (adjusted <= 1)).all()
    assert (adjusted >= raw - 1e-15).all()
    order = np.argsort(raw, kind="stable")
    assert (np.diff(adjusted[order]) >= -1e-12).all()


# -------------------------------------------------------- hypothesis tests
samples = st.lists(
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False), min_size=3, max_size=25
)


@settings(max_examples=40, deadline=None)
@given(samples, samples)
def test_p_values_lie_in_the_unit_interval_and_sides_flip_the_effect(a, b):
    for test in (welch_t, mann_whitney, brown_forsythe):
        forward, backward = test(a, b), test(b, a)
        assert forward.assessed == backward.assessed
        if not forward.assessed:  # e.g. a dynamic range that cannot be rescaled losslessly
            assert forward.reason and forward.p_value is None
            continue
        assert 0 <= forward.p_value <= 1
        assert forward.p_value == pytest.approx(backward.p_value, rel=1e-9, abs=1e-12)
        if forward.effect is not None and forward.effect_name != "sd_ratio":
            assert forward.effect == pytest.approx(-backward.effect, rel=1e-9, abs=1e-9)


# ------------------------------------------------------------------ effects
def test_effect_sizes_by_hand():
    b, c = [2.0, 4.0, 6.0], [5.0, 7.0, 9.0, 11.0]
    # pooled variance = (2*4 + 3*6.6667) / 5 = 5.6, d = (8 - 4) / sqrt(5.6)
    assert effects.cohen_d(b, c) == pytest.approx(4 / math.sqrt(5.6))
    j = math.gamma(2.5) / (math.sqrt(2.5) * math.gamma(2.0))
    assert effects.hedges_g(b, c) == pytest.approx(4 / math.sqrt(5.6) * j)
    assert effects.rank_biserial(b, c) == pytest.approx((11 - 1) / 12)  # 11 wins, 1 loss
    assert effects.cramers_v([[10, 0], [0, 10]]) == pytest.approx(1.0)
    assert effects.cramers_v([[10, 10], [10, 10]]) == pytest.approx(0.0)
    assert effects.rate_ratio(10, 2.0, 30, 3.0) == pytest.approx(2.0)
    assert effects.rate_ratio(0, 2.0, 30, 3.0) is None
    assert effects.odds_ratio([[2, 8], [5, 5]]) == pytest.approx(2 * 5 / (8 * 5))
    assert effects.cohen_d([1.0, 1.0], [2.0, 2.0]) is None


def test_agreement_metrics_by_hand():
    b, c = [1.0, 2.0, 3.0, 4.0], [1.5, 1.5, 3.5, 5.0]
    d = np.array([0.5, -0.5, 0.5, 1.0])
    assert effects.bias(b, c) == pytest.approx(0.375)
    assert effects.mae(b, c) == pytest.approx(0.625)
    assert effects.rmse(b, c) == pytest.approx(math.sqrt((0.25 * 3 + 1) / 4))
    agreement = effects.bland_altman(b, c)
    sd = float(d.std(ddof=1))
    assert (agreement.bias, agreement.sd) == pytest.approx((0.375, sd))
    assert (agreement.lower, agreement.upper) == pytest.approx(
        (0.375 - 1.96 * sd, 0.375 + 1.96 * sd)
    )
    mb, mc = 2.5, 2.875
    sxy = np.mean((np.array(b) - mb) * (np.array(c) - mc))
    expected = 2 * sxy / (np.var(b) + np.var(c) + (mb - mc) ** 2)
    assert effects.concordance(b, c) == pytest.approx(expected)
    assert effects.mape(b, c) == pytest.approx(np.mean(np.abs(d) / np.array(b)) * 100)
    assert effects.smape([0.0, 2.0], [0.0, 4.0]) == pytest.approx((0 + 2 * 2 / 6) / 2 * 100)
    assert effects.r_squared(b, c) == pytest.approx(1 - (d**2).sum() / 5.0)
    assert effects.mape([0.0, 1.0], [1.0, 1.0]) is None
    assert effects.r_squared([2.0, 2.0], [1.0, 3.0]) is None


# ------------------------------------------------- explicit edge behaviour
def test_constant_samples_have_explicit_outcomes():
    same = welch_t([3.0, 3.0, 3.0], [3.0, 3.0])
    assert same.assessed and same.p_value == 1.0 and same.statistic == 0.0
    apart = welch_t([3.0, 3.0, 3.0], [4.0, 4.0])
    assert apart.p_value == 0.0 and apart.statistic == math.inf and apart.effect == 1.0
    assert welch_t([3.0, 3.0], [4.0, 4.0], alternative="less").p_value == 1.0
    assert mann_whitney([2.0, 2.0], [2.0, 2.0, 2.0]).p_value == 1.0
    assert anderson_darling([1.0, 1.0], [1.0, 1.0]).p_value == 1.0
    assert brown_forsythe([1.0, 1.0], [5.0, 5.0]).p_value == 1.0
    assert brown_forsythe([1.0, 1.0, 1.0], [0.0, 2.0, 0.0, 2.0]).p_value == 0.0
    flat = paired_t([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert flat.p_value == 0.0 and flat.effect == 1.0
    unchanged = [1.0, 2.0, 3.0]
    for test in (paired_t, wilcoxon_signed_rank, sign_test):
        assert test(unchanged, unchanged).p_value == 1.0
    assert mann_kendall([4.0, 4.0, 4.0, 4.0]).p_value == 1.0
    assert poisson_rate((0, 0)).p_value == 1.0
    assert two_proportion((0, 0), (10, 20)).p_value == 1.0
    assert mcnemar([[5, 0], [0, 7]]).p_value == 1.0
    assert stuart_maxwell(np.diag([4, 5, 6])).p_value == 1.0
    exact = slope_difference(np.arange(6.0), np.zeros(6), np.arange(6.0) * 2)
    assert exact.p_value == 0.0 and exact.effect == pytest.approx(2.0)


def test_undefined_or_too_small_samples_are_not_assessed():
    cases = [
        welch_t([1.0], [1.0, 2.0]),
        mann_whitney([], [1.0, 2.0]),
        paired_t([1.0], [2.0]),
        paired_t([1.0, 2.0], [2.0]),
        welch_t([1.0, math.nan], [1.0, 2.0]),
        welch_t([1.0, math.inf, 2.0], [1.0, 2.0]),
        correlation_test([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
        correlation_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        mann_kendall([1.0, 2.0]),
        chi_square([[0, 0], [3, 4]]),
        morans_i([2.0, 2.0, 2.0], np.ones((3, 3))),
        morans_i([1.0, 2.0, 3.0], np.zeros((3, 3))),
        two_proportion((1, 2), (0, 5)),
        welch_t_from_moments(Moments(1, 1.0, 0.0), Moments(3, 1.0, 1.0)),
        lag_test({0: 1.0, 1: 2.0}, {0: 1.0, 1: 2.0}, 1),
    ]
    for result in cases:
        assert not result.assessed and result.reason and result.p_value is None
        assert result.significant is None


@pytest.mark.parametrize("name", sorted(TESTS.names()))
def test_each_component_enforces_its_declared_minimum(name):
    test = TESTS.create(name)
    if test.needs in ("counts", "proportions"):
        assert test.run(TestInput(counts=(test.minimum, 1), totals=(3, 3))).assessed
        if test.needs == "proportions":
            assert not test.run(TestInput(counts=(0, 0), totals=(0, 3))).assessed
        return

    def values(n, shift=0.0):
        return np.arange(n, dtype=float) * 1.5 + shift + np.array([0.3, -0.1] * n)[:n] ** 2

    if test.needs == "moments":
        enough = Moments(test.minimum, 1.0, 1.0), Moments(test.minimum, 2.0, 1.5)
        short = Moments(test.minimum - 1, 1.0, 1.0), enough[1]
        assert test.run(TestInput(moments=enough)).assessed
        assert not test.run(TestInput(moments=short)).assessed
        return
    n = test.minimum
    assert test.run(TestInput(baseline=values(n), candidate=values(n, 0.7))).assessed
    shorter = values(n - 1)
    result = test.run(TestInput(baseline=shorter, candidate=values(n - 1, 0.7)))
    assert not result.assessed and result.reason


def test_without_scipy_every_test_is_not_assessed(monkeypatch):
    for module in ("scipy", "scipy.stats"):
        monkeypatch.setitem(sys.modules, module, None)
    x, y = [1.0, 2.0, 3.0, 4.0], [2.0, 3.5, 3.0, 6.0]
    for result in (
        welch_t(x, y), mann_whitney(x, y), paired_t(x, y), wilcoxon_signed_rank(x, y),
        poisson_rate((3, 5)), two_proportion((3, 5), (10, 10)), mann_kendall(y),
        correlation_test(x, y), tost_paired(x, y, 1.0), chi_square([[1, 2], [3, 4]]),
    ):  # fmt: skip
        assert not result.assessed and "qavf[stats]" in result.reason
    decision = decide("material_and_significant", True, welch_t(x, y))
    assert decision.outcome == "not_assessed"


def test_resampling_is_deterministic_by_seed():
    rng = np.random.default_rng(3)
    b, c = rng.normal(size=40), rng.normal(0.3, 1, 45)
    first = bootstrap_difference(b, c, seed=11)
    assert first == bootstrap_difference(b, c, seed=11)
    assert first.seed == 11 and first.resamples == 2000
    other = bootstrap_difference(b, c, seed=12)
    assert (other.p_value, other.ci_low) != (first.p_value, first.ci_low)
    for test in (permutation, paired_permutation):
        long_b, long_c = rng.normal(size=30), rng.normal(size=30)
        assert test(long_b, long_c, seed=5) == test(long_b, long_c, seed=5)
        assert test(long_b, long_c, seed=5).p_value != test(long_b, long_c, seed=6).p_value
    ring = {i: [(i + 1) % 20] for i in range(20)}
    values = rng.normal(size=20)
    assert morans_i(values, ring, seed=1) == morans_i(values, ring, seed=1)


def test_the_seed_fixes_the_resamples_exactly():
    # The documented scheme, rebuilt from NumPy's PCG64 (platform independent):
    # baseline indices, then candidate indices, for all resamples of a batch.
    b = np.linspace(0.0, 1.0, 15)
    c = np.linspace(0.2, 1.3, 17)
    rng = np.random.Generator(np.random.PCG64(7))
    drawn_b = b[rng.integers(0, 15, size=(999, 15))].mean(axis=1)
    drawn_c = c[rng.integers(0, 17, size=(999, 17))].mean(axis=1)
    distribution = drawn_c - drawn_b
    expected = min(
        1.0, 2 * min(np.sum(distribution <= 0) + 1, np.sum(distribution >= 0) + 1) / 1000
    )
    result = bootstrap_difference(b, c, resamples=999, seed=7)
    assert result.p_value == expected
    assert result.ci_low == pytest.approx(np.quantile(distribution, 0.025), rel=1e-12)
    assert result == bootstrap_difference(list(b), list(c), resamples=999, seed=7)


def test_work_budget_makes_tests_not_assessed():
    x = np.arange(100.0)
    for result in (
        bootstrap_difference(x, x + 1, resamples=2000, max_work=1000),
        permutation(x, x + 1, resamples=2000, max_work=1000),
        slope_difference(x, x, x * 2 + np.sin(x), resamples=2000, max_work=1000),
        morans_i(x, {i: [(i + 1) % 100] for i in range(100)}, max_work=10),
        mann_kendall(x, max_work=100),
    ):
        assert not result.assessed and "max_work" in result.reason


def test_results_round_trip_through_json_with_non_finite_values():
    result = welch_t([3.0, 3.0, 3.0], [4.0, 4.0])
    again = TestResult.model_validate_json(result.model_dump_json())
    assert again == result and again.statistic == math.inf
    family = Significance(
        method="welch_t", decision="material_and_significant", alpha=0.05, correction="bh",
        family_size=1, tests=[result],
    )  # fmt: skip
    assert Significance.model_validate_json(family.model_dump_json()) == family


# ---------------------------------------------------------------- decisions
def _judged(p: float | None, assessed: bool = True) -> TestResult:
    if not assessed:
        return TestResult(method="t", assessed=False, reason="too few values")
    return TestResult(method="t", p_value=p).judged(0.05)


@pytest.mark.parametrize(
    "policy,material,p,outcome",
    [
        ("material_and_significant", True, 0.01, "fail"),
        ("material_and_significant", True, 0.20, "pass"),
        ("material_and_significant", False, 0.01, "pass"),
        ("material_and_significant", False, None, "pass"),
        ("material_and_significant", True, "na", "not_assessed"),
        ("material_and_significant", False, "na", "pass"),
        ("material", True, 0.20, "fail"),
        ("material", False, 0.01, "pass"),
        ("material", True, "na", "fail"),
        ("significant", False, 0.01, "fail"),
        ("significant", True, 0.20, "pass"),
        ("significant", False, "na", "not_assessed"),
        ("equivalence", True, 0.01, "pass"),
        ("equivalence", False, 0.20, "fail"),
        ("equivalence", False, "na", "not_assessed"),
    ],
)
def test_decision_truth_table(policy, material, p, outcome):
    test = None if p is None else _judged(None, assessed=False) if p == "na" else _judged(p)
    if p is None and policy != "material_and_significant":
        pytest.skip("covered elsewhere")
    assert decide(policy, material, test).outcome == outcome


@pytest.mark.parametrize("policy", ["material_and_significant", "significant", "equivalence"])
def test_measures_without_a_sound_test_fall_back_to_materiality(policy):
    for material, outcome in ((True, "fail"), (False, "pass")):
        decision = decide(policy, material, None, applicable=False)
        assert decision.outcome == outcome and decision.basis == "materiality"
        assert "no sound test" in decision.reason


def test_equivalence_needs_the_margin_to_be_shown():
    rng = np.random.default_rng(9)
    b = rng.normal(10, 1, 200)
    close = tost_paired(b, b + rng.normal(0.02, 0.1, 200), 0.1).judged(0.05)
    far = tost_paired(b, b + rng.normal(0.3, 0.1, 200), 0.1).judged(0.05)
    assert decide("equivalence", False, close).outcome == "pass"
    assert decide("equivalence", True, far).outcome == "fail"
    noisy = tost_paired(b[:5], b[:5] + rng.normal(0, 5, 5), 0.1).judged(0.05)
    assert decide("equivalence", False, noisy).outcome == "fail"  # not shown is not a pass


# ------------------------------------------------------------------ plugins
class Halved(InferenceTest):
    """A plugin test: half the Welch p-value (for registration tests only)."""

    name: ClassVar[str] = "halved_welch"
    needs: ClassVar = "moments"

    def run(self, data: TestInput) -> TestResult:
        assert data.moments is not None
        result = welch_t_from_moments(*data.moments)
        return result.model_copy(update={"method": self.name, "p_value": result.p_value / 2})


@pytest.fixture
def restore_tests():
    TESTS.discover()
    saved = (dict(TESTS._components), dict(TESTS._installed), dict(TESTS.plugins))
    loaded = TESTS._entry_points_loaded
    yield
    TESTS._components, TESTS._installed, TESTS.plugins = saved
    TESTS._entry_points_loaded = loaded


def test_tests_register_in_code(restore_tests):
    register_test(Halved)
    test = TESTS.create("halved_welch")
    data = TestInput(moments=(Moments(10, 1.0, 1.0), Moments(10, 2.0, 1.0)))
    assert test.run(data).p_value == pytest.approx(welch_t_from_moments(*data.moments).p_value / 2)
    with pytest.raises(ValueError, match="duplicate statistical test 'welch_t'"):

        class Other(Halved):
            name: ClassVar[str] = "welch_t"

        register_test(Other)
    with pytest.raises(ValueError, match="unknown statistical test 'nope'"):
        TESTS.create("nope")


def test_tests_load_from_entry_points_only_when_enabled(restore_tests, monkeypatch):
    module = ModuleType("qavf_fake_inference_plugin")
    Halved.__module__ = module.__name__
    module.Halved = Halved
    try:
        monkeypatch.setattr(
            "qavf.plugins.registry.entry_points",
            lambda group: (
                [SimpleNamespace(name="halving", load=lambda: module)]
                if group == "qavf.inference.tests"
                else []
            ),
        )
        TESTS.refresh()
        assert "halved_welch" not in TESTS
        with entry_point_scope(True):
            assert "halved_welch" in TESTS
        assert TESTS.plugins == {"halving": "loaded"}
        from qavf.plugins.registry import plugin_status

        assert plugin_status()["inference.tests:halving"] == "loaded"
    finally:
        Halved.__module__ = __name__
