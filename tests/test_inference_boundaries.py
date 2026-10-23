"""Inference results against independent oracles, and resource boundaries."""

from __future__ import annotations

import math
import tracemalloc
from itertools import combinations

import numpy as np
import pytest

from qavf.inference import (
    TestResult,
    adjust,
    bootstrap_difference,
    correct,
    correlation_test,
    decide,
    local_morans_i,
    morans_i,
    paired_bootstrap,
    paired_permutation,
    permutation,
    poisson_rate,
    resampling,
    stuart_maxwell,
)
from qavf.inference.effects import quantile_deltas
from qavf.inference.independent import _jackknife
from qavf.inference.resampling import permutation_p_value
from qavf.inference.trend import lag_test, slope_difference


@pytest.mark.parametrize("p", [None, math.nan, math.inf, -0.1, 1.1])
def test_invalid_probabilities_never_become_a_pass(p):
    result = TestResult(method="plugin", p_value=p)
    assert not result.assessed
    for policy in ("significant", "material_and_significant", "equivalence"):
        assert decide(policy, True, result).outcome == "not_assessed"
    corrected = correct([result], "bh", 0.05)
    assert not corrected[0].assessed
    assert not corrected[0].significant
    assert TestResult.model_validate_json(result.model_dump_json()).assessed is False


def test_probability_state_is_enforced_on_copy_and_decision_boundary():
    original = TestResult(method="plugin", p_value=0.2).judged(0.05)
    assert not original.model_copy(update={"p_value": math.nan}).assessed
    # An unvalidated third-party object must still fail closed at the decision boundary.
    unvalidated = TestResult.model_construct(method="plugin", p_value=math.nan, significant=False)
    assert decide("significant", True, unvalidated).outcome == "not_assessed"
    changed = original.model_copy(update={"p_value": 0.001, "significant": False})
    assert changed.significant is True
    assert decide("significant", True, changed).outcome == "fail"
    with pytest.raises(ValueError, match="alpha"):
        original.judged(math.nan)


def test_valid_infinite_statistics_remain_assessed():
    result = TestResult(method="constant_difference", statistic=math.inf, p_value=0.0)
    assert result.assessed
    assert decide("significant", True, result).outcome == "fail"


@pytest.mark.parametrize(
    "method,reference",
    [
        ("holm", "holm"),
        ("bh", "fdr_bh"),
        ("by", "fdr_by"),
    ],
)
def test_implicit_probability_padding_matches_statsmodels(method, reference):
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    p = np.array([0.2, 0.0, 1.0, 0.001, 0.2, 0.8])
    for size in [6, 7, 129, 1000]:
        expected = multipletests(np.r_[p, np.ones(size - len(p))], method=reference)[1][: len(p)]
        np.testing.assert_allclose(adjust(p, method, size), expected, atol=1e-15, rtol=2e-14)


def test_single_probability_large_family_uses_small_owned_output():
    tracemalloc.start()
    result = adjust([0.01], "holm", 1_000_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.tolist() == [1.0]
    assert result.base is None
    assert peak < 200_000, peak


@pytest.mark.parametrize("method", [permutation, paired_bootstrap, bootstrap_difference])
def test_resampling_rejects_nan_and_wrong_shape_callbacks(method):
    baseline, candidate = [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]
    bad = method(baseline, candidate, statistic=lambda a: a.mean(axis=-1) * math.nan, resamples=99)
    assert not bad.assessed
    assert "not finite" in bad.reason
    wrong = method(baseline, candidate, statistic=lambda a: np.array(1.0), resamples=99)
    assert not wrong.assessed
    assert "shape" in wrong.reason


def test_permutation_arbitrary_callback_preserves_original_units():
    stats = pytest.importorskip("scipy.stats")
    before, after = np.array([1.0, 2.0, 3.0]), np.array([20.0, 21.0, 22.0])

    def statistic(a):
        return (a > 10).mean(axis=-1)

    expected = stats.permutation_test(
        (before, after),
        lambda b, c, axis: (c > 10).mean(axis=axis) - (b > 10).mean(axis=axis),
        n_resamples=np.inf,
    )
    result = permutation(before, after, statistic=statistic, resamples=99)
    assert result.assessed
    assert result.effect == expected.statistic == 1
    assert result.p_value == pytest.approx(expected.pvalue)


def test_paired_variance_callback_uses_squared_original_units():
    before, after = np.zeros(5), np.array([1.0, 5.0, 2.0, 8.0, 4.0])

    def statistic(a):
        return a.var(axis=-1)

    result = paired_bootstrap(before, after, statistic=statistic, resamples=99)
    assert result.effect == pytest.approx(np.var(after))
    with pytest.raises(ValueError, match="interval"):
        paired_bootstrap(before, after, interval="typo")


@pytest.mark.parametrize("method", [permutation, paired_permutation, paired_bootstrap])
def test_resample_counts_are_validated(method):
    with pytest.raises(ValueError, match="resamples"):
        method([1.0, 2.0], [3.0, 4.0], resamples=0)


def test_batched_exact_permutation_matches_enumeration(monkeypatch):
    monkeypatch.setattr(resampling, "_CHUNK", 12)
    before, after = np.array([1.0, 3.0, 8.0]), np.array([2.0, 6.0, 11.0])
    values = np.r_[before, after]
    distribution = []
    for picked in combinations(range(6), 3):
        selected = set(picked)
        distribution.append(
            np.mean([v for i, v in enumerate(values) if i in selected])
            - np.mean([v for i, v in enumerate(values) if i not in selected])
        )
    expected = permutation_p_value(np.array(distribution), after.mean() - before.mean(), exact=True)
    result = permutation(before, after, resamples=99)
    assert result.p_value == pytest.approx(expected)


def test_batched_jackknife_matches_independent_leave_one_out(monkeypatch):
    monkeypatch.setattr(resampling, "_CHUNK", 15)
    values = np.array([2.0, 7.0, 3.0, 13.0, 8.0, 9.0])

    def statistic(a):
        return np.median(a, axis=-1)

    expected = np.array([np.median(np.delete(values, i)) for i in range(len(values))])
    np.testing.assert_array_equal(_jackknife(values, statistic), expected)


def test_block_bootstrap_batches_preserve_seeded_results(monkeypatch):
    x = np.arange(20.0)
    b, c = np.sin(x), np.sin(x) + 0.1 * x + np.cos(x * 0.7)
    initial_slope = slope_difference(x, b, c, resamples=99, seed=18)
    initial_lag = lag_test(dict(enumerate(b)), dict(enumerate(c)), 1, resamples=99, seed=18)
    monkeypatch.setattr(resampling, "_CHUNK", 60)
    batched_slope = slope_difference(x, b, c, resamples=99, seed=18)
    batched_lag = lag_test(dict(enumerate(b)), dict(enumerate(c)), 1, resamples=99, seed=18)
    for first, second in [(initial_slope, batched_slope), (initial_lag, batched_lag)]:
        assert first.assessed and second.assessed
        assert second.p_value == pytest.approx(first.p_value, abs=1e-14)
        assert second.ci_low == pytest.approx(first.ci_low, abs=1e-14)
        assert second.ci_high == pytest.approx(first.ci_high, abs=1e-14)


def test_sparse_spatial_work_includes_all_permuted_locations():
    values = np.arange(2000, dtype=float)
    edges = {0: {1: 1.0}, 1: {0: 1.0}}
    local = local_morans_i(values, edges, permutations=199, max_work=400)
    assert all(not result.assessed for result in local.results)
    assert "max_work" in local.results[0].reason
    assert not morans_i(values, edges, permutations=199, max_work=400).assessed


def test_local_moran_batched_randomization_matches_independent_oracle(monkeypatch):
    values = np.array([3.0, 2.0, 7.0, 9.0, 4.0])
    weights = {0: {1: 1.0, 2: 1.0}, 1: {0: 1.0}, 2: {0: 1.0}, 3: {4: 1.0}}
    z = values - values.mean()
    denominator = (z * z).mean()
    rng = np.random.Generator(np.random.PCG64(23))
    draws = np.stack([rng.permutation(4)[:2].copy() for _ in range(99)])
    expected = []
    for i in range(5):
        row = weights.get(i, {})
        if not row:
            expected.append(None)
            continue
        observed = z[i] * np.mean([z[j] for j in row]) / denominator
        positions = draws[:, : len(row)]
        positions = positions + (positions >= i)
        randomized = z[i] * z[positions].mean(axis=1) / denominator
        extreme = np.count_nonzero(randomized >= observed - abs(observed) * 1e-12)
        expected.append((extreme + 1) / 100)
    monkeypatch.setattr(resampling, "_CHUNK", 8)
    result = local_morans_i(values, weights, permutations=99, seed=23)
    for i, p in enumerate(expected):
        if p is None:
            assert not result.results[i].assessed
        else:
            assert result.results[i].p_value == pytest.approx(p)


def test_poisson_common_exposure_scaling_leaves_probability_unchanged():
    expected = poisson_rate((5, 5), (1.0, 1.0))
    actual = poisson_rate((5, 5), (1e308, 1e308))
    assert actual.assessed and expected.assessed
    assert actual.p_value == expected.p_value == 1
    assert actual.effect == expected.effect == 1


def test_singular_category_changes_are_explicitly_unassessed():
    result = stuart_maxwell([[5, 1, 0, 0], [2, 5, 0, 0], [0, 0, 5, 3], [0, 0, 1, 5]])
    assert not result.assessed
    assert "singular covariance" in result.reason


def test_effective_sample_size_has_work_admission():
    x = np.arange(100.0)
    result = correlation_test(np.sin(x), np.cos(x), dependence="effective_n", max_work=100)
    assert not result.assessed
    assert "max_work" in result.reason


def test_vector_quantiles_preserve_probability_order_and_duplicates():
    before, after = np.array([1.0, 5.0, 3.0, 8.0]), np.array([4.0, 2.0, 7.0, 9.0])
    probabilities = [0.9, 0.0, 0.5, 0.9, 1.0]
    expected = [(p, np.quantile(before, p), np.quantile(after, p)) for p in probabilities]
    assert quantile_deltas(before, after, iter(probabilities)) == expected


def test_compact_adjacency_inference_matches_mapping_without_row_iteration(monkeypatch):
    from qavf._adjacency import CompactAdjacency

    compact = CompactAdjacency(
        np.array([0, 2, 3, 4, 4]),
        np.array([1, 2, 0, 0]),
        np.array([0.5, 0.5, 1.0, 1.0]),
    )
    weights = {0: {1: 0.5, 2: 0.5}, 1: {0: 1.0}, 2: {0: 1.0}, 3: {}}
    expected = morans_i([1.0, 5.0, 2.0, 7.0], weights, permutations=99)

    def no_rows(_self):
        raise AssertionError("compact adjacency must use its numerical edge arrays")

    monkeypatch.setattr(CompactAdjacency, "items", no_rows)
    actual = morans_i([1.0, 5.0, 2.0, 7.0], compact, permutations=99)
    assert actual.statistic == expected.statistic
    assert actual.p_value == expected.p_value


def test_rate_ratio_avoids_intermediate_overflow_and_underflow():
    from fractions import Fraction

    from qavf.inference.effects import rate_ratio

    for counts, exposures in [
        ((1e300, 1e300), (1e-300, 1e-300)),
        ((1e-300, 1e-300), (1e300, 1e300)),
        ((10.0, 5.0), (1e-310, 1.0)),
    ]:
        b, c = counts
        eb, ec = exposures
        expected = float(Fraction(c) * Fraction(eb) / (Fraction(b) * Fraction(ec)))
        assert rate_ratio(b, eb, c, ec) == pytest.approx(expected, abs=0)
    assert rate_ratio(5, 1e-310, 0, 1) == 0
    assert rate_ratio(5, 1e308, 5, 1e-308) is None


def test_jackknife_peak_allocation_stays_below_quadratic_sample_matrix(monkeypatch):
    monkeypatch.setattr(resampling, "_CHUNK", 4096)
    values = np.linspace(1.0, 5.0, 1500)
    expected = (values.sum() - values) / (len(values) - 1)
    tracemalloc.start()
    actual = _jackknife(values, lambda a: a.mean(axis=-1))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    np.testing.assert_allclose(actual, expected)
    assert peak < 2_000_000, peak


@pytest.mark.parametrize("paired", [False, True])
def test_exact_permutation_peak_allocation_is_batched(monkeypatch, paired):
    monkeypatch.setattr(resampling, "_CHUNK", 4096)
    n = 14 if paired else 9
    before = np.arange(n, dtype=float)
    after = before + 1
    method = paired_permutation if paired else permutation
    tracemalloc.start()
    result = method(before, after, resamples=100_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.assessed
    assert result.seed is None  # An exact enumeration, rather than a sampling shortcut.
    assert peak < 2_000_000, peak


def test_array_inference_is_core_only_and_registry_exports_remain_compatible():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""\
        import importlib.abc
        import sys

        class NoExtras(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'scipy', 'pandas', 'pyproj', 'statsmodels'}:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, NoExtras())
        from qavf.inference import morans_i, permutation, welch_t
        assert 'qavf.engine.context' not in sys.modules
        assert 'qavf.reports.models' not in sys.modules
        assert permutation([1, 2, 3], [2, 4, 6], resamples=99).assessed
        assert morans_i([1, 2, 3], {0: [1], 1: [2]}, permutations=99).assessed
        assert not welch_t([1, 2, 3], [2, 4, 6]).assessed

        from qavf.inference import TESTS, InferenceTest, TestInput, register_test
        from qavf.inference.registry import TESTS as direct_registry
        from qavf.reports.models import SnapshotModel as compatibility_base
        from qavf.reports.models import decode_non_finite, encode_non_finite
        from qavf.snapshots import SnapshotModel
        assert TESTS is direct_registry
        assert compatibility_base is SnapshotModel
        assert decode_non_finite(encode_non_finite(float('inf'))) == float('inf')
        assert callable(register_test) and isinstance(TestInput(), TestInput)
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
