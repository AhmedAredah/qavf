"""Global and local Moran's I and the rank correlation test against independent oracles,
and their calibration under spatial randomness (fixed seeds)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qavf.inference import correct, local_morans_i, morans_i, rank_correlation_test
from qavf.inference.spatial import morans_statistic

ALPHA = 0.05


def _rook(n: int) -> dict[int, dict[int, float]]:
    """Rook neighbours of an ``n x n`` lattice, index ``row * n + column`` (binary)."""
    rows: dict[int, dict[int, float]] = {}
    for r in range(n):
        for c in range(n):
            near = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            rows[r * n + c] = {i * n + j: 1.0 for i, j in near if 0 <= i < n and 0 <= j < n}
    return rows


def _dense(weights: dict[int, dict[int, float]], size: int, standardize: bool) -> np.ndarray:
    matrix = np.zeros((size, size))
    for i, row in weights.items():
        for j, w in row.items():
            matrix[i, j] = w
    if standardize:
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def _tolerance(reps: int) -> float:
    return 3.5 * math.sqrt(ALPHA * (1 - ALPHA) / reps)


@pytest.mark.parametrize("standardize", [False, True])
def test_global_morans_i_matches_a_numpy_formula(standardize):
    rng = np.random.default_rng(7)
    values = rng.normal(size=36) + np.repeat(np.arange(6), 6) * 0.5
    weights = _rook(6)
    matrix = _dense(weights, 36, standardize)
    z = values - values.mean()
    expected = len(z) / matrix.sum() * (z @ matrix @ z) / (z @ z)
    result = morans_i(values, weights, row_standardize=standardize, seed=1)
    assert result.statistic == pytest.approx(expected, rel=1e-12)
    statistic = morans_statistic(values, weights, row_standardize=standardize)
    assert statistic == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="constant"):
        morans_statistic(np.ones(36), weights)


def test_local_morans_i_matches_the_lisa_formula_and_names_quadrants():
    rng = np.random.default_rng(11)
    values = rng.normal(size=25)
    weights = _rook(5)
    matrix = _dense(weights, 25, True)
    z = values - values.mean()
    expected = z * (matrix @ z) / ((z @ z) / len(z))
    local = local_morans_i(values, weights, permutations=99, seed=3)
    assert [r.statistic for r in local.results] == pytest.approx(list(expected), rel=1e-12)
    lag = matrix @ z
    for i, quadrant in enumerate(local.quadrants):
        assert quadrant == ("H" if z[i] > 0 else "L") + ("H" if lag[i] > 0 else "L")
    # Local statistics sum to n times the global one (row-standardized weights).
    globally = morans_i(values, weights, row_standardize=True, seed=1).statistic
    assert sum(expected) / len(z) == pytest.approx(globally, rel=1e-10)


def test_islands_and_unusable_values_are_not_assessed():
    weights = {0: {1: 1.0}, 1: {0: 1.0}, 2: {1: 1.0}, 3: {}}
    local = local_morans_i([1.0, 2.0, 3.0, 4.0], weights, permutations=19)
    assert not local.results[3].assessed and "no neighbours" in local.results[3].reason
    assert all(result.assessed for result in local.results[:3])
    constant = local_morans_i([2.0] * 4, weights, permutations=19)
    assert not any(result.assessed for result in constant.results)
    assert "constant" in constant.results[0].reason
    budget = local_morans_i([1.0, 2.0, 3.0, 4.0], weights, permutations=999, max_work=10)
    assert "max_work" in budget.results[0].reason


def test_lisa_finds_a_planted_hot_spot_and_nothing_in_noise():
    n = 10
    rng = np.random.default_rng(5)
    values = rng.normal(size=n * n)
    block = {r * n + c for r in range(2, 5) for c in range(2, 5)}
    for i in block:
        values[i] += 4.0
    local = local_morans_i(values, _rook(n), permutations=999, seed=2)
    found = {
        i
        for i, result in enumerate(correct(local.results, "bh", ALPHA))
        if result.significant and local.quadrants[i] == "HH"
    }
    # The block's centre (all neighbours in the block) and some of its edge are found;
    # nothing outside it is.
    assert 3 * n + 3 in found and len(found) >= 3 and found <= block, found
    noise_hits = 0
    for seed in range(10):
        noise = np.random.default_rng(100 + seed).normal(size=n * n)
        quiet = local_morans_i(noise, _rook(n), permutations=499, seed=seed)
        noise_hits += sum(bool(r.significant) for r in correct(quiet.results, "bh", ALPHA))
    assert noise_hits <= 2


def test_global_morans_i_holds_its_level_under_spatial_randomness():
    reps, weights = 400, _rook(8)
    rng = np.random.default_rng(21)
    rejected = sum(
        morans_i(rng.normal(size=64), weights, permutations=199, seed=rep).p_value < ALPHA
        for rep in range(reps)
    )
    assert abs(rejected / reps - ALPHA) <= _tolerance(reps)


def test_local_morans_i_holds_its_level_and_bh_controls_the_family():
    # 1999 permutations make the smallest p-value (1/2000) small enough for the
    # correction of 36 zones to reject at all (alpha / 36 = 0.0014).
    reps, n = 150, 6
    weights = _rook(n)
    rng = np.random.default_rng(31)
    raw = families = 0
    for rep in range(reps):
        local = local_morans_i(rng.normal(size=n * n), weights, permutations=1999, seed=rep)
        raw += sum(r.p_value < ALPHA for r in local.results)
        families += any(r.significant for r in correct(local.results, "bh", ALPHA))
    tests = reps * n * n
    # Per location the conditional permutation test rejects at about alpha (one-sided).
    assert abs(raw / tests - ALPHA) <= _tolerance(tests) * 3
    # Under the complete null the false discovery rate is the family-wise rate.
    assert families / reps <= ALPHA + _tolerance(reps)


def test_rank_correlation_test_matches_an_independent_formula():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(8)
    x = rng.normal(size=30)
    y = x + rng.normal(scale=1.5, size=30)
    result = rank_correlation_test(x, y, rho=0.2, alternative="less")
    rho = scipy_stats.spearmanr(x, y).statistic
    z = (math.atanh(rho) - math.atanh(0.2)) * math.sqrt(27 / 1.06)
    assert result.effect == pytest.approx(rho, rel=1e-12)
    assert result.statistic == pytest.approx(z, rel=1e-12)
    assert result.p_value == pytest.approx(scipy_stats.norm.cdf(z), rel=1e-10)
    assert not rank_correlation_test(x[:3], y[:3]).assessed
    assert not rank_correlation_test(np.ones(10), y[:10]).assessed


def test_rank_correlation_test_holds_its_level():
    pytest.importorskip("scipy")
    reps = 1000
    rng = np.random.default_rng(41)
    rejected = sum(
        rank_correlation_test(rng.normal(size=25), rng.normal(size=25)).p_value < ALPHA
        for _ in range(reps)
    )
    assert abs(rejected / reps - ALPHA) <= _tolerance(reps)
