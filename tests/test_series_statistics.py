"""Exact series statistics against NumPy/SciPy and closed-form oracles."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qavf.statistics.series import (
    average_ranks,
    best_lag,
    normal_equations,
    pearson,
    spearman,
)


def test_least_squares_slope_is_exact_for_large_and_rational_values():
    x = list(range(6))
    y = [10**20 + 3 * k for k in x]
    assert normal_equations(x, y).slope == 3
    rational = [Fraction(1, 3) * k + Fraction(7, 2) for k in x]
    assert normal_equations([Fraction(k) for k in x], rational).slope == Fraction(1, 3)
    assert normal_equations([2, 2, 2], [1, 2, 3]).slope is None


@settings(max_examples=50, deadline=None)
@given(st.lists(st.integers(-1000, 1000), min_size=3, max_size=30))
def test_slope_matches_numpy(values):
    x = list(range(len(values)))
    slope = normal_equations(x, values).slope
    assert slope is not None
    assert float(slope) == pytest.approx(np.polyfit(x, values, 1)[0], abs=1e-9)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.tuples(st.integers(-50, 50), st.integers(-50, 50)), min_size=3, max_size=40))
def test_pearson_and_spearman_match_scipy(pairs):
    scipy_stats = pytest.importorskip("scipy.stats")
    x, y = [p for p, _ in pairs], [q for _, q in pairs]
    ours = pearson(x, y)
    if len(set(x)) < 2 or len(set(y)) < 2:
        assert ours is None
        return
    assert ours is not None
    assert ours.value == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-12)
    rank = spearman(x, y)
    assert rank is not None
    assert rank.value == pytest.approx(scipy_stats.spearmanr(x, y).statistic, abs=1e-12)


def test_average_ranks_share_ties():
    assert average_ranks([10, 20, 10, 30]) == [Fraction(3, 2), 3, Fraction(3, 2), 4]


def test_thresholds_are_decided_exactly():
    perfect = pearson([0, 1, 2], [0, 2, 4])
    assert perfect is not None and perfect.value == 1.0 and perfect.at_least(1.0)
    inverse = pearson([0, 1, 2], [4, 2, 0])
    assert inverse is not None and not inverse.at_least(0.0) and inverse.at_least(-1.0)
    # r = 6 / sqrt(172) by hand (n=4: Sxy=6, Sxx=4, Syy=43, all times n).
    partial = pearson([0, 0, 1, 1], [0, 1, 0, 4])
    assert partial is not None
    assert (partial.numerator, partial.squared_denominator) == (6, 4 * 43)
    assert math.isclose(partial.value, 6 / math.sqrt(172))
    assert partial.at_least(partial.value - 1e-12)
    assert not partial.at_least(partial.value + 1e-9)
    assert inverse < partial < perfect


def test_best_lag_finds_a_known_shift():
    signal = [0, 1, 4, 9, 3, 0, 2, 8, 5, 1, 0, 6, 2, 7, 3, 0]
    baseline = dict(enumerate(signal))
    candidate = {position + 2: value for position, value in baseline.items()}
    lag = best_lag(baseline, candidate, max_lag=4, min_pairs=5)
    assert lag is not None and lag.lag == 2 and lag.correlation.value == pytest.approx(1.0)
    assert lag.at_zero is not None and lag.at_zero < lag.correlation
    same = best_lag(baseline, dict(baseline), max_lag=4, min_pairs=5)
    assert same is not None and same.lag == 0
    # A cyclic profile wraps around: hour 23 is next to hour 0.
    profile = {h: (h * 7) % 24 for h in range(24)}
    wrapped = {(h - 3) % 24: v for h, v in profile.items()}
    cyclic = best_lag(profile, wrapped, max_lag=6, min_pairs=4, period=24)
    assert cyclic is not None and cyclic.lag == -3
