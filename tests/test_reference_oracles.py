"""Statistics checked against exact or published reference values."""

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats
from scipy.sparse import coo_matrix

from qavf.inference.counts import two_proportion
from qavf.inference.effects import cramers_v, quantile_deltas
from qavf.inference.intervals import bca_interval, percentile_interval
from qavf.inference.spatial import local_morans_i, morans_i, morans_statistic
from qavf.inference.trend import kendall_tau, mann_kendall, seasonal_kendall, slope_difference


@pytest.mark.parametrize("complement", [False, True])
def test_proportion_rare_outcomes_do_not_round_to_no_difference(complement):
    n = 2**60
    counts = (100, 0) if complement else (n - 100, n)
    result = two_proportion(counts, (n, n))
    pb, pc = (Fraction(count, n) for count in counts)
    pooled = (pb + pc) / 2
    variance = pooled * (1 - pooled) * Fraction(2, n)
    expected_z = float(pc - pb) / math.sqrt(float(variance))
    assert result.assessed
    assert result.effect == float(pc - pb)
    assert result.statistic == pytest.approx(expected_z, rel=2e-15)
    assert result.p_value == pytest.approx(2 * stats.norm.sf(abs(expected_z)), rel=1e-13)
    assert result.p_value < 1e-20


def test_proportions_keep_exact_all_equal_and_ordinary_reference_behavior():
    assert two_proportion((10, 10), (10, 10)).p_value == 1
    assert two_proportion((0, 0), (10, 10)).p_value == 1
    result = two_proportion((35, 56), (100, 120))
    pooled = Fraction(91, 220)
    difference = Fraction(56, 120) - Fraction(35, 100)
    variance = pooled * (1 - pooled) * (Fraction(1, 100) + Fraction(1, 120))
    assert result.statistic == pytest.approx(float(difference) / math.sqrt(float(variance)))


def test_direct_moran_row_normalization_matches_exact_normalized_graph():
    values = [0.0, 1.0, 3.0, 4.0]
    raw = {0: {1: 1e308, 2: 1e308}, 1: {0: 1.0}, 2: {3: 1.0}, 3: {2: 1.0}}
    normalized = {0: {1: 0.5, 2: 0.5}, 1: {0: 1.0}, 2: {3: 1.0}, 3: {2: 1.0}}
    actual = morans_i(values, raw, row_standardize=True, permutations=29)
    expected = morans_i(values, normalized, permutations=29)
    assert actual.statistic == expected.statistic == pytest.approx(0.6)
    assert actual.p_value == expected.p_value
    actual_local = local_morans_i(values, raw, permutations=29)
    expected_local = local_morans_i(values, normalized, permutations=29)
    assert actual_local.quadrants == expected_local.quadrants
    assert [(r.statistic, r.p_value) for r in actual_local.results] == [
        (r.statistic, r.p_value) for r in expected_local.results
    ]


@pytest.mark.parametrize("weight", [1e308, math.ulp(0.0)])
def test_global_moran_is_invariant_to_extreme_common_weight_units(weight):
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    values = [1.0, 2.0, 5.0]
    assert morans_statistic(values, matrix * weight) == pytest.approx(
        morans_statistic(values, matrix)
    )


def test_duplicate_sparse_neighbors_are_one_location_in_conditional_draws():
    repeated = coo_matrix((np.ones(4), ([0] * 4, [1] * 4)), shape=(3, 3))
    actual = local_morans_i([1.0, 2.0, 4.0], repeated, permutations=29)
    expected = local_morans_i([1.0, 2.0, 4.0], {0: {1: 4.0}}, permutations=29)
    assert actual.results[0].n["neighbours"] == 1
    assert actual.results[0].statistic == expected.results[0].statistic
    assert actual.results[0].p_value == expected.results[0].p_value
    assert not actual.results[1].assessed


def test_moran_rejects_fractional_neighbor_identity_and_unsafe_local_range():
    with pytest.raises(ValueError, match="indices must be integers"):
        morans_statistic([1.0, 2.0, 4.0], {0: {1.9: 1.0}})
    with pytest.raises(ValueError, match="indices must lie"):
        morans_statistic([1.0, 2.0, 4.0], {0: {2**100: 1.0}})
    result = local_morans_i(
        [0.0, 1.0, 3.0, 4.0], {0: {1: 1e308, 2: 1e308}},
        row_standardize=False, permutations=29,
    )
    assert all(not test.assessed for test in result.results)
    assert all("finite float range" in test.reason for test in result.results)


def test_extreme_linear_quantiles_and_intervals_stay_between_finite_endpoints():
    values = np.array([-1e308, 1e308])
    probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
    actual = quantile_deltas(values, values, probabilities)
    for probability, before, after in actual:
        exact = Fraction(float(values[0])) + Fraction(probability) * (
            Fraction(float(values[1])) - Fraction(float(values[0]))
        )
        assert before == after == float(exact)
    low, high = percentile_interval(values, 0.5)
    assert (low, high) == (-5e307, 5e307)
    low, high = bca_interval(0.0, values, [np.zeros(3)], confidence=0.5)
    assert (low, high) == pytest.approx((-5e307, 5e307))


@pytest.mark.parametrize("large, small", [(1e308, 1e308), (1e308, 1e-308), (1.0, 1.0)])
def test_cramers_v_perfect_association_survives_count_range(large, small):
    assert cramers_v([[large, 0], [0, small]]) == pytest.approx(1.0)


def test_cramers_v_preserves_small_association_and_rejects_nonfinite_counts():
    table = np.array([[10.00000001, 10.0], [10.0, 10.00000001]])
    chi2 = stats.chi2_contingency(table, correction=False).statistic
    assert cramers_v(table) == pytest.approx(math.sqrt(chi2 / table.sum()), rel=1e-7, abs=0)
    with pytest.raises(ValueError, match="non-negative counts"):
        cramers_v([[1.0, math.nan], [0.0, 1.0]])


def test_cramers_v_preserves_representable_effect_with_subnormal_squared_effect():
    smallest = math.ulp(0.0)
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float(smallest)
        # 2x2 phi = (a*d-b*c) / sqrt((a+b)*(c+d)*(a+c)*(b+d)).
        expected = float(d / (2 * d * (1 + d)).sqrt())
    assert cramers_v([[1.0, 1.0], [0.0, smallest]]) == pytest.approx(expected, rel=1e-15, abs=0)
    assert expected > 0


@pytest.mark.parametrize("table", [
    [[1.0, 1e-300], [1e-300, 0.0]],
    [[1e-100, 0.0], [1.0, math.ulp(0.0)]],
    [[1e-100, 1e-200], [1e100, 1.0]],
])
def test_cramers_v_tiny_association_matches_exact_decimal_oracle(table):
    (a, b), (c, d) = [[Fraction(value) for value in row] for row in table]
    square = (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    with localcontext() as context:
        context.prec = 100
        expected = float((Decimal(square.numerator) / Decimal(square.denominator)).sqrt())
    actual = cramers_v(table)
    assert actual == pytest.approx(expected, rel=2e-15, abs=0)
    assert actual > 0


def test_monotone_slopes_preserve_signed_float_extremes():
    values = [-1e308, 0.0, 1e308]
    seasonal = seasonal_kendall(values, [0, 0, 0], positions=values)
    ordinary = mann_kendall(values, positions=values)
    for result in (seasonal, ordinary):
        assert result.assessed
        assert result.effect == 1.0
        assert result.statistic == 3.0
    assert ordinary.ci_low == ordinary.ci_high == 1.0


def test_zero_slope_does_not_multiply_by_an_overflowing_unit_ratio():
    # The y/x unit ratio overflows, but the exactly constant difference has slope0.
    result = slope_difference(
        [1e-308, 2e-308, 3e-308, 4e-308],
        [1e308, 1.1e308, 1.2e308, 1.3e308],
        [1e308, 1.1e308, 1.2e308, 1.3e308], dependence="none",
    )
    assert result.assessed
    assert result.effect == result.ci_low == result.ci_high == 0.0


@pytest.mark.parametrize("x, y", [
    ([[0, 1], [2, 3]], [[0, 1], [2, 3]]),
    ([0, 1], [0, 1, 2]),
    ([0, 1, 2], [0, 1, math.inf]),
    (np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]"), [0, 1]),
])
def test_kendall_admits_only_paired_finite_one_dimensional_ranks(x, y):
    assert not kendall_tau(x, y).assessed


def test_kendall_retains_exact_large_integer_and_nanosecond_rank_order():
    integers = np.array([2**60, 2**60 + 1, 2**60 + 2], dtype=np.int64)
    times = np.array([1, 2, 3], dtype="datetime64[ns]")
    result = kendall_tau(integers, times)
    assert result.assessed and result.effect == 1.0
    assert result.n == {"pairs": 3}


@pytest.mark.parametrize("missing", [math.inf, math.nan, Decimal("Infinity"), Decimal("NaN")])
def test_kendall_object_numeric_missing_values_are_unassessed(missing):
    assert not kendall_tau(np.array([0, 1, missing], dtype=object), [0, 1, 2]).assessed
