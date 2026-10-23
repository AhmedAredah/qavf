"""Statistical inference admission checked against independent computations."""

import math
from datetime import UTC, datetime
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from scipy import stats

from qavf import DataContext
from qavf.inference import (
    Moments,
    bootstrap_difference,
    chi_square,
    decide,
    permutation,
    tost_from_moments,
    welch_t_from_moments,
)
from qavf.inference.agreement import fit_line, identity_test
from qavf.inference.effects import jensen_shannon, mape, odds_ratio, smape
from qavf.inference.independent import mean_statistic, resampled_difference
from qavf.inference.trend import slope_difference
from qavf.spatial.hierarchy import apply_crosswalk
from qavf.spatial.layers import ZoneLayer, load_zones
from qavf.spatial.weights import from_table
from qavf.temporal import Bucket, TimeBucketing, bucket_grid, bucket_labels, sequence_positions


@pytest.mark.parametrize(
    "median,candidate",
    [
        (math.ldexp(1.0, -1074), math.ldexp(100.0, -1074)),
        (-math.ldexp(1.0, -1074), -math.ldexp(100.0, -1074)),
        (math.ldexp(1.0, -1074), 1e308),
        (math.ldexp(1.0, -1074), -1e308),
        (2e-308, 3.0),
        (-2e-308, -3.0),
        (math.ldexp(1.0, -1022), 1e-307),
        (1e-300, 1.5e-300),
        (1e308, 1.7e308),
        (1e308, math.ldexp(1.0, -1074)),
    ],
)
def test_joined_group_median_ratios_match_fraction_oracle(median, candidate):
    from qavf.rules.spatial_checks import GroupComparisonRule

    values = [median] * 7 + [candidate]
    exact = Fraction(candidate) / Fraction(median)
    expected_bad = exact < Fraction(1, 2) or exact > 2
    frame = pl.DataFrame({"id": range(8), "v": values})
    with DataContext(frames={"d": frame}) as ctx:
        [finding] = GroupComparisonRule(
            {
                "max_ratio": 2,
                "metrics": [
                    {"dataset": "d", "entity_column": "id", "value": "v", "aggregate": "none"}
                ],
            }
        ).evaluate(ctx)
    assert finding.assessed
    assert finding.passed is (not expected_bad)
    assert finding.affected_count == (1 if expected_bad else None)
    if expected_bad:
        assert finding.affected[0]["id"] == 7
        try:
            expected_ratio = float(exact)
        except OverflowError:
            expected_ratio = -math.inf if exact < 0 else math.inf
        assert finding.affected[0]["ratio_to_group"] == pytest.approx(expected_ratio)


@pytest.mark.parametrize(
    "table",
    [
        [[float("inf"), 5], [5, 5]],
        np.array([[2**63, 10], [20, 10]], dtype=np.uint64),
        np.array([[2**62, 10], [2**62, 100]], dtype=np.int64),
        np.empty((0, 2)),
    ],
)
def test_unsafe_contingency_counts_never_pass(table):
    result = chi_square(table)
    assert not result.assessed
    assert decide("material_and_significant", True, result).outcome == "not_assessed"


def test_mean_permutation_matches_integer_assignment_oracle_after_translation():
    b = 2 * np.arange(12, dtype=np.int64)
    c = b + 8
    joined = np.concatenate([b, c])
    draws = np.random.Generator(np.random.PCG64(0)).permuted(
        np.broadcast_to(joined, (2000, len(joined))), axis=1
    )
    numerators = draws[:, 12:].sum(axis=1) - draws[:, :12].sum(axis=1)
    expected = min(
        1, 2 * min(((numerators >= 96).sum() + 1) / 2001, ((numerators <= 96).sum() + 1) / 2001)
    )
    for offset in (0, 1e16):
        result = permutation(b + offset, c + offset, resamples=2000)
        assert result.p_value == pytest.approx(expected)
        assert result.effect == 8


def test_default_mean_bootstrap_is_translation_invariant():
    b = 2 * np.arange(12, dtype=float)
    c = b + 8
    for interval in ("percentile", "bca"):
        plain = bootstrap_difference(b, c, interval=interval)
        offset = bootstrap_difference(b + 1e16, c + 1e16, interval=interval)
        assert (offset.p_value, offset.ci_low, offset.ci_high) == pytest.approx(
            (plain.p_value, plain.ci_low, plain.ci_high)
        )


def test_unequal_sample_bca_matches_scipy_with_the_identical_draws():
    b = np.array([0.0, 1.0, 2.0, 3.0])
    c = np.concatenate([np.linspace(0, 3, 27), [9.0, 20.0, 60.0]]) - 0.2
    _, distribution = resampled_difference(b, c, mean_statistic, 2000, 0, 20_000_000)
    reference = stats.bootstrap(
        (b, c),
        lambda b, c, axis=-1: c.mean(axis=axis) - b.mean(axis=axis),
        vectorized=True,
        n_resamples=0,
        method="BCa",
        bootstrap_result=SimpleNamespace(bootstrap_distribution=distribution),
    )
    result = bootstrap_difference(b, c, interval="bca")
    assert (result.ci_low, result.ci_high) == pytest.approx(reference.confidence_interval)
    assert result.ci_low > 0


@pytest.mark.parametrize("variance", [1e200, 1e-200, 1.0])
def test_summary_welch_and_tost_use_scale_free_reference(variance):
    b, c = Moments(10, 0, variance), Moments(10, np.sqrt(variance), variance)
    expected_t = np.sqrt(5.0)
    result = welch_t_from_moments(b, c)
    assert result.statistic == pytest.approx(expected_t)
    assert result.p_value == pytest.approx(2 * stats.t.sf(expected_t, 18))
    result = tost_from_moments(b, c, 2 * np.sqrt(variance))
    assert result.p_value == pytest.approx(stats.t.sf(expected_t, 18))


def test_shifted_ols_matches_exact_rational_regression():
    b = 1e16 + np.arange(4.0) * 2
    c = np.arange(4.0)
    exact_b, exact_c = list(map(Fraction, b)), list(map(Fraction, c))
    mb, mc = sum(exact_b) / 4, sum(exact_c) / 4
    slope = sum((x - mb) * (y - mc) for x, y in zip(exact_b, exact_c, strict=True)) / sum(
        (x - mb) ** 2 for x in exact_b
    )
    result = fit_line(b, c)
    assert result.slope == float(slope)
    assert result.intercept == float(mc - slope * mb)


def test_identity_cannot_pass_a_constant_discrepancy_after_translation():
    for offset in (0.0, 1e16):
        b = offset + np.array([0.0, 2.0, 4.0])
        c = offset + np.array([2.0, 4.0, 6.0])
        result = identity_test(b, c)
        assert result.assessed and result.p_value == 0
        assert decide("material_and_significant", True, result).failed


@pytest.mark.parametrize("method", ["ols", "deming"])
def test_intercept_survives_overflowing_intermediate_product(method):
    result = fit_line([9e307, 9.5e307, 1e308], [3e307, 4e307, 5e307], method=method)
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(-1.5e308)


@pytest.mark.parametrize("ratio", [0.1, 1.0, 10.0, 1e200])
def test_deming_recovers_a_small_nonzero_exact_line(ratio):
    b = np.arange(10.0)
    result = fit_line(b, b * 1e-10, method="deming", error_ratio=ratio)
    assert result.slope == pytest.approx(1e-10, rel=1e-12, abs=0)
    assert result.intercept == pytest.approx(0, abs=1e-24)


def test_hac_and_block_follow_positions_not_input_row_order():
    rng = np.random.default_rng(2)
    x = np.arange(80.0)
    noise = rng.normal(size=80)
    for j in range(1, 80):
        noise[j] += 0.8 * noise[j - 1]
    y = noise + 0.015 * x
    order = np.random.default_rng(42).permutation(80)
    for dependence in ("hac", "block"):
        a = slope_difference(x, np.zeros(80), y, dependence=dependence)
        b = slope_difference(x[order], np.zeros(80), y[order], dependence=dependence)
        assert (a.statistic, a.p_value, a.ci_low, a.ci_high) == pytest.approx(
            (b.statistic, b.p_value, b.ci_low, b.ci_high)
        )
    # Independent OLS / Bartlett sandwich, not a second call to QAVF.
    X = np.column_stack([np.ones(80), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    scores = X * (y - X @ beta)[:, None]
    meat = scores.T @ scores
    for lag in range(1, 4):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / 4) * (cross + cross.T)
    bread = np.linalg.inv(X.T @ X)
    covariance = bread @ meat @ bread * 80 / 78
    p = 2 * stats.t.sf(abs(beta[1] / np.sqrt(covariance[1, 1])), 78)
    assert slope_difference(x, np.zeros(80), y, dependence="hac").p_value == pytest.approx(p)


@pytest.mark.parametrize(
    "zone,every,instants",
    [
        (
            "America/New_York",
            2,
            [datetime(2024, 3, 10, 7, 15, tzinfo=UTC), datetime(2024, 3, 10, 8, 15, tzinfo=UTC)],
        ),
        (
            "Australia/Lord_Howe",
            1,
            [datetime(2024, 10, 5, 15, 45, tzinfo=UTC), datetime(2024, 10, 5, 16, 15, tzinfo=UTC)],
        ),
    ],
)
def test_gap_adjusted_first_bucket_preserves_the_nominal_phase(zone, every, instants):
    series = pl.Series("t", instants).dt.convert_time_zone(zone)
    bucket = Bucket(unit="hour", every=every)
    expression, _ = bucket_labels("t", series.dtype, TimeBucketing(bucket=bucket))
    labels = series.to_frame().select(expression.alias("label"))["label"]
    grid = bucket_grid(labels.head(1), labels.tail(1), bucket, limit=2)
    assert grid.equals(labels)
    assert sequence_positions(labels, bucket).to_list() == [0, 1]


def test_temporal_limits_include_cyclic_and_repeated_physical_buckets():
    with pytest.raises(ValueError, match="limit"):
        bucket_grid(pl.Series([0]), pl.Series([23]), Bucket(unit="hour_of_day"), limit=1)
    series = pl.Series(
        [datetime(2024, 11, 3, 4, tzinfo=UTC), datetime(2024, 11, 3, 8, tzinfo=UTC)]
    ).dt.convert_time_zone("America/New_York")
    with pytest.raises(ValueError, match="limit"):
        bucket_grid(series.head(1), series.tail(1), Bucket(unit="hour"), limit=4)


def test_crosswalk_never_selects_or_drops_a_user_target_column():
    frame = pl.DataFrame({"zone": ["a"], "v": [7], "tmp_to": ["original"]})
    cw = pl.DataFrame({"from": ["a"], "to": ["P"], "weight": [1.0]})
    output = apply_crosswalk(frame.lazy(), "zone", cw, [], taken="tmp").collect()
    assert output.to_dict(as_series=False) == {"zone": ["P"], "v": [7], "tmp_to": ["original"]}


def test_effect_ratios_avoid_overflow_when_the_answer_is_finite():
    assert mape([-1e308], [1e308]) == 200
    assert mape(np.ones(1000), np.full(1000, 1e306)) == pytest.approx(1e308)
    assert smape([-1e308], [1e308]) == 200
    assert jensen_shannon([1e308, 1e308, 0], [0, 0, 1e308]) == 1
    assert odds_ratio([[1e200, 1e200], [1e200, 2e200]]) == 2


def test_weight_table_rejects_null_weights_with_its_documented_error():
    table = pl.DataFrame(
        {"from": ["a"], "to": ["b"], "weight": [None]}, schema_overrides={"weight": pl.Float64}
    )
    with pytest.raises(ValueError, match="finite and positive"):
        from_table(pl.Series(["a", "b"]), table)


def test_repair_cannot_admit_an_empty_polygon():
    import duckdb

    con = duckdb.connect()
    try:
        try:
            con.execute("LOAD spatial")
        except duckdb.Error:
            pytest.skip("installed spatial extension unavailable; no network installation")

        class Context:
            def project(self, *args):
                return pl.DataFrame({"id": ["flat"], "geometry": ["POLYGON ((0 0,1 0,2 0,0 0))"]})

            def resolve_path(self, path):
                return path

            def memo(self, key, fn):
                return fn()

        with pytest.raises(ValueError, match=r"repair left.*nonempty valid polygon"):
            load_zones(con, Context(), ZoneLayer(dataset="x", id="id", repair=True))
    finally:
        con.close()
