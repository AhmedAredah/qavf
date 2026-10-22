"""The relationships level and the identity fit: hand-computed agreement metrics,
constructed fits with known answers, and a planted correlation change."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf import compare_datasets
from qavf.compare import DatasetComparison
from qavf.inference.agreement import fit_line, identity_test

pytest.importorskip("scipy")
import statsmodels.api as sm


def _check(result, name):
    return next(c for c in result.level("relationships").checks if c.check == name)


# ------------------------------------------------------------------ agreement
BASE = [1.0, 2.0, 3.0, 4.0]
CAND = [1.5, 2.5, 2.5, 5.0]
"""Differences 0.5, 0.5, -0.5, 1: worked by hand below."""


def test_agreement_metrics_equal_hand_computed_values():
    base = pl.DataFrame({"k": [1, 2, 3, 4], "x": BASE})
    cand = pl.DataFrame({"k": [1, 2, 3, 4], "x": CAND})
    level = {"level": "relationships", "agreement": {"columns": ["x"], "min_ccc": None,
                                                      "max_slope_deviation": None}}  # fmt: skip
    result = compare_datasets(base, cand, keys=["k"], levels=[level], decision="material")
    [agreement] = result.level("relationships").agreement
    sd = math.sqrt(1.1875 / 3)  # squared deviations from the mean difference 0.375
    assert agreement.pairs == 4 and result.level("relationships").matched == 4
    assert agreement.bias == pytest.approx(0.375)
    assert agreement.sd == pytest.approx(sd)
    assert (agreement.lower, agreement.upper) == pytest.approx(
        (0.375 - 1.96 * sd, 0.375 + 1.96 * sd)
    )
    assert agreement.mae == pytest.approx(0.625)
    assert agreement.rmse == pytest.approx(math.sqrt(0.4375))
    assert agreement.mape == pytest.approx((0.5 + 0.25 + 0.5 / 3 + 0.25) / 4 * 100)
    assert agreement.smape == pytest.approx((0.4 + 1 / 4.5 + 1 / 5.5 + 2 / 9) / 4 * 100)
    # Lin: 2 * 1.3125 / (1.25 + 1.671875 + 0.375 ** 2) with 1/n moments.
    assert agreement.ccc == pytest.approx(2.625 / 3.0625)
    assert agreement.r_squared == pytest.approx(1 - 1.75 / 5)
    assert agreement.material == [] and _check(result, "agreement: x").passed
    strict = {**level, "agreement": {"columns": ["x"], "max_abs_bias": 0.1}}
    failed = compare_datasets(base, cand, keys=["k"], levels=[strict], decision="material")
    assert not _check(failed, "agreement: x").passed
    assert failed.level("relationships").agreement[0].material == ["bias", "slope", "ccc"]


def test_the_identity_fit_has_known_answers_on_constructed_data():
    b = np.arange(1.0, 9.0)
    e = 0.1 * np.array([1, -1, -1, 1, 1, -1, -1, 1])  # orthogonal to 1 and to b
    on_line = identity_test(b, b + e)
    assert on_line.effect == pytest.approx(1.0) and on_line.p_value == pytest.approx(1.0)
    assert "intercept" in on_line.notes[0]
    steeper = identity_test(b, 1.2 * b + e)
    assert steeper.effect == pytest.approx(1.2)
    model = sm.OLS(1.2 * b + e, sm.add_constant(b)).fit()
    oracle = model.f_test("const = 0, x1 = 1")
    assert steeper.statistic == pytest.approx(float(oracle.fvalue), rel=1e-9)
    assert steeper.p_value == pytest.approx(float(oracle.pvalue), rel=1e-9)
    assert identity_test(b, b).p_value == 1.0  # a perfect fit on the line
    assert identity_test(b, b + 1).p_value == 0.0  # a perfect fit off it
    fit = fit_line(b, 3 + 2 * b)
    assert (fit.slope, fit.intercept) == pytest.approx((2.0, 3.0))
    assert not identity_test(b[:2], b[:2]).assessed


def test_deming_regression_is_symmetric_and_tested_by_jackknife():
    rng = np.random.default_rng(2)
    truth = rng.uniform(0, 100, 200)
    b, c = truth + rng.normal(0, 2, 200), truth + rng.normal(0, 2, 200)
    deming = fit_line(b, c, method="deming")
    reverse = fit_line(c, b, method="deming")
    assert deming.slope == pytest.approx(1 / reverse.slope, rel=1e-9)
    ols = fit_line(b, c)
    assert ols.slope < deming.slope  # OLS is attenuated by the baseline's own error
    assert identity_test(b, c, method="deming").p_value > 0.05
    biased = identity_test(b, 1.1 * c, method="deming")
    assert biased.p_value < 1e-6 and biased.effect > deming.slope


# --------------------------------------------------------------- correlation
def test_a_planted_correlation_change_is_detected_and_an_unchanged_pair_is_not():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.normal(0, 1, n)
    base = pl.DataFrame({"x": x, "y": x + rng.normal(0, 0.4, n), "z": rng.normal(0, 1, n)})
    cand = base.with_columns(y=pl.Series(rng.normal(0, 1, n)))  # y no longer follows x
    level = {"level": "relationships", "correlation": {"columns": ["x", "y", "z"]}}
    result = compare_datasets(base, cand, levels=[level])
    for method in ("pearson", "spearman"):
        check = _check(result, f"correlation: {method}")
        assert not check.passed and check.affected_count == 1
        assert check.affected_records[0]["columns"] == "x, y"
        assert check.significance.family_size == 3
    changes = {(c.method, tuple(c.columns)): c for c in result.level("relationships").correlations}
    planted = changes[("pearson", ("x", "y"))]
    assert planted.baseline == pytest.approx(np.corrcoef(base["x"], base["y"])[0, 1], abs=1e-12)
    assert planted.test is not None and planted.test.method == "fisher_z"
    assert not changes[("pearson", ("x", "z"))].material


# --------------------------------------------------------------------- ratios
def test_ratio_invariants_per_group_with_poisson_tests():
    base = pl.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "fatal": [1, 2, 1, 2, 3, 1, 2, 2],
                         "crashes": [100] * 8})  # fmt: skip
    cand = base.with_columns(
        fatal=pl.when(pl.col("g") == "a").then(pl.col("fatal") * 20).otherwise(pl.col("fatal"))
    )
    level = {"level": "relationships", "group_by": ["g"], "ratios": ["fatal / crashes"]}
    result = compare_datasets(base, cand, levels=[level])
    check = _check(result, "ratio: fatal / crashes")
    assert not check.passed and [r["g"] for r in check.affected_records] == ["a"]
    assert check.significance.method == "poisson"
    ratios = {r.group["g"]: r for r in result.level("relationships").ratios}
    assert ratios["a"].baseline == 6 / 400 and ratios["a"].candidate == 120 / 400
    assert ratios["b"].change == 0
    floats = base.with_columns(pl.col("fatal").cast(pl.Float64) * 0.3)
    judged = compare_datasets(floats, floats, levels=[level])
    assert _check(judged, "ratio: fatal / crashes").significance.applicable is False


def test_configuration_and_unmeasurable_members_fail_closed():
    with pytest.raises(ValidationError, match="needs the comparison's 'keys'"):
        agreement = [{"level": "relationships", "agreement": {"columns": ["x"]}}]
        compare_datasets(pl.DataFrame({"x": [1.0]}), pl.DataFrame({"x": [1.0]}), levels=agreement)
    with pytest.raises(ValidationError, match="configure 'correlation'"):
        compare_datasets(pl.DataFrame({"x": [1.0]}), pl.DataFrame({"x": [1.0]}),
                         levels=["relationships"])  # fmt: skip
    flat = pl.DataFrame({"x": [1.0] * 10, "y": np.arange(10.0)})
    level = {"level": "relationships", "correlation": {"columns": ["x", "y"]}}
    result = compare_datasets(flat, flat, levels=[level])
    check = _check(result, "correlation: pearson")
    assert not check.assessed and "constant" in check.message
    zero = pl.DataFrame({"n": [1, 2], "d": [0, 0]})
    ratio = compare_datasets(zero, zero, levels=[{"level": "relationships", "ratios": ["n / d"]}])
    assert not _check(ratio, "ratio: n / d").assessed
    dup = pl.DataFrame({"k": [1, 1, 2], "x": [1.0, 2.0, 3.0]})
    agreement = {"level": "relationships", "agreement": {"columns": ["x"]}}
    repeated = compare_datasets(dup, dup, keys=["k"], levels=[agreement])
    assert not _check(repeated, "agreement").assessed


def test_every_policy_and_a_json_round_trip():
    rng = np.random.default_rng(5)
    base = pl.DataFrame({"k": range(50), "x": rng.normal(10, 1, 50)})
    cand = base.with_columns(x=pl.col("x") * 1.5)
    level = {"level": "relationships", "agreement": {"columns": ["x"]}}
    outcomes = {}
    for decision in ("material_and_significant", "material", "significant", "equivalence"):
        result = compare_datasets(base, cand, keys=["k"], levels=[level], decision=decision)
        check = _check(result, "agreement: x")
        outcomes[decision] = (check.passed, check.assessed)
        assert DatasetComparison.model_validate_json(result.model_dump_json()) == result
    assert outcomes == {
        "material_and_significant": (False, True),
        "material": (False, True),
        "significant": (False, True),
        "equivalence": (False, False),
    }
