"""The distribution level: metrics per column and member against independent oracles,
categories that appear or disappear, families corrected across members, and every
decision policy failing closed."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from qavf import compare_datasets
from qavf.compare import DatasetComparison

pytest.importorskip("scipy")
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.stats.multitest import multipletests

GROUPS = ["a", "b", "c", "d"]


def _check(result, name):
    return next(c for c in result.level("distribution").checks if c.check == name)


def _frames(shift: float = 2.0, n: int = 200, seed: int = 8):
    """Fresh draws per side; only group ``a`` moves (by ``shift`` standard deviations)."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(GROUPS, n)
    base = pl.DataFrame({"g": groups, "x": rng.normal(0, 1, len(groups))})
    moved = rng.normal(0, 1, len(groups)) + np.where(groups == "a", shift, 0.0)
    return base, pl.DataFrame({"g": groups, "x": moved})


def _level(**options):
    return [{"level": "distribution", "columns": ["x"], "group_by": ["g"], **options}]


def test_metrics_per_member_match_scipy_and_numpy():
    base, cand = _frames()
    result = compare_datasets(base, cand, levels=_level())
    members = {m.group["g"]: m for m in result.level("distribution").members}
    for group in GROUPS:
        b = base.filter(pl.col("g") == group)["x"].to_numpy()
        c = cand.filter(pl.col("g") == group)["x"].to_numpy()
        member = members[group]
        assert member.ks == pytest.approx(ks_2samp(b, c).statistic, abs=1e-12)
        assert member.wasserstein == pytest.approx(wasserstein_distance(b, c), rel=1e-9)
        shares_b = np.array([s.baseline for s in member.bins])
        shares_c = np.array([s.candidate for s in member.bins])
        assert member.jensen_shannon == pytest.approx(
            jensenshannon(shares_b, shares_c, base=2) ** 2, abs=1e-12
        )
        for quantile in member.quantiles:
            assert quantile.baseline == np.quantile(b, quantile.p)
            assert quantile.candidate == np.quantile(c, quantile.p)
        assert (member.baseline_n, member.candidate_n) == (200, 200)


def test_a_planted_shift_fails_one_group_and_the_family_is_corrected():
    base, cand = _frames()
    result = compare_datasets(base, cand, levels=_level())
    check = _check(result, "x")
    assert not check.passed and check.assessed
    assert [r["g"] for r in check.affected_records] == ["a"]
    significance = check.significance
    assert significance.method == "ks" and significance.family_size == 4
    raw = [t.p_value for t in significance.tests]
    adjusted = [t.adjusted_p for t in significance.tests]
    # Members that needed no test count as p = 1 (immaterial under the default policy).
    padded = raw + [1.0] * (significance.family_size - len(raw))
    expected = multipletests(padded, method="fdr_bh")[1][: len(raw)]
    np.testing.assert_allclose(adjusted, expected, rtol=1e-12)
    [column] = result.level("distribution").columns
    assert column.compared == 4 and column.material >= 1


def test_noise_alone_is_not_a_distribution_change():
    base, cand = _frames(shift=0.0)
    assert _check(compare_datasets(base, cand, levels=_level()), "x").passed


@pytest.mark.parametrize(
    ("decision", "passed", "assessed"),
    [
        ("material_and_significant", False, True),
        ("significant", False, True),
        ("material", False, True),
        ("equivalence", False, False),
    ],
)
def test_every_decision_policy_decides_or_fails_closed(decision, passed, assessed):
    base, cand = _frames()
    check = _check(compare_datasets(base, cand, levels=_level(), decision=decision), "x")
    assert (check.passed, check.assessed) == (passed, assessed)
    if decision == "material":
        assert check.significance is None
    if decision == "equivalence":
        assert "no equivalence form" in check.message


def test_anderson_darling_can_decide_and_thresholds_are_per_metric():
    base, cand = _frames()
    result = compare_datasets(base, cand, levels=_level(test="anderson_darling"))
    assert _check(result, "x").significance.method == "anderson_darling"
    members = result.level("distribution").members
    assert all(m.anderson_darling is None or m.anderson_darling >= 0 for m in members)
    # Thresholds off and quantiles off: nothing is material, nothing fails.
    quiet = _level(thresholds={"max_psi": None, "max_ks": None, "max_js": None}, quantiles=None)
    assert _check(compare_datasets(base, cand, levels=quiet), "x").passed


def test_categories_that_appear_or_disappear_are_reported():
    base = pl.DataFrame({"g": ["a"] * 6 + ["b"] * 6, "fc": list("ppqqrr") * 2})
    cand = pl.DataFrame({"g": ["a"] * 6 + ["b"] * 6, "fc": list("ppqqss") + list("ppqqrr")})
    level = [{"level": "distribution", "columns": ["fc"], "group_by": ["g"], "min_values": 2}]
    result = compare_datasets(base, cand, levels=level, decision="material")
    categories = _check(result, "fc: categories")
    assert not categories.passed and categories.affected_count == 2
    assert {(r["g"], r["fc"], r["change"]) for r in categories.affected_records} == {
        ("a", "r", "disappeared"), ("a", "s", "appeared")
    }  # fmt: skip
    members = {m.group["g"]: m for m in result.level("distribution").members}
    assert members["a"].appeared == ["s"] and members["a"].disappeared == ["r"]
    assert members["b"].psi == 0.0 and members["b"].jensen_shannon == 0.0
    # PSI over categories by hand: shares 1/3 each; r: 1/3 -> 1e-6, s: 1e-6 -> 1/3.
    third = 1 / 3
    expected = 2 * (third - 1e-6) * np.log(third / 1e-6)
    assert members["a"].psi == pytest.approx(expected, rel=1e-12)
    assert not _check(result, "fc").passed
    allowed = [{**level[0], "new_categories": "ignore"}]
    again = compare_datasets(base, cand, levels=allowed, decision="material")
    assert _check(again, "fc: categories").passed


def test_chi_square_tests_categorical_members():
    rng = np.random.default_rng(3)
    base = pl.DataFrame({"fc": rng.choice(["x", "y", "z"], 600, p=[0.5, 0.3, 0.2])})
    cand = pl.DataFrame({"fc": rng.choice(["x", "y", "z"], 600, p=[0.3, 0.3, 0.4])})
    level = [{"level": "distribution", "columns": ["fc"]}]
    check = _check(compare_datasets(base, cand, levels=level), "fc")
    assert not check.passed and check.significance.method == "chi_square"
    assert check.significance.tests[0].p_value < 1e-6


def test_time_buckets_and_zones_are_members():
    days = [date(2024, 1, 1) + timedelta(days=i % 60) for i in range(600)]
    rng = np.random.default_rng(4)
    base = pl.DataFrame({"zone": ["n", "s"] * 300, "day": days, "x": rng.normal(0, 1, 600)})
    cand = base.with_columns(x=pl.Series(rng.normal(0, 1, 600)))
    level = [{"level": "distribution", "columns": ["x"], "group_by": ["zone"],
              "time": {"time_column": "day", "bucket": "month"}}]  # fmt: skip
    result = compare_datasets(base, cand, levels=level)
    distribution = result.level("distribution")
    assert distribution.bucket == "month" and len(distribution.members) == 4
    assert {tuple(m.group) for m in distribution.members} == {("zone", "bucket")}
    assert _check(result, "buckets").passed


def test_too_few_values_and_bad_columns_are_not_assessed():
    base = pl.DataFrame({"g": ["a", "b", "b", "b"], "x": [1.0, 1.0, 2.0, 3.0]})
    level = _level(min_values=2)
    check = _check(compare_datasets(base, base, levels=level), "x")
    assert not check.assessed and "min_values" in check.message
    missing = compare_datasets(base, base, levels=[{"level": "distribution", "columns": ["zz"]}])
    assert not _check(missing, "zz").assessed
    numeric = [{"level": "distribution", "columns": [{"column": "g", "kind": "numeric"}]}]
    assert not _check(compare_datasets(base, base, levels=numeric), "g").assessed
    budget = compare_datasets(base, base, levels=_level(max_work=3))
    assert not _check(budget, "x").assessed and "max_work" in _check(budget, "x").message


def test_identical_sides_pass_and_round_trip():
    base, _ = _frames()
    result = compare_datasets(base, base, levels=_level())
    assert result.passed, result.summary()
    assert DatasetComparison.model_validate_json(result.model_dump_json()) == result
