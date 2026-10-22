"""TMP-002, TMP-003 and STA-002 on the shared inference module: unchanged by default.

Two kinds of evidence that the retrofit changed no default result:

* golden values recorded by running the rules on the commit before the
  retrofit (``1804f7c``, SciPy 1.18), compared to 1e-12 (SciPy versions differ
  in the last digits of the t tail and the KS statistic; the floor is 1.11);
* the pre-retrofit formulas, written out here with SciPy directly, compared
  exactly on further inputs (ties, short series, several comparisons).

The new ``correction`` options agree with statsmodels' ``multipletests``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qavf.engine.severity import Status
from qavf.rules.temporal_checks import _trend_outlier_test
from qavf.testing import rule_results, run_rule

stats = pytest.importorskip("scipy.stats")


def _series(values, years=None, sites=None):
    years = years or list(range(2010, 2010 + len(values)))
    return {"s": pl.DataFrame({"year": years, "v": values, "site": sites or ["a"] * len(values)})}


def _params(**extra):
    return {"series": [{"dataset": "s", "time_column": "year", "value": "v", **extra}]}


def _trend(seed: int = 3):
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2016))
    values = [1000.0 * 0.97 ** (y - 2000) * (1 + rng.normal(0, 0.002)) for y in years]
    values[6] *= 1.4
    values[11] *= 0.8
    return years, values, rng


def _drift_frames(rng):
    return {
        "ref": pl.DataFrame({"x": rng.normal(100, 10, 3000)}),
        "cur": pl.DataFrame({"x": rng.normal(103, 10, 2500)}),
        "sref": pl.DataFrame({"x": rng.normal(0, 1, 300)}),
        "scur": pl.DataFrame({"x": rng.normal(0.4, 1, 280)}),
    }


KS = [
    {"dataset": "cur", "reference": "ref", "column": "x", "method": "ks"},
    {"dataset": "scur", "reference": "sref", "column": "x", "method": "ks"},
]


# ------------------------------------------------------------ golden values
def test_golden_values_recorded_before_the_retrofit():
    falling = [10.0, 9.0, 8.5, 7.0, 6.0, 6.5, 5.0]
    result = run_rule(
        "TMP-002", _series(falling), {**_params(expected_trend="increasing"), "alpha": 0.05}
    )
    [record] = result.affected_records
    assert (record["kendall_tau"], record["p_value"]) == (-0.9048, 0.002778)
    years, values, rng = _trend()
    [record] = run_rule("TMP-003", _series(values, years), _params()).affected_records
    assert record["year"] == 2006
    assert record["adjusted_p"] == pytest.approx(0.0012957361236595206, rel=1e-12)
    results = rule_results("STA-002", _drift_frames(rng), {"comparisons": KS})
    observed = [(r.status, r.metrics["ks_statistic"], r.metrics["p_value"]) for r in results]
    expected = [
        (Status.FAIL, 0.10266666666666663, 5.828489753506503e-13),
        (Status.FAIL, 0.1702380952380953, 0.00038127226417131405),
    ]
    for (status, statistic, p), (want_status, want_statistic, want_p) in zip(
        observed, expected, strict=True
    ):
        assert status == want_status
        assert (statistic, p) == pytest.approx((want_statistic, want_p), rel=1e-12)


# ---------------------------------------------------- pre-retrofit formulas
@pytest.mark.parametrize(
    "values",
    [
        [10.0, 9.0, 8.5, 7.0, 6.0],
        [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0],  # ties
        [9.0, 7.0, 8.0, 6.0, 7.0, 5.0, 4.0, 4.5, 3.0, 2.0, 2.5, 1.0],
    ],
)
def test_tmp002_p_values_equal_scipy_kendalltau_exactly(values):
    years = list(range(2000, 2000 + len(values)))
    result = run_rule("TMP-002", _series(values, years), _params(expected_trend="increasing"))
    [record] = result.affected_records
    before = stats.kendalltau(np.arange(len(values)), np.array(values))
    assert record["kendall_tau"] == round(float(before.statistic), 4)
    assert record["p_value"] == round(float(before.pvalue), 6)
    assert record["adjusted_p"] == record["p_value"] and record["correction"] == "none"
    assert record["effect"] == record["kendall_tau"]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_tmp003_adjusted_p_values_equal_the_bonferroni_formula_exactly(seed):
    rng = np.random.default_rng(seed)
    times = np.arange(12, dtype=float)
    values = 50 + 2 * times + rng.normal(0, 1, 12)
    values[5] += 6
    test = _trend_outlier_test(times, values, False)
    assert test is not None
    before = np.minimum(1.0, 2 * stats.t.sf(np.abs(test.residuals), 12 - 3) * 12)
    np.testing.assert_array_equal(test.adjusted_p, before)
    np.testing.assert_array_equal(test.p_value, 2 * stats.t.sf(np.abs(test.residuals), 9))


def test_sta002_default_decisions_follow_the_bonferroni_divided_alpha():
    rng = np.random.default_rng(8)
    frames = _drift_frames(rng)
    results = rule_results("STA-002", frames, {"comparisons": KS})
    for result, comparison in zip(results, KS, strict=True):
        reference = frames[comparison["reference"]]["x"].to_numpy()
        current = frames[comparison["dataset"]]["x"].to_numpy()
        before = stats.ks_2samp(reference, current, method="auto")
        assert result.metrics["ks_statistic"] == float(before.statistic)
        assert result.metrics["p_value"] == float(before.pvalue)
        assert result.metrics["alpha_bonferroni"] == 0.01 / 2
        assert result.metrics["adjusted_p"] == min(1.0, float(before.pvalue) * 2)
        assert result.metrics["effect"] == result.metrics["ks_statistic"]
        drifted = before.pvalue < 0.01 / 2 and before.statistic >= 0.1
        assert (result.status is Status.FAIL) == drifted


# -------------------------------------------------------------- corrections
def test_tmp002_correction_across_series_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(5)
    sites, years, values = [], [], []
    for site in range(6):
        drift = -0.3 * site
        for year in range(2000, 2010):
            sites.append(f"s{site}")
            years.append(year)
            values.append(100 + drift * (year - 2000) + rng.normal(0, 1))
    frames = _series(values, years, sites)
    spec = _params(expected_trend="increasing", group_by=["site"])
    raw = run_rule("TMP-002", frames, {**spec, "min_tau": 0.01})
    corrected = run_rule("TMP-002", frames, {**spec, "min_tau": 0.01, "correction": "holm"})
    family = []
    for site in range(6):
        y = [v for s, v in zip(sites, values, strict=True) if s == f"s{site}"]
        family.append(stats.kendalltau(np.arange(10), y).pvalue)
    expected = multipletests(family, method="holm")[1]
    holm = {r["site"]: r["adjusted_p"] for r in corrected.affected_records}
    for site, p in holm.items():
        assert p == round(float(expected[int(site[1:])]), 6)
    assert {r["site"] for r in raw.affected_records} >= set(holm)
    strict = run_rule("TMP-002", frames, {**spec, "min_tau": 0.01, "alpha": 0.01})
    strict_holm = run_rule(
        "TMP-002", frames, {**spec, "min_tau": 0.01, "alpha": 0.01, "correction": "holm"}
    )
    assert len(strict_holm.affected_records) <= len(strict.affected_records)


@pytest.mark.parametrize("correction", ["holm", "bh", "by", "none"])
def test_tmp003_corrections_match_statsmodels(correction):
    from statsmodels.stats.multitest import multipletests

    years, values, _ = _trend()
    test = _trend_outlier_test(
        np.array(years, dtype=float), np.array(values), True, correction=correction
    )
    assert test is not None
    methods = {"holm": "holm", "bh": "fdr_bh", "by": "fdr_by"}
    expected = (
        test.p_value
        if correction == "none"
        else multipletests(test.p_value, method=methods[correction])[1]
    )
    np.testing.assert_allclose(test.adjusted_p, expected, rtol=1e-12)
    result = run_rule("TMP-003", _series(values, years), _params())
    assert result.affected_records[0]["correction"] == "bonferroni"
    assert "p_value" in result.affected_records[0] and "effect" in result.affected_records[0]


def test_sta002_benjamini_hochberg_option():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(9)
    frames = _drift_frames(rng)
    results = rule_results("STA-002", frames, {"comparisons": KS, "correction": "bh"})
    raw = [r.metrics["p_value"] for r in results]
    expected = multipletests(raw, method="fdr_bh")[1]
    for result, q in zip(results, expected, strict=True):
        assert result.metrics["adjusted_p"] == pytest.approx(q, rel=1e-12)
        assert result.metrics["correction"] == "bh" and "alpha_bonferroni" not in result.metrics
