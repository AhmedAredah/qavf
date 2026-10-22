"""Distribution distances and the shared STA-002 binning, against independent oracles.

PSI is checked against values worked by hand, Jensen-Shannon against
``scipy.spatial.distance.jensenshannon``, Wasserstein against
``scipy.stats.wasserstein_distance``, the KS distance against
``scipy.stats.ks_2samp`` and quantile deltas against ``numpy.quantile``. The
binning STA-002 shares with the ``distribution`` level is pinned by golden
values recorded on the commit before it was shared (``2a452f1``).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from qavf.compare.levels.distribution import numeric_bins
from qavf.inference.effects import (
    jensen_shannon,
    ks_distance,
    population_stability,
    quantile_deltas,
    wasserstein,
)
from qavf.rules import _distribution
from qavf.statistics.ordered import bin_index, quantile_cutpoints
from qavf.testing import rule_results

scipy = pytest.importorskip("scipy")
from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy.stats import ks_2samp, wasserstein_distance  # noqa: E402

SAMPLES = [
    (np.random.default_rng(1).normal(0, 1, 300), np.random.default_rng(2).normal(0.4, 1.3, 250)),
    (np.array([1.0, 2.0, 2.0, 5.0]), np.array([2.0, 3.0, 9.0])),
    (
        1e12 + np.random.default_rng(3).normal(0, 1, 50),
        1e12 + np.random.default_rng(4).normal(0, 1, 60),
    ),
]


def test_psi_matches_a_hand_computed_value():
    # Shares 0.5/0.5 against 0.25/0.75: (0.25 - 0.5) ln(0.5) + (0.75 - 0.5) ln(1.5).
    expected = -0.25 * math.log(0.5) + 0.25 * math.log(1.5)
    assert population_stability([(50, 25), (50, 75)], (100, 100)) == pytest.approx(expected, 1e-15)
    # An empty bin is floored at 1e-6 rather than infinite.
    floored = (1 - 1e-6) * math.log(1 / 1e-6) + (1e-6 - 1) * math.log(1e-6 / 1)
    assert population_stability([(0, 10), (10, 0)], (10, 10)) == pytest.approx(floored, 1e-12)
    assert population_stability([(3, 3), (7, 7)], (10, 10)) == 0.0


@pytest.mark.parametrize(("baseline", "candidate"), [([5, 5, 0], [1, 2, 7]), ([1, 0], [0, 1]),
                                                     ([3, 3, 3], [3, 3, 3])])  # fmt: skip
def test_jensen_shannon_is_scipys_squared_distance(baseline, candidate):
    p, q = np.array(baseline) / sum(baseline), np.array(candidate) / sum(candidate)
    distance = jensenshannon(p, q, base=2)
    assert jensen_shannon(baseline, candidate) == pytest.approx(distance**2, abs=1e-12)


@pytest.mark.parametrize(("baseline", "candidate"), SAMPLES)
def test_wasserstein_and_ks_distances_match_scipy(baseline, candidate):
    assert wasserstein(baseline, candidate) == pytest.approx(
        wasserstein_distance(baseline, candidate), rel=1e-9, abs=1e-9
    )
    assert ks_distance(baseline, candidate) == pytest.approx(
        ks_2samp(baseline, candidate).statistic, abs=1e-12
    )


def test_quantile_deltas_are_numpys_linear_quantiles():
    baseline, candidate = SAMPLES[0]
    for p, qb, qc in quantile_deltas(baseline, candidate, (0.1, 0.5, 0.9)):
        assert (qb, qc) == (np.quantile(baseline, p), np.quantile(candidate, p))


def test_level_bins_are_the_baselines_quantiles_with_support_tails():
    rng = np.random.default_rng(5)
    baseline, candidate = rng.normal(0, 1, 1000), rng.normal(0, 1.5, 900)
    labels, b, c = numeric_bins(baseline, candidate, 10)
    assert len(labels) == len(b) == len(c) == 12  # 10 quantile bins and two tails
    assert b.sum() == 1000 and c.sum() == 900
    assert b[0] == 0 and b[-1] == 0  # the baseline never falls in its own tails
    assert min(b[1:-1]) >= 80 and max(b[1:-1]) <= 120
    # Few distinct values: one bin per value of either side, as STA-002 does.
    labels, b, c = numeric_bins(np.array([1.0, 1.0, 2.0]), np.array([2.0, 3.0]), 10)
    assert labels == ["1", "2", "3"] and list(b) == [2, 1, 0] and list(c) == [0, 1, 1]
    edges = quantile_cutpoints([(v, 1) for v in (1.0, 2.0, 3.0, 4.0)], 4, 2)
    assert list(bin_index(edges, [0.5, 1.0, 2.5, 4.0, 5.0])) == [0, 1, 2, 2, 3]


def test_sta002_golden_values_are_unchanged_by_the_shared_binning(monkeypatch):
    """Values recorded with the pre-sharing implementation (commit 2a452f1)."""
    raw = []
    original = _distribution.Distribution.psi

    def spy(self, bins):
        value = original(self, bins)
        raw.append(value)
        return value

    monkeypatch.setattr(_distribution.Distribution, "psi", spy)
    rng = np.random.default_rng(20251022)
    frames = {
        "ref": pl.DataFrame({"x": rng.normal(100, 10, 3000),
                             "c": rng.choice(["a", "b", "c", "d"], 3000, p=[.4, .3, .2, .1])}),
        "cur": pl.DataFrame({"x": rng.normal(103, 11, 2500),
                             "c": rng.choice(["a", "b", "c", "e"], 2500, p=[.35, .3, .25, .1])}),
        "iref": pl.DataFrame({"x": rng.integers(0, 6, 400)}),
        "icur": pl.DataFrame({"x": rng.integers(0, 8, 380)}),
        "wref": pl.DataFrame({"x": 1e12 + rng.normal(0, 1, 1000)}),
        "wcur": pl.DataFrame({"x": 1e12 + rng.normal(0.3, 1, 900)}),
    }  # fmt: skip
    cases = [
        {"dataset": "cur", "reference": "ref", "column": "x"},
        {"dataset": "cur", "reference": "ref", "column": "x", "bins": 20, "name": "b20"},
        {"dataset": "cur", "reference": "ref", "column": "c", "name": "cat"},
        {"dataset": "icur", "reference": "iref", "column": "x", "name": "ints"},
        {"dataset": "wcur", "reference": "wref", "column": "x", "name": "wide"},
    ]
    results = rule_results("STA-002", frames, {"comparisons": cases})
    assert [(r.status.name, r.metrics["psi"], r.metrics["bins"]) for r in results] == [
        ("PASS", 0.113962, 12),
        ("PASS", 0.123423, 22),
        ("FAIL", 2.461109, 5),
        ("FAIL", 3.127381, 8),
        ("PASS", 0.108117, 12),
    ]
    assert raw == [
        (0.11396244077444291, 12, "reference_quantiles_with_support_tails"),
        (0.12342310172422152, 22, "reference_quantiles_with_support_tails"),
        (2.4611094285861848, 5, "distinct_values"),
        (3.127381487328194, 8, "distinct_values"),
        (0.108116668389478, 12, "reference_quantiles_with_support_tails"),
    ]
