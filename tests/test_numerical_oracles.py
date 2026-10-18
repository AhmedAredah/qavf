"""Independent exact oracles for the release-blocking numerical findings."""

from fractions import Fraction
from itertools import permutations
from statistics import median, stdev

import numpy as np
import polars as pl
import pytest

from qavf.numeric import exact_float_sum, stable_float_statistic
from qavf.rules.helpers import aggregate
from qavf.stats import detect_outliers, flag_outliers
from qavf.testing import rule_results


@pytest.mark.parametrize("values", list(permutations([1e16, 1.0, -1e16])))
def test_reductions_preserve_cancellation_residual_for_every_permutation(values):
    oracle = sum(map(Fraction, values))
    assert oracle == 1
    assert exact_float_sum(values) == oracle
    assert stable_float_statistic(values, "mean") == oracle / len(values)
    frame = pl.DataFrame({"g": [0] * len(values), "v": values})
    assert aggregate(frame, ["g"], "v")["v"][0] == 1.0
    assert aggregate(frame, ["g"], "v", how="mean")["v"][0] == float(Fraction(1, 3))
    results = rule_results(
        "REC-001",
        {"a": frame, "b": pl.DataFrame({"v": [0.0]})},
        {
            "reconciliations": [
                {
                    "left": {"dataset": "a", "value": "v"},
                    "right": {"dataset": "b", "value": "v"},
                    "rtol": 0,
                }
            ]
        },
    )
    assert any(r.status.value == "FAIL" for r in results)


def test_standard_deviation_keeps_variation_at_large_offset():
    for offset in (0, 1e16, -1e16):
        values = [offset, offset + 2]
        assert stable_float_statistic(values, "std") == stdev(values) == 2**0.5


def test_mean_preserves_extreme_dynamic_range_and_subnormal_residual():
    values = [1e308, 1e-300, -1e308]
    assert stable_float_statistic(values, "mean") == Fraction(1e-300) / 3
    assert exact_float_sum([1e308, 1e308, -1e308]) == Fraction(1e308)


CASES = {
    "zscore": [-40, -8, 10, 4, -8, -4, -10, -6, 8, 6, 6, 0, -6, -8, 6, 10, 10, 10, -10, 10],
    "modified_zscore": [20, 2, -8, 6, 2, 0, -4, -10, -10, 6, 2, -10, -2, -4, -8, 0, -4, 10, -4, 0],
    "iqr": [-20, 8, 6, 8, 8, 8, -8, 10, 6, -10, 6, 4, -6, 2, 0, 8, -2, -10, -2, 0],
}


def quantile(values, p):
    ordered = sorted(map(Fraction, values))
    index = (len(ordered) - 1) * p
    left = index.numerator // index.denominator
    fraction = index - left
    return (
        ordered[left]
        if not fraction
        else (ordered[left] * (1 - fraction) + ordered[left + 1] * fraction)
    )


def exact_flags(values, method):
    values = list(map(Fraction, values))
    if method == "zscore":
        center = sum(values) / len(values)
        variance = sum((v - center) ** 2 for v in values) / (len(values) - 1)
        return [(v - center) ** 2 > 9 * variance for v in values]
    if method == "modified_zscore":
        center = median(values)
        mad = median([abs(v - center) for v in values])
        return [Fraction("0.6745") * abs(v - center) > Fraction("3.5") * mad for v in values]
    q1, q3 = quantile(values, Fraction(1, 4)), quantile(values, Fraction(3, 4))
    return [
        v < q1 - Fraction("1.5") * (q3 - q1) or v > q3 + Fraction("1.5") * (q3 - q1) for v in values
    ]


@pytest.mark.parametrize("method", CASES)
@pytest.mark.parametrize("offset", [0, 1e16, -1e16])
def test_array_grouped_and_rule_outliers_match_exact_translation_oracle(method, offset):
    values = np.array(CASES[method], dtype=float)
    shifted = values + offset
    assert all(
        Fraction(x) - Fraction(offset) == Fraction(y) for x, y in zip(shifted, values, strict=True)
    )
    expected = exact_flags(values, method)
    assert expected == [True] + [False] * (len(values) - 1)
    assert detect_outliers(shifted, method=method).mask.tolist() == expected
    frame = pl.DataFrame({"g": ["x"] * len(shifted), "v": shifted})
    scan = flag_outliers(frame, "v", ["g"], method=method)
    assert scan.flags.to_list() == expected
    results = rule_results(
        "STA-001",
        {"d": frame},
        {
            "targets": [
                {"dataset": "d", "value": "v", "method": method, "min_relative_deviation": 0}
            ]
        },
    )
    assert any(r.status.value == "FAIL" and r.affected_count == 1 for r in results)


def test_valid_neighbor_cannot_hide_unassessable_spatial_target():
    frames = {
        "small": pl.DataFrame({"id": [1, 2], "v": [1.0, 2.0]}),
        "large": pl.DataFrame({"id": [1, 2, 3, 4, 5], "v": [1.0] * 5}),
    }
    results = rule_results(
        "SPA-001",
        frames,
        {"metrics": [{"dataset": n, "value": "v", "entity_column": "id"} for n in frames]},
    )
    assert any("small" in str(r.datasets) and "Not assessed" in r.message for r in results)
    ledger = next(r.metrics["target_outcomes"] for r in results if "target_outcomes" in r.metrics)
    assert [(r["target"], r["outcome"]) for r in ledger] == [
        ("small.v", "unassessed"),
        ("large.v", "pass"),
    ]


def test_spatial_target_ledger_includes_invalid_entities_before_scan():
    results = rule_results(
        "SPA-001",
        {"d": pl.DataFrame({"id": range(6), "v": [1.0] * 5 + [None]})},
        {"metrics": [{"dataset": "d", "value": "v", "entity_column": "id"}]},
    )
    ledger = next(r.metrics["target_outcomes"] for r in results if "target_outcomes" in r.metrics)
    assert ledger[0]["outcome"] == "unassessed"
    assert ledger[0]["examined"] == 5
    assert ledger[0]["unassessed"] == 1
    assert ledger[0]["unassessed_by_unit"] == {"entity values": 1}


def test_spatial_target_ledger_survives_all_targets_unavailable():
    results = rule_results(
        "SPA-001",
        {},
        {"metrics": [{"dataset": "absent", "value": "v", "entity_column": "id"}]},
    )
    ledger = next(r.metrics["target_outcomes"] for r in results if "target_outcomes" in r.metrics)
    assert [(r["target"], r["outcome"]) for r in ledger] == [("absent.v", "unassessed")]
    assert ledger[0]["unassessed"] is None  # No fabricated entity count for an absent source.
