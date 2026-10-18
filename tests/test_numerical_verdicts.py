"""Numerical verdicts checked against exact or metamorphic independent oracles."""

import math
from fractions import Fraction
from statistics import median

import numpy as np
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.rules.spatial_checks import GroupComparisonRule
from qavf.rules.statistical_checks import DistributionDriftRule
from qavf.rules.temporal_checks import StepChangeRule, _best_split
from qavf.testing import rule_results


def metric(normalizer=None):
    spec = {"dataset": "n", "entity_column": "id", "value": "v"}
    if normalizer is not None:
        spec["normalize_by"] = normalizer
    return spec


@pytest.mark.parametrize("rule", ["SPA-001", "SPA-002"])
def test_nonzero_normalization_underflow_is_explicitly_unassessed(rule):
    frames = {
        "n": pl.DataFrame({"id": range(8), "v": [1e-200] * 7 + [1e-198]}),
        "d": pl.DataFrame({"id": range(8), "w": [1e200] * 8}),
    }
    assert Fraction(frames["n"]["v"][-1]) / Fraction(frames["n"]["v"][0]) > 99
    results = rule_results(rule, frames, {"metrics": [metric({"dataset": "d", "value": "w"})]})
    assert not any(r.status is Status.PASS for r in results)
    excluded = next(r for r in results if r.affected_count == 8)
    assert excluded.status is Status.ERROR
    assert all("underflows" in row["problem"] for row in excluded.affected_records)


def test_zero_numerator_is_valid_and_representable_normalization_still_detects_outlier():
    for values, expected in [([0.0] * 8, Status.PASS), ([1.0] * 7 + [100.0], Status.FAIL)]:
        frames = {
            "n": pl.DataFrame({"id": range(8), "v": values}),
            "d": pl.DataFrame({"id": range(8), "w": [1e200] * 8}),
        }
        results = rule_results(
            "SPA-002", frames, {"metrics": [metric({"dataset": "d", "value": "w"})]}
        )
        assert {r.status for r in results} == {expected}


def test_subnormal_normalization_cannot_merge_materially_different_ratios():
    frames = {
        "n": pl.DataFrame({"id": range(8), "v": [2.6e-16] * 7 + [7.3e-16]}),
        "d": pl.DataFrame({"id": range(8), "w": [1e308] * 8}),
    }
    assert Fraction(7.3e-16) / Fraction(2.6e-16) > 2
    results = rule_results(
        "SPA-002", frames, {"metrics": [metric({"dataset": "d", "value": "w"})]}
    )
    assert {r.status for r in results} == {Status.ERROR}
    assert "subnormal-value precision" in results[0].message


def test_exact_subnormal_normalization_remains_supported():
    smallest = math.ldexp(1.0, -1074)
    frames = {
        "n": pl.DataFrame({"id": range(8), "v": [smallest] * 7 + [smallest * 100]}),
        "d": pl.DataFrame({"id": range(8), "w": [1.] * 8}),
    }
    results = rule_results(
        "SPA-002", frames, {"metrics": [metric({"dataset": "d", "value": "w"})]}
    )
    assert {r.status for r in results} == {Status.FAIL}
    assert results[0].affected_count == 1


def exact_split(values, minimum=4):
    values = list(map(Fraction, values))
    best = None
    for split in range(minimum, len(values) - minimum + 1):
        left, right = values[:split], values[split:]
        before, after = median(left), median(right)
        residuals = [abs(v - before) for v in left] + [abs(v - after) for v in right]
        spread = Fraction("1.4826") * median(residuals)
        gap = abs(after - before)
        if not gap:
            continue
        score = float(gap / spread) if spread else math.inf
        if best is None or score > best[1]:
            best = split, score
    return best


@pytest.mark.parametrize("offset", [0.0, 1e16, -1e16])
def test_level_shift_preserves_exact_offset_with_rational_median_mad_oracle(offset):
    base = np.array([18.0, 18.0, 0.0, 12.0, 8.0, 10.0, 8.0, 12.0, -18.0, -2.0, -12.0, -20.0])
    values = base + offset
    assert all(
        Fraction(v) - Fraction(offset) == Fraction(x) for x, v in zip(base, values, strict=True)
    )
    expected = exact_split(values)
    result = _best_split(values, 4, 0)
    assert expected is not None and result is not None
    assert result[0] == expected[0] == 8
    assert result[3] == pytest.approx(expected[1], rel=1e-14)
    results = rule_results(
        "TMP-008",
        {"d": pl.DataFrame({"t": range(len(values)), "v": values})},
        {
            "series": [{"dataset": "d", "value": "v", "time_column": "t"}],
            "min_relative_shift": 0,
        },
    )
    assert {r.status for r in results} == {Status.FAIL}


def test_level_shift_materiality_uses_original_scale():
    values = np.array([10.0] * 4 + [20.0] * 4)
    assert _best_split(values, 4, 0.2) is not None
    assert _best_split(values + 1e16, 4, 0.2) is None


def exact_quantile_psi(reference, current, bins=10):
    reference, current = sorted(map(Fraction, reference)), list(map(Fraction, current))
    edges = []
    for i in range(bins + 1):
        index = Fraction(i * (len(reference) - 1), bins)
        left = index.numerator // index.denominator
        remainder = index - left
        edge = reference[left]
        if remainder:
            edge += remainder * (reference[left + 1] - reference[left])
        if not edges or edge != edges[-1]:
            edges.append(edge)
    expected, actual = [0] * (len(edges) + 1), [0] * (len(edges) + 1)
    for values, counts in [(reference, expected), (current, actual)]:
        for value in values:
            index = sum(value >= edge for edge in edges[:-1]) + (value > edges[-1])
            counts[index] += 1
    return sum(
        (q - p) * math.log(q / p)
        for p, q in (
            (max(a / len(reference), 1e-6), max(b / len(current), 1e-6))
            for a, b in zip(expected, actual, strict=True)
        )
    )


@pytest.mark.parametrize("scale", [1.0, math.ldexp(1.0, 1023)])
def test_psi_extreme_quantiles_match_exact_bin_oracle(scale):
    reference = np.array(
        [
            -1.5e308,
            -1.4e308,
            -1.3e308,
            -1.2e308,
            -1.1e308,
            -1e308,
            1e308,
            1.1e308,
            1.2e308,
            1.3e308,
            1.4e308,
            1.5e308,
        ]
    )
    current = reference.copy()
    current[6] = -1e308
    expected = exact_quantile_psi(reference, current)
    assert np.array_equal((reference / scale) * scale, reference)
    results = rule_results(
        "STA-002",
        {
            "r": pl.DataFrame({"v": reference / scale}),
            "c": pl.DataFrame({"v": current / scale}),
        },
        {"comparisons": [{"dataset": "c", "reference": "r", "column": "v", "min_samples": 2}]},
    )
    assert {r.status for r in results} == {Status.FAIL}
    assert results[0].metrics["psi"] == pytest.approx(expected, abs=5e-7)


def test_psi_unrepresentable_scaling_is_unassessed():
    values = [math.ldexp(1.0, -1074)] + [1e308 + i * 1e306 for i in range(11)]
    frame = pl.DataFrame({"v": values})
    results = rule_results(
        "STA-002",
        {"r": frame, "c": frame},
        {
            "comparisons": [{"dataset": "c", "reference": "r", "column": "v", "min_samples": 2}],
        },
    )
    assert {r.status for r in results} == {Status.ERROR}
    assert "precision loss" in results[0].message


def test_normalizer_infers_actual_shared_keys_and_rejects_empty_intersection():
    frames = {
        "n": pl.DataFrame({"id": [1, 2, 3], "year": [2020] * 3, "v": [1.0, 1.0, 100.0]}),
        "d": pl.DataFrame({"year": [2020], "w": [1.0]}),
    }
    spec = {**metric({"dataset": "d", "value": "w"}), "group_by": ["year"]}
    inferred = rule_results("SPA-002", frames, {"metrics": [spec]})
    explicit = rule_results(
        "SPA-002",
        frames,
        {
            "metrics": [
                {
                    **spec,
                    "normalize_by": {"dataset": "d", "value": "w", "on": ["year"]},
                }
            ]
        },
    )
    assert {r.status for r in inferred} == {Status.FAIL}
    assert inferred[0].affected_records == explicit[0].affected_records
    frames["d"] = pl.DataFrame({"w": [1.0]})
    missing = rule_results("SPA-002", frames, {"metrics": [spec]})
    assert {r.status for r in missing} == {Status.ERROR}
    assert "shared keys" in missing[0].message


@pytest.mark.parametrize(
    "rule,params",
    [
        (
            DistributionDriftRule,
            {"comparisons": [{"dataset": "d", "reference": "r", "column": "v", "min_samples": 2}]},
        ),
        (StepChangeRule, {"series": [{"dataset": "d", "value": "v", "time_column": "t"}]}),
        (GroupComparisonRule, {"metrics": [{"dataset": "d", "value": "v", "entity_column": "id"}]}),
    ],
)
def test_narrow_statistical_rules_do_not_materialize_wide_lazy_frame(rule, params, monkeypatch):
    frame = pl.DataFrame(
        {"id": range(12), "t": range(12), "v": [1.0] * 12, "unused": ["x" * 1000] * 12}
    )
    with DataContext(frames={"d": frame.lazy(), "r": frame.lazy()}) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *args: pytest.fail("eager whole-frame access"))
        findings = rule(params).evaluate(ctx)
    assert findings and all(f.assessed for f in findings)
