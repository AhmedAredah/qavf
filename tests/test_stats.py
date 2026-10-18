"""Statistical outlier primitives."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from qavf.stats import (
    OutlierMethod,
    detect_outliers,
    flag_outliers,
    iqr_outliers,
    modified_zscore_outliers,
    zscore_outliers,
)

# Twelve well-behaved values and one gross outlier at index 12.
SAMPLE = np.array([10.1, 9.8, 10.3, 10.0, 9.9, 10.2, 10.4, 9.7, 10.0, 10.1, 9.9, 10.2, 25.0])


@pytest.mark.parametrize("method", list(OutlierMethod))
def test_each_method_finds_the_gross_outlier(method: OutlierMethod) -> None:
    result = detect_outliers(SAMPLE, method)
    assert result.mask.tolist() == [False] * 12 + [True]
    assert result.count == 1
    assert result.lower < 9.7 and result.upper < 25.0
    assert result.method is method


def test_zscore_matches_definition() -> None:
    result = zscore_outliers(SAMPLE, threshold=2.0)
    expected = (SAMPLE - SAMPLE.mean()) / SAMPLE.std(ddof=1)
    np.testing.assert_allclose(result.scores, expected)


def test_iqr_fences_and_scores() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    result = iqr_outliers(values, threshold=1.5)
    q1, q3 = np.percentile(values, [25, 75])
    assert result.lower == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert result.upper == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert result.scores[:5].tolist() == [0.0] * 5
    assert result.scores[5] == pytest.approx((100.0 - result.upper) / (q3 - q1))


def test_modified_zscore_resists_masking() -> None:
    # Two large outliers inflate the standard deviation enough to hide each other
    # from the plain z-score, but not from the median/MAD version.
    values = np.array([10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 60.0, 62.0])
    assert zscore_outliers(values).count == 0
    assert modified_zscore_outliers(values).mask.tolist() == [False] * 6 + [True, True]


@pytest.mark.parametrize("method", list(OutlierMethod))
def test_degenerate_samples_yield_no_flags(method: OutlierMethod) -> None:
    for values in ([], [1.0, 2.0], [5.0] * 10, [np.nan] * 6):
        result = detect_outliers(values, method)
        assert result.count == 0
        assert len(result.mask) == len(values)


def test_nans_are_ignored_not_flagged() -> None:
    values = np.append(SAMPLE, np.nan)
    result = detect_outliers(values, OutlierMethod.IQR)
    assert result.mask.tolist() == [False] * 12 + [True, False]


def test_custom_threshold_changes_sensitivity() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.5])
    assert detect_outliers(values, "iqr", threshold=1.0).count == 1
    assert detect_outliers(values, "iqr", threshold=3.0).count == 0


@pytest.mark.parametrize("method", list(OutlierMethod))
def test_grouped_polars_scan_matches_array_functions(method: OutlierMethod) -> None:
    rng = np.random.default_rng(7)
    frames = []
    for group in range(6):
        values = rng.normal(100, 5, size=30)
        values[rng.integers(0, 30)] *= 3  # plant an outlier
        frames.append(pl.DataFrame({"g": [group] * 30, "x": values}))
    frame = pl.concat(frames)

    scanned = flag_outliers(frame, "x", ["g"], method).frame
    for group in range(6):
        part = scanned.filter(pl.col("g") == group)
        reference = detect_outliers(part["x"].to_numpy(), method)
        assert part["is_outlier"].to_list() == reference.mask.tolist()
        np.testing.assert_allclose(part["score"].to_numpy(), reference.scores, rtol=1e-9)
        assert part["lower_bound"][0] == pytest.approx(reference.lower)
        assert part["upper_bound"][0] == pytest.approx(reference.upper)
    assert scanned["is_outlier"].sum() >= 6


def test_grouped_scan_respects_group_size_and_nulls() -> None:
    frame = pl.DataFrame(
        {"g": ["a"] * 13 + ["b"] * 3 + ["a"], "x": [*SAMPLE.tolist(), 1.0, 1.0, 900.0, None]}
    )
    scanned = flag_outliers(frame, "x", ["g"], "modified_zscore", min_group_size=5).frame
    assert scanned.height == 16  # the null row is dropped
    small = scanned.filter(pl.col("g") == "b")
    assert small["is_outlier"].sum() == 0 and small["score"].null_count() == 3
    assert scanned.filter(pl.col("is_outlier"))["x"].to_list() == [25.0]
    # Ungrouped: the whole frame is one group.
    assert flag_outliers(frame, "x", method="iqr").flags.sum() == 4
