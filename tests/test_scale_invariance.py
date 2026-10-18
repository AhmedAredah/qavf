"""Scale-invariant statistics and regression edge cases.

Numerically stable outlier statistics, non-finite transitions and NaN keys in
regression comparison, unambiguous group membership, threshold
boundaries free of rounding noise and invalid time values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.regression.compare_outputs import compare_frames
from qavf.rules.spatial_checks import GroupComparisonParams
from qavf.settings import QAVFSettings
from qavf.stats import OutlierMethod, detect_outliers, flag_outliers

METHODS = [OutlierMethod.ZSCORE, OutlierMethod.MODIFIED_ZSCORE, OutlierMethod.IQR]


# =========================================================================
# Outlier statistics are stable at any magnitude, and implementations agree
# =========================================================================
def _parity(values: list[float], method: OutlierMethod, threshold: float | None = None):
    array = detect_outliers(values, method, threshold)
    grouped = flag_outliers(pl.DataFrame({"v": values}), "v", method=method, threshold=threshold)
    assert grouped.flags.to_list() == array.mask.tolist(), method
    assert np.isfinite(array.scores[np.isfinite(values)]).all() or method is OutlierMethod.IQR
    return array


SCALES = {
    "near overflow": [1e308 + i * 1e306 for i in range(9)] + [1.7e308],
    "huge": [1e160 * (1 + 0.01 * i) for i in range(9)] + [1e162],
    "subnormal": [1e-320 * (1 + i % 3) for i in range(9)] + [1e-310],
    "ordinary": [10.0 + 0.1 * (i % 3) for i in range(9)] + [1000.0],
}


@pytest.mark.parametrize("method", METHODS, ids=str)
@pytest.mark.parametrize("scale", list(SCALES), ids=str)
def test_planted_outlier_found_identically_at_every_scale(method, scale) -> None:
    values = SCALES[scale]
    result = _parity(values, method, 2.5 if method is OutlierMethod.ZSCORE else None)
    assert result.mask.tolist() == [False] * 9 + [True], (scale, method)


@pytest.mark.parametrize("method", METHODS, ids=str)
def test_identical_huge_values_are_clean(method) -> None:
    result = _parity([1.5e308] * 12, method)
    assert not result.mask.any()


@pytest.mark.parametrize("method", METHODS, ids=str)
def test_results_are_scale_invariant(method) -> None:
    base = SCALES["ordinary"]
    reference = detect_outliers(base, method).scores
    for factor in (1e-300, 1e150, 1e300):
        scaled = detect_outliers([v * factor for v in base], method).scores
        np.testing.assert_allclose(scaled, reference, rtol=1e-9)


def test_bounds_are_reported_in_data_units() -> None:
    result = detect_outliers(SCALES["huge"], OutlierMethod.MODIFIED_ZSCORE)
    assert math.isfinite(result.lower) and math.isfinite(result.upper)
    assert result.lower < 1e160 < result.upper < 1e162


# =========================================================================
# Non-finite transitions are counted; NaN keys are rejected
# =========================================================================
def _frame(values: list[Any], keys: list[Any] | None = None) -> pl.DataFrame:
    return pl.DataFrame({"k": keys or list(range(len(values))), "v": values})


def test_finite_maxima_survive_non_finite_transitions() -> None:
    result = compare_frames(_frame([math.nan, 1.0]), _frame([3.0, 10.0]), ["k"])
    assert result.values_changed == 2 and result.max_abs_diff == 9.0
    assert result.nonfinite_changes == 1


@pytest.mark.parametrize(
    ("before", "after"),
    [(math.inf, 1.0), (1.0, -math.inf), (math.inf, -math.inf), (math.nan, math.inf)],
)
def test_non_finite_transitions_are_changes(before, after) -> None:
    result = compare_frames(_frame([before, 5.0]), _frame([after, 5.0]), ["k"])
    assert result.values_changed == 1 and result.nonfinite_changes == 1
    assert result.max_abs_diff is None


def test_nan_equals_nan_in_values() -> None:
    assert compare_frames(_frame([math.nan]), _frame([math.nan]), ["k"]).equal


@pytest.mark.parametrize("key", [math.nan])
def test_nan_keys_are_rejected(key) -> None:
    frame = _frame([1.0], [key])
    with pytest.raises(ValueError, match="NaN"):
        compare_frames(frame, frame, ["k"])


def test_null_keys_match_each_other_consistently() -> None:
    schema = {"k": pl.Int64, "v": pl.Float64}
    before = pl.DataFrame({"k": [None], "v": [1.0]}, schema=schema)
    after = pl.DataFrame({"k": [None], "v": [2.0]}, schema=schema)
    result = compare_frames(before, after, ["k"])
    assert (result.values_changed, result.rows_added, result.rows_removed) == (1, 0, 0)
    assert compare_frames(before, before, ["k"]).equal


# =========================================================================
# Group membership is unambiguous
# =========================================================================
ENTITIES = pl.DataFrame({"e": list("abcde"), "v": [1.0, 1.0, 100.0, 100.0, 100.0]})
METRIC = {"dataset": "m", "value": "v", "entity_column": "e"}


def _spa002(groups: dict[str, list[str]] | None = None, frame: pl.DataFrame = ENTITIES, **params):
    config = {"metrics": [METRIC], **({"groups": groups} if groups is not None else {}), **params}
    settings = QAVFSettings.model_validate(
        {
            "datasets": {"m": {}},
            "rules": {"SPA-002": {"params": config}},
            "engine": {"run_unconfigured_rules": False},
            "plugins": {"entry_points": False},
        }
    )
    from qavf.testing import discovered_rules

    return QARunner(settings, discovered_rules(entry_points=False)).run(frames={"m": frame})


def test_overlapping_groups_are_rejected() -> None:
    with pytest.raises(ValidationError, match="'c'"):
        GroupComparisonParams.model_validate(
            {"metrics": [METRIC], "groups": {"g1": ["a", "b", "c"], "g2": ["c", "d", "e"]}}
        )


def test_empty_groups_are_rejected() -> None:
    with pytest.raises(ValidationError, match="empty"):
        GroupComparisonParams.model_validate({"metrics": [METRIC], "groups": {"empty": []}})


def test_configured_group_too_small_to_compare_is_unassessed() -> None:
    report = _spa002({"g1": ["a", "b"], "g2": ["c", "d", "e"]})
    statuses = {r.check: r.status for r in report.results}
    assert any(s is Status.ERROR for s in statuses.values()), statuses
    blocked = next(r for r in report.results if r.status is Status.ERROR)
    assert "g1" in blocked.message
    assert any(r.status is Status.PASS for r in report.results)


def test_without_groups_all_entities_form_one_group() -> None:
    report = _spa002(None)
    assert report.get("SPA-002").total_records == 5


# =========================================================================
# Exact thresholds are accepted despite binary rounding
# =========================================================================
def _tmp001(values: list[float], years: list[int], limit: float):
    settings = QAVFSettings.model_validate(
        {
            "datasets": {"s": {}},
            "rules": {
                "TMP-001": {
                    "params": {"series": [{"dataset": "s", "value": "v"}], "max_change_rate": limit}
                }
            },
            "engine": {"run_unconfigured_rules": False},
            "plugins": {"entry_points": False},
        }
    )
    from qavf.testing import discovered_rules

    frame = pl.DataFrame({"year": years, "v": values})
    return QARunner(settings, discovered_rules(entry_points=False)).run(frames={"s": frame})


@pytest.mark.parametrize(
    ("values", "years"),
    [
        ([100.0, 110.0], [2020, 2021]),
        ([100.0, 90.0], [2020, 2021]),
        ([100.0, 121.0], [2020, 2022]),
        ([0.3, 0.33], [2020, 2021]),
        ([1e-9, 1.1e-9], [2020, 2021]),
    ],
)
def test_change_exactly_at_the_limit_passes(values, years) -> None:
    assert _tmp001(values, years, 0.1).get("TMP-001").status is Status.PASS


@pytest.mark.parametrize(
    ("values", "years"), [([100.0, 110.001], [2020, 2021]), ([100.0, 89.99], [2020, 2021])]
)
def test_change_just_beyond_the_limit_fails(values, years) -> None:
    assert _tmp001(values, years, 0.1).get("TMP-001").status is Status.FAIL


def test_group_ratio_exactly_at_the_limit_passes() -> None:
    frame = pl.DataFrame({"e": list("abc"), "v": [0.1, 0.3, 0.3]})
    report = _spa002(None, frame, max_ratio=3.0)
    assert report.get("SPA-002").status is Status.PASS


# =========================================================================
# Invalid time values are diagnosed, never crash or count as periods
# =========================================================================
def _temporal(rule: str, frame: pl.DataFrame, **params):
    settings = QAVFSettings.model_validate(
        {
            "datasets": {"s": {}},
            "rules": {
                rule: {
                    "params": {
                        "series": [{"dataset": "s", "value": "v", "group_by": ["g"]}],
                        **params,
                    }
                }
            },
            "engine": {"run_unconfigured_rules": False},
            "plugins": {"entry_points": False},
        }
    )
    from qavf.testing import discovered_rules

    return QARunner(settings, discovered_rules(entry_points=False)).run(frames={"s": frame})


GAPS = pl.DataFrame(
    {
        "g": ["a", "a", "a", "b", "b", "b"],
        "year": [2020, None, 2022, 2020, 2021, 2022],
        "v": [1.0] * 6,
    }
)


@pytest.mark.parametrize("rule", ["TMP-001", "TMP-004"])
def test_missing_time_values_are_reported_with_evidence(rule) -> None:
    report = _temporal(rule, GAPS)
    results = report.results_for(rule)
    blocked = [r for r in results if r.status is Status.ERROR]
    assert len(blocked) == 1 and "time" in blocked[0].message
    assert blocked[0].affected_count == 1 and blocked[0].affected_records[0]["g"] == "a"
    # The unaffected group is still assessed, and nothing is invented from the null.
    assessed = [r for r in results if r.status in (Status.PASS, Status.FAIL)]
    assert assessed and all(r.status is Status.PASS for r in assessed)


def test_nan_time_values_are_invalid() -> None:
    frame = GAPS.with_columns(pl.col("year").cast(pl.Float64).fill_null(math.nan))
    blocked = [r for r in _temporal("TMP-004", frame).results if r.status is Status.ERROR]
    assert len(blocked) == 1 and blocked[0].affected_count == 1


@pytest.mark.parametrize("rule", ["TMP-001", "TMP-004"])
def test_non_numeric_time_columns_are_unassessed(rule) -> None:
    frame = GAPS.with_columns(pl.col("year").cast(pl.String))
    [result] = _temporal(rule, frame).results_for(rule)
    assert result.status is Status.ERROR and "numeric" in result.message
