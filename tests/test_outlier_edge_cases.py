"""Outlier and coverage edge cases.

Degenerate spread and non-positive medians, invalid normalisation, untestable temporal
series and missing coverage periods,
scale invariance, and silently altered configuration.
"""

from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings, load_settings
from qavf.stats import OutlierMethod, detect_outliers, flag_outliers
from qavf.testing import rule_results, run_rule

SPIKE = [10.0] * 9 + [1e9]

HAS_STATS = all(importlib.util.find_spec(m) is not None for m in ("scipy", "statsmodels"))
needs_stats = pytest.mark.skipif(not HAS_STATS, reason="needs the qavf[stats] extra")


# =========================================================================
# Zero spread must not hide extreme values
# =========================================================================
MOSTLY_ZERO = [0.0] * 8 + [1.0, 0.0, 0.0, 7.0]


@pytest.mark.parametrize(
    ("method", "values", "flagged"),
    [
        ("iqr", SPIKE, [1e9]),
        ("iqr", [10.0] * 9 + [-1e9], [-1e9]),
        ("iqr", [0.0] * 12 + [3.0, 5.0], [3.0, 5.0]),
        # Tukey's fences are both 0, so every non-zero value lies outside them.
        ("iqr", MOSTLY_ZERO, [1.0, 7.0]),
        ("modified_zscore", SPIKE, [1e9]),
        ("modified_zscore", [10.0] * 9 + [-1e9], [-1e9]),
        ("modified_zscore", [0.0] * 12 + [3.0, 5.0], [3.0, 5.0]),
        # With a zero MAD the mean absolute deviation stands in: 7 is far, 1 is not.
        ("modified_zscore", MOSTLY_ZERO, [7.0]),
    ],
)
def test_departures_from_a_tied_majority_are_flagged(method, values, flagged) -> None:
    result = detect_outliers(values, method)
    assert [v for v, m in zip(values, result.mask, strict=True) if m] == flagged
    assert np.isfinite(result.lower) and np.isfinite(result.upper)
    scan = flag_outliers(pl.DataFrame({"x": values}), "x", method=method)
    assert scan.flags.to_list() == result.mask.tolist()  # grouped and array forms agree
    assert scan.tested == len(values)


@pytest.mark.parametrize("method", list(OutlierMethod))
def test_identical_values_are_tested_and_clean(method) -> None:
    # 12 values: enough for every method (a z-score at 3 needs 11 to flag anything).
    result = detect_outliers([4.0] * 12, method)
    assert result.count == 0
    scan = flag_outliers(pl.DataFrame({"x": [4.0] * 12}), "x", method=method)
    assert scan.tested == 12 and scan.flags.sum() == 0


def test_statistical_rule_flags_the_spike(registry) -> None:
    frame = pl.DataFrame({"x": SPIKE})
    params = {"targets": [{"dataset": "t", "value": "x"}], "min_group_size": 5}
    result = run_rule("STA-001", {"t": frame}, params, registry=registry)
    assert result.status is Status.FAIL and result.total_records == 10
    assert [r["x"] for r in result.affected_records] == [1e9]


@pytest.mark.parametrize(
    ("values", "flagged"),
    [([0.0, 0.0, 1000.0], ["c"]), ([-10.0, -10.0, -1000.0], ["c"]), ([5.0, 5.0, -5.0], ["c"])],
    ids=["zero-median", "negative-median", "sign-flip"],
)
def test_group_comparison_handles_non_positive_medians(values, flagged, registry) -> None:
    frame = pl.DataFrame({"e": ["a", "b", "c"], "v": values})
    params = {"metrics": [{"dataset": "t", "value": "v", "entity_column": "e"}]}
    result = run_rule("SPA-002", {"t": frame}, params, registry=registry)
    assert result.status is Status.FAIL
    assert [r["e"] for r in result.affected_records] == flagged


def test_group_comparison_all_zero_group_passes(registry) -> None:
    frame = pl.DataFrame({"e": ["a", "b", "c"], "v": [0.0, 0.0, 0.0]})
    params = {"metrics": [{"dataset": "t", "value": "v", "entity_column": "e"}]}
    assert run_rule("SPA-002", {"t": frame}, params, registry=registry).status is Status.PASS


# =========================================================================
# Invalid normalisation never erases an entity
# =========================================================================
@pytest.mark.parametrize("bad_denominator", ["missing", 0.0, -1.0, math.nan, None])
@pytest.mark.parametrize("rule_id", ["SPA-001", "SPA-002"])
def test_entities_with_invalid_denominators_are_reported(
    rule_id, bad_denominator, registry
) -> None:
    entities = list("abcdef")
    numerator = pl.DataFrame({"e": entities, "v": [10.0, 10.1, 10.2, 9.9, 9.8, 1000.0]})
    denominators = [1.0] * 5 + ([] if bad_denominator == "missing" else [bad_denominator])
    denominator = pl.DataFrame(
        {"e": entities[: len(denominators)], "d": denominators},
        schema={"e": pl.String, "d": pl.Float64},
    )
    metric = {
        "dataset": "n",
        "value": "v",
        "entity_column": "e",
        "normalize_by": {"dataset": "d", "value": "d"},
    }
    results = rule_results(
        rule_id, {"n": numerator, "d": denominator}, {"metrics": [metric]}, registry=registry
    )
    (blocked,) = [r for r in results if r.status is Status.ERROR]
    assert [r["e"] for r in blocked.affected_records] == ["f"]
    assert "d" in blocked.datasets and "n" in blocked.datasets


# =========================================================================
# Untestable series and absent periods are not certified
# =========================================================================
def _series(values, years=None) -> dict[str, pl.DataFrame]:
    years = years or list(range(2020, 2020 + len(values)))
    return {"t": pl.DataFrame({"year": years, "g": ["a"] * len(values), "v": values})}


def _params(**extra) -> dict:
    return {"series": [{"dataset": "t", "value": "v", "group_by": ["g"], **extra}]}


@pytest.mark.parametrize(
    ("rule_id", "values", "extra"),
    [
        pytest.param(
            "TMP-002",
            [5.0, 4.0, math.nan, 2.0, 1.0],
            {"expected_trend": "increasing"},
            marks=needs_stats,
        ),
        pytest.param(
            "TMP-003", [1.0, 2.0, math.nan, 100.0, 5.0, 6.0, 7.0, 8.0], {}, marks=needs_stats
        ),
        ("TMP-001", [1.0, math.inf, 2.0], {}),
    ],
)
def test_non_finite_series_values_are_not_certified(rule_id, values, extra, registry) -> None:
    results = rule_results(rule_id, _series(values), _params(**extra), registry=registry)
    assert any(r.status is Status.ERROR and "non-finite" in r.message for r in results)
    assert not any(r.status is Status.PASS for r in results)


@needs_stats
def test_too_short_series_are_disclosed(registry) -> None:
    frames = {
        "t": pl.DataFrame(
            {
                "year": [2020, 2021, 2020, 2021, 2022, 2023, 2024],
                "g": ["a"] * 2 + ["b"] * 5,
                "v": [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
    }
    result = run_rule(
        "TMP-002",
        frames,
        _params(
            expected_trend="increasing",
        ),
        registry=registry,
    )
    # A clean assessed subset does not certify the group with insufficient data.
    assert result.status is Status.ERROR and "too few periods" in result.message


@needs_stats
def test_untestable_inputs_are_not_passes(registry) -> None:
    two_points = run_rule(
        "TMP-002", _series([1.0, 2.0]), _params(expected_trend="increasing"), registry=registry
    )
    assert two_points.status is Status.ERROR
    empty = {
        "t": pl.DataFrame(
            {"year": [], "g": [], "v": []},
            schema={"year": pl.Int64, "g": pl.String, "v": pl.Float64},
        )
    }
    missing = run_rule(
        "TMP-004", empty, {**_params(), "expected_periods": [2020]}, registry=registry
    )
    assert missing.status is Status.ERROR


def test_coverage_checks_absent_periods_and_empty_tables(registry) -> None:
    t = pl.DataFrame({"e": ["a", "b"], "year": [2020, 2020]})
    u = pl.DataFrame({"e": ["a", "b", "a", "b"], "year": [2020, 2020, 2021, 2021]})
    params = {"entity_column": "e", "time_column": "year", "expected_entities": ["a", "b"]}
    result = run_rule("SPA-003", {"t": t, "u": u}, params, registry=registry)
    assert result.status is Status.FAIL
    assert result.affected_records == [
        {"dataset": "t", "year": 2021, "missing_entities": ["a", "b"]}
    ]

    explicit = run_rule(
        "SPA-003", {"t": t}, {**params, "expected_periods": [2020, 2022]}, registry=registry
    )
    assert [r["year"] for r in explicit.affected_records] == [2022]

    empty = pl.DataFrame({"e": [], "year": []}, schema={"e": pl.String, "year": pl.Int64})
    lone = run_rule("SPA-003", {"t": empty}, params, registry=registry)
    assert lone.status is not Status.PASS
    with_peer = run_rule("SPA-003", {"t": empty, "u": u}, params, registry=registry)
    assert with_peer.status is Status.FAIL
    assert {r["year"] for r in with_peer.affected_records if r["dataset"] == "t"} == {2020, 2021}


# =========================================================================
# Verdicts do not depend on units
# =========================================================================
@pytest.mark.parametrize("scale", [1.0, 1e-9, 1e9, -1.0])
@pytest.mark.parametrize("log_transform", [True, False])
@needs_stats
def test_trend_outlier_verdict_is_scale_invariant(scale, log_transform, registry) -> None:
    base = [0.0, 0.1, 0.2, 0.3, 4.0, 0.5, 0.6, 0.7, 0.8]
    values = [v * scale for v in base]
    params = {**_params(), "log_transform": log_transform}
    result = run_rule("TMP-003", _series(values), params, registry=registry)
    assert result.status is Status.FAIL, (scale, log_transform, result.message)
    assert [r["year"] for r in result.affected_records] == [2024]


@needs_stats
def test_a_truly_constant_series_is_consistent(registry) -> None:
    result = run_rule("TMP-003", _series([3e-12] * 6), _params(), registry=registry)
    assert result.status is Status.PASS


# =========================================================================
# Configuration cannot silently change the intended checks
# =========================================================================
def test_duplicate_yaml_keys_are_rejected_with_location(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "datasets:\n  t:\n    schema:\n      columns:\n        x: {ge: 0}\n        x: {le: 100}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate key 'x'.*line 6"):
        load_settings(path)


@pytest.mark.parametrize(
    "document",
    [
        '{"datasets": {"t": {}, "t": {}}}',
        '{"rules": {"TMP-001": {}}, "rules": {}}',
        '{"datasets": {"t": {"schema": {"columns": {"x": {"ge": 0, "ge": 1}}}}}}',
    ],
)
def test_duplicate_json_keys_are_rejected(tmp_path, document) -> None:
    path = tmp_path / "c.json"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_settings(path)


def test_dataset_names_that_differ_only_in_case_are_rejected() -> None:
    with pytest.raises(ValueError, match="differ only in case"):
        QAVFSettings.model_validate({"datasets": {"A": {}, "a": {}}})
    with pytest.raises(ValueError, match="differ only in case"):
        DataContext(frames={"A": pl.DataFrame({"x": [1]}), "a": pl.DataFrame({"x": [2]})})


def test_valid_yaml_and_json_still_load(tmp_path) -> None:
    (tmp_path / "c.yaml").write_text("datasets:\n  a: {}\n  b: {}\n", encoding="utf-8")
    assert set(load_settings(tmp_path / "c.yaml").datasets) == {"a", "b"}
    (tmp_path / "c.json").write_text(json.dumps({"datasets": {"a": {}, "b": {}}}), encoding="utf-8")
    assert set(load_settings(tmp_path / "c.json").datasets) == {"a", "b"}
