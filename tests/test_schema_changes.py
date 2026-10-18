"""Schema changes and comparison edge cases.

Schema changes in regression comparison, key names that collide with evidence
fields in REG-001, the hidden-difference count of ``assert_outputs_match``, and
z-score tests that cannot flag anything in small groups.
"""

from __future__ import annotations

import math
import sys

import polars as pl
import pytest

import qavf.regression  # noqa: F401  (the package re-exports a function of the module's name)
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings
from qavf.stats import detect_outliers, flag_outliers, zscore_min_samples
from qavf.testing import discovered_rules

co = sys.modules["qavf.regression.compare_outputs"]
REGISTRY = discovered_rules(entry_points=False)


def _run(frames, **config):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False, "parallel": False},
            "datasets": {name: {} for name in frames},
            **config,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


def _frame(values, dtype) -> pl.DataFrame:
    return pl.DataFrame({"k": list(range(len(values))), "v": pl.Series(values, dtype=dtype)})


# =========================================================================
# Schema changes are differences, never a silent pass
# =========================================================================
@pytest.mark.parametrize(
    ("base", "cand"),
    [
        (([], pl.Float64), ([], pl.String)),
        (([1.0], pl.Float64), (["1.0"], pl.String)),
        (([True], pl.Boolean), ([1], pl.Int64)),
        (([None], pl.Date), ([None], pl.Datetime("us"))),
    ],
)
def test_kind_changes_are_reported(base, cand) -> None:
    result = co.compare_frames(_frame(*base), _frame(*cand), ["k"])
    assert not result.equal and result.dtypes_changed == ["v"]
    [change] = [d for d in result.differences if d.kind == "dtype_changed"]
    assert change.column == "v" and change.baseline != change.candidate


def test_time_zone_changes_are_kind_changes() -> None:
    from datetime import datetime

    naive = _frame([datetime(2020, 1, 1)], pl.Datetime("us"))
    aware = _frame([datetime(2020, 1, 1)], pl.Datetime("us", "UTC"))
    assert co.compare_frames(naive, aware, ["k"]).dtypes_changed == ["v"]


@pytest.mark.parametrize(
    ("base", "cand"),
    [
        (([1, 2], pl.Int64), ([1.0, 2.0], pl.Float64)),
        (([1, 2], pl.Int32), ([1, 2], pl.Int64)),
        ((["a"], pl.String), (["a"], pl.Categorical)),
        (([None], pl.Null), ([None], pl.Float64)),
    ],
)
def test_compatible_changes_pass_by_default_but_not_when_strict(base, cand) -> None:
    assert co.compare_frames(_frame(*base), _frame(*cand), ["k"]).equal
    strict = co.compare_frames(_frame(*base), _frame(*cand), ["k"], schema="strict")
    assert strict.dtypes_changed == ["v"] and not strict.equal


def test_schema_checks_can_be_ignored() -> None:
    assert co.compare_frames(
        _frame([], pl.Float64), _frame([], pl.String), ["k"], schema="ignore"
    ).equal


def test_reg001_reports_schema_changes(tmp_path) -> None:
    path = tmp_path / "b.parquet"
    _frame([1.0, 2.0], pl.Float64).write_parquet(path)
    baselines = [{"dataset": "out", "path": str(path), "keys": ["k"]}]
    report = _run(
        {"out": _frame(["1.0", "2.0"], pl.String)},
        rules={"REG-001": {"params": {"baselines": baselines}}},
    )
    result = report.get("REG-001")
    assert result.status is Status.FAIL
    assert any(r["kind"] == "dtype_changed" for r in result.affected_records)


# =========================================================================
# Key columns never overwrite REG-001 evidence fields
# =========================================================================
def test_reg001_keys_are_nested_in_evidence(tmp_path) -> None:
    base = pl.DataFrame({"dataset": [1], "column": ["x"], "kind": ["k"], "v": [1.0]})
    path = tmp_path / "b.parquet"
    base.write_parquet(path)
    baselines = [{"dataset": "out", "path": str(path), "keys": ["dataset", "column", "kind"]}]
    report = _run(
        {"out": base.with_columns(pl.col("v") + 1)},
        rules={"REG-001": {"params": {"baselines": baselines}}},
    )
    [record] = report.get("REG-001").affected_records
    assert (
        record["dataset"] == "out" and record["kind"] == "value_changed" and record["column"] == "v"
    )
    assert record["key"] == {"dataset": 1, "column": "x", "kind": "k"}


# =========================================================================
# assert_outputs_match says how many differences it did not show
# =========================================================================
def test_hidden_count_includes_every_kind_of_difference() -> None:
    base = pl.DataFrame({"k": [1], **{f"c{i}": [1] for i in range(15)}})
    with pytest.raises(AssertionError) as info:
        co.assert_outputs_match(base, pl.DataFrame({"k": [1]}), ["k"], show=3)
    assert info.value.args[0].splitlines()[-1].strip() == "... and 12 more"


# =========================================================================
# A z-score test that cannot flag anything is untested, not clean
# =========================================================================
@pytest.mark.parametrize(("threshold", "minimum"), [(3.0, 11), (2.0, 6), (2.5, 9), (1.0, 3)])
def test_minimum_samples_for_a_z_score_to_be_able_to_flag(threshold, minimum) -> None:
    n = zscore_min_samples(threshold)
    assert n == minimum
    assert (n - 1) / math.sqrt(n) > threshold >= (n - 2) / math.sqrt(n - 1) or n == 3


def test_small_samples_are_untested_by_the_z_score() -> None:
    spike = [1.0] * 9 + [1e9]
    array = detect_outliers(spike, "zscore", 3.0)
    assert not array.mask.any() and all(math.isnan(s) for s in array.scores)
    scan = flag_outliers(pl.DataFrame({"v": spike}), "v", method="zscore", threshold=3.0)
    assert scan.tested == 0
    bigger = [1.0] * 10 + [1e9]
    assert detect_outliers(bigger, "zscore", 3.0).mask.tolist() == [False] * 10 + [True]


def test_sta001_explains_why_a_z_score_could_not_be_assessed() -> None:
    frame = pl.DataFrame({"v": [1.0] * 9 + [1e9]})
    params = {"targets": [{"dataset": "t", "value": "v"}], "method": "zscore", "min_group_size": 8}
    report = _run({"t": frame}, rules={"STA-001": {"params": params}})
    [result] = report.results_for("STA-001")
    assert result.status is Status.ERROR and "11" in result.message and "z-score" in result.message


def test_reg001_schema_policy_is_configurable() -> None:
    from qavf.rules.regression_checks import BaselineSpec

    spec = BaselineSpec.model_validate(
        {"dataset": "o", "path": "b.csv", "keys": ["k"], "schema": "strict"}
    )
    assert spec.schema_policy == "strict"
    assert (
        BaselineSpec.model_validate({"dataset": "o", "path": "b.csv", "keys": ["k"]}).schema_policy
        == "compatible"
    )
