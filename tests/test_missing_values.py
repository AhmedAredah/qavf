"""Missing-value markers and relational input contracts.

Sentinel and blank missing values, decimal-place limits, conditional checks,
row evidence alongside column-level failures, alternate unique keys, duplicate
rows, foreign-key null matching and cardinality, and dataset-selection typos.
"""

from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import ConfigurationError, QARunner
from qavf.engine.severity import Status
from qavf.settings import ColumnCheck, ColumnSpec, QAVFSettings, SchemaSpec
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


def _run(frames, datasets, **config):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False, "parallel": False},
            "rules": {"INP-003": {}, "INP-004": {}, "INP-005": {}},
            "datasets": datasets,
            **config,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


def _checks(report, rule="INP-003"):
    return {r.check: r for r in report.results_for(rule)}


# ------------------------------------------------------------------ missing values
def test_sentinels_count_as_missing_and_are_excluded_from_limits() -> None:
    frame = pl.DataFrame({"v": [10.0, -999.0, 20.0, -999.0]})
    report = _run(
        {"t": frame},
        {
            "t": {
                "schema": {
                    "columns": {
                        "v": {"missing_values": [-999], "ge": 0, "mean": {"ge": 15, "le": 15}}
                    }
                }
            }
        },
    )
    checks = _checks(report)
    not_null = checks["t.v: not null"]
    assert not_null.status is Status.FAIL and not_null.affected_count == 2
    # -999 never violates ge 0, and the mean is of 10 and 20 only.
    assert all(r.status is Status.PASS for k, r in checks.items() if k != "t.v: not null")


def test_sentinels_feed_the_null_fraction() -> None:
    frame = pl.DataFrame({"v": [1, -1, -1, 4]})
    spec = {"nullable": True, "missing_values": [-1], "max_null_fraction": 0.25}
    report = _run({"t": frame}, {"t": {"schema": {"columns": {"v": spec}}}})
    [result] = [r for r in report.results_for("INP-003") if r.failed]
    assert "missing fraction 0.5" in result.message


def test_blank_text_can_be_missing() -> None:
    frame = pl.DataFrame({"s": ["a", "", "  ", None]})
    report = _run({"t": frame}, {"t": {"schema": {"columns": {"s": {"blank_as_missing": True}}}}})
    assert _checks(report)["t.s: not null"].affected_count == 3


def test_blank_as_missing_needs_text() -> None:
    with pytest.raises(ValidationError, match="blank_as_missing"):
        ColumnSpec(dtype="int", blank_as_missing=True)


# ------------------------------------------------------------------ decimal places
@pytest.mark.parametrize(
    ("values", "dtype", "limits", "bad"),
    [
        ([1.0, 2.5, 3.0], pl.Float64, {"integral": True}, 1),
        ([1.25, 0.1 + 0.2, 1.255], pl.Float64, {"max_decimals": 2}, 1),
        ([1, 2, 3], pl.Int64, {"max_decimals": 0}, 0),
        (
            [Decimal("1.20"), Decimal("1.25"), Decimal("1.257")],
            pl.Decimal(10, 3),
            {"max_decimals": 2},
            1,
        ),
    ],
)
def test_decimal_place_limits(values, dtype, limits, bad) -> None:
    frame = pl.DataFrame({"v": pl.Series(values, dtype=dtype)})
    failed = [
        r
        for r in _run({"t": frame}, {"t": {"schema": {"columns": {"v": limits}}}}).results
        if r.failed
    ]
    assert [r.affected_count for r in failed] == ([bad] if bad else [])


def test_decimal_limits_need_numbers() -> None:
    with pytest.raises(ValidationError, match="integral"):
        ColumnSpec(dtype="str", integral=True)


# ------------------------------------------------------------------ conditional checks
def test_where_limits_a_check_to_matching_rows() -> None:
    frame = pl.DataFrame({"status": ["valid", "invalid", "valid"], "v": [5.0, -1.0, -2.0]})
    check = {"ge": 0, "where": "status = 'valid'", "name": "valid rows"}
    report = _run({"t": frame}, {"t": {"schema": {"columns": {"v": {"checks": [check]}}}}})
    result = _checks(report)["t.v: valid rows"]
    assert result.status is Status.FAIL and result.affected_count == 1 and result.total_records == 2
    assert result.affected_records[0]["v"] == -2.0


def test_where_applies_to_column_level_limits() -> None:
    frame = pl.DataFrame({"k": ["a", "a", "b"], "v": [1.0, 1.0, 100.0]})
    check = {"max": {"le": 10}, "where": "k = 'a'", "name": "a only"}
    report = _run({"t": frame}, {"t": {"schema": {"columns": {"v": {"checks": [check]}}}}})
    [summary] = [r for r in report.results_for("INP-003") if r.status is Status.PASS]
    assert "t.v: a only" in summary.metrics["passed_checks"]
    assert not any(r.failed for r in report.results)


def test_too_few_applicable_rows_is_unassessed() -> None:
    frame = pl.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]})
    check = {"ge": 0, "where": "k = 'a'", "min_applicable_rows": 5, "name": "a"}
    report = _run({"t": frame}, {"t": {"schema": {"columns": {"v": {"checks": [check]}}}}})
    result = _checks(report)["t.v: a"]
    assert result.status is Status.ERROR and "fewer than min_applicable_rows" in result.message


def test_where_conditions_are_validated() -> None:
    with pytest.raises(ValidationError, match="where"):
        ColumnCheck(ge=0, where="((")
    with pytest.raises(ValidationError, match="min_applicable_rows"):
        ColumnCheck(ge=0, min_applicable_rows=3)


# ------------------------------------------------------------------ row evidence with aggregates
def test_column_level_failure_still_reports_offending_rows() -> None:
    frame = pl.DataFrame({"v": [1.0, -5.0, 2.0]})
    report = _run(
        {"t": frame}, {"t": {"schema": {"columns": {"v": {"ge": 0, "mean": {"ge": 10}}}}}}
    )
    [result] = [r for r in report.results_for("INP-003") if r.failed]
    assert result.unit == "rows" and result.affected_count == 1
    assert result.affected_records[0]["v"] == -5.0
    assert "mean" in result.message and result.metrics["column_problems"]


# ------------------------------------------------------------------ unique keys / rows
def test_alternate_unique_keys() -> None:
    frame = pl.DataFrame({"id": [1, 2, 3, 4], "code": ["a", "a", None, None], "site": [1, 1, 2, 2]})
    datasets = {"t": {"schema": {"primary_key": ["id"], "unique_keys": [["code", "site"]]}}}
    report = _run({"t": frame}, datasets)
    checks = _checks(report, "INP-004")
    result = checks["t: unique (code, site)"]
    assert result.status is Status.FAIL and result.affected_rows == 2
    summary = [r for r in report.results_for("INP-004") if r.status is Status.PASS]
    assert not summary  # the dataset is not clean
    clean = _run({"t": frame.with_columns(pl.Series("code", ["a", "b", None, None]))}, datasets)
    [ok] = clean.results_for("INP-004")
    assert ok.status is Status.PASS
    assert ok.metrics["rows_exempt_null_unique_keys"] == {"t (code, site)": 2}


def test_duplicate_full_rows() -> None:
    frame = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    report = _run({"t": frame}, {"t": {"schema": {"unique_rows": True}}})
    result = _checks(report, "INP-004")["t: duplicate rows"]
    assert result.status is Status.FAIL and result.affected_count == 2


@pytest.mark.parametrize(
    "config",
    [{"unique_keys": [[]]}, {"unique_keys": [["a", "a"]]}, {"unique_keys": [["a"], ["a"]]}],
)
def test_unique_keys_are_validated(config) -> None:
    with pytest.raises(ValidationError):
        SchemaSpec.model_validate(config)


# ------------------------------------------------------------------ foreign keys
PARENT = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})


def _fk(child, **fk):
    datasets = {
        "parent": {},
        "child": {"foreign_keys": [{"columns": ["a", "b"], "ref_dataset": "parent", **fk}]},
    }
    return _run({"parent": PARENT, "child": child}, datasets)


def test_partly_null_keys_are_exempt_by_default_and_counted() -> None:
    child = pl.DataFrame({"a": [1, None, None], "b": ["x", "q", None]})
    report = _fk(child)
    [result] = report.results_for("INP-005")
    assert result.status is Status.PASS
    assert result.metrics["rows_exempt_null_keys"] == {"child(a, b) -> parent": 2}


def test_match_full_rejects_partly_null_keys() -> None:
    child = pl.DataFrame({"a": [1, None, None], "b": ["x", "q", None]})
    report = _fk(child, match="full")
    result = _checks(report, "INP-005")["child(a, b) -> parent: partly null keys"]
    assert result.status is Status.FAIL and result.affected_count == 1


def test_many_to_one_requires_unique_parent_keys() -> None:
    child = pl.DataFrame({"a": [1], "b": ["x"]})
    parent = pl.concat([PARENT, PARENT.head(1)])
    datasets = {
        "parent": {},
        "child": {
            "foreign_keys": [
                {"columns": ["a", "b"], "ref_dataset": "parent", "cardinality": "many_to_one"}
            ]
        },
    }
    report = _run({"parent": parent, "child": child}, datasets)
    result = _checks(report, "INP-005")["child(a, b) -> parent: parent keys not unique"]
    assert result.status is Status.FAIL and result.affected_count == 1


def test_one_to_one_requires_unique_child_keys() -> None:
    child = pl.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    report = _fk(child, cardinality="one_to_one")
    assert _checks(report, "INP-005")["child(a, b) -> parent: child keys not unique"].failed


# ------------------------------------------------------------------ selection typos
def test_dataset_selection_typos_fail_at_plan_time() -> None:
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {"counts": {}},
            "rules": {"INP-003": {"params": {"datasets": ["cuonts"]}}},
        }
    )
    with pytest.raises(ConfigurationError, match="cuonts"):
        QARunner(settings, REGISTRY).plan()
