"""A filtered ``expression`` is assessed on the rows its ``where`` selects.

Review findings R4 (denominator), R5 (``min_applicable_rows``) and R8 (disabled
row-check stubs); both backends agree (see ``test_backend_report_parity``).
"""

from __future__ import annotations

import polars as pl
import pytest

from qavf import Status, validate
from qavf.settings import QAVFSettings

BACKENDS = ("duckdb", "polars")


def _expression_results(report):
    return [r for r in report.results if r.rule_id == "INP-006" and r.check_id == "row.expression"]


def _config(item, require=()):
    return {"datasets": {"d": {"checks": [item]}}, "run": {"require": list(require)}}


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_filtered_expression_counts_only_the_rows_it_applies_to(backend):
    frame = pl.DataFrame({"x": [5, 20] + [-1] * 99})
    item = {"check": "expression", "expr": "x < 10", "where": "x > 0"}
    report = validate({"d": frame}, config=_config(item), backend=backend)
    (result,) = _expression_results(report)
    assert result.status == Status.FAIL
    assert result.total_records == 2
    assert result.affected_count == 1
    assert result.message == "1 row of 2 matching 'where: x > 0' in d violates 'x < 10'."
    assert result.metrics["applicable_rows"] == 2
    assert result.metrics["excluded_rows"] == 99
    assert result.metrics["population_rows"] == 101
    assert result.check == "d: x < 10 where x > 0"
    assert "CASE" not in result.message and "CASE" not in (result.check or "")
    assert result.item_ids == ["datasets.d.checks[0]"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_null_scope_values_are_excluded(backend):
    frame = pl.DataFrame({"x": [None, 50, 1]}, schema={"x": pl.Int64})
    item = {"check": "expression", "name": "small", "expr": "x < 10", "where": "x > 0"}
    (result,) = _expression_results(validate({"d": frame}, config=_config(item), backend=backend))
    assert result.total_records == 2
    assert result.metrics["excluded_rows"] == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_too_few_applicable_rows_leave_a_required_expression_not_assessed(backend):
    frame = pl.DataFrame({"x": [-1, -2]})
    item = {
        "check": "expression",
        "name": "lt10",
        "expr": "x < 10",
        "where": "x > 0",
        "min_applicable_rows": 1,
    }
    report = validate({"d": frame}, config=_config(item, ["lt10"]), backend=backend)
    (result,) = _expression_results(report)
    assert result.status == Status.ERROR
    assert "fewer than min_applicable_rows 1" in result.message
    assert result.metrics["applicable_rows"] == 0
    assert result.item_ids == ["datasets.d.checks[0]"]
    assert report.verdict() != "passed"


@pytest.mark.parametrize("backend", BACKENDS)
def test_enough_applicable_rows_assess_the_expression(backend):
    frame = pl.DataFrame({"x": [1, -2]})
    item = {"check": "expression", "expr": "x < 10", "where": "x > 0", "min_applicable_rows": 1}
    report = validate({"d": frame}, config=_config(item), backend=backend)
    assert report.verdict() == "passed"


@pytest.mark.parametrize(
    "item",
    [
        {"check": "expression", "expr": "x < 10", "min_applicable_rows": 1},
        {"check": "expression", "expr": "x < 10", "where": "x > 0", "min_applicable_rows": True},
        {"check": "expression", "expr": "x < 10", "where": "x > 0", "min_applicable_rows": -1},
    ],
)
def test_min_applicable_rows_needs_where_and_a_count(item):
    with pytest.raises(ValueError, match="min_applicable_rows"):
        QAVFSettings.model_validate(_config(item))


@pytest.mark.parametrize(
    "config",
    [
        {"datasets": {"d": {"checks": [{"check": "expression", "enabled": False}]}}},
        {"checks": [{"check": "sql", "enabled": False}]},
    ],
)
def test_disabled_row_check_stubs_are_accepted_like_other_checks(config):
    QAVFSettings.model_validate(config)


def test_a_disabled_row_check_with_settings_is_still_validated():
    config = {"datasets": {"d": {"checks": [{"check": "expression", "enabled": False, "x": 1}]}}}
    with pytest.raises(ValueError, match="unknown setting"):
        QAVFSettings.model_validate(config)


def test_a_disabled_stub_runs_nothing():
    config = {"datasets": {"d": {"checks": [{"check": "expression", "enabled": False}]}}}
    report = validate({"d": pl.DataFrame({"x": [1]})}, config=config)
    assert not _expression_results(report)
