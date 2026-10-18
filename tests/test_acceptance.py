"""Fail-closed acceptance: no false pass and no lost evidence.

Each test reproduces a scenario that could report success without assessing the
data, or lose evidence of a failure, and pins the correct behaviour.
"""

from __future__ import annotations

import math
from typing import ClassVar

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import ConfigurationError, QARunner
from qavf.engine.severity import Severity, Status
from qavf.regression import compare_frames
from qavf.testing import make_settings, rule_results, run_rule

BIG = 2**53  # 9007199254740992: the first integer whose successor Float64 cannot represent


def _statuses(results) -> list[tuple[str | None, Status]]:
    return [(r.check, r.status) for r in results]


# =========================================================================
# Missing work must not produce a successful gate
# =========================================================================
def test_required_dataset_without_source_or_frame_fails(registry) -> None:
    datasets = {"t": {"required": True, "schema": {"columns": {"x": {"dtype": "float"}}}}}
    results = rule_results("INP-001", {}, datasets=datasets, registry=registry)
    (failed,) = [r for r in results if r.status is Status.FAIL]
    assert failed.check == "t" and failed.severity is Severity.CRITICAL
    # The schema rules report the required dataset as not assessed instead of ignoring it.
    schema = run_rule("INP-002", {}, datasets=datasets, registry=registry)
    assert schema.status is Status.ERROR and "t" in schema.message


def test_optional_absent_dataset_is_not_an_error(registry) -> None:
    datasets = {"t": {"required": False, "schema": {"columns": {"x": {}}}}}
    for rule_id in ("INP-001", "INP-002", "INP-003"):
        assert all(
            r.status is not Status.ERROR
            for r in rule_results(rule_id, {}, datasets=datasets, registry=registry)
        )


def test_unknown_category_filters_are_configuration_errors(registry) -> None:
    with pytest.raises(ConfigurationError, match="unknown categor"):
        QARunner(make_settings(include_categories=["audti"]), registry).plan()
    with pytest.raises(ConfigurationError, match="unknown categor"):
        QARunner(make_settings(exclude_categories=["nope"]), registry).plan()


class _Noop(Rule):
    rule_id: ClassVar[str] = "T-NOOP"
    name: ClassVar[str] = "Noop"
    description: ClassVar[str] = "Never applies."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.skip("nothing to do")


class _NeedsD(Rule):
    rule_id: ClassVar[str] = "T-NEEDS-D"
    name: ClassVar[str] = "Needs d"
    description: ClassVar[str] = "Requires dataset d."
    category: ClassVar[str] = "test"
    requires: ClassVar[tuple[str, ...]] = ("d",)

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.ok("fine")


def _registry(*rules: type[Rule]) -> RuleRegistry:
    registry = RuleRegistry()
    for rule in rules:
        registry.register(rule)
    return registry


def test_a_run_that_assessed_nothing_is_incomplete() -> None:
    report = QARunner(make_settings(), _registry(_Noop)).run(frames={})
    assert report.summary.assessed == 0
    assert report.verdict() == "incomplete" and report.exit_code() == 1
    empty = QARunner(make_settings(), RuleRegistry()).run(frames={})
    assert empty.verdict() == "incomplete" and empty.exit_code() == 1
    lenient = QARunner(make_settings(fail_on_incomplete=False), _registry(_Noop)).run(frames={})
    assert lenient.verdict() == "incomplete" and lenient.exit_code() == 0


def test_explicitly_configured_rule_with_missing_data_is_unassessed() -> None:
    registry = _registry(_NeedsD, _Noop)
    implicit = QARunner(make_settings(), registry).run(frames={}).get("T-NEEDS-D")
    assert implicit.status is Status.SKIPPED  # auto-discovered and inapplicable: fine
    settings = make_settings(rules={"T-NEEDS-D": {}})
    explicit = QARunner(settings, registry).run(frames={}).get("T-NEEDS-D")
    assert explicit.status is Status.ERROR and explicit.severity >= Severity.ERROR


def test_configured_series_that_cannot_be_assessed_are_reported(demo, registry) -> None:
    good = {"dataset": "counts", "value": "aadt", "group_by": ["site_id"], "time_column": "year"}
    missing = {**good, "dataset": "never_loaded"}
    results = rule_results("TMP-001", demo, {"series": [good, missing]}, registry=registry)
    by_status = {r.status for r in results}
    assert by_status == {Status.PASS, Status.ERROR}
    (blocked,) = [r for r in results if r.status is Status.ERROR]
    assert blocked.check == "never_loaded.aadt" and "never_loaded" in blocked.message


def test_coverage_with_an_unavailable_listed_dataset_is_unassessed(demo, registry) -> None:
    params = {"entity_column": "site_id", "datasets": ["counts", "never_loaded"]}
    statuses = {r.status for r in rule_results("SPA-003", demo, params, registry=registry)}
    assert Status.ERROR in statuses


@pytest.mark.parametrize(
    "values",
    [[float("nan")] * 10, [float("inf")] * 10, [1.0, 2.0]],
    ids=["all-nan", "all-inf", "too-small"],
)
def test_outlier_scan_with_nothing_testable_is_unassessed(values, registry) -> None:
    frame = pl.DataFrame({"x": values})
    result = run_rule(
        "STA-001", {"t": frame}, {"targets": [{"dataset": "t", "value": "x"}]}, registry=registry
    )
    assert result.status is Status.ERROR and result.total_records in (0, None)


def test_outlier_scan_reports_a_missing_target_beside_a_healthy_one(registry) -> None:
    frame = pl.DataFrame({"x": [float(i) for i in range(20)]})
    targets = [{"dataset": "t", "value": "x"}, {"dataset": "gone", "value": "x"}]
    results = rule_results("STA-001", {"t": frame}, {"targets": targets}, registry=registry)
    assert {r.status for r in results} == {Status.PASS, Status.ERROR}


def test_foreign_key_to_an_unavailable_dataset_is_unassessed(small, registry) -> None:
    datasets = {
        "child": {"foreign_keys": [{"columns": ["id"], "ref_dataset": "parent"}]},
        "parent": {"required": True},
    }
    result = run_rule("INP-005", {"child": small["child"]}, datasets=datasets, registry=registry)
    assert result.status is Status.ERROR


# =========================================================================
# A later missing dependency must not erase an earlier failure
# =========================================================================
def _expressions(*exprs: str, **dataset) -> dict:
    return {"t": {"expressions": [{"expr": e} for e in exprs]}, **dataset}


def test_string_literals_and_comments_are_not_dataset_dependencies(tmp_path, registry) -> None:
    frame = pl.DataFrame({"x": [-1], "label": ["fine"]})
    datasets = {
        **_expressions("x >= 0", "label != 'ghost'"),
        "ghost": {"path": str(tmp_path / "missing.csv"), "required": False},
    }
    results = rule_results("INP-006", {"t": frame}, datasets=datasets, registry=registry)
    assert _statuses(results) == [("t: x >= 0", Status.FAIL), (None, Status.PASS)]


def test_unavailable_dependency_marks_only_its_expression(tmp_path, registry) -> None:
    frame = pl.DataFrame({"x": [-1]})
    datasets = {
        **_expressions("x >= 0", "x IN (SELECT x FROM ghost)"),
        "ghost": {"path": str(tmp_path / "missing.csv"), "required": False},
    }
    results = rule_results("INP-006", {"t": frame}, datasets=datasets, registry=registry)
    assert _statuses(results) == [
        ("t: x >= 0", Status.FAIL),
        ("t: x IN (SELECT x FROM ghost)", Status.ERROR),
    ]


def test_sql_dataset_references_are_case_insensitive_and_parsed() -> None:
    ctx = DataContext(
        frames={"actual": pl.DataFrame({"k": [1]}), "other": pl.DataFrame({"k": [2]})}
    )
    assert ctx.referenced("SELECT * FROM ACTUAL") == ["actual"]
    assert ctx.referenced("SELECT 'other' AS label -- other\nFROM actual") == ["actual"]
    assert ctx.sql("SELECT k FROM ACTUAL")["k"].to_list() == [1]
    # A query DuckDB cannot bind without the tables still resolves its references.
    assert ctx.referenced(
        "WITH c AS (SELECT * FROM actual) SELECT * FROM c JOIN other USING (k)"
    ) == [
        "actual",
        "other",
    ]


def test_failing_assertion_dependencies_are_unassessed_not_failed(registry) -> None:
    assertions = [{"name": "uses ghost", "sql": "SELECT * FROM ghost"}]
    result = run_rule(
        "INP-007", {"t": pl.DataFrame({"x": [1]})}, assertions=assertions, registry=registry
    )
    assert result.status is Status.ERROR


# =========================================================================
# NaN and infinity have one explicit policy
# =========================================================================
def _column(values, **spec) -> tuple[dict, dict]:
    return {"t": pl.DataFrame({"x": values})}, {"t": {"schema": {"columns": {"x": spec}}}}


def _failed_checks(frames, datasets, registry) -> dict[str, object]:
    results = rule_results("INP-003", frames, datasets=datasets, registry=registry)
    return {r.check: r for r in results if r.status is Status.FAIL}


def test_nan_counts_as_missing(registry) -> None:
    frames, datasets = _column([1.0, math.nan], ge=0, le=10)
    failed = _failed_checks(frames, datasets, registry)
    assert set(failed) == {"t.x: not null"}
    assert "NaN" in failed["t.x: not null"].message

    frames, datasets = _column([1.0, math.nan], nullable=True, max_null_fraction=0)
    assert set(_failed_checks(frames, datasets, registry)) == {"t.x: max_null_fraction 0"}

    frames, datasets = _column([1.0, math.nan, 3.0], nullable=True, mean={"ge": 2, "le": 2})
    assert _failed_checks(frames, datasets, registry) == {}  # NaN ignored like null


def test_explicitly_forbidden_nan_is_enforced(registry) -> None:
    frames, datasets = _column([math.nan, 1.0], nullable=True, notin=[math.nan])
    failed = _failed_checks(frames, datasets, registry)
    (result,) = failed.values()
    assert result.affected_count == 1 and result.affected_records[0]["_row"] == 0


def test_infinity_is_rejected_unless_allowed(registry) -> None:
    frames, datasets = _column([1.0, math.inf, -math.inf])
    failed = _failed_checks(frames, datasets, registry)
    assert failed["t.x: finite"].affected_count == 2
    frames, datasets = _column([1.0, math.inf], allow_infinite=True)
    assert _failed_checks(frames, datasets, registry) == {}


# =========================================================================
# Large integers keep their identity and exactness
# =========================================================================
def test_foreign_keys_compare_large_integers_exactly(registry) -> None:
    child = pl.DataFrame({"id": [BIG + 1]}, schema={"id": pl.UInt64})
    parent = pl.DataFrame({"id": [BIG]}, schema={"id": pl.Int64})
    datasets = {
        "child": {"foreign_keys": [{"columns": ["id"], "ref_dataset": "parent"}]},
        "parent": {},
    }
    result = run_rule(
        "INP-005", {"child": child, "parent": parent}, datasets=datasets, registry=registry
    )
    assert result.status is Status.FAIL and result.affected_records[0]["id"] == BIG + 1


def test_foreign_keys_across_int_and_float(registry) -> None:
    datasets = {
        "child": {"foreign_keys": [{"columns": ["id"], "ref_dataset": "parent"}]},
        "parent": {},
    }
    exact = {"child": pl.DataFrame({"id": [1, 2]}), "parent": pl.DataFrame({"id": [1.0, 2.0]})}
    assert run_rule("INP-005", exact, datasets=datasets, registry=registry).status is Status.PASS
    lossy = {"child": pl.DataFrame({"id": [1]}), "parent": pl.DataFrame({"id": [1.5]})}
    result = run_rule("INP-005", lossy, datasets=datasets, registry=registry)
    assert result.status is Status.ERROR and "compare" in result.message
    text = {"child": pl.DataFrame({"id": [1]}), "parent": pl.DataFrame({"id": ["1"]})}
    assert run_rule("INP-005", text, datasets=datasets, registry=registry).status is Status.ERROR


@pytest.mark.parametrize(
    ("spec", "expect_fail"),
    [
        ({"nullable": True, "isin": [BIG]}, True),
        ({"nullable": True, "gt": BIG}, False),
        ({"nullable": True, "le": BIG}, True),
        ({"nullable": True, "sum": {"le": BIG}}, True),
        ({"nullable": True, "max": {"ge": BIG + 1, "le": BIG + 1}}, False),
    ],
)
def test_integer_limits_are_exact(spec, expect_fail, registry) -> None:
    frames = {"t": pl.DataFrame({"x": [BIG + 1, None]}, schema={"x": pl.Int64})}
    datasets = {"t": {"schema": {"columns": {"x": spec}}}}
    failed = _failed_checks(frames, datasets, registry)
    assert bool(failed) is expect_fail, failed


def test_regression_compares_large_integers_exactly() -> None:
    base = pl.DataFrame({"k": [1], "v": [BIG]})
    cand = pl.DataFrame({"k": [1], "v": [BIG + 1]})
    result = compare_frames(base, cand, ["k"], rtol=0, atol=0)
    assert not result.equal and result.values_changed == 1
    assert compare_frames(base, base, ["k"], rtol=0, atol=0).equal


def test_regression_keys_are_never_matched_lossily() -> None:
    base = pl.DataFrame({"k": [BIG + 1], "v": [1.0]})
    cand = pl.DataFrame({"k": [float(BIG)], "v": [1.0]})
    with pytest.raises(ValueError, match="key column 'k'"):
        compare_frames(base, cand, ["k"])
    distinct = pl.DataFrame({"k": [BIG, BIG + 1], "v": [1.0, 2.0]})
    assert compare_frames(distinct, distinct, ["k"]).equal  # distinct keys stay distinct


# =========================================================================
# Nested output changes are detected
# =========================================================================
def test_list_cell_changes_are_detected() -> None:
    base = pl.DataFrame({"k": [1, 2], "v": [[1, 2], [3]]})
    for changed in ([1, 2, 3], [1], [2, 1]):
        cand = pl.DataFrame({"k": [1, 2], "v": [changed, [3]]})
        result = compare_frames(base, cand, ["k"])
        assert not result.equal, changed
        assert (result.values_changed, result.rows_changed, result.rows_added) == (1, 1, 0)
        (difference,) = result.differences
        assert (
            difference.column == "v"
            and difference.baseline == [1, 2]
            and difference.candidate == changed
        )


def test_struct_field_changes_are_cell_changes_not_rows() -> None:
    base = pl.DataFrame({"k": [1], "s": [{"a": 1}]})
    cand = pl.DataFrame({"k": [1], "s": [{"a": 1, "b": 2}]})
    result = compare_frames(base, cand, ["k"])
    assert (result.values_changed, result.rows_added, result.rows_removed) == (1, 0, 0)
    assert result.differences[0].path == ["b"]


def test_nested_numbers_respect_tolerance_and_row_changes_combine() -> None:
    base = pl.DataFrame({"k": [1, 2], "v": [[1.0, 2.0], [3.0]]})
    cand = pl.DataFrame({"k": [1, 3], "v": [[1.0, 2.0 + 1e-12], [5.0]]})
    result = compare_frames(base, cand, ["k"])
    assert (result.values_changed, result.rows_added, result.rows_removed) == (0, 1, 1)


def test_incomplete_runs_are_visible_in_cli_and_html(tmp_path, capsys) -> None:
    import yaml

    from qavf.cli import EXIT_OK, EXIT_QA_FAILED, main
    from qavf.reports.html import render_html
    from qavf.settings import load_settings

    config = {
        "rules": {"TMP-001": {"params": {"series": [{"dataset": "gone", "value": "v"}]}}},
        "engine": {"include_categories": ["temporal"], "parallel": False},
        "plugins": {"entry_points": False},
        "report": {"output_dir": "out"},
    }
    path = tmp_path / "qavf.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "-c", str(path)]) == EXIT_QA_FAILED
    assert "QA INCOMPLETE: 1 configured check(s) could not be assessed" in capsys.readouterr().out
    report = QARunner(load_settings(path)).run()
    assert "QA incomplete" in render_html(report)
    assert report.model_dump(mode="json")["summary"]["assessed"] == 0

    config["engine"]["fail_on_incomplete"] = False
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "-c", str(path)]) == EXIT_OK
    assert "QA INCOMPLETE" in capsys.readouterr().out
