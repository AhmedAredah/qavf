"""Parser-proven SQL keeps native projections and components narrow (EP-04).

A row expression or ``where`` condition reads its own dataset's columns and,
through subqueries, the datasets it names. Only those are retained or joined
into its component; SQL whose reads cannot be proven keeps the conservative
plan (one component, every column) and says why.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

import qavf
from qavf.engine.planning import build_plan
from qavf.rules.input_checks.expressions import ExpressionRule
from qavf.settings import QAVFSettings
from qavf.sql_dependencies import PredicateDependencies, SqlDependencies

NAMES = ("a", "b", "c")


def _frames() -> dict[str, pa.Table]:
    return {name: pa.table({"id": [1, 2], "unused_payload": ["x", "y"]}) for name in NAMES}


def _datasets() -> dict[str, Any]:
    return {name: {"schema": {"columns": {"id": {"dtype": "int"}}}} for name in NAMES}


def _retained(report: Any) -> dict[str, list[str]]:
    return {info.name: info.notes["retained_columns"] for info in report.metadata.datasets}


def _validate(datasets: dict[str, Any], evidence: int = 0, frames: Any = None) -> Any:
    return qavf.validate(
        frames if frames is not None else _frames(),
        config={"datasets": datasets, "engine": {"max_affected_records": evidence}},
        backend="duckdb",
    )


@pytest.mark.parametrize(
    ("predicate", "columns", "datasets"),
    [
        ("id >= 0", {"id"}, set()),
        ("a.id >= 0 AND lower(name) = 'x'", {"a", "id", "name"}, set()),
        ("year IN (SELECT year FROM b)", None, {"b"}),
        ("EXISTS (SELECT 1 FROM B WHERE B.id = a.id)", None, {"b"}),
        ("id IN (SELECT id FROM a)", None, set()),
        ("id IN (WITH b AS (SELECT 1 AS id) SELECT id FROM b)", None, set()),
        ("COLUMNS('x.*') > 0", None, set()),
    ],
)
def test_predicate_reads_are_proven(predicate, columns, datasets):
    found = PredicateDependencies(NAMES).of(predicate, "a")
    assert found.proven
    assert found.columns == (None if columns is None else frozenset(columns))
    assert found.datasets == frozenset(datasets)


@pytest.mark.parametrize(
    ("predicate", "reason"),
    [
        ("id IN (SELECT id FROM read_csv('x.csv'))", "table function"),
        ("id IN (SELECT id FROM other.main.b)", "qualified relation"),
        ("id IN (SELECT id FROM temp.b)", "qualified relation"),
        ("id IN (SELECT id FROM nowhere)", "'nowhere', which is not a dataset"),
        ("id >= 0) ; SELECT (1", "could not be parsed"),
        ("id >= (", "could not be parsed"),
    ],
)
def test_unprovable_predicates_say_why(predicate, reason):
    found = PredicateDependencies(NAMES).of(predicate, "a")
    assert not found.proven
    assert reason in str(found.reason)


def test_a_local_expression_keeps_other_datasets_narrow_and_separate():
    baseline = _validate(_datasets())
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "name": "positive", "expr": "id >= 0"}]
    report = _validate(datasets)
    for run in (baseline, report):
        assert run.metadata.resource_usage["dependency_components"] == 3
        assert "projection_fallback" not in run.metadata.resource_usage
    assert _retained(report) == {"a": ["id"], "b": ["id"], "c": ["id"]}
    statuses = {r.status.value for r in report.results if r.rule_id == "INP-006"}
    assert statuses == {"PASS"}


def test_expression_columns_are_added_to_the_projection():
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "expr": "unused_payload <> 'z'"}]
    report = _validate(datasets)
    assert _retained(report)["a"] == ["id", "unused_payload"]
    assert _retained(report)["b"] == ["id"]


def test_expression_evidence_keeps_its_own_dataset_whole():
    """Expression evidence shows whole rows on both backends: only that dataset widens."""
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "expr": "id > 1"}]
    report = _validate(datasets, evidence=5)
    assert _retained(report) == {"a": ["id", "unused_payload"], "b": ["id"], "c": ["id"]}
    assert report.metadata.resource_usage["dependency_components"] == 3
    failed = next(r for r in report.results if r.check_id == "row.expression")
    assert failed.affected_records[0]["unused_payload"] == "x"


def test_a_subquery_joins_exactly_the_datasets_it_reads():
    frames = _frames()
    frames["b"] = pa.table({"id": [1], "unused_payload": ["x"]})
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "expr": "id IN (SELECT id FROM b)"}]
    report = _validate(datasets, frames=frames)
    assert report.metadata.resource_usage["dependency_components"] == 2
    retained = _retained(report)
    # The subquery may bind any column of either side: both stay whole.
    assert retained["a"] == retained["b"] == ["id", "unused_payload"]
    assert retained["c"] == ["id"]
    failed = next(r for r in report.results if r.check_id == "row.expression")
    assert (failed.status.value, failed.affected_count) == ("FAIL", 1)


def test_a_where_subquery_on_a_column_check_joins_its_dataset():
    datasets = _datasets()
    datasets["a"]["schema"]["columns"]["id"]["conditions"] = [
        {"where": "id IN (SELECT id FROM c)", "ge": 0}
    ]
    report = _validate(datasets)
    assert report.metadata.resource_usage["dependency_components"] == 2
    assert _retained(report)["b"] == ["id"]


@pytest.mark.parametrize(
    "expr",
    ["id IN (SELECT id FROM nowhere_view)", "id IN (SELECT 1 FROM range(3))"],
)
def test_unprovable_sql_keeps_the_conservative_plan(expr):
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "expr": expr}]
    report = _validate(datasets)
    usage = report.metadata.resource_usage
    assert usage["dependency_components"] == 1
    assert usage["projection_fallback"] and "a:" in usage["projection_fallback"][0]
    assert all(columns == ["id", "unused_payload"] for columns in _retained(report).values())


def test_explain_shows_retained_columns_and_fallback_reasons():
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "expr": "id IN (SELECT id FROM b)"}]
    plan = qavf.explain(
        _frames(),
        config={"datasets": datasets, "engine": {"max_affected_records": 0}},
        backend="duckdb",
    )
    assert plan.projection_fallback == ()
    by_name = {dataset.name: dataset for dataset in plan.datasets}
    assert by_name["c"].retained_columns == ("id",) and by_name["c"].component == ("c",)
    assert by_name["b"].retained_columns is None
    assert by_name["b"].retention == "a subquery of a reads it"
    assert by_name["a"].component == ("a", "b")
    datasets["a"]["checks"] = [{"check": "expression", "expr": "id IN (SELECT 1 FROM range(2))"}]
    plan = qavf.explain(_frames(), config={"datasets": datasets}, backend="duckdb")
    assert "table function" in plan.projection_fallback[0]
    assert all(dataset.retained_columns is None for dataset in plan.datasets)


def _reads_nothing(self: Any, predicate: str, dataset: str) -> SqlDependencies:
    """An analysis that wrongly proves a predicate reads no column."""
    return SqlDependencies()


def test_a_column_the_plan_missed_is_an_error_never_a_pass(monkeypatch):
    """Should the analysis ever under-report, the expression fails to bind and is
    reported not assessed, naming the column, rather than silently passing."""
    monkeypatch.setattr(PredicateDependencies, "of", _reads_nothing)
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "expr": "unused_payload = 'x'"}]
    report = _validate(datasets)
    assert _retained(report)["a"] == ["id"]
    result = next(r for r in report.results if r.check_id == "row.expression")
    assert result.status.value == "ERROR"
    assert "unused_payload" in result.message


def _expression_rule(datasets: dict[str, Any]) -> tuple[ExpressionRule, QAVFSettings]:
    settings = QAVFSettings.model_validate({"datasets": datasets})
    return ExpressionRule(), settings


def test_polars_expression_lifetimes_follow_proven_reads():
    datasets: dict[str, Any] = {name: {"path": f"{name}.csv"} for name in NAMES}
    datasets["a"]["checks"] = [{"check": "expression", "expr": "id IN (SELECT id FROM b)"}]
    rule, settings = _expression_rule(datasets)
    requirements = rule.data_requirements(settings)
    assert requirements is not None and requirements.datasets == frozenset({"a", "b"})
    (described,) = build_plan([rule], settings, NAMES).describe()
    assert described["dependency_mode"] == "declared"
    datasets["c"]["checks"] = [{"check": "expression", "expr": "id IN (SELECT 1 FROM range(2))"}]
    rule, settings = _expression_rule(datasets)
    assert rule.data_requirements(settings) is None


def test_polars_reads_another_dataset_after_lifetime_planning():
    frames = _frames()
    frames["b"] = pa.table({"id": [1], "unused_payload": ["x"]})
    datasets = _datasets()
    datasets["a"]["checks"] = [{"check": "expression", "expr": "id IN (SELECT id FROM b)"}]
    reports = [
        qavf.validate(
            frames,
            config={"datasets": datasets, "engine": {"max_affected_records": 5}},
            backend=backend,
        )
        for backend in ("polars", "duckdb")
    ]
    found = [
        next(r for r in report.results if r.check_id == "row.expression") for report in reports
    ]
    assert [(r.status.value, r.affected_count) for r in found] == [("FAIL", 1)] * 2
    assert found[0].affected_records == found[1].affected_records


def test_explain_text_lists_what_each_dataset_reads(tmp_path):
    from qavf.cli import main

    config = tmp_path / "qavf.yaml"
    config.write_text(
        "datasets:\n"
        "  a:\n"
        "    path: a.csv\n"
        "    schema: {columns: {id: {dtype: int}}}\n"
        "    checks: [{check: expression, expr: 'id IN (SELECT id FROM b)'}]\n"
        "  b: {path: b.csv}\n"
        "  c:\n"
        "    path: c.csv\n"
        "    schema: {columns: {id: {dtype: int}}}\n"
    )
    output = tmp_path / "plan.txt"
    assert main(["explain", "-c", str(config), "--backend", "duckdb", "-o", str(output)]) == 0
    text = output.read_text()
    assert "reads: every column (a subquery of a reads it); loaded with a" in text
    assert "reads: id\n" in text
