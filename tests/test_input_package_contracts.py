"""Discovery and backend contracts for the modular input-rule package."""

from __future__ import annotations

import polars as pl
import pytest

from qavf import Status, validate
from qavf.engine.registry import RuleRegistry
from qavf.rules import input_checks
from qavf.testing import run_rules


def test_input_category_discovery_runs_all_exported_rules_once():
    registry = RuleRegistry()
    found = registry.discover_package("qavf.rules.input_checks")
    expected = {
        input_checks.RequiredDatasetsRule,
        input_checks.SchemaConformanceRule,
        input_checks.ColumnCheckRule,
        input_checks.DuplicateKeyRule,
        input_checks.ReferentialIntegrityRule,
        input_checks.ExpressionRule,
        input_checks.SqlAssertionRule,
    }
    assert len(found) == len(expected)
    assert set(found) == expected
    report = run_rules(
        {
            "child": pl.DataFrame({"id": [1, 2], "value": [1, 2]}),
            "parent": pl.DataFrame({"id": [1, 2]}),
        },
        registry=registry,
        rules={rule.rule_id: {} for rule in expected},
        datasets={
            "child": {
                "schema": {
                    "columns": {"id": {"dtype": "int"}, "value": {"ge": 0}},
                    "primary_key": ["id"],
                },
                "foreign_keys": [{"columns": ["id"], "ref_dataset": "parent"}],
                "expressions": [{"expr": "value >= id"}],
            },
            "parent": {"schema": {"columns": {"id": {"dtype": "int"}}}},
        },
        assertions=[{"name": "nonnegative", "sql": 'SELECT * FROM "child" WHERE value < 0'}],
        run_unconfigured_rules=False,
    )
    assert {result.rule_id for result in report.results} == {rule.rule_id for rule in expected}
    assert all(result.status is Status.PASS for result in report.results)


@pytest.mark.parametrize("backend", ["eager", "lazy", "native", "mixed"])
@pytest.mark.parametrize("limit", [0, 1])
def test_foreign_key_findings_preserve_counts_severity_and_bounded_evidence(
    backend, limit, tmp_path
):
    child = pl.DataFrame({"a": [1, 1, 2, None, None], "b": [10, 10, 20, 30, None]})
    parent = pl.DataFrame({"a": [1, 1, 3], "b": [10, 10, 30]})
    inputs = {"child": child, "parent": parent}
    if backend == "lazy":
        inputs = {name: frame.lazy() for name, frame in inputs.items()}
    elif backend in ("native", "mixed"):
        for name in ("child", "parent") if backend == "native" else ("parent",):
            path = tmp_path / f"{name}.parquet"
            inputs[name].write_parquet(path)
            inputs[name] = path
    report = validate(
        inputs,
        config={
            "datasets": {
                "child": {
                    "foreign_keys": [
                        {
                            "columns": ["a", "b"],
                            "ref_dataset": "parent",
                            "match": "full",
                            "cardinality": "one_to_one",
                            "severity": "WARNING",
                        }
                    ],
                },
                "parent": {},
            },
            "rules": {"INP-005": {}},
            "plugins": {"entry_points": False},
            "engine": {
                "parallel": False,
                "run_unconfigured_rules": False,
                "max_affected_records": limit,
            },
        },
    )
    results = report.results_for("INP-005")
    label = "child(a, b) -> parent"
    expected = {
        f"{label}: partly null keys": (1, 5, "rows", 1),
        f"{label}: parent keys not unique": (1, None, "keys", 2),
        f"{label}: child keys not unique": (1, None, "keys", 2),
        label: (1, 2, "keys", 1),
    }
    assert len(results) == len(expected), [result.message for result in results]
    for result in results:
        assert result.status is Status.FAIL, result.message
        assert result.severity.value == "WARNING"
        assert (
            result.affected_count,
            result.total_records,
            result.unit,
            result.affected_rows,
        ) == expected[result.check]
        assert len(result.affected_records) == limit
    assert [
        result.metrics["rows_exempt_null_keys"]
        for result in results
        if "rows_exempt_null_keys" in result.metrics
    ] == [{label: 1}]
