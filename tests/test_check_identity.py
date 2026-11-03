"""Machine check contracts survive summaries, statuses, budgets and round trips."""

from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.engine.severity import Severity, Status
from qavf.reports.models import QAReport, RuleResult
from qavf.settings import QAVFSettings


@pytest.mark.parametrize("status", list(Status))
def test_legacy_result_constructor_has_identity_for_every_status(status):
    result = RuleResult(
        rule_id="CUSTOM",
        name="Old plugin",
        description="",
        category="custom",
        status=status,
        severity=Severity.ERROR,
        message="",
        datasets=["d"],
    )
    assert result.check_id == "CUSTOM"
    assert result.scope == "dataset"
    assert result.identity.rule_id == "CUSTOM"
    assert result.columns == [] and result.parameters == {}


def test_summary_and_failure_have_structured_column_contracts():
    settings = QAVFSettings(
        plugins={"entry_points": False},
        datasets={
            "d": {
                "schema": {
                    "columns": {
                        "odd.name: x": {"dtype": "int", "ge": 0, "le": 5},
                        "b": {"dtype": "int", "ge": 0},
                    }
                }
            }
        },
    )
    report = QARunner(settings).run(frames={"d": pl.DataFrame({"odd.name: x": [7], "b": [2]})})
    results = [r for r in report.results if r.rule_id == "INP-003"]
    failure = next(r for r in results if r.status == Status.FAIL)
    assert failure.columns == ["odd.name: x"]
    assert failure.parameters["le"] == 5
    summary = next(r for r in results if r.status == Status.PASS)
    assert any(c.columns == ["b"] and c.parameters.get("ge") == 0 for c in summary.checks)
    assert QAReport.model_validate_json(report.model_dump_json()) == report


def test_identity_is_outcome_ledger_even_when_evidence_budget_is_zero():
    class Check(Rule):
        rule_id = "CUSTOM"
        check_name = "custom"
        name = description = "Plugin check"
        category = "custom"

        def evaluate(self, ctx):
            return Finding.fail(
                "message",
                check="anything",
                check_id="decimal.bound",
                columns=("x",),
                datasets=("d",),
                parameters={"ge": Decimal("1.2")},
                metrics={"discard": "optional"},
            )

    registry = RuleRegistry()
    registry.register(Check)
    settings = QAVFSettings(engine={"max_evidence_bytes": 0}, checks=[{"check": Check.check_name}])
    report = QARunner(settings, registry).run()
    result = report.results[0]
    assert result.parameters["ge"] == Decimal("1.2")
    assert result.metrics == {}
    assert result.identity.columns == ["x"]
    restored = QAReport.model_validate_json(report.model_dump_json())
    assert restored.results[0].parameters["ge"] == Decimal("1.2")
    assert b"".join(report.iter_json_chunks()) == report.model_dump_json().encode()


def test_duplicate_evidence_completeness_survives_zero_byte_budget():
    class Check(Rule):
        rule_id = "DUPLICATE"
        check_name = "duplicate"
        name = description = "Duplicate keys"
        category = "custom"

        def evaluate(self, ctx):
            return Finding.fail(
                "duplicate",
                affected=[{"x": 7}],
                affected_count=1,
                metrics={
                    "duplicate_positions_complete": False,
                    "duplicate_positions_truncated": True,
                },
            )

    registry = RuleRegistry()
    registry.register(Check)
    settings = QAVFSettings(engine={"max_evidence_bytes": 0}, checks=[{"check": Check.check_name}])
    report = QARunner(settings, registry).run()
    result = report.results[0]
    assert result.affected_count == 1
    assert result.evidence_complete is False and result.evidence_truncated
    assert result.metrics == {}
