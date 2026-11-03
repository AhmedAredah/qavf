"""Named check items: per-item identity, load-time validation, plugins and reports."""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.compilation import ConfigurationError
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleParams, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.reports.models import RuleResult
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

SERIES = {"value": "aadt", "group_by": ["site_id"], "time_column": "year"}
COUNTS = pl.DataFrame(
    {
        "site_id": ["S1", "S1", "S1", "S2", "S2"],
        "year": [2016, 2017, 2019, 2016, 2019],
        "aadt": [100, 110, 120, 50, 60],
    }
)


def _settings(raw: dict[str, Any]) -> QAVFSettings:
    return QAVFSettings.model_validate({"engine": {"parallel": False}, **raw})


def _counts(*checks: dict[str, Any]) -> dict[str, Any]:
    return {"datasets": {"counts": {"checks": list(checks)}}}


def test_two_items_of_one_check_keep_their_own_severity_and_name() -> None:
    settings = _settings(
        _counts(
            {"check": "missing_periods", "series": SERIES, "expected": [2016, 2017]},
            {
                "check": "missing_periods",
                "name": "no missing years",
                "series": SERIES,
                "expected": [2016, 2017, 2018, 2019],
                "severity": "CRITICAL",
                "description": "every year since 2016",
            },
        )
    )
    report = QARunner(settings, registry=discovered_rules()).run(frames={"counts": COUNTS})
    results = report.results_for("TMP-004")
    assert [(r.item_name, r.severity.value, r.status.value) for r in results] == [
        ("missing_periods on counts", "ERROR", "FAIL"),
        ("no missing years", "CRITICAL", "FAIL"),
    ]
    assert results[1].description == "every year since 2016"
    assert results[0].label == "missing_periods · counts (TMP-004)"
    assert results[1].label == "missing_periods · counts · no missing years (TMP-004)"
    assert [rule.item_name for rule in report.metadata.rules if rule.rule_id == "TMP-004"] == [
        "missing_periods on counts",
        "no missing years",
    ]


def test_disabled_items_do_not_run_and_fail_a_requirement() -> None:
    settings = _settings(
        {
            **_counts(
                {"check": "missing_periods", "name": "years", "series": SERIES, "enabled": False}
            ),
            "run": {"require": ["years"]},
        }
    )
    runner = QARunner(settings, registry=discovered_rules())
    report = runner.run(frames={"counts": COUNTS})
    assert not [
        r for r in report.results if r.rule_id == "TMP-004" and r.check != "required: years"
    ]
    (required,) = [r for r in report.results if r.check == "required: years"]
    assert required.status.value == "ERROR" and "disabled" in required.message
    assert runner.excluded == {"years": "disabled in the configuration"}


def test_dataset_requirements_cover_every_check_on_the_dataset() -> None:
    settings = _settings(
        {
            **_counts({"check": "expression", "expr": "aadt > 0"}),
            "run": {"require": ["counts", "expression on counts"]},
        }
    )
    report = QARunner(settings, registry=discovered_rules()).run(frames={"counts": COUNTS})
    assert not [r for r in report.results if (r.check or "").startswith("required:")]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            _counts({"check": "change_rate", "series": SERIES, "max": -1}),
            "datasets.counts.checks[0].max: Input should be greater than 0",
        ),
        (
            _counts({"check": "change_rate", "series": {**SERIES, "valu": "x"}}),
            "datasets.counts.checks[0].series.valu",
        ),
        (
            _counts({"check": "missing_periods", "series": SERIES, "expectd": [1]}),
            "datasets.counts.checks[0].expectd",
        ),
        (
            _counts({"check": "change_rate", "series": "aadt_by_site"}),
            "datasets.counts.checks[0].series: unknown series 'aadt_by_site'",
        ),
        (
            _counts({"check": "outliers", "column": "aadt", "where": "year > 2000"}),
            "datasets.counts.checks[0].where: 'where' is not supported by the outliers check",
        ),
        (
            _counts({"check": "coverage", "entity": "site_id"}),
            "datasets.counts.checks[0].check: coverage goes in the top-level checks list",
        ),
        (
            {"datasets": {"a": {}, "b": {}}, "checks": [{"check": "fresh", "column": "x"}]},
            "checks[0].check: fresh goes in a dataset's checks list",
        ),
        (
            _counts(
                {"check": "expression", "expr": "a > 0", "name": "x"},
                {"check": "expression", "expr": "b", "name": "x"},
            ),
            "datasets.counts.checks[1].name: 'x' is already used in this list",
        ),
        (
            {"datasets": {"a": {}}, "checks": [{"check": "sql", "sql": "SELECT 1"}]},
            "checks[0].name: a sql check needs a name",
        ),
        (
            {**_counts(), "run": {"require": ["nothing"]}},
            "run.require[0]: 'nothing' matches no check name or dataset",
        ),
        (
            {
                "datasets": {"a": {}, "b": {}},
                "checks": [
                    {
                        "check": "totals_match",
                        "datasets": ["a", "b"],
                        "sides": {"a": {"keys": ["k"]}},
                    }
                ],
            },
            "checks[0].sides: a: keys must list 0 column(s)",
        ),
        (
            {
                "datasets": {"a": {}, "b": {}, "c": {}},
                "checks": [{"check": "compare", "datasets": ["a", "b", "c"], "keys": ["k"]}],
            },
            "checks[0].datasets: relating more than two datasets in one check arrives in a "
            "later version of QAVF",
        ),
        (
            {
                "datasets": {"a": {}, "b": {}},
                "checks": [{"check": "drift", "datasets": ["a", "z"], "column": "x"}],
            },
            "checks[0].datasets: unknown dataset 'z'",
        ),
        (
            {
                "datasets": {"a": {}, "b": {}},
                "checks": [
                    {
                        "check": "drift",
                        "datasets": ["a", "b"],
                        "column": "x",
                        "strategy": "all_pairs",
                    }
                ],
            },
            "checks[0].strategy: strategy 'all_pairs' arrives in a later version",
        ),
    ],
)
def test_invalid_items_fail_when_the_configuration_is_loaded(
    raw: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError) as error:
        QAVFSettings.model_validate(raw)
    assert message in str(error.value)
    assert "phase" not in str(error.value).lower()


class _PluginParams(RuleParams):
    limit: int


class PluginCheckRule(Rule):
    rule_id: ClassVar[str] = "ACME-901"
    check_name: ClassVar[str] = "acme_limit"
    name: ClassVar[str] = "Acme limit"
    description: ClassVar[str] = "Values stay below the limit."
    category: ClassVar[str] = "custom"
    Params: ClassVar[type[RuleParams]] = _PluginParams

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.ok(f"limit {self.params.limit}")


class UnnamedRule(Rule):
    rule_id: ClassVar[str] = "ACME-902"
    name: ClassVar[str] = "Unnamed"
    description: ClassVar[str] = "No check name."
    category: ClassVar[str] = "custom"

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.ok("ok")


class ClashingRule(UnnamedRule):
    rule_id: ClassVar[str] = "ACME-903"
    check_name: ClassVar[str] = "missing_periods"


def _plugin_registry() -> RuleRegistry:
    registry = discovered_rules(entry_points=False)
    registry.register(PluginCheckRule)
    return registry


def test_plugin_checks_validate_when_the_run_is_planned() -> None:
    settings = _settings(_counts({"check": "acme_limit", "limit": "high"}))
    with pytest.raises(ConfigurationError, match=r"datasets\.counts\.checks\[0\]\.limit"):
        QARunner(settings, registry=_plugin_registry()).plan()
    with pytest.raises(ConfigurationError, match="unknown check 'acme_limit'"):
        QARunner(settings, registry=discovered_rules(entry_points=False)).plan()
    good = _settings(_counts({"check": "acme_limit", "limit": 3, "severity": "INFO"}))
    report = QARunner(good, registry=_plugin_registry()).run(frames={"counts": COUNTS})
    (result,) = report.results_for("ACME-901")
    assert (result.check_name, result.item_name, result.severity.value) == (
        "acme_limit",
        "acme_limit on counts",
        "INFO",
    )


def test_rules_need_a_unique_check_name() -> None:
    with pytest.raises(TypeError, match="check_name"):
        RuleRegistry().register(UnnamedRule)
    with pytest.raises(ValueError, match="duplicate check name 'missing_periods'"):
        discovered_rules(entry_points=False).register(ClashingRule)


def test_only_configured_checks_run() -> None:
    settings = _settings({"datasets": {"counts": {"schema": {"columns": {"aadt": {"ge": 0}}}}}})
    report = QARunner(settings, registry=discovered_rules()).run(frames={"counts": COUNTS})
    assert {r.rule_id for r in report.results} == {
        "INP-001",
        "INP-002",
        "INP-003",
        "INP-004",
        "INP-005",
        "INP-009",
    }
    assert {r.check_name for r in report.results} == {
        "datasets_available",
        "schema",
        "column_limits",
        "unique_keys",
        "references",
        "not_applicable",
    }


def test_reports_written_before_named_checks_still_load() -> None:
    old = {
        "rule_id": "TMP-004",
        "name": "No missing periods",
        "description": "d",
        "category": "temporal",
        "check": "counts.aadt",
        "status": "FAIL",
        "severity": "ERROR",
        "message": "m",
    }
    result = RuleResult.model_validate(old)
    assert result.check_name is None and result.item_name is None
    assert result.label == "TMP-004 [counts.aadt]"
