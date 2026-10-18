"""Severity vocabulary, rule registry, runner and data context."""

from __future__ import annotations

import textwrap
from typing import ClassVar

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleParams, RuleRegistry
from qavf.engine.runner import ConfigurationError, QARunner
from qavf.engine.severity import Severity, Status, highest_severity, meets_threshold
from qavf.testing import make_settings


# ------------------------------------------------------------------- severity
def test_severity_orders_by_rank_not_alphabetically() -> None:
    assert Severity.INFO < Severity.WARNING < Severity.ERROR < Severity.CRITICAL
    assert sorted([Severity.ERROR, Severity.CRITICAL, Severity.INFO])[-1] is Severity.CRITICAL
    assert Severity("warning") is Severity.WARNING
    with pytest.raises(ValueError):
        Severity("fatal")


def test_highest_severity_and_threshold() -> None:
    assert highest_severity([Severity.INFO, Severity.ERROR]) is Severity.ERROR
    assert highest_severity([]) is None
    assert meets_threshold(Severity.CRITICAL, Severity.ERROR)
    assert not meets_threshold(Severity.WARNING, Severity.ERROR)
    assert not meets_threshold(None, Severity.INFO)


# ------------------------------------------------------------- test rules
class _Params(RuleParams):
    limit: int = 1


class _AlwaysFails(Rule):
    rule_id: ClassVar[str] = "T-FAIL"
    name: ClassVar[str] = "Always fails"
    description: ClassVar[str] = "Test rule."
    category: ClassVar[str] = "test"
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = "Fix it."
    Params: ClassVar[type[RuleParams]] = _Params

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.fail("nope", affected=[{"i": i} for i in range(10)])


class _Crashes(Rule):
    rule_id: ClassVar[str] = "T-CRASH"
    name: ClassVar[str] = "Crashes"
    description: ClassVar[str] = "Test rule."
    category: ClassVar[str] = "test"
    default_severity: ClassVar[Severity] = Severity.INFO

    def evaluate(self, ctx: DataContext) -> Finding:
        raise RuntimeError("boom")


class _NeedsData(Rule):
    rule_id: ClassVar[str] = "T-NEEDS"
    name: ClassVar[str] = "Needs data"
    description: ClassVar[str] = "Test rule."
    category: ClassVar[str] = "other"
    requires: ClassVar[tuple[str, ...]] = ("absent",)

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.ok("fine")


class _MultiCheck(Rule):
    rule_id: ClassVar[str] = "T-MULTI"
    name: ClassVar[str] = "Several checks"
    description: ClassVar[str] = "Test rule."
    category: ClassVar[str] = "test"
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = "Look at the checks."

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        return [
            Finding.fail("a failed", check="a", severity=Severity.WARNING),
            Finding.fail("b failed", check="b"),
            Finding.ok("c passed", check="c"),
        ]


def _registry(*rules: type[Rule]) -> RuleRegistry:
    registry = RuleRegistry()
    for rule in rules:
        registry.register(rule)
    return registry


# ------------------------------------------------------------------- registry
def test_discovers_builtin_rules(registry: RuleRegistry) -> None:
    ids = {rule.rule_id for rule in registry}
    assert {f"INP-00{i}" for i in range(1, 8)} <= ids
    assert {"TMP-001", "SPA-001", "STA-001", "REG-001"} <= ids
    assert not any(rule_id.startswith("MOV") for rule_id in ids)  # domain rules are plugins
    assert all(not rule.__module__.rsplit(".", 1)[-1].startswith("_") for rule in registry)


def test_every_builtin_rule_is_documented_and_defaults_validate(registry: RuleRegistry) -> None:
    for rule in registry:
        assert rule.name and rule.description and rule.category, rule.rule_id
        assert rule.recommendation, f"{rule.rule_id} has no recommendation"
        rule()


def test_every_builtin_rule_skips_on_unrelated_data(registry: RuleRegistry) -> None:
    report = QARunner(make_settings(), registry).run(frames={"x": pl.DataFrame({"a": [1]})})
    assert {r.status for r in report.results} == {Status.SKIPPED}


def test_discovers_rules_from_plugin_path(tmp_path) -> None:
    (tmp_path / "my_rules.py").write_text(
        textwrap.dedent(
            """
            from qavf import Finding, Rule

            class Plugged(Rule):
                rule_id = "PLUG-1"
                name = "Plugged"
                description = "From a file."
                category = "plugin"

                def evaluate(self, ctx):
                    return Finding.ok("ok")
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "_private.py").write_text("raise RuntimeError('must not import')")
    registry = RuleRegistry()
    assert [rule.rule_id for rule in registry.discover_path(tmp_path)] == ["PLUG-1"]
    with pytest.raises(FileNotFoundError):
        registry.discover_path(tmp_path / "missing")


def test_register_rejects_duplicates_and_incomplete_rules() -> None:
    registry = _registry(_AlwaysFails)
    registry.register(_AlwaysFails)  # idempotent for the same class

    class Clash(_Crashes):
        rule_id: ClassVar[str] = "T-FAIL"

    with pytest.raises(ValueError, match="duplicate rule id"):
        registry.register(Clash)

    class NoMetadata(Rule):
        def evaluate(self, ctx: DataContext) -> Finding:
            return Finding.ok("x")

    with pytest.raises(TypeError, match="missing rule metadata"):
        registry.register(NoMetadata)
    with pytest.raises(TypeError, match="abstract"):
        registry.register(Rule)  # type: ignore[type-abstract]
    with pytest.raises(KeyError):
        registry.get("NOPE")
    assert _AlwaysFails({"limit": 3}).params.limit == 3
    with pytest.raises(ValueError):
        _AlwaysFails({"unknown": 1})


# --------------------------------------------------------------------- runner
def test_runner_turns_outcomes_into_results() -> None:
    settings = make_settings(max_affected_records=3)
    report = QARunner(settings, _registry(_AlwaysFails, _Crashes, _NeedsData)).run(frames={})

    failed = report.get("T-FAIL")
    assert failed.status is Status.FAIL and failed.severity is Severity.WARNING
    assert failed.affected_count == 10 and len(failed.affected_records) == 3

    crashed = report.get("T-CRASH")
    assert crashed.status is Status.ERROR and "boom" in crashed.message
    assert crashed.severity is Severity.ERROR  # escalated: unassessed is never harmless

    assert report.get("T-NEEDS").status is Status.SKIPPED
    summary = report.summary
    assert (summary.failed, summary.errored, summary.skipped, summary.passed) == (1, 1, 1, 0)
    assert report.exit_code(Severity.ERROR) == 1
    # The crashed rule left data unassessed: incomplete even when no failure meets the
    # threshold, and that fails the run unless fail_on_incomplete is switched off.
    assert report.verdict(Severity.CRITICAL) == "incomplete"
    assert report.exit_code(Severity.CRITICAL) == 1


def test_multi_check_rules_grade_each_check_separately() -> None:
    report = QARunner(make_settings(), _registry(_MultiCheck)).run(frames={})
    results = report.results_for("T-MULTI")
    assert [(r.check, r.status, r.severity) for r in results] == [
        ("a", Status.FAIL, Severity.WARNING),
        ("b", Status.FAIL, Severity.ERROR),
        ("c", Status.PASS, Severity.ERROR),
    ]
    assert results[0].label == "T-MULTI [a]"
    assert report.get("T-MULTI").check == "b"  # most severe failure
    assert report.get("T-MULTI", check="c").status is Status.PASS
    (advice,) = report.recommendations()
    assert advice.checks == ["b", "a"]


def test_plan_honours_overrides_and_filters() -> None:
    registry = _registry(_AlwaysFails, _Crashes, _NeedsData)
    rules = {
        "T-FAIL": {"severity": "critical", "params": {"limit": 5}},
        "T-CRASH": {"enabled": False},
    }
    plan = QARunner(make_settings(rules=rules), registry).plan()
    assert [rule.rule_id for rule in plan] == ["T-FAIL", "T-NEEDS"]
    assert plan[0].severity is Severity.CRITICAL and plan[0].params.limit == 5

    only = QARunner(make_settings(rules=rules, run_unconfigured_rules=False), registry)
    assert [rule.rule_id for rule in only.plan()] == ["T-FAIL"]
    included = QARunner(make_settings(include_categories=["other"]), registry)
    assert [rule.rule_id for rule in included.plan()] == ["T-NEEDS"]
    excluded = QARunner(make_settings(exclude_categories=["test"]), registry)
    assert [rule.rule_id for rule in excluded.plan()] == ["T-NEEDS"]


def test_plan_rejects_unknown_rules_and_bad_params() -> None:
    registry = _registry(_AlwaysFails)
    with pytest.raises(ConfigurationError, match="unknown rule id"):
        QARunner(make_settings(rules={"TYPO-1": {}}), registry).plan()
    with pytest.raises(ConfigurationError, match="T-FAIL"):
        QARunner(make_settings(rules={"T-FAIL": {"params": {"limit": "many"}}}), registry).plan()


def test_parallel_and_sequential_runs_agree(demo, registry: RuleRegistry) -> None:
    rules = {
        "TMP-001": {
            "params": {"series": [{"dataset": "counts", "value": "aadt", "group_by": ["site_id"]}]}
        },
        "SPA-003": {"params": {"entity_column": "site_id", "time_column": "year"}},
    }

    def outcome(parallel: bool) -> list[tuple[str, str | None, Status, str]]:
        settings = make_settings(rules=rules, parallel=parallel, max_workers=4)
        report = QARunner(settings, registry).run(frames=demo)
        assert report.metadata.parallel is parallel
        return [(r.rule_id, r.check, r.status, r.message) for r in report.results]

    assert outcome(parallel=True) == outcome(parallel=False)


# -------------------------------------------------------------------- context
def test_context_accepts_polars_pandas_and_arrow_and_normalises_columns() -> None:
    frame = pl.DataFrame({"siteid": ["a"], "YEAR": [2020], "v": [1.0]})
    ctx = DataContext(
        frames={"p": frame, "d": frame.to_pandas(), "a": frame.to_arrow()},
        canonical_columns=["siteID", "year"],
    )
    for name in ("p", "d", "a"):
        assert ctx.get(name).columns == ["siteID", "year", "v"]
    first_view = ctx.pandas("p")
    assert first_view.equals(ctx.pandas("p"))
    first_view.loc[0, "v"] = 999
    assert ctx.pandas("p").loc[0, "v"] == 1.0
    assert isinstance(ctx.pandas("p"), pd.DataFrame)
    assert ctx.has("p") and not ctx.has("missing")
    assert "not configured" in str(ctx.unavailable_reason("missing"))
    assert ctx.memo("k", lambda: 1) == 1 and ctx.memo("k", lambda: 2) == 1
    with pytest.raises(Exception, match="expected a DataFrame"):
        DataContext(frames={"bad": [1, 2]})
    assert isinstance(pa.table({"x": [1]}), pa.Table)


def test_context_sql_and_extensions() -> None:
    ctx = DataContext(
        frames={"a": pl.DataFrame({"k": [1, 2, 3]}), "b": pl.DataFrame({"k": [2]})},
        extensions={"myplugin": {"version": "2"}},
    )
    result = ctx.sql("SELECT k FROM a WHERE k NOT IN (SELECT k FROM b) ORDER BY k")
    assert result["k"].to_list() == [1, 3]
    assert ctx.referenced("select * from a join b using (k)") == ["a", "b"]
    assert ctx.referenced("select a_b from bb") == []
    assert ctx.extension("myplugin") == {"version": "2"} and ctx.extension("other") == {}


def test_dataset_rename_applies_after_loading() -> None:
    settings = make_settings(datasets={"t": {"rename": {"old": "new"}}})
    ctx = DataContext.from_settings(settings, frames={"t": pl.DataFrame({"old": [1]})})
    assert ctx.get("t").columns == ["new"]
