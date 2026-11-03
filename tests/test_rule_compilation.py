"""Rule compilation owns diagnostics without loading or partly publishing a plan."""

from dataclasses import FrozenInstanceError

import pytest

from qavf.checks.items import CheckItem
from qavf.engine.compilation import (
    ConfigurationError,
    RulePlan,
    compile_rules,
    plugin_status,
)
from qavf.engine.registry import Finding, Rule, RuleParams, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceRegistry


class RequiredParams(RuleParams):
    threshold: int


class TargetParams(RuleParams):
    limit: int = 1


class NeedsConfiguration(Rule[RequiredParams]):
    rule_id = "A-NEEDS"
    check_name = "a_needs"
    name = "Needs configuration"
    description = "Required parameter fixture"
    category = "compilation"
    Params = RequiredParams

    def evaluate(self, ctx):
        raise AssertionError("Compilation must not execute a rule")


class OptionalRule(Rule):
    rule_id = "B-OPTIONAL"
    check_name = "b_optional"
    name = "Optional rule"
    description = "Exclusion fixture"
    category = "compilation"

    def evaluate(self, ctx):
        raise AssertionError("Compilation must not execute a rule")


class TargetRule(Rule[TargetParams]):
    rule_id = "Z-TARGET"
    check_name = "z_target"
    name = "Target rule"
    description = "Late validation fixture"
    category = "compilation"
    Params = TargetParams

    def configuration_errors(self, settings):
        return ["limit cannot be negative"] if self.params.limit < 0 else []

    def evaluate(self, ctx):
        return Finding.ok("Target ran")


def inputs():
    registry = RuleRegistry()
    for rule in (TargetRule, OptionalRule, NeedsConfiguration):
        registry.register(rule)
    settings = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "checks": [{"check": "z_target"}]}
    )
    return settings, registry, SourceRegistry()


def test_compilation_returns_owned_immutable_diagnostics_without_reading_sources(monkeypatch):
    settings, registry, sources = inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError("Compilation must not build or load sources")

    monkeypatch.setattr(SourceRegistry, "build", forbidden)
    compiled = compile_rules(settings, registry, sources)
    # Only configured checks run: unconfigured rules are not planned at all.
    assert tuple(rule.rule_id for rule in compiled.rules) == ("Z-TARGET",)
    assert compiled.rules[0].key == "checks[0]"
    assert compiled.excluded == {}
    with pytest.raises(TypeError):
        compiled.excluded["B-OPTIONAL"] = "changed"
    with pytest.raises(FrozenInstanceError):
        compiled.rules = ()


def test_plan_owns_diagnostics_instead_of_borrowing_constructor_mappings():
    excluded = {"b_optional": "disabled"}
    plan = RulePlan((), excluded)
    excluded["b_optional"] = "changed"
    assert plan.excluded == {"b_optional": "disabled"}


@pytest.mark.parametrize("previous_success", [False, True])
@pytest.mark.parametrize("failure", ["parameters", "configuration"])
def test_failed_compilation_never_publishes_partial_runner_diagnostics(previous_success, failure):
    settings, registry, sources = inputs()
    runner = QARunner(settings, registry, sources)
    if previous_success:
        assert len(runner.plan()) == 1
    previous_excluded = dict(runner.excluded)
    settings.checks = [
        CheckItem(check="b_optional", enabled=False),
        CheckItem.model_validate(
            {"check": "z_target", "limit": "invalid" if failure == "parameters" else -1}
        ),
    ]
    with pytest.raises(ConfigurationError, match=r"checks\[1\]"):
        runner.plan()
    assert runner.excluded == previous_excluded
    settings.checks[1] = CheckItem.model_validate({"check": "z_target", "limit": 2})
    assert [rule.rule_id for rule in runner.plan()] == ["Z-TARGET"]
    assert runner.excluded == {"b_optional": "disabled in the configuration"}


def test_supplied_empty_registries_do_not_trigger_discovery(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit empty registries must remain empty")

    monkeypatch.setattr(RuleRegistry, "discover", forbidden)
    monkeypatch.setattr(SourceRegistry, "discover", forbidden)
    runner = QARunner(QAVFSettings(), RuleRegistry(), SourceRegistry())
    assert runner.plan() == []
    assert runner.plugin_status() == {}


def test_compilation_keeps_required_plugin_components_separate_and_redacted():
    settings, registry, sources = inputs()
    registry.plugins["pack"] = "error: password=top-secret"
    sources.plugins["pack"] = "loaded"
    settings.plugins.required = ["pack"]
    with pytest.raises(ConfigurationError, match="rules:pack") as error:
        compile_rules(settings, registry, sources)
    assert "top-secret" not in str(error.value)
    assert "top-secret" not in plugin_status(registry, sources)["rules:pack"]
    settings.plugins.required = ["sources:pack"]
    assert compile_rules(settings, registry, sources).rules


def test_compiler_rechecks_live_mapping_keys_before_constructing_rules(monkeypatch):
    settings, registry, sources = inputs()
    settings.extensions["probe"] = {1: "private"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid configuration reached rule construction")

    monkeypatch.setattr(OptionalRule, "__init__", forbidden)
    with pytest.raises(ConfigurationError, match="mapping keys must be strings") as error:
        compile_rules(settings, registry, sources)
    assert "private" not in str(error.value)
