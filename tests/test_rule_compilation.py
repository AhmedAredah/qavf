"""Rule compilation owns diagnostics without loading or partly publishing a plan."""

from dataclasses import FrozenInstanceError

import pytest

from qavf.engine.compilation import (
    ConfigurationError,
    RulePlan,
    compile_rules,
    plugin_status,
)
from qavf.engine.registry import Finding, Rule, RuleParams, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.settings import QAVFSettings, RuleSettings
from qavf.sources.base import SourceRegistry


class RequiredParams(RuleParams):
    threshold: int


class TargetParams(RuleParams):
    limit: int = 1


class NeedsConfiguration(Rule[RequiredParams]):
    rule_id = "A-NEEDS"
    name = "Needs configuration"
    description = "Required parameter fixture"
    category = "compilation"
    Params = RequiredParams

    def evaluate(self, ctx):
        raise AssertionError("Compilation must not execute a rule")


class OptionalRule(Rule):
    rule_id = "B-OPTIONAL"
    name = "Optional rule"
    description = "Exclusion fixture"
    category = "compilation"

    def evaluate(self, ctx):
        raise AssertionError("Compilation must not execute a rule")


class TargetRule(Rule[TargetParams]):
    rule_id = "Z-TARGET"
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
        {"plugins": {"entry_points": False}, "rules": {TargetRule.rule_id: {}}}
    )
    return settings, registry, SourceRegistry()


def test_compilation_returns_owned_immutable_diagnostics_without_reading_sources(monkeypatch):
    settings, registry, sources = inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError("Compilation must not build or load sources")

    monkeypatch.setattr(SourceRegistry, "build", forbidden)
    compiled = compile_rules(settings, registry, sources)
    assert tuple(rule.rule_id for rule in compiled.rules) == ("B-OPTIONAL", "Z-TARGET")
    assert compiled.needs_configuration == {"A-NEEDS": ("threshold",)}
    assert compiled.excluded == {}
    with pytest.raises(TypeError):
        compiled.needs_configuration["A-NEEDS"] = ("changed",)
    with pytest.raises(TypeError):
        compiled.excluded["B-OPTIONAL"] = "changed"
    with pytest.raises(FrozenInstanceError):
        compiled.rules = ()
    settings.engine.run_unconfigured_rules = False
    newer = compile_rules(settings, registry, sources)
    assert newer.needs_configuration == {}
    assert compiled.needs_configuration == {"A-NEEDS": ("threshold",)}


def test_plan_owns_diagnostics_instead_of_borrowing_constructor_mappings():
    needed = {"A-NEEDS": ("threshold",)}
    excluded = {"B-OPTIONAL": "disabled"}
    plan = RulePlan((), needed, excluded)
    needed.clear()
    excluded["B-OPTIONAL"] = "changed"
    assert plan.needs_configuration == {"A-NEEDS": ("threshold",)}
    assert plan.excluded == {"B-OPTIONAL": "disabled"}


@pytest.mark.parametrize("previous_success", [False, True])
@pytest.mark.parametrize("failure", ["parameters", "configuration"])
def test_failed_compilation_never_publishes_partial_runner_diagnostics(previous_success, failure):
    settings, registry, sources = inputs()
    runner = QARunner(settings, registry, sources)
    if previous_success:
        assert len(runner.plan()) == 2
    previous_needed = dict(runner.needs_configuration)
    previous_excluded = dict(runner.excluded)
    settings.rules["B-OPTIONAL"] = RuleSettings(enabled=False)
    settings.rules["Z-TARGET"] = RuleSettings(
        params={"limit": "invalid" if failure == "parameters" else -1}
    )
    with pytest.raises(ConfigurationError, match="Z-TARGET"):
        runner.plan()
    assert runner.needs_configuration == previous_needed
    assert runner.excluded == previous_excluded
    settings.rules["Z-TARGET"] = RuleSettings(params={"limit": 2})
    settings.engine.run_unconfigured_rules = False
    assert [rule.rule_id for rule in runner.plan()] == ["Z-TARGET"]
    assert runner.needs_configuration == {}
    assert set(runner.excluded) == {"A-NEEDS", "B-OPTIONAL"}


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
