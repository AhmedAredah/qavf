"""Public plans are data-free and named profiles preserve explicit QA contracts."""

from contextlib import contextmanager

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import BatchSource, Dataset, File, Loader, Query, explain, validate

CONFIG = {"plugins": {"entry_points": False}, "engine": {"parallel": False}}
SCHEMA = {"columns": {"x": {"ge": 0}}}


def test_explain_does_not_read_files_or_invoke_loaders(tmp_path):
    def forbidden():
        raise AssertionError("loader must not execute while planning")

    for source in (Loader(forbidden), File(tmp_path / "missing.csv"), Query("select 1 as x")):
        result = explain(source, schema=SCHEMA, config=CONFIG, profile="validity")
        assert result.complete and not result.data_read
        assert result.datasets[0].observed_schema is None
        assert result.datasets[0].source_evaluations is None
        assert result.datasets[0].declared_columns == ("x",)
        # Schema contracts always run; nothing else is configured.
        assert {rule.rule_id for rule in result.rules} == {
            "INP-001",
            "INP-002",
            "INP-003",
            "INP-004",
            "INP-005",
            "INP-009",
        }


def test_explain_does_not_open_batch_factory():
    @contextmanager
    def forbidden():
        raise AssertionError("stream must not be opened")
        yield  # pragma: no cover

    result = explain(BatchSource(forbidden, {"x": pl.Int64}), schema=SCHEMA, config=CONFIG)
    assert not result.data_read
    assert result.datasets[0].input_kind == "BatchSource"


def test_explain_does_not_consume_arrow_interchange():
    class NotConsumed:
        def __arrow_c_stream__(self, requested_schema=None):
            raise AssertionError("interchange must not be consumed")

    result = explain(NotConsumed(), schema=SCHEMA, config=CONFIG)
    assert not result.data_read


def test_explain_does_not_import_configured_plugin_code(tmp_path):
    marker = tmp_path / "marker"
    plugin = tmp_path / "plugin.py"
    plugin.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    result = explain(
        Loader(lambda: None),
        schema=SCHEMA,
        config={
            "plugins": {"entry_points": False, "paths": [plugin]},
            "checks": [{"check": "custom_001"}],
        },
    )
    assert not result.complete
    assert any("custom_001" in reason for reason in result.unresolved)
    assert not marker.exists()


def test_plan_is_immutable_and_roundtrips():
    result = explain(Loader(lambda: None), schema=SCHEMA, config=CONFIG, profile="validity")
    with pytest.raises(ValidationError):
        result.complete = False
    with pytest.raises(TypeError):
        result.resources["memory_budget_bytes"] = 0
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_validity_profile_does_not_infer_a_contract():
    with pytest.raises(ValueError, match="explicit dataset QA contract"):
        validate(pl.DataFrame({"x": [1]}), config=CONFIG, profile="validity")


@pytest.mark.parametrize("profile", ["regression", "change"])
def test_analysis_profile_requires_explicit_comparison_before_loader_runs(profile):
    def forbidden():
        raise AssertionError("invalid configuration must be rejected before loading")

    with pytest.raises(ValueError, match="requires a configured"):
        validate(Loader(forbidden), schema=SCHEMA, config=CONFIG, profile=profile)


def test_change_profile_checks_invalid_but_unchanged_data():
    frame = pl.DataFrame({"id": [1], "x": [-1]})
    schema = {"primary_key": ["id"], **SCHEMA}
    config = CONFIG | {"checks": [{"check": "compare", "datasets": ["a", "b"], "keys": ["id"]}]}
    report = validate(
        {"a": Dataset(frame, schema=schema), "b": Dataset(frame, schema=schema)},
        config=config,
        profile="change",
    )
    assert report.verdict() == "failed"
    assert any(
        result.rule_id == "INP-003" and result.status.value == "FAIL" for result in report.results
    )
    assert report.metadata.config["extensions"]["qavf.profile"]["name"] == "change"


def test_profile_preserves_configuration_object():
    from copy import deepcopy

    before = deepcopy(CONFIG)
    result = validate(pl.DataFrame({"x": [1]}), schema=SCHEMA, config=CONFIG, profile="validity")
    assert result.verdict() == "passed"
    assert before == CONFIG


@pytest.fixture
def registered_rule(monkeypatch):
    from qavf import DataRequirements, Finding, Rule
    from qavf.engine.registry import default_registry

    # Keep the public process registry shared with the runner, but isolate this
    # test's registrations from other plugin tests and restore them afterwards.
    monkeypatch.setattr(default_registry, "_rules", {})
    monkeypatch.setattr(default_registry, "_origins", {})
    hooks = []

    class RegisteredRule(Rule):
        rule_id = "CUSTOM-900"
        check_name = "custom_900"
        name = "Registered check"
        description = "A process-registered check without a discovery module."
        category = "input"

        def configuration_errors(self, settings):
            hooks.append("configuration")
            return []

        def data_requirements(self, settings):
            hooks.append("dependencies")
            return DataRequirements(frozenset(settings.datasets))

        def evaluate(self, ctx):
            return Finding.fail("registered failure")

    default_registry.register(RegisteredRule)
    return hooks, RegisteredRule, default_registry


def test_explain_discloses_registered_rules_without_running_configuration_hooks(registered_rule):
    hooks, rule, _ = registered_rule
    config = CONFIG | {"checks": [{"check": rule.check_name}]}
    plan = explain(Loader(lambda: None), schema=SCHEMA, config=config)
    assert not plan.complete
    assert any(rule.check_name in reason for reason in plan.unresolved)
    assert rule.rule_id not in {entry.rule_id for entry in plan.rules}
    assert hooks == []

    resolved = explain(Loader(lambda: None), schema=SCHEMA, config=config, load_plugins=True)
    assert resolved.complete
    assert rule.rule_id in {entry.rule_id for entry in resolved.rules}
    assert hooks == ["configuration", "dependencies"]


@pytest.mark.parametrize(
    "engine,profile",
    [
        ({}, None),
        ({"max_workers": 1}, None),
        ({}, "validity"),
    ],
)
def test_explain_can_resolve_when_registered_default_is_not_selected(
    registered_rule, engine, profile
):
    hooks, rule, _ = registered_rule
    config = CONFIG | {"engine": {"parallel": False, **engine}}
    plan = explain(Loader(lambda: None), schema=SCHEMA, config=config, profile=profile)
    assert plan.complete
    assert not any(rule.rule_id in reason for reason in plan.unresolved)
    assert hooks == []


def test_registered_builtin_collision_is_unresolved_even_if_rule_not_selected(registered_rule):
    hooks, rule, registry = registered_rule
    registry._rules.clear()
    rule.rule_id = "INP-003"
    registry.register(rule)
    plan = explain(
        Loader(lambda: None),
        schema=SCHEMA,
        config=CONFIG | {"engine": {}},
    )
    assert not plan.complete
    assert any("INP-003" in reason and "conflicts" in reason for reason in plan.unresolved)
    assert hooks == []
