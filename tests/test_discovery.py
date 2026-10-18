"""Public contracts across discovery, normalization and project generation."""
from __future__ import annotations

import importlib
import json
import math
import threading
from decimal import Decimal
from enum import StrEnum
from types import SimpleNamespace
from typing import ClassVar, Literal

import jsonschema
import polars as pl
import pytest
import yaml
from pydantic import AliasChoices, Field, ValidationError, model_validator

from qavf import Finding, QARunner, QAVFSettings, Rule, RuleParams, RuleRegistry, validate
from qavf.cli import EXIT_OK, EXIT_USAGE, main
from qavf.engine.registry import default_registry, registration_transaction
from qavf.engine.runner import ConfigurationError, build_registry
from qavf.scaffold import config_json_schema, profile_frame, render_config
from qavf.settings import RuleSettings, SchemaSpec, load_settings
from qavf.sources import DataSource, SourceParams, SourceRegistry


class Needed(RuleParams):
    threshold: float


class NeededRule(Rule[Needed]):
    rule_id: ClassVar[str] = "REV3-NEEDED"
    name: ClassVar[str] = "Needs configuration"
    description: ClassVar[str] = "Cannot run until a threshold is set."
    category: ClassVar[str] = "custom"
    Params = Needed

    def evaluate(self, ctx):
        return Finding.ok("Configured")


class ProbeSourceParams(SourceParams):
    type: Literal["probe_source"] = "probe_source"


class ProbeSource(DataSource[ProbeSourceParams]):
    kind: ClassVar[str] = "probe_source"
    Params = ProbeSourceParams

    def load(self):
        return pl.DataFrame({"x": [1]})


@pytest.fixture(autouse=True)
def isolate_global_registration(monkeypatch):
    monkeypatch.setattr(default_registry, "_rules", dict(default_registry._rules))


@pytest.mark.parametrize(
    "required,rules,sources,allowed",
    [
        (["pack"], "error: failed", "loaded", False),
        (["pack"], "loaded", "error: failed", False),
        (["pack"], "loaded", "loaded", True),
        (["pack"], "loaded", None, True),
        (["pack"], None, "loaded", True),
        (["pack"], None, None, False),
        (["rules:pack"], "loaded", "error: failed", True),
        (["rules:pack"], None, "loaded", False),
        (["sources:pack"], "error: failed", "loaded", True),
    ],
)
def test_required_components_cannot_mask_each_other(required, rules, sources, allowed):
    registry, source_registry = RuleRegistry(), SourceRegistry()
    if rules is not None:
        registry.plugins["pack"] = rules
    if sources is not None:
        source_registry.plugins["pack"] = sources
    settings = QAVFSettings.model_validate({"plugins": {"required": required}})
    runner = QARunner(settings, registry=registry, sources=source_registry)
    if allowed:
        assert runner.plan() == []
    else:
        with pytest.raises(ConfigurationError, match="required plugin"):
            runner.plan()


def test_required_plugin_uses_real_discovery_failures(monkeypatch):
    def fail():
        raise ImportError("unavailable rules")
    monkeypatch.setattr("qavf.engine.registry.entry_points", lambda **kw: [
        SimpleNamespace(name="pack", load=fail),
    ])
    monkeypatch.setattr("qavf.sources.base.entry_points", lambda **kw: [
        SimpleNamespace(name="pack", load=lambda: ProbeSource),
    ])
    runner = QARunner(QAVFSettings.model_validate({"plugins": {"required": ["pack"]}}))
    assert runner.plugin_status()["sources:pack"] == "loaded"
    with pytest.raises(ConfigurationError, match="rules:pack"):
        runner.plan()


@pytest.mark.parametrize("group", ["rules", "sources"])
def test_duplicate_entrypoint_names_load_neither_candidate(monkeypatch, group):
    calls = []
    entries = [SimpleNamespace(name="duplicate", load=lambda: calls.append(1)) for _ in range(2)]
    if group == "rules":
        monkeypatch.setattr("qavf.engine.registry.entry_points", lambda **kw: entries)
        registry = RuleRegistry()
        registry.discover_entry_points()
    else:
        monkeypatch.setattr("qavf.sources.base.entry_points", lambda **kw: entries)
        registry = SourceRegistry().discover()
    assert not calls
    assert registry.plugins["duplicate"].startswith("error: duplicate")


def _plugin_text(*, fail=True):
    return '''from qavf import Finding, Rule, register
@register
class Partial(Rule):
    rule_id = 'REV3-PARTIAL'
    name = 'Partial'
    description = 'A staged declaration'
    category = 'custom'
    def evaluate(self, ctx):
        return Finding.ok('ok')
''' + ("raise RuntimeError('initialization failed')\n" if fail else "")


@pytest.mark.parametrize("mechanism", ["rules", "sources", "module", "path"])
def test_failed_import_discards_public_decorators(tmp_path, monkeypatch, mechanism):
    plugin = tmp_path / "custom_import.py"
    plugin.write_text(_plugin_text())
    monkeypatch.syspath_prepend(str(tmp_path))
    default_registry.register(NeededRule)
    entry = SimpleNamespace(name="broken", load=lambda: importlib.import_module(plugin.stem))
    registry = RuleRegistry()
    if mechanism == "rules":
        monkeypatch.setattr("qavf.engine.registry.entry_points", lambda **kw: [entry])
        registry.discover_entry_points()
        assert registry.plugins["broken"].startswith("error:")
    elif mechanism == "sources":
        monkeypatch.setattr("qavf.sources.base.entry_points", lambda **kw: [entry])
        sources = SourceRegistry().discover()
        assert sources.plugins["broken"].startswith("error:")
    elif mechanism == "path":
        with pytest.raises(ImportError):
            registry.discover_path(plugin)
    else:
        with pytest.raises(ImportError):
            registry.discover(packages=(), modules=[plugin.stem], use_entry_points=False)
    assert "REV3-PARTIAL" not in registry
    assert "REV3-PARTIAL" not in default_registry
    assert "REV3-NEEDED" in default_registry
    fresh = build_registry(QAVFSettings.model_validate({"plugins": {"entry_points": False}}))
    assert "REV3-PARTIAL" not in fresh


def test_failed_module_registration_also_discards_decorators(tmp_path):
    plugin = tmp_path / "invalid_declaration.py"
    plugin.write_text(_plugin_text(fail=False) + '''
class Invalid(Rule):
    rule_id = 'REV3-INVALID'
    def evaluate(self, ctx):
        return Finding.ok('ok')
''')
    with pytest.raises(TypeError, match="metadata"):
        RuleRegistry().discover_path(plugin)
    assert "REV3-PARTIAL" not in default_registry


def test_successful_import_publishes_decorator_once(tmp_path):
    plugin = tmp_path / "valid_declaration.py"
    plugin.write_text(_plugin_text(fail=False))
    first, second = RuleRegistry(), RuleRegistry()
    first.discover_path(plugin)
    second.discover_path(plugin)
    assert first.get("REV3-PARTIAL") is second.get("REV3-PARTIAL")
    assert first.get("REV3-PARTIAL") is default_registry.get("REV3-PARTIAL")


@pytest.mark.parametrize("fail", [False, True])
def test_transaction_preserves_unrelated_concurrent_registration(fail):
    staged = threading.Event()
    finish = threading.Event()
    errors = []
    def discover():
        try:
            with registration_transaction():
                default_registry.register(NeededRule)
                staged.set()
                assert finish.wait(5)
                if fail:
                    raise ValueError("failed discovery")
        except ValueError as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=discover)
    worker.start()
    try:
        assert staged.wait(5)
        assert "REV3-NEEDED" not in default_registry
        other = type("ConcurrentRule", (NeededRule,), {"rule_id": "REV3-OTHER"})
        default_registry.register(other)
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive()
    assert "REV3-OTHER" in default_registry
    assert ("REV3-NEEDED" in default_registry) is not fail
    assert bool(errors) is fail


@pytest.mark.parametrize("form", ["mapping", "settings", "path"])
@pytest.mark.parametrize("explicit_params", [False, True])
def test_disabled_required_params_have_identical_facade_semantics(tmp_path, form, explicit_params):
    default_registry.register(NeededRule)
    rule = {"enabled": False}
    if explicit_params:
        rule["params"] = {}
    raw = {"rules": {NeededRule.rule_id: rule}, "datasets": {"data": {"schema": {"min_rows": 1}}},
           "plugins": {"entry_points": False}, "engine": {"parallel": False}}
    config = raw
    if form == "settings":
        config = QAVFSettings.model_validate(raw)
    elif form == "path":
        config = tmp_path / "qavf.yaml"
        config.write_text(yaml.safe_dump(raw))
    frame = pl.DataFrame({"x": [1]})
    if explicit_params:
        with pytest.raises(ConfigurationError, match="disabled rule"):
            validate(frame, config=config)
    else:
        report = validate(frame, config=config)
        assert report.exit_code() == 0
        assert report.metadata.excluded_rules[NeededRule.rule_id] == "disabled in the configuration"


def test_omission_serializer_keeps_mutated_parameters_and_explicit_empty_mapping():
    omitted = RuleSettings(enabled=False)
    assert "params" not in RuleSettings.model_validate(omitted.model_dump()).model_fields_set
    omitted.params["threshold"] = 2
    assert omitted.model_dump()["params"] == {"threshold": 2}
    explicit = RuleSettings.model_validate(RuleSettings(params={}).model_dump())
    assert "params" in explicit.model_fields_set


class Aliased(RuleParams):
    source_name: str = Field(alias="source")


class ValidationAliased(RuleParams):
    source_name: str = Field(validation_alias=AliasChoices("source", "alternate"))


class Conditional(RuleParams):
    left: str | None = None
    right: str | None = None

    @model_validator(mode="after")
    def one_source(self):
        if self.left is None and self.right is None:
            raise ValueError("left or right is required")
        return self


class Nested(RuleParams):
    source: Aliased


@pytest.mark.parametrize("params", [Aliased, ValidationAliased, Conditional, Nested])
def test_plugin_scaffold_handles_aliases_and_conditional_defaults(params):
    rule = type("CustomParamsRule", (NeededRule,), {"Params": params})
    registry = RuleRegistry()
    registry.register(rule)
    text = render_config(registry)
    raw = yaml.safe_load(text)
    entry = raw["rules"][rule.rule_id]
    assert entry["enabled"] is False
    assert "params" not in entry
    assert QARunner(QAVFSettings.model_validate(raw), registry=registry).plan() == []
    schema = config_json_schema(registry, SourceRegistry())
    jsonschema.validate(raw, schema)
    enabled = {"rules": {rule.rule_id: {"enabled": True}}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(enabled, schema)
    assert "source" in text if params is not Conditional else "left or right" in text


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json", ".JSON"])
def test_init_formats_reload_with_exact_profiled_bounds(tmp_path, suffix):
    data = tmp_path / "values.parquet"
    pl.DataFrame({"value": [Decimal("123456789012345678.0000000001")]}).write_parquet(data)
    target = tmp_path / ("generated" + suffix)
    assert main(["init", "-o", str(target), "-d", str(data)]) == EXIT_OK
    settings = load_settings(target)
    assert settings.datasets["values"].schema_spec.columns["value"].ge == Decimal(
        "123456789012345678.0000000001"
    )
    if suffix.lower() == ".json":
        assert json.loads(target.read_text())["$schema"] == "qavf.schema.json"


@pytest.mark.parametrize("name", ["bad.toml", "qavf.schema.json"])
def test_invalid_init_output_plan_writes_nothing(tmp_path, name):
    target = tmp_path / name
    assert main(["init", "-o", str(target)]) == EXIT_USAGE
    assert list(tmp_path.iterdir()) == []


def test_schema_generation_failure_does_not_publish_config(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("schema unavailable")
    monkeypatch.setattr("qavf.cli.config_json_schema", fail)
    target = tmp_path / "qavf.yaml"
    assert main(["init", "-o", str(target)]) == EXIT_USAGE
    assert list(tmp_path.iterdir()) == []


def test_nonzero_profile_margin_avoids_intermediate_float_overflow():
    frame = pl.DataFrame({"x": [-1e308, 1e308]})
    schema = SchemaSpec.model_validate(profile_frame(frame, margin=0.1))
    spec = schema.columns["x"]
    assert math.isfinite(spec.ge) and math.isfinite(spec.le)
    assert spec.ge <= -1.2e308 and spec.le >= 1.2e308
    with pytest.raises(ValueError, match="finite float range"):
        profile_frame(frame, margin=10)


@pytest.fixture
def config_key_plugin():
    calls = []

    class KeySensitiveRule(Rule):
        rule_id: ClassVar[str] = "REV3-KEYS"
        name: ClassVar[str] = "Configuration key semantics"
        description: ClassVar[str] = "Nested mapping keys retain their runtime meaning."
        category: ClassVar[str] = "custom"

        def evaluate(self, ctx):
            mapping = ctx.extension("probe")["mapping"]
            calls.append(mapping)
            return Finding.ok("integer key") if 1 in mapping else Finding.fail("string key")

    registry = RuleRegistry()
    registry.register(KeySensitiveRule)
    return registry, calls


@pytest.mark.parametrize("container", [lambda x: x, lambda x: [x], lambda x: (x,)])
def test_nonstring_config_mapping_keys_rejected_before_plugin_evaluation(
    config_key_plugin, container
):
    registry, calls = config_key_plugin
    with pytest.raises(ValidationError, match="configuration mapping keys must be strings"):
        settings = QAVFSettings.model_validate(
            {"extensions": {"probe": {"mapping": container({1: "secret-value"})}}}
        )
        QARunner(settings, registry=registry, sources=SourceRegistry()).run()
    assert calls == []


@pytest.mark.parametrize("mutate_after_runner", [False, True])
def test_runner_preflight_rejects_late_nonstring_config_key(config_key_plugin, mutate_after_runner):
    registry, calls = config_key_plugin
    settings = QAVFSettings.model_validate(
        {"extensions": {"probe": {"mapping": {"1": "secret-value"}}}}
    )
    if mutate_after_runner:
        runner = QARunner(settings, registry=registry, sources=SourceRegistry())
    settings.extensions["probe"]["mapping"] = {1: "secret-value"}
    with pytest.raises(
        ConfigurationError, match="configuration mapping keys must be strings"
    ) as exc:
        if not mutate_after_runner:
            runner = QARunner(settings, registry=registry, sources=SourceRegistry())
        runner.run()
    assert calls == []
    assert "configuration.extensions" in str(exc.value)
    assert "secret-value" not in str(exc.value)


def test_string_config_mapping_key_reaches_real_plugin(config_key_plugin):
    registry, calls = config_key_plugin
    settings = QAVFSettings.model_validate({"extensions": {"probe": {"mapping": {"1": "x"}}}})
    report = QARunner(settings, registry=registry, sources=SourceRegistry()).run()
    assert calls == [{"1": "x"}]
    assert report.results[0].status.value == "FAIL"
    assert report.metadata.config_digest


def test_config_key_error_does_not_echo_arbitrary_keys_or_values():
    with pytest.raises(ValidationError) as exc:
        QAVFSettings.model_validate({
            "extensions": {"secret-plugin-name": {"secret-nested-key": {7: "secret-value"}}}
        })
    message = str(exc.value)
    assert "configuration.extensions" in message
    assert "secret" not in message


def test_string_enum_config_keys_are_admitted():
    class Key(StrEnum):
        NAME = "name"

    settings = QAVFSettings.model_validate({"extensions": {"probe": {"mapping": {Key.NAME: 1}}}})
    assert settings.extensions["probe"]["mapping"]["name"] == 1
