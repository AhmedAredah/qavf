"""Public contracts stay equivalent across input forms and plugin boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from decimal import Decimal, localcontext
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import duckdb
import jsonschema
import polars as pl
import pytest
import yaml

from qavf import Loader, Query, ResourceLimits, validate
from qavf.cli import EXIT_USAGE, main
from qavf.engine.registry import Finding, Rule, RuleRegistry, default_registry
from qavf.engine.runner import QARunner, build_source_registry
from qavf.inputs.adapters import _QueryInput
from qavf.scaffold import config_json_schema, profile_frame, render_config
from qavf.settings import Bounds, ColumnSpec, DatasetSpec, QAVFSettings, SchemaSpec, load_settings
from qavf.sources import FileSource, SourceEnvironment, SourceRegistry
from qavf.sources.parameters import FileParams

SCHEMA = {"columns": {"x": {"dtype": "int", "ge": 0}}}


class EagerQueryRule(Rule):
    rule_id: ClassVar[str] = "CUSTOM-EAGER"
    name: ClassVar[str] = "Eager query consumer"
    description: ClassVar[str] = "Exercise the explicit materialization boundary."
    category: ClassVar[str] = "custom"

    def evaluate(self, ctx):
        frame = ctx.get("data")
        return Finding.ok(f"Read {frame.height} rows", total_records=frame.height)


def _register_eager(monkeypatch):
    monkeypatch.setattr(default_registry, "_rules", dict(default_registry._rules))
    default_registry.register(EagerQueryRule)
    return {
        "rules": {EagerQueryRule.rule_id: {}},
        "engine": {"run_unconfigured_rules": False, "parallel": False},
        "plugins": {"entry_points": False},
    }


def test_owned_query_materialization_matches_native_and_borrowed_paths(monkeypatch):
    config = _register_eager(monkeypatch)
    query = "SELECT $x AS x"
    with duckdb.connect() as connection:
        for item in (Query(query, parameters={"x": 1}), Query(query, connection, {"x": 1})):
            report = validate(item, config=config)
            assert report.exit_code() == 0
            assert report.results[0].total_records == 1
        assert connection.execute("SELECT 42").fetchone() == (42,)
    assert validate(Query(query, parameters={"x": 1}), schema=SCHEMA).exit_code() == 0


@pytest.mark.parametrize("sql", ["SELECT 1 AS x", "SELECT * FROM missing_table"])
def test_owned_query_releases_managed_connection_on_success_and_error(sql):
    class Budget:
        acquired = 0
        released = 0

        def acquire(self):
            self.acquired += 1
            return object(), 0, None

        def release(self, token):
            self.released += 1

    budget = Budget()
    source = _QueryInput(Query(sql), SourceEnvironment(connection_budget=budget), threading.Lock())
    if "missing_table" in sql:
        with pytest.raises(duckdb.Error):
            source.load()
    else:
        assert source.load().height == 1
    assert budget.acquired == budget.released == 1


@pytest.mark.parametrize("optimized", [False, True])
def test_query_materialization_does_not_depend_on_python_assertions(optimized):
    script = """
from qavf import Query, Rule, Finding, register, validate
@register
class Eager(Rule):
    rule_id='CUSTOM-EAGER-SUBPROCESS'
    name='eager'
    description='query materialization'
    category='test'
    def evaluate(self, ctx):
        return Finding.ok('loaded', total_records=ctx.get('data').height)
report=validate(Query('SELECT 1 AS x'), config={
    'rules': {Eager.rule_id:{}},
    'plugins': {'entry_points':False},
    'engine': {'parallel':False,'run_unconfigured_rules':False},
})
if report.exit_code() != 0 or report.results[0].total_records != 1:
    raise RuntimeError(report.model_dump_json())
"""
    subprocess.run([sys.executable, *(["-O"] if optimized else []), "-c", script], check=True)


def test_file_location_override_keeps_format_options_projection_and_policy(tmp_path):
    path = tmp_path / "original.data"
    other = tmp_path / "replacement.data"
    for file in (path, other):
        file.write_text("x;unused\n1;100\n2;200\n")
    source = {
        "type": "file", "path": str(path), "format": "csv",
        "options": {"delim": ";", "header": True}, "columns": ["x"],
        "row_budget": 2, "retries": 2, "retry_delay_s": 0,
    }
    settings = QAVFSettings.model_validate({
        "datasets": {"data": {"source": source, "schema": {**SCHEMA, "strict": True}}},
        "plugins": {"entry_points": False}, "engine": {"parallel": False},
    })
    before = settings.model_dump()
    for replacement in (None, path, {"data": other}):
        report = validate(replacement, config=settings)
        assert report.exit_code() == 0
        effective = report.metadata.config["datasets"]["data"]
        assert effective["source"]["format"] == "csv"
        assert effective["source"]["options"] == source["options"]
        assert effective["source"]["retries"] == 2
        assert effective["source"]["row_budget"] == 2
    assert settings.model_dump() == before


@pytest.mark.parametrize("kind", ["path", "frame", "lazy", "query", "loader", "source"])
def test_input_override_does_not_weaken_existing_row_budget(kind, tmp_path):
    path = tmp_path / "values.csv"
    frame = pl.DataFrame({"x": [1, 2]})
    frame.write_csv(path)
    inputs = {
        "path": path, "frame": frame, "lazy": frame.lazy(),
        "query": Query("SELECT * FROM (VALUES (1), (2)) t(x)"),
        "loader": Loader(lambda: frame),
        "source": FileSource(FileParams(path=str(path)), SourceEnvironment()),
    }
    report = validate(inputs[kind], config={
        "datasets": {"data": {
            "source": {"type": "file", "path": str(path), "row_budget": 1},
            "schema": SCHEMA,
        }},
        "plugins": {"entry_points": False}, "engine": {"parallel": False},
    })
    assert report.exit_code() == 1
    assert any("row_budget" in result.message for result in report.results)


def test_policy_roundtrips_preserve_only_explicit_overrides():
    spec = DatasetSpec.model_validate({
        "source": {"type": "file", "path": "input.csv", "row_budget": 3, "retries": 2},
        "load_policy": {"row_budget": 4},
    })
    for _ in range(3):
        assert spec.effective_load_policy().row_budget == 4
        assert spec.effective_load_policy().retries == 2
        spec = DatasetSpec.model_validate(spec.model_dump())
    assert spec.model_dump()["load_policy"] == {"row_budget": 4}


def test_loader_does_not_retry_after_inheriting_retry_policy(tmp_path):
    calls = []

    def load():
        from qavf.sources import SourceError
        calls.append(1)
        raise SourceError("temporarily unavailable", retryable=True)

    report = validate(Loader(load), config={
        "datasets": {"data": {
            "source": {"type": "file", "path": "prior.csv", "retries": 2, "retry_delay_s": 0},
            "schema": SCHEMA,
        }}, "plugins": {"entry_points": False},
    })
    assert report.exit_code() == 1
    assert calls == [1]


def test_replay_safe_loader_retries_only_with_explicit_policy():
    from qavf.sources import SourceError

    calls = []

    def load():
        calls.append(1)
        if len(calls) < 3:
            raise SourceError("temporarily unavailable", retryable=True)
        return pl.DataFrame({"x": [1]})

    report = validate(Loader(load, replayable=True, retry_safe=True), config={
        "datasets": {"data": {
            "load_policy": {"retries": 2, "retry_delay_s": 0}, "schema": SCHEMA,
        }}, "plugins": {"entry_points": False}, "engine": {"parallel": False},
    })
    assert report.exit_code() == 0
    assert len(calls) == 3


def test_direct_source_retains_its_environment_and_works_after_validation(tmp_path):
    pl.DataFrame({"x": [1, 2]}).write_csv(tmp_path / "source.csv")
    original_env = SourceEnvironment(base_dir=tmp_path)
    source = FileSource(FileParams(path="source.csv"), original_env)
    original = source.load()
    original_notes = dict(source.load_notes)
    report = validate(source, schema=SCHEMA)
    assert report.exit_code() == 0
    assert source.env is original_env
    assert source.load_notes == original_notes
    assert source.load().equals(original)


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("value_kind", ["absolute", "relative", "list"])
def test_bound_file_queries_support_paths_without_literal_sql(tmp_path, borrowed, value_kind):
    file = tmp_path / "first.csv"
    pl.DataFrame({"x": [1]}).write_csv(file)
    config = tmp_path / "q.json"
    config.write_text("{}")
    value = file if value_kind == "absolute" else "first.csv"
    expected = 1
    if value_kind == "list":
        pl.DataFrame({"x": [2]}).write_csv(tmp_path / "second.csv")
        value = ["first.csv", "second.csv"]
        expected = 2
    with duckdb.connect() as connection:
        query = Query(
            "SELECT * FROM read_csv($path)", connection if borrowed else None, {"path": value}
        )
        report = validate(
            query, schema={**SCHEMA, "min_rows": expected, "max_rows": expected}, config=config
        )
        assert report.exit_code() == 0
        assert connection.execute("SELECT 3").fetchone() == (3,)


def _plugin_rule(name, rule_id, description="A normal plugin."):
    return type(name, (Rule,), {
        "__module__": "custom_plugin", "rule_id": rule_id, "name": name,
        "description": description, "category": "custom",
        "evaluate": lambda self, ctx: Finding.ok("ok"),
    })


def test_failed_entrypoint_registration_is_transactional(monkeypatch):
    import qavf.engine.registry as registry_module
    registry = RuleRegistry()
    registry.register(_plugin_rule("Existing", "DUPLICATE"))
    module = ModuleType("custom_plugin")
    module.Good = _plugin_rule("Good", "GOOD-001")
    module.Bad = _plugin_rule("Bad", "DUPLICATE")
    monkeypatch.setattr(registry_module, "entry_points", lambda **kw: [
        SimpleNamespace(name="mixed", load=lambda: module),
    ])
    assert registry.discover_entry_points() == []
    assert [rule.rule_id for rule in registry] == ["DUPLICATE"]
    assert registry.plugins["mixed"].startswith("error:")


def test_plugin_ids_and_multiline_metadata_roundtrip_yaml_and_schema(tmp_path):
    registry = RuleRegistry()
    for index, rule_id in enumerate(("CUSTOM-001", "CUSTOM_001", "ON", "a: rule")):
        registry.register(_plugin_rule(f"Name{index}", rule_id, "First line.\nSecond line."))
    text = render_config(registry)
    path = tmp_path / "q.yaml"
    path.write_text(text, encoding="utf-8")
    settings = load_settings(path)
    assert set(settings.rules) == {rule.rule_id for rule in registry}
    assert len(QARunner(settings, registry).plan()) == len(registry)
    schema = config_json_schema(registry, build_source_registry(QAVFSettings()))
    assert set(schema["$defs"]["Rules"]["properties"]) == set(settings.rules)
    assert jsonschema.Draft202012Validator(schema).is_valid(yaml.safe_load(text))
    assert config_json_schema(registry, SourceRegistry())["title"]


@pytest.mark.parametrize("margin", [0.0, 0.5])
def test_decimal_profile_preserves_exact_bounds_through_json_and_yaml(margin):
    low = Decimal("9007199254740993.0000000001")
    high = Decimal("9007199254740993.0000000003")
    frame = pl.DataFrame({"amount": [low, high]})
    inferred = profile_frame(frame, margin=margin)
    with localcontext() as context:
        context.prec = 80
        widen = (high - low) * Decimal(str(margin))
        expected = (low - widen, high + widen)
    for parsed in (json.loads(json.dumps(inferred)), yaml.safe_load(yaml.safe_dump(inferred))):
        column = SchemaSpec.model_validate(parsed).columns["amount"]
        assert column.dtype == "decimal"
        assert (column.ge, column.le) == expected
    assert validate(frame, schema=inferred).exit_code() == 0
    outlier = pl.DataFrame({"amount": [Decimal("9007199254740999.0")]})
    assert validate(outlier, schema=inferred).exit_code() == 1
    missing = pl.DataFrame({"amount": pl.Series([None], dtype=pl.Decimal(38, 10))})
    assert profile_frame(missing)["columns"]["amount"] == {"dtype": "decimal", "nullable": True}


def test_cli_handles_resource_budget_failure_without_traceback(tmp_path, capsys):
    pl.DataFrame({"x": [1]}).write_csv(tmp_path / "data.csv")
    config = tmp_path / "q.json"
    config.write_text(json.dumps({
        "datasets": {"data": {"path": "data.csv", "schema": SCHEMA}},
        "engine": {"max_report_bytes": 1, "parallel": False},
        "plugins": {"entry_points": False},
    }))
    assert main(["run", "-c", str(config)]) == EXIT_USAGE
    output = capsys.readouterr()
    assert "Increase the budget" in output.err
    assert "Traceback" not in output.err
    process = subprocess.run(
        [sys.executable, "-m", "qavf.cli", "run", "-c", str(config)],
        capture_output=True, text=True,
    )
    assert process.returncode == EXIT_USAGE
    assert "Traceback" not in process.stderr
    assert not (tmp_path / "qa_output").exists()


def test_resource_units_reject_incomplete_binary_prefix():
    with pytest.raises(ValueError):
        ResourceLimits(target_memory="2ib")


@pytest.mark.parametrize("value", [30.29405, 0.1887, 2**70 + 1, Decimal("0.1887")])
def test_numeric_bounds_preserve_input_meaning_across_pydantic_versions(value):
    for bounds in (ColumnSpec(ge=value, le=value), Bounds(ge=value, le=value)):
        assert type(bounds.ge) is type(value)
        assert type(bounds.le) is type(value)
        assert bounds.ge == bounds.le == value
    assert ColumnSpec(ge="0.1887").ge == Decimal("0.1887")


def test_profiled_float_extrema_survive_json_and_yaml_roundtrip():
    frame = pl.DataFrame({"x": [0.0476, 0.1887, 30.29405, 33.3717]})
    inferred = profile_frame(frame)
    variants = (
        inferred,
        json.loads(json.dumps(inferred)),
        yaml.safe_load(yaml.safe_dump(inferred)),
    )
    for parsed in variants:
        assert validate(frame, schema=parsed).exit_code() == 0


@pytest.mark.parametrize("model", [Bounds, ColumnSpec])
def test_boolean_is_not_an_implicit_numeric_bound(model):
    with pytest.raises(ValueError, match="must not be boolean"):
        model(ge=True)
