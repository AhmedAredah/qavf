"""Configuration: loading, environment variables, ``qavf init``, profiling and JSON Schema."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import jsonschema
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from qavf.cli import EXIT_OK, EXIT_USAGE, main
from qavf.demo import demo_config, write_demo_project
from qavf.engine.runner import QARunner, build_source_registry
from qavf.engine.severity import Severity
from qavf.scaffold import config_json_schema, profile_frame, render_config
from qavf.settings import QAVFSettings, expand_env, load_settings

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- settings
def test_yaml_and_json_load_identically(tmp_path) -> None:
    document = {
        "project": "demo",
        "datasets": {"a": {"path": "data/a.csv", "schema": {"primary_key": ["k"]}}},
        "rules": {"TMP-001": {"severity": "error", "params": {"max_change_rate": 0.1}}},
        "report": {"json": False},
        "extensions": {"moves": {"version": "MOVES4"}},
    }
    (tmp_path / "c.json").write_text(json.dumps({"$schema": "x.json", **document}))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(document))
    from_json, from_yaml = load_settings(tmp_path / "c.json"), load_settings(tmp_path / "c.yaml")
    assert from_json.model_dump() == from_yaml.model_dump()
    assert from_json.rules["TMP-001"].severity is Severity.ERROR
    assert from_json.report.json_ is False and from_json.extensions["moves"]["version"] == "MOVES4"
    assert from_json.resolve("data/a.csv") == tmp_path / "data" / "a.csv"


def test_environment_variables_are_expanded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QAVF_TEST_DIR", "warehouse")
    (tmp_path / "c.yaml").write_text(
        "datasets:\n"
        "  a: {path: '${QAVF_TEST_DIR}/a.csv'}\n"
        "  b: {path: '${QAVF_UNSET:-fallback}/b.csv'}\n"
    )
    settings = load_settings(tmp_path / "c.yaml")
    assert settings.datasets["a"].path == "warehouse/a.csv"
    assert settings.datasets["b"].path == "fallback/b.csv"
    with pytest.raises(ValueError, match="QAVF_NOPE"):
        expand_env({"x": ["${QAVF_NOPE}"]}, environ={})


def test_invalid_configurations_are_rejected(tmp_path) -> None:
    for bad, match in [
        ({"enginee": {}}, "enginee"),
        (
            {"datasets": {"a": {"path": "x.csv", "source": {"type": "file"}}}},
            "either 'path' or 'source'",
        ),
        (
            {"datasets": {"a": {"source": {"type": "query", "sql": "x"}, "format": "csv"}}},
            "'format' goes with 'path'",
        ),
        (
            {"datasets": {"a": {"foreign_keys": [{"columns": ["x"], "ref_dataset": "nope"}]}}},
            "unknown dataset",
        ),
        (
            {"datasets": {"a": {"source": {"type": "database", "connection": "db", "table": "t"}}}},
            "unknown connection",
        ),
        ({"assertions": [{"name": "x", "sql": "s"}, {"name": "x", "sql": "s"}]}, "unique"),
    ]:
        with pytest.raises(ValidationError, match=match):
            QAVFSettings.model_validate(bad)
    (tmp_path / "c.toml").write_text("x = 1")
    with pytest.raises(ValueError, match="unsupported configuration format"):
        load_settings(tmp_path / "c.toml")
    (tmp_path / "list.yaml").write_text("- a\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_settings(tmp_path / "list.yaml")


def test_canonical_columns_from_import_reference() -> None:
    settings = QAVFSettings.model_validate(
        {"engine": {"canonical_columns": "string:ascii_letters"}}
    )
    assert settings.engine.resolved_canonical_columns()[:3] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="module:attribute"):
        QAVFSettings.model_validate(
            {"engine": {"canonical_columns": "nomodule"}}
        ).engine.resolved_canonical_columns()


@pytest.mark.parametrize("config", ["config/qavf.minimal.yaml", "config/qavf.full.yaml"])
def test_shipped_example_configurations_plan(config: str) -> None:
    settings = load_settings(ROOT / config)
    settings.plugins.entry_points = False
    assert QARunner(settings).plan()


# ------------------------------------------------------------------ profiling
def test_profile_frame_infers_limits() -> None:
    frame = pl.DataFrame(
        {
            "id": [f"ID-{i:03d}" for i in range(30)],
            "kind": ["a", "b", "c"] * 10,
            "n": list(range(30)),
            "x": [float(i) / 2 for i in range(29)] + [None],
            "d": [date(2020, 1, 1 + i % 28) for i in range(30)],
            "flag": [True, False] * 15,
        }
    )
    columns = profile_frame(frame)["columns"]
    assert columns["id"] == {"dtype": "str", "min_length": 6, "max_length": 6}
    assert columns["kind"] == {"dtype": "str", "isin": ["a", "b", "c"]}
    assert columns["n"] == {"dtype": "int", "ge": 0, "le": 29}
    assert columns["x"] == {"dtype": "float", "nullable": True, "ge": 0, "le": 14}
    assert columns["d"] == {"dtype": "date", "ge": date(2020, 1, 1), "le": date(2020, 1, 28)}
    assert columns["flag"] == {"dtype": "bool"}
    widened = profile_frame(frame, margin=0.1, bounds=True)["columns"]["n"]
    # Integer bounds stay integers and are rounded outwards (-2.9 -> -3, 31.9 -> 32).
    assert (widened["ge"], widened["le"]) == (-3, 32)
    assert "ge" not in profile_frame(frame, bounds=False)["columns"]["n"]


def test_profiled_schema_validates_its_own_data(demo) -> None:
    from qavf.testing import run_rules

    datasets = {name: {"schema": profile_frame(frame)} for name, frame in demo.items()}
    report = run_rules(demo, datasets=datasets, rules={"INP-007": {"enabled": False}})
    assert not report.failures(), [(r.label, r.message) for r in report.failures()]


# ----------------------------------------------------------------- init/schema
def test_render_config_lists_every_rule_and_loads(tmp_path, registry) -> None:
    text = render_config(
        registry, {"counts": {"path": "counts.csv", "schema": {"columns": {"x": {"ge": 0}}}}}
    )
    path = tmp_path / "qavf.yaml"
    path.write_text(text, encoding="utf-8")
    settings = load_settings(path)
    assert set(settings.rules) == {rule.rule_id for rule in registry}
    assert settings.datasets["counts"].schema_spec.columns["x"].ge == 0
    settings.plugins.entry_points = False
    assert len(QARunner(settings, registry).plan()) == len(registry)


def test_json_schema_validates_configurations(registry) -> None:
    schema = config_json_schema(registry, build_source_registry(QAVFSettings()))
    validator = jsonschema.Draft202012Validator(schema)
    # Editors see YAML dates as strings; mirror that by round-tripping through JSON.
    good = json.loads(json.dumps(yaml.safe_load(demo_config()), default=str))
    assert not list(validator.iter_errors(good))
    assert {"INP-003", "TMP-001", "REG-001"} <= set(schema["$defs"]["Rules"]["properties"])
    for bad in (
        {"rules": {"TMP-001": {"params": {"max_change_rate": "fast"}}}},
        {"rules": {"TMP-001": {"params": {"typo": 1}}}},
        {"datasets": {"a": {"source": {"type": "file", "pth": "x"}}}},
        {"datasets": {"a": {"schema": {"columns": {"x": {"severity": "LOUD"}}}}}},
    ):
        assert list(validator.iter_errors(bad)), bad


def test_init_command(tmp_path, capsys) -> None:
    write_demo_project(tmp_path / "demo", clean=True)
    output = tmp_path / "project" / "qavf.yaml"
    counts = tmp_path / "demo" / "data" / "counts.csv"
    crashes = tmp_path / "demo" / "data" / "crashes.parquet"
    assert (
        main(
            [
                "init",
                "-o",
                str(output),
                "-d",
                str(counts),
                "-d",
                f"crash_data={crashes}",
                "--margin",
                "0.05",
            ]
        )
        == EXIT_OK
    )
    assert (output.parent / "qavf.schema.json").is_file()
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# yaml-language-server: $schema=qavf.schema.json")
    settings = load_settings(output)
    assert settings.datasets["counts"].path == "../demo/data/counts.csv"
    assert settings.datasets["crash_data"].schema_spec.columns["fatal"].dtype == "int"
    report = QARunner(settings).run()
    assert not report.failures(), [(r.label, r.message) for r in report.failures()]

    assert main(["init", "-o", str(output)]) == EXIT_USAGE  # exists
    assert (
        main(["init", "-o", str(output), "--force", "--no-schema", "-d", "bad name=x.csv"])
        == EXIT_USAGE
    )
    assert main(["schema", "-o", str(tmp_path / "s.json")]) == EXIT_OK
    assert json.loads((tmp_path / "s.json").read_text())["title"] == "QAVF configuration"
    capsys.readouterr()


def test_generated_reference_docs_are_current() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_docs", ROOT / "scripts" / "generate_docs.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stale = [
        path.name
        for path, text in module.pages().items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    assert not stale, f"run: uv run python scripts/generate_docs.py  (stale: {stale})"
