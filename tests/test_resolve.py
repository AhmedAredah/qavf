"""``qavf resolve`` and :func:`qavf.resolve`: the resolved, redacted configuration with
the origin of each value, deterministic and written without replacing an input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qavf import ResolvedConfig, resolve
from qavf.cli import EXIT_OK, EXIT_USAGE, main
from qavf.settings import QAVFSettings, load_settings


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(tmp_path: Path) -> Path:
    _write(
        tmp_path / "team" / "base.yaml",
        "vars: {key: [id, day]}\n"
        "datasets:\n"
        "  counts:\n"
        "    path: counts.csv\n"
        "    across: {houston: {path: h.csv}, dallas: {path: d.csv}}\n"
        "    schema: {min_rows: 1}\n"
        "checks:\n"
        "  - {check: sql, name: base check, sql: 'SELECT 1 WHERE false'}\n"
        "overrides:\n"
        "  checks:\n"
        "    base check: {severity: WARNING}\n",
    )
    return _write(
        tmp_path / "qa.yaml",
        "include: [team/base.yaml]\n"
        "vars: {rows: 5}\n"
        "defaults:\n"
        "  expression: {severity: WARNING}\n"
        "connections:\n"
        "  warehouse: {type: postgres, uri: 'host=db user=qa password=hunter2'}\n"
        "datasets:\n"
        "  roads:\n"
        "    path: roads.csv\n"
        "    schema: {min_rows: '${vars.rows}', primary_key: '${vars.key}'}\n"
        "    checks:\n"
        "      - {check: expression, name: positive, expr: 'a > 0'}\n"
        "overrides:\n"
        "  datasets:\n"
        "    counts@houston: {schema: {min_rows: 7}}\n",
    )


def test_resolve_applies_every_reuse_step_and_redacts(tmp_path) -> None:
    resolved = resolve(_project(tmp_path))
    assert isinstance(resolved, ResolvedConfig)
    datasets = resolved.configuration["datasets"]
    assert list(datasets) == ["counts@houston", "counts@dallas", "roads"]
    assert datasets["counts@houston"]["schema"]["min_rows"] == 7
    assert datasets["roads"]["schema"] == {"min_rows": 5, "primary_key": ["id", "day"]}
    assert datasets["roads"]["checks"][0]["severity"] == "WARNING"
    assert resolved.configuration["checks"][0]["severity"] == "WARNING"
    assert "overrides" not in resolved.configuration and "include" not in resolved.configuration
    text = resolved.to_yaml(origins=True) + resolved.to_json(origins=True)
    assert "hunter2" not in text and "***" in text
    assert [applied.instance for applied in resolved.overrides] == [
        "checks[0]",
        "datasets.counts@houston",
    ]


def test_the_resolved_document_reads_back_as_the_same_settings(tmp_path) -> None:
    config = _project(tmp_path)
    resolved = resolve(config)
    for text in (resolved.to_yaml(), resolved.to_yaml(origins=True), resolved.to_json()):
        assert yaml.safe_load(text) == resolved.configuration
    again = _write(tmp_path / "resolved.yaml", resolved.to_yaml(origins=True))
    assert resolve(again).configuration == resolved.configuration


def test_origins_name_file_line_and_how_each_value_arrived(tmp_path) -> None:
    resolved = resolve(_project(tmp_path))
    origins = {path: origin.describe() for path, origin in resolved.origins.items()}
    assert origins["datasets.roads.schema.min_rows"] == "qa.yaml:10:14 (via vars.rows)"
    assert origins["datasets.roads.schema.primary_key[1]"] == "team/base.yaml:1:18 (via vars.key)"
    assert origins["datasets.roads.checks[0].severity"] == "qa.yaml:4:16 (from defaults.expression)"
    assert (
        origins["datasets.counts@dallas.schema.min_rows"]
        == "team/base.yaml:6:14 (from across copy counts@dallas)"
    )
    assert (
        origins["datasets.counts@houston.schema.min_rows"]
        == "qa.yaml:15:31 (override counts@houston)"
    )
    assert origins["checks[0].severity"] == "team/base.yaml:11:18 (override base check)"
    yaml_text = resolved.to_yaml(origins=True)
    assert "      min_rows: 7  # qa.yaml:15:31 (override counts@houston)\n" in yaml_text
    assert "#   checks[0] ('base check'): severity from team/base.yaml:11\n" in yaml_text
    document = json.loads(resolved.to_json(origins=True))
    assert document["origins"]["checks[0].severity"] == origins["checks[0].severity"]
    assert document["overrides"][1] == {
        "instance": "datasets.counts@houston",
        "target": "counts@houston",
        "section": "datasets",
        "fields": ["schema.min_rows"],
        "origin": "qa.yaml",
        "line": 15,
    }


def test_output_is_deterministic(tmp_path) -> None:
    config = _project(tmp_path)
    first = [
        resolve(config).render(fmt, origins=flag)
        for fmt in ("yaml", "json")
        for flag in (False, True)
    ]
    second = [
        resolve(config).render(fmt, origins=flag)
        for fmt in ("yaml", "json")
        for flag in (False, True)
    ]
    assert first == second


def test_multiline_and_special_strings_stay_on_one_line(tmp_path) -> None:
    config = _write(
        tmp_path / "qa.yaml",
        "project: 'a: b # c'\n"
        "datasets:\n  x:\n    path: x.csv\n"
        "checks:\n"
        "  - check: sql\n"
        "    name: multi\n"
        "    sql: |\n      SELECT *\n      FROM x\n      WHERE a < 0\n",
    )
    resolved = resolve(config)
    text = resolved.to_yaml(origins=True)
    assert '    sql: "SELECT *\\nFROM x\\nWHERE a < 0\\n"  # qa.yaml:8:5' in text
    assert yaml.safe_load(text) == resolved.configuration


def test_settings_and_mappings_resolve_without_origins() -> None:
    raw = {"datasets": {"x": {"path": "x.csv"}}}
    for config in (raw, QAVFSettings.model_validate(raw)):
        resolved = resolve(config)
        assert resolved.configuration == {"datasets": {"x": {"path": "x.csv"}}}
        assert resolved.origins == {}


def test_loaded_settings_keep_their_origins(tmp_path) -> None:
    settings = load_settings(_project(tmp_path))
    assert resolve(settings).origins


def test_an_invalid_configuration_raises_with_origins(tmp_path) -> None:
    _write(
        tmp_path / "m" / "c.yaml", "datasets:\n  x:\n    path: x.csv\n    schema: {min_rowz: 1}\n"
    )
    config = _write(tmp_path / "qa.yaml", "include: [m/c.yaml]\n")
    with pytest.raises(ValidationError):
        resolve(config)


def test_cli_prints_and_exits_zero(tmp_path, capsys) -> None:
    config = _project(tmp_path)
    assert main(["resolve", "-c", str(config), "--origins"]) == EXIT_OK
    out = capsys.readouterr().out
    assert out == resolve(config).to_yaml(origins=True)
    assert main(["resolve", "-c", str(config), "--format", "json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out) == resolve(config).configuration


def test_cli_reports_an_invalid_configuration_with_exit_two(tmp_path, capsys) -> None:
    _write(
        tmp_path / "m" / "c.yaml", "datasets:\n  x:\n    path: x.csv\n    schema: {min_rowz: 1}\n"
    )
    config = _write(tmp_path / "qa.yaml", "include: [m/c.yaml]\n")
    assert main(["resolve", "-c", str(config)]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "datasets.x.schema.min_rowz (m/c.yaml:4): Extra inputs are not permitted" in (
        captured.err
    )


def test_cli_writes_a_file_but_never_replaces_an_input(tmp_path, capsys) -> None:
    config = _project(tmp_path)
    original = config.read_text(encoding="utf-8")
    included = tmp_path / "team" / "base.yaml"
    for target in (config, included, tmp_path / "roads.csv"):
        if not target.exists():
            target.write_text("id\n1\n", encoding="utf-8")
        before = target.read_text(encoding="utf-8")
        assert main(["resolve", "-c", str(config), "-o", str(target), "--force"]) == EXIT_USAGE
        assert target.read_text(encoding="utf-8") == before
    assert config.read_text(encoding="utf-8") == original
    assert "would replace the input" in capsys.readouterr().err
    output = tmp_path / "out" / "resolved.yaml"
    assert main(["resolve", "-c", str(config), "-o", str(output)]) == EXIT_OK
    assert output.read_text(encoding="utf-8") == resolve(config).to_yaml()
    assert main(["resolve", "-c", str(config), "-o", str(output)]) == EXIT_USAGE
    assert "refusing to overwrite" in capsys.readouterr().err
    assert main(["resolve", "-c", str(config), "-o", str(output), "--force", "--origins"]) == 0
    assert "# qa.yaml:" in output.read_text(encoding="utf-8")


def test_resolve_reads_no_data(tmp_path, monkeypatch) -> None:
    import qavf.sources.base as sources

    monkeypatch.setattr(sources.SourceRegistry, "build", _no_build)
    assert resolve(_project(tmp_path)).configuration


def _no_build(*args: object, **kwargs: object) -> None:
    raise AssertionError("resolve built a data source")
