"""CFG-10b: every configuration value keeps the file, line and column it was written
at, through include, vars/env, defaults, across and overrides, and errors name it."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qavf import validate
from qavf.cli import EXIT_USAGE, main
from qavf.origins import Origin, OriginIndex, format_path, parse_path
from qavf.redaction import exception_details
from qavf.settings import QAVFSettings, load_settings, settings_from_text


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _where(settings: QAVFSettings, *path: str | int) -> str | None:
    assert settings.origins is not None
    origin = settings.origins.find(path)
    return origin.describe() if origin is not None else None


def test_paths_format_and_parse() -> None:
    path = ("datasets", "counts@houston", "checks", 2, "max")
    assert format_path(path) == "datasets.counts@houston.checks[2].max"
    assert parse_path(format_path(path)) == path


def test_yaml_keys_items_and_anchors_are_located(tmp_path) -> None:
    config = _write(
        tmp_path / "qa.yaml",
        "datasets:\n"
        "  a: &shared\n"
        "    path: a.csv\n"
        "    schema: {min_rows: 1, primary_key: [id, day]}\n"
        "  b:\n"
        "    <<: *shared\n"
        "    path: b.csv\n",
    )
    settings = load_settings(config)
    assert _where(settings, "datasets", "a", "schema", "min_rows") == "qa.yaml:4:14"
    assert _where(settings, "datasets", "a", "schema", "primary_key", 1) == "qa.yaml:4:45"
    # A merge key's values point at the anchor; an explicit key wins.
    assert _where(settings, "datasets", "b", "schema", "min_rows") == "qa.yaml:4:14"
    assert _where(settings, "datasets", "b", "path") == "qa.yaml:7:5"


def test_json_values_are_located(tmp_path) -> None:
    config = _write(
        tmp_path / "qa.json",
        '{\n  "datasets": {\n    "x": {\n      "path": "x.csv",\n'
        '      "schema": {"min_rows": 3}\n    }\n  }\n}\n',
    )
    settings = load_settings(config)
    assert _where(settings, "datasets", "x", "schema", "min_rows") == "qa.json:5:18"


def test_included_values_keep_their_file_and_appended_checks_shift(tmp_path) -> None:
    _write(
        tmp_path / "team" / "base.yaml",
        "datasets:\n  x:\n    path: x.csv\n    schema: {min_rows: 1}\n"
        "checks:\n  - {check: sql, name: first, sql: 'SELECT 1 WHERE false'}\n",
    )
    config = _write(
        tmp_path / "qa.yaml",
        "include: [team/base.yaml]\n"
        "checks:\n  - {check: sql, name: second, sql: 'SELECT 2 WHERE false'}\n",
    )
    settings = load_settings(config)
    assert _where(settings, "datasets", "x", "schema", "min_rows") == "team/base.yaml:4:14"
    assert _where(settings, "checks", 0, "name") == "team/base.yaml:6:18"
    assert _where(settings, "checks", 1, "name") == "qa.yaml:3:18"


def test_vars_and_env_substitution_keep_the_reference_with_a_note(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QAVF_ORIGIN_DIR", "data")
    config = _write(
        tmp_path / "qa.yaml",
        "vars: {rows: 5, key: [id, day]}\n"
        "datasets:\n"
        "  x:\n"
        "    path: '${QAVF_ORIGIN_DIR}/x.csv'\n"
        "    schema: {min_rows: '${vars.rows}', primary_key: '${vars.key}'}\n",
    )
    settings = load_settings(config)
    assert _where(settings, "datasets", "x", "path") == "qa.yaml:4:5 (via ${QAVF_ORIGIN_DIR})"
    assert _where(settings, "datasets", "x", "schema", "min_rows") == "qa.yaml:5:14 (via vars.rows)"
    # The parts of a whole-variable value point at the variable's definition.
    assert (
        _where(settings, "datasets", "x", "schema", "primary_key", 1)
        == "qa.yaml:1:27 (via vars.key)"
    )


def test_defaults_across_and_overrides_point_at_what_supplied_the_value(tmp_path) -> None:
    config = _write(
        tmp_path / "qa.yaml",
        "defaults:\n"
        "  expression: {severity: WARNING}\n"
        "datasets:\n"
        "  counts:\n"
        "    path: counts.csv\n"
        "    across: {houston: {path: h.csv}, dallas: {path: d.csv}}\n"
        "    schema: {min_rows: 1}\n"
        "    checks:\n"
        "      - {check: expression, name: positive, expr: 'x > 0'}\n"
        "overrides:\n"
        "  datasets:\n"
        "    counts@houston: {schema: {min_rows: 7}}\n",
    )
    settings = load_settings(config)
    assert (
        _where(settings, "datasets", "counts@dallas", "checks", 0, "severity")
        == "qa.yaml:2:16 (from defaults.expression)"
    )
    assert (
        _where(settings, "datasets", "counts@dallas", "source", "path")
        == "qa.yaml:6:47 (from across copy counts@dallas)"
    )
    assert (
        _where(settings, "datasets", "counts@houston", "schema", "min_rows")
        == "qa.yaml:12:31 (override counts@houston)"
    )
    [applied] = settings.applied_overrides
    assert (applied.origin, applied.line) == (None, 12)
    assert applied.describe() == "datasets.counts@houston: schema.min_rows (line 12)"


def test_included_overrides_record_their_file_and_line(tmp_path) -> None:
    _write(
        tmp_path / "prod" / "prod.yaml",
        "overrides:\n  datasets:\n    counts: {schema: {min_rows: 100}}\n",
    )
    config = _write(
        tmp_path / "qa.yaml",
        "include: [prod/prod.yaml]\n"
        "datasets:\n  counts:\n    path: counts.csv\n    schema: {min_rows: 1}\n",
    )
    settings = load_settings(config)
    [applied] = settings.applied_overrides
    assert (applied.origin, applied.line) == ("prod/prod.yaml", 3)
    assert (
        _where(settings, "datasets", "counts", "schema", "min_rows")
        == "prod/prod.yaml:3:23 (override counts)"
    )


def test_settings_from_mappings_have_no_origins() -> None:
    assert QAVFSettings.model_validate({"datasets": {"x": {"path": "x.csv"}}}).origins is None
    assert settings_from_text("datasets: {x: {path: x.csv}}\n", "qa.yaml").origins is not None


def test_an_included_typo_names_its_file_and_line(tmp_path) -> None:
    _write(
        tmp_path / "m" / "c.yaml", "datasets:\n  x:\n    path: x.csv\n    schema: {min_rowz: 1}\n"
    )
    config = _write(tmp_path / "qavf.yaml", "include: [m/c.yaml]\n")
    with pytest.raises(ValidationError) as caught:
        load_settings(config)
    message = exception_details(caught.value, location=False)
    assert "datasets.x.schema.min_rowz (m/c.yaml:4): Extra inputs" in message
    assert "datasets.x.schema.min_rowz: written at m/c.yaml:4:14" in caught.value.__notes__


def test_a_missing_setting_names_the_enclosing_value(tmp_path) -> None:
    config = _write(
        tmp_path / "qa.yaml",
        "datasets:\n  x:\n    path: x.csv\n    series:\n      s: {time_column: t}\n",
    )
    with pytest.raises(ValidationError) as caught:
        load_settings(config)
    assert "datasets.x.series.s.value (qa.yaml:5): Field required" in exception_details(
        caught.value
    )


def test_a_configuration_error_message_names_the_value_it_mentions(tmp_path) -> None:
    config = _write(
        tmp_path / "qa.yaml",
        "datasets:\n  counts:\n    path: c.csv\n"
        "overrides:\n  datasets:\n    countz: {schema: {min_rows: 2}}\n",
    )
    with pytest.raises(ValidationError) as caught:
        load_settings(config)
    assert "(qa.yaml:" in exception_details(caught.value, location=False)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_every_entry_point_reports_the_origin(tmp_path, capsys, backend) -> None:
    _write(
        tmp_path / "m" / "c.yaml", "datasets:\n  x:\n    path: x.csv\n    schema: {min_rowz: 1}\n"
    )
    config = _write(tmp_path / "qavf.yaml", "include: [m/c.yaml]\n")
    expected = "datasets.x.schema.min_rowz (m/c.yaml:4)"
    with pytest.raises(ValidationError) as caught:
        validate(config=config, backend=backend)
    assert expected in exception_details(caught.value)
    for command in (["run", "--backend", backend], ["explain", "--backend", backend]):
        assert main([command[0], "-c", str(config), *command[1:]]) == EXIT_USAGE
        assert expected in capsys.readouterr().err


def test_an_unknown_path_has_no_origin() -> None:
    index = OriginIndex()
    index.record(("datasets",), Origin("qa.yaml", 1, 1))
    assert index.find(("checks", 0)) is None
    assert index.locate(("datasets", "x", "path")) == Origin("qa.yaml", 1, 1)
    assert index.mentioned("nothing configured here") is None


def test_alias_cycles_end(tmp_path) -> None:
    index = OriginIndex()
    index.alias(("a",), ("b",))
    index.alias(("b",), ("a",))
    assert index.find(("a", "x")) is None
