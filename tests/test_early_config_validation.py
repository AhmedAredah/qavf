"""CFG-10a: source settings and named series/metrics are validated when a configuration
loads, without reading data, and fail identically on both backends."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qavf import validate
from qavf.cli import EXIT_USAGE, main
from qavf.settings import QAVFSettings, load_settings
from qavf.sources.builtin import BUILTIN_SOURCES
from qavf.sources.parameters import BUILTIN_SOURCE_PARAMS

BAD_SOURCE = "datasets:\n  x:\n    source: {type: file, path: x.csv, headre: false}\n"
UNUSED_BAD_SERIES = "datasets:\n  x:\n    path: x.csv\n    series:\n      s: {bad_key: 7}\n"
UNUSED_BAD_METRIC = (
    "datasets:\n  x:\n    path: x.csv\n    metrics:\n"
    "      m: {value: v, entity: e, per: nodot, zzz: 1}\n"
)
CASES = {
    "source": (BAD_SOURCE, {("datasets", "x", "source", "headre")}),
    "series": (
        UNUSED_BAD_SERIES,
        {("datasets", "x", "series", "s", "bad_key"), ("datasets", "x", "series", "s", "value")},
    ),
    "metric": (
        UNUSED_BAD_METRIC,
        {("datasets", "x", "metrics", "m", "per"), ("datasets", "x", "metrics", "m", "zzz")},
    ),
}


def _project(tmp_path: Path, text: str) -> Path:
    (tmp_path / "x.csv").write_text("a\n1\n", encoding="utf-8")
    config = tmp_path / "qavf.yaml"
    config.write_text(text, encoding="utf-8")
    return config


def _locations(exc: ValidationError) -> set[tuple[str | int, ...]]:
    return {tuple(error["loc"]) for error in exc.errors()}


def test_model_validate_rejects_the_review_mapping() -> None:
    raw = {
        "datasets": {
            "x": {
                "path": "x.csv",
                "series": {"s": {"bad_key": 7}},
                "metrics": {"m": {"zzz": 1}},
            }
        }
    }
    with pytest.raises(ValidationError) as caught:
        QAVFSettings.model_validate(raw)
    assert {
        ("datasets", "x", "series", "s", "bad_key"),
        ("datasets", "x", "metrics", "m", "zzz"),
        ("datasets", "x", "metrics", "m", "entity"),
    } <= _locations(caught.value)


@pytest.mark.parametrize("case", sorted(CASES))
def test_load_fails_with_the_field_path(tmp_path, case) -> None:
    text, expected = CASES[case]
    with pytest.raises(ValidationError) as caught:
        load_settings(_project(tmp_path, text))
    assert expected <= _locations(caught.value)


@pytest.mark.parametrize("case", sorted(CASES))
def test_both_backends_fail_identically_before_reading(tmp_path, case) -> None:
    config = _project(tmp_path, CASES[case][0])
    failures = {}
    for backend in ("duckdb", "polars"):
        with pytest.raises(ValidationError) as caught:
            validate(config=config, backend=backend)
        failures[backend] = sorted((error["loc"], error["type"]) for error in caught.value.errors())
    assert failures["duckdb"] == failures["polars"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_cli_backends_report_the_same_usage_error(tmp_path, capsys, case) -> None:
    config = _project(tmp_path, CASES[case][0])
    messages = {}
    for backend in ("duckdb", "polars"):
        code = main(["run", "-c", str(config), "--backend", backend])
        assert code == EXIT_USAGE
        messages[backend] = capsys.readouterr().err
    assert messages["duckdb"] == messages["polars"]
    assert ".".join(str(part) for part in sorted(CASES[case][1])[0]) in messages["duckdb"]


def test_valid_named_blocks_and_sources_still_load(tmp_path) -> None:
    settings = load_settings(
        _project(
            tmp_path,
            "datasets:\n  x:\n    source: {type: file, path: x.csv, retries: 1}\n"
            "    series:\n      s: {value: a, time_column: t, group_by: [site]}\n"
            "    metrics:\n      m: {value: a, entity: e, per: y.aadt}\n"
            "  q:\n    source: {type: database, connection: db, table: t}\n"
            "connections:\n  db: {type: duckdb, path: ':memory:'}\n",
        )
    )
    assert settings.datasets["x"].series["s"]["group_by"] == ["site"]


def test_a_plugin_source_type_is_left_for_the_run() -> None:
    settings = QAVFSettings.model_validate(
        {"datasets": {"x": {"source": {"type": "http_csv", "anything": 1}}}}
    )
    assert settings.datasets["x"].source == {"type": "http_csv", "anything": 1}


def test_builtin_parameter_models_match_the_registry() -> None:
    assert {source.kind: source.Params for source in BUILTIN_SOURCES} == dict(BUILTIN_SOURCE_PARAMS)
