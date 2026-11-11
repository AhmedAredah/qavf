"""Explicit ``overrides`` patch known datasets and check instances (review CFG-09).

They apply after include, vars, defaults and ``across`` expansion and before
validation: mappings merge, lists and scalars replace, ``null`` is an error, an
unknown target names the closest match and an ambiguous name asks for the
instance id. Each applied override is recorded, and ``qavf explain`` lists them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from qavf import explain
from qavf.overrides import AppliedOverride
from qavf.settings import QAVFSettings, load_settings


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _base() -> dict[str, Any]:
    return {
        "vars": {"limit": 0.5},
        "defaults": {"change_rate": {"max": "${vars.limit}"}},
        "datasets": {
            "counts": {
                "path": "counts.csv",
                "schema": {"min_rows": 1, "columns": {"aadt": {"dtype": "int", "ge": 0}}},
                "series": {"by_site": {"value": "aadt", "time_column": "year"}},
                "checks": [
                    {
                        "check": "missing_periods",
                        "name": "no missing years",
                        "series": "by_site",
                        "expected": [2020, 2021],
                    },
                    {"check": "change_rate", "series": "by_site"},
                    {"check": "change_rate", "series": "by_site", "name": "jumps", "max": 0.3},
                ],
            },
            "stations": {
                "path": "{city}.csv",
                "across": {"city": ["houston", "austin"]},
                "checks": [{"check": "expression", "name": "positive", "expr": "x > 0"}],
            },
        },
        "checks": [{"check": "sql", "name": "nothing", "sql": "SELECT 1 WHERE false"}],
    }


def _with(overrides: Any) -> QAVFSettings:
    return QAVFSettings.model_validate({**_base(), "overrides": overrides})


def test_a_check_override_merges_into_the_item_after_defaults() -> None:
    settings = _with({"checks": {"datasets.counts.checks[1]": {"max": 0.2, "severity": "WARNING"}}})
    item = settings.datasets["counts"].checks[1]
    assert item.settings["max"] == 0.2
    assert item.severity is not None and item.severity.value == "WARNING"
    # The default still reaches items the override does not name.
    assert settings.datasets["counts"].checks[2].settings["max"] == 0.3
    [applied] = settings.applied_overrides
    assert applied == AppliedOverride(
        "checks", "datasets.counts.checks[1]", "datasets.counts.checks[1]", ("max", "severity")
    )


def test_a_check_is_addressed_by_its_name_or_its_instance_id() -> None:
    settings = _with(
        {"checks": {"no missing years": {"expected": [2019]}, "checks[0]": {"enabled": False}}}
    )
    assert settings.datasets["counts"].checks[0].settings["expected"] == [2019]
    assert settings.checks[0].enabled is False
    assert [a.instance for a in settings.applied_overrides] == [
        "datasets.counts.checks[0]",
        "checks[0]",
    ]


def test_a_dataset_override_merges_mappings_and_replaces_lists() -> None:
    settings = _with(
        {
            "datasets": {
                "counts": {
                    "schema": {"min_rows": 100, "columns": {"aadt": {"le": 10}}},
                    "description": "audited",
                }
            }
        }
    )
    counts = settings.datasets["counts"]
    assert counts.schema_spec is not None and counts.schema_spec.min_rows == 100
    column = counts.schema_spec.columns["aadt"]
    assert (column.ge, column.le) == (0, 10)
    assert counts.description == "audited"
    assert settings.applied_overrides[0].fields == (
        "schema.min_rows",
        "schema.columns.aadt.le",
        "description",
    )
    replaced = _with(
        {"datasets": {"counts": {"checks": [{"check": "change_rate", "series": "by_site"}]}}}
    )
    assert len(replaced.datasets["counts"].checks) == 1


def test_an_across_copy_and_its_checks_are_addressable() -> None:
    settings = _with(
        {
            "datasets": {"stations@houston": {"description": "gulf"}},
            "checks": {"datasets.stations@austin.checks[0]": {"expr": "x >= 0"}},
        }
    )
    assert settings.datasets["stations@houston"].description == "gulf"
    assert settings.datasets["stations@austin"].description is None
    austin = settings.datasets["stations@austin"].checks[0]
    assert austin.settings["expr"] == "x >= 0"
    assert settings.datasets["stations@houston"].checks[0].settings["expr"] == "x > 0"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"checks": {"no missing yaers": {"max": 1}}}, "did you mean 'no missing years'"),
        ({"datasets": {"count": {"description": "x"}}}, "did you mean 'counts'"),
        ({"datasets": {"stations": {"description": "x"}}}, "did you mean 'stations@"),
        ({"checks": {"positive": {"expr": "x > 1"}}}, "names 2 instances"),
        ({"checks": {"jumps": {"max": None}}}, "null is not allowed"),
        ({"datasets": {"counts": {"schema": {"min_rows": None}}}}, "null is not allowed"),
        ({"checks": {"jumps": {}}}, "non-empty mapping"),
        ({"checks": {"jumps": 0.2}}, "non-empty mapping"),
        ({"rules": {}}, "unknown section 'rules'"),
    ],
)
def test_override_errors_name_the_problem(overrides: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _with(overrides)


def test_an_ambiguous_name_lists_the_instance_ids() -> None:
    with pytest.raises(ValidationError) as caught:
        _with({"checks": {"positive": {"expr": "x > 1"}}})
    text = str(caught.value)
    assert "datasets.stations@houston.checks[0]" in text
    assert "datasets.stations@austin.checks[0]" in text


def test_overridden_values_are_validated() -> None:
    with pytest.raises(ValidationError, match="min_rows"):
        _with({"datasets": {"counts": {"schema": {"min_rows": "many"}}}})


def test_settings_without_overrides_record_none_and_round_trip() -> None:
    settings = _with({"checks": {"jumps": {"max": 0.1}}})
    dumped = settings.model_dump(mode="json", exclude_unset=True)
    assert "overrides" not in dumped
    again = QAVFSettings.model_validate(dumped)
    assert again.datasets["counts"].checks[2].settings["max"] == 0.1
    assert again.applied_overrides == ()
    assert QAVFSettings.model_validate(_base()).applied_overrides == ()


def test_overrides_from_included_files_apply_in_order_and_the_root_s_last(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "base.yaml",
        "datasets:\n"
        "  counts:\n"
        "    path: counts.csv\n"
        "    schema: {min_rows: 1}\n"
        "    checks:\n"
        "      - {check: expression, name: positive, expr: 'x > 0', severity: ERROR}\n",
    )
    _write(
        tmp_path / "team" / "team.yaml",
        "include: [../base.yaml]\n"
        "overrides:\n"
        "  checks:\n"
        "    positive: {severity: WARNING, expr: 'x >= 0'}\n"
        "  datasets:\n"
        "    counts: {schema: {min_rows: 10}}\n",
    )
    config = _write(
        tmp_path / "qa.yaml",
        "include: [team/team.yaml]\n"
        "vars: {rows: 100}\n"
        "overrides:\n"
        "  datasets:\n"
        "    counts: {schema: {min_rows: '${vars.rows}'}}\n",
    )
    settings = load_settings(config)
    counts = settings.datasets["counts"]
    assert counts.schema_spec is not None and counts.schema_spec.min_rows == 100
    item = counts.checks[0]
    assert item.settings["expr"] == "x >= 0"
    assert item.severity is not None and item.severity.value == "WARNING"
    assert [(a.instance, a.fields, a.origin) for a in settings.applied_overrides] == [
        ("datasets.counts", ("schema.min_rows",), "team/team.yaml"),
        ("datasets.counts.checks[0]", ("severity", "expr"), "team/team.yaml"),
        ("datasets.counts", ("schema.min_rows",), "qa.yaml"),
    ]


def test_explain_lists_the_applied_overrides(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "qa.yaml",
        "datasets:\n"
        "  counts:\n"
        "    path: counts.csv\n"
        "    checks:\n"
        "      - {check: expression, name: positive, expr: 'x > 0'}\n"
        "overrides:\n"
        "  checks:\n"
        "    positive: {severity: WARNING}\n",
    )
    plan = explain(config=config)
    assert plan.overrides == ("datasets.counts.checks[0] ('positive'): severity (line 8)",)


def test_a_file_path_set_by_an_included_override_is_relative_to_that_file(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "prod" / "prod.yaml",
        "overrides:\n  datasets:\n    counts: {path: data/counts.csv}\n",
    )
    config = _write(
        tmp_path / "qa.yaml",
        "include: [prod/prod.yaml]\ndatasets:\n  counts: {path: counts.csv}\n",
    )
    assert load_settings(config).datasets["counts"].path == "prod/data/counts.csv"


def test_the_editor_schema_accepts_one_overrides_mapping_only() -> None:
    import jsonschema

    from qavf.cli_core import config_json_schema

    schema = config_json_schema(QAVFSettings())
    validator = jsonschema.Draft202012Validator(schema)
    written = {
        "overrides": {
            "checks": {"no missing years": {"max": 0.2}},
            "datasets": {"counts": {"schema": {"min_rows": 100}}},
        }
    }
    assert not list(validator.iter_errors(written))
    assert list(validator.iter_errors({"overrides": {"rules": {}}}))
    assert list(validator.iter_errors({"overrides": [{"origin": "a.yaml", "checks": {}}]}))
    assert schema["properties"]["overrides"]["ui_group"] == "Reuse"


def test_the_explain_command_prints_the_applied_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from qavf.cli import EXIT_OK, main

    _write(
        tmp_path / "base.yaml",
        "datasets:\n  counts:\n    path: counts.csv\n    schema: {min_rows: 1}\n",
    )
    config = _write(
        tmp_path / "qa.yaml",
        "include: [base.yaml]\noverrides:\n  datasets:\n    counts: {schema: {min_rows: 5}}\n",
    )
    assert main(["explain", "-c", str(config)]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Overrides applied (1):\n  datasets.counts: schema.min_rows from qa.yaml:4\n" in text
