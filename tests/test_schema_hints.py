"""Editor hints stay in step with the configuration models."""

from __future__ import annotations

import json

import pytest

from qavf.cli import main
from qavf.schema_hints import HINTS


def _schema(capsys, backend):
    assert main(["schema", "--backend", backend]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_every_hint_names_a_real_property(capsys, backend):
    definitions = _schema(capsys, backend)["$defs"]
    hinted = 0
    for name, hints in HINTS.items():
        if name not in definitions:
            continue
        properties = definitions[name]["properties"]
        for field in hints:
            assert field in properties, f"{name}.{field} is not a property"
            assert {"ui_group", "ui_title", "ui_advanced"} <= set(properties[field])
            hinted += 1
    assert hinted > 150


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_column_and_engine_fields_are_all_hinted(capsys, backend):
    definitions = _schema(capsys, backend)["$defs"]
    for name in ("ColumnSpec", "ColumnCheck", "SchemaSpec", "EngineSettings", "ReportSettings"):
        missing = set(definitions[name]["properties"]) - set(HINTS[name])
        assert not missing, f"{name} fields without hints: {sorted(missing)}"


_CONFIGURATION_MODELS = (
    "RunSettings",
    "TagPolicy",
    "ViewSpec",
    "ConsensusSettings",
    "ColumnStats",
    "Bounds",
    "_SideSettings",
)
"""Models of the 0.7 configuration sections: run policy, views, consensus, stats."""


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_every_configuration_section_and_check_setting_is_hinted(capsys, backend):
    schema = _schema(capsys, backend)
    root = {name for name in schema["properties"] if name != "$schema"}
    assert not {n for n in root if "ui_group" not in schema["properties"][n]}
    definitions = schema["$defs"]
    datasets = "NativeDataset" if backend == "duckdb" else "Dataset"
    checked = [datasets, *(n for n in _CONFIGURATION_MODELS if n in definitions)]
    checked += [n for n in definitions if n.startswith(("Check_", "Inline_"))]
    checked += [n for n in ("ExpressionItem", "SqlItem") if n in definitions]
    if backend == "polars":
        assert {"Check_compare", "ConsensusSettings", "Inline_series_trend"} <= set(checked)
    for name in checked:
        properties = definitions[name]["properties"]
        missing = [field for field, spec in properties.items() if "ui_group" not in spec]
        assert not missing, f"{name} fields without hints: {missing}"


def test_new_column_settings_are_hinted(capsys):
    definitions = _schema(capsys, "polars")["$defs"]
    column = definitions["ColumnSpec"]["properties"]
    for field in ("between", "stats", "conditions"):
        assert column[field]["ui_group"]
    assert "checks" not in column and "mean" not in column
    compare = definitions["Check_compare"]["properties"]
    assert compare["strategy"]["ui_title"] == "Pairing strategy"
    assert compare["consensus"]["ui_group"] == "Datasets"
