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
