"""Configuration contracts: no discarded constraints or lossy Decimal bounds."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from qavf.settings import Bounds, ColumnSpec, EngineSettings, QAVFSettings, load_settings


@pytest.mark.parametrize(
    "mapping",
    [
        "datasets: {d: {schema: {min_rows: 0, min_rows: 10}}}",
        "engine: {parallel: true, parallel: false}",
        "datasets: {d: {path: a.csv}, d: {path: b.csv}}",
    ],
)
def test_inline_duplicate_keys_are_rejected(tmp_path, mapping):
    config = tmp_path / "q.yaml"
    config.write_text(mapping)
    with pytest.raises(ValueError, match="duplicate key"):
        load_settings(config)


def test_yaml_merge_explicit_override_is_not_a_duplicate(tmp_path):
    config = tmp_path / "q.yaml"
    config.write_text(
        "datasets:\n"
        "  a: &base\n"
        "    path: a.csv\n"
        "    schema: {min_rows: 1}\n"
        "  b:\n"
        "    <<: *base\n"
        "    path: b.csv\n"
    )
    settings = load_settings(config)
    assert settings.datasets["b"].path == "b.csv"
    assert settings.datasets["b"].schema_spec.min_rows == 1


def test_duplicate_inside_merged_anchor_is_rejected(tmp_path):
    config = tmp_path / "q.yaml"
    config.write_text("datasets: {a: &base {path: a.csv, path: b.csv}, b: {<<: *base}}")
    with pytest.raises(ValueError, match="duplicate key"):
        load_settings(config)


def test_case_colliding_connection_names_rejected():
    with pytest.raises(ValidationError, match="differ only in case"):
        QAVFSettings.model_validate(
            {
                "connections": {
                    "Warehouse": {"type": "duckdb", "path": "a.duckdb"},
                    "warehouse": {"type": "duckdb", "path": "b.duckdb"},
                }
            }
        )


def test_duplicate_explicit_connections_rejected():
    with pytest.raises(ValidationError, match="duplicate source connections"):
        QAVFSettings.model_validate(
            {
                "connections": {"db": {"type": "duckdb", "path": "a.duckdb"}},
                "datasets": {
                    "d": {
                        "source": {
                            "type": "query",
                            "sql": "SELECT 1",
                            "connections": ["db", "db"],
                        }
                    }
                },
            }
        )


def test_decimal_bounds_remain_exact():
    bound = Decimal("9007199254740993.0000000001")
    assert Bounds(ge=bound).ge == bound
    assert isinstance(Bounds(ge=bound).ge, Decimal)
    with pytest.raises(ValidationError, match="finite"):
        Bounds(ge=Decimal("Infinity"))


@pytest.mark.parametrize(
    "name,value",
    [
        ("memory_budget_bytes", 0),
        ("cache_max_bytes", -1),
        ("spill_max_bytes", True),
        ("max_evidence_bytes", -1),
        ("native_threads", 0),
        ("timeout_grace_s", float("inf")),
        ("rule_timeout_s", float("nan")),
    ],
)
def test_resource_limits_validate(name, value):
    with pytest.raises(ValidationError):
        EngineSettings.model_validate({name: value})


def test_quoted_decimal_bounds_and_aggregates_are_exact():
    raw = "0.10000000000000000001"
    column = ColumnSpec.model_validate(
        {
            "dtype": "decimal",
            "ge": raw,
            "sum": {"ge": raw},
            "checks": [{"le": raw, "min": {"le": raw}}],
        }
    )
    assert column.ge == Decimal(raw) and isinstance(column.ge, Decimal)
    assert column.sum.ge == Decimal(raw)
    assert column.checks[0].le == Decimal(raw)
    assert column.checks[0].min.le == Decimal(raw)
