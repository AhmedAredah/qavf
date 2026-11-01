"""Reader type hints cannot round decimal text before schema validation."""

from decimal import Decimal

import pytest

from qavf import Dataset, File, validate
from qavf.settings import DatasetSpec


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("option", ["types", "dtypes", "column_types", "columns"])
def test_decimal_reader_precision_never_erases_significant_digits(tmp_path, backend, option):
    path = tmp_path / "exact.csv"
    path.write_text("raw\n1.234\n9007199254740993.125\n")
    source = Dataset(
        File(path, options={option: {"raw": "DECIMAL(6,2)"}}),
        rename={"raw": "value"},
        schema={
            "columns": {
                "value": {
                    "dtype": "decimal",
                    "isin": [Decimal("1.234"), Decimal("9007199254740993.125")],
                }
            }
        },
    )
    report = validate({"data": source}, backend=backend)
    assert report.verdict() == "passed", [(r.rule_id, r.message) for r in report.results]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("allowed_underscores", [False, True])
def test_decimal_reader_hint_cannot_bypass_number_grammar(tmp_path, backend, allowed_underscores):
    path = tmp_path / "grammar.csv"
    path.write_text("value\n1_000.250\n")
    report = validate(
        File(path, options={"types": {"value": "DECIMAL(10,2)"}}),
        backend=backend,
        schema={
            "columns": {
                "value": {
                    "dtype": "decimal",
                    "number_grammar": {"underscores": allowed_underscores},
                    "isin": [Decimal("1000.250")],
                }
            }
        },
    )
    assert report.verdict() == ("passed" if allowed_underscores else "failed")
    assert not [r for r in report.results if r.status.value == "ERROR"]


def test_positional_numeric_reader_types_report_decimal_policy_conflict():
    spec = DatasetSpec.model_validate(
        {
            "source": {"type": "file", "path": "input.csv", "options": {"types": ["DECIMAL(6,2)"]}},
            "schema": {"columns": {"value": {"dtype": "decimal"}}},
        }
    )
    with pytest.raises(
        ValueError, match="positional types conflicts with declared column validation"
    ):
        spec.source_config()


def test_unhinted_decimal_is_read_as_raw_text():
    spec = DatasetSpec.model_validate(
        {
            "path": "input.csv",
            "schema": {"columns": {"value": {"dtype": "decimal"}}},
        }
    )
    assert spec.source_config()["text_columns"] == ["value"]
