"""Declared parsing must see CSV spellings before native reader coercion."""

from __future__ import annotations

from decimal import Decimal

import pytest

from qavf import Dataset, File, validate


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("reader_option", ["types", "dtypes", "column_types", "columns"])
@pytest.mark.parametrize("raw", ["7.0", "2.5", "1_000"])
def test_explicit_numeric_reader_hints_cannot_bypass_integer_grammar(
    tmp_path, backend, reader_option, raw
):
    path = tmp_path / "data.csv"
    path.write_text(f"k\n{raw}\n")
    report = validate(
        File(path, options={"header": True, reader_option: {"k": "BIGINT"}}),
        schema={"columns": {"k": {"dtype": "int"}}},
        backend=backend,
    )
    assert report.verdict() == "failed"
    assert any(result.rule_id == "INP-002" and result.failed for result in report.results)


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("reader_option", ["types", "columns"])
def test_governed_reader_preserves_valid_opted_in_integer_spelling(
    tmp_path, backend, reader_option
):
    path = tmp_path / "data.csv"
    path.write_text("k\n7.0\n")
    report = validate(
        File(path, options={"header": True, reader_option: {"k": "BIGINT"}}),
        schema={"columns": {"k": {"dtype": "int", "number_grammar": {"int_decimal": True}}}},
        backend=backend,
    )
    assert report.verdict() == "passed"


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize(
    "dtype,native,raw,column,expected",
    [
        ("bool", "BOOLEAN", "1", {}, "failed"),
        ("bool", "BOOLEAN", "Y", {"true_values": ["Y"], "false_values": ["N"]}, "passed"),
        ("date", "DATE", "2024/01/02", {}, "failed"),
        ("date", "DATE", "01/02/2024", {"format": "%m/%d/%Y"}, "passed"),
        ("decimal", "DECIMAL(6,2)", "1.234", {"isin": [Decimal("1.234")]}, "passed"),
    ],
)
def test_other_declared_types_keep_original_values(
    tmp_path, backend, dtype, native, raw, column, expected
):
    path = tmp_path / "data.csv"
    path.write_text(f"k\n{raw}\n")
    report = validate(
        File(path, options={"header": True, "types": {"k": native}}),
        schema={"columns": {"k": {"dtype": dtype, **column}}},
        backend=backend,
    )
    assert report.verdict() == expected


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_unguverned_explicit_reader_type_retains_precedence(tmp_path, backend):
    path = tmp_path / "data.csv"
    path.write_text("k\n2.5\n")
    report = validate(
        Dataset(
            File(path, text_columns=["k"], options={"header": True, "types": {"k": "BIGINT"}}),
            schema={"min_rows": 1},
        ),
        backend=backend,
    )
    assert report.verdict() == "passed"
    assert "INT" in report.metadata.datasets[0].dtypes["k"].upper()


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_positional_numeric_reader_types_report_a_clear_conflict(tmp_path, backend):
    path = tmp_path / "data.csv"
    path.write_text("k\n2.5\n")
    report = validate(
        File(path, options={"header": True, "types": ["BIGINT"]}),
        schema={"columns": {"k": {"dtype": "int"}}},
        backend=backend,
    )
    assert report.verdict() != "passed"
    assert any("positional types conflicts" in result.message for result in report.results)


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_case_insensitive_reader_hints_do_not_create_duplicate_type_entries(tmp_path, backend):
    path = tmp_path / "data.csv"
    path.write_text("k\n2.5\n")
    report = validate(
        File(path, options={"header": True, "types": {"K": "BIGINT"}}),
        schema={"columns": {"k": {"dtype": "int"}}},
        backend=backend,
    )
    assert report.verdict() == "failed"
    assert any(result.rule_id == "INP-002" and result.failed for result in report.results)
