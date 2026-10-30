"""Inputs read the same on every machine, and absent or unreadable inputs say why.

Zoned timestamps compare in UTC regardless of the host's zone; absent columns
name their likely spelling; unsupported workbook formats say what to do.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from qavf import Dataset, validate
from qavf.engine.severity import Status
from qavf.sources.formats import infer_format


def _failed(source: object, schema: dict, rule: str) -> list:
    report = validate({"t": Dataset(source, schema=schema)})
    return [r for r in report.results_for(rule) if r.status is not Status.PASS]


# =========================================================================
# Zoned timestamps compare in UTC on every machine
# =========================================================================
def test_zoned_csv_timestamps_load_and_compare_in_utc(tmp_path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("ts\n2024-01-05T13:00:00Z\n2024-01-05T11:00:00Z\n")
    schema = {"columns": {"ts": {"dtype": "datetime", "ge": datetime(2024, 1, 5, 12)}}}
    report = validate({"t": Dataset(str(path), schema=schema)})
    assert report.metadata.datasets[0].dtypes["ts"].endswith("time_zone='UTC')")
    [bound] = [r for r in report.results_for("INP-003") if r.status is Status.FAIL]
    assert [r["_row"] for r in bound.affected_records] == [1]


# =========================================================================
# An absent column names its likely spelling
# =========================================================================
@pytest.mark.parametrize(
    ("declared", "present", "hint"),
    [
        ("id", "ID", "the data has 'ID'"),
        ("site_id", "Site ID", "the data has 'Site ID'"),
        ("volume", "volumes", "did you mean 'volumes'"),
    ],
)
def test_absent_columns_suggest_the_data_spelling(declared, present, hint) -> None:
    frame = pl.DataFrame({present: [1]})
    schema = {"primary_key": [declared], "columns": {declared: {"dtype": "int"}}}
    [schema_failure] = _failed(frame, schema, "INP-002")
    assert hint in schema_failure.affected_records[0]["hint"]
    [column] = _failed(frame, schema, "INP-003")
    assert hint in column.message
    [key] = _failed(frame, schema, "INP-004")
    assert hint in key.message


def test_unrelated_absent_columns_get_no_hint() -> None:
    [schema_failure] = _failed(pl.DataFrame({"a": [1]}), {"columns": {"zzz": {}}}, "INP-002")
    assert "hint" not in schema_failure.affected_records[0]


# =========================================================================
# Workbook formats
# =========================================================================
def test_macro_workbooks_read_as_excel_and_legacy_workbooks_say_why() -> None:
    assert infer_format("book.xlsm") == "excel"
    with pytest.raises(ValueError, match=r"save the workbook as \.xlsx"):
        infer_format("book.xls")
