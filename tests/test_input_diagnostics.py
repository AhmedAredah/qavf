"""Inputs read the same on every machine, and absent or unreadable inputs say why.

Zoned timestamps compare in UTC regardless of the host's zone; absent columns
name their likely spelling; unsupported workbook formats say what to do.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from qavf import Dataset, File, validate
from qavf.engine.severity import Status
from qavf.sources.csv_probe import CsvProbe
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


# =========================================================================
# Unreadable lines are located, once, with advice
# =========================================================================
CLEAN_ROWS = 25_000
"""More rows than the reader samples to choose column types (20,480)."""


def _late(tmp_path, last: str, name: str = "t.csv") -> str:
    """Clean rows past the reader's type sample, then ``last``."""
    path = tmp_path / name
    rows = "".join(f"{i},n{i},{i}\n" for i in range(CLEAN_ROWS))
    path.write_text(f"id,name,v\n{rows}{last}")
    return str(path)


def _errors(source: object, schema: dict) -> set[str]:
    report = validate({"t": Dataset(source, schema=schema)})
    return {r.message for r in report.results if r.status is Status.ERROR}


@pytest.mark.parametrize(
    ("last", "located", "advice"),
    [
        ("1,x,n/a\n", "line 25,002, column 'v' (cast): '1,x,n/a'", "declare the column's dtype"),
        ("1,Smith, John,5\n", "line 25,002 (too many columns)", "quote values"),
    ],
)
def test_an_unreadable_line_is_located_with_advice(tmp_path, last, located, advice) -> None:
    # Without a declared dtype, the reader types "v" from its first rows.
    errors = _errors(_late(tmp_path, last), {"columns": {"v": {"ge": 0}}})
    assert errors and all("1 row of" in e and located in e and advice in e for e in errors)
    assert not any("Possible fixes" in e or e.endswith("..") for e in errors)


def test_lossy_reads_report_the_lines_they_left_out(tmp_path) -> None:
    path = _late(tmp_path, "1,Smith, John,5\n2,a,b,c,d\n")
    schema = {"columns": {"v": {"dtype": "int"}}}
    lossy = File(path, options={"store_rejects": True}, acknowledge_lossy=True)
    [left_out] = [f for f in _failed(lossy, schema, "INP-002") if "unreadable" in f.check]
    assert left_out.affected_count == 2
    assert [(r["line"], r["error"]) for r in left_out.affected_records] == [
        (25_002, "TOO MANY COLUMNS"),
        (25_003, "TOO MANY COLUMNS"),
    ]
    tolerated = File(
        path, options={"store_rejects": True}, acknowledge_lossy=True, max_rejected_rows=2
    )
    assert _failed(tolerated, schema, "INP-002") == []


# =========================================================================
# Header names the reader had to change
# =========================================================================
def test_duplicate_and_blank_headers_are_reported(tmp_path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("id,v,v,,V\n1,2,-3,4,5\n")
    [schema_failure] = _failed(str(path), {"columns": {"v": {"ge": 0}}}, "INP-002")
    assert [(r["column"], r["problem"]) for r in schema_failure.affected_records] == [
        ("v_1", "duplicate header 'v', read as 'v_1'"),
        ("column3", "blank header, read as 'column3'"),
        ("V_2", "duplicate header 'V', read as 'V_2'"),
    ]


def test_an_unrecognized_header_row_is_explained(tmp_path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("id,name,v\n" + "".join(f"{i},n{i},{i}\n" for i in range(39)) + "40,a, b,5\n")
    [schema_failure] = _failed(str(path), {"columns": {"v": {}}}, "INP-002")
    problems = [r["problem"] for r in schema_failure.affected_records]
    assert any("header row was not recognized" in p for p in problems)


def _never_read(*_args: object) -> list[str | None]:
    raise AssertionError("a clean header must not be re-read")


def test_clean_headers_need_no_raw_header_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(CsvProbe, "raw_header", _never_read)
    path = tmp_path / "t.csv"
    path.write_text("id,v_1,column_a\n1,2,3\n")
    assert _failed(str(path), {"columns": {"id": {"dtype": "int"}}}, "INP-002") == []
