"""Declared types in text, text columns, located duplicates and case-insensitive values.

A stray cell leaves a column as text: INP-002 locates it and INP-003 checks the
rest. Codes stay text through the reader. Duplicates carry their rows, and
allowed values can ignore case.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import Dataset, File, validate
from qavf.engine.severity import Status
from qavf.profiling import profile_input
from qavf.rules._declared_types import convert
from qavf.settings import ColumnSpec
from qavf.sources.parameters import FileParams


def _results(source: object, schema: dict, rule: str) -> list:
    report = validate({"t": Dataset(source, schema=schema)})
    return [r for r in report.results if r.rule_id == rule]


def _failed(source: object, schema: dict, rule: str) -> list:
    return [r for r in _results(source, schema, rule) if r.status is not Status.PASS]


def _csv(tmp_path: Path, text: str, name: str = "t.csv") -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# =========================================================================
# A declared type read as text: INP-002 locates the cells, INP-003 the rest
# =========================================================================
def test_a_stray_cell_is_located_and_the_limits_still_apply(tmp_path) -> None:
    rows = "".join(f"{i},{i}\n" for i in range(5))
    path = _csv(tmp_path, f"id,n\n{rows}5,12x\n6,-5\n")
    schema = {"primary_key": ["id"], "columns": {"n": {"dtype": "int", "ge": 0}}}

    [schema_failure] = _failed(path, schema, "INP-002")
    assert schema_failure.check == "t.n: dtype int"
    assert schema_failure.affected_count == 1 and schema_failure.total_records == 7
    assert schema_failure.affected_records == [
        {"id": 5, "n": "12x", "_row": 5, "violates": "not an int"}
    ]

    [bound] = _failed(path, schema, "INP-003")
    assert bound.status is Status.FAIL and bound.affected_count == 1
    assert bound.affected_records[0]["_row"] == 6 and bound.affected_records[0]["n"] == -5


def test_text_missing_markers_let_a_numeric_column_conform(tmp_path) -> None:
    path = _csv(tmp_path, "n\n1\nN/A\n3\n")
    schema = {"columns": {"n": {"dtype": "int", "nullable": True, "missing_values": ["N/A"]}}}
    assert _failed(path, schema, "INP-002") == []
    checked = {**schema["columns"]["n"], "ge": 2}
    [bound] = _failed(path, {"columns": {"n": checked}}, "INP-003")
    assert bound.affected_count == 1 and bound.affected_records[0]["_row"] == 0


@pytest.mark.parametrize("kind", ["csv", "frame"])
def test_text_markers_are_ignored_once_the_column_holds_its_type(tmp_path, kind) -> None:
    path = _csv(tmp_path, "n\n1.5\n-999\n3.0\n")
    source = path if kind == "csv" else pl.read_csv(path)
    spec = {"dtype": "float", "nullable": True, "missing_values": ["N/A", -999], "ge": 0}
    results = _results(source, {"columns": {"n": spec}}, "INP-003")
    assert all(r.status is Status.PASS for r in results), [r.message for r in results]


def test_invalid_cells_are_not_counted_as_missing(tmp_path) -> None:
    path = _csv(tmp_path, "n\n1\nx\n3\n")
    results = _results(path, {"columns": {"n": {"dtype": "int"}}}, "INP-003")
    assert all(r.status is Status.PASS for r in results)
    assert "t.n: not null" in results[0].metrics["passed_checks"]


@pytest.mark.parametrize(
    ("dtype", "text", "expected"),
    [
        ("int", [" 7", "+3", "1.0", "x"], [7, 3, None, None]),
        ("float", ["1e3", "2.5", "1,0"], [1000.0, 2.5, None]),
        ("decimal", ["3.50", "-1", "1e3"], [Decimal("3.50"), Decimal("-1.00"), None]),
        ("bool", ["TRUE", "false", "yes"], [True, False, None]),
        ("date", ["2024-01-05", "2024-13-01", "05/01/2024"], [date(2024, 1, 5), None, None]),
        (
            "datetime",
            ["2024-01-05 10:30:00", "2024-01-05T10:30", "2024-01-05", "noon"],
            [
                datetime(2024, 1, 5, 10, 30),
                datetime(2024, 1, 5, 10, 30),
                datetime(2024, 1, 5),
                None,
            ],
        ),
    ],
)
def test_text_converts_to_each_declared_type(dtype, text, expected) -> None:
    converted = convert(pl.Series("v", text), ColumnSpec(dtype=dtype))
    assert converted.values.to_list() == expected
    assert converted.invalid.to_list() == [value is None for value in expected]


def test_markers_and_blanks_are_missing_rather_than_invalid() -> None:
    spec = ColumnSpec(dtype="int", missing_values=["N/A", -999], blank_as_missing=True)
    converted = convert(pl.Series("v", ["N/A", " ", None, "-999", "x"]), spec)
    assert converted.invalid.to_list() == [False, False, False, False, True]
    assert converted.values.to_list() == [None, None, None, -999, None]


# =========================================================================
# Columns declared as text are read as text
# =========================================================================
def test_codes_declared_as_text_keep_leading_zeros(tmp_path) -> None:
    rows = "".join(f"48{i % 999:03d}\n" for i in range(30_000))
    path = _csv(tmp_path, f"code\n{rows}01001\n")
    schema = {"columns": {"code": {"dtype": "str", "regex": r"\d{5}"}}}
    assert _failed(path, schema, "INP-002") == []
    assert _failed(path, schema, "INP-003") == []


def test_declared_text_follows_renames_and_explicit_types_win(tmp_path) -> None:
    path = _csv(tmp_path, "Code,n\n48201,1\n")
    report = validate(
        {
            "t": Dataset(
                path,
                rename={"Code": "code"},
                schema={"columns": {"code": {"dtype": "str"}, "n": {"dtype": "str"}}},
            )
        }
    )
    assert report.metadata.datasets[0].dtypes == {"code": "String", "n": "String"}

    source = File(path, text_columns=["code", "absent"], options={"types": {"Code": "BIGINT"}})
    report = validate({"t": Dataset(source, schema={"min_rows": 1})})
    assert report.metadata.datasets[0].dtypes == {"Code": "Int64", "n": "Int64"}


def test_text_columns_apply_to_csv_only() -> None:
    with pytest.raises(ValidationError, match="applies to CSV"):
        FileParams(path="t.parquet", text_columns=["code"])


def test_profiles_draft_leading_zero_codes_as_text(tmp_path) -> None:
    rows = "".join(f"{i:05d},{i}\n" for i in range(1, 40))
    profile = profile_input(_csv(tmp_path, f"code,n\n{rows}"))
    columns = profile.schema["columns"]
    assert columns["code"]["dtype"] == "str" and columns["n"]["dtype"] == "int"


# =========================================================================
# Duplicates carry their rows
# =========================================================================
@pytest.mark.parametrize("kind", ["csv", "frame", "lazy"])
def test_duplicated_keys_list_their_rows(tmp_path, kind) -> None:
    text = "k,v\n1,a\n2,b\n1,c\n-9,d\n-9,e\n2,f\n"
    path = _csv(tmp_path, text)
    frame = pl.read_csv(path)
    source = {"csv": path, "frame": frame, "lazy": frame.lazy()}[kind]
    schema = {
        "primary_key": ["k"],
        "columns": {"k": {"dtype": "int", "nullable": False, "missing_values": [-9]}},
    }
    duplicated = [r for r in _failed(source, schema, "INP-004") if "duplicated" in r.message]
    assert duplicated[0].affected_records == [
        {"k": 1, "occurrences": 2, "rows": [0, 2]},
        {"k": 2, "occurrences": 2, "rows": [1, 5]},
    ]


def test_a_unique_column_lists_its_duplicated_rows(tmp_path) -> None:
    path = _csv(tmp_path, "opt\nA\nB\nA\nC\n")
    [unique] = _failed(path, {"columns": {"opt": {"unique": True}}}, "INP-003")
    assert unique.affected_count == 2
    assert [r["_row"] for r in unique.affected_records] == [0, 2]
    assert {r["violates"] for r in unique.affected_records} == {"unique"}


# =========================================================================
# Allowed values and uniqueness can ignore case
# =========================================================================
@pytest.mark.parametrize("kind", ["csv", "frame"])
def test_ignore_case_applies_to_allowed_values_and_uniqueness(tmp_path, kind) -> None:
    path = _csv(tmp_path, "s\nValid\nINVALID\nvalid\nmaybe\n")
    source = path if kind == "csv" else pl.read_csv(path)
    allowed = {"isin": ["valid", "invalid"], "ignore_case": True}
    [isin] = _failed(source, {"columns": {"s": allowed}}, "INP-003")
    assert [r["s"] for r in isin.affected_records] == ["maybe"]

    unique = {"unique": True, "ignore_case": True}
    [repeat] = _failed(source, {"columns": {"s": unique}}, "INP-003")
    assert [r["_row"] for r in repeat.affected_records] == [0, 2]
    assert _failed(source, {"columns": {"s": {"unique": True}}}, "INP-003") == []


def test_ignore_case_needs_text_membership_limits() -> None:
    with pytest.raises(ValidationError, match="goes with 'isin'"):
        ColumnSpec(ignore_case=True, regex="a")
    with pytest.raises(ValidationError, match="applies to str columns"):
        ColumnSpec(dtype="int", isin=[1], ignore_case=True)
