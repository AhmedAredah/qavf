"""Evidence locates its rows: ``_row`` everywhere, and the file for multi-file inputs.

Conditional required fields and too-frequent missing values list the missing
rows; row expressions number their violating rows; evidence from several files
names the file and the row within it.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from qavf import Dataset, validate
from qavf.engine.row_locator import FileIndex
from qavf.engine.severity import Status
from qavf.settings import ColumnSpec

ROWS = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "status": ["open", "closed", "closed", "open", "closed"],
        "closed_on": ["2024-01-01", None, "2024-02-01", None, None],
        "x": [1.0, None, None, None, 2.0],
    }
)


def _source(kind: str, tmp_path: Path) -> object:
    if kind == "csv":
        path = tmp_path / "t.csv"
        ROWS.write_csv(path)
        return str(path)
    return ROWS if kind == "frame" else ROWS.lazy()


def _failed(source: object, rule: str, **contract: object) -> list:
    report = validate({"t": Dataset(source, **contract)})
    return [r for r in report.results_for(rule) if r.status is Status.FAIL]


# =========================================================================
# Missing values are listed by row
# =========================================================================
@pytest.mark.parametrize("kind", ["csv", "frame", "lazy"])
def test_a_conditionally_required_field_lists_its_missing_rows(tmp_path, kind) -> None:
    check = {"name": "closed needs a date", "where": "status = 'closed'", "not_null": True}
    column = {"nullable": True, "conditions": [check]}
    schema = {"primary_key": ["id"], "columns": {"closed_on": column}}
    [missing] = _failed(_source(kind, tmp_path), "INP-003", schema=schema)
    assert (missing.affected_count, missing.total_records) == (2, 3)
    assert [(r["id"], r["_row"]) for r in missing.affected_records] == [(2, 1), (5, 4)]


@pytest.mark.parametrize("kind", ["csv", "frame", "lazy"])
def test_too_many_missing_values_are_listed(tmp_path, kind) -> None:
    schema = {"primary_key": ["id"], "columns": {"x": {"nullable": True, "max_null_fraction": 0.2}}}
    [missing] = _failed(_source(kind, tmp_path), "INP-003", schema=schema)
    assert "missing fraction 0.6" in missing.message
    assert [(r["id"], r["violates"]) for r in missing.affected_records] == [
        (2, "missing"),
        (3, "missing"),
        (4, "missing"),
    ]


def test_not_null_belongs_in_a_check() -> None:
    with pytest.raises(ValidationError, match="nullable: false"):
        ColumnSpec(not_null=True)


# =========================================================================
# Row expressions number their violating rows
# =========================================================================
@pytest.mark.parametrize("kind", ["csv", "frame", "lazy"])
def test_row_expression_failures_carry_their_rows(tmp_path, kind) -> None:
    expressions = [{"check": "expression", "expr": "t.x IS NULL OR status = 'open'"}]
    [violated] = _failed(_source(kind, tmp_path), "INP-006", checks=expressions)
    assert violated.affected_count == 1
    assert [(r["id"], r["_row"]) for r in violated.affected_records] == [(5, 4)]


# =========================================================================
# Evidence from several files names the file and the row within it
# =========================================================================
@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_evidence_from_several_files_names_each_file(tmp_path, suffix) -> None:
    first, second = tmp_path / f"a.{suffix}", tmp_path / f"b.{suffix}"
    for path, frame in [(first, ROWS.head(3)), (second, ROWS.tail(2))]:
        with duckdb.connect() as con:
            con.register("frame", frame.to_arrow())
            con.execute(f"COPY frame TO '{path.as_posix()}'")
    schema = {"primary_key": ["id"], "columns": {"x": {"nullable": True, "ge": 1.5}}}
    [bound] = _failed(str(tmp_path / f"*.{suffix}"), "INP-003", schema=schema)
    [record] = bound.affected_records
    assert (record["_row"], Path(record["_file"]).name, record["_file_row"]) == (
        0,
        first.name,
        0,
    )
    located = _failed(
        str(tmp_path / f"*.{suffix}"),
        "INP-003",
        schema={"primary_key": ["id"], "columns": {"x": {"nullable": True, "le": 1.5}}},
    )
    [record] = located[0].affected_records
    assert (record["_row"], Path(record["_file"]).name, record["_file_row"]) == (
        4,
        second.name,
        1,
    )


def test_single_file_evidence_needs_no_file(tmp_path) -> None:
    schema = {"columns": {"x": {"nullable": True, "ge": 1.5}}}
    [bound] = _failed(_source("csv", tmp_path), "INP-003", schema=schema)
    assert "_file" not in bound.affected_records[0]


def test_file_index_maps_positions_to_files() -> None:
    index = FileIndex([("a", 3), ("b", 0), ("c", 2)])
    assert [index.locate(p) for p in (0, 2, 3, 4, 5)] == [
        ("a", 0),
        ("a", 2),
        ("c", 0),
        ("c", 1),
        (None, None),
    ]
