"""Counterexamples from the 0.5.0 code-quality review, kept as regression tests.

Each test is named after the review finding it reproduces. A false pass, an
overwritten input or a silent omission must fail here before any fix is trusted.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from qavf import Dataset, File, ResourceLimits, validate
from qavf.engine.severity import Status
from qavf.rules._declared_types import convert
from qavf.settings import ColumnSpec
from qavf.sources.file import FileSource


def _statuses(report, *rules: str) -> list[tuple[str, Status]]:
    return [(r.rule_id, r.status) for r in report.results if r.rule_id in rules]


# =========================================================================
# F01: decimal text is converted exactly, never rounded
# =========================================================================
def test_f01_a_decimal_beyond_its_scale_is_invalid_not_rounded() -> None:
    original = "0." + "0" * 37 + "11"  # 1.1e-38 needs 39 fractional digits
    report = validate(
        {"d": Dataset(pl.DataFrame({"x": [original]}), schema={
            "columns": {"x": {"dtype": "decimal", "le": Decimal("1e-38")}}
        })}
    )
    [schema] = [r for r in report.results_for("INP-002") if r.status is Status.FAIL]
    assert schema.affected_records[0]["x"] == original
    assert report.verdict() == "failed"


@pytest.mark.parametrize(
    ("texts", "values", "invalid"),
    [
        (["1.2300", "12", "-3.5", ".5"], ["1.23", "12", "-3.5", "0.5"], [False] * 4),
        (["0." + "0" * 37 + "1"], ["1E-38"], [False]),
        (["0." + "0" * 37 + "11"], [None], [True]),
        # Both fit alone, not together in one 38-digit column: neither is rounded.
        (["9" * 38, "0.5"], ["9" * 38, None], [False, True]),
        (["0.25", "0.250000000000000000000000000000000000000000"], ["0.25", "0.25"], [False] * 2),
    ],
)
def test_f01_decimal_conversion_is_exact(texts, values, invalid) -> None:
    converted = convert(pl.Series("v", texts), ColumnSpec(dtype="decimal"))
    expected = [None if v is None else Decimal(v) for v in values]
    assert converted.values.to_list() == expected
    assert converted.invalid.to_list() == invalid


# =========================================================================
# F08: legitimate header names are never reported as broken
# =========================================================================
@pytest.mark.parametrize(
    ("header", "columns"),
    [("column0,column1", ["column0", "column1"]), ('"rate,count"', ["rate,count"]),
     ('"rate;count"', ["rate;count"])],
)
def test_f08_names_that_look_generated_are_accepted_when_written(tmp_path, header, columns):
    path = tmp_path / "t.csv"
    path.write_text(f"{header}\n{','.join('1' for _ in columns)}\n")
    report = validate(
        File(path, options={"header": True, "delim": ","}),
        schema={"columns": {c: {"ge": 0} for c in columns}},
    )
    assert report.verdict() == "passed", [r.affected_records for r in report.results_for("INP-002")]


# =========================================================================
# F18: declared-text checks stay within a modest memory target
# =========================================================================
def test_f18_text_conversion_needs_no_loaded_column() -> None:
    source = pl.DataFrame({"v": ["000000000000000001"] * 200_000}).lazy()
    report = validate(
        {"d": Dataset(source, schema={"columns": {"v": {"dtype": "int", "ge": 0}}})},
        resources=ResourceLimits(
            target_memory="1MiB", cache_memory=0, max_spill="32MiB", max_workers=1,
            native_threads=1, evidence_rows=0,
        ),
    )
    assert _statuses(report, "INP-002", "INP-003") == [
        ("INP-002", Status.PASS), ("INP-003", Status.PASS)
    ]


# =========================================================================
# Review I-6: Parquet file positions come from footers, not a scan
# =========================================================================
def test_parquet_file_rows_come_from_footers(tmp_path, monkeypatch) -> None:
    for name, rows in (("a", 3), ("b", 2)):
        pl.DataFrame({"x": [-1.0] * rows}).write_parquet(tmp_path / f"{name}.parquet")
    monkeypatch.setattr(FileSource, "_count_file_rows", _never_scanned)
    report = validate(
        str(Path(tmp_path) / "*.parquet"), schema={"columns": {"x": {"ge": 0}}}
    )
    [bound] = [r for r in report.results_for("INP-003") if r.status is Status.FAIL]
    assert [(Path(r["_file"]).name, r["_file_row"]) for r in bound.affected_records][-1] == (
        "b.parquet", 1
    )


def _never_scanned(*_args: object) -> list[tuple[str, int]]:
    raise AssertionError("Parquet row counts must come from footers")
