"""Polars and DuckDB recognize a declared type in text identically.

Typed CSV reads convert in DuckDB; frames and text columns convert in Polars. Every
text is either the same value in both engines or invalid in both.
"""

from __future__ import annotations

import math

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from qavf.text_types import TextType, polars_value, sql_value

TEXTS = [
    # integers and floats, with the spellings each engine's own cast treats differently
    "7", " 7 ", "\t8\n", "+3", "-0", "007", "1.0", "2.5", "1e3", "1E-2", ".5", "5.", "+.5",
    "0x10", "1,0", "1_000", "1__0", "1.2.3", "e5", "1e", ".e1", "1e+", "--1", "+-1", "1-2",
    "+", "-", ".", "inf", "-Infinity", "NaN", "-nan", "+inf", "9223372036854775807",
    "9223372036854775808", "١٢",
    # booleans
    "true", "FALSE", " True ", "yes", "Y", "N", "no", "1", "0",
    # dates and datetimes
    "2024-01-05", "2024-1-5", "+024-01-05", "2024-01-0a", "2024-02-29", "2023-02-29",
    "2024-13-01", "2024-01-05 10:30", "2024-01-05T10:30", "2024-01-05 10:30:15",
    "2024-01-05T10:30:00.123456", "2024-01-05T10:30:00.1234567", "2024-01-05T10:30:00Z",
    "2024-01-05 25:00", "01/05/2024", "1/5/2024", "13/40/2024", "Jan 5 2024", "-infinity",
    # missing
    "N/A", "", "  ", None,
]
SPECS = [
    TextType(dtype="int", missing=["N/A"], blank_as_missing=True),
    TextType(dtype="int"),
    TextType(dtype="float", missing=["N/A"], blank_as_missing=True),
    TextType(dtype="bool"),
    TextType(dtype="bool", true_values=["Y", "yes", "1"], false_values=["N", "no", "0"]),
    TextType(dtype="date", blank_as_missing=True),
    TextType(dtype="datetime"),
    TextType(dtype="date", format="%m/%d/%Y"),
    TextType(dtype="datetime", format="%m/%d/%Y"),
    TextType(dtype="date", format="%b %d %Y"),
]


def _same(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def _spec_id(spec: TextType) -> str:
    return f"{spec.dtype}-{spec.format or 'iso'}"


@pytest.mark.parametrize("spec", SPECS, ids=_spec_id)
def test_both_engines_accept_exactly_the_same_text(spec: TextType) -> None:
    frame = pl.DataFrame({"t": pl.Series(TEXTS, dtype=pl.String)})
    value, missing = polars_value(pl.col("t"), spec)
    polars = frame.select(value.alias("v"), (~missing & value.is_null()).alias("bad"))
    sql_v, sql_bad = sql_value('"t"', spec)
    with duckdb.connect() as con:
        con.register("f", frame.to_arrow())
        native = con.execute(f"SELECT {sql_v} AS v, {sql_bad} AS bad FROM f").pl()
    differences = [
        (text, a, b)
        for text, a, b, x, y in zip(
            TEXTS, polars["v"], native["v"], polars["bad"], native["bad"], strict=True
        )
        if not _same(a, b) or x != y
    ]
    assert differences == []


def test_formats_use_only_directives_both_engines_share() -> None:
    with pytest.raises(ValidationError, match="%z"):
        TextType(dtype="datetime", format="%Y-%m-%d %z")
    with pytest.raises(ValidationError, match="%H"):
        TextType(dtype="date", format="%Y-%m-%d %H")
    with pytest.raises(ValidationError, match="incomplete"):
        TextType(dtype="date", format="%Y-%m-%")


def test_true_and_false_words_cannot_overlap() -> None:
    with pytest.raises(ValidationError, match="both true and false"):
        TextType(dtype="bool", true_values=["Y"], false_values=[" y "])
