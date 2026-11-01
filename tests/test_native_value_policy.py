"""The optional dataframe path and DuckDB-only path admit identical literals."""

import operator as comparisons
from datetime import date, datetime
from decimal import Decimal

import duckdb
import polars as pl
import pytest

from qavf.core.policy import (
    ColumnValue,
    DecimalColumnScale,
    NativeValueDomain,
    decimal_places_condition,
    native_bound_condition,
)
from qavf.rules._predicates import row_tests
from qavf.settings import ColumnSpec
from qavf.value_policy import members


@pytest.mark.parametrize(
    "native,eager",
    [
        ("TINYINT", pl.Int8),
        ("BIGINT", pl.Int64),
        ("UBIGINT", pl.UInt64),
        ("FLOAT", pl.Float32),
        ("DOUBLE", pl.Float64),
        ("DECIMAL(38,3)", pl.Decimal(38, 3)),
        ("VARCHAR", pl.String),
        ("BOOLEAN", pl.Boolean),
        ("DATE", pl.Date),
        ("TIMESTAMP", pl.Datetime("us")),
    ],
)
def test_literal_admission_matches_optional_engine(native, eager):
    values = [
        None,
        float("nan"),
        float("inf"),
        "07",
        "1.0",
        True,
        False,
        7,
        7.0,
        7.1,
        9007199254740993,
        2**63,
        -(2**63),
        256,
        Decimal("0.1"),
        0.1,
        Decimal("12.500"),
        date(2024, 1, 1),
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 1),
    ]
    assert NativeValueDomain(native).members(values) == members(values, eager)


def test_typed_missing_markers_apply_after_text_conversion():
    spec = ColumnSpec(dtype="int", missing_values=[-999, "absent"])
    policy = ColumnValue.compile("v", "VARCHAR", spec)
    with duckdb.connect() as connection:
        rows = connection.execute(
            f"SELECT {policy.value}, {policy.invalid}, {policy.missing} "
            "FROM (VALUES ('-999'), ('absent'), ('7'), ('invalid')) t(v)"
        ).fetchall()
    assert rows == [(None, False, True), (None, False, True), (7, False, False), (None, True, True)]


def test_all_decimal_scales_share_one_preflight_and_preserve_exact_digits():
    columns = {
        "a": ColumnSpec(dtype="decimal"),
        "b": ColumnSpec(
            dtype="decimal", number_grammar={"underscores": True}, trim_whitespace="unicode"
        ),
    }
    with duckdb.connect() as connection:
        connection.execute("create table input(a varchar, b varchar)")
        connection.executemany(
            "insert into input values (?,?)",
            [("9007199254740993.125", "\u00a01_000.50\u00a0"), ("1.00", "1__0.0"), ("x", "2.5")],
        )
        scales = DecimalColumnScale.resolve_many(
            connection, "input", {"a": "VARCHAR", "b": "VARCHAR"}, columns
        )
        assert scales == {"a": 3, "b": 1}
        a = ColumnValue.compile("a", "VARCHAR", columns["a"], decimal_scale=scales["a"])
        b = ColumnValue.compile("b", "VARCHAR", columns["b"], decimal_scale=scales["b"])
        assert connection.execute(f"SELECT {a.value}, {b.value} FROM input").fetchall() == [
            (Decimal("9007199254740993.125"), Decimal("1000.5")),
            (Decimal("1.000"), None),
            (None, Decimal("2.5")),
        ]


@pytest.mark.parametrize(
    "dtype, values",
    [
        (pl.Float64, [0.0, 0.1, 0.125, 1.00000000000001, 9007199254740992.0, 1e-100]),
        (pl.Decimal(38, 3), [Decimal("0.000"), Decimal("7.125"), Decimal("9007199254740993.250")]),
        (pl.Int64, [7, 9007199254740993, 2**63 - 1]),
    ],
)
@pytest.mark.parametrize("places", [0, 1, 2, 10, 324])
def test_decimal_place_sql_is_identical_to_reference(dtype, values, places):
    frame = pl.DataFrame({"v": pl.Series(values, dtype=dtype)})
    expected = frame.with_columns(
        row_tests({"max_decimals": places}, dtype)["max_decimals"][1]("v").alias("pass")
    )["pass"].to_list()
    with duckdb.connect() as connection:
        connection.register("input", frame.to_arrow())
        actual_type = connection.execute("describe input").fetchone()[1]
        condition = decimal_places_condition("v", actual_type, places)
        actual = [r[0] for r in connection.execute(f"SELECT {condition} FROM input").fetchall()]
    assert actual == expected


@pytest.mark.parametrize("operator", ["ge", "le", "gt", "lt"])
@pytest.mark.parametrize(
    "bound", [Decimal("0.1"), Decimal("9007199254740993.1"), 10**100, -(10**100)]
)
@pytest.mark.parametrize(
    "dtype, values",
    [
        (pl.Float64, [0.0, 0.1, 9007199254740992.0, 9007199254740994.0]),
        (pl.Int64, [0, 1, 9007199254740993, 2**63 - 1]),
        (pl.Decimal(38, 3), [Decimal("0.100"), Decimal("9007199254740993.125")]),
    ],
)
def test_bound_sql_retains_exact_numeric_thresholds(operator, bound, dtype, values):
    frame = pl.DataFrame({"v": pl.Series(values, dtype=dtype)})
    expected = [getattr(comparisons, operator)(value, bound) for value in frame["v"]]
    with duckdb.connect() as connection:
        connection.register("input", frame.to_arrow())
        physical = connection.execute("describe input").fetchone()[1]
        condition = native_bound_condition("v", physical, operator, bound)
        actual = [r[0] for r in connection.execute(f"SELECT {condition} FROM input").fetchall()]
    assert actual == expected


def test_native_floating_values_declared_int_are_assessed_without_rounding():
    policy = ColumnValue.compile("v", "DOUBLE", ColumnSpec(dtype="int"))
    with duckdb.connect() as connection:
        rows = connection.execute(
            f"SELECT {policy.value}, {policy.invalid}, {policy.missing} "
            "FROM (VALUES (7.0::DOUBLE), (7.1), ('inf'::DOUBLE), ('nan'::DOUBLE)) t(v)"
        ).fetchall()
    assert rows == [
        (7.0, False, False),
        (7.1, True, False),
        (float("inf"), True, False),
        (None, False, True),
    ]
