"""Configurable lexical admission is exact and shared by both execution engines."""

from decimal import Decimal

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from qavf import Dataset, validate
from qavf.number_grammar import UNICODE_WHITESPACE, NumberGrammar
from qavf.rules._declared_types import convert
from qavf.settings import ColumnSpec
from qavf.text_types import TextType, polars_value, sql_value


def accepted(texts, spec):
    value, missing = polars_value(pl.col("v"), spec)
    frame = pl.DataFrame({"v": pl.Series(texts, dtype=pl.String)})
    eager = frame.select(value.alias("parsed"), (~missing & value.is_null()).alias("bad"))
    sql, invalid = sql_value('"v"', spec)
    with duckdb.connect() as connection:
        connection.register("input", frame.to_arrow())
        native = connection.execute(f"SELECT {sql}, {invalid} FROM input").fetchall()
    assert eager.rows() == native
    return [row[0] for row in native]


@pytest.mark.parametrize("policy", ["ascii", "unicode", "none"])
def test_whitespace_is_an_explicit_identical_character_set(policy):
    chars = list(UNICODE_WHITESPACE)
    texts = [f"{char}7{char}" for char in chars]
    values = accepted(texts, TextType(dtype="int", trim_whitespace=policy))
    expected = [
        7 if policy == "unicode" or (policy == "ascii" and char == " ") else None for char in chars
    ]
    assert values == expected


def test_integral_decimal_never_rounds_and_preserves_large_integer_digits():
    texts = [
        "07",
        "+7",
        "7.0",
        "7.",
        "7.00000000000000000000000001",
        "9007199254740993.0",
        "9223372036854775807.0",
        "9223372036854775808.0",
    ]
    values = accepted(texts, TextType(dtype="int", number_grammar={"int_decimal": True}))
    assert values == [7, 7, 7, 7, None, 9007199254740993, 9223372036854775807, None]
    assert accepted(["7.0"], TextType(dtype="int")) == [None]


def test_ascii_numeric_fast_path_cannot_hide_unicode_in_a_prepared_alias():
    spec = TextType(dtype="int", number_grammar={"int_decimal": True, "underscores": True})
    sql, invalid = sql_value("v", spec, "trim(v)")
    with duckdb.connect() as connection:
        rows = connection.execute(
            f"SELECT {sql}, {invalid} FROM (VALUES "
            "('  +7.0  '), (' 1_000.0_0 '), ('\u00a07\u00a0'), ('\t7\t')) t(v)"
        ).fetchall()
    assert rows == [(7, False), (1000, False), (None, True), (None, True)]


@pytest.mark.parametrize("dtype", ["int", "float"])
def test_underscores_only_separate_digits(dtype):
    texts = ["1_000", "1__000", "_1", "1_", "+1_2", "1_.0", "1._0"]
    assert accepted(texts, TextType(dtype=dtype, number_grammar={"underscores": True})) == [
        1000,
        None,
        None,
        None,
        12,
        None,
        None,
    ]
    assert accepted(["1_000"], TextType(dtype=dtype)) == [None]


def test_float_exponents_and_special_tokens_can_be_disabled():
    spec = TextType(dtype="float", number_grammar={"exponent": False, "special_values": False})
    assert accepted(["7.0", "1e3", "nan", "+INF", "-infinity"], spec) == [
        7.0,
        None,
        None,
        None,
        None,
    ]


def test_decimal_grammar_is_exact_and_uses_the_same_whitespace_setting():
    spec = ColumnSpec(
        dtype="decimal", number_grammar={"underscores": True}, trim_whitespace="unicode"
    )
    values = convert(pl.Series(["\u00a01_000.250\u00a0", "1__0.5", "1e2"]), spec)
    assert values.values.to_list() == [Decimal("1000.25"), None, None]
    assert values.invalid.to_list() == [False, True, True]


def test_option_conversion_is_opt_in_and_preserves_exact_integer_literals():
    raw = {"dtype": "int", "isin": ["07", "+7", "7.0", "9007199254740993.0"]}
    assert ColumnSpec(**raw).all_checks()[0].isin == raw["isin"]
    spec = ColumnSpec(**raw, convert_options=True, number_grammar={"int_decimal": True})
    assert spec.all_checks()[0].isin == [7, 7, 7, 9007199254740993]
    with pytest.raises(ValidationError, match="numeric dtype"):
        ColumnSpec(dtype="str", isin=["07"], convert_options=True)


@pytest.mark.parametrize(
    "values, passed",
    [([1, 2], True), ([1], False), ([1, 1], False), ([1, 2, 3], False), ([], False)],
)
def test_each_allowed_once_is_a_builtin_contract(values, passed):
    source = pl.DataFrame({"v": pl.Series(values, dtype=pl.Int64)})
    report = validate(
        {
            "t": Dataset(
                source,
                schema={
                    "columns": {
                        "v": {
                            "dtype": "int",
                            "isin": ["01", 1, "2.0"],
                            "convert_options": True,
                            "number_grammar": {"int_decimal": True},
                            "each_allowed_once": True,
                        }
                    }
                },
            )
        }
    )
    results = [result for result in report.results if result.rule_id == "INP-003"]
    assert results
    assert all(str(result.status.value) == "PASS" for result in results) == passed


def test_case_collisions_and_normalized_domain_cardinality():
    spec = ColumnSpec(dtype="str", isin=["A", "a", "B"], normalize=["case"], each_allowed_once=True)
    check = spec.all_checks()[0]
    assert check.unique and check.distinct_count.ge == check.distinct_count.le == 2
    report = validate(
        {"t": Dataset(pl.DataFrame({"v": ["a", "A", "B"]}), schema={"columns": {"v": spec}})}
    )
    failed = [
        result
        for result in report.results
        if result.rule_id == "INP-003" and result.status.value == "FAIL"
    ]
    assert failed and failed[0].affected_count == 2


def test_domain_trim_preserves_existing_unicode_white_space_contract():
    spec = ColumnSpec(
        dtype="str",
        isin=["A", "\x1cA\x1c", "\u00a0A\u00a0"],
        normalize=["trim"],
        each_allowed_once=True,
    )
    assert spec.all_checks()[0].distinct_count.ge == 2


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_normalized_coverage_preserves_unicode_lowercase_semantics(backend):
    import pyarrow as pa

    schema = {
        "columns": {
            "v": {
                "dtype": "str",
                "isin": ["İ", "\u039f\u03a3"],
                "normalize": ["case"],
                "each_allowed_once": True,
            }
        }
    }
    report = validate(pa.table({"v": ["i\u0307", "\u03bf\u03c2"]}), schema=schema, backend=backend)
    assert report.verdict() == "passed"
    report = validate(pa.table({"v": ["i", "\u03bf\u03c3"]}), schema=schema, backend=backend)
    assert report.verdict() == "failed"


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_each_allowed_once_exempts_nullable_values_and_preserves_exact_counts(backend):
    import pyarrow as pa

    schema = {
        "columns": {
            "v": {"dtype": "int", "isin": [1, 2], "each_allowed_once": True, "nullable": True}
        }
    }
    report = validate(pa.table({"v": [1, 2, None]}), schema=schema, backend=backend)
    assert report.verdict() == "passed"
    report = validate(pa.table({"v": [1, 1, 2, None]}), schema=schema, backend=backend)
    assert report.verdict() == "failed"
    assert not [r for r in report.results if r.status.value == "ERROR"]
    duplicates = [r for r in report.results if r.rule_id == "INP-003" and r.affected_count == 2]
    assert duplicates


def test_domain_configuration_rejects_missing_and_inconsistent_cardinality():
    with pytest.raises(ValidationError, match="requires 'isin'"):
        ColumnSpec(dtype="int", each_allowed_once=True)
    with pytest.raises(ValidationError, match="missing values"):
        ColumnSpec(dtype="int", isin=[1, None], each_allowed_once=True)
    with pytest.raises(ValidationError, match="contradicts"):
        ColumnSpec(dtype="int", isin=[1, 2], each_allowed_once=True, distinct_count={"le": 1})


def test_configuration_literals_do_not_pass_through_float():
    grammar = NumberGrammar(int_decimal=True)
    assert grammar.parse("9007199254740993.0", "int") == 9007199254740993
    with pytest.raises(ValueError):
        grammar.parse("7.00000000000000000001", "int")
