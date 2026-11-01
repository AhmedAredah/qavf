"""Engine-independent exact decimal, Unicode, locale and datetime contracts."""

from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from qavf import File, Query, validate
from qavf.number_grammar import NumberGrammar
from qavf.settings import ColumnSpec
from qavf.text_policies import datetime_text, register_sql_functions, requires_sql_functions


@pytest.mark.parametrize(
    "spec,required",
    [
        ({"dtype": "float"}, False),
        ({"dtype": "bool"}, False),
        ({"dtype": "datetime"}, False),
        ({"dtype": "datetime", "datetime_unit": "ns"}, True),
        ({"dtype": "bool", "true_values": ["i\u0307"]}, True),
        ({"dtype": "str", "normalize": ["trim"]}, False),
        ({"dtype": "str", "normalize": ["NFC"]}, True),
        ({"dtype": "str", "checks": [{"normalize": ["casefold"]}]}, True),
    ],
)
def test_native_functions_are_requested_only_by_relevant_policies(spec, required):
    assert requires_sql_functions({"t": {"schema": {"columns": {"v": spec}}}}) is required


def test_optional_native_registration_needs_no_catalog_scan_or_large_memory():
    import duckdb

    with duckdb.connect(config={"threads": 1, "memory_limit": "1MB"}) as connection:
        register_sql_functions(connection)
        register_sql_functions(connection)
        assert connection.execute("SELECT __qavf_unicode_lower('İ')").fetchone() == ("i\u0307",)


def report(values, spec, backend, **schema):
    result = validate(
        pa.table({"v": values}), backend=backend, schema={"columns": {"v": spec}, **schema}
    )
    assert not [r.message for r in result.results if r.status.value == "ERROR"]
    return result


def invalid_types(result):
    return sum(r.affected_count for r in result.results if r.rule_id == "INP-002")


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_impossible_decimal_outlier_does_not_change_other_values(backend):
    result = report(
        ["1.25", "9" * 39, "1.2500"], {"dtype": "decimal"}, backend, unique_keys=[["v"]]
    )
    assert invalid_types(result) == 1
    duplicate = [r for r in result.results if r.rule_id == "INP-004" and r.affected_count]
    assert len(duplicate) == 1 and duplicate[0].affected_rows == 2


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("physical", [False, True])
def test_explicit_decimal_contract_never_rounds(backend, physical):
    values = ["99.99", "100", "1.234", "1.230", "-99.99"]
    if physical:
        values = pa.array([Decimal(v) for v in values], type=pa.decimal128(6, 3))
    result = report(
        values, {"dtype": "decimal", "decimal_precision": 4, "decimal_scale": 2}, backend
    )
    assert invalid_types(result) == 2


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_physical_decimal_domain_ignores_lexical_locale(backend):
    values = pa.array([Decimal("12.30"), Decimal("12.30")], type=pa.decimal128(6, 2))
    result = report(
        values,
        {
            "dtype": "decimal",
            "decimal_precision": 5,
            "decimal_scale": 2,
            "number_grammar": {"decimal_separator": ","},
        },
        backend,
        primary_key=["v"],
    )
    assert invalid_types(result) == 0
    assert sum(r.affected_rows or 0 for r in result.results if r.rule_id == "INP-004") == 2


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_unicode_boolean_lowercase_has_shared_meaning(backend):
    result = report(
        ["İ", "i\u0307", "NO", "i"],
        {"dtype": "bool", "true_values": ["i\u0307"], "false_values": ["no"]},
        backend,
    )
    assert invalid_types(result) == 1


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_ascii_boolean_words_accept_unicode_lowercase_equivalents(backend):
    assert (
        report(
            ["\u212a", "K", "n"],
            {"dtype": "bool", "true_values": ["k"], "false_values": ["n"]},
            backend,
        ).verdict()
        == "passed"
    )
    assert (
        invalid_types(
            report(["İ"], {"dtype": "bool", "true_values": ["i"], "false_values": ["n"]}, backend)
        )
        == 1
    )


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_unicode_normalization_and_casefold_apply_to_values_and_options(backend):
    result = report(
        ["Straße", "cafe\u0301"],
        {
            "dtype": "str",
            "normalize": ["casefold", "NFC"],
            "isin": ["STRASSE", "café"],
            "each_allowed_once": True,
        },
        backend,
    )
    assert result.verdict() == "passed"
    collided = report(
        ["Straße", "STRASSE"], {"dtype": "str", "normalize": ["casefold"], "unique": True}, backend
    )
    assert collided.verdict() == "failed"


@pytest.mark.parametrize(
    "form,values,options",
    [
        ("NFC", ["e\u0301"], ["é"]),
        ("NFD", ["é"], ["e\u0301"]),
        ("NFKC", ["①"], ["1"]),
        ("NFKD", ["①"], ["1"]),
    ],
)
@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_unicode_forms(backend, form, values, options):
    assert (
        report(values, {"dtype": "str", "normalize": [form], "isin": options}, backend).verdict()
        == "passed"
    )


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_locale_number_grammar_admits_only_unambiguous_grouping(backend):
    result = report(
        ["1.234,50", "1234,5", "+1.234,500", "12.34,5", "1,234.5", "1.234,567"],
        {
            "dtype": "decimal",
            "decimal_precision": 8,
            "decimal_scale": 2,
            "number_grammar": {"decimal_separator": ",", "grouping_separator": "."},
            "isin": ["1.234,50"],
            "convert_options": True,
        },
        backend,
    )
    assert invalid_types(result) == 3
    assert not [
        r
        for r in result.results
        if r.rule_id == "INP-003" and r.affected_count and "isin" in r.parameters
    ]


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_nanosecond_timezone_keys_are_lossless(backend):
    result = report(
        [
            "2024-01-01T00:00:00.123456789",
            "2024-01-01T05:00:00.123456789Z",
            "2024-01-01T05:00:00.123456788+00:00",
        ],
        {"dtype": "datetime", "datetime_unit": "ns", "timezone": "America/New_York"},
        backend,
        primary_key=["v"],
    )
    assert invalid_types(result) == 0
    duplicate = [r for r in result.results if r.rule_id == "INP-004" and r.affected_count]
    assert len(duplicate) == 1 and duplicate[0].affected_rows == 2


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
def test_physical_zoned_nanoseconds_preserve_distinct_keys(backend, zone):
    values = pa.array([1704067200123456789, 1704067200123456788], type=pa.timestamp("ns", tz=zone))
    result = report(
        values,
        {"dtype": "datetime", "datetime_unit": "ns", "timezone": zone},
        backend,
        primary_key=["v"],
    )
    assert result.verdict() == "passed"


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_physical_zoned_precision_contract_rejects_rounding(backend):
    values = pa.array([1704067200123456000, 1704067200123456789], type=pa.timestamp("ns", tz="UTC"))
    result = report(
        values, {"dtype": "datetime", "datetime_unit": "us", "timezone": "UTC"}, backend
    )
    assert invalid_types(result) == 1


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize(
    "unit,fraction,invalid",
    [("ms", "123000", 0), ("ms", "123001", 1), ("us", "123456789", 1), ("ns", "123456789", 0)],
)
def test_temporal_precision_rejects_loss(backend, unit, fraction, invalid):
    result = report(
        ["2024-01-01T00:00:00." + fraction],
        {"dtype": "datetime", "datetime_unit": unit, "timezone": "UTC"},
        backend,
    )
    assert invalid_types(result) == invalid


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_dst_fold_requires_a_policy_and_gap_is_invalid(backend):
    values = ["2024-11-03T01:30:00", "2024-03-10T02:30:00"]
    spec = {"dtype": "datetime", "timezone": "America/New_York"}
    assert invalid_types(report(values, spec, backend)) == 2
    assert invalid_types(report(values, {**spec, "ambiguous": "earliest"}, backend)) == 1
    assert (
        datetime_text(values[0], "us", spec["timezone"], "earliest", "raise")
        == "2024-11-03 05:30:00.000000"
    )
    assert (
        datetime_text(values[0], "us", spec["timezone"], "latest", "raise")
        == "2024-11-03 06:30:00.000000"
    )
    assert invalid_types(report(["2024-01-01Z", "2024-01-01+05:00"], spec, backend)) == 2


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_timezone_temporal_options_compare_instants(backend):
    result = report(
        ["2024-01-01T00:00:00"],
        {
            "dtype": "datetime",
            "timezone": "America/New_York",
            "isin": [datetime(2024, 1, 1, 5, tzinfo=UTC)],
            "ge": datetime(2024, 1, 1),
        },
        backend,
    )
    assert result.verdict() == "passed"


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_timezone_nanosecond_options_and_bounds_preserve_last_digit(backend):
    pandas = pytest.importorskip("pandas")
    stamp = pandas.Timestamp("2024-01-01T05:00:00.123456789Z")
    result = report(
        ["2024-01-01T00:00:00.123456789", "2024-01-01T00:00:00.123456788"],
        {
            "dtype": "datetime",
            "datetime_unit": "ns",
            "timezone": "America/New_York",
            "isin": [stamp],
            "ge": stamp,
        },
        backend,
    )
    assert invalid_types(result) == 0
    failures = [r for r in result.results if r.rule_id == "INP-003" and r.affected_count]
    assert failures and all(r.affected_count == 1 for r in failures)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("source", ["csv", "query"])
def test_timezone_contract_keeps_csv_and_query_precision(tmp_path, backend, source):
    values = ["2024-01-01T00:00:00.123456789Z", "2024-01-01T00:00:00.123456788Z"]
    if source == "csv":
        path = tmp_path / "dates.csv"
        path.write_text("v\n" + "\n".join(values) + "\n")
        data = File(path, options={"types": {"v": "TIMESTAMPTZ"}})
    else:
        data = Query("SELECT * FROM (VALUES ('" + "'), ('".join(values) + "')) t(v)")
    result = validate(
        data,
        backend=backend,
        schema={
            "columns": {"v": {"dtype": "datetime", "datetime_unit": "ns", "timezone": "UTC"}},
            "primary_key": ["v"],
        },
    )
    assert result.verdict() == "passed", [(r.status, r.message) for r in result.results if r.failed]


@pytest.mark.parametrize(
    "config",
    [
        {"decimal_precision": 3},
        {"decimal_precision": 2, "decimal_scale": 3},
        {"dtype": "str", "datetime_unit": "ns"},
    ],
)
def test_invalid_type_policy_rejected(config):
    with pytest.raises(ValueError):
        ColumnSpec.model_validate({"dtype": "decimal", **config})


@pytest.mark.parametrize(
    "grammar",
    [
        {"decimal_separator": ",", "grouping_separator": ","},
        {"grouping_separator": ".", "underscores": True},
        {"decimal_separator": " "},
        {"grouping_separator": "1"},
    ],
)
def test_ambiguous_number_separators_rejected(grammar):
    with pytest.raises(ValueError):
        NumberGrammar(**grammar)
