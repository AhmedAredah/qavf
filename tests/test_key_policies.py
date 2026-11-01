"""Canonical key equality, explicit null semantics and exact duplicate evidence."""

from datetime import date
from decimal import Decimal
from unittest.mock import patch

import polars as pl
import pyarrow as pa
import pytest
from pydantic import ValidationError

from qavf import Dataset, File, Query, Status, validate
from qavf.rules._duplicate_rows import NativeDuplicateRows
from qavf.settings import SchemaSpec


def checked(source, schema, *, evidence=50):
    report = validate(
        {"data": Dataset(source, schema=schema)},
        config={
            "plugins": {"entry_points": False},
            "engine": {
                "run_unconfigured_rules": False,
                "parallel": False,
                "max_affected_records": evidence,
            },
            "rules": {"INP-004": {}},
        },
    )
    results = report.results_for("INP-004")
    assert results and not any(result.status is Status.ERROR for result in results), [
        result.message for result in report.results
    ]
    return results


def source_for(tmp_path, kind, frame):
    if kind == "frame":
        return frame
    if kind == "lazy":
        return frame.lazy()
    if kind == "arrow":
        return frame.to_arrow()
    if kind == "pandas":
        return frame.to_pandas()
    if kind == "csv":
        path = tmp_path / "keys.csv"
        frame.write_csv(path)
        return File(path, options={"types": {name: "VARCHAR" for name in frame.columns}})
    path = tmp_path / "keys.parquet"
    frame.write_parquet(path)
    if kind == "parquet":
        return str(path)
    return Query("SELECT * FROM read_parquet($path)", parameters={"path": str(path)})


@pytest.mark.parametrize("kind", ["csv", "frame", "lazy", "arrow", "pandas", "parquet", "query"])
def test_declared_int_key_compares_converted_values_on_every_adapter(tmp_path, kind):
    source = source_for(tmp_path, kind, pl.DataFrame({"id": ["07", "7", "+7", "7.0", "8"]}))
    schema = {
        "primary_key": ["id"],
        "columns": {"id": {"dtype": "int", "number_grammar": {"int_decimal": True}}},
    }
    [result] = checked(source, schema)
    assert result.status is Status.FAIL
    assert (result.affected_count, result.affected_rows, result.total_records) == (1, 4, 2)
    assert result.affected_records == [{"id": 7, "occurrences": 4, "rows": [0, 1, 2, 3]}]
    assert result.check_id == "key.primary"
    assert result.columns == ["id"]
    assert result.parameters["nulls"] == "forbid"


@pytest.mark.parametrize("kind", ["frame", "query"])
@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        ("decimal", ["03.50", "3.5", "+3.500"], Decimal("3.5")),
        ("date", ["2024-01-01", "2024-01-01", "2024-01-02"], date(2024, 1, 1)),
        ("bool", ["TRUE", "true", "false"], True),
    ],
)
def test_key_conversion_matches_value_conversion_for_other_declared_types(
    tmp_path, kind, dtype, values, expected
):
    source = source_for(tmp_path, kind, pl.DataFrame({"id": values}))
    [result] = checked(source, {"primary_key": ["id"], "columns": {"id": {"dtype": dtype}}})
    assert result.status is Status.FAIL
    assert result.affected_records[0]["rows"] == ([0, 1, 2] if dtype == "decimal" else [0, 1])
    if dtype != "decimal":
        assert result.affected_records[0]["id"] == expected


@pytest.mark.parametrize("kind", ["frame", "query"])
def test_identifiers_declared_as_strings_preserve_leading_zeros(tmp_path, kind):
    source = source_for(tmp_path, kind, pl.DataFrame({"id": ["07", "7", "+7", "7.0"]}))
    [result] = checked(source, {"primary_key": ["id"], "columns": {"id": {"dtype": "str"}}})
    assert result.status is Status.PASS


@pytest.mark.parametrize("kind", ["frame", "query"])
def test_invalid_key_text_follows_null_contract_not_raw_text_equality(tmp_path, kind):
    source = source_for(tmp_path, kind, pl.DataFrame({"id": ["x", "x", "2"]}))
    [result] = checked(source, {"primary_key": ["id"], "columns": {"id": {"dtype": "int"}}})
    assert result.check == "data: null keys"
    assert result.affected_count == 2


@pytest.mark.parametrize("kind", ["frame", "lazy", "query"])
@pytest.mark.parametrize("policy", ["exempt", "equal", "forbid"])
def test_nullable_composite_key_policy_and_positions(tmp_path, kind, policy):
    frame = pl.DataFrame({"a": [1, 1, 1, 1, None, None], "b": [None, None, 2, 2, None, None]})
    source = source_for(tmp_path, kind, frame)
    schema = {
        "columns": {"a": {"nullable": True}, "b": {"nullable": True}},
        "unique_keys": [{"columns": ["a", "b"], "nulls": policy}],
    }
    results = checked(source, schema)
    duplicate = next(result for result in results if result.unit == "keys")
    assert duplicate.affected_count == (3 if policy == "equal" else 1)
    assert duplicate.affected_rows == (6 if policy == "equal" else 2)
    assert duplicate.parameters["nulls"] == policy
    rows = {tuple(record["rows"]) for record in duplicate.affected_records}
    assert rows == ({(0, 1), (2, 3), (4, 5)} if policy == "equal" else {(2, 3)})
    null_failures = [result for result in results if "null keys" in (result.check or "")]
    assert len(null_failures) == (1 if policy == "forbid" else 0)
    if null_failures:
        assert null_failures[0].affected_count == 4
    assert ("rows_exempt_null_unique_keys" in duplicate.metrics) == (policy == "exempt")


@pytest.mark.parametrize("kind", ["frame", "lazy", "query"])
@pytest.mark.parametrize("limit", [1, 12, "all"])
def test_duplicate_positions_limit_never_changes_exact_totals(tmp_path, kind, limit):
    frame = pl.DataFrame({"id": ["07", "7", "+7"] * 9 + ["8", "8"]})
    [result] = checked(
        source_for(tmp_path, kind, frame),
        {
            "primary_key": ["id"],
            "columns": {"id": {"dtype": "int"}},
            "duplicate_rows_per_key": limit,
        },
        evidence=1,
    )
    assert (result.affected_count, result.affected_rows, result.metrics["surplus_rows"]) == (
        2,
        29,
        27,
    )
    [record] = result.affected_records
    assert record["occurrences"] == 27
    assert record["rows"] == list(range(27 if limit == "all" else limit))
    assert result.metrics["duplicate_positions_available"] is True
    assert result.metrics["duplicate_positions_complete"] == (limit == "all")
    assert result.metrics["duplicate_positions_truncated"] == (limit != "all")


def test_duplicate_position_lookup_failure_is_explicit_and_keeps_exact_totals(tmp_path):
    source = source_for(tmp_path, "query", pl.DataFrame({"id": [1, 1, 1]}))
    with patch.object(NativeDuplicateRows, "locate", side_effect=RuntimeError("lookup failed")):
        [result] = checked(source, {"primary_key": ["id"]})
    assert result.affected_count == 1 and result.affected_rows == 3
    assert result.metrics["duplicate_positions_available"] is False
    assert result.metrics["duplicate_positions_complete"] is False
    assert result.affected_records == [{"id": 1, "occurrences": 3}]


@pytest.mark.parametrize("kind", ["frame", "lazy", "query"])
def test_na_tokens_are_counted_separately_and_preserve_original_positions(tmp_path, kind):
    source = source_for(tmp_path, kind, pl.DataFrame({"id": ["N/A", "07", "7", "N/A", None]}))
    results = checked(
        source, {"primary_key": ["id"], "columns": {"id": {"dtype": "int", "na_values": ["N/A"]}}}
    )
    duplicates = next(result for result in results if result.unit == "keys")
    nulls = next(result for result in results if result.unit == "rows")
    assert duplicates.affected_records[0]["rows"] == [1, 2]
    assert nulls.affected_count == 1
    assert any(result.metrics.get("exempt_na_rows") == {"data (id)": 2} for result in results)


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "0", "ALL"])
def test_invalid_duplicate_position_limits_are_rejected(limit):
    with pytest.raises(ValidationError):
        SchemaSpec(duplicate_rows_per_key=limit)


def test_legacy_unique_key_list_shape_and_default_are_preserved():
    schema = SchemaSpec(unique_keys=[["a", "b"]])
    assert schema.unique_keys == [["a", "b"]]
    assert schema.alternate_keys[0].nulls == "exempt"
    assert schema.duplicate_rows_per_key == 10


def test_sql_core_keys_share_policies_and_preserve_exact_totals():
    from qavf.core.runner import validate_core

    frame = pa.table({"id": ["N/A", "07", "+7", "7.0", None, None]})
    for nulls, expected_keys, expected_rows in [
        ("equal", 2, 5),
        ("exempt", 1, 3),
        ("forbid", 1, 3),
    ]:
        report = validate_core(
            {
                "data": Dataset(
                    frame,
                    schema={
                        "columns": {
                            "id": {
                                "dtype": "int",
                                "nullable": True,
                                "na_values": ["N/A"],
                                "number_grammar": {"int_decimal": True},
                            }
                        },
                        "unique_keys": [{"columns": ["id"], "nulls": nulls}],
                        "duplicate_rows_per_key": "all",
                    },
                )
            },
            config={"plugins": {"entry_points": False}},
        )
        failures = report.results_for("INP-004")
        assert not any(r.status is Status.ERROR for r in report.results), [
            r.message for r in report.results
        ]
        duplicates = next(r for r in failures if r.unit == "keys")
        assert (duplicates.affected_count, duplicates.affected_rows) == (
            expected_keys,
            expected_rows,
        )
        assert duplicates.metrics["exempt_na_rows"] == {"data (id)": 1}
        assert duplicates.check_id == "key.unique"
        assert duplicates.parameters == {"nulls": nulls, "rows_per_key": "all"}
        assert {tuple(record["rows"]) for record in duplicates.affected_records} == (
            {(1, 2, 3), (4, 5)} if nulls == "equal" else {(1, 2, 3)}
        )


@pytest.mark.parametrize("backend", ["legacy", "core"])
def test_all_duplicate_positions_respect_evidence_byte_budget(backend):
    from qavf.audit_codec import evidence_json_size
    from qavf.core.runner import validate_core

    run = validate if backend == "legacy" else validate_core
    report = run(
        {
            "data": Dataset(
                pa.table({"id": [7] * 10_000}),
                schema={
                    "primary_key": ["id"],
                    "duplicate_rows_per_key": "all",
                },
            )
        },
        config={
            "plugins": {"entry_points": False},
            "engine": {"max_evidence_bytes": 4096, "run_unconfigured_rules": False},
            "rules": {"INP-004": {}},
        },
    )
    [result] = report.results_for("INP-004")
    assert result.status is Status.FAIL and result.affected_rows == 10_000
    assert result.metrics["duplicate_positions_truncated"] is True
    assert result.metrics["duplicate_positions_complete"] is False
    assert evidence_json_size(result.affected_records) <= 4096
    assert result.affected_records and len(result.affected_records[0]["rows"]) > 0


def test_sql_core_unique_rows_uses_raw_rows_and_row_units():
    from qavf.core.runner import validate_core

    source = pa.table({"id": ["07", "7", "7"], "v": ["x", "x", "x"]})
    report = validate_core(
        {
            "data": Dataset(
                source,
                schema={
                    "columns": {"id": {"dtype": "int"}},
                    "unique_rows": True,
                },
            )
        },
        config={"plugins": {"entry_points": False}},
    )
    [result] = report.results_for("INP-004")
    assert result.unit == "rows" and result.affected_count == 2 and result.total_records == 3
    assert result.check == "data: duplicate rows"
    assert result.check_id == "key.unique_rows"
    assert result.affected_records == [
        {"id": "7", "v": "x", "_row": 1, "violates": "duplicate row"},
        {"id": "7", "v": "x", "_row": 2, "violates": "duplicate row"},
    ]


@pytest.mark.parametrize("backend", ["legacy", "core"])
def test_duplicate_positions_preserve_nanosecond_key_identity(tmp_path, backend):
    from qavf.core.runner import validate_core

    run = validate if backend == "legacy" else validate_core
    frame = pl.DataFrame({"stamp": pl.Series([0, 0, 1, 1], dtype=pl.Datetime("ns"))})
    source = source_for(tmp_path, "query", frame)
    report = run(
        {"data": Dataset(source, schema={"primary_key": ["stamp"]})},
        config={
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "rules": {"INP-004": {}},
        },
    )
    [result] = report.results_for("INP-004")
    assert result.status is Status.FAIL
    assert result.affected_count == 2 and result.affected_rows == 4
    assert len({str(record["stamp"]) for record in result.affected_records}) == 2
    assert [record["rows"] for record in result.affected_records] == [[0, 1], [2, 3]]


@pytest.mark.parametrize("kind", ["frame", "query"])
def test_numeric_key_missing_sentinels_apply_after_conversion(tmp_path, kind):
    frame = pl.DataFrame({"id": ["-0999", "-999", "+7", "07"]})
    results = checked(
        source_for(tmp_path, kind, frame),
        {
            "primary_key": ["id"],
            "columns": {"id": {"dtype": "int", "missing_values": [-999]}},
        },
    )
    nulls = next(result for result in results if result.unit == "rows")
    duplicates = next(result for result in results if result.unit == "keys")
    assert nulls.affected_count == 2
    assert duplicates.affected_records == [{"id": 7, "occurrences": 2, "rows": [2, 3]}]


def test_native_binary_key_positions_are_not_stringified(tmp_path):
    frame = pl.DataFrame({"id": [b"\x00\xff", b"\x00\xff", b"text"]})
    [result] = checked(source_for(tmp_path, "query", frame), {"primary_key": ["id"]})
    assert result.metrics["duplicate_positions_available"] is True
    assert result.affected_records[0]["rows"] == [0, 1]


def test_native_timezone_key_positions_remain_available():
    source = Query(
        "SELECT TIMESTAMPTZ '2024-01-01 00:00:00+00' AS id UNION ALL "
        "SELECT TIMESTAMPTZ '2023-12-31 19:00:00-05'"
    )
    [result] = checked(source, {"primary_key": ["id"]})
    assert result.metrics["duplicate_positions_available"] is True
    assert result.affected_records[0]["rows"] == [0, 1]
