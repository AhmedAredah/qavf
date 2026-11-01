"""General grouped domain coverage and conditional composite-key contracts."""

import pyarrow as pa
import pytest

from qavf import validate
from qavf.settings import SchemaSpec


def run(data, schema, backend="duckdb"):
    return validate(
        pa.table(data), schema=schema, backend=backend, config={"engine": {"native_threads": 2}}
    )


def results(report):
    return [r for r in report.results if r.rule_id == "INP-004" and r.status.value != "SKIPPED"]


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_conditional_key_only_checks_canonical_selected_rows(backend):
    report = run(
        {"a": ["07", "7", "7"], "b": ["x", "x", "x"], "active": [True, False, False]},
        {
            "columns": {"a": {"dtype": "int"}},
            "unique_keys": [{"columns": ["a", "b"], "where": "active", "name": "active identity"}],
        },
        backend,
    )
    assert report.verdict() == "passed", [(r.status, r.message) for r in report.results]
    assert results(report)[0].parameters["name"] == "active identity"


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_grouped_exactly_once_reports_missing_duplicate_and_unexpected_values(backend):
    report = run(
        {"group": ["a", "a", "b", "b", "b"], "code": ["07", "8", "7", "7", "9"]},
        {
            "columns": {"code": {"dtype": "int"}},
            "grouped_values": [{"column": "code", "group_by": ["group"], "allowed_values": [7, 8]}],
        },
        backend,
    )
    findings = results(report)
    assert len(findings) == 1 and findings[0].status.value == "FAIL", [
        (r.status, r.message) for r in report.results
    ]
    assert findings[0].affected_count == 3
    evidence = findings[0].affected_records
    assert {(row["group"], row["code"], row["occurrences"]) for row in evidence} == {
        ("b", 7, 2),
        ("b", 8, 0),
        ("b", 9, 1),
    }


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_group_coverage_keeps_empty_filtered_and_explicit_absent_groups(backend):
    report = run(
        {"group": ["a", "b"], "code": [7, 7], "active": [True, False]},
        {
            "grouped_values": [
                {
                    "column": "code",
                    "group_by": ["group"],
                    "allowed_values": [7],
                    "where": "active",
                    "expected_groups": [["c"]],
                }
            ],
        },
        backend,
    )
    findings = results(report)
    assert len(findings) == 1 and findings[0].affected_count == 2, [
        (r.status, r.message) for r in report.results
    ]
    assert {row["group"] for row in findings[0].affected_records} == {"b", "c"}


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_group_nulls_compare_equal_and_na_members_are_exempt(backend):
    report = run(
        {"group": [None, None, "N/A"], "code": ["7", "8", "invalid"]},
        {
            "columns": {"group": {"na_values": ["N/A"], "nullable": True}},
            "grouped_values": [
                {"column": "code", "group_by": ["group"], "allowed_values": ["7", "8"]}
            ],
        },
        backend,
    )
    assert report.verdict() == "passed", [(r.status, r.message) for r in report.results]


def test_conditional_keys_on_same_columns_have_distinct_scope():
    schema = SchemaSpec.model_validate(
        {
            "unique_keys": [
                {"columns": ["x"], "where": "active"},
                {"columns": ["x"], "where": "NOT active"},
            ]
        }
    )
    assert len(schema.alternate_keys) == 2


@pytest.mark.parametrize(
    "raw",
    [
        {"column": "x", "group_by": ["g", "g"], "allowed_values": [1]},
        {"column": "x", "group_by": ["x"], "allowed_values": [1]},
        {"column": "x", "group_by": ["g"], "allowed_values": [None]},
        {"column": "x", "group_by": ["g"], "allowed_values": [1], "min_occurrences": 2},
        {"column": "x", "group_by": ["g"], "allowed_values": [1], "expected_groups": [[1, 2]]},
    ],
)
def test_grouped_contract_rejects_ambiguous_configuration(raw):
    with pytest.raises(ValueError):
        SchemaSpec.model_validate({"grouped_values": [raw]})


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_explicit_groups_are_assessed_for_completely_empty_input(backend):
    data = pa.table(
        {"group": pa.array([], type=pa.string()), "code": pa.array([], type=pa.int64())}
    )
    report = validate(
        data,
        backend=backend,
        schema={
            "min_rows": 0,
            "grouped_values": [
                {
                    "column": "code",
                    "group_by": ["group"],
                    "allowed_values": [1, 2],
                    "expected_groups": [["a"], ["b"]],
                }
            ],
        },
        config={"engine": {"native_threads": 2}},
    )
    assert results(report)[0].affected_count == 4


def test_bad_grouped_condition_does_not_suppress_independent_group_contract():
    report = run(
        {"group": ["a"], "code": [1]},
        {
            "grouped_values": [
                {"column": "code", "group_by": ["group"], "allowed_values": [1], "where": "absent"},
                {"column": "code", "group_by": ["group"], "allowed_values": [2]},
            ]
        },
    )
    assert [r.status.value for r in results(report)] == ["ERROR", "FAIL"]


@pytest.mark.parametrize("field", ["domain", "group"])
def test_grouped_contract_cannot_require_exempt_na_literal(field):
    rule = {"column": "x", "group_by": ["g"], "allowed_values": ["value"]}
    if field == "domain":
        rule["allowed_values"] = [" N/A "]
    else:
        rule["expected_groups"] = [[" N/A "]]
    with pytest.raises(ValueError, match="declared N/A token"):
        SchemaSpec.model_validate(
            {
                "columns": {"g": {"na_values": ["N/A"]}, "x": {"na_values": ["N/A"]}},
                "grouped_values": [rule],
            }
        )


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("na_selected", [True, False])
def test_conditional_null_key_denominator_matches_selected_population(backend, na_selected):
    report = run(
        {"x": [None, None, "N/A"], "active": [True, False, na_selected]},
        {
            "columns": {"x": {"nullable": True, "na_values": ["N/A"]}},
            "unique_keys": [{"columns": ["x"], "where": "active", "nulls": "forbid"}],
        },
        backend,
    )
    failure = next(r for r in results(report) if r.status.value == "FAIL")
    assert failure.affected_count == 1
    assert failure.total_records == 1 + int(na_selected)
