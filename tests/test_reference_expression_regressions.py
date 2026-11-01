"""Contract applicability and identities across the public execution backends."""

import pyarrow as pa
import pytest

from qavf import Dataset, validate

CONFIG = {
    "engine": {"parallel": False, "native_threads": 2},
    "plugins": {"entry_points": False},
    "report": {"hash_inputs": False},
}


def outcomes(report, rule):
    return [r for r in report.results if r.rule_id == rule and r.status.value != "SKIPPED"]


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("match", ["simple", "full"])
def test_na_composite_foreign_keys_are_exempt_before_cardinality(backend, match):
    child = Dataset(
        pa.table({"a": ["N/A", "N/A", "7"], "b": [1, 1, 1]}),
        schema={"columns": {"a": {"dtype": "str", "na_values": ["N/A"]}}},
        foreign_keys=[
            {
                "columns": ["a", "b"],
                "ref_dataset": "parent",
                "match": match,
                "cardinality": "one_to_one",
            }
        ],
    )
    report = validate(
        {"child": child, "parent": pa.table({"a": ["7"], "b": [1]})},
        backend=backend,
        config=CONFIG,
    )
    checked = outcomes(report, "INP-005")
    assert checked and all(r.status.value == "PASS" for r in checked)
    if backend == "polars":
        assert (
            sum(
                r.metrics.get("exempt_na_rows", {}).get("child(a, b) -> parent", 0) for r in checked
            )
            == 2
        )


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_parent_na_tokens_cannot_resolve_ordinary_child_values(backend):
    report = validate(
        {
            "child": Dataset(
                pa.table({"x": ["N/A"]}),
                foreign_keys=[
                    {
                        "columns": ["x"],
                        "ref_dataset": "parent",
                        "cardinality": "many_to_one",
                    }
                ],
            ),
            "parent": Dataset(
                pa.table({"x": ["N/A", "N/A"]}),
                schema={
                    "columns": {"x": {"dtype": "str", "na_values": ["N/A"]}},
                },
            ),
        },
        backend=backend,
        config=CONFIG,
    )
    checked = outcomes(report, "INP-005")
    failures = [r for r in checked if r.status.value == "FAIL"]
    assert len(failures) == 1 and failures[0].affected_count == 1
    assert not [r for r in checked if r.status.value == "ERROR"]


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_duplicate_expression_labels_do_not_replace_contract_identity(backend):
    report = validate(
        Dataset(
            pa.table({"x": [1]}),
            expressions=[
                {"name": "same", "expr": "x > 0"},
                {"name": "same", "expr": "x < 0"},
                {"name": "same", "expr": "unknown > 0"},
            ],
        ),
        backend=backend,
        config=CONFIG,
    )
    checked = outcomes(report, "INP-006")
    ledger = [
        (r.status.value, identity.parameters)
        for r in checked
        for identity in ([*r.checks] or [r.identity])
    ]
    assert sorted((s, p["expr"]) for s, p in ledger) == sorted(
        [("PASS", "x > 0"), ("FAIL", "x < 0"), ("ERROR", "unknown > 0")]
    )
    assert all(p["name"] == "same" for _, p in ledger)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_all_na_primary_keys_retain_specific_contract_identity(backend):
    report = validate(
        pa.table({"x": ["N/A", "N/A"]}),
        backend=backend,
        config=CONFIG,
        schema={"columns": {"x": {"dtype": "str", "na_values": ["N/A"]}}, "primary_key": ["x"]},
    )
    checked = outcomes(report, "INP-004")
    assert checked and all(r.status.value == "PASS" for r in checked)
    identities = [identity for r in checked for identity in [r.identity, *r.checks]]
    assert any(
        i.check_id == "key.primary" and i.columns == ["x"] and i.parameters["nulls"] == "forbid"
        for i in identities
    )
    assert all(r.check_id != "INP-004" for r in checked)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_conditional_duplicates_retain_original_positions_and_other_successes(backend):
    report = validate(
        pa.table({"x": ["7", "07", "7", "7"], "active": [False, True, False, True]}),
        backend=backend,
        config=CONFIG,
        schema={
            "columns": {"x": {"dtype": "int"}},
            "unique_keys": [
                {"columns": ["x"], "where": "active", "name": "selected"},
                {"columns": ["x"], "where": "x < 0", "name": "empty"},
            ],
            "duplicate_rows_per_key": "all",
        },
    )
    checked = outcomes(report, "INP-004")
    failure = next(r for r in checked if r.status.value == "FAIL")
    assert failure.affected_count == 1 and failure.affected_rows == 2
    assert failure.affected_records[0]["rows"] == [1, 3]
    assert failure.parameters["where"] == "active"
    successes = [
        i for r in checked if r.status.value == "PASS" for i in ([*r.checks] or [r.identity])
    ]
    assert any(i.parameters.get("name") == "empty" for i in successes)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("dataset", ["source", "groups", "domain", "counts"])
def test_grouped_domains_do_not_shadow_dataset_names(backend, dataset):
    report = validate(
        {
            dataset: Dataset(
                pa.table({"g": ["a"], "x": [7]}),
                schema={
                    "grouped_values": [{"column": "x", "group_by": ["g"], "allowed_values": [7]}],
                },
            )
        },
        backend=backend,
        config=CONFIG,
    )
    checked = outcomes(report, "INP-004")
    assert checked and all(r.status.value == "PASS" for r in checked), [r.message for r in checked]


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_conditional_key_counts_scope_nulls_and_na_exemptions(backend):
    report = validate(
        pa.table({"x": [None, "N/A", "N/A", "7"], "active": [True, True, False, False]}),
        backend=backend,
        config=CONFIG,
        schema={
            "columns": {"x": {"dtype": "int", "nullable": True, "na_values": ["N/A"]}},
            "unique_keys": [{"columns": ["x"], "where": "active", "nulls": "forbid"}],
        },
    )
    checked = outcomes(report, "INP-004")
    failure = next(r for r in checked if r.status.value == "FAIL")
    assert failure.affected_count == 1 and failure.total_records == 2
    assert sum(sum(r.metrics.get("exempt_na_rows", {}).values()) for r in checked) == 1
