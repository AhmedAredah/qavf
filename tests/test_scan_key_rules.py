"""Public key contracts retain exact outcomes while sharing grouped scans."""

from unittest.mock import patch

import polars as pl
import pytest

from qavf import Query, Status, validate
from qavf.engine.context import DataContext
from qavf.engine.observation import Observations


def run_rule(tmp_path, backend, frames, rule, *, datasets=None, params=None, limit=1):
    inputs = {}
    for index, (name, frame) in enumerate(frames.items()):
        if backend == "lazy":
            inputs[name] = frame.lazy()
        else:
            path = tmp_path / f"{index}.parquet"
            frame.write_parquet(path)
            inputs[name] = Query(
                "SELECT * FROM read_parquet($path)", parameters={"path": str(path)}
            )
    calls = []
    original_sql, original_collect = DataContext.sql, DataContext.collect_many
    original_observed_collect = Observations.collect

    def record(result):
        if any(name.startswith("__qavf_summary_") for name in result.columns):
            calls.append(result.height)
        return result

    def sql(self, *args, **kwargs):
        return record(original_sql(self, *args, **kwargs))

    def collect(self, *args, **kwargs):
        return record(original_collect(self, *args, **kwargs))

    def observed_collect(self, *args, **kwargs):
        return record(original_observed_collect(self, *args, **kwargs))

    with (
        patch.object(DataContext, "sql", sql),
        patch.object(DataContext, "collect_many", collect),
        patch.object(Observations, "collect", observed_collect),
    ):
        report = validate(inputs, config={
            "plugins": {"entry_points": False},
            "engine": {
                "parallel": False,
                "run_unconfigured_rules": False,
                "max_affected_records": limit,
            },
            "datasets": datasets or {name: {} for name in frames},
            "rules": {rule: {"params": params or {}}},
        })
    assert all(result.status in (Status.PASS, Status.FAIL) for result in report.results), [
        result.message for result in report.results
    ]
    results = report.results_for(rule)
    assert results
    assert all(len(result.affected_records) <= limit for result in results)
    return results, calls


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("limit", [0, 1])
def test_primary_and_unique_contracts_share_one_grouped_scan(tmp_path, backend, primary, limit):
    frame = pl.DataFrame({
        "id": [1, 1, 2, 2, 3, None, -999],
        "kind": ["a", "a", "b", "b", "c", "z", "NA"],
    })
    schema = {
        "columns": {
            "id": {"nullable": not primary, "missing_values": [-999]},
            "kind": {"nullable": not primary, "missing_values": ["NA"]},
        },
        **({"primary_key": ["id", "kind"]} if primary else {"unique_keys": [["id", "kind"]]}),
    }
    results, calls = run_rule(
        tmp_path, backend, {"data": frame}, "INP-004",
        datasets={"data": {"schema": schema}}, limit=limit,
    )
    assert calls == [1]  # All/null/valid/duplicate subsets share one collection.
    failures = {result.check: result for result in results if result.status is Status.FAIL}
    label = "data" if primary else "data: unique (id, kind)"
    assert set(failures) == ({label, "data: null keys"} if primary else {label})
    duplicates = failures[label]
    assert (
        duplicates.affected_count, duplicates.affected_rows, duplicates.total_records
    ) == (2, 4, 3)
    assert duplicates.metrics["surplus_rows"] == 2
    for record in duplicates.affected_records:
        assert (record["id"], record["kind"]) in {(1, "a"), (2, "b")}
        assert record["occurrences"] == 2
    if primary:
        missing = failures["data: null keys"]
        assert (missing.affected_count, missing.total_records) == (2, 7)
    else:
        assert duplicates.metrics["rows_exempt_null_unique_keys"] == {"data (id, kind)": 2}


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("match", ["simple", "full"])
@pytest.mark.parametrize("cardinality", ["any", "many_to_one", "one_to_one"])
def test_foreign_key_fusion_preserves_null_and_cardinality_facts(
    tmp_path, backend, match, cardinality,
):
    child = pl.DataFrame({
        "id": [1, 1, 2, 2, None, None, -999, 3],
        "kind": ["a", "a", "b", "b", "a", None, "NA", "c"],
    })
    parent = pl.DataFrame({"pid": [1, 1, 4, None], "pkind": ["a", "a", "d", None]})
    datasets = {
        "child": {
            "schema": {"columns": {
                "id": {"missing_values": [-999]}, "kind": {"missing_values": ["NA"]},
            }},
            "foreign_keys": [{
                "columns": ["id", "kind"], "ref_dataset": "parent",
                "ref_columns": ["pid", "pkind"], "match": match, "cardinality": cardinality,
            }],
        },
        "parent": {},
    }
    results, calls = run_rule(
        tmp_path, backend, {"child": child, "parent": parent}, "INP-005", datasets=datasets,
    )
    assert calls == [1] * (2 if cardinality == "any" else 3)
    label = "child(id, kind) -> parent"
    failures = {result.check: result for result in results if result.status is Status.FAIL}
    expected = {label: 2}
    if match == "full":
        expected[f"{label}: partly null keys"] = 1
    if cardinality != "any":
        expected[f"{label}: parent keys not unique"] = 1
    if cardinality == "one_to_one":
        expected[f"{label}: child keys not unique"] = 2
    assert {name: result.affected_count for name, result in failures.items()} == expected
    assert (failures[label].affected_rows, failures[label].total_records) == (3, 3)
    for record in failures[label].affected_records:
        assert (record["id"], record["kind"], record["rows"]) in {(2, "b", 2), (3, "c", 1)}
    assert any(
        result.metrics.get("rows_exempt_null_keys") == {label: 2 if match == "full" else 3}
        for result in results
    )


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("right,expected", [
    ([1, 1], Status.PASS), ([2, 2], Status.FAIL), ([1, None], Status.FAIL), ([], Status.FAIL),
])
def test_batch_identity_uses_one_summary_per_dataset_without_losing_nanoseconds(
    tmp_path, backend, limit, right, expected,
):
    frames = {
        "left": pl.DataFrame({"batch": pl.Series([1, 1], dtype=pl.Datetime("ns"))}),
        "right": pl.DataFrame({"batch": pl.Series(right, dtype=pl.Datetime("ns"))}),
    }
    results, calls = run_rule(
        tmp_path, backend, frames, "REC-005", limit=limit,
        params={"batches": [{"datasets": ["left", "right"], "column": "batch"}]},
    )
    assert calls == [1, 1]  # Three subsets share each source's grouped evaluation.
    assert len(results) == 1 and results[0].status is expected


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_uppercase_private_frequency_name_cannot_hide_duplicate_primary_keys(tmp_path, backend):
    column = "__QAVF_OCCURRENCES"
    results, calls = run_rule(
        tmp_path, backend, {"data": pl.DataFrame({column: [0, 0, 0]})}, "INP-004",
        datasets={"data": {"schema": {
            "columns": {column: {"nullable": False}}, "primary_key": [column],
        }}},
    )
    assert calls == [1]
    assert len(results) == 1 and results[0].status is Status.FAIL
    assert (results[0].affected_count, results[0].affected_rows) == (1, 3)
    assert results[0].affected_records == [{column: 0, "occurrences": 3}]


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_uppercase_private_frequency_name_preserves_foreign_key_cardinality(tmp_path, backend):
    column = "__QAVF_ROWS"
    results, _ = run_rule(
        tmp_path, backend,
        {"child": pl.DataFrame({column: [0, 0, 9]}), "parent": pl.DataFrame({"pid": [0]})},
        "INP-005",
        datasets={
            "child": {"foreign_keys": [{
                "columns": [column], "ref_dataset": "parent", "ref_columns": ["pid"],
                "cardinality": "one_to_one",
            }]},
            "parent": {},
        },
    )
    label = f"child({column}) -> parent"
    failures = {result.check: result for result in results}
    assert set(failures) == {label, f"{label}: child keys not unique"}
    assert all(result.status is Status.FAIL for result in results)
    assert (failures[label].affected_count, failures[label].affected_rows) == (1, 1)
    assert failures[label].affected_records == [{column: 9, "rows": 1}]
    duplicate = failures[f"{label}: child keys not unique"]
    assert (duplicate.affected_count, duplicate.affected_rows) == (1, 2)
    assert duplicate.affected_records == [{column: 0, "occurrences": 2}]


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_uppercase_private_frequency_name_keeps_zero_batch_identifiers_valid(tmp_path, backend):
    column = "__QAVF_ROWS"
    results, calls = run_rule(
        tmp_path, backend,
        {name: pl.DataFrame({column: [0, 0]}) for name in ("left", "right")},
        "REC-005",
        params={"batches": [{"datasets": ["left", "right"], "column": column}]},
    )
    assert calls == [1, 1]
    assert len(results) == 1 and results[0].status is Status.PASS


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_uppercase_private_frequency_name_preserves_key_set_difference(tmp_path, backend):
    column = "__QAVF_ROWS"
    results, _ = run_rule(
        tmp_path, backend,
        {"left": pl.DataFrame({column: [0, 0]}), "right": pl.DataFrame({column: [1, 1]})},
        "REC-004", limit=2,
        params={"pairs": [{
            "left": {"dataset": "left", "columns": [column]},
            "right": {"dataset": "right", "columns": [column]},
        }]},
    )
    assert len(results) == 1 and results[0].status is Status.FAIL
    assert results[0].affected_count == 2
    assert {(row[column], row["side"]) for row in results[0].affected_records} == {
        (0, "only in left"), (1, "only in right"),
    }
