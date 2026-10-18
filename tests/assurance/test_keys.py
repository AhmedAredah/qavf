"""Exact key counts, null policies and cardinality agree with independent counters."""

import polars as pl
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from qavf import Status

from .support.adapters import BACKENDS, adapted_frames
from .support.contracts import run_checks
from .support.oracles import MissingPolicy, key_counts
from .support.strategies import KEY_ROWS

pytestmark = pytest.mark.assurance

POLICIES = (MissingPolicy((-999,)), MissingPolicy(("NA",), True))
COLUMNS = {
    "id": {"nullable": True, "missing_values": [-999]},
    "kind": {"nullable": True, "missing_values": ["NA"], "blank_as_missing": True},
}
JOIN_BACKENDS = [*BACKENDS, "query_lazy", "lazy_query"]


def frame_for(rows):
    return pl.DataFrame(rows, schema={"id": pl.Int64, "kind": pl.String}, orient="row")


def joined_backend(backend):
    if backend == "query_lazy":
        return {"child": "query", "parent": "lazy"}
    if backend == "lazy_query":
        return {"child": "lazy", "parent": "query"}
    return backend


@pytest.mark.parametrize("backend", BACKENDS)
@given(rows=KEY_ROWS, primary=st.booleans())
@example(rows=[], primary=True)
@example(rows=[(1, "a"), (1, "a"), (None, "a"), (-999, "NA")], primary=True)
@example(rows=[(None, "a"), (None, "a"), (1, "a"), (1, "a")], primary=False)
def test_composite_key_counts_match_python_counter(backend, rows, primary):
    counts, null_rows, _ = key_counts(rows, POLICIES)
    duplicated = {key: count for key, count in counts.items() if count > 1}
    schema = {
        "columns": {
            name: {**spec, "nullable": not primary} for name, spec in COLUMNS.items()
        },
        **({"primary_key": ["id", "kind"]} if primary else {"unique_keys": [["id", "kind"]]}),
    }
    with adapted_frames({"data": frame_for(rows)}, backend) as inputs:
        results = run_checks(inputs, "INP-004", datasets={"data": {"schema": schema}})
    failures = {result.check: result for result in results if result.status is Status.FAIL}
    expected = {}
    if primary and null_rows:
        expected["data: null keys"] = null_rows
    duplicate_label = "data" if primary else "data: unique (id, kind)"
    if duplicated:
        expected[duplicate_label] = len(duplicated)
    assert {name: result.affected_count for name, result in failures.items()} == expected
    if duplicated:
        result = failures[duplicate_label]
        assert result.total_records == len(counts)
        assert result.affected_rows == sum(duplicated.values())
        assert result.metrics["surplus_rows"] == sum(duplicated.values()) - len(duplicated)
        assert len(result.affected_records) == min(4, len(duplicated))
        for record in result.affected_records:
            assert duplicated[record["id"], record["kind"]] == record["occurrences"]
    if primary and null_rows:
        assert failures["data: null keys"].total_records == len(rows)
    if not primary and null_rows:
        assert any(
            result.metrics.get("rows_exempt_null_unique_keys") == {"data (id, kind)": null_rows}
            for result in results
        )


@pytest.mark.parametrize("backend", JOIN_BACKENDS)
@given(
    children=KEY_ROWS,
    parents=KEY_ROWS,
    match=st.sampled_from(["simple", "full"]),
    cardinality=st.sampled_from(["any", "many_to_one", "one_to_one"]),
)
@example(
    children=[(1, "a"), (2, "b"), (2, "b"), (None, "a"), (None, None)],
    parents=[(1, "a"), (1, "a")],
    match="full",
    cardinality="one_to_one",
)
def test_foreign_keys_match_exact_oracle(backend, children, parents, match, cardinality):
    child_counts, null_rows, partial_rows = key_counts(children, POLICIES)
    parent_counts, _, _ = key_counts(parents, POLICIES)
    orphans = {key: count for key, count in child_counts.items() if key not in parent_counts}
    datasets = {
        "child": {
            "schema": {"columns": COLUMNS},
            "foreign_keys": [
                {
                    "columns": ["id", "kind"],
                    "ref_dataset": "parent",
                    "match": match,
                    "cardinality": cardinality,
                }
            ],
        },
        "parent": {"schema": {"columns": COLUMNS}},
    }
    with adapted_frames(
        {"child": frame_for(children), "parent": frame_for(parents)}, joined_backend(backend)
    ) as inputs:
        results = run_checks(inputs, "INP-005", datasets=datasets)
    label = "child(id, kind) -> parent"
    expected = {}
    if orphans:
        expected[label] = len(orphans)
    if match == "full" and partial_rows:
        expected[f"{label}: partly null keys"] = partial_rows
    for side, counts in (("parent", parent_counts), ("child", child_counts)):
        constrained = cardinality == "one_to_one" or (
            side == "parent" and cardinality == "many_to_one"
        )
        duplicate_count = sum(count > 1 for count in counts.values())
        if constrained and duplicate_count:
            expected[f"{label}: {side} keys not unique"] = duplicate_count
    failures = {result.check: result for result in results if result.status is Status.FAIL}
    assert {name: result.affected_count for name, result in failures.items()} == expected
    if orphans:
        result = failures[label]
        assert result.affected_rows == sum(orphans.values())
        assert result.total_records == len(child_counts)
        assert len(result.affected_records) == min(4, len(orphans))
        for record in result.affected_records:
            assert orphans[record["id"], record["kind"]] == record["rows"]
    exempt_rows = null_rows - partial_rows if match == "full" else null_rows
    if exempt_rows:
        assert any(
            result.metrics.get("rows_exempt_null_keys") == {label: exempt_rows}
            for result in results
        )
    else:
        assert all(
            label not in result.metrics.get("rows_exempt_null_keys", {}) for result in results
        )


@pytest.mark.parametrize("backend", JOIN_BACKENDS)
@given(children=KEY_ROWS, parents=KEY_ROWS, reverse=st.booleans())
@example(children=[(1, "a"), (1, "a"), (None, "a")], parents=[(2, "b")], reverse=True)
def test_key_set_difference_ignores_order_and_multiplicity(backend, children, parents, reverse):
    # One valid shared key makes this a genuinely assessed comparison even when
    # the generated remainder contains no usable keys.
    children = [*children, (99, "anchor")]
    parents = [*parents, (99, "anchor")]
    left, _, _ = key_counts(children, POLICIES)
    right, _, _ = key_counts(parents, POLICIES)
    expected = set(left) ^ set(right)
    frames = {
        "child": frame_for(list(reversed(children)) if reverse else children),
        "parent": frame_for(parents + parents),
    }
    with adapted_frames(frames, joined_backend(backend)) as inputs:
        results = run_checks(
            inputs,
            "REC-004",
            datasets={name: {"schema": {"columns": COLUMNS}} for name in frames},
            params={
                "pairs": [
                    {
                        "left": {"dataset": "child", "columns": ["id", "kind"]},
                        "right": {"dataset": "parent", "columns": ["id", "kind"]},
                    }
                ]
            },
        )
    assert len(results) == 1
    result = results[0]
    assert result.status is (Status.FAIL if expected else Status.PASS)
    assert result.affected_count == len(expected)
    assert len(result.affected_records) == min(4, len(expected))
    for record in result.affected_records:
        key = record["id"], record["kind"]
        assert key in expected
        assert record["side"] == ("only in child" if key in left else "only in parent")
