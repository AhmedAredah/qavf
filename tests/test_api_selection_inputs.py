"""A dataset selection never adapts, converts or consumes an excluded input.

Both backends must leave an excluded Arrow stream unread, while a dataset a
selected check references (a foreign key's parent) still loads.
"""

from __future__ import annotations

import pytest

import qavf
from qavf.checks.selection import SelectionError

pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")

BACKENDS = ("duckdb", "polars")


class TrackedStream:
    """An Arrow C stream that counts how often it is consumed."""

    def __init__(self, table: object) -> None:
        self.table = table
        self.calls = 0

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        self.calls += 1
        return self.table.__arrow_c_stream__(requested_schema)


class BrokenStream:
    """An Arrow C stream whose conversion always fails."""

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        raise RuntimeError("excluded input was converted")


def _int_schema(column: str) -> dict[str, object]:
    return {"columns": {column: {"dtype": "int"}}}


@pytest.mark.parametrize("backend", BACKENDS)
def test_excluded_stream_is_never_consumed(backend: str) -> None:
    stream = TrackedStream(pa.table({"y": [1, 2, 3]}))
    report = qavf.validate(
        {
            "a": qavf.Dataset(pl.DataFrame({"x": [1, 2]}), schema=_int_schema("x")),
            "b": qavf.Dataset(stream, schema=_int_schema("y")),
        },
        datasets="a",
        backend=backend,
    )
    assert report.verdict("ERROR") == "passed"
    assert stream.calls == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_failing_excluded_input_does_not_fail_the_run(backend: str) -> None:
    report = qavf.validate(
        {
            "a": qavf.Dataset(pl.DataFrame({"x": [1, 2]}), schema=_int_schema("x")),
            "b": qavf.Dataset(BrokenStream(), schema=_int_schema("y")),
        },
        datasets="a",
        backend=backend,
    )
    assert report.verdict("ERROR") == "passed"


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_selection_fails_before_any_conversion(backend: str) -> None:
    stream = TrackedStream(pa.table({"y": [1]}))
    with pytest.raises(SelectionError):
        qavf.validate(
            {"b": qavf.Dataset(stream, schema=_int_schema("y"))},
            datasets="missing",
            backend=backend,
        )
    assert stream.calls == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_referenced_parent_of_a_selected_dataset_still_loads(backend: str) -> None:
    parent = TrackedStream(pa.table({"id": [1, 2]}))
    child = qavf.Dataset(
        pl.DataFrame({"id": [1, 3]}),
        schema=_int_schema("id"),
        foreign_keys=[{"columns": ["id"], "ref_dataset": "parent"}],
    )
    report = qavf.validate(
        {"child": child, "parent": qavf.Dataset(parent, schema=_int_schema("id"))},
        datasets="child",
        backend=backend,
    )
    assert report.verdict("ERROR") == "failed"  # 3 has no parent row
    assert parent.calls >= 1
