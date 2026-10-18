"""Physical partitioning and presentation order cannot change complete results."""

import polars as pl
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from qavf import Loader, Status

from .support.adapters import PARTITIONED_BACKENDS, adapted_frames
from .support.contracts import run_checks, signatures

pytestmark = pytest.mark.assurance


@pytest.mark.parametrize("backend", PARTITIONED_BACKENDS)
@given(
    keys=st.lists(st.integers(-8, 8), min_size=1, max_size=20),
    partition_size=st.integers(1, 7),
)
@example(keys=[1, 2, 3, 1], partition_size=1)
def test_duplicates_across_partitions_preserve_complete_counts(backend, keys, partition_size):
    # Every duplicate has a partner in a distant partition, with a distinct tail
    # key to catch accidental truncation or incorrectly reused partition state.
    values = [*keys, 1000, *reversed(keys)]
    frame = pl.DataFrame({"id": values, "payload": ["ignored"] * len(values)})
    datasets = {"data": {"schema": {"primary_key": ["id"]}}}
    baseline = run_checks({"data": frame}, "INP-004", datasets=datasets)
    with adapted_frames(
        {"data": frame.reverse()}, backend, partition_size=partition_size
    ) as inputs:
        actual = run_checks(inputs, "INP-004", datasets=datasets)
    assert signatures(actual) == signatures(baseline)
    failures = [result for result in actual if result.status is Status.FAIL]
    assert len(failures) == 1
    assert failures[0].affected_count == len(set(keys))
    assert failures[0].affected_rows == 2 * len(keys)
    assert failures[0].total_records == len(set(keys)) + 1


@pytest.mark.parametrize("backend", PARTITIONED_BACKENDS)
@given(size=st.integers(0, 25), partition_size=st.integers(1, 8))
def test_last_partition_violations_cannot_be_lost(backend, size, partition_size):
    frame = pl.DataFrame({"x": [*range(size), -1]})
    datasets = {"data": {"schema": {"columns": {"x": {"ge": 0}}}}}
    with adapted_frames({"data": frame}, backend, partition_size=partition_size) as inputs:
        results = run_checks(inputs, "INP-003", datasets=datasets)
    failures = [result for result in results if result.status is Status.FAIL]
    assert len(failures) == 1
    assert failures[0].affected_count == 1
    assert failures[0].total_records == size + 1
    assert failures[0].affected_records[0]["x"] == -1


@given(values=st.lists(st.integers(-100, 100), max_size=25))
def test_loader_is_evaluated_once_and_keeps_same_contract(values):
    calls = []
    frame = pl.DataFrame({"x": values}, schema={"x": pl.Int64})

    def load():
        calls.append(1)
        return frame.lazy()

    datasets = {"data": {"schema": {"columns": {"x": {"ge": 0, "le": 99}}}}}
    expected = run_checks({"data": frame}, "INP-003", datasets=datasets)
    actual = run_checks({"data": Loader(load)}, "INP-003", datasets=datasets)
    assert calls == [1]
    assert signatures(actual) == signatures(expected)
    assert sum(result.affected_count for result in actual) == sum(
        value < 0 or value > 99 for value in values
    )
