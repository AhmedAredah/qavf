"""Generated keyed regression oracles independent of QAVF comparison kernels."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import polars as pl
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from qavf.audit_codec import evidence_json_size
from qavf.regression import compare_frames, compare_outputs
from qavf.regression.compare_outputs import ComparisonResult

from .support.adapters import adapted_frames

KEYS = ("id", "group")
VALUES = ("quantity", "label", "nested")
SCHEMA = {
    "id": pl.Int64,
    "group": pl.String,
    "quantity": pl.Int64,
    "label": pl.String,
    "nested": pl.List(pl.Int64),
}
_keys = st.tuples(st.one_of(st.none(), st.integers(-5, 5)), st.sampled_from(["a", "b", ""]))
_values = st.tuples(
    st.one_of(st.none(), st.integers(-(2**55), 2**55)),
    st.one_of(st.none(), st.sampled_from(["", "\u03b1", "0", "same", "different"])),
    st.one_of(st.none(), st.lists(st.one_of(st.none(), st.integers(-5, 5)), max_size=4)),
)
_tables = st.dictionaries(_keys, _values, max_size=10)


def _frame(rows):
    return pl.DataFrame(
        [(*key, *values) for key, values in rows.items()], schema=SCHEMA, orient="row"
    )


def _oracle(baseline, candidate):
    """Count set differences and Python cell inequality on exact typed values."""
    base_keys, cand_keys = set(baseline), set(candidate)
    events = {("row_added", key, None) for key in cand_keys - base_keys}
    events |= {("row_removed", key, None) for key in base_keys - cand_keys}
    changed_rows = set()
    changed_values = 0
    for key in base_keys & cand_keys:
        for index, column in enumerate(VALUES):
            if baseline[key][index] != candidate[key][index]:
                changed_rows.add(key)
                changed_values += 1
                events.add(("value_changed", key, column))
    return (
        len(cand_keys - base_keys),
        len(base_keys - cand_keys),
        len(changed_rows),
        changed_values,
    ), events


def _counts(result):
    return result.rows_added, result.rows_removed, result.rows_changed, result.values_changed


@pytest.mark.parametrize("backend", ["eager", "lazy", "arrow", "parquet", "partitioned"])
@example(
    baseline={(None, "a"): (2**53 + 1, "x", [None, 1]), (1, "b"): (2, "", [])},
    candidate={(None, "a"): (2**53, "y", [1, None]), (2, "b"): (3, None, None)},
    partition_size=1,
)
@given(baseline=_tables, candidate=_tables, partition_size=st.integers(1, 5))
def test_keyed_counts_and_evidence_survive_representation_and_permutations(
    backend,
    baseline,
    candidate,
    partition_size,
):
    expected_counts, expected_events = _oracle(baseline, candidate)
    base, cand = _frame(baseline), _frame(candidate)
    # Reversing row and column order must change neither identity nor counts.
    inputs = {"baseline": base.reverse(), "candidate": cand.reverse().select(cand.columns[::-1])}
    with adapted_frames(inputs, backend, partition_size=partition_size) as adapted:
        result = compare_outputs(
            adapted["baseline"],
            adapted["candidate"],
            KEYS,
            rtol=0,
            atol=0,
            max_differences=100,
            max_evidence_bytes=100_000,
        )
    assert _counts(result) == expected_counts
    assert result.equal == (not expected_events)
    assert not result.truncated
    actual_events = {
        (item.kind, tuple(item.key[name] for name in KEYS), item.column)
        for item in result.differences
    }
    assert actual_events == expected_events
    restored = ComparisonResult.model_validate_json(result.model_dump_json())
    assert _counts(restored) == expected_counts
    assert restored.equal == result.equal
    # Reversing comparison swaps additions/removals; changed cells are symmetric.
    inverse = compare_frames(cand, base, KEYS, rtol=0, atol=0, max_differences=0)
    assert _counts(inverse) == (expected_counts[1], expected_counts[0], *expected_counts[2:])


@given(
    baseline=_tables,
    candidate=_tables,
    evidence_rows=st.integers(0, 4),
    evidence_bytes=st.sampled_from([0, 64, 512, 4096]),
)
def test_evidence_caps_never_change_exact_regression_counts(
    baseline,
    candidate,
    evidence_rows,
    evidence_bytes,
):
    expected, events = _oracle(baseline, candidate)
    result = compare_frames(
        _frame(baseline),
        _frame(candidate),
        KEYS,
        rtol=0,
        atol=0,
        max_differences=evidence_rows,
        max_evidence_bytes=evidence_bytes,
    )
    assert _counts(result) == expected
    assert result.equal == (not events)
    assert len(result.differences) <= evidence_rows
    if result.differences:
        assert (
            evidence_json_size([item.model_dump(mode="python") for item in result.differences])
            <= evidence_bytes
        )
    assert result.truncated == (len(events) > len(result.differences))


@pytest.mark.parametrize("kind", ["integer", "decimal"])
@given(
    values=st.lists(
        st.tuples(st.integers(2**53, 2**53 + 100), st.integers(-3, 3)), min_size=1, max_size=15
    ),
    atol=st.sampled_from([0.0, 0.01, 0.1, 1.0, 2.0]),
)
def test_large_exact_values_obey_rational_absolute_tolerance(kind, values, atol):
    scale = 100 if kind == "decimal" else 1
    left = [Decimal(value).scaleb(-2) if kind == "decimal" else value for value, _ in values]
    right = [
        Decimal(value + delta).scaleb(-2) if kind == "decimal" else value + delta
        for value, delta in values
    ]
    dtype = pl.Decimal(24, 2) if kind == "decimal" else pl.Int64
    baseline = pl.DataFrame({"id": range(len(left)), "x": pl.Series(left, dtype=dtype)})
    candidate = pl.DataFrame(
        {"id": range(len(right)), "x": pl.Series(right, dtype=dtype)}
    ).reverse()
    expected = sum(Fraction(abs(delta), scale) > Fraction(atol) for _, delta in values)
    result = compare_frames(baseline, candidate, "id", rtol=0, atol=atol)
    assert result.values_changed == result.rows_changed == expected
    assert result.equal == (expected == 0)


@given(
    epochs=st.lists(st.integers(1_600_000_000_000, 1_800_000_000_000), min_size=1, max_size=15),
    changed=st.sets(st.integers(0, 14)),
    zone=st.sampled_from(["UTC", "America/New_York", "Asia/Tokyo"]),
)
def test_temporal_instants_preserve_one_nanosecond_changes_across_zones(epochs, changed, zone):
    baseline = pl.DataFrame(
        {"id": range(len(epochs)), "time": pl.Series(epochs, dtype=pl.Datetime("ms", "UTC"))}
    )
    nanos = [epoch * 1_000_000 + int(index in changed) for index, epoch in enumerate(epochs)]
    candidate = pl.DataFrame(
        {"id": range(len(epochs)), "time": pl.Series(nanos, dtype=pl.Datetime("ns", "UTC"))}
    )
    candidate = candidate.with_columns(pl.col("time").dt.convert_time_zone(zone)).reverse()
    expected = len(changed & set(range(len(epochs))))
    result = compare_frames(baseline, candidate, "id", rtol=1, atol=1)
    assert result.values_changed == result.rows_changed == expected
    assert result.dtypes_changed == []
    assert result.equal == (expected == 0)
