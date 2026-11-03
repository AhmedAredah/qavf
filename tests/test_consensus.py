"""The consensus of several members, key by key: every configurable option.

Numbers combine by median, mean or trimmed mean; other values by majority or
plurality; ``min_agreement`` sets how many members must agree; a tie leaves the
key without a consensus value (``report``) or takes the reference member's value
(``reference``). A key without a consensus value says why and is never guessed.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from qavf.compare.consensus import NO_CONSENSUS, ConsensusBuilder, ConsensusError
from qavf.compare.pairing import ConsensusSettings


def _members(*values: list[Any], column: str = "v") -> list[pl.DataFrame]:
    """One member per list of values, keyed 1, 2, ... (``...`` leaves a key out)."""
    frames = []
    for member in values:
        keys = [k for k, value in enumerate(member, start=1) if value is not ...]
        kept = [value for value in member if value is not ...]
        frames.append(pl.DataFrame({"k": keys, column: kept}))
    return frames


def _build(frames: list[pl.DataFrame], reference: str | None = None, **settings: Any):
    names = [f"m{i}" for i in range(len(frames))]
    builder = ConsensusBuilder(ConsensusSettings(**settings), ["k"], names, reference=reference)
    return {row["k"]: row for row in builder.build(frames).to_dicts()}


# --------------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("method", "expected"),
    [("median", 12.0), ("mean", 26.8), ("trimmed_mean", 13.0)],
)
def test_numbers_combine_by_the_configured_method(method: str, expected: float) -> None:
    frames = _members([10.0], [11.0], [12.0], [16.0], [85.0])
    row = _build(frames, numeric=method, trim=0.2)[1]
    assert row["v"] == pytest.approx(expected)
    assert row[NO_CONSENSUS] is None


def test_trimming_removes_the_same_count_from_each_end() -> None:
    frames = _members([1.0], [2.0], [3.0], [4.0], [100.0], [1000.0])
    # trim 0.2 of 6 values drops one from each end: mean of 2, 3, 4, 100.
    assert _build(frames, numeric="trimmed_mean", trim=0.2)[1]["v"] == pytest.approx(27.25)
    # trim 0 is the plain mean.
    assert _build(frames, numeric="trimmed_mean", trim=0.0)[1]["v"] == pytest.approx(185.0)


def test_whole_number_consensus_keeps_the_members_integer_type() -> None:
    frames = _members([10], [12], [14])
    built = ConsensusBuilder(ConsensusSettings(), ["k"], ["a", "b", "c"]).build(frames)
    assert built.schema["v"] == pl.Int64
    assert built["v"].to_list() == [12]


def test_a_number_needs_enough_members_reporting_it() -> None:
    frames = _members([10.0], [None], [None], [None])
    # One of four members reports a value, three report none.
    assert _build(frames)[1]["v"] is None  # most members agree there is no value
    assert _build(frames)[1][NO_CONSENSUS] is None
    frames = _members([10.0], [12.0], [None], [None])
    row = _build(frames, min_agreement=0.75)[1]
    assert row["v"] is None
    assert row[NO_CONSENSUS] == "v: too little agreement"
    assert _build(frames, min_agreement=0.5)[1]["v"] == pytest.approx(11.0)


# ----------------------------------------------------------------------- other values


def test_majority_needs_more_than_half() -> None:
    frames = _members(["x"], ["x"], ["y"], ["z"], column="c")
    majority = _build(frames)[1]
    assert majority["c"] is None
    assert majority[NO_CONSENSUS] == "c: too little agreement"
    plurality = _build(frames, categorical="plurality")[1]
    assert plurality["c"] == "x" and plurality[NO_CONSENSUS] is None


def test_min_agreement_raises_the_share_that_must_agree() -> None:
    frames = _members(["x"], ["x"], ["x"], ["y"], column="c")
    assert _build(frames, min_agreement=0.75)[1]["c"] == "x"
    row = _build(frames, min_agreement=0.8)[1]
    assert row["c"] is None and row[NO_CONSENSUS] == "c: too little agreement"


@pytest.mark.parametrize("categorical", ["majority", "plurality"])
def test_a_tie_is_reported_by_default(categorical: str) -> None:
    frames = _members(["x"], ["y"], ["x"], ["y"], column="c")
    row = _build(frames, categorical=categorical)[1]
    assert row["c"] is None
    assert row[NO_CONSENSUS] == "c: tie"


@pytest.mark.parametrize("categorical", ["majority", "plurality"])
def test_a_tie_can_take_the_reference_members_value(categorical: str) -> None:
    frames = _members(["x"], ["y"], ["x"], ["y"], column="c")
    row = _build(frames, reference="m1", categorical=categorical, ties="reference")[1]
    assert row["c"] == "y" and row[NO_CONSENSUS] is None


# --------------------------------------------------------------------------- presence


def test_a_key_belongs_to_the_consensus_when_most_members_hold_it() -> None:
    frames = _members([1.0, 2.0, 3.0], [1.0, 2.0, ...], [1.0, ..., ...])
    built = _build(frames)
    assert sorted(built) == [1, 2]  # key 3 is held by one member of three: not a key
    assert built[2]["v"] == pytest.approx(2.0)


def test_presence_ties_follow_the_tie_mode() -> None:
    frames = _members([1.0, 2.0], [1.0, ...])
    built = _build(frames)
    assert built[2][NO_CONSENSUS] == "presence: tie"
    by_reference = _build(frames, reference="m0", ties="reference")
    assert by_reference[2]["v"] == pytest.approx(2.0)
    assert by_reference[2][NO_CONSENSUS] is None
    assert 2 not in _build(frames, reference="m1", ties="reference")


def test_a_key_repeated_within_a_member_has_no_consensus() -> None:
    frames = [
        pl.DataFrame({"k": [1, 1], "v": [1.0, 2.0]}),
        pl.DataFrame({"k": [1], "v": [1.0]}),
        pl.DataFrame({"k": [1], "v": [1.0]}),
    ]
    assert _build(frames)[1][NO_CONSENSUS] == "duplicate key in m0"


def test_members_without_the_keys_are_an_error() -> None:
    builder = ConsensusBuilder(ConsensusSettings(), ["k"], ["a", "b"])
    with pytest.raises(ConsensusError, match="b has no key column"):
        builder.build([pl.DataFrame({"k": [1]}), pl.DataFrame({"j": [1]})])


def test_totals_combine_each_members_sum_per_key() -> None:
    frames = [
        pl.DataFrame({"k": [1, 1, 2], "v": [1.0, 2.0, 5.0]}),
        pl.DataFrame({"k": [1, 2], "v": [3.0, 5.0]}),
        pl.DataFrame({"k": [1, 2], "v": [9.0, 5.0]}),
    ]
    builder = ConsensusBuilder(ConsensusSettings(), ["k"], ["a", "b", "c"], columns=["v"])
    built = builder.build(builder.aggregated(frames, "v"))
    assert built.sort("k")["v"].to_list() == [3.0, 5.0]
    rows = ConsensusBuilder(ConsensusSettings(), ["k"], ["a", "b", "c"], columns=["rows"])
    counted = rows.build(rows.aggregated(frames, None))
    assert counted.columns == ["k", "rows", NO_CONSENSUS]
