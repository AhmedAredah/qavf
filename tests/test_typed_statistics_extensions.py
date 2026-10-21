"""Shared capabilities added for dataset comparison, tested against independent oracles.

Checked casts, exact scaling, exact quantiles and means, native grouped
reductions, byte-bounded evidence and declared-only missing normalization are
general helpers; each expected value here is computed with plain Python
(``fractions``, ``statistics``) rather than the code under test.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qavf.audit_codec import evidence_json_size
from qavf.engine.context import DataContext
from qavf.numeric import exact_rational, report_number
from qavf.rules._execution import normalized_plan
from qavf.rules._grouped import ProjectedRelation, grouped_mean, grouped_quantile, grouped_reduce
from qavf.rules.helpers import BoundedEvidence, LargestEvidence
from qavf.statistics.ordered import linear_quantile, quantile_ranks
from qavf.typed_values import checked_cast, exact_scale, logical_type


def _cast(values, source, target):
    frame = pl.DataFrame({"v": values}, schema={"v": source})
    converted, lost = checked_cast(pl.col("v"), source, target)
    result = frame.select(converted.alias("c"), lost.alias("lost"))
    return result["c"].to_list(), result["lost"].to_list()


@pytest.mark.parametrize(
    "values,source,target,lost",
    [
        ([1.0, 1.5, None], pl.Float64, pl.Int64, [False, True, False]),
        ([2**53, 2**53 + 1], pl.Int64, pl.Float64, [False, True]),
        (["1", "01", "x"], pl.String, pl.Int64, [False, True, True]),
        ([1, 22], pl.Int64, pl.String, [False, False]),
        ([True, False], pl.Boolean, pl.Int8, [False, False]),
        ([Decimal("1.25"), Decimal("1.20")], pl.Decimal(10, 2), pl.Decimal(10, 1), [True, False]),
        (
            [datetime(2024, 1, 1), datetime(2024, 1, 1, 6)],
            pl.Datetime("us"),
            pl.Date,
            [False, True],
        ),
        ([float("nan")], pl.Float64, pl.Int64, [True]),
        ([float("nan"), 0.5], pl.Float64, pl.Float32, [False, False]),
    ],
)
def test_checked_cast_flags_every_value_it_cannot_reproduce(values, source, target, lost):
    assert _cast(values, source, target)[1] == lost


def test_checked_cast_keeps_lossless_values():
    converted, lost = _cast(["2024-02-29", None], pl.String, pl.Date)
    assert converted == [date(2024, 2, 29), None] and lost == [False, False]


def test_logical_types_require_a_decimal_scale():
    assert logical_type("decimal", 3) == pl.Decimal(38, 3)
    with pytest.raises(ValueError, match="scale"):
        logical_type("decimal")
    with pytest.raises(ValueError, match="scale"):
        logical_type("int", 2)


@pytest.mark.parametrize(
    "values,dtype,factor,expected",
    [
        ([2**62, -3], pl.Int64, 1000, [2**62 * 1000, -3000]),
        ([7, None], pl.Int64, 0.001, [Decimal("0.007"), None]),
        ([Decimal("1.2345")], pl.Decimal(38, 4), Decimal("0.001"), [Decimal("0.0012345")]),
        ([Decimal("1.5")], pl.Decimal(38, 1), 3, [Decimal("4.5")]),
    ],
)
def test_exact_scale_never_rounds_exact_values(values, dtype, factor, expected):
    frame = pl.DataFrame({"v": values}, schema={"v": dtype})
    result = frame.select(exact_scale(pl.col("v"), dtype, factor))["v"].to_list()
    assert [None if v is None else Fraction(v) for v in result] == [
        None if v is None else Fraction(v) for v in expected
    ]


def test_exact_scale_floats_and_invalid_factors():
    frame = pl.DataFrame({"v": [0.5]})
    assert frame.select(exact_scale(pl.col("v"), pl.Float64, 4))["v"].item() == 2.0
    for factor in (0, float("inf"), True):
        with pytest.raises(ValueError):
            exact_scale(pl.col("v"), pl.Float64, factor)
    with pytest.raises(ValueError, match="numeric"):
        exact_scale(pl.col("v"), pl.String, 2)


def _oracle_quantile(values, p):
    """NumPy's linear method with exact arithmetic, written independently."""
    ordered = sorted(Fraction(v) for v in values)
    position = (len(ordered) - 1) * Fraction(str(p))
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@given(
    st.lists(st.integers(-(2**70), 2**70), min_size=1, max_size=25),
    st.sampled_from([0, 0.1, 0.25, 0.5, 0.9, 1]),
)
@settings(max_examples=60, deadline=None)
def test_quantile_ranks_and_interpolation_match_exact_oracle(values, p):
    ordered = sorted(values)
    lower, upper, weight = quantile_ranks(len(values), p)
    assert linear_quantile(ordered[lower], ordered[upper], weight) == _oracle_quantile(values, p)


def test_quantile_ranks_reject_invalid_inputs():
    with pytest.raises(ValueError):
        quantile_ranks(0, 0.5)
    with pytest.raises(ValueError):
        quantile_ranks(3, 1.5)


def _relation(ctx, observation, name="data"):
    return ProjectedRelation(
        ctx,
        name,
        observation.schema(),
        lazy=observation.scan(),
        observation=observation,
        collector=observation.collect,
    )


@pytest.mark.parametrize("storage", ["frame", "spill"])
def test_grouped_exact_quantiles_and_means_match_oracles(storage, tmp_path):
    big = 2**60
    frame = pl.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "c"],
            "i": [big + 1, big + 2, big + 4, 1, 2, None],
            "d": pl.Series(
                [Decimal("0.1"), Decimal("0.2"), Decimal("0.4"), None, Decimal("1"), None],
                dtype=pl.Decimal(10, 1),
            ),
            "f": [0.1, 0.2, float("nan"), 1.0, 2.0, None],
        }
    )
    with (
        DataContext(
            frames={"data": frame},
            cache_max_bytes=0 if storage == "spill" else 2**20,
            temp_directory=tmp_path,
        ) as ctx,
        ctx.observe("data", frame.columns) as observation,
    ):
        relation = _relation(ctx, observation)
        results = {}
        for column in ("i", "d", "f"):
            for p in (0.5, 0.9):
                q = grouped_quantile(relation, ["g"], column, "q", p).sort("g")
                results[(column, p)] = dict(zip(q["g"], q["q"], strict=True))
            mean = grouped_mean(relation, ["g"], column, "m").sort("g")
            results[(column, "mean")] = dict(zip(mean["g"], mean["m"], strict=True))
    groups = {"a": [0, 1, 2], "b": [3, 4], "c": [5]}
    for column in ("i", "d", "f"):
        for group, rows in groups.items():
            usable = [
                frame[column][r]
                for r in rows
                if frame[column][r] is not None and frame[column][r] == frame[column][r]
            ]
            for p in (0.5, 0.9):
                expected = _oracle_quantile(usable, p) if usable else None
                actual = results[(column, p)][group]
                assert (None if actual is None else Fraction(actual)) == expected
            mean = sum(map(Fraction, usable)) / len(usable) if usable else None
            actual = results[(column, "mean")][group]
            assert (None if actual is None else Fraction(actual)) == mean


def test_grouped_reduce_collects_one_row_per_group(tmp_path):
    frame = pl.DataFrame({"g": [1, 1, 2], "v": ["x", "y", "x"]})
    with (
        DataContext(frames={"data": frame}, temp_directory=tmp_path) as ctx,
        ctx.observe("data") as observation,
    ):
        relation = _relation(ctx, observation)
        result = grouped_reduce(relation, ["g"], {"n": pl.col("v").n_unique()}).sort("g")
        total = grouped_reduce(relation, [], {"n": pl.len()})
    assert result.rows() == [(1, 2), (2, 1)] and total.item() == 3
    with pytest.raises(ValueError, match="Polars"):
        grouped_reduce(ProjectedRelation(ctx, "data", {}, sql="SELECT 1"), [], {})


def test_bounded_evidence_byte_limit_keeps_complete_counts_and_order():
    records = [{"i": i, "text": "x" * 40} for i in range(10)]
    size = evidence_json_size(records[0])
    evidence = BoundedEvidence(limit=100, byte_limit=3 * size)
    evidence.extend(records)
    assert evidence.count == 10
    assert [record["i"] for record in evidence.records] == [0, 1, 2]
    assert evidence.truncated
    # A later, smaller record cannot jump the queue once the allowance is spent.
    evidence.append({"i": 99})
    assert [record["i"] for record in evidence.records] == [0, 1, 2]
    frame = BoundedEvidence(limit=2, byte_limit=None)
    frame.add_frame(pl.DataFrame({"i": [1, 2, 3]}))
    assert frame.records == [{"i": 1}, {"i": 2}] and frame.count == 3 and frame.truncated


def test_declared_only_normalization_keeps_nan_as_a_value(tmp_path):
    frame = pl.DataFrame({"a": [float("nan"), -999.0], "b": [float("nan"), 1.0]})
    config = {"a": {"schema": {"columns": {"a": {"missing_values": [-999]}}}}}
    from qavf.settings import DatasetSpec

    specs = {"a": DatasetSpec.model_validate(config["a"])}
    with DataContext(frames={"a": frame}, specs=specs, temp_directory=tmp_path) as ctx:
        plan = normalized_plan(ctx, "a", frame.lazy(), declared_only=True).collect()
        everything = normalized_plan(ctx, "a", frame.lazy()).collect()
    assert plan["a"].to_list()[1] is None and plan["b"].is_nan().to_list() == [True, False]
    assert everything["b"].to_list() == [None, 1.0]


def test_largest_evidence_keeps_the_largest_magnitudes_with_earliest_ties():
    evidence = LargestEvidence(3)
    items = [(5.0, 0), (1.0, 1), (5.0, 2), (9.0, 3), (5.0, 4), (0.5, 5)]
    for magnitude, ordinal in items:
        evidence.add(magnitude, ordinal, {"ordinal": ordinal})
    oracle = sorted(items, key=lambda item: (-item[0], item[1]))[:3]
    assert [r["ordinal"] for r in evidence.records] == [ordinal for _, ordinal in oracle]
    assert not LargestEvidence(0).admits(1.0, 0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (2**70, Fraction(2**70)),
        (Decimal("0.1"), Fraction(1, 10)),
        (0.1, Fraction(0.1)),
        ("7/2", Fraction(7, 2)),
        ("x", None),
        (float("nan"), None),
        (Decimal("Infinity"), None),
        (True, None),
        (None, None),
    ],
)
def test_exact_rational_conversion(value, expected):
    assert exact_rational(value) == expected


def test_report_numbers_keep_integers_exact():
    assert report_number(Fraction(2**70)) == 2**70
    assert report_number(Fraction(1, 4)) == 0.25
    assert report_number(Fraction(10**400, 3)) == math.inf
    assert report_number(None) is None


def test_statistics_oracle_sanity():
    # Guard the oracle itself against the standard library.
    assert float(_oracle_quantile([1, 2, 3, 4], 0.5)) == statistics.median([1, 2, 3, 4])
