"""Shared typed policies retain exact decisions across execution representations."""

from decimal import Decimal, localcontext
from fractions import Fraction
from unittest.mock import patch

import polars as pl
import pytest

from qavf.aggregation import checked_accumulator_dtype, exact_sum_expr
from qavf.engine.context import DataContext
from qavf.keys import KeyTypeError, align_keys
from qavf.rules._execution import KeyRelation, align_key_relations


@pytest.mark.parametrize(
    "left,right,safe,proofs",
    [
        (pl.Series([1, 2], dtype=pl.Int8), pl.Series([1, 2], dtype=pl.UInt64), True, 0),
        (pl.Series([2**63], dtype=pl.UInt64), pl.Series([-1], dtype=pl.Int64), True, 0),
        (pl.Series([2**53], dtype=pl.Int64), pl.Series([float(2**53)]), True, 2),
        (pl.Series([2**53 + 1], dtype=pl.Int64), pl.Series([float(2**53)]), False, 1),
        (pl.Series([1.5]), pl.Series([1], dtype=pl.Int64), False, 1),
        (pl.Series([None], dtype=pl.Null), pl.Series([1], dtype=pl.Int64), True, 0),
        (pl.Series([0.5], dtype=pl.Float32), pl.Series([0.5]), True, 0),
        (pl.Series(["1"]), pl.Series([1], dtype=pl.Int64), False, 0),
    ],
)
def test_eager_and_lazy_keys_keep_values_and_only_scan_for_required_proofs(
    left, right, safe, proofs
):
    frames = [series.rename("key").to_frame() for series in (left, right)]
    with DataContext() as ctx:
        relations = [
            KeyRelation(
                ctx,
                (),
                ("key",),
                "frequency",
                {**frame.schema, "frequency": pl.UInt64()},
                lazy=frame.with_columns(pl.lit(1, dtype=pl.UInt64).alias("frequency")).lazy(),
            )
            for frame in frames
        ]
        with patch.object(ctx, "collect_many", wraps=ctx.collect_many) as collect:
            if not safe:
                with pytest.raises(KeyTypeError) as eager:
                    align_keys(*frames, ["key"])
                with pytest.raises(KeyTypeError) as lazy:
                    align_key_relations(*relations)
                assert str(eager.value) == str(lazy.value)
            else:
                eager_frames = align_keys(*frames, ["key"])
                aligned = align_key_relations(*relations)
                assert collect.call_count == proofs
                for original, eager_frame, relation in zip(
                    frames, eager_frames, aligned, strict=True
                ):
                    actual = relation.lazy.collect()["key"].to_list()
                    assert actual == original["key"].to_list() == eager_frame["key"].to_list()
            assert collect.call_count == proofs


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("x", [2**63, 1, None], dtype=pl.UInt64),
        pl.Series("x", [-(2**70), 2**70, 1], dtype=pl.Int128),
        pl.Series("x", [Decimal("123456789.1234"), Decimal("-0.0001"), None]),
        pl.Series("x", [None, None], dtype=pl.Decimal(20, 8)),
        pl.Series("x", [], dtype=pl.Int64),
    ],
)
def test_shared_sum_guard_preserves_exact_totals_and_undefined_empty_values(series):
    values = [Fraction(value) for value in series if value is not None]
    expected = sum(values, Fraction()) if values else None
    with localcontext() as context:
        context.prec = 3
        result = series.to_frame().select(exact_sum_expr(series)).item()
    assert (Fraction(result) if result is not None else None) == expected


@pytest.mark.parametrize(
    "dtype,minimum,maximum,count",
    [
        (pl.Int128(), 2**126, 2**126, 2),
        (pl.Int128(), -(2**126), 2**126, 2),
        (pl.Decimal(38, 8), Decimal("5e29"), Decimal("5e29"), 2),
    ],
)
def test_shared_guard_rejects_potential_overflow_before_backend_execution(
    dtype, minimum, maximum, count
):
    with pytest.raises(ValueError, match="exact sum may exceed accumulator precision"):
        checked_accumulator_dtype(dtype, minimum, maximum, count)


def test_incompatible_scalar_metadata_fails_explicitly():
    with pytest.raises(TypeError, match="nonnumeric extremum"):
        checked_accumulator_dtype(pl.Int64(), "1", 2, 2)
