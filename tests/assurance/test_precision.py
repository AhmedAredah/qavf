"""Generated numerical contracts with exact, backend-independent reference values."""

from __future__ import annotations

import math
import operator
from decimal import Decimal, localcontext
from fractions import Fraction

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qavf.engine.severity import Status
from qavf.numeric import exact_float_sum

from .support.adapters import BACKENDS, adapted_frames
from .support.contracts import run_checks

_COMPARISONS = {"ge": operator.ge, "gt": operator.gt, "le": operator.le, "lt": operator.lt}
_FINITE = st.floats(allow_nan=False, allow_infinity=False, width=64)


def _fraction_sum(values: list[float]) -> Fraction:
    """Independent rational oracle; does not use QAVF's integer accumulator."""
    return sum((Fraction(value) for value in values), Fraction())


@given(values=st.lists(_FINITE, min_size=1, max_size=24), split=st.integers(1, 24))
def test_exact_float_accumulator_is_order_and_partition_invariant(values, split):
    expected = _fraction_sum(values)
    assert exact_float_sum(values) == expected
    assert exact_float_sum(reversed(values)) == expected
    parts = [values[offset : offset + split] for offset in range(0, len(values), split)]
    assert sum((exact_float_sum(part) for part in parts), Fraction()) == expected
    assert exact_float_sum([None, math.nan, *values, None]) == expected


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    exponent=st.integers(53, 62),
    displacement=st.integers(-5, 5),
    comparison=st.sampled_from(tuple(_COMPARISONS)),
)
def test_float_bounds_compare_exact_integer_thresholds(exponent, displacement, backend, comparison):
    boundary = 2**exponent + displacement
    origin = float(boundary)
    values = [math.nextafter(origin, -math.inf), origin, math.nextafter(origin, math.inf)]
    expected = sum(
        not _COMPARISONS[comparison](Fraction(value), Fraction(boundary)) for value in values
    )
    with adapted_frames({"d": pl.DataFrame({"value": values})}, backend) as inputs:
        results = run_checks(
            inputs,
            "INP-003",
            datasets={"d": {"schema": {"columns": {"value": {comparison: boundary}}}}},
        )
    assert sum(result.affected_count for result in results) == expected
    assert any(result.status is Status.FAIL for result in results) == bool(expected)


@pytest.mark.parametrize("backend", (*BACKENDS, "partitioned"))
@given(
    exponent=st.integers(54, 1000),
    tail=st.integers(-32, 32),
    denominator=st.integers(0, 10),
    delta=st.sampled_from([0, 1]),
    order=st.permutations((0, 1, 2)),
    partition_size=st.integers(1, 3),
)
def test_reconciliation_cancellation_matches_exact_totals_across_partitions(
    exponent, tail, denominator, delta, order, partition_size, backend
):
    large = math.ldexp(1.0, exponent)
    residual = math.ldexp(float(tail), -denominator)
    left = [large, residual, -large]
    right = [large, -large, residual + delta]
    expected = int(_fraction_sum(left) != _fraction_sum(right))
    frames = {
        "a": pl.DataFrame({"key": [1] * 3, "value": [left[index] for index in order]}),
        "b": pl.DataFrame({"key": [1] * 3, "value": right}),
    }
    with adapted_frames(frames, backend, partition_size=partition_size) as inputs:
        results = run_checks(
            inputs,
            "REC-001",
            params={
                "reconciliations": [
                    {
                        "left": {"dataset": "a", "value": "value"},
                        "right": {"dataset": "b", "value": "value"},
                        "on": ["key"],
                        "rtol": 0,
                        "atol": 0,
                    }
                ]
            },
        )
    assert sum(result.affected_count for result in results) == expected
    assert any(result.status is Status.FAIL for result in results) == bool(expected)


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    coefficient=st.integers(10**15, 10**16),
    scale=st.integers(1, 8),
    delta=st.sampled_from([0, 1]),
)
def test_decimal_totals_ignore_ambient_decimal_rounding(coefficient, scale, delta, backend):
    # Construct exact decimal tuples without using the ambient arithmetic context.
    def decimal(value):
        return Decimal((0, tuple(map(int, str(value))), -scale))

    values = [decimal(coefficient), decimal(1)]
    control = decimal(coefficient + 1 + delta)
    expected = int(sum(map(Fraction, values), Fraction()) != Fraction(control))
    frames = {
        "a": pl.DataFrame({"value": pl.Series(values, dtype=pl.Decimal(38, scale))}),
        "b": pl.DataFrame({"value": pl.Series([control], dtype=pl.Decimal(38, scale))}),
    }
    with adapted_frames(frames, backend) as inputs, localcontext() as context:
        context.prec = 4
        results = run_checks(
            inputs,
            "REC-001",
            params={
                "reconciliations": [
                    {
                        "left": {"dataset": "a", "value": "value"},
                        "right": {"dataset": "b", "value": "value"},
                        "rtol": 0,
                        "atol": 0,
                    }
                ]
            },
        )
    assert sum(result.affected_count for result in results) == expected
    assert any(result.status is Status.FAIL for result in results) == bool(expected)


@given(
    odd=st.integers(0, 1000).map(lambda value: 2 * value + 1),
    backend=st.sampled_from(BACKENDS),
)
def test_statistics_cannot_silently_round_unrepresentable_integer_values(odd, backend):
    value = 2**53 + odd
    assert Fraction(float(value)) != value
    frame = pl.DataFrame({"value": [value] * 7 + [value + 100]})
    with adapted_frames({"d": frame}, backend) as inputs:
        results = run_checks(
            inputs,
            "STA-001",
            params={"targets": [{"dataset": "d", "value": "value"}]},
            assessed_only=False,
        )
    assert {result.status for result in results} == {Status.ERROR}
    assert all(
        "exact" in result.message.lower() or "precision" in result.message.lower()
        for result in results
    )


@pytest.mark.parametrize("nonfinite", [math.inf, -math.inf])
def test_exact_sum_rejects_nonfinite_observations(nonfinite):
    with pytest.raises(ValueError, match="non-finite"):
        exact_float_sum([1.0, nonfinite, -1.0])
