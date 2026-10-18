"""Lossless key comparison policy, independent of rules and execution contexts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl


class KeyTypeError(ValueError):
    """Two key columns cannot be compared without losing or inventing information."""


FLOAT_EXACT_LIMIT = 2**53


def key_casts(
    column: str,
    left: pl.DataType,
    right: pl.DataType,
    *,
    exact_float: Callable[[], bool],
    integer_storage: Callable[[], pl.DataType] = pl.Int128,
) -> tuple[pl.DataType | None, pl.DataType | None]:
    """Plan lossless casts, asking the backend for value proofs only if needed.

    Eager callers can prove that both integer columns fit Int64. Lazy/native
    callers use Int128 without scanning for compaction. Cross integer/float
    alignment always requires the same exact-value proof from the caller.
    """
    if left == right:
        return None, None
    if left == pl.Null:
        return right, None
    if right == pl.Null:
        return None, left
    if left.is_integer() and right.is_integer():
        target = integer_storage()
        return target, target
    if left.is_float() and right.is_float():
        return pl.Float64(), pl.Float64()
    if (left.is_integer() or left.is_float()) and (right.is_integer() or right.is_float()):
        if exact_float():
            return pl.Float64(), pl.Float64()
        raise KeyTypeError(
            f"key column {column!r} cannot be compared exactly: {left} vs {right} with "
            "fractional values or magnitudes beyond 2**53; cast one side in its source"
        )
    raise KeyTypeError(
        f"key column {column!r} cannot be compared: {left} vs {right}; "
        "make both sides the same type in their sources"
    )


def _integral_and_exact(series: pl.Series) -> bool:
    """Whether each non-null value is an integer exactly representable as Float64."""
    values = series.drop_nulls()
    if values.is_empty():
        return True
    if values.dtype.is_float() and not (
        values.is_finite().all() and (values == values.round(0)).all()
    ):
        return False
    return all(
        isinstance(value, int | float) and abs(value) <= FLOAT_EXACT_LIMIT
        for value in (values.min(), values.max())
    )


def _fits_int64(series: pl.Series) -> bool:
    values = series.drop_nulls()
    if values.is_empty():
        return True
    return int(values.min()) >= -(2**63) and int(values.max()) < 2**63  # type: ignore[arg-type]


def align_keys(
    left: pl.DataFrame, right: pl.DataFrame, columns: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Give same-named keys a common dtype without changing their values.

    Integer widths/signedness align as Int64 when both fit, otherwise Int128.
    Cross integer/float keys must be integral with magnitudes at most 2**53.
    Incompatible domains raise KeyTypeError rather than guessing a conversion.
    """
    left_types: dict[str, pl.DataType] = {}
    right_types: dict[str, pl.DataType] = {}
    left_schema, right_schema = left.schema, right.schema
    for column in columns:

        def exact_float(key: str = column) -> bool:
            return _integral_and_exact(left[key]) and _integral_and_exact(right[key])

        def integer_storage(key: str = column) -> pl.DataType:
            return pl.Int64() if _fits_int64(left[key]) and _fits_int64(right[key]) else pl.Int128()

        a, b = key_casts(
            column,
            left_schema[column],
            right_schema[column],
            exact_float=exact_float,
            integer_storage=integer_storage,
        )
        if a is not None:
            left_types[column] = a
        if b is not None:
            right_types[column] = b
    if left_types:
        left = left.cast(left_types)  # type: ignore[arg-type]
    if right_types:
        right = right.cast(right_types)  # type: ignore[arg-type]
    return left, right
