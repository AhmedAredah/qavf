"""Schema compatibility and lossless temporal normalization for comparison."""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from qavf.regression.models import SchemaPolicy
from qavf.typed_values import encode_temporal as _encode_temporal

__all__ = ["_encode_temporal"]

# Private aliases retain the comparison module boundary while other exact
# batch consumers share the same temporal normalization implementation.


def _kind(dtype: pl.DataType) -> Any:
    """What a column holds, for the ``compatible`` schema policy."""
    if dtype == pl.Null:
        return "null"
    if dtype.is_numeric():
        return "number"
    if dtype == pl.String or isinstance(dtype, pl.Categorical | pl.Enum):
        return "text"
    if isinstance(dtype, pl.Datetime):
        return ("datetime", dtype.time_zone is not None)
    if isinstance(dtype, pl.Duration):
        return "duration"
    if isinstance(dtype, pl.List | pl.Array):
        return ("list", _kind(cast(pl.DataType, dtype.inner)))
    if isinstance(dtype, pl.Struct):
        return (
            "struct",
            tuple((field.name, _kind(cast(pl.DataType, field.dtype))) for field in dtype.fields),
        )
    return str(dtype)


def _type_changed(a: pl.DataType, b: pl.DataType, policy: SchemaPolicy) -> bool:
    if policy == "ignore" or a == b:
        return False
    if policy == "strict":
        return True
    if a == pl.Null or b == pl.Null:
        return False
    if isinstance(a, pl.List | pl.Array) and isinstance(b, pl.List | pl.Array):
        return _type_changed(cast(pl.DataType, a.inner), cast(pl.DataType, b.inner), policy)
    if isinstance(a, pl.Struct) and isinstance(b, pl.Struct):
        left = {field.name: cast(pl.DataType, field.dtype) for field in a.fields}
        right = {field.name: cast(pl.DataType, field.dtype) for field in b.fields}
        return any(
            _type_changed(dtype, right[name], policy)
            for name, dtype in left.items()
            if name in right
        )
    return _kind(a) != _kind(b)


def _structure_changed(a: pl.DataType, b: pl.DataType) -> bool:
    """Detect nested schema loss even when no non-null cells expose it."""
    if isinstance(a, pl.Struct) and isinstance(b, pl.Struct):
        left = {field.name: field.dtype for field in a.fields}
        right = {field.name: field.dtype for field in b.fields}
        return left.keys() != right.keys() or any(
            _structure_changed(cast(pl.DataType, left[name]), cast(pl.DataType, right[name]))
            for name in left
        )
    if isinstance(a, pl.List | pl.Array) and isinstance(b, pl.List | pl.Array):
        if isinstance(a, pl.Array) and isinstance(b, pl.Array) and a.shape != b.shape:
            return True
        return _structure_changed(cast(pl.DataType, a.inner), cast(pl.DataType, b.inner))
    return False


def _nested(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.List | pl.Array | pl.Struct | pl.Object)
