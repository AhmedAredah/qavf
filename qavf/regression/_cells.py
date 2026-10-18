"""Bounded-state comparison of nested residual values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import polars as pl

from qavf.regression._numeric import _is_number, _nonfinite, numbers_close
from qavf.regression._schema import _type_changed

_MISSING = object()


def _cell_change(
    before: Any,
    after: Any,
    column: str,
    tolerance: Callable,
    before_dtype: pl.DataType | None = None,
    after_dtype: pl.DataType | None = None,
) -> tuple[bool, list[Any], bool]:
    """One changed-cell decision and path, without a complete leaf-difference graph.

    Traversal stops at the first difference when neither cell can contain nonfinite
    numbers. Otherwise it continues only until a changed nonfinite leaf is found,
    preserving the exact changed-cell nonfinite diagnostic with O(depth) state.
    """
    rtol, atol = tolerance(column)
    first: list[Any] | None = None
    nonfinite = False

    def may_nonfinite(value: Any) -> bool:
        if isinstance(value, pl.Series):

            def numeric(dtype: pl.DataType) -> bool:
                if isinstance(dtype, pl.List | pl.Array):
                    return numeric(cast(pl.DataType, dtype.inner))
                if isinstance(dtype, pl.Struct):
                    return any(numeric(cast(pl.DataType, field.dtype)) for field in dtype.fields)
                return dtype.is_float() or isinstance(dtype, pl.Object)

            if value.dtype.is_float():
                return not bool(value.is_finite().all())
            return numeric(value.dtype)
        if isinstance(value, (list, tuple, dict)):
            return True
        return _nonfinite(value)

    inspect_nonfinite = may_nonfinite(before) or may_nonfinite(after)

    def visit(
        a: Any,
        b: Any,
        path: list[Any],
        da: pl.DataType | None = None,
        db: pl.DataType | None = None,
    ) -> None:
        nonlocal first, nonfinite
        if first is not None and (nonfinite or not inspect_nonfinite):
            return
        if _is_number(a) and _is_number(b):
            different = not numbers_close(a, b, rtol, atol)
        elif isinstance(a, Mapping) and isinstance(b, Mapping):
            fields_a = (
                {field.name: cast(pl.DataType, field.dtype) for field in da.fields}
                if isinstance(da, pl.Struct)
                else {}
            )
            fields_b = (
                {field.name: cast(pl.DataType, field.dtype) for field in db.fields}
                if isinstance(db, pl.Struct)
                else {}
            )
            for key in a:
                if key not in b:
                    visit(a[key], _MISSING, [*path, key], fields_a.get(key), None)
                else:
                    visit(a[key], b[key], [*path, key], fields_a.get(key), fields_b.get(key))
                if first is not None and (nonfinite or not inspect_nonfinite):
                    return
            for key in b:
                if key not in a:
                    visit(_MISSING, b[key], [*path, key], None, fields_b.get(key))
            return
        elif (
            isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and type(a) is not type(b)
        ):
            different = True
        elif isinstance(a, (list, tuple, pl.Series)) and isinstance(b, (list, tuple, pl.Series)):
            inner_a = cast(pl.DataType, da.inner) if isinstance(da, pl.List | pl.Array) else None
            inner_b = cast(pl.DataType, db.inner) if isinstance(db, pl.List | pl.Array) else None
            for index in range(max(len(a), len(b))):
                visit(
                    a[index] if index < len(a) else _MISSING,
                    b[index] if index < len(b) else _MISSING,
                    [*path, index],
                    inner_a,
                    inner_b,
                )
                if first is not None and (nonfinite or not inspect_nonfinite):
                    return
            return
        else:
            different = type(a) is not type(b) or a != b
            if a is not None and b is not None and da is not None and db is not None:
                different |= _type_changed(da, db, "compatible")
        if different:
            if first is None:
                first = path
            nonfinite |= _nonfinite(a) or _nonfinite(b)
            if a is _MISSING or b is _MISSING:

                def contains_nonfinite(value: Any) -> bool:
                    if isinstance(value, Mapping):
                        return any(contains_nonfinite(item) for item in value.values())
                    if isinstance(value, (list, tuple, pl.Series)):
                        return any(contains_nonfinite(item) for item in value)
                    return _nonfinite(value)

                nonfinite |= contains_nonfinite(b if a is _MISSING else a)

    visit(before, after, [], before_dtype, after_dtype)
    return first is not None, first or [], nonfinite
