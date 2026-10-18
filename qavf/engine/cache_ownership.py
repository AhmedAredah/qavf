"""Measure and detach supported cache values at the ownership boundary.

Polars reports visible slice sizes, not the parent allocation it keeps alive.
Insertion therefore compacts its buffers before measuring them. Subsequent hits
only clone the immutable native buffers and detach the Python wrapper.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow as pa

_IMMUTABLE = (
    str,
    bytes,
    int,
    float,
    complex,
    bool,
    np.generic,
    date,
    datetime,
    time,
    timedelta,
    Decimal,
    Enum,
)


@dataclass(frozen=True, slots=True)
class _ArrowSnapshot:
    """An Arrow value backed exclusively by owned, immutable IPC bytes."""

    value: Any
    payload: bytes


def _own_arrow(value: pa.Table | pa.Array | pa.ChunkedArray) -> _ArrowSnapshot:
    table = value if isinstance(value, pa.Table) else pa.table({"value": value})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    # BufferOutputStream's allocation is writable. Python bytes establishes the
    # immutable ownership boundary, including nested/dictionary/schema buffers.
    payload = sink.getvalue().to_pybytes()
    restored = pa.ipc.open_stream(payload).read_all()
    result = restored if isinstance(value, pa.Table) else restored.column(0)
    if isinstance(value, pa.Array):
        result = result.chunk(0) if result.num_chunks else pa.array([], type=value.type)
    return _ArrowSnapshot(result, payload)


def _index_payload_size(index: Any, seen: set[int]) -> int:
    """Count pandas index allocations, labels and names, without losing aliases."""
    if id(index) in seen:
        return 0
    seen.add(id(index))
    pd = sys.modules["pandas"]
    if isinstance(index, pd.MultiIndex):
        return (
            sum(_index_payload_size(level, seen) for level in index.levels)
            + sum(int(code.nbytes) for code in index.codes)
            + retained_size(tuple(index.names), seen)
        )
    size = int(index.memory_usage(deep=False))
    if isinstance(index, pd.CategoricalIndex):
        # memory_usage already counts category slots, but not their object payloads.
        size += _index_payload_size(index.categories, seen)
    elif _object_pandas_dtype(index.dtype):
        size += sum(retained_size(item, seen) for item in index)
    return size + retained_size(index.name, seen)


def _object_pandas_dtype(dtype: Any) -> bool:
    pd = sys.modules["pandas"]
    return dtype == np.dtype(object) or (
        isinstance(dtype, pd.StringDtype) and dtype.storage == "python"
    )


def _arrow_pandas_dtype(dtype: Any) -> bool:
    pd = sys.modules["pandas"]
    return isinstance(dtype, pd.ArrowDtype) or getattr(dtype, "storage", None) in (
        "pyarrow",
        "pyarrow_numpy",
    )


def _snapshot_index(index: Any, memo: dict[int, Any], *, compact: bool) -> Any:
    if id(index) in memo:
        return memo[id(index)]
    pd = sys.modules["pandas"]
    if _arrow_pandas_dtype(index.dtype) and not isinstance(index.dtype, pd.StringDtype):
        raise TypeError("Arrow-backed pandas indexes require an explicit Arrow cache value")
    name = _snapshot(index.name, memo, compact=compact)
    if isinstance(index, pd.MultiIndex):
        result = pd.MultiIndex(
            levels=[_snapshot_index(level, memo, compact=compact) for level in index.levels],
            codes=[code.copy() for code in index.codes],
            names=[_snapshot(n, memo, compact=compact) for n in index.names],
            sortorder=index.sortorder,
            verify_integrity=False,
        )
    elif isinstance(index, pd.CategoricalIndex):
        categories = _snapshot_index(index.categories, memo, compact=compact)
        result = pd.CategoricalIndex(
            pd.Categorical.from_codes(index.codes.copy(), categories, ordered=index.ordered),
            name=name,
        )
    elif _object_pandas_dtype(index.dtype) or isinstance(index.dtype, pd.StringDtype):
        # pandas 3 defaults string labels to Arrow storage. Reconstruct labels
        # from owned scalars instead of retaining a sliced Arrow parent. This
        # keeps ordinary frames cacheable without admitting arbitrary Arrow
        # extension values whose ownership contract is handled separately.
        values = np.empty(len(index), dtype=object)
        values[:] = [_snapshot(item, memo, compact=compact) for item in index]
        result = pd.Index(values, dtype=index.dtype, name=name, tupleize_cols=False)
    else:
        result = index.copy(deep=True)
        result.name = name
    memo[id(index)] = result
    return result


def contains_object(dtype: pl.DataType | pl.datatypes.DataTypeClass) -> bool:
    """Reject Object wherever it occurs, before any native conversion is attempted."""
    if dtype == pl.Object:
        return True
    if isinstance(dtype, pl.List | pl.Array):
        return contains_object(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(contains_object(field.dtype) for field in dtype.fields)
    return False


def _check_polars(value: pl.DataFrame | pl.Series) -> None:
    dtypes = value.dtypes if isinstance(value, pl.DataFrame) else [value.dtype]
    if any(contains_object(dtype) for dtype in dtypes):
        raise TypeError("Polars Object values require an explicit supported cache representation")


def _native_copy(value: pl.Series) -> pl.Series:
    stream = io.BytesIO()
    value.to_frame().write_ipc(stream, compression="uncompressed")
    return pl.read_ipc(stream.getvalue()).to_series()


def _compact_series(value: pl.Series) -> pl.Series:
    dtype = value.dtype
    if dtype.is_numeric() or dtype.is_temporal() or dtype in (pl.Boolean, pl.Null):
        # clone/rechunk/shrink_to_fit can all retain a sliced external parent.
        # Native gather copies fixed-width buffers, including Int128 and ns times.
        return value.gather(pl.int_range(0, len(value), dtype=pl.get_index_type(), eager=True))
    if dtype.is_nested():
        # Polars 1.32/Arrow 16 can import nested values with inconsistent physical
        # children and panic on newest Arrow export. Native IPC preserves these.
        return _native_copy(value)
    try:
        # The default compatibility level exports offset arrays (not StringView),
        # and concat copies the visible values and nested child buffers. This also
        # avoids retaining a large parent behind a single long-string view.
        copied = pa.concat_arrays([value.to_arrow()])
        result = cast(pl.Series, pl.from_arrow(copied)).rename(value.name)
        return result if result.dtype == dtype else result.cast(dtype)
    except (pa.ArrowException, TypeError, NotImplementedError):
        # Arrow cannot represent every Polars dtype (notably nested Int128).
        # Native IPC preserves the schema and precision, without Python scalars.
        return _native_copy(value)


def compact_polars(value: pl.DataFrame | pl.Series) -> pl.DataFrame | pl.Series:
    """Return native data detached from arbitrary caller-owned parent buffers."""
    _check_polars(value)
    if isinstance(value, pl.Series):
        return _compact_series(value)
    if not value.width:
        return value.clone()
    return pl.DataFrame([_compact_series(column) for column in value])


def _polars_size(value: pl.DataFrame | pl.Series) -> int:
    _check_polars(value)
    if isinstance(value, pl.DataFrame):
        return sum(_polars_size(column) for column in value)
    if value.dtype.is_nested():
        # Native IPC may retain a serialized parent. Count its complete backing
        # and decoded buffers; avoid nested newest-compat exports on Polars 1.32.
        stream = io.BytesIO()
        value.to_frame().write_ipc(stream, compression="uncompressed")
        return 2 * stream.tell()
    try:
        # Unlike estimated_size, this includes StringView descriptors and enum
        # dictionaries. Parent independence is established by compact_polars.
        return int(value.to_arrow(compat_level=pl.CompatLevel.newest()).get_total_buffer_size())
    except (pa.ArrowException, TypeError, NotImplementedError):
        if value.dtype == pl.Int128:
            return int(value.estimated_size())
        # An unsupported nested dtype uses the native IPC ownership fallback.
        # Count both an entire serialized backing allocation and decoded buffers
        # conservatively; IPC includes their descriptors, validity and metadata.
        stream = io.BytesIO()
        value.to_frame().write_ipc(stream, compression="uncompressed")
        return 2 * stream.tell()


def retained_size(value: Any, seen: set[int] | None = None) -> int:
    """Measure retained cache snapshots; arbitrary Polars inputs need compaction first."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, _ArrowSnapshot):
        return sys.getsizeof(value) + sys.getsizeof(value.payload)
    if isinstance(value, pl.DataFrame | pl.Series):
        try:
            return _polars_size(value)
        except TypeError:
            return sys.maxsize
    pd = sys.modules.get("pandas")
    # A native worker may import optional pandas while another thread measures
    # report evidence. sys.modules exposes the module before its classes exist.
    dataframe = getattr(pd, "DataFrame", ())
    series = getattr(pd, "Series", ())
    if isinstance(value, dataframe):
        assert pd is not None
        if any(_arrow_pandas_dtype(dtype) for dtype in value.dtypes):
            return sys.maxsize
        size = int(value.memory_usage(index=False, deep=False).sum())
        size += _index_payload_size(value.index, seen) + _index_payload_size(value.columns, seen)
        size += retained_size(value.attrs, seen)
        size += sum(
            _index_payload_size(value.iloc[:, i].cat.categories, seen)
            for i, dtype in enumerate(value.dtypes)
            if isinstance(dtype, pd.CategoricalDtype)
        )
        return size + sum(
            retained_size(item, seen)
            for i, dtype in enumerate(value.dtypes)
            if _object_pandas_dtype(dtype)
            for item in value.iloc[:, i]
        )
    if isinstance(value, series):
        assert pd is not None
        if _arrow_pandas_dtype(value.dtype):
            return sys.maxsize
        size = int(value.memory_usage(index=False, deep=False))
        size += _index_payload_size(value.index, seen) + retained_size(value.name, seen)
        size += retained_size(value.attrs, seen)
        if isinstance(value.dtype, pd.CategoricalDtype):
            size += _index_payload_size(value.cat.categories, seen)
        return size + (
            sum(retained_size(item, seen) for item in value)
            if _object_pandas_dtype(value.dtype)
            else 0
        )
    if isinstance(value, np.ndarray):
        return sys.maxsize if value.dtype.hasobject else int(value.nbytes)
    if isinstance(value, pa.Table | pa.Array | pa.ChunkedArray):
        return int(value.get_total_buffer_size())
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(retained_size(k, seen) + retained_size(v, seen) for k, v in value.items())
    if isinstance(value, tuple | list | set | frozenset):
        return size + sum(retained_size(v, seen) for v in value)
    if value is None or value is getattr(pd, "NA", None) or isinstance(value, _IMMUTABLE):
        return size
    return sys.maxsize


def _snapshot(value: Any, memo: dict[int, Any] | None = None, *, compact: bool = False) -> Any:
    """Detach wrappers on hits; additionally detach native parents on insertion."""
    memo = {} if memo is None else memo
    result: Any
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, _ArrowSnapshot):
        return value.value
    if isinstance(value, pl.DataFrame | pl.Series):
        _check_polars(value)
        result = memo[id(value)] = compact_polars(value) if compact else value.clone()
        return result
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays require an explicit cache representation")
        result = memo[id(value)] = value.copy()
        return result
    pd = sys.modules.get("pandas")
    dataframe = getattr(pd, "DataFrame", ())
    series = getattr(pd, "Series", ())
    if isinstance(value, (dataframe, series)):
        assert pd is not None
        result = memo[id(value)] = value.copy(deep=True)
        result.index = _snapshot_index(value.index, memo, compact=compact)
        result.attrs = _snapshot(value.attrs, memo, compact=compact)
        if isinstance(result, dataframe):
            result.columns = _snapshot_index(value.columns, memo, compact=compact)
            for i, dtype in enumerate(result.dtypes):
                if _arrow_pandas_dtype(dtype):
                    raise TypeError(
                        "Arrow-backed pandas columns require an explicit Arrow cache value"
                    )
                if _object_pandas_dtype(dtype):
                    result.isetitem(
                        i,
                        result.iloc[:, i]
                        .map(lambda item: _snapshot(item, memo, compact=compact))
                        .astype(dtype),
                    )
                elif isinstance(dtype, pd.CategoricalDtype):
                    column = result.iloc[:, i]
                    result.isetitem(
                        i,
                        pd.Categorical.from_codes(
                            column.cat.codes.to_numpy(copy=True),
                            _snapshot_index(column.cat.categories, memo, compact=compact),
                            ordered=dtype.ordered,
                        ),
                    )
        elif _object_pandas_dtype(result.dtype):
            result = result.map(lambda item: _snapshot(item, memo, compact=compact)).astype(
                value.dtype
            )
            memo[id(value)] = result
        elif _arrow_pandas_dtype(result.dtype):
            raise TypeError("Arrow-backed pandas series require an explicit Arrow cache value")
        elif isinstance(result.dtype, pd.CategoricalDtype):
            result = pd.Series(
                pd.Categorical.from_codes(
                    result.cat.codes.to_numpy(copy=True),
                    _snapshot_index(result.cat.categories, memo, compact=compact),
                    ordered=result.dtype.ordered,
                ),
                index=result.index,
                name=result.name,
            )
            result.attrs = _snapshot(value.attrs, memo, compact=compact)
            memo[id(value)] = result
        return result
    if isinstance(value, pa.Table | pa.Array | pa.ChunkedArray):
        result = memo[id(value)] = _own_arrow(value) if compact else value
        return result
    if isinstance(value, dict):
        mapped: dict[Any, Any] = {}
        memo[id(value)] = mapped
        mapped.update(
            {
                _snapshot(k, memo, compact=compact): _snapshot(v, memo, compact=compact)
                for k, v in value.items()
            }
        )
        return mapped
    if isinstance(value, list):
        items: list[Any] = []
        memo[id(value)] = items
        items.extend(_snapshot(v, memo, compact=compact) for v in value)
        return items
    if isinstance(value, tuple | set | frozenset):
        result = type(value)(_snapshot(v, memo, compact=compact) for v in value)
        memo[id(value)] = result
        return result
    if value is None or value is getattr(pd, "NA", None) or isinstance(value, _IMMUTABLE):
        return value
    raise TypeError("opaque cache values require a supported immutable representation")
