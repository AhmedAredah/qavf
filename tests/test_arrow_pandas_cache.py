"""Ownership and retained-size regressions for Arrow and pandas cache values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from qavf.engine.resources import ByteCache


@pytest.mark.parametrize("kind", ["array", "chunked", "table"])
def test_arrow_cache_owns_immutable_buffers(kind):
    source = np.arange(4, dtype=np.int64)
    array = pa.array(source)
    value = (
        array
        if kind == "array"
        else pa.chunked_array([array])
        if kind == "chunked"
        else pa.table({"x": array})
    )
    cache = ByteCache(10000)
    cache["value"] = value
    source[0] = 99
    hit = cache["value"]
    column = hit.column(0) if kind == "table" else hit
    restored = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
    assert restored.to_pylist() == [0, 1, 2, 3]
    assert not restored.buffers()[1].is_mutable
    with pytest.raises(ValueError):
        np.frombuffer(restored.buffers()[1], dtype=np.int64)[0] = 88
    assert cache["value"].equals(value) is False


def test_arrow_nested_dictionary_and_schema_survive_owned_snapshot():
    table = pa.table(
        {
            "nested": pa.array([[1, None], None, []]),
            "category": pa.array(["a", "b", "a"]).dictionary_encode(),
        }
    ).replace_schema_metadata({b"origin": b"test"})
    cache = ByteCache(10000)
    cache["value"] = {"a": table, "b": table}
    hit = cache["value"]
    assert hit["a"] is hit["b"]
    assert hit["a"].equals(table, check_metadata=True)


def test_arrow_ipc_metadata_backing_counts_toward_budget():
    table = pa.table({"x": [1]}).replace_schema_metadata({b"large": b"x" * 10000})
    cache = ByteCache(1024)
    cache["value"] = table
    assert "value" not in cache


@pytest.mark.parametrize("kind", ["frame", "series"])
def test_pandas_index_aliases_are_detached_on_insert_and_hit(kind):
    source = pd.DataFrame({"x": [1, 2]}, index=["first", "second"])
    if kind == "series":
        source = source["x"]
    cache = ByteCache(10000)
    cache["value"] = source
    source.index.values[0] = "outside"
    hit = cache["value"]
    assert hit.index.tolist() == ["first", "second"]
    hit.index.values[1] = "hit"
    assert cache["value"].index.tolist() == ["first", "second"]


def test_pandas_column_index_is_detached():
    source = pd.DataFrame([[1, 2]], columns=["a", "b"])
    cache = ByteCache(10000)
    cache["value"] = source
    source.columns.values[0] = "outside"
    hit = cache["value"]
    hit.columns.values[1] = "hit"
    assert cache["value"].columns.tolist() == ["a", "b"]


@pytest.mark.parametrize("kind", ["object_index", "multi_index", "columns", "categorical", "attrs"])
def test_pandas_large_label_and_metadata_payloads_cannot_bypass_budget(kind):
    large = "x" * 100000
    frame = pd.DataFrame({"x": [1, 2]})
    if kind == "object_index":
        frame.index = [large, "y"]
    elif kind == "multi_index":
        frame.index = pd.MultiIndex.from_tuples([(large, 1), ("y", 2)])
    elif kind == "columns":
        frame.columns = [large]
    elif kind == "categorical":
        frame["x"] = pd.Categorical([large, "y"])
    else:
        frame.attrs = {"description": large}
    cache = ByteCache(1000)
    cache["value"] = frame
    assert "value" not in cache
    assert cache.bytes == 0


def test_pandas_categorical_index_and_values_are_detached():
    frame = pd.DataFrame({"x": pd.Categorical(["a", "b"])}, index=pd.CategoricalIndex(["a", "b"]))
    cache = ByteCache(10000)
    cache["value"] = frame
    frame.index.categories.values[0] = "outside"
    frame["x"].cat.categories.values[0] = "outside"
    hit = cache["value"]
    assert hit.index.tolist() == ["a", "b"]
    assert hit["x"].tolist() == ["a", "b"]


def test_pandas_arrow_extension_is_safely_uncached():
    source = np.arange(3)
    frame = pd.DataFrame({"x": pd.Series(pa.array(source), dtype=pd.ArrowDtype(pa.int64()))})
    cache = ByteCache(10000)
    cache["value"] = frame
    assert "value" not in cache


def test_python_string_extension_payload_is_counted_and_dtype_preserved():
    frame = pd.DataFrame({"s": pd.Series(["short", pd.NA], dtype="string[python]")})
    cache = ByteCache(10000)
    cache["value"] = frame
    frame.loc[0, "s"] = "changed"
    hit = cache["value"]
    assert hit["s"].dtype == frame["s"].dtype
    assert hit.loc[0, "s"] == "short"
    assert hit.loc[1, "s"] is pd.NA
    large = pd.DataFrame({"s": pd.Series(["x" * 100000], dtype="string[python]")})
    cache["large"] = large
    assert "large" not in cache


def test_arrow_backed_pandas_index_is_safely_uncached():
    source = np.arange(3)
    index = pd.Index(pd.array(pa.array(source), dtype=pd.ArrowDtype(pa.int64())))
    cache = ByteCache(10000)
    cache["value"] = pd.DataFrame({"x": [1, 2, 3]}, index=index)
    assert "value" not in cache
