"""Regression gates for retained ownership, cache copies and spill reuse."""

from __future__ import annotations

import gc
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from unittest.mock import patch

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from qavf.engine import resources
from qavf.engine.cache_ownership import compact_polars, contains_object
from qavf.engine.resources import (
    ByteCache,
    FrameStore,
    ResourceLimitError,
    RunResources,
    retained_size,
)


def test_cache_helpers_tolerate_concurrent_optional_pandas_initialization(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", ModuleType("pandas"))
    value = {"sample": [1, 2, 3]}
    cache = ByteCache(4096)
    assert 0 < retained_size(value) < cache.limit
    cache["sample"] = value
    copied = cache["sample"]
    assert copied == value
    assert copied is not value
    assert 0 < retained_size(copied) < cache.bytes


@pytest.mark.parametrize("kind", ["series", "frame", "struct", "list", "chunked"])
@pytest.mark.parametrize("offset", [0, 7])
def test_polars_slice_does_not_retain_numpy_parent(kind, offset):
    parent = np.arange(100_000, dtype=np.int64)
    reference = weakref.ref(parent)
    series = pl.Series("x", parent).slice(offset, 2)
    if kind == "series":
        value = series
    elif kind == "frame":
        value = series.to_frame()
    elif kind == "struct":
        value = series.to_frame().to_struct("nested")
    elif kind == "list":
        value = pl.from_arrow(pa.LargeListArray.from_arrays([0, 2], series.to_arrow()))
    else:
        value = pl.concat([series, pl.Series("x", [99])], rechunk=False)
    cache = ByteCache(4096)
    cache["key"] = value
    expected = value.clone()
    # Expected must not itself keep the external allocation alive.
    expected = compact_polars(expected)
    del parent, series, value
    gc.collect()
    assert reference() is None
    assert cache["key"].equals(expected)
    assert 0 < cache.bytes <= cache.limit


def test_many_slices_release_every_external_parent():
    cache = ByteCache(4096)
    references = []
    for number in range(40):
        parent = np.arange(100_000, dtype=np.int64)
        references.append(weakref.ref(parent))
        cache[number] = pl.Series("x", parent).slice(number, 1)
    del parent
    gc.collect()
    assert all(reference() is None for reference in references)
    assert cache.bytes <= cache.limit


def test_frame_store_compacts_source_owned_parent(tmp_path):
    parent = np.arange(100_000, dtype=np.int64)
    reference = weakref.ref(parent)
    frame = pl.DataFrame({"x": parent}).slice(5, 1)
    store = FrameStore(128, 10000, tmp_path, materialization_bytes=128)
    store["data"] = frame
    del parent, frame
    gc.collect()
    assert reference() is None
    assert store["data"].item() == 5
    assert store.bytes == 8


def test_long_string_slice_compacts_backing_storage(tmp_path):
    frame = pl.DataFrame({"text": ["x" * 1000] * 1000}).slice(7, 1)
    store = FrameStore(2048, 10000, tmp_path)
    store["data"] = frame
    assert store["data"].equals(frame)
    assert 1000 <= store.bytes <= 2048


def test_string_view_descriptors_and_enum_categories_are_counted(tmp_path):
    strings = pl.DataFrame({"x": ["a"] * 1000})
    store = FrameStore(2000, 100000, tmp_path)
    store["strings"] = strings
    assert "strings" in store.paths  # the 16-byte StringView slots exceed RAM
    enum = pl.Series("x", ["a"], dtype=pl.Enum(["a", "b" * 10000]))
    assert retained_size(compact_polars(enum)) >= 10000
    cache = ByteCache(2048)
    cache["enum"] = enum
    assert "enum" not in cache


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("x", [2**100, None, -(2**110)], dtype=pl.Int128),
        pl.Series("x", [1, None, -3], dtype=pl.Duration("ns")),
        pl.Series("x", [1, None, 3], dtype=pl.Datetime("ns", "UTC")),
        pl.Series("x", ["b", None, "a"], dtype=pl.Enum(["b", "a", "unused"])),
        pl.Series("x", ["b", None, "a"], dtype=pl.Categorical),
        pl.Series("x", [[1, None], [], None], dtype=pl.List(pl.Duration("ns"))),
        pl.Series("x", [["2025-01-01T00:00:00.000000000"], [], None]),
        pl.Series("x", [{"s": "hello"}, None], dtype=pl.Struct({"s": pl.String})),
        pl.Series("x", [["a", "b"], None], dtype=pl.Array(pl.String, 2)),
        pl.Series(
            "x",
            [{"n": 2**100, "s": "hello"}, None],
            dtype=pl.Struct({"n": pl.Int128, "s": pl.String}),
        ),
        pl.Series("x", [True, None, False]),
        pl.Series("x", [float("nan"), None, float("inf")]),
        pl.Series("x", [], dtype=pl.Int128),
    ],
)
def test_exact_dtype_and_values_survive_cache_and_spill(series, tmp_path):
    cache = ByteCache(100000)
    cache["value"] = series
    assert cache["value"].dtype == series.dtype
    assert cache["value"].equals(series)
    store = FrameStore(0, 100000, tmp_path)
    store["data"] = series.to_frame()
    result = store["data"].to_series()
    assert result.dtype == series.dtype
    assert result.equals(series)
    assert store.bytes == 0
    if "data" in store.paths:
        assert store.spill_read_count == 1
        assert store.spill_read_bytes == store.paths["data"].stat().st_size
    else:
        assert len(series) == 0
        assert store.spill_read_count == 0


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Object,
        pl.List(pl.Object),
        pl.Array(pl.Object, 2),
        pl.Struct({"outer": pl.List(pl.Struct({"inner": pl.Object}))}),
    ],
)
def test_object_rejection_checks_every_dtype_level(dtype):
    # Current Polars rejects construction of nested Object itself. Validate the
    # recursive guard without provoking a Rust panic in those constructors.
    assert contains_object(dtype)


@pytest.mark.parametrize(
    "wrapper", [lambda v: v, lambda v: v.to_frame(), lambda v: {"outer": [v]}, lambda v: (v,)]
)
def test_object_series_is_never_retained_in_any_container(wrapper):
    payload = {"state": [], "native_payload": bytearray(1_000_000)}
    value = wrapper(pl.Series("object", [payload], dtype=pl.Object))
    cache = ByteCache(4096)
    cache["key"] = value
    payload["state"].append("mutated")
    assert "key" not in cache
    assert cache.bytes == 0
    with pytest.raises(KeyError):
        cache["key"]


def test_frame_store_explicitly_rejects_object_and_keeps_old_value(tmp_path):
    store = FrameStore(4096, 10000, tmp_path)
    store["data"] = pl.DataFrame({"x": [1]})
    with pytest.raises(TypeError, match="Polars Object"):
        store["data"] = pl.DataFrame([pl.Series("x", [object()], dtype=pl.Object)])
    assert store["data"].item() == 1
    assert store.bytes == 8


def test_object_polars_in_pandas_cell_is_not_cached():
    pd = pytest.importorskip("pandas")
    value = pd.DataFrame({"x": [pl.Series("nested", [object()], dtype=pl.Object)]})
    cache = ByteCache(4096)
    cache["key"] = value
    assert "key" not in cache


def test_cache_membership_clear_discard_do_not_snapshot():
    cache = ByteCache(100000)
    cache["a"] = np.arange(1000)
    cache["b"] = np.arange(1000)
    with patch.object(resources, "_snapshot", wraps=resources._snapshot) as snapshot:
        assert "a" in cache and "absent" not in cache
        assert len(cache) == 2
        list(cache)
        cache.discard("absent")
        cache.discard("a")
        del cache["b"]
        cache.clear()
        assert snapshot.call_count == 0
    assert cache.bytes == 0
    assert cache.peak_bytes > 0


def test_cache_hit_and_pop_each_make_one_snapshot():
    cache = ByteCache(100000)
    cache["a"] = np.arange(1000)
    with patch.object(resources, "_snapshot", wraps=resources._snapshot) as snapshot:
        hit = cache["a"]
        assert snapshot.call_count == 1
        hit[0] = 999
        assert cache.pop("a")[0] == 0
        assert snapshot.call_count == 2
    assert cache.bytes == 0


def test_frame_hits_and_insertion_are_isolated(tmp_path):
    store = FrameStore(1000, 10000, tmp_path)
    original = pl.DataFrame({"x": [1, 2]})
    store["data"] = original
    original[0, "x"] = 9
    hit = store["data"]
    hit[0, "x"] = 7
    assert store["data"][0, "x"] == 1


def test_spill_promotes_only_after_capacity_is_free(tmp_path):
    store = FrameStore(8, 10000, tmp_path)
    store["a"] = pl.DataFrame({"x": [1]})
    store["b"] = pl.DataFrame({"x": [2]})
    size = store.paths["a"].stat().st_size
    writes = store.spill_write_count
    for _ in range(2):
        assert store["a"].item() == 1
    assert store.spill_read_count == 2
    assert store.spill_read_bytes == 2 * size
    assert store.spill_promotions == 0
    assert store.spill_write_count == writes
    del store["b"]
    assert store["a"].item() == 1
    assert store.spill_promotions == 1
    assert store.spill_read_bytes == 3 * size
    assert store.spill_bytes == 0
    assert store.bytes == 8
    assert not store.paths and not list(store.directory.iterdir())
    assert store["a"].item() == 1
    assert store.spill_read_count == 3
    assert store.spill_write_count == writes


def test_parallel_spill_hits_promote_once(tmp_path):
    store = FrameStore(8, 10000, tmp_path)
    store["a"] = pl.DataFrame({"x": [1]})
    store["b"] = pl.DataFrame({"x": [2]})
    del store["b"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(lambda _: store["a"].item(), range(20))) == [1] * 20
    assert store.spill_read_count == store.spill_promotions == 1
    assert store.bytes <= store.limit


def test_frame_clear_and_membership_never_read_spill(tmp_path):
    store = FrameStore(0, 10000, tmp_path)
    store["a"] = pl.DataFrame({"x": [1]})
    with patch.object(pl, "read_ipc", side_effect=AssertionError("unexpected read")):
        assert "a" in store
        store.discard("missing")
        store.clear()
    assert store.bytes == store.spill_bytes == 0
    assert not store.sizes and not store.paths
    assert not list(store.directory.iterdir())


def test_run_resources_exports_exact_io_and_promotion_counts(tmp_path):
    owned = RunResources(16, 20000, None, tmp_path)
    owned.frames["a"] = pl.DataFrame({"x": [1]})
    owned.frames["b"] = pl.DataFrame({"x": [2]})
    size = owned.frames.paths["a"].stat().st_size
    del owned.frames["b"]
    owned.frames["a"]
    usage = owned.usage()
    assert usage["frame_spill_read_count"] == 1
    assert usage["frame_spill_read_bytes"] == size
    assert usage["frame_spill_promotions"] == 1
    assert usage["frame_spill_write_count"] == 1
    assert usage["frame_spill_write_bytes"] == size
    owned.close()
    assert not owned.directory.exists()


def test_failed_spill_write_stays_within_budget_and_leaves_no_partial_file(tmp_path):
    store = FrameStore(0, 10, tmp_path)
    with pytest.raises(ResourceLimitError):
        store["a"] = pl.DataFrame({"x": [1]})
    assert not store.paths and not store.sizes
    assert store.bytes == store.spill_bytes == 0
    assert not list(store.directory.iterdir())
