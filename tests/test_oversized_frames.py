"""A frame that can never fit the cache is spilled directly (EP-03).

It must not evict or spill the in-memory entries first: that wastes writes and
can falsely exhaust a spill budget that holds the large frame alone.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from qavf.engine.resources import FrameStore
from qavf.resource_errors import ResourceLimitError

pl = pytest.importorskip("polars")

HOT = pl.DataFrame({"x": list(range(40))})
LARGE = pl.DataFrame({"x": list(range(1000))})


def _no_metadata() -> None:
    return None


def _store(tmp_path: Path, spill: int = 9000) -> FrameStore:
    return FrameStore(1024, spill, tmp_path)


def test_oversized_frame_spills_alone_and_keeps_the_hot_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store["hot"] = HOT
    store["large"] = LARGE  # fits the spill budget alone, not with hot spilled too
    assert list(store.data) == ["hot"]
    assert sorted(store._spill_files) == ["large"]
    assert store.spill_write_count == 1
    assert store["large"].equals(LARGE) and store["hot"].equals(HOT)
    store.close()
    assert not list(tmp_path.rglob("*.arrow"))


def test_replacing_a_key_with_an_oversized_frame(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store["hot"] = HOT
    store["other"] = HOT
    store["other"] = LARGE
    assert list(store.data) == ["hot"]
    assert sorted(store._spill_files) == ["other"]
    assert store.spill_write_count == 1
    assert store["other"].equals(LARGE)
    store.close()


def test_shared_lease_survives_an_oversized_insertion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store["hot"] = HOT
    frame, lease = store.observe("hot")
    store["large"] = LARGE
    assert list(store.data) == ["hot"] and store.spill_write_count == 1
    assert frame is not None and frame.equals(HOT)
    lease.release()
    assert store["hot"].equals(HOT)
    store.close()


def test_failed_spill_write_keeps_unrelated_entries(tmp_path: Path) -> None:
    store = _store(tmp_path, spill=4000)  # too small for the large frame
    store["hot"] = HOT
    with pytest.raises(ResourceLimitError):
        store["large"] = LARGE
    assert list(store.data) == ["hot"] and not store._spill_files
    assert store.spill_write_count == 0
    assert store["hot"].equals(HOT) and "large" not in store
    store.close()
    assert not list(tmp_path.rglob("*.arrow"))


def test_publish_does_not_churn_for_an_oversized_member(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store["hot"] = HOT
    store.publish({"large": LARGE}, _no_metadata, contextlib.nullcontext())
    assert list(store.data) == ["hot"]
    assert sorted(store._spill_files) == ["large"]
    assert store.spill_write_count == 1
    assert store["large"].equals(LARGE)
    store.close()


def test_publish_still_makes_room_for_a_fitting_frame(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store["a"] = HOT
    store["b"] = HOT
    store["c"] = HOT
    store.publish({"new": HOT}, _no_metadata, contextlib.nullcontext())
    assert "new" in store.data  # newly loaded data replaces the least recently used
    assert "a" in store._spill_files
    store.close()
