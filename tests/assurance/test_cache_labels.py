"""Pandas label ownership across explicit storage choices and dependency versions."""

import pandas as pd
import pytest

from qavf.engine.resources import ByteCache

pytestmark = pytest.mark.assurance


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("axis", ["index", "columns", "categories"])
def test_string_labels_keep_ordinary_frames_cacheable(storage, axis):
    dtype = pd.StringDtype(storage=storage)
    labels = pd.Index(pd.array(["alpha", "beta"], dtype=dtype))
    if axis == "index":
        frame = pd.DataFrame({"x": [1, 2]}, index=labels)
    elif axis == "columns":
        frame = pd.DataFrame([[1, 2]], columns=labels)
    else:
        frame = pd.DataFrame({"x": pd.Categorical.from_codes([0, 1], categories=labels)})
    expected = frame.copy(deep=True)
    cache = ByteCache(10000)
    cache["frame"] = frame
    hit = cache["frame"]
    pd.testing.assert_frame_equal(hit, expected)
    if axis == "index":
        hit.index = ["changed", "labels"]
        frame.index = ["source", "changed"]
    elif axis == "columns":
        hit.columns = ["changed", "labels"]
        frame.columns = ["source", "changed"]
    else:
        hit["x"] = hit["x"].cat.rename_categories(["changed", "labels"])
        frame["x"] = frame["x"].cat.rename_categories(["source", "changed"])
    pd.testing.assert_frame_equal(cache["frame"], expected)


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_large_string_label_payload_is_not_omitted(storage):
    labels = pd.Index(pd.array(["x" * 100_000], dtype=pd.StringDtype(storage=storage)))
    cache = ByteCache(1024)
    cache["frame"] = pd.DataFrame({"x": [1]}, index=labels)
    assert "frame" not in cache
    assert cache.bytes == 0
