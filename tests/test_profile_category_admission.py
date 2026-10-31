"""Early cardinality refusal preserves complete profiling semantics."""

import polars as pl

from qavf.profiling import profile_frame
from qavf.profiling.accumulator import ProfileAccumulator


def test_high_cardinality_retains_exact_lengths_and_missing_count():
    values = [f"code-{i}" for i in range(65536)] + [None, "longest-code-in-the-input"]
    result = profile_frame(pl.DataFrame({"code": values}), max_categories=20)
    column = result["columns"]["code"]
    assert "isin" not in column
    assert column["nullable"] is True
    assert column["min_length"] == 6
    assert column["max_length"] == len("longest-code-in-the-input")


def test_late_categories_are_not_hidden_by_a_low_cardinality_prefix():
    accumulator = ProfileAccumulator({"code": pl.String}, max_categories=2, bounds=True)
    accumulator.update(pl.DataFrame({"code": ["a"] * 1000}))
    accumulator.update(pl.DataFrame({"code": ["a"] * 1000 + ["b", "ccc"]}))
    result = accumulator.finish(0)
    assert result.rows == 2002
    assert result.schema["columns"]["code"] == {"dtype": "str", "min_length": 1, "max_length": 3}


def test_low_cardinality_enumeration_is_complete_across_batches():
    accumulator = ProfileAccumulator({"code": pl.String}, max_categories=3, bounds=True)
    for value in ("c", "a", "b", "c"):
        accumulator.update(pl.DataFrame({"code": [value] * 1000}))
    assert accumulator.finish(0).schema["columns"]["code"]["isin"] == ["a", "b", "c"]


def test_oversized_category_still_obeys_byte_admission():
    result = profile_frame(
        pl.DataFrame({"code": ["a"] * 300 + ["z" * 100]}), max_category_value_bytes=32
    )
    column = result["columns"]["code"]
    assert "isin" not in column
    assert "byte limit" in column["description"]
    assert column["max_length"] == 100
