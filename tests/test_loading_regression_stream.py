"""Exact streamed-file regression matches the independent in-memory API contract."""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from qavf import ResourceLimits
from qavf.engine.sql_read import SqlReadSession
from qavf.regression import _comparison, compare_frames, compare_outputs
from qavf.sources.file import FileSource


def write_pair(tmp_path, baseline, candidate, file_format="parquet"):
    left, right = tmp_path / f"baseline.{file_format}", tmp_path / f"candidate.{file_format}"
    for frame, path in ((baseline, left), (candidate, right)):
        if file_format == "csv":
            frame.write_csv(path)
        else:
            frame.write_parquet(path)
    return left, right


def semantic_result(result):
    # JSON mode preserves nonfinite evidence as comparable tagged values.
    return result.model_dump(mode="json", exclude={"execution", "fallback_reason", "name"})


def assert_file_parity(tmp_path, baseline, candidate, keys, *, file_format="parquet", **options):
    expected = compare_frames(baseline, candidate, keys, **options)
    paths = write_pair(tmp_path, baseline, candidate, file_format)
    actual = compare_outputs(*paths, keys, resources=ResourceLimits(native_threads=1), **options)
    assert semantic_result(actual) == semantic_result(expected)
    return actual


def forbid_eager_native_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native file comparison must not fully load or join Python frames")

    monkeypatch.setattr(FileSource, "load", forbidden)
    monkeypatch.setattr(pl.DataFrame, "join", forbidden)


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_scalar_native_files_never_load_or_join_full_python_frames(
    tmp_path, monkeypatch, file_format
):
    baseline = pl.DataFrame(
        {"id": [4, 1, 3, 2], "value": [0.25, 1.0, 3.5, 2.0], "label": ["d", "a", "c", "b"]}
    )
    candidate = pl.DataFrame(
        {"id": [2, 3, 1, 5], "value": [2.0, 3.75, 1.0, 5.0], "label": ["B", "c", "a", "e"]}
    )
    expected = compare_frames(baseline, candidate, "id", rtol=0, atol=0)
    paths = write_pair(tmp_path, baseline, candidate, file_format)
    forbid_eager_native_work(monkeypatch)
    actual = compare_outputs(
        *paths, "id", rtol=0, atol=0, resources=ResourceLimits(native_threads=1)
    )
    assert actual.execution == "native_stream" and actual.fallback_reason is None
    assert semantic_result(actual) == semantic_result(expected)
    assert (actual.rows_added, actual.rows_removed, actual.rows_changed, actual.values_changed) == (
        1,
        1,
        2,
        2,
    )


@pytest.mark.parametrize("rows,batch_rows", [(16_391, 16_384), (17, 5)])
def test_streaming_preserves_evidence_order_across_batches_and_unmatched_prefixes(
    tmp_path, monkeypatch, rows, batch_rows
):
    monkeypatch.setattr(_comparison, "_BATCH_ROWS", batch_rows)
    changed = sorted({0, 1, batch_rows - 2, batch_rows - 1, batch_rows, batch_rows + 1, rows - 1})
    removed, added = [-3, -1, -2], [rows + 4, rows + 2, rows + 3, rows + 1]
    baseline = pl.DataFrame(
        {"id": removed + list(range(rows)), "first": [0] * (rows + 3), "second": [0.0] * (rows + 3)}
    )
    candidate_keys = added + list(reversed(range(rows)))
    candidate = pl.DataFrame(
        {
            "id": candidate_keys,
            "first": [1 if key in changed else 0 for key in candidate_keys],
            "second": [1.0 if key in changed else 0.0 for key in candidate_keys],
        }
    )
    expected = compare_frames(baseline, candidate, "id", rtol=0, atol=0)
    paths = write_pair(tmp_path, baseline, candidate)
    received = []
    original = SqlReadSession.batches

    def bounded_batches(self, query, batch_rows=65536):
        for batch in original(self, query, batch_rows=batch_rows):
            received.append(batch.height)
            assert batch.height <= batch_rows
            yield batch

    monkeypatch.setattr(SqlReadSession, "batches", bounded_batches)
    forbid_eager_native_work(monkeypatch)
    actual = compare_outputs(
        *paths, "id", rtol=0, atol=0, resources=ResourceLimits(native_threads=1)
    )
    assert actual.execution == "native_stream"
    assert semantic_result(actual) == semantic_result(expected)
    assert actual.rows_changed == len(changed) and actual.values_changed == 2 * len(changed)
    assert actual.rows_removed == len(removed) and actual.rows_added == len(added)
    assert len(received) > 1 and sum(received) == rows + len(removed) + len(added)
    evidence_order = [(item.kind, item.key["id"], item.column) for item in actual.differences]
    expected_order = [("row_removed", key, None) for key in removed]
    expected_order += [("row_added", key, None) for key in added]
    for start in range(0, rows, batch_rows):
        for column in ("first", "second"):
            expected_order += [
                ("value_changed", key, column)
                for key in changed
                if start <= key < start + batch_rows
            ]
    assert evidence_order == expected_order


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_null_composite_keys_match_exactly_and_reordering_is_irrelevant(tmp_path, file_format):
    baseline = pl.DataFrame(
        {"id": [None, None, 1, 2], "region": ["A", "a", None, "é"], "value": [10, 20, 30, 40]}
    )
    candidate = baseline.reverse().with_columns(
        pl.when(pl.col("region") == "a").then(21).otherwise(pl.col("value")).alias("value")
    )
    result = assert_file_parity(
        tmp_path, baseline, candidate, ["id", "region"], file_format=file_format, rtol=0, atol=0
    )
    assert result.execution == "native_stream"
    assert result.rows_added == result.rows_removed == 0
    assert result.values_changed == 1 and result.differences[0].key == {"id": None, "region": "a"}


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_text_keys_preserve_unicode_nuls_and_literal_blob_escape_sequences(tmp_path, file_format):
    keys = ["A", "a", r"\x41", "é", "e\u0301", "汉字", "🙂", "a\x00b", r"back\slash"]
    baseline = pl.DataFrame({"id": keys, "value": list(range(len(keys)))})
    candidate = baseline.reverse().with_columns(
        pl.when(pl.col("id") == r"\x41")
        .then(pl.col("value") + 1)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    result = assert_file_parity(
        tmp_path, baseline, candidate, "id", file_format=file_format, rtol=0, atol=0
    )
    assert result.execution == "native_stream" and result.values_changed == 1
    assert result.rows_added == result.rows_removed == 0
    assert result.differences[0].key == {"id": r"\x41"}


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_native_date_keys_and_values_remain_exact_under_numeric_tolerance(tmp_path, file_format):
    baseline = pl.DataFrame(
        {
            "id": [date(2024, 1, 1), date(2024, 1, 2), None],
            "value": [date(2020, 2, 28), date(2020, 2, 29), None],
        }
    )
    candidate = baseline.reverse().with_columns(
        pl.when(pl.col("id") == date(2024, 1, 1))
        .then(pl.lit(date(2020, 2, 29)))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    result = assert_file_parity(
        tmp_path, baseline, candidate, "id", file_format=file_format, rtol=1, atol=1e9
    )
    assert result.execution == "native_stream" and result.values_changed == 1
    assert result.differences[0].key == {"id": "2024-01-01"}


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize("invalid", ["nan", "duplicates", "null_duplicates"])
def test_invalid_native_keys_raise_the_same_error_before_comparison(
    tmp_path, file_format, side, invalid
):
    valid = pl.DataFrame({"id": [1.0, 2.0, 3.0], "value": [10, 20, 30]})
    invalid_keys = {
        "nan": [1.0, math.nan, 3.0],
        "duplicates": [1.0, 1.0, 3.0],
        "null_duplicates": [None, None, 3.0],
    }[invalid]
    bad = valid.with_columns(pl.Series("id", invalid_keys, dtype=pl.Float64))
    baseline, candidate = (bad, valid) if side == "baseline" else (valid, bad)
    paths = write_pair(tmp_path, baseline, candidate, file_format)
    with pytest.raises(ValueError) as expected:
        compare_frames(baseline, candidate, "id", name="keys")
    with pytest.raises(ValueError) as actual:
        compare_outputs(*paths, "id", name="keys", resources=ResourceLimits(native_threads=1))
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_quoted_reserved_and_internal_looking_column_names_are_literal(tmp_path, file_format):
    key = 'select" key'
    baseline = pl.DataFrame(
        {
            key: [2, 1, 3],
            "from": [2, 1, 3],
            "br": ["b", "a", "c"],
            "b0": [True, False, True],
            "__qavf_comparison_ordinal": [2.0, 1.0, 3.0],
            "__qavf_brow": [2, 1, 3],
            "value; DROP TABLE baseline; --": [20, 10, 30],
        }
    )
    candidate = baseline.reverse().with_columns((pl.col("from") + 1).alias("from"))
    result = assert_file_parity(tmp_path, baseline, candidate, key, file_format=file_format)
    assert result.execution == "native_stream" and result.values_changed == 3


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_large_signed_integer_differences_are_not_rounded_through_float(tmp_path, file_format):
    baseline = pl.DataFrame({"id": [1, 2, 3], "value": [2**53 + 1, 2**63 - 2, -(2**63)]})
    candidate = baseline.with_columns((pl.col("value") + 1).alias("value"))
    result = assert_file_parity(
        tmp_path, baseline, candidate, "id", file_format=file_format, rtol=0, atol=0
    )
    assert result.execution == "native_stream" and result.values_changed == 3
    assert result.max_abs_diff == 1
    assert result.differences[0].baseline == 2**53 + 1


def test_unsigned_integer_boundaries_remain_exact(tmp_path):
    baseline = pl.DataFrame(
        {"id": [1, 2, 3], "value": pl.Series([2**64 - 2, 2**63 + 1, 2**53 + 1], dtype=pl.UInt64)}
    )
    candidate = baseline.with_columns((pl.col("value") + 1).alias("value"))
    result = assert_file_parity(tmp_path, baseline, candidate, "id", rtol=0, atol=0)
    assert result.execution == "native_stream" and result.values_changed == 3
    assert result.differences[0].candidate == 2**64 - 1


def test_decimal_38_digit_values_keep_small_exact_differences(tmp_path):
    before = [Decimal("12345678901234567890.123456789012345678"), Decimal("0.100000000000000000")]
    after = [Decimal("12345678901234567890.123456789012345679"), Decimal("0.100000000000000001")]
    baseline = pl.DataFrame({"id": [1, 2], "value": pl.Series(before, dtype=pl.Decimal(38, 18))})
    candidate = pl.DataFrame({"id": [1, 2], "value": pl.Series(after, dtype=pl.Decimal(38, 18))})
    result = assert_file_parity(tmp_path, baseline, candidate, "id", rtol=0, atol=0)
    assert result.execution == "native_stream" and result.values_changed == 2
    assert result.max_abs_diff == 1e-18
    assert result.differences[0].baseline == before[0]


@pytest.mark.parametrize("atol,changed", [(0, 1), (1e-16, 0)])
def test_mixed_decimal_and_float_values_use_exact_tolerance(tmp_path, atol, changed):
    baseline = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": pl.Series([Decimal("0.1"), Decimal("1"), None], dtype=pl.Decimal(10, 1)),
        }
    )
    candidate = pl.DataFrame({"id": [1, 2, 3], "value": [0.1, 1.0, None]})
    result = assert_file_parity(tmp_path, baseline, candidate, "id", rtol=0, atol=atol)
    assert result.execution == "native_stream" and result.values_changed == changed


@pytest.mark.parametrize("atol", [0, 1e-12])
def test_float_nonfinite_overflow_subnormal_and_null_edges_preserve_evidence(tmp_path, atol):
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    smallest = float.fromhex("0x0.0000000000001p-1022")
    before = [math.nan, math.inf, -math.inf, -0.0, 1.0, largest, smallest, None, None]
    after = [
        math.nan,
        -math.inf,
        math.inf,
        0.0,
        math.nextafter(1.0, 2.0),
        -largest,
        0.0,
        1.0,
        math.nan,
    ]
    baseline = pl.DataFrame({"id": range(len(before)), "value": before})
    candidate = pl.DataFrame({"id": range(len(after)), "value": after})
    result = assert_file_parity(tmp_path, baseline, candidate, "id", rtol=0, atol=atol)
    assert result.execution == "native_stream" and result.nonfinite_changes == 3
    assert result.values_changed == (7 if atol == 0 else 5)
    assert math.isinf(result.max_abs_diff)


def test_float32_precision_and_column_tolerance_survive_native_reading(tmp_path):
    baseline = pl.DataFrame({"id": [1], "exact": pl.Series([1.0], dtype=pl.Float32)})
    baseline = baseline.with_columns(pl.col("exact").alias("tolerated"))
    candidate = baseline.with_columns(pl.all().exclude("id") + 2**-23)
    result = assert_file_parity(
        tmp_path,
        baseline,
        candidate,
        "id",
        rtol=0,
        atol=0,
        column_tolerances={"tolerated": (0, 1e-6)},
    )
    assert result.execution == "native_stream"
    assert result.values_changed == 1 and result.differences[0].column == "exact"


@pytest.mark.parametrize("empty_side", ["baseline", "candidate", "both"])
def test_empty_native_tables_keep_original_row_and_schema_counts(tmp_path, empty_side):
    populated = pl.DataFrame({"id": [3, 1], "value": [30, 10]})
    empty = populated.head(0)
    baseline = empty if empty_side in ("baseline", "both") else populated
    candidate = empty if empty_side in ("candidate", "both") else populated
    result = assert_file_parity(tmp_path, baseline, candidate, "id")
    assert result.execution == "native_stream"
    assert result.baseline_rows == baseline.height and result.candidate_rows == candidate.height


@pytest.mark.parametrize("typed", ["nested", "int128", "nanosecond"])
def test_typed_files_fall_back_without_losing_exact_value_semantics(tmp_path, typed):
    pairs = {
        "nested": (pl.List(pl.Int64), [[1, 2], [3]], [[1, 3], [3]]),
        "int128": (pl.Int128, [2**100, 2**100 + 2], [2**100 + 1, 2**100 + 2]),
        "nanosecond": (pl.Datetime("ns"), [1001, 2000], [1002, 2000]),
    }
    dtype, before, after = pairs[typed]
    baseline = pl.DataFrame({"id": [1, 2], "value": pl.Series(before, dtype=dtype)})
    candidate = pl.DataFrame({"id": [1, 2], "value": pl.Series(after, dtype=dtype)})
    result = assert_file_parity(tmp_path, baseline, candidate, "id", rtol=0, atol=0)
    assert result.execution == ("native_stream" if typed == "nested" else "typed_stream")
    assert result.values_changed == 1


def test_case_distinct_columns_remain_separate_in_typed_fallback(tmp_path):
    baseline = pl.DataFrame({"id": [1, 2], "Value": [10, 20], "value": [30, 40]})
    candidate = baseline.reverse().with_columns((pl.col("Value") + 1).alias("Value"))
    result = assert_file_parity(tmp_path, baseline, candidate, "id", rtol=0, atol=0)
    assert result.execution == "typed_stream" and result.fallback_reason
    assert result.values_changed == 2
    assert [item.column for item in result.differences] == ["Value", "Value"]
