"""Native equal-payload elision keeps exact differences and bounded evidence intact."""

from __future__ import annotations

import polars as pl
import pyarrow.parquet as pq
import pytest

from qavf import ResourceLimits
from qavf.audit_codec import evidence_json_size
from qavf.engine.sql_read import SqlReadSession
from qavf.regression import compare_frames, compare_outputs


def write_pair(tmp_path, baseline, candidate):
    left, right = tmp_path / "baseline.parquet", tmp_path / "candidate.parquet"
    baseline.write_parquet(left)
    candidate.write_parquet(right)
    return left, right


def semantic_result(result):
    return result.model_dump(exclude={"execution", "fallback_reason", "name"})


def observe_decoded_pairs(monkeypatch, *, nocase=False):
    batches = []
    original_batches = SqlReadSession.batches

    def capture(self, query, batch_rows=65536):
        for batch in original_batches(self, query, batch_rows=batch_rows):
            batches.append(batch)
            yield batch

    monkeypatch.setattr(SqlReadSession, "batches", capture)
    if nocase:
        original_init = SqlReadSession.__init__

        def initialize(self, ctx, connection, sources):
            connection.execute("SET default_collation='nocase'")
            original_init(self, ctx, connection, sources)

        monkeypatch.setattr(SqlReadSession, "__init__", initialize)
    return batches


def payload_pair():
    """Distinct keys and explicit expected wire payloads, including unmatched rows."""
    large_text = "L" * 199_999
    large_blob = b"Z" * 199_999
    records = [
        ("A", "Keepé\x00\\x41", "Keepé\x00\\x41", b"\x00\xff\\x41", b"\x00\xff\\x41"),
        ("a", "ALPHA", "alpha", b"A", b"a"),
        (r"\x41", r"\x41", r"\x41", b"\\x41", b"\\x41"),
        ("nul\x00key", "nul\x00text", "nul\x00text", b"\x00", b"\x00"),
        ("é", "é", "e\u0301", b"\xc3\xa9", b"e\xcc\x81"),
        ("e\u0301", None, None, None, None),
        ("🙂", "☺", "☺", "🙂".encode(), "🙂".encode()),
        ("null-left", None, "new", None, b"\x00new"),
        ("null-right", "old", None, b"\x00old", None),
        ("large", large_text + "x", large_text + "y", large_blob + b"x", large_blob + b"y"),
        (None, "null-key-payload", "null-key-payload", b"null-key-payload", b"null-key-payload"),
        ("empty", "", "", b"", b""),
    ]
    baseline = pl.DataFrame(
        {
            "id": [row[0] for row in records] + ["removed"],
            "text": [row[1] for row in records] + ["removed\x00é"],
            "blob": [row[3] for row in records] + [b"\xffremoved"],
        }
    )
    candidate = pl.DataFrame(
        {
            "id": ["added"] + [row[0] for row in reversed(records)],
            "text": ["added\x00é"] + [row[2] for row in reversed(records)],
            "blob": [b"\x80added"] + [row[4] for row in reversed(records)],
        }
    )
    expected = {
        "A": ("", "", b"", b""),
        "a": ("ALPHA", "alpha", b"A", b"a"),
        r"\x41": ("", "", b"", b""),
        "nul\x00key": ("", "", b"", b""),
        "é": ("é", "e\u0301", b"\xc3\xa9", b"e\xcc\x81"),
        "e\u0301": (None, None, None, None),
        "🙂": ("", "", b"", b""),
        "null-left": (None, "new", None, b"\x00new"),
        "null-right": ("old", None, b"\x00old", None),
        "large": (large_text + "x", large_text + "y", large_blob + b"x", large_blob + b"y"),
        None: ("", "", b"", b""),
        "empty": ("", "", b"", b""),
        "removed": ("removed\x00é", None, b"\xffremoved", None),
        "added": (None, "added\x00é", None, b"\x80added"),
    }
    return baseline, candidate, expected


@pytest.mark.parametrize("nocase", [False, True])
def test_decoded_equal_pairs_are_empty_but_original_differences_and_keys_survive(
    tmp_path, monkeypatch, nocase
):
    baseline, candidate, expected_pairs = payload_pair()
    expected = compare_frames(baseline, candidate, "id", rtol=0, atol=0)
    paths = write_pair(tmp_path, baseline, candidate)
    batches = observe_decoded_pairs(monkeypatch, nocase=nocase)
    actual = compare_outputs(
        *paths, "id", rtol=0, atol=0, resources=ResourceLimits(native_threads=1)
    )

    assert actual.execution == "native_stream" and actual.fallback_reason is None
    assert semantic_result(actual) == semantic_result(expected)
    assert (actual.rows_added, actual.rows_removed, actual.rows_changed, actual.values_changed) == (
        1,
        1,
        5,
        10,
    )
    decoded = pl.concat(batches)
    assert decoded.schema["b1"] == decoded.schema["c1"] == pl.String
    assert decoded.schema["b2"] == decoded.schema["c2"] == pl.Binary
    actual_pairs = {}
    for row in decoded.iter_rows(named=True):
        key = row["b0"] if row["br"] is not None else row["c0"]
        actual_pairs[key] = (row["b1"], row["c1"], row["b2"], row["c2"])
        if key == "removed":
            assert row["cr"] is None and row["c0"] is None
        elif key == "added":
            assert row["br"] is None and row["b0"] is None
        else:
            assert row["br"] is not None and row["cr"] is not None
            assert row["b0"] == row["c0"] == key
    assert actual_pairs == expected_pairs
    assert decoded.height == len(expected_pairs)
    previews = [item for item in actual.differences if item.key == {"id": "large"}]
    assert len(previews) == 2
    for preview in previews:
        assert preview.baseline_previewed and preview.candidate_previewed
        assert preview.baseline_length == preview.candidate_length == 200_000
    assert previews[0].baseline == previews[0].candidate == "L" * 256
    assert previews[1].baseline == previews[1].candidate == "<binary preview omitted>"


@pytest.mark.parametrize(
    "row_cap,byte_cap", [(0, 8192), (1, 8192), (4, 8192), (100, 0), (100, 1), (100, 700)]
)
def test_equal_payload_elision_preserves_counts_and_evidence_under_caps(
    tmp_path, monkeypatch, row_cap, byte_cap
):
    baseline, candidate, _ = payload_pair()
    options = {"max_differences": row_cap, "max_evidence_bytes": byte_cap}
    expected = compare_frames(baseline, candidate, "id", **options)
    paths = write_pair(tmp_path, baseline, candidate)
    batches = observe_decoded_pairs(monkeypatch, nocase=True)
    actual = compare_outputs(*paths, "id", resources=ResourceLimits(native_threads=1), **options)

    assert semantic_result(actual) == semantic_result(expected)
    assert actual.execution == "native_stream" and actual.truncated
    assert actual.values_changed == 10 and actual.rows_changed == 5
    assert len(actual.differences) <= row_cap
    if byte_cap >= 2:
        assert evidence_json_size([item.model_dump() for item in actual.differences]) <= byte_cap
    if row_cap == 0 or byte_cap < 2:
        assert actual.differences == []
    equal = pl.concat(batches).filter(pl.col("b0") == "A").row(0, named=True)
    assert (equal["b1"], equal["c1"], equal["b2"], equal["c2"]) == ("", "", b"", b"")


def test_elision_keeps_logical_evidence_blocks_across_a_real_16k_boundary(tmp_path, monkeypatch):
    rows = 16_389
    changed = [0, 16_383, 16_384, 16_388]
    keys = [f"row-{index}" for index in range(rows)]
    baseline = pl.DataFrame(
        {
            "id": ["removed", *keys],
            "text": ["removed-original", *(["é\x00" + "x" * 60] * rows)],
            "blob": [b"removed-original", *([b"\x00\xff" + b"x" * 60] * rows)],
        }
    )
    candidate = pl.DataFrame(
        {
            "id": [*keys, "added"],
            "text": ["changed" if index in changed else "é\x00" + "x" * 60 for index in range(rows)]
            + ["added-original"],
            "blob": [
                b"changed" if index in changed else b"\x00\xff" + b"x" * 60 for index in range(rows)
            ]
            + [b"added-original"],
        }
    ).reverse()
    expected = compare_frames(baseline, candidate, "id")
    paths = write_pair(tmp_path, baseline, candidate)
    batches = observe_decoded_pairs(monkeypatch, nocase=True)
    actual = compare_outputs(*paths, "id", resources=ResourceLimits(native_threads=1))

    assert actual.execution == "native_stream"
    assert semantic_result(actual) == semantic_result(expected)
    assert actual.rows_changed == 4 and actual.values_changed == 8
    assert [batch.height for batch in batches] == [16_384, rows + 2 - 16_384]
    decoded = pl.concat(batches)
    equal_rows = decoded.filter(
        pl.col("br").is_not_null()
        & pl.col("cr").is_not_null()
        & ~pl.col("b0").is_in([keys[index] for index in changed])
    )
    assert equal_rows.height == rows - len(changed)
    assert equal_rows.select(
        (pl.col("b1") == "").all(),
        (pl.col("c1") == "").all(),
        (pl.col("b2") == b"").all(),
        (pl.col("c2") == b"").all(),
    ).row(0) == (True, True, True, True)
    assert [(item.key["id"], item.column) for item in actual.differences] == [
        ("removed", None),
        ("added", None),
        ("row-0", "text"),
        ("row-16383", "text"),
        ("row-0", "blob"),
        ("row-16383", "blob"),
        ("row-16384", "text"),
        ("row-16388", "text"),
        ("row-16384", "blob"),
        ("row-16388", "blob"),
    ]


def test_different_original_string_binary_dtypes_are_never_elided(tmp_path, monkeypatch):
    baseline = pl.DataFrame({"id": ["A", "a"], "text": ["same", None], "blob": [b"same", None]})
    candidate = pl.DataFrame({"id": ["a", "A"], "text": [None, b"same"], "blob": [None, "same"]})
    expected = compare_frames(baseline, candidate, "id", schema="ignore")
    paths = write_pair(tmp_path, baseline, candidate)
    batches = observe_decoded_pairs(monkeypatch, nocase=True)
    actual = compare_outputs(
        *paths, "id", schema="ignore", resources=ResourceLimits(native_threads=1)
    )
    assert actual.execution == "native_stream"
    assert semantic_result(actual) == semantic_result(expected)
    assert actual.rows_changed == 1 and actual.values_changed == 2
    row = pl.concat(batches).filter(pl.col("b0") == "A").row(0, named=True)
    assert (row["b1"], row["c1"], row["b2"], row["c2"]) == ("same", b"same", b"same", "same")


def test_equal_nonpayload_numeric_and_boolean_cells_keep_their_original_values(
    tmp_path, monkeypatch
):
    baseline = pl.DataFrame({"id": ["A", "a"], "text": [7, 11], "blob": [True, False]})
    candidate = baseline.reverse()
    paths = write_pair(tmp_path, baseline, candidate)
    batches = observe_decoded_pairs(monkeypatch, nocase=True)
    actual = compare_outputs(*paths, "id", resources=ResourceLimits(native_threads=1))
    assert actual.execution == "native_stream" and actual.equal
    rows = pl.concat(batches).select("b0", "b1", "c1", "b2", "c2").rows()
    assert rows == [("A", 7, 7, True, True), ("a", 11, 11, False, False)]


def test_dictionary_encoded_parquet_string_keys_remain_exact_under_nocase(tmp_path, monkeypatch):
    keys = ["A", "a", r"\x41", "é", "e\u0301", "a\x00b", None]
    baseline = pl.DataFrame({"id": keys, "text": ["payload"] * len(keys)})
    candidate = baseline.reverse().with_columns(
        pl.when(pl.col("id") == "é")
        .then(pl.lit("changed"))
        .otherwise(pl.col("text"))
        .alias("text")
    )
    expected = compare_frames(baseline, candidate, "id")
    paths = [tmp_path / "baseline.parquet", tmp_path / "candidate.parquet"]
    for frame, path in zip((baseline, candidate), paths, strict=True):
        pq.write_table(frame.to_arrow(), path, use_dictionary=["id"])
        encodings = pq.read_metadata(path).row_group(0).column(0).encodings
        assert {"RLE_DICTIONARY", "PLAIN_DICTIONARY"}.intersection(encodings)
    batches = observe_decoded_pairs(monkeypatch, nocase=True)
    actual = compare_outputs(*paths, "id", resources=ResourceLimits(native_threads=1))

    assert actual.execution == "native_stream" and actual.fallback_reason is None
    assert semantic_result(actual) == semantic_result(expected)
    assert actual.baseline_rows == actual.candidate_rows == 7
    assert actual.rows_added == actual.rows_removed == 0
    assert actual.rows_changed == actual.values_changed == 1
    [difference] = actual.differences
    assert difference.key == {"id": "é"} and difference.column == "text"
    assert (difference.baseline, difference.candidate) == ("payload", "changed")
    decoded = pl.concat(batches)
    assert decoded["b0"].to_list() == decoded["c0"].to_list() == keys
    assert decoded.filter(pl.col("b1") == "").height == len(keys) - 1
