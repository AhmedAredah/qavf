"""Column mapping in the keyed regression comparison and every entry point to it.

A renamed column must compare as the same column, on every execution path
(eager frames, native file streams, typed indexed snapshots, ad-hoc REG-001
baselines and the CLI), and a mapped column the baseline lacks must never pass.
"""

from __future__ import annotations

import duckdb
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qavf import File, Query, Status
from qavf.cli import EXIT_OK, EXIT_QA_FAILED, EXIT_USAGE, main
from qavf.regression import assert_outputs_match, compare_frames, compare_outputs
from qavf.testing import rule_results

BASELINE = pl.DataFrame({"site": ["a", "b", "c"], "aadt": [10, 20, 30], "share": [0.1, 0.2, 0.3]})
CANDIDATE = pl.DataFrame(
    {"site_id": ["a", "b", "c"], "volume": [10, 21, 30], "share": [0.1, 0.2, 0.3]}
)
MAP = {"site": "site_id", "aadt": "volume"}


def test_mapped_frames_compare_under_candidate_names():
    result = compare_frames(BASELINE, CANDIDATE, ["site_id"], rtol=0, atol=0, column_map=MAP)
    assert result.values_changed == 1 and result.rows_changed == 1
    assert result.column_changes == {"volume": 1}
    assert not result.columns_added and not result.columns_removed
    assert result.column_map == MAP
    (difference,) = result.differences
    assert difference.key == {"site_id": "b"} and difference.column == "volume"
    assert (difference.baseline, difference.candidate) == (20, 21)


def test_without_a_map_the_rename_is_a_schema_change():
    result = compare_frames(BASELINE.rename({"site": "site_id"}), CANDIDATE, ["site_id"])
    assert result.columns_removed == ["aadt"] and result.columns_added == ["volume"]


@pytest.mark.parametrize(
    "column_map,message",
    [
        ({"missing": "volume"}, "baseline lacks mapped column"),
        ({"site": "site_id", "share": "site_id"}, "several baseline columns"),
        ({"aadt": "share"}, "several baseline columns"),
    ],
)
def test_invalid_maps_never_compare_nothing(column_map, message):
    with pytest.raises(ValueError, match=message):
        compare_frames(BASELINE, CANDIDATE, ["site_id"], column_map=column_map)


def test_tolerances_and_ignored_columns_use_candidate_names():
    result = compare_frames(
        BASELINE,
        CANDIDATE,
        ["site_id"],
        rtol=0,
        atol=0,
        column_map=MAP,
        column_tolerances={"volume": (0.1, 0)},
    )
    assert result.equal
    assert compare_frames(
        BASELINE, CANDIDATE, ["site_id"], column_map=MAP, ignore_columns=["volume"]
    ).equal


@pytest.mark.parametrize("kind", ["parquet", "csv", "query"])
def test_file_and_query_paths_apply_the_map(kind, tmp_path):
    if kind == "query":
        BASELINE.write_parquet(tmp_path / "b.parquet")
        location = (tmp_path / "b.parquet").as_posix()
        baseline = Query(f"SELECT * FROM read_parquet('{location}')")
    else:
        baseline = tmp_path / f"b.{kind}"
        getattr(BASELINE, f"write_{kind}")(baseline)
    candidate = tmp_path / "c.parquet"
    CANDIDATE.write_parquet(candidate)
    result = compare_outputs(baseline, candidate, ["site_id"], rtol=0, atol=0, column_map=MAP)
    assert result.column_changes == {"volume": 1}, result.execution
    assert result.execution == ("typed_stream" if kind == "query" else "native_stream")
    assert not result.columns_added and not result.columns_removed
    with pytest.raises(ValueError, match="baseline lacks mapped column"):
        compare_outputs(baseline, candidate, ["site_id"], column_map={"nope": "volume"})


def test_assert_outputs_match_accepts_a_map():
    assert_outputs_match(
        BASELINE,
        CANDIDATE.with_columns(volume=pl.Series([10, 20, 30])),
        ["site_id"],
        column_map=MAP,
    )
    with pytest.raises(AssertionError, match="1 value"):
        assert_outputs_match(BASELINE, CANDIDATE, ["site_id"], rtol=0, atol=0, column_map=MAP)


frames = st.integers(1, 6).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(-5, 5) | st.none(), min_size=n, max_size=n),
        st.lists(st.integers(-5, 5) | st.none(), min_size=n, max_size=n),
        st.lists(st.sampled_from(["x", "y", None]), min_size=n, max_size=n),
        st.lists(st.sampled_from(["x", "y", None]), min_size=n, max_size=n),
    )
)


@given(frames)
@settings(max_examples=60, deadline=None)
def test_mapping_then_comparing_equals_comparing_renamed_frames(values):
    """Property: a map is exactly a rename of the baseline before comparison."""
    a, b, c, d = values
    keys = list(range(len(a)))
    baseline = pl.DataFrame({"k": keys, "old_v": a, "old_s": c})
    candidate = pl.DataFrame({"key": keys, "v": b, "s": d})
    column_map = {"k": "key", "old_v": "v", "old_s": "s"}
    mapped = compare_frames(baseline, candidate, ["key"], rtol=0, atol=0, column_map=column_map)
    renamed = compare_frames(baseline.rename(column_map), candidate, ["key"], rtol=0, atol=0)
    assert mapped.model_dump(exclude={"column_map"}) == renamed.model_dump(exclude={"column_map"})
    expected = sum(x != y for x, y in zip(a, b, strict=True)) + sum(
        x != y for x, y in zip(c, d, strict=True)
    )
    assert mapped.values_changed == expected
    assert sum(mapped.column_changes.values()) == expected


def _baseline_params(path, **extra):
    return {
        "baselines": [
            {
                "dataset": "counts",
                "path": str(path),
                "keys": ["site_id"],
                "rtol": 0,
                "atol": 0,
                "rename": MAP,
                **extra,
            }
        ]
    }


def test_reg001_rename_maps_baseline_columns(tmp_path):
    path = tmp_path / "baseline.parquet"
    BASELINE.write_parquet(path)
    (result,) = rule_results("REG-001", {"counts": CANDIDATE}, _baseline_params(path))
    assert result.status is Status.FAIL and result.affected_count == 1
    assert result.affected_records[0]["column"] == "volume"
    same = CANDIDATE.with_columns(volume=pl.Series([10, 20, 30]))
    (result,) = rule_results("REG-001", {"counts": same}, _baseline_params(path))
    assert result.status is Status.PASS


def test_reg001_rename_of_absent_column_is_not_assessed(tmp_path):
    path = tmp_path / "baseline.parquet"
    BASELINE.write_parquet(path)
    params = _baseline_params(path)
    params["baselines"][0]["rename"] = {"nope": "volume", "site": "site_id"}
    (result,) = rule_results("REG-001", {"counts": CANDIDATE}, params)
    assert result.status is Status.ERROR and "baseline lacks mapped column" in result.message


def test_reg001_rename_rejects_duplicate_targets(tmp_path):
    params = _baseline_params(tmp_path / "b.parquet")
    params["baselines"][0]["rename"] = {"site": "x", "aadt": "x"}
    with pytest.raises(Exception, match="several columns"):
        rule_results("REG-001", {"counts": CANDIDATE}, params)


def test_reg001_rename_with_a_database_baseline(tmp_path):
    database = tmp_path / "b.duckdb"
    with duckdb.connect(str(database)) as con:
        con.register("b", BASELINE)
        con.execute("CREATE TABLE counts AS SELECT * FROM b")
    params = {
        "baselines": [
            {
                "dataset": "counts",
                "source": {"type": "database", "connection": "accepted", "table": "counts"},
                "keys": ["site_id"],
                "rtol": 0,
                "atol": 0,
                "rename": MAP,
            }
        ]
    }
    connections = {"accepted": {"type": "duckdb", "path": str(database)}}
    (result,) = rule_results("REG-001", {"counts": CANDIDATE}, params, connections=connections)
    assert result.status is Status.FAIL and result.affected_count == 1


def test_cli_compare_map(tmp_path, capsys):
    baseline, candidate = tmp_path / "b.parquet", tmp_path / "c.parquet"
    BASELINE.write_parquet(baseline)
    CANDIDATE.write_parquet(candidate)
    args = ["compare", str(baseline), str(candidate), "-k", "site_id", "--rtol", "0", "--atol", "0"]
    mapped = [*args, "--map", "site=site_id", "--map", "aadt=volume"]
    assert main(mapped) == EXIT_QA_FAILED
    assert "1 value(s) changed" in capsys.readouterr().out
    assert main([*mapped, "--rtol", "0.1"]) == EXIT_OK
    assert main([*args, "--map", "site"]) == EXIT_USAGE
    assert main([*args, "--map", "site=a", "--map", "site=b"]) == EXIT_USAGE


def test_file_descriptor_inputs_apply_the_map(tmp_path):
    baseline, candidate = tmp_path / "b.parquet", tmp_path / "c.parquet"
    BASELINE.write_parquet(baseline)
    CANDIDATE.write_parquet(candidate)
    result = compare_outputs(
        File(baseline), File(candidate), ["site_id"], column_map=MAP, rtol=0, atol=0
    )
    assert result.column_changes == {"volume": 1}
