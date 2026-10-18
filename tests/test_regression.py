"""Baseline comparison (DeepDiff) and the REG-001 rule."""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from qavf.demo import inject_defects, make_demo_data
from qavf.engine.severity import Status
from qavf.regression import assert_outputs_match, compare_frames, compare_outputs, save_baseline
from qavf.testing import run_rule

BASELINES = Path(__file__).parent / "baselines"
KEYS = ["county", "year"]
BASE = pl.DataFrame(
    {
        "county": [1, 1, 2, 2],
        "year": [2020, 2025, 2020, 2025],
        "nox": [100.0, 90.0, 200.0, 180.0],
        "label": ["a", "b", "c", "d"],
    }
)


def test_identical_and_reordered_tables_match() -> None:
    assert compare_frames(BASE, BASE, KEYS).equal
    shuffled = BASE.sort("nox").select("label", "year", "nox", "county")
    result = compare_frames(BASE, shuffled, KEYS)
    assert result.equal and "matches baseline" in result.summary()


def test_numeric_tolerance_is_relative_and_absolute() -> None:
    assert compare_frames(BASE, BASE.with_columns(pl.col("nox") * (1 + 1e-9)), KEYS).equal
    drifted = BASE.with_columns(pl.col("nox") * 1.001)
    assert not compare_frames(BASE, drifted, KEYS).equal
    assert compare_frames(BASE, drifted, KEYS, rtol=0.01).equal
    near_zero = pl.DataFrame({"k": [1], "v": [0.0]}), pl.DataFrame({"k": [1], "v": [1e-12]})
    assert compare_frames(*near_zero, ["k"]).equal
    assert not compare_frames(*near_zero, ["k"], atol=0).equal


def test_value_changes_are_located_by_key_and_column() -> None:
    changed = BASE.with_columns(
        pl.when((pl.col("county") == 2) & (pl.col("year") == 2025))
        .then(pl.col("nox") * 1.5)
        .otherwise(pl.col("nox"))
        .alias("nox"),
        pl.when(pl.col("label") == "a").then(pl.lit("z")).otherwise(pl.col("label")).alias("label"),
    )
    result = compare_frames(BASE, changed, KEYS)
    assert (result.values_changed, result.rows_changed) == (2, 2)
    by_column = {d.column: d for d in result.differences}
    assert by_column["nox"].key == {"county": 2, "year": 2025}
    assert (by_column["nox"].baseline, by_column["nox"].candidate) == (180.0, 270.0)
    assert by_column["nox"].rel_diff == pytest.approx(0.5)
    assert (by_column["label"].baseline, by_column["label"].candidate) == ("a", "z")


def test_added_removed_rows_and_columns() -> None:
    candidate = (
        pl.concat(
            [
                BASE.filter(pl.col("county") == 1),
                pl.DataFrame({"county": [3], "year": [2020], "nox": [5.0], "label": ["e"]}),
            ]
        )
        .with_columns(pl.lit(1).alias("extra"))
        .drop("label")
    )
    result = compare_frames(BASE, candidate, KEYS)
    assert (result.rows_added, result.rows_removed, result.values_changed) == (1, 2, 0)
    assert result.columns_added == ["extra"] and result.columns_removed == ["label"]
    ignored = compare_frames(
        BASE, BASE.with_columns(pl.lit("now").alias("ts")), KEYS, ignore_columns=["ts"]
    )
    assert ignored.equal


def test_mostly_disjoint_tables_still_report_row_level_differences() -> None:
    # Guards DeepDiff's ``threshold_to_diff_deeper``, which would otherwise collapse
    # everything into one opaque root-level change when few keys are shared.
    baseline = pl.DataFrame({"k": list(range(100)), "v": [1.0] * 100})
    candidate = pl.DataFrame({"k": [0, 1], "v": [2.0, 1.0]})
    result = compare_frames(baseline, candidate, ["k"])
    assert (result.rows_removed, result.values_changed, result.rows_added) == (98, 1, 0)


def test_dtype_null_and_key_handling() -> None:
    assert compare_frames(
        pl.DataFrame({"k": [1], "v": [1]}), pl.DataFrame({"k": [1], "v": [1.0]}), ["k"]
    ).equal
    with_null = pl.DataFrame({"k": [1, 2], "v": [1.0, None]})
    assert compare_frames(with_null, with_null, ["k"]).equal
    with pytest.raises(ValueError, match="lacks key column"):
        compare_frames(BASE, BASE, ["nope"])
    with pytest.raises(ValueError, match="duplicate key"):
        compare_frames(BASE, BASE, ["county"])
    with pytest.raises(ValueError, match="at least one key"):
        compare_frames(BASE, BASE, [])
    many = pl.DataFrame({"k": list(range(50)), "v": [1.0] * 50})
    capped = compare_frames(many, many.with_columns(pl.col("v") + 1), ["k"], max_differences=5)
    assert capped.values_changed == 50 and len(capped.differences) == 5 and capped.truncated


def test_file_round_trip_and_assertion_helper(tmp_path) -> None:
    baseline = save_baseline(BASE.sort("nox", descending=True), tmp_path / "base" / "out.csv", KEYS)
    assert pl.read_csv(baseline)["county"].to_list() == [1, 1, 2, 2]
    candidate = tmp_path / "out.parquet"
    BASE.write_parquet(candidate)
    assert compare_outputs(baseline, candidate, KEYS).name == "out.parquet"
    assert_outputs_match(baseline, candidate, KEYS)
    BASE.with_columns(pl.col("nox") * 2).write_parquet(candidate)
    with pytest.raises(AssertionError) as excinfo:
        assert_outputs_match(baseline, candidate, KEYS, show=2)
    assert "4 value(s) changed" in str(excinfo.value) and "and 2 more" in str(excinfo.value)


# ------------------------------------------------------------------ REG-001
def _params(**baseline) -> dict:
    return {"baselines": [{"dataset": "counts", "keys": ["site_id", "year"], **baseline}]}


def test_rule_compares_dataset_with_baseline_file(demo, tmp_path) -> None:
    path = save_baseline(demo["counts"], tmp_path / "counts.parquet")
    assert run_rule("REG-001", demo, _params(path=str(path))).status is Status.PASS
    result = run_rule(
        "REG-001", inject_defects(demo, ["aadt_spike", "missing_count"]), _params(path=str(path))
    )
    assert result.status is Status.FAIL and result.affected_count == 2
    assert {r["kind"] for r in result.affected_records} == {"value_changed", "row_removed"}
    tolerant = run_rule(
        "REG-001", inject_defects(demo, ["aadt_spike"]), {**_params(path=str(path)), "rtol": 5}
    )
    assert tolerant.status is Status.PASS


def test_baseline_can_live_in_a_database(demo, tmp_path) -> None:
    with duckdb.connect(str(tmp_path / "b.duckdb")) as con:
        con.register("c", demo["counts"])
        con.execute("CREATE TABLE counts AS SELECT * FROM c")
    params = _params(source={"type": "database", "connection": "accepted", "table": "counts"})
    connections = {"accepted": {"type": "duckdb", "path": str(tmp_path / "b.duckdb")}}
    assert run_rule("REG-001", demo, params, connections=connections).status is Status.PASS


def test_rule_fails_when_comparison_cannot_run(demo, tmp_path) -> None:
    assert run_rule("REG-001", demo).status is Status.SKIPPED
    missing = run_rule("REG-001", demo, _params(path=str(tmp_path / "never.csv")))
    # An unavailable comparison is unassessed, independent of severity.
    assert missing.status is Status.ERROR and "could not run" in missing.message
    path = save_baseline(demo["counts"], tmp_path / "counts.csv")
    duplicated = run_rule(
        "REG-001", inject_defects(demo, ["duplicate_count"]), _params(path=str(path))
    )
    assert duplicated.status is Status.ERROR and "duplicate key" in duplicated.message
    with pytest.raises(ValueError):
        run_rule("REG-001", demo, _params())  # neither path nor source


@pytest.mark.regression
def test_demo_counts_match_stored_baseline() -> None:
    """The pattern to copy: compare generated output with a baseline under version control."""
    summary = make_demo_data(seed=0)["counts"].group_by("year").agg(pl.col("aadt").sum())
    assert_outputs_match(
        BASELINES / "demo_counts_by_year.csv", summary, keys=["year"], name="counts by year"
    )
