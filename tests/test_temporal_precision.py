"""Temporal precision, formats and column-name collisions.

Nanosecond precision, Excel blank rows, configuration-relative SQL
paths, compressed formats and internal column-name collisions.
"""

from __future__ import annotations

import gzip

import duckdb
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.io import read_table
from qavf.regression import compare_frames
from qavf.sources import FileSource, SourceEnvironment, SourceRegistry, infer_format
from qavf.sources.parameters import FileParams
from qavf.stats import flag_outliers
from qavf.testing import make_settings, run_rule


# =========================================================================
# Nanosecond temporal values keep their precision
# =========================================================================
def _ns(values: list[int], dtype: pl.DataType) -> pl.Series:
    return pl.Series(values, dtype=pl.Int64).cast(dtype)


@pytest.mark.parametrize("dtype", [pl.Datetime("ns"), pl.Duration("ns"), pl.Time()])
def test_sub_microsecond_value_changes_are_detected(dtype) -> None:
    base = pl.DataFrame({"k": [1], "t": _ns([1000], dtype)})
    cand = pl.DataFrame({"k": [1], "t": _ns([1001], dtype)})
    result = compare_frames(base, cand, ["k"], rtol=0, atol=0)
    assert not result.equal and result.values_changed == 1
    assert compare_frames(base, base, ["k"]).equal


def test_nanosecond_keys_stay_distinct_and_exact() -> None:
    base = pl.DataFrame({"k": _ns([1000, 1001, 1002], pl.Datetime("ns")), "v": [1.0, 2.0, 3.0]})
    assert compare_frames(base, base, ["k"]).equal  # three keys within one microsecond
    shifted = base.with_columns(_ns([1000, 1001, 1003], pl.Datetime("ns")).alias("k"))
    result = compare_frames(base, shifted, ["k"])
    assert (result.rows_added, result.rows_removed) == (1, 1)


def test_negative_durations_and_mixed_time_units() -> None:
    base = pl.DataFrame({"k": [1], "d": _ns([-5], pl.Duration("ns"))})
    cand = pl.DataFrame({"k": [1], "d": _ns([-6], pl.Duration("ns"))})
    assert not compare_frames(base, cand, ["k"]).equal
    # The same instant stored with different time units is the same value ...
    micro = pl.DataFrame({"k": [1], "t": _ns([5], pl.Datetime("us"))})
    nano = pl.DataFrame({"k": [1], "t": _ns([5000], pl.Datetime("ns"))})
    assert compare_frames(micro, nano, ["k"]).equal
    # ... and the same key.
    assert compare_frames(micro.rename({"k": "v"}), nano.rename({"k": "v"}), ["t"]).equal


def test_nanosecond_evidence_is_readable() -> None:
    base = pl.DataFrame({"k": [1], "t": _ns([1000], pl.Datetime("ns"))})
    cand = pl.DataFrame({"k": [1], "t": _ns([1001], pl.Datetime("ns"))})
    (difference,) = compare_frames(base, cand, ["k"]).differences
    assert str(difference.baseline).endswith(".000001000")
    assert str(difference.candidate).endswith(".000001001")


# =========================================================================
# Excel blank rows do not hide later rows
# =========================================================================
def _write_xlsx(path, frame: pl.DataFrame) -> None:
    try:
        with duckdb.connect() as con:
            con.execute("INSTALL excel; LOAD excel")
            con.register("f", frame)
            con.execute(f"COPY f TO '{path.as_posix()}' (FORMAT xlsx, HEADER true)")
    except duckdb.Error as exc:  # pragma: no cover - needs DuckDB >= 1.2 with the excel extension
        pytest.skip(f"DuckDB cannot write Excel here: {exc}")


def test_excel_rows_after_a_blank_row_are_loaded(tmp_path, registry) -> None:
    frame = pl.DataFrame({"id": [1, None, 2], "value": [10.0, None, -999.0]})
    _write_xlsx(tmp_path / "book.xlsx", frame)
    loaded = read_table(tmp_path / "book.xlsx")
    assert loaded.height == 3 and loaded["value"].to_list()[-1] == -999.0

    datasets = {
        "book": {
            "path": str(tmp_path / "book.xlsx"),
            "schema": {"columns": {"id": {}, "value": {"ge": 0}}},
        }
    }
    settings = make_settings(datasets=datasets, include_categories=["input"])
    report = QARunner(settings, registry).run()
    failed = {r.check for r in report.results if r.status is Status.FAIL}
    assert {"book.value: ge 0", "book.id: not null"} <= failed
    assert report.exit_code() == 1


def test_excel_trailing_blank_rows_are_trimmed_and_options_respected(tmp_path) -> None:
    frame = pl.DataFrame({"id": [1, None, 2, None, None], "value": [1.0, None, 2.0, None, None]})
    _write_xlsx(tmp_path / "book.xlsx", frame)
    assert read_table(tmp_path / "book.xlsx").height == 3  # interior blank kept, trailing dropped
    # stop_at_empty can leave rows out, so it must be acknowledged.
    params = FileParams(path="book.xlsx", options={"stop_at_empty": True}, acknowledge_lossy=True)
    explicit = FileSource(params, SourceEnvironment(base_dir=tmp_path)).load()
    assert explicit.height == 1  # an explicit choice is honoured


# =========================================================================
# Relative SQL file paths resolve against the configuration directory
# =========================================================================
def _competing_files(tmp_path, name: str) -> tuple:
    config_dir, cwd = tmp_path / "config, with comma", tmp_path / "cwd"
    for directory, value in ((config_dir, 1), (cwd, 999)):
        directory.mkdir()
        pl.DataFrame({"x": [value]}).write_parquet(directory / f"{name}.parquet")
        pl.DataFrame({"x": [value]}).write_csv(directory / f"{name}.csv")
    return config_dir, cwd


def test_query_reads_the_configuration_relative_file_not_the_cwd_copy(
    tmp_path, monkeypatch
) -> None:
    config_dir, cwd = _competing_files(tmp_path, "data")
    monkeypatch.chdir(cwd)
    registry = SourceRegistry().discover(use_entry_points=False)
    env = SourceEnvironment(base_dir=config_dir)
    for sql in (
        "SELECT * FROM read_parquet('data.parquet')",
        "SELECT * FROM read_parquet(['data.parquet'])",
        "SELECT * FROM 'data.parquet'",
        "SELECT * FROM read_csv('data.csv', header = true)",
        "SELECT * FROM read_parquet('./data.parquet') -- 'data.csv' in a comment",
    ):
        loaded = registry.build({"type": "query", "sql": sql}, env).load()
        assert loaded["x"].to_list() == [1], sql
    # String values that are not file arguments are left alone.
    value = registry.build({"type": "query", "sql": "SELECT 'data.csv' AS name"}, env).load()
    assert value["name"].to_list() == ["data.csv"]


def test_assertions_resolve_relative_files_against_the_configuration(tmp_path, monkeypatch) -> None:
    config_dir, cwd = _competing_files(tmp_path, "ref")
    monkeypatch.chdir(cwd)
    ctx = DataContext(
        frames={"t": pl.DataFrame({"x": [1]})}, env=SourceEnvironment(base_dir=config_dir)
    )
    assert ctx.sql("SELECT * FROM t WHERE x NOT IN (SELECT x FROM 'ref.parquet')").is_empty()


# =========================================================================
# Compressed files are read in their real format
# =========================================================================
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.csv.gz", "csv"),
        ("a.tsv.gz", "csv"),
        ("a.json.gz", "json"),
        ("a.ndjson.gz", "ndjson"),
        ("a.jsonl.zst", "ndjson"),
        ("a.csv", "csv"),
    ],
)
def test_compression_suffixes_are_peeled_before_format_inference(name, expected) -> None:
    assert infer_format(name) == expected


@pytest.mark.parametrize("name", ["a.gz", "a.parquet.gz", "a.unknown.gz"])
def test_unknown_compressed_formats_are_rejected(name) -> None:
    with pytest.raises(ValueError, match="format"):
        infer_format(name)


def test_compressed_ndjson_loads_every_record(tmp_path) -> None:
    with gzip.open(tmp_path / "data.ndjson.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"id": 1, "value": 2}\n{"id": 2, "value": -99}\n')
    loaded = read_table(tmp_path / "data.ndjson.gz")
    assert loaded.columns == ["id", "value"] and loaded["value"].to_list() == [2, -99]


# =========================================================================
# Column names never collide with internal names
# =========================================================================
RESERVED = [
    "group_size",
    "group_median",
    "score",
    "lower_bound",
    "upper_bound",
    "is_outlier",
    "_centre",
    "_spread",
    "_q1",
    "_q3",
    "value",
    "metric",
    "denominator",
    "group",
]
SAMPLE = [10.1, 9.8, 10.3, 10.0, 9.9, 10.2, 10.4, 9.7, 10.0, 10.1, 9.9, 10.2, 25.0]


@pytest.mark.parametrize("method", ["zscore", "modified_zscore", "iqr"])
@pytest.mark.parametrize("name", RESERVED)
def test_outlier_scan_is_invariant_to_column_names(method, name) -> None:
    groups = ["a"] * len(SAMPLE)
    reference = flag_outliers(pl.DataFrame({"g": groups, "v": SAMPLE}), "v", ["g"], method)
    renamed = flag_outliers(pl.DataFrame({"g": groups, name: SAMPLE}), name, ["g"], method)
    assert renamed.flags.to_list() == reference.flags.to_list()
    assert renamed.frame[name].to_list() == SAMPLE  # the input column is untouched

    as_group = flag_outliers(pl.DataFrame({name: groups, "v": SAMPLE}), "v", [name], method)
    assert as_group.flags.to_list() == reference.flags.to_list()


def test_unrelated_user_columns_survive_and_outputs_never_overwrite_them() -> None:
    frame = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 100.0], "_q1": ["keep"] * 5, "score": [7] * 5})
    scan = flag_outliers(frame, "v", method="iqr", threshold=1.5)
    assert scan.frame["_q1"].to_list() == ["keep"] * 5
    assert scan.frame["score"].to_list() == [7] * 5  # the user's column, not the outlier score
    assert scan.columns.score != "score"
    assert scan.flags.to_list() == [False] * 4 + [True]


def test_group_comparison_respects_a_user_column_named_group(registry) -> None:
    frame = pl.DataFrame(
        {"entity": list("abcdef"), "group": ["x"] * 3 + ["y"] * 3, "v": [1.0] * 3 + [100.0] * 3}
    )
    metric = {"dataset": "t", "value": "v", "entity_column": "entity", "group_by": ["group"]}
    result = run_rule("SPA-002", {"t": frame}, {"metrics": [metric]}, registry=registry)
    assert result.status is Status.PASS, result.message


@pytest.mark.parametrize("name", ["value", "metric", "denominator", "group_median", "score"])
def test_spatial_metric_with_reserved_value_name(name, registry) -> None:
    values = [*([10.0, 10.5, 9.5, 10.2] * 3)[:11], 40.0]
    frame = pl.DataFrame({"e": list(range(12)), name: values})
    params = {"metrics": [{"dataset": "t", "value": name, "entity_column": "e"}]}
    result = run_rule("SPA-001", {"t": frame}, params, registry=registry)
    assert result.status is Status.FAIL
    assert [r["e"] for r in result.affected_records] == [11]


@pytest.mark.parametrize("name", ["value", "previous_value", "previous_period", "change_rate"])
def test_temporal_series_with_reserved_group_name(name, registry) -> None:
    frame = pl.DataFrame(
        {"year": [2020, 2021, 2020, 2021], name: ["a", "a", "b", "b"], "v": [1.0, 1.05, 1.0, 3.0]}
    )
    params = {"series": [{"dataset": "t", "value": "v", "group_by": [name], "time_column": "year"}]}
    result = run_rule("TMP-001", {"t": frame}, params, registry=registry)
    assert result.status is Status.FAIL
    assert [r[name] for r in result.affected_records] == ["b"]
