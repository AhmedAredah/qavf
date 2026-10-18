"""Normalized baseline inputs, bounded round trips and failed-publication cleanup."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from qavf import Dataset, File, Loader, Query, ResourceLimits
from qavf.cli import EXIT_OK, main
from qavf.engine.context import DataContext
from qavf.engine.resources import ResourceLimitError
from qavf.io import read_table
from qavf.regression import accept_baseline, read_manifest, verify_baseline
from qavf.regression.baselines import manifest_path

FRAME = pl.DataFrame({"key": [3, 1, 2], "value": [30, 10, 20], "label": ["c", "a", "b"]})


@pytest.mark.parametrize(
    "suffix", ["csv", "tsv", "parquet", "json", "ndjson", "jsonl.gz", "csv.gz", "json.gz"]
)
def test_streamed_formats_remain_sorted_and_verified(suffix, tmp_path):
    target = tmp_path / f"baseline.{suffix}"
    accepted = accept_baseline(FRAME.lazy(), target, "key")
    assert_frame_equal(read_table(target), FRAME.sort("key"), check_dtypes=False)
    assert accepted.rows == 3
    assert read_manifest(target) == accepted
    assert verify_baseline(target, ["key"]) == []


@pytest.mark.parametrize("kind", ["path", "file", "query", "loader", "dataset"])
def test_baseline_acceptance_normalizes_explicit_inputs(kind, tmp_path):
    source = tmp_path / "source.parquet"
    FRAME.write_parquet(source)
    calls = []

    def load():
        calls.append(1)
        return FRAME.lazy()

    inputs = {
        "path": source,
        "file": File(source),
        "query": Query("SELECT * FROM read_parquet($path)", parameters={"path": str(source)}),
        "loader": Loader(load),
        "dataset": Dataset(File(source), rename={"label": "description"}),
    }
    target = tmp_path / "baseline.parquet"
    accepted = accept_baseline(inputs[kind], target, ["key"])
    expected = FRAME.rename({"label": "description"}) if kind == "dataset" else FRAME
    assert_frame_equal(pl.read_parquet(target), expected.sort("key"))
    assert accepted.rows == 3
    assert calls == ([1] if kind == "loader" else [])


def test_large_native_baseline_never_uses_eager_source_reader(tmp_path, monkeypatch):
    frame = pl.DataFrame({"key": range(80_000, 0, -1), "value": ["value"] * 80_000})
    source = tmp_path / "source.parquet"
    frame.write_parquet(source)
    target = tmp_path / "baseline.parquet"
    monkeypatch.setattr(DataContext, "get", lambda *a, **k: pytest.fail("full source load"))
    monkeypatch.setattr("qavf.io.read_table", lambda *a, **k: pytest.fail("eager reader"))
    accepted = accept_baseline(source, target, "key")
    assert accepted.rows == 80_000
    assert pl.scan_parquet(target).select((pl.col("key").diff() > 0).all()).collect().item()


def test_typed_nested_nanosecond_values_round_trip_exactly(tmp_path):
    frame = pl.DataFrame(
        {
            "key": [2, 1],
            "when": pl.Series([1, 1_001], dtype=pl.Int64).cast(pl.Datetime("ns")),
            "nested": [[{"value": "x"}], [{"value": "y"}, {"value": None}]],
        }
    )
    target = tmp_path / "baseline.parquet"
    accept_baseline(frame.lazy(), target, "key")
    assert_frame_equal(pl.read_parquet(target), frame.sort("key"))


def test_spill_failure_preserves_previous_pair_and_cleans_temporary_files(tmp_path):
    target = tmp_path / "baseline.parquet"
    accept_baseline(FRAME, target, "key")
    before = (target.read_bytes(), manifest_path(target).read_bytes())
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ResourceLimitError, match="spill budget"):
        accept_baseline(
            FRAME.with_columns(pl.col("value") + 1),
            target,
            "key",
            replace=True,
            resources=ResourceLimits(max_spill=1, temp_dir=scratch),
        )
    assert (target.read_bytes(), manifest_path(target).read_bytes()) == before
    assert verify_baseline(target) == []
    assert not list(scratch.iterdir())
    assert not target.with_name(f".{target.name}.transaction").exists()


def test_baseline_rejects_unsupported_isolation_before_loading(tmp_path):
    target = tmp_path / "baseline.parquet"
    with pytest.raises(ValueError, match="isolated baseline acceptance"):
        accept_baseline("missing.parquet", target, "key", resources=ResourceLimits(isolate=True))
    assert not target.exists()
    assert not manifest_path(target).exists()
    assert not target.with_name(f".{target.name}.transaction").exists()


def test_baseline_keeps_dataset_loading_safeguards(tmp_path):
    source = tmp_path / "source.parquet"
    FRAME.write_parquet(source)
    with pytest.raises(Exception, match=r"row budget|row_budget"):
        accept_baseline(
            Dataset(source, load_policy={"row_budget": 2}),
            tmp_path / "baseline.parquet",
            "key",
        )


def test_baseline_cli_uses_source_descriptor_without_eager_reader(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    target = tmp_path / "baseline.parquet"
    FRAME.write_parquet(source)
    monkeypatch.setattr("qavf.io.read_table", lambda *a, **k: pytest.fail("eager CLI reader"))
    assert main(["baseline", str(source), str(target), "-k", "key"]) == EXIT_OK
    assert verify_baseline(target) == []
