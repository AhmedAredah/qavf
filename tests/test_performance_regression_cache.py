"""Benchmark oracles reject wrong results and prove the intended real work occurred."""

from __future__ import annotations

import copy
import importlib

import polars as pl
import pytest

from scripts.performance import cache, regression
from scripts.performance.models import WorkloadOptions


def test_regression_fixture_exercises_keyed_scalar_and_nested_changes(tmp_path):
    fixture = tmp_path / "fixture"
    regression.prepare_fixture(fixture, rows=303, seed=0, chunk_rows=37)
    workload = regression.build_workload(
        fixture, tmp_path / "run", WorkloadOptions(backend="file", evidence_rows=100)
    )
    result = workload.run()
    verified = workload.verify(result)
    assert verified["rows_added"] == verified["rows_removed"] == 3
    assert verified["rows_changed"] == 15
    assert verified["values_changed"] == 18
    assert verified["evidence_rows"] == 24
    assert verified["truncated"] is False
    changes = result["differences"]
    assert {item["column"] for item in changes if item["kind"] == "value_changed"} == {
        "amount",
        "status",
        "readings",
        "details",
    }
    assert all(item["path"] == [3] for item in changes if item["column"] == "readings")
    assert all(item["path"] == ["score"] for item in changes if item["column"] == "details")


@pytest.fixture(scope="module")
def regression_findings(tmp_path_factory):
    directory = tmp_path_factory.mktemp("regression-findings")
    regression.prepare_fixture(directory / "fixture", rows=303, seed=0, chunk_rows=37)
    workload = regression.build_workload(
        directory / "fixture",
        directory / "run",
        WorkloadOptions(backend="file", evidence_rows=100),
    )
    result = workload.run()
    assert workload.verify(result)["truncated"] is False
    return workload, result


@pytest.mark.parametrize(
    "mutation",
    [
        "nonsense",
        "kind",
        "key_missing",
        "key_extra",
        "key_bool",
        "key_outside",
        "key_wrong_event",
        "added_key",
        "row_value",
        "column",
        "tolerated_cell",
        "path",
        "baseline",
        "candidate",
        "nested_baseline",
        "nested_candidate",
        "preview",
        "duplicate",
    ],
)
def test_regression_oracle_rejects_forged_evidence_with_correct_counts(
    regression_findings, mutation
):
    workload, original = regression_findings
    result = copy.deepcopy(original)
    differences = result["differences"]
    amount = next(item for item in differences if item["column"] == "amount")
    nested = next(item for item in differences if item["column"] == "readings")
    if mutation == "nonsense":
        differences[0] = {"nonsense": "not a comparison finding"}
    elif mutation == "kind":
        amount["kind"] = "column_removed"
    elif mutation == "key_missing":
        amount["key"] = {}
    elif mutation == "key_extra":
        amount["key"]["other"] = 0
    elif mutation == "key_bool":
        amount["key"]["id"] = True
    elif mutation == "key_outside":
        amount["key"]["id"] = 9999
    elif mutation == "key_wrong_event":
        amount["key"]["id"] = 2
    elif mutation == "added_key":
        next(item for item in differences if item["kind"] == "row_added")["key"]["id"] = 0
    elif mutation == "row_value":
        differences[0]["baseline"] = "invented removed-row contents"
    elif mutation == "column":
        amount["column"] = "metric_0"
    elif mutation == "tolerated_cell":
        amount.update(key={"id": 6}, column="measurement", baseline=1006.0, candidate=1006.0 + 1e-8)
    elif mutation == "path":
        nested["path"] = [2]
    elif mutation == "baseline":
        amount["baseline"] += 1
    elif mutation == "candidate":
        amount["candidate"] += 1
    elif mutation == "nested_baseline":
        nested["baseline"][0] += 1
    elif mutation == "nested_candidate":
        nested["candidate"][3] += 1
    elif mutation == "preview":
        nested["baseline_previewed"] = True
    elif mutation == "duplicate":
        differences[1] = copy.deepcopy(differences[0])
    with pytest.raises(AssertionError, match="regression evidence"):
        workload.verify(result)


@pytest.mark.parametrize("threads", [1, 3])
def test_regression_native_readers_use_requested_thread_count(tmp_path, monkeypatch, threads):
    fixture = tmp_path / "fixture"
    regression.prepare_fixture(fixture, rows=303, seed=0, chunk_rows=37)
    owner = importlib.import_module(regression.FileSource.__module__)
    original_connect = owner.connect
    observed = []

    def observed_connect(environment, *args, **kwargs):
        connection = original_connect(environment, *args, **kwargs)
        observed.append(connection.execute("SELECT current_setting('threads')").fetchone()[0])
        return connection

    monkeypatch.setattr(owner, "connect", observed_connect)
    workload = regression.build_workload(
        fixture, tmp_path / "run", WorkloadOptions(backend="file", threads=threads)
    )
    workload.verify(workload.run())
    assert observed == [threads, threads]


@pytest.mark.parametrize(("rows", "seed", "chunk_rows"), [(101, 17, 3), (137, -9, 61)])
@pytest.mark.parametrize(("evidence_rows", "evidence_bytes"), [(0, 65536), (20, 0), (20, 1200)])
def test_regression_complete_counts_survive_evidence_limits(
    tmp_path, rows, seed, chunk_rows, evidence_rows, evidence_bytes
):
    fixture = tmp_path / "fixture"
    manifest = regression.prepare_fixture(fixture, rows=rows, seed=seed, chunk_rows=chunk_rows)
    workload = regression.build_workload(
        fixture,
        tmp_path / "run",
        WorkloadOptions(backend="file", evidence_rows=evidence_rows, evidence_bytes=evidence_bytes),
    )
    result = workload.run()
    verified = workload.verify(result)
    assert verified["values_changed"] == manifest["expected"]["values_changed"]
    assert verified["truncated"]
    if not evidence_rows or not evidence_bytes:
        assert result["differences"] == []
    wrong = copy.deepcopy(result)
    wrong["rows_changed"] += 1
    with pytest.raises(AssertionError, match="rows_changed"):
        workload.verify(wrong)


def test_cache_spill_is_physical_reused_promoted_and_cleaned(tmp_path):
    fixture = tmp_path / "fixture"
    cache.prepare_fixture(fixture, rows=257, seed=73, chunk_rows=47)
    workload = cache.build_workload(fixture, tmp_path / "run", WorkloadOptions(backend="context"))
    result = workload.run()
    verified = workload.verify(result)
    assert verified["spill_read_count"] == 3
    assert verified["spill_promotions"] == verified["spill_write_count"] == 1
    assert verified["spill_read_bytes"] == 3 * verified["spill_write_bytes"]
    assert verified["spill_write_bytes"] > 257 * 32
    assert verified["temporary_files_after_close"] == 0
    assert result["summaries"]["hit_a"] == {
        "rows": 257,
        "id_min": 0,
        "id_max": 256,
        "id_sum": 32896,
        "trip_sum": 51657,
        "load_sum": 100487,
        "category_sum": 2041,
        "all_values_exact": True,
    }
    wrong = copy.deepcopy(result)
    wrong["snapshots"]["load_b"]["physical_spill_files"] = 0
    with pytest.raises(AssertionError, match="did not create"):
        workload.verify(wrong)
    wrong = copy.deepcopy(result)
    wrong["snapshots"]["hit_a"]["usage"]["frame_spill_read_count"] += 1
    with pytest.raises(AssertionError, match="counters at hit_a"):
        workload.verify(wrong)


@pytest.mark.parametrize("cache_bytes", [0, 10, 1_000_000])
def test_cache_rejects_budgets_that_do_not_exercise_one_frame_capacity(tmp_path, cache_bytes):
    fixture = tmp_path / "fixture"
    cache.prepare_fixture(fixture, rows=257, seed=0)
    with pytest.raises(ValueError, match="fits one frame but not two"):
        cache.build_workload(
            fixture,
            tmp_path / "run",
            WorkloadOptions(backend="context", cache_max_bytes=cache_bytes),
        )


def test_cache_oracle_detects_corruption_that_preserves_column_checksums(tmp_path):
    fixture = tmp_path / "fixture"
    cache.prepare_fixture(fixture, rows=257, seed=73)
    path = fixture / "a.parquet"
    frame = pl.read_parquet(path).with_columns(
        pl.when(pl.col("id") == 0)
        .then(10)
        .when(pl.col("id") == 1)
        .then(7)
        .otherwise(pl.col("load"))
        .alias("load")
    )
    frame.write_parquet(path)
    workload = cache.build_workload(fixture, tmp_path / "run", WorkloadOptions(backend="context"))
    result = workload.run()
    assert result["summaries"]["hit_a"]["load_sum"] == 100487
    assert result["summaries"]["hit_a"]["all_values_exact"] is False
    with pytest.raises(AssertionError, match="exact dataset values"):
        workload.verify(result)


@pytest.mark.parametrize("module", [cache, regression])
def test_benchmark_rejects_unsupported_backend(tmp_path, module):
    with pytest.raises(ValueError, match="supports only"):
        module.build_workload(tmp_path, tmp_path / "run", WorkloadOptions(backend="native"))
