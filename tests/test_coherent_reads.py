"""Behavioral regression gates for coherent reads and resource-efficient ownership."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.resources import ByteCache, FrameStore, ResourceLimitError
from qavf.engine.runner import QARunner
from qavf.settings import DatasetSpec, QAVFSettings
from qavf.sources.base import DataSource, SourceEnvironment, SourceError, SourceParams


@pytest.mark.parametrize("operation", ["project", "reduce", "get", "join"])
def test_lazy_budget_and_accepted_values_share_one_observation(operation):
    calls = []

    def change(frame):
        calls.append(1)
        return pl.DataFrame({"x": range(1 if len(calls) == 1 else 10)})

    plan = pl.DataFrame({"x": [0]}).lazy().map_batches(change, schema={"x": pl.Int64})
    with DataContext(
        frames={"d": plan}, specs={"d": DatasetSpec(load_policy={"row_budget": 2})}
    ) as ctx:
        if operation == "project":
            result = ctx.project("d", ["x"])
        elif operation == "reduce":
            result = ctx.reduce("d", lambda data: data.select(pl.len()))
        elif operation == "get":
            result = ctx.get("d")
        else:
            data = ctx.scan("d")
            result = ctx.collect_many(["d"], data.join(data, on="x"))
        assert result.height == 1
        assert len(calls) == 1
        assert ctx.resource_usage()["retained_total_bytes"] <= ctx._resources.retained_budget.limit


def test_typed_quota_projection_evaluates_once_and_releases_admission():
    calls = []

    class Typed(DataSource):
        kind, Params = "review_typed", SourceParams

        def load(self):
            raise AssertionError("typed reader expected")

        def scan_polars(self):
            return pl.DataFrame({"x": range(100)}).lazy().map_batches(
                lambda frame: (calls.append(frame.height), frame)[1], schema={"x": pl.Int64}
            )

    source = Typed(SourceParams(type=Typed.kind, row_budget=1000), SourceEnvironment())
    with DataContext(sources={"d": source}) as ctx:
        assert ctx.project("d", ["x"]).height == 100
        assert calls == [100]
        with ctx.observe("d", ["x"]) as observed:
            assert observed.reduce(lambda data: data.select(pl.len())).item() == 100
        assert calls == [100]
        ctx.release(["d"])
        assert ctx.resource_usage()["retained_total_bytes"] == 0


def test_quota_admission_normalizes_chained_column_names_only_once():
    frame = pl.DataFrame({"a": [1], "b": [2]})
    spec = DatasetSpec(rename={"a": "b", "b": "c"}, load_policy={"row_budget": 10})
    with DataContext(frames={"d": frame.lazy()}, specs={"d": spec}) as ctx:
        assert ctx.schema("d") == {"b": pl.Int64, "c": pl.Int64}
        assert ctx.project("d", ["b"]).item() == 1
        with ctx.observe("d", ["b"]) as observed:
            assert observed.reduce(lambda data: data).item() == 1
        assert ctx.get("d").to_dicts() == [{"b": 1, "c": 2}]


def test_ad_hoc_budget_schema_projection_and_observation_share_values():
    calls = []

    class Typed(DataSource):
        kind, Params = "review_adhoc", SourceParams

        def load(self):
            raise AssertionError("typed reader expected")

        def scan_polars(self):
            def change(frame):
                calls.append(1)
                return pl.DataFrame({"x": range(1 if len(calls) == 1 else 10)})

            return pl.DataFrame({"x": [0]}).lazy().map_batches(change, schema={"x": pl.Int64})

    class Registry:
        def build(self, config, env):
            return Typed(SourceParams(type=Typed.kind, row_budget=2), env)

    def consume(view):
        assert view.schema() == {"x": pl.Int64}
        assert view.project(["x"]).height == 1
        with view.observe(["x"]) as observed:
            assert observed.reduce(lambda data: data.select(pl.len())).item() == 1

    with DataContext(source_registry=Registry()) as ctx:
        ctx.read_source({"type": "review_adhoc"}, consume)
        assert calls == [1]
        assert ctx.resource_usage()["retained_total_bytes"] == 0


def test_failed_quota_releases_owned_memory_and_files():
    with DataContext(
        frames={"d": pl.DataFrame({"x": range(100)}).lazy()},
        specs={"d": DatasetSpec(load_policy={"row_budget": 2})},
    ) as ctx:
        with pytest.raises(SourceError, match="more than 2 rows"):
            ctx.scan("d")
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


@pytest.mark.parametrize("cache_bytes", [0, 100_000])
def test_admitted_observation_lease_survives_dataset_release(cache_bytes):
    with DataContext(
        frames={"d": pl.DataFrame({"x": range(10)}).lazy()},
        specs={"d": DatasetSpec(load_policy={"row_budget": 10})},
        cache_max_bytes=cache_bytes,
    ) as ctx:
        ctx.scan("d")  # Admit the quota and its coherent input exactly once.
        with ctx.observe("d") as observed:
            before = ctx.resource_usage()
            ctx.release(["d"])
            after = ctx.resource_usage()
            assert before["retained_total_bytes"] == after["retained_total_bytes"]
            assert before["owned_spill_bytes"] == after["owned_spill_bytes"]
            assert observed.reduce(lambda data: data.select(pl.sum("x"))).item() == 45
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


def test_nested_cache_admission_serializes_once_and_preserves_values(tmp_path):
    frame = pl.DataFrame({"nested": [[1, 2, 3]] * 10000})
    original, calls = pl.DataFrame.write_ipc, []

    def record(self, file, *args, **kwargs):
        calls.append(self.height)
        return original(self, file, *args, **kwargs)

    for store in (ByteCache(10_000_000), FrameStore(10_000_000, 10_000_000, tmp_path)):
        calls.clear()
        with patch.object(pl.DataFrame, "write_ipc", record):
            store["d"] = frame
        assert calls == [10000]
        assert store["d"].equals(frame)
        store.clear()


def test_small_lazy_observation_uses_available_backend_and_identity_derivation_reuses_it():
    frame = pl.DataFrame({"x": range(1000)})
    batches_available = hasattr(pl.LazyFrame, "collect_batches")
    spill_limit = 0 if batches_available else 100_000
    with DataContext(
        frames={"d": frame.lazy()}, cache_max_bytes=100_000, spill_max_bytes=spill_limit
    ) as ctx:
        with ctx.observe("d") as observed:
            before = ctx.resource_usage()
            if batches_available:
                assert before["retained_observation_bytes"] >= frame.estimated_size()
                assert before["owned_spill_bytes"] == 0
            else:
                assert before["retained_observation_bytes"] == 0
                assert 0 < before["owned_spill_bytes"] <= spill_limit
            with observed.derived(lambda data: data) as derived:
                assert derived.reduce(lambda data: data.select(pl.len())).item() == 1000
                during = ctx.resource_usage()
                assert during["retained_observation_bytes"] == before["retained_observation_bytes"]
                assert during["owned_spill_bytes"] == before["owned_spill_bytes"]
            assert observed.reduce(lambda data: data.select(pl.len())).item() == 1000
            assert observed.reduce(lambda data: data).equals(frame)
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


def test_observation_and_frame_cache_share_one_retention_envelope():
    batches_available = hasattr(pl.LazyFrame, "collect_batches")
    with DataContext(
        frames={"d": pl.DataFrame({"x": range(500)}).lazy()},
        cache_max_bytes=8000, spill_max_bytes=100_000,
    ) as ctx:
        with ctx.observe("d") as observed:
            ctx._frames["another"] = pl.DataFrame({"x": range(600)})
            usage = ctx.resource_usage()
            assert usage["retained_total_bytes"] <= 8000
            assert 0 < usage["owned_spill_bytes"] <= 100_000
            if batches_available:
                assert usage["retained_observation_bytes"] > 0
                assert "another" in ctx._frames.paths
            else:
                # The exact IPC observation leaves the resident budget for the cache.
                assert usage["retained_observation_bytes"] == 0
                assert usage["retained_cache_bytes"] > 0
                assert "another" not in ctx._frames.paths
            assert ctx._frames["another"].equals(pl.DataFrame({"x": range(600)}))
            assert observed.reduce(lambda data: data.select(pl.sum("x"))).item() == 124750
        assert ctx.resource_usage()["retained_observation_bytes"] == 0
        if not batches_available:
            assert ctx.resource_usage()["owned_spill_bytes"] == 0
        ctx.release(["another"])
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


def test_materialized_observation_remains_charged_after_cache_release():
    with DataContext(frames={"d": pl.DataFrame({"x": range(100)})}) as ctx:
        with ctx.observe("d") as observed:
            before = ctx.resource_usage()["retained_total_bytes"]
            ctx.release(["d"])
            assert ctx.resource_usage()["retained_cache_bytes"] == 0
            assert ctx.resource_usage()["retained_observation_bytes"] == before
            assert observed.reduce(lambda data: data.select(pl.sum("x"))).item() == 4950
        assert ctx.resource_usage()["retained_total_bytes"] == 0


def test_incremental_observation_spills_without_replaying_source():
    calls = []
    original = pl.DataFrame({"x": range(150000)})
    plan = original.lazy().map_batches(
        lambda frame: (calls.append(1), frame)[1], schema={"x": pl.Int64}
    )
    with DataContext(frames={"d": plan}, cache_max_bytes=750000) as ctx:
        with ctx.observe("d") as observed:
            assert ctx.resource_usage()["owned_spill_bytes"] > 0
            assert ctx.resource_usage()["retained_total_bytes"] <= 750000
            assert pl.concat(list(observed.batches(10000))).equals(original)
            assert len(calls) == 1
        assert ctx.resource_usage()["owned_spill_bytes"] == 0
        assert ctx.resource_usage()["retained_total_bytes"] == 0


def test_no_memory_or_disk_capacity_fails_without_leaking_partial_observation():
    with DataContext(
        frames={"d": pl.DataFrame({"x": range(1000)}).lazy()},
        cache_max_bytes=0, spill_max_bytes=100,
    ) as ctx:
        with pytest.raises(ResourceLimitError), ctx.observe("d"):
            pytest.fail("over-capacity observation was accepted")
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


def test_small_memory_target_reduces_rule_concurrency_before_native_oom():
    class Native(Rule):
        rule_id = "REVIEW-MEMORY"
        name = description = "Memory scheduling gate"
        category = "custom"

        def evaluate(self, ctx):
            assert ctx.sql(
                "SELECT i % 10000 AS k, count(*) AS n "
                "FROM range(100000) AS t(i) GROUP BY k"
            ).height == 10000
            return Finding.ok("assessed")

    registry = RuleRegistry()
    for index in range(8):
        registry.register(type(f"Native{index}", (Native,), {"rule_id": f"MEM-{index}"}))
    settings = QAVFSettings(
        plugins={"entry_points": False},
        engine={"memory_budget_bytes": 32 * 1024**2, "native_threads": 1, "max_workers": 8},
    )
    report = QARunner(settings, registry).run()
    assert report.metadata.max_workers == 1
    assert report.exit_code() == 0
    with DataContext(memory_budget_bytes=32 * 1024**2) as ctx:
        assert ctx.resource_usage()["native_memory_bytes_per_connection"] == 8 * 1024**2
