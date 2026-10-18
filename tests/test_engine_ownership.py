"""Public invariants for resource ownership, cancellation, policy and evidence."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from qavf import DataContext, Finding, QARunner, QAVFSettings, Rule, RuleRegistry
from qavf.engine.resources import ByteCache, ResourceLimitError
from qavf.reports.models import evidence_json_size
from qavf.sources.base import SourceEnvironment, SourceIntegrityError
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource


def settings(**engine):
    return QAVFSettings.model_validate({"plugins": {"entry_points": False}, "engine": engine})


def runner(rule, **engine):
    registry = RuleRegistry()
    registry.register(rule)
    return QARunner(settings(**engine), registry)


class BaseRule(Rule):
    rule_id = "PROBE"
    name = "probe"
    description = "resource contract"
    category = "probe"


def test_constructor_failure_reclaims_prior_successful_spill(tmp_path):
    for _ in range(3):
        with pytest.raises(ResourceLimitError):
            DataContext(
                frames={
                    "a": pl.DataFrame({"x": range(100)}),
                    "b": pl.DataFrame({"x": range(2000)}),
                },
                cache_max_bytes=0,
                spill_max_bytes=20000,
                temp_directory=tmp_path,
            )
    assert list(tmp_path.iterdir()) == []


def test_planner_failure_reclaims_context(tmp_path):
    class Broken(BaseRule):
        def data_requirements(self, settings):
            raise ValueError("invalid dependencies")

        def evaluate(self, ctx):
            raise AssertionError("unreachable")

    with pytest.raises(ValueError, match="invalid dependencies"):
        runner(Broken, cache_max_bytes=0, temp_directory=tmp_path).run(
            frames={"data": pl.DataFrame({"x": range(100)})}
        )
    assert list(tmp_path.iterdir()) == []


def test_cache_native_series_and_keys_are_counted():
    cache = ByteCache(1024)
    cache["large"] = pl.Series("x", range(1_000_000))
    cache["x" * 2000] = 1
    assert len(cache) == 0 and cache.bytes == 0


def test_cache_detaches_native_and_nested_mutation():
    cache = ByteCache(10000)
    value = {"series": pl.Series("x", [1, 2]), "items": [1]}
    cache["key"] = value
    value["items"].append(999)
    value["series"][0] = 999
    hit = cache["key"]
    assert hit["items"] == [1] and hit["series"].to_list() == [1, 2]
    hit["items"].append(999)
    hit["series"][0] = 999
    assert cache["key"]["items"] == [1]
    assert cache["key"]["series"].to_list() == [1, 2]


def test_opaque_cache_values_are_recomputed_without_retention():
    class NativeHolder:
        pass

    with DataContext() as ctx:
        first = ctx.memo("opaque", NativeHolder)
        assert ctx.memo("opaque", NativeHolder) is not first


def test_memo_reclaims_inflight_lock_metadata_and_computes_once_concurrently():
    with DataContext() as ctx:
        for key in range(1000):
            ctx.memo(key, lambda: 1)
        assert len(ctx._key_locks.entries) == 0
        gate = threading.Event()
        calls = []

        def compute():
            calls.append(1)
            gate.wait(1)
            return 42

        output = []
        workers = [
            threading.Thread(target=lambda: output.append(ctx.memo("shared", compute)))
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
        time.sleep(0.01)
        gate.set()
        for worker in workers:
            worker.join(1)
        assert output == [42] * 4 and len(calls) == 1
        assert len(ctx._key_locks.entries) == 0


def test_availability_does_not_deserialize_spill():
    with DataContext(
        frames={"data": pl.DataFrame({"x": range(1000)})}, cache_max_bytes=0, spill_max_bytes=100000
    ) as ctx:
        with patch("polars.read_ipc", wraps=pl.read_ipc) as read:
            for _ in range(3):
                assert ctx.has("data")
                assert ctx.get("data").height == 1000
        assert read.call_count == 3


def test_borrowed_source_environment_and_lifetime_remain_independent(tmp_path):
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "data.parquet")
    source = FileSource(FileParams(path="data.parquet"), SourceEnvironment(base_dir=tmp_path))
    original = source.env
    with DataContext(sources={"data": source}) as first:
        with DataContext(sources={"data": source}) as second:
            assert second.get("data").item() == 1
        assert first.get("data").item() == 1
    assert source.env is original
    assert source.load().item() == 1


def test_timeout_does_not_poison_reused_context():
    gate = threading.Event()
    calls = []

    class SlowFirst(BaseRule):
        def evaluate(self, ctx):
            calls.append(1)
            if len(calls) == 1:
                gate.wait(2)
            return Finding.ok("complete")

    execute = runner(SlowFirst, rule_timeout_s=0.03, timeout_grace_s=0.001)
    with DataContext() as ctx:
        try:
            assert execute.run(ctx).get("PROBE").status.value == "ERROR"
        finally:
            gate.set()
        deadline = time.monotonic() + 1
        while ctx._active and time.monotonic() < deadline:
            time.sleep(0.001)
        for _ in range(3):
            assert execute.run(ctx).get("PROBE").status.value == "PASS"
    assert len(calls) == 4


def test_deadline_interrupts_owned_eager_query_connection():
    done = threading.Event()

    class Eager(BaseRule):
        def evaluate(self, ctx):
            try:
                ctx.get("data")
                return Finding.ok("complete")
            finally:
                done.set()

    source = QuerySource(
        QueryParams(sql=("SELECT sum(a.i * b.i) FROM range(100000) a(i), range(100000) b(i)")),
        SourceEnvironment(threads=1),
    )
    with DataContext(sources={"data": source}, env=SourceEnvironment(threads=1)) as ctx:
        try:
            report = runner(Eager, rule_timeout_s=0.08, timeout_grace_s=0.005).run(ctx)
            assert report.get("PROBE").status.value == "ERROR"
            assert done.wait(1), "owned eager DuckDB work was not interrupted"
        finally:
            for _, connection in tuple(ctx._session.connections.values()):
                connection.interrupt()
            done.wait(1)


def test_native_retry_uses_same_source_policy(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)
    source = FileSource(FileParams(path=str(path), retries=1, retry_delay_s=0), SourceEnvironment())
    from qavf.engine import connections as module

    original = module.connect
    calls = []

    def connect(*args, **kwargs):
        con = original(*args, **kwargs)
        sql = con.sql

        def retry_once(*values, **options):
            calls.append(1)
            if len(calls) == 1:
                raise duckdb.HTTPException("HTTP Error: 503 Service Unavailable")
            return sql(*values, **options)

        con.sql = retry_once
        return con

    with DataContext(sources={"data": source}) as ctx, patch.object(module, "connect", connect):
        assert ctx.has("data")
        assert len(calls) == 2
        assert ctx._catalog.sources["data"].load_notes["retries_used"] == 1


def test_native_identity_change_is_not_silently_accepted(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    with DataContext(sources={"data": source}) as ctx:
        assert ctx.sql("SELECT x FROM data").item() == 1
        pl.DataFrame({"x": [1234]}).write_parquet(path)
        with pytest.raises(SourceIntegrityError):
            ctx.sql("SELECT x FROM data")
        assert "data" in ctx.verify_inputs()


def test_final_identity_error_preserves_completed_rule_outcome(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)

    class ModifyAfterRead(BaseRule):
        def evaluate(self, ctx):
            assert ctx.sql("SELECT x FROM data").item() == 1
            pl.DataFrame({"x": [999]}).write_parquet(path)
            return Finding.fail("observed an independent problem", affected_count=1)

    report = runner(ModifyAfterRead).run(
        frames={"data": FileSource(FileParams(path=str(path)), SourceEnvironment())}
    )
    assert report.get("PROBE").status.value == "FAIL"
    assert report.get("QAVF-INPUT-INTEGRITY").status.value == "ERROR"
    assert "data" in report.metadata.unavailable_datasets


def test_optional_diagnostic_and_canonical_evidence_share_budget():
    class Large(BaseRule):
        def evaluate(self, ctx):
            return Finding.fail(
                "x" * 1_000_000,
                affected=[{"x": float("nan")}] * 100,
                metrics={"detail": "x" * 40},
                affected_count=999,
            )

    result = runner(Large, max_evidence_bytes=256).run().get("PROBE")
    sizes = [
        evidence_json_size(value)
        for value in (result.affected_records, result.metrics, result.message)
        if value
    ]
    assert sum(sizes) <= 256
    assert result.status.value == "FAIL" and result.affected_count == 999


def test_sql_sample_accepts_lexical_terminator_and_trailing_comment():
    with DataContext() as ctx:
        sample, count = ctx.sql_sample("SELECT 1 AS x; -- comment", 3)
        assert count == 1 and sample.item() == 1


def test_bound_database_source_uses_its_own_connection_and_rejects_alias_conflict(tmp_path):
    from qavf.settings import ConnectionSpec
    from qavf.sources.parameters import DatabaseParams
    from qavf.sources.query import DatabaseSource

    paths = [tmp_path / "intended.duckdb", tmp_path / "other.duckdb"]
    for path, value in zip(paths, [1, 999], strict=True):
        with duckdb.connect(str(path)) as connection:
            connection.execute(f"CREATE TABLE t AS SELECT {value} AS x")
    source = DatabaseSource(
        DatabaseParams(connection="w", table="t"),
        SourceEnvironment(
            base_dir=tmp_path, connections={"w": ConnectionSpec(type="duckdb", path=paths[0].name)}
        ),
    )
    conflicting = SourceEnvironment(
        connections={"w": ConnectionSpec(type="duckdb", path=str(paths[1]))}
    )
    with DataContext(sources={"data": source}, env=conflicting) as ctx:
        assert ctx.sql("SELECT x FROM data").item() == 1
        with pytest.raises(Exception, match="conflicting connection bindings"):
            ctx.sql("SELECT x FROM data UNION ALL SELECT x FROM w.t")
    with DataContext(sources={"data": source}) as ctx:
        assert ctx.sql("SELECT x FROM data").item() == 1
    from qavf import validate

    assert validate(source, schema={"columns": {"x": {"ge": 1, "le": 1}}}).exit_code() == 0


def test_shared_nested_cache_values_remain_one_detached_allocation():
    import pandas as pd

    cache = ByteCache(20000)
    frame = pd.DataFrame({"x": range(1000)})
    cache["value"] = {"a": frame, "b": frame}
    result = cache["value"]
    assert result["a"] is result["b"]
    result["a"].iloc[0, 0] = 999
    assert cache["value"]["a"].iloc[0, 0] == 0


@pytest.mark.parametrize("container", ["dataframe", "series"])
def test_nested_pandas_object_payload_is_counted(container):
    import pandas as pd

    from qavf.engine.resources import retained_size

    value = (
        pd.DataFrame({"x": [["x" * 50000]]})
        if container == "dataframe"
        else pd.Series([["x" * 50000]])
    )
    assert retained_size(value) >= 50000
    cache = ByteCache(1024)
    cache["value"] = value
    assert len(cache) == 0


@pytest.mark.parametrize("container", ["array", "chunked", "table"])
def test_arrow_slice_cache_counts_the_retained_parent_buffers(container):
    from qavf.engine.resources import retained_size

    parent = pa.array(range(1_000_000), type=pa.int64())
    value = parent.slice(0, 1)
    if container == "chunked":
        value = pa.chunked_array([value])
    elif container == "table":
        value = pa.table({"x": value})
    assert value.nbytes == 8
    assert retained_size(value) >= 8_000_000
    cache = ByteCache(1024)
    cache["value"] = value
    assert len(cache) == 0 and cache.bytes == 0


def test_report_finalization_does_not_wait_on_overdue_source_identity_lock(tmp_path):
    gate = threading.Event()
    holding = threading.Event()
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)

    class HoldsSourceLock(BaseRule):
        def evaluate(self, ctx):
            with ctx._catalog.sources["data"]._state_lock:
                holding.set()
                gate.wait(2)
            return Finding.ok("done")

    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    with DataContext(sources={"data": source}) as ctx:
        # Establish input use before the deadline: this test measures finalization
        # around an overdue identity lock, not cold native imports or file binding.
        ctx.sql("SELECT x FROM data")
        started = time.monotonic()
        try:
            report = runner(HoldsSourceLock, rule_timeout_s=0.1, timeout_grace_s=0.001).run(ctx)
            assert holding.is_set()
            assert time.monotonic() - started < 0.8
            assert report.get("PROBE").status.value == "ERROR"
            assert report.metadata.datasets[0].notes["identity_verified"] is False
        finally:
            gate.set()
            deadline = time.monotonic() + 1
            while ctx._active and time.monotonic() < deadline:
                time.sleep(0.001)


def test_completed_deadline_workers_release_borrowed_context_connections():
    class ReadsSQL(BaseRule):
        def evaluate(self, ctx):
            result = ctx.sql("SELECT 1")
            assert result.item() == 1
            return Finding.ok("complete")

    with DataContext(connection_slots=2) as ctx:
        execute = runner(ReadsSQL, rule_timeout_s=1, timeout_grace_s=0.01)
        for _ in range(5):
            report = execute.run(ctx)
            assert report.get("PROBE").status.value == "PASS"
        deadline = time.monotonic() + 1
        while ctx._resources.reservations and time.monotonic() < deadline:
            time.sleep(0.001)
        assert len(ctx._resources.reservations) == 0


def test_parallel_lazy_access_uses_distinct_python_wrappers():
    from concurrent.futures import ThreadPoolExecutor

    with DataContext(frames={"data": pl.DataFrame({"x": range(1000)}).lazy()}) as ctx:

        def read(_):
            assert ctx.schema("data") == {"x": pl.Int64}
            return ctx.scan("data").select(pl.col("x").sum()).collect().item()

        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(read, range(20))) == [499500] * 20
