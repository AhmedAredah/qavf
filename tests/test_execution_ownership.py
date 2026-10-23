"""Execution ownership boundaries: memo values, deadlines and cleanup."""

from __future__ import annotations

import gc
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import pytest

from qavf import Finding, QARunner, QAVFSettings, Rule, RuleRegistry
from qavf.engine.cache_ownership import retained_size
from qavf.engine.context import DataContext
from qavf.engine.relations import bind_scan
from qavf.engine.resources import ByteCache, ResourceLimitError
from qavf.sources.base import SourceEnvironment
from qavf.sources.file import FileSource
from qavf.sources.native import OwnedConnection, connect
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource


@pytest.mark.parametrize("record_array", [False, True])
def test_memo_structured_scalar_is_detached_from_parent_and_every_hit(record_array):
    parent = np.array([(1, (2, 3))], dtype=[("count", "i8"), ("coordinates", "i8", (2,))])
    if record_array:
        parent = parent.view(np.recarray)
    scalar = parent[0]
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return scalar

    with DataContext(cache_max_bytes=8192) as context:
        context.memo("structured", compute)
        parent["count"][0] = 99
        parent["coordinates"][0, 0] = 98
        hit = context.memo("structured", compute)
        assert isinstance(hit, type(scalar))
        assert hit["count"] == 1
        assert hit["coordinates"].tolist() == [2, 3]
        hit["count"] = 97
        hit["coordinates"][0] = 96
        next_hit = context.memo("structured", compute)
        assert next_hit["count"] == 1
        assert next_hit["coordinates"].tolist() == [2, 3]
        assert calls == 1


def test_structured_scalar_cache_does_not_keep_large_parent_alive():
    parent = np.zeros(100_000, dtype=[("count", "i8"), ("value", "f8")])
    parent[10] = (10, 2.5)
    reference = weakref.ref(parent)
    scalar = parent[10]
    cache = ByteCache(4096)
    cache["scalar"] = scalar
    del scalar, parent
    gc.collect()
    assert reference() is None
    assert cache["scalar"].tolist() == (10, 2.5)
    assert 16 <= cache.bytes <= cache.limit
    cache.clear()
    assert cache.budget.used == 0


@pytest.mark.parametrize("wrapped", [False, True])
def test_object_bearing_numpy_scalar_is_not_cached(wrapped):
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        array = np.empty(1, dtype=[("payload", object)])
        array["payload"][0] = {"mutable": [], "large": bytearray(1_000_000)}
        return {"record": [array[0]]} if wrapped else array[0]

    with DataContext(cache_max_bytes=8192) as context:
        first = context.memo("objects", compute)
        record = first["record"][0] if wrapped else first
        record["payload"]["mutable"].append("changed")
        second = context.memo("objects", compute)
        second_record = second["record"][0] if wrapped else second
        assert second_record["payload"]["mutable"] == []
        assert calls == 2
        assert context.resource_usage()["retained_total_bytes"] == 0


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("placement", ["top", "field", "subarray", "title"])
def test_numpy_dtype_python_payloads_are_not_retained(scalar, placement):
    payload = {"items": [1], "large": bytearray(1_000_000)}
    child = np.dtype("i8", metadata={"payload": payload})
    if placement == "top":
        dtype = np.dtype([("x", "i8")], metadata={"payload": payload})
    elif placement == "field":
        dtype = np.dtype([("x", child)])
    elif placement == "subarray":
        dtype = np.dtype([("x", child, (2,))])
    else:
        dtype = np.dtype([((payload, "x"), "i8")])
    array = np.zeros(1, dtype=dtype)
    value = array[0] if scalar else array
    assert not value.dtype.hasobject
    assert retained_size(value) == sys.maxsize
    with DataContext(cache_max_bytes=8192) as context:
        assert context.memo("value", lambda: value) is value
        payload["items"].append(2)
        replacement = object()
        assert context.memo("value", lambda: replacement) is replacement
        assert context.resource_usage()["retained_total_bytes"] == 0


def test_plain_string_field_titles_remain_cacheable():
    array = np.zeros(1, dtype=[(("title", "field"), "i8")])
    array["field"][0] = 3
    with DataContext(cache_max_bytes=8192) as context:
        context.memo("value", lambda: array)
        array["field"][0] = 9
        result = context.memo("value", lambda: None)
        assert result.dtype == array.dtype
        assert result["title"].tolist() == [3]


@pytest.mark.parametrize(
    "placement", ["series", "frame", "series_index", "frame_index", "columns", "categories"]
)
def test_pandas_wrappers_do_not_retain_numpy_dtype_metadata(placement):
    pd = pytest.importorskip("pandas")
    payload = {"items": [1], "large": bytearray(1_000_000)}
    array = np.array([1], dtype=np.dtype("i8", metadata={"payload": payload}))
    if placement == "series":
        value = pd.Series(array)
    elif placement == "frame":
        value = pd.DataFrame({"x": array})
    elif placement == "series_index":
        value = pd.Series([1], index=pd.Index(array))
    elif placement == "frame_index":
        value = pd.DataFrame({"x": [1]}, index=pd.Index(array))
    elif placement == "columns":
        value = pd.DataFrame([[1]], columns=pd.Index(array))
    else:
        value = pd.Series(pd.Categorical.from_codes([0], categories=pd.Index(array)))
    assert retained_size(value) >= sys.maxsize
    with DataContext(cache_max_bytes=8192) as context:
        assert context.memo("value", lambda: value) is value
        payload["items"].append(2)
        replacement = object()
        assert context.memo("value", lambda: replacement) is replacement
        assert context.resource_usage()["retained_total_bytes"] == 0


class _ReadsSQL(Rule):
    rule_id = "CLEANUP-PROBE"
    name = "worker cleanup"
    description = "deadline worker connection lifetime"
    category = "test"

    def evaluate(self, ctx):
        assert ctx.sql("SELECT 1").item() == 1
        return Finding.ok("complete")


def _deadline_runner(*rules, timeout=10):
    registry = RuleRegistry()
    for rule in rules or (_ReadsSQL,):
        registry.register(rule)
    settings = QAVFSettings.model_validate({
        "plugins": {"entry_points": False},
        "engine": {"parallel": False, "rule_timeout_s": timeout, "timeout_grace_s": 0.01},
    })
    return QARunner(settings, registry)


def test_deadline_report_waits_for_worker_connection_cleanup(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    execute = _deadline_runner()
    with DataContext(connection_slots=1) as ctx, ThreadPoolExecutor(max_workers=1) as calls:
        original = ctx.finish_worker

        def delayed(owner=None):
            entered.set()
            assert release.wait(5)
            original(owner)

        monkeypatch.setattr(ctx, "finish_worker", delayed)
        try:
            for _ in range(3):
                entered.clear()
                release.clear()
                result = calls.submit(execute.run, ctx)
                assert entered.wait(3)
                assert len(ctx._resources.reservations) == 1
                with pytest.raises(TimeoutError):
                    result.result(timeout=0.1)
                release.set()
                report = result.result(timeout=3)
                assert report.get(_ReadsSQL.rule_id).status.value == "PASS"
                assert not ctx._resources.reservations
        finally:
            release.set()


def test_deadline_cleanup_timeout_retains_worker_capacity(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    workers = []
    execute = _deadline_runner(timeout=0.5)
    with DataContext(connection_slots=1) as ctx, ThreadPoolExecutor(max_workers=1) as calls:
        original = ctx.finish_worker

        def delayed(owner=None):
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5)
            original(owner)

        monkeypatch.setattr(ctx, "finish_worker", delayed)
        try:
            result = calls.submit(execute.run, ctx)
            assert entered.wait(3)
            report = result.result(timeout=2)
            finding = report.get(_ReadsSQL.rule_id)
            assert finding.status.value == "ERROR"
            assert "did not finish" in finding.message
            assert len(ctx._resources.reservations) == 1
            blocked = execute.run(ctx).get(_ReadsSQL.rule_id)
            assert blocked.status.value == "ERROR"
            assert "worker capacity is retained" in blocked.message
        finally:
            release.set()
            for worker in workers:
                worker.join(3)
            monkeypatch.setattr(ctx, "finish_worker", original)
        assert all(not worker.is_alive() for worker in workers)
        assert not ctx._resources.reservations
        assert execute.run(ctx).get(_ReadsSQL.rule_id).status.value == "PASS"


@pytest.mark.parametrize("boundary", ["finish_worker", "end_rule"])
@pytest.mark.parametrize("error", [RuntimeError, SystemExit])
def test_deadline_worker_cleanup_errors_are_explicit_outcomes(monkeypatch, boundary, error):
    with DataContext(connection_slots=1) as ctx:
        original = getattr(ctx, boundary)

        def broken(*args):
            original(*args)
            raise error(f"failed {boundary}")

        monkeypatch.setattr(ctx, boundary, broken)
        report = _deadline_runner(timeout=1).run(ctx)
        errors = [result for result in report.results if result.status.value == "ERROR"]
        assert len(errors) == 1
        assert f"failed {boundary}" in errors[0].message
        assert report.exit_code() != 0
        assert not ctx._resources.reservations


def test_deadline_worker_reuses_connection_for_queued_rules(monkeypatch):
    rules = [
        type(f"ReadsSQL{index}", (_ReadsSQL,), {"rule_id": f"SQL-{index}"}) for index in range(3)
    ]
    with DataContext(connection_slots=1) as ctx:
        original = ctx._resources.acquire
        acquisitions = []

        def acquire():
            acquisitions.append(1)
            return original()

        monkeypatch.setattr(ctx._resources, "acquire", acquire)
        report = _deadline_runner(*rules).run(ctx)
        assert len(report.results) == 3
        assert all(result.status.value == "PASS" for result in report.results)
        assert acquisitions == [1]
        assert not ctx._resources.reservations


@pytest.mark.parametrize("error", [RuntimeError, SystemExit])
def test_deadline_worker_exception_does_not_abandon_queued_rules(monkeypatch, error):
    rules = [
        type(f"ReadsSQL{index}", (_ReadsSQL,), {"rule_id": f"SQL-{index}"}) for index in range(3)
    ]
    execute = _deadline_runner(*rules)
    original = execute._execute

    def broken(rule, *args, **kwargs):
        if rule.rule_id == "SQL-0":
            raise error("unexpected invocation failure")
        return original(rule, *args, **kwargs)

    monkeypatch.setattr(execute, "_execute", broken)
    with DataContext(connection_slots=1) as ctx:
        report = execute.run(ctx)
        assert len(report.results) == 3
        assert report.get("SQL-0").status.value == "ERROR"
        assert report.get("SQL-1").status.value == "PASS"
        assert report.get("SQL-2").status.value == "PASS"
        assert not ctx._resources.reservations


def test_owned_context_deadline_does_not_race_a_blocked_native_close():
    entered, release = threading.Event(), threading.Event()
    contexts, workers = [], []

    class DelayedNative:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def close(self):
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5)
            self.connection.close()

    class ReadsWithDelayedClose(_ReadsSQL):
        def evaluate(self, ctx):
            outcome = super().evaluate(ctx)
            contexts.append(ctx)
            connection = next(iter(ctx._sql_connections._local.connections.values()))
            connection._connection = DelayedNative(connection._connection)
            return outcome

    execute = _deadline_runner(ReadsWithDelayedClose, timeout=0.5)
    with ThreadPoolExecutor(max_workers=1) as calls:
        try:
            result = calls.submit(execute.run)
            assert entered.wait(3)
            report = result.result(timeout=2)
            assert report.get(_ReadsSQL.rule_id).status.value == "ERROR"
            ctx = contexts[0]
            assert ctx._active
            assert len(ctx._resources.reservations) == 1
            assert ctx._resources.directory.exists()
        finally:
            release.set()
            for worker in workers:
                worker.join(3)
        assert all(not worker.is_alive() for worker in workers)
        assert not ctx._active and not ctx._resources.reservations
        assert not ctx._resources.directory.exists()


def test_cancelled_eager_source_cannot_start_query_after_idle_interrupt(monkeypatch):
    from qavf.sources import query as query_module

    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    errors, workers = [], []
    original_fetch = query_module.fetch
    # This is an already-loaded query boundary, not a cold-import timing test.
    warm = QuerySource(QueryParams(sql="SELECT 1"), SourceEnvironment(threads=1))
    warm.load()

    def delayed_fetch(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_fetch(*args, **kwargs)

    class Eager(_ReadsSQL):
        def evaluate(self, ctx):
            workers.append(threading.current_thread())
            try:
                ctx.get("data")
                return Finding.ok("complete")
            except Exception as exc:
                errors.append(exc)
                raise
            finally:
                done.set()

    source = QuerySource(
        QueryParams(sql="SELECT sum(a.i * b.i) FROM range(100000) a(i), range(100000) b(i)"),
        SourceEnvironment(threads=1),
    )
    monkeypatch.setattr(query_module, "fetch", delayed_fetch)
    with DataContext(sources={"data": source}, env=SourceEnvironment(threads=1)) as ctx:
        with ThreadPoolExecutor(max_workers=1) as calls:
            try:
                result = calls.submit(_deadline_runner(Eager, timeout=0.5).run, ctx)
                assert entered.wait(3)
                report = result.result(timeout=2)
                assert report.get(_ReadsSQL.rule_id).status.value == "ERROR"
                assert not done.is_set()
                release.set()
                assert done.wait(1), "a query started after the owned invocation was cancelled"
                assert errors and "cancelled" in str(errors[0])
            finally:
                release.set()
                for _, connection in tuple(ctx._session.connections.values()):
                    connection.interrupt()
                for worker in workers:
                    worker.join(3)
        assert not ctx._active and not ctx._resources.reservations


@pytest.mark.parametrize("operation", ["execute", "executemany", "sql", "fetchone"])
def test_cancelled_owned_connection_checks_captured_and_chained_calls(operation):
    with DataContext() as ctx:
        invocation = ctx.begin_rule()
        connection = connect(ctx.env)
        try:
            chained = connection.execute("SELECT 1")
            assert chained is connection
            captured = getattr(chained, operation)
            ctx.cancel_rule(invocation)
            args = {
                "execute": ("SELECT 2",),
                "executemany": ("SELECT ?", [[2]]),
                "sql": ("SELECT 2",),
                "fetchone": (),
            }[operation]
            with pytest.raises(ResourceLimitError, match="cancelled"):
                captured(*args)
            with pytest.raises(ResourceLimitError, match="cancelled"):
                chained.execute("SELECT 3")
            connection.interrupt()
        finally:
            connection.close()
            ctx.end_rule()
        assert not ctx._resources.reservations
        # Cancelling one invocation must not poison the reusable context.
        with connect(ctx.env) as fresh:
            assert fresh.execute("SELECT 4").fetchone() == (4,)


def test_cancelled_owned_connection_can_rollback_and_unregister():
    import pyarrow as pa

    with DataContext() as ctx:
        invocation = ctx.begin_rule()
        connection = connect(ctx.env)
        try:
            assert connection.register("data", pa.table({"x": [1]})) is connection
            connection.execute("BEGIN TRANSACTION")
            ctx.cancel_rule(invocation)
            assert connection.rollback() is connection
            assert connection.unregister("data") is connection
            with pytest.raises(ResourceLimitError, match="cancelled"):
                connection.unregister("data").execute("SELECT 2")
            connection.interrupt()
        finally:
            connection.close()
            ctx.end_rule()
        assert not ctx._resources.reservations


@pytest.mark.parametrize(
    "operation", ["budget_count", "schema", "create_view", "reader", "columns"]
)
def test_cancelled_owned_file_relation_checks_deferred_consumers(tmp_path, operation):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(path)
    with DataContext() as ctx:
        invocation = ctx.begin_rule()
        connection = connect(ctx.env)
        source = FileSource(FileParams(path=str(path), row_budget=4), ctx.env)
        try:
            relation = bind_scan(connection, source, source.scan())
            # Capture terminal methods before cancellation, after lazy transforms.
            if operation == "budget_count":
                consume = relation.limit(5).aggregate("count(*)").fetchone
            elif operation == "schema":
                consume = relation.limit(0).pl
            elif operation == "create_view":
                create = relation.project("x AS value").create_view

                def consume():
                    return create("data", replace=True)

            elif operation == "reader":
                read = getattr(relation, "to_arrow_reader", None) or relation.fetch_arrow_reader

                def consume():
                    return read(1)

            else:

                def consume():
                    return relation.columns

            ctx.cancel_rule(invocation)
            with pytest.raises(ResourceLimitError, match="cancelled"):
                consume()
            with pytest.raises(ResourceLimitError, match="cancelled"):
                relation.limit(1)
        finally:
            connection.close()
            ctx.end_rule()
        assert not ctx._resources.reservations


def test_owned_relations_preserve_values_and_direct_relation_arguments():
    with DataContext() as ctx, connect(ctx.env) as connection:
        left = connection.sql("SELECT 1 AS x").set_alias("left_values")
        right = connection.sql("SELECT 1 AS y").set_alias("right_values")
        assert left.columns == ["x"]
        assert left.join(right, "x = y").fetchall() == [(1, 1)]
        assert left.union(union_rel=connection.sql("SELECT 2 AS x")).order("x").fetchall() == [
            (1,), (2,)
        ]


@pytest.mark.parametrize("raised", [False, True])
def test_cancelled_sql_operation_releases_bindings_without_masking_errors(raised):
    with DataContext() as ctx:
        invocation = ctx.begin_rule()
        connections = ctx._sql_connections
        connection = connections.connection(())
        original_error = ValueError("original operation failure")

        def operation():
            with connections.operation():
                connections.register_frame(connection, 'input"name', pl.DataFrame({"x": [1, 2]}))
                connection.sql('SELECT * FROM "input""name"').create_view("derived")
                connections.remember("derived")
                assert connections.usage()["active_sql_registration_bytes_upper_bound"] > 0
                ctx.cancel_rule(invocation)
                if raised:
                    raise original_error

        try:
            if raised:
                with pytest.raises(ValueError) as caught:
                    operation()
                assert caught.value is original_error
            else:
                operation()
            assert connections.usage()["active_sql_registration_bytes_upper_bound"] == 0
            assert not connections._local.registered
        finally:
            ctx.end_rule()
        assert connection.execute("SHOW TABLES").fetchall() == []
        ctx.finish_worker()
        assert not ctx._resources.reservations


def test_cancelled_observation_sample_drops_staged_table(monkeypatch):
    from qavf.engine.observation_sql import sample_observed

    with (
        DataContext(frames={"data": pl.DataFrame({"x": [1, 2]})}) as ctx,
        ctx.observe("data") as observed,
    ):
        invocation = ctx.begin_rule()
        original = OwnedConnection.__getattr__
        staged = []

        def intercept(connection, name):
            value = original(connection, name)
            if name != "execute":
                return value

            def execute(statement, *args, **kwargs):
                result = value(statement, *args, **kwargs)
                if statement.startswith("CREATE TEMP TABLE"):
                    staged.append(connection)
                    ctx.cancel_rule(invocation)
                return result

            return execute

        monkeypatch.setattr(OwnedConnection, "__getattr__", intercept)
        try:
            with pytest.raises(ResourceLimitError, match="cancelled"):
                sample_observed(ctx, "SELECT * FROM data", {"data": observed}, 1)
        finally:
            ctx.end_rule()
        assert len(staged) == 1
        assert staged[0].execute("SHOW TABLES").fetchall() == []
        assert ctx._sql_connections.usage()["active_sql_registration_bytes_upper_bound"] == 0


def test_owned_relations_support_legacy_acquire_release_only_budget():
    class Budget:
        def __init__(self):
            self.released = []

        def acquire(self):
            return "legacy", None, None

        def release(self, token):
            self.released.append(token)

    budget = Budget()
    with connect(SourceEnvironment(connection_budget=budget)) as connection:
        assert connection.sql("SELECT 7 AS x").limit(1).pl().item() == 7
    assert budget.released == ["legacy"]
