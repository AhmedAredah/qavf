"""Projection ownership and operation-scoped streaming policy contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from qavf import DataContext
from qavf.engine.context import DatasetNotAvailableError
from qavf.engine.resources import ResourceLimitError
from qavf.engine.sql_read import NativeReadUnavailable
from qavf.sources.base import SourceEnvironment, SourceError, SourceIntegrityError
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource


def query_source(*, row_budget=None, retries=0):
    return QuerySource(
        QueryParams(
            sql="SELECT i AS x, repeat('payload', 1000) AS wide FROM range(9) t(i)",
            row_budget=row_budget,
            retries=retries,
            retry_delay_s=0,
        ),
        SourceEnvironment(threads=1),
    )


def fail_full(*args, **kwargs):
    raise AssertionError("full dataset load is forbidden for this operation")


@pytest.mark.parametrize("backend", ["native", "lazy", "cached", "spilled"])
def test_project_prunes_without_full_get_and_preserves_order(backend, monkeypatch):
    expected = pl.DataFrame({"x": list(range(9)), "wide": ["payload" * 1000] * 9})
    kwargs = (
        {"sources": {"s": query_source()}}
        if backend == "native"
        else {"frames": {"s": expected.lazy() if backend == "lazy" else expected}}
    )
    if backend == "spilled":
        kwargs["cache_max_bytes"] = 0
    with DataContext(**kwargs) as ctx:
        monkeypatch.setattr(ctx, "get", fail_full)
        result = ctx.project("s", ["x", "x"])
        assert_frame_equal(result, expected.select("x"))
        assert ctx.resource_usage()["frame_spill_promotions"] == 0
        if backend == "spilled":
            assert ctx.resource_usage()["frame_spill_read_count"] == 0
            assert ctx.resource_usage()["frame_spill_read_bytes"] == 0
            assert ctx.resource_usage()["frame_spill_project_count"] == 1
            assert ctx.resource_usage()["frame_spill_projected_read_bytes"] is None


def test_owned_spill_selected_types_and_results_survive_cleanup(monkeypatch):
    frame = pl.DataFrame(
        {
            "huge": pl.Series([2**100, -(2**100)], dtype=pl.Int128),
            "enum": pl.Series(["b", "a"], dtype=pl.Enum(["a", "b", "unused"])),
            "nested": [[1, 2], None],
            "payload": ["a" * 10000, "b" * 10000],
        }
    )
    with DataContext(frames={"s": frame}, cache_max_bytes=0) as ctx:
        path = ctx._frames.paths["s"]
        monkeypatch.setattr(ctx, "get", fail_full)
        result = ctx.project("s", ["enum", "huge", "nested"])
        reduced = ctx.reduce("s", lambda plan: plan.select(pl.col("huge").max()))
        assert path.exists()
        assert ctx._frames.spill_promotions == 0
        assert ctx._frames.spill_read_count == 0
        assert ctx._frames.spill_reduce_count == 1
    assert not path.exists()
    assert_frame_equal(result, frame.select("enum", "huge", "nested"))
    assert reduced.item() == 2**100


def test_reduce_holds_owned_file_until_collection_ends():
    entered, release_attempted = Event(), Event()
    with DataContext(frames={"s": pl.DataFrame({"x": [1, 2]})}, cache_max_bytes=0) as ctx:
        path = ctx._frames.paths["s"]

        def build(plan):
            entered.set()
            assert release_attempted.wait(5)
            assert path.exists()
            return plan.select(pl.col("x").sum())

        def release():
            assert entered.wait(5)
            release_attempted.set()
            ctx.release(["s"])

        with ThreadPoolExecutor(max_workers=2) as pool:
            read = pool.submit(ctx.reduce, "s", build)
            cleanup = pool.submit(release)
            assert read.result(timeout=10).item() == 3
            cleanup.result(timeout=10)
        assert not path.exists()
        with pytest.raises(DatasetNotAvailableError, match="released"):
            ctx.project("s", ["x"])


def test_project_checks_selected_materialization_target():
    frame = pl.DataFrame({"x": range(100), "wide": ["x" * 10000] * 100})
    with DataContext(frames={"s": frame.lazy()}, memory_budget_bytes=5000) as ctx:
        assert ctx.project("s", ["x"]).height == 100
        with pytest.raises(ResourceLimitError, match="materialized"):
            ctx.project("s", ["wide"])


def test_native_batch_reader_shape_and_metadata_without_full_load(monkeypatch):
    with DataContext(sources={"s": query_source()}) as ctx:
        monkeypatch.setattr(ctx, "get", fail_full)
        sessions = []

        def consume(session):
            sessions.append(session)
            assert session.schema('SELECT "x" FROM s') == {"x": pl.Int64}
            assert session.frame("SELECT count(*) FROM s").item() == 9
            parts = []
            for batch in session.batches("SELECT x FROM s ORDER BY x", batch_rows=2):
                assert batch.width == 1
                assert 0 < batch.height <= 2
                parts.extend(batch["x"].to_list())
            return parts

        assert ctx.consume_sql(lambda: consume, datasets=["s"]) == list(range(9))
        assert ctx.resource_usage()["active_connection_reservations"] == 0
        assert not ctx.is_materialized("s")
        with pytest.raises(RuntimeError, match="ended"):
            sessions[0].frame("SELECT 1")


def test_partial_attempt_retry_resets_consumer_and_closes_connection():
    with DataContext(sources={"s": query_source(retries=1)}) as ctx:
        attempts = []

        def factory():
            values = []
            attempts.append(values)

            def consume(session):
                for batch in session.batches("SELECT x FROM s ORDER BY x", 2):
                    values.extend(batch["x"].to_list())
                    if len(attempts) == 1:
                        raise SourceError("transient after first batch", retryable=True)
                return values

            return consume

        assert ctx.consume_sql(factory, datasets=["s"]) == list(range(9))
        assert attempts == [[0, 1], list(range(9))]
        assert ctx.resource_usage()["active_connection_reservations"] == 0
        assert ctx._catalog.sources["s"].load_notes["retries_used"] == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt, ValueError])
def test_batch_consumer_errors_close_owned_connection_and_propagate(error):
    with DataContext(sources={"s": query_source(retries=2)}) as ctx:
        factories = []

        def factory():
            factories.append(1)

            def consume(session):
                for _ in session.batches("SELECT x FROM s", 2):
                    raise error("stop")

            return consume

        with pytest.raises(error, match="stop"):
            ctx.consume_sql(factory, datasets=["s"])
        assert factories == [1]
        assert ctx.resource_usage()["active_connection_reservations"] == 0


def test_incomplete_or_escaped_reader_cannot_publish_success():
    with DataContext(sources={"s": query_source()}) as ctx:
        retained = []

        def consume(session):
            iterator = session.batches("SELECT x FROM s", 2)
            retained.append(iterator)
            next(iterator)
            return "partial success"

        with pytest.raises(RuntimeError, match="must exhaust"):
            ctx.consume_sql(lambda: consume, datasets=["s"])
        assert ctx.resource_usage()["active_connection_reservations"] == 0
        retained[0].close()
        escaped = ctx.consume_sql(
            lambda: lambda session: session.batches("SELECT x FROM s"), datasets=["s"]
        )
        with pytest.raises(RuntimeError, match="ended"):
            next(escaped)


def test_stream_complete_source_budget_precedes_filtered_result():
    with DataContext(sources={"s": query_source(row_budget=2)}) as ctx:
        called = []
        with pytest.raises(SourceError, match="row_budget"):
            ctx.consume_sql(
                lambda: lambda session: called.append(session.frame("SELECT x FROM s LIMIT 1")),
                datasets=["s"],
            )
        assert called == []
        assert ctx.resource_usage()["active_connection_reservations"] == 0


@pytest.mark.parametrize("kind", ["lazy", "typed_file", "cached"])
def test_native_capability_rejection_never_eager_loads(kind, tmp_path, monkeypatch):
    frame = pl.DataFrame({"category": pl.Series(["a"], dtype=pl.Enum(["a", "b"]))})
    if kind == "typed_file":
        path = tmp_path / "typed.parquet"
        frame.write_parquet(path)
        kwargs = {"sources": {"s": FileSource(FileParams(path=str(path)), SourceEnvironment())}}
    else:
        kwargs = {"frames": {"s": frame.lazy() if kind == "lazy" else frame}}
    with DataContext(**kwargs) as ctx:
        monkeypatch.setattr(ctx, "get", fail_full)
        with pytest.raises(NativeReadUnavailable):
            ctx.consume_sql(
                lambda: lambda session: session.frame("SELECT * FROM s"), datasets=["s"]
            )
        assert ctx.resource_usage()["active_connection_reservations"] == 0


def test_stream_identity_change_after_last_batch_never_publishes(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n2\n")
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    with DataContext(sources={"s": source}, hash_inputs=True) as ctx:

        def consume(session):
            assert sum(batch.height for batch in session.batches("SELECT x FROM s", 1)) == 2
            path.write_text("x\n3\n4\n")
            return "must not publish"

        with pytest.raises(SourceIntegrityError, match="changed"):
            ctx.consume_sql(lambda: consume, datasets=["s"])
        assert ctx.resource_usage()["active_connection_reservations"] == 0
        assert ctx._catalog.schema("s") is None


@pytest.mark.parametrize("typed", [False, True])
def test_ad_hoc_view_is_scoped_projected_and_uses_one_source(tmp_path, monkeypatch, typed):
    path = tmp_path / "data.parquet"
    dtype = pl.Enum(["a", "b"]) if typed else pl.String
    frame = pl.DataFrame({"key": pl.Series(["a", "b"], dtype=dtype), "wide": ["x" * 10000] * 2})
    frame.write_parquet(path)
    views = []
    with DataContext() as ctx:
        monkeypatch.setattr(FileSource, "load", fail_full)

        def consume(view):
            views.append(view)
            assert view.schema() == dict(frame.schema)
            assert view._source.env.connection_budget is ctx._resources
            result = view.project(["key"])
            fingerprint = view.fingerprint()
            return result, fingerprint

        projected, fingerprint = ctx.read_source({"type": "file", "path": str(path)}, consume)
        assert_frame_equal(projected, frame.select("key"))
        assert fingerprint is not None
        assert ctx.resource_usage()["active_connection_reservations"] == 0
        assert ctx.declared() == []
        with pytest.raises(RuntimeError, match="ended"):
            views[0].project(["key"])


def test_projected_ipc_passes_path_and_selected_columns_to_native_reader(monkeypatch):
    from pathlib import Path

    from qavf.engine import owned_ipc

    frame = pl.DataFrame({"x": range(5), "wide": ["y" * 10000] * 5})
    with DataContext(frames={"s": frame}, cache_max_bytes=0) as ctx:
        original = owned_ipc.pl.read_ipc
        calls = []

        def read(source, *, columns=None, **kwargs):
            assert isinstance(source, Path), "Python buffers would eagerly read every IPC column"
            assert columns == ["x"]
            calls.append(source)
            return original(source, columns=columns, **kwargs)

        monkeypatch.setattr(owned_ipc.pl, "read_ipc", read)
        assert ctx.project("s", ["x"]).height == 5
        assert len(calls) == 1


def test_batch_invocation_cancellation_cleans_reservation():
    with DataContext(sources={"s": query_source(retries=2)}) as ctx:
        invocation = ctx.begin_rule()
        calls = []

        def consume(session):
            for batch in session.batches("SELECT x FROM s", 2):
                calls.append(batch.height)
                ctx.cancel_rule(invocation)
            return "must not publish"

        try:
            with pytest.raises(ResourceLimitError, match="cancelled"):
                ctx.consume_sql(lambda: consume, datasets=["s"])
            assert calls == [2]
            assert ctx.resource_usage()["active_connection_reservations"] == 0
        finally:
            ctx.end_rule()


def test_ad_hoc_projection_cannot_hide_complete_typed_source_row_budget(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"category": pl.Series(["a"] * 3, dtype=pl.Enum(["a", "b"]))}).write_parquet(path)
    with DataContext() as ctx, pytest.raises(SourceError, match="row_budget"):
        ctx.read_source(
            {"type": "file", "path": str(path), "row_budget": 2},
            lambda view: view.project(["category"]),
        )


def test_ad_hoc_and_configured_participants_use_strictest_retry_policy():
    with DataContext(sources={"s": query_source(retries=2)}) as ctx:
        calls = []

        def factory():
            calls.append(1)

            def consume(session):
                for _ in session.batches("SELECT x FROM s", 1):
                    raise SourceError("transient but second source disallows retry", retryable=True)

            return consume

        with pytest.raises(SourceError, match="disallows retry"):
            ctx.consume_sql(
                factory,
                datasets=["s"],
                sources={"baseline": {"type": "query", "sql": "SELECT 1 AS x", "retries": 0}},
            )
        assert calls == [1]
        assert ctx.resource_usage()["active_connection_reservations"] == 0


@pytest.mark.parametrize("typed", [False, True])
def test_repeatable_scan_is_verified_local_file_capability(tmp_path, monkeypatch, typed):
    path = tmp_path / "data.parquet"
    dtype = pl.Enum(["a", "b"]) if typed else pl.String
    frame = pl.DataFrame({"x": pl.Series(["a", "b"], dtype=dtype)})
    frame.write_parquet(path)
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    with DataContext(sources={"s": source}, hash_inputs=True) as ctx:
        monkeypatch.setattr(ctx, "get", fail_full)
        assert ctx.repeatable_scan("s") is True
        assert ctx.is_materialized("s") is False
        assert_frame_equal(ctx.project("s", ["x"]), frame)
        frame.head(1).write_parquet(path)
        with pytest.raises(SourceIntegrityError, match="changed"):
            ctx.repeatable_scan("s")


def test_repeatable_scan_rejects_volatile_and_unverified_backends(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    local = FileSource(FileParams(path=str(path)), SourceEnvironment())

    class PluginFile(FileSource):
        def scan(self):
            raise AssertionError("a subclass cannot inherit a repeatability promise")

    sources = {
        "query": query_source(),
        "remote": FileSource(
            FileParams(path="https://example.invalid/data.csv"), SourceEnvironment()
        ),
        "plugin": PluginFile(FileParams(path=str(path)), SourceEnvironment()),
        "overridden": local,
    }
    with DataContext(
        frames={
            "lazy": pl.DataFrame({"x": [1]}).lazy(),
            "overridden": pl.DataFrame({"x": [2]}).lazy(),
        },
        sources=sources,
    ) as ctx:
        monkeypatch.setattr(ctx, "get", fail_full)
        monkeypatch.setattr(ctx, "collect", fail_full)
        for name in ("query", "remote", "plugin", "lazy", "overridden"):
            assert ctx.repeatable_scan(name) is False
        assert ctx.resource_usage()["active_connection_reservations"] == 0


def test_session_repeatability_uses_exact_bound_sources_for_both_alias_kinds(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n2\n")
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    sessions = []
    with DataContext(sources={"configured": source}) as ctx:
        assert ctx.repeatable_scan("configured") is True

        def consume(session):
            sessions.append(session)
            assert session.repeatable_scan("configured") is True
            assert session.repeatable_scan("baseline") is True
            assert session.repeatable_scan("volatile") is False
            return session.frame("SELECT count(*) FROM configured").item()

        assert (
            ctx.consume_sql(
                lambda: consume,
                datasets=["configured"],
                sources={
                    "baseline": {"type": "file", "path": str(path)},
                    "volatile": {"type": "query", "sql": "SELECT random() AS x"},
                },
            )
            == 2
        )
        with pytest.raises(RuntimeError, match="ended"):
            sessions[0].repeatable_scan("baseline")
