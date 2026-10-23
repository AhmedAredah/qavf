"""Shared projection ownership, failure isolation and last-consumer cleanup."""

import contextlib
import gc
import weakref

import polars as pl
import pytest

from qavf.engine.context import DataContext


def lazy(frame, calls):
    def read(value):
        calls.append(1)
        return value

    return frame.lazy().map_batches(
        read,
        schema=dict(frame.schema),
        projection_pushdown=False,
        predicate_pushdown=False,
        slice_pushdown=False,
    )


def test_plan_unions_once_and_releases_each_input_at_its_last_contract(tmp_path, monkeypatch):
    calls, opened = {"a": [], "b": []}, []
    frame = pl.DataFrame({"x": [1, 2], "y": [3, 4], "unused": ["secret"] * 2})
    with DataContext(
        frames={name: lazy(frame, seen) for name, seen in calls.items()},
        cache_max_bytes=0,
        temp_directory=tmp_path,
    ) as ctx:
        original = ctx.observe

        @contextlib.contextmanager
        def observe(name, columns=None):
            opened.append((name, tuple(columns)))
            with original(name, columns) as observation:
                yield observation

        monkeypatch.setattr(ctx, "observe", observe)
        with ctx.observation_plan([{"a": ["x"]}, {"b": ["x"]}, {"a": ["y"]}]) as plan:
            assert opened == []
            with plan.contract(0) as first:
                view = first["a"]
                assert view.schema() == {"x": pl.Int64}
                assert view.reduce(lambda data: data.select(pl.sum("x"))).item() == 3
                assert all(batch.columns == ["x"] for batch in view.batches(1))
                retained_a = ctx.resource_usage()["owned_spill_bytes"]
                assert retained_a > 0
            assert ctx.resource_usage()["owned_spill_bytes"] == retained_a
            with pytest.raises(RuntimeError, match="ended"):
                view.scan()
            with plan.contract(1) as second:
                assert second["b"].schema() == {"x": pl.Int64}
                assert ctx.resource_usage()["owned_spill_bytes"] > retained_a
            assert ctx.resource_usage()["owned_spill_bytes"] == retained_a
            with plan.contract(2) as third:
                assert third["a"].reduce(lambda data: data.select(pl.sum("y"))).item() == 7
            assert ctx.resource_usage()["owned_spill_bytes"] == 0
        assert opened == [("a", ("x", "y")), ("b", ("x",))]
        assert calls == {"a": [1], "b": [1]}
    assert not list(tmp_path.iterdir())


def test_failure_cache_drops_tracebacks_and_allows_unrelated_input(tmp_path, monkeypatch):
    calls, references = [], []

    class Payload:
        pass

    with DataContext(
        frames={"good": pl.DataFrame({"x": [1]}).lazy()}, temp_directory=tmp_path
    ) as ctx:
        original = ctx.observe

        @contextlib.contextmanager
        def observe(name, columns=None):
            calls.append(name)
            if name == "bad":
                payload = Payload()
                references.append(weakref.ref(payload))
                raise RuntimeError("failed input")
            with original(name, columns) as observation:
                yield observation

        monkeypatch.setattr(ctx, "observe", observe)
        with ctx.observation_plan([{"bad": ["x"]}, {"good": ["x"]}, {"bad": ["x"]}]) as plan:
            for index in (0, 2):
                with plan.contract(index) as observed, pytest.raises(
                    RuntimeError, match="failed input"
                ):
                    observed["bad"]
                gc.collect()
                assert references[0]() is None
            with plan.contract(1) as observed:
                assert observed["good"].reduce(lambda data: data).item() == 1
            assert ctx.resource_usage()["owned_spill_bytes"] == 0
        assert calls == ["bad", "good"]
    assert not list(tmp_path.iterdir())


def test_view_close_never_closes_shared_parent_and_unused_contracts_do_not_read(tmp_path):
    calls = []
    with DataContext(
        frames={"a": lazy(pl.DataFrame({"x": [1]}), calls)},
        cache_max_bytes=0,
        temp_directory=tmp_path,
    ) as ctx:
        with ctx.observation_plan([{"a": ["x"]}, {"a": ["x"]}, {"absent": ["x"]}]) as plan:
            with plan.contract(0) as observed:
                observed["a"].close()
            assert ctx.resource_usage()["owned_spill_bytes"] > 0
            with plan.contract(1) as observed:
                assert observed["a"].reduce(lambda data: data).item() == 1
            assert ctx.resource_usage()["owned_spill_bytes"] == 0
        assert calls == [1]
    assert not list(tmp_path.iterdir())


def test_plan_cleanup_propagates_base_exception_and_invalidates_open_views(tmp_path):
    class Stop(BaseException):
        pass

    with DataContext(
        frames={"a": pl.DataFrame({"x": [1]}).lazy()}, temp_directory=tmp_path
    ) as ctx:
        with (
            pytest.raises(Stop),
            ctx.observation_plan([{"a": ["x"]}, {"a": ["x"]}]) as plan,
            plan.contract(0) as observed,
        ):
            view = observed["a"]
            raise Stop()
        assert ctx.resource_usage()["owned_spill_bytes"] == 0
        with pytest.raises(RuntimeError, match="ended"):
            view.scan()
    assert not list(tmp_path.iterdir())
