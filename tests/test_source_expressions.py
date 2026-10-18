"""Complete source-expression, identity, ownership, and execution-policy contracts."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.session import RunSession
from qavf.sources import SourceEnvironment, SourceError, SourceIntegrityError, SourceParams
from qavf.sources.base import DataSource, SourceNotFoundError, SourceRegistry
from qavf.sources.execution import execute_with_source_policy
from qavf.sources.file import FileSource
from qavf.sources.native import connect
from qavf.sources.parameters import DatabaseParams, FileParams, QueryParams
from qavf.sources.query import DatabaseSource, QuerySource
from qavf.sources.sql import normalize_sql_paths, resolve_sql_paths


@pytest.mark.parametrize("argument", [
    "['a.csv'] || ['b.csv']", "['a.csv'][1]", "['a.csv']::VARCHAR[]",
    "'a' || '.csv'", "list_concat(['a.csv'], ['b.csv'])",
])
def test_complete_dynamic_file_argument_rejected_atomically(argument, tmp_path):
    with pytest.raises(SourceError, match="dynamic"):
        normalize_sql_paths(f"SELECT * FROM read_csv({argument})", tmp_path)


@pytest.mark.parametrize("argument,name", [
    ("E'a\\tb.csv'", "a\tb.csv"),
    ("E'a\\\'b.csv'", "a'b.csv"),
    ("$tag$a]b.csv$tag$", "a]b.csv"),
    ("'é.csv'", "é.csv"),
    ("['a.csv']", "a.csv"),
])
def test_literal_forms_resolve_to_configuration_root(argument, name, tmp_path):
    if os.name == "nt" and any(ord(character) < 32 for character in name):
        pytest.skip("Windows forbids control characters in literal file names")
    pl.DataFrame({"x": [7]}).write_csv(tmp_path / name)
    source = QuerySource(
        QueryParams(sql=f"SELECT 'é' AS label, x FROM main.read_csv({argument})"),
        SourceEnvironment(base_dir=tmp_path),
    )
    assert source.load().to_dicts() == [{"label": "é", "x": 7}]


@pytest.mark.parametrize("value", ["a.csv", Path("a.csv"), ["a.csv", Path("b.csv")]])
def test_named_paths_are_bound_and_input_mapping_is_unchanged(value, tmp_path):
    parameters = {"path": value, "label": "read_csv('ignore.csv')"}
    sql = "SELECT $label AS label, * FROM read_csv($path)"
    normalized, bound = normalize_sql_paths(sql, tmp_path, parameters)
    expected = (
        [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]
        if isinstance(value, list) else str(tmp_path / "a.csv")
    )
    assert normalized == sql
    assert bound["path"] == expected
    assert parameters["path"] == value
    assert bound["label"] == parameters["label"]


def test_url_parameter_and_nonpath_literals_remain_unchanged(tmp_path):
    uri = "https://example.org/a.csv?signature=secret"
    sql = "SELECT * FROM read_csv($path)"
    assert normalize_sql_paths(sql, tmp_path, {"path": uri}) == (sql, {"path": uri})
    literal = "SELECT $$read_csv('data.csv')$$ AS x"
    assert resolve_sql_paths(literal, tmp_path) == literal


def test_file_parameter_names_follow_duckdb_case_insensitivity(tmp_path):
    sql = "SELECT * FROM read_csv($PaTh)"
    assert normalize_sql_paths(sql, tmp_path, {"path": "a.csv"}) == (
        sql, {"path": str(tmp_path / "a.csv")}
    )


def test_borrowed_source_binding_is_independent_and_refreshes_manifest(tmp_path):
    path = tmp_path / "a.csv"
    pl.DataFrame({"x": [1]}).write_csv(path)
    env = SourceEnvironment(base_dir=tmp_path)
    source = FileSource(FileParams(path="*.csv"), env)
    with (
        DataContext(sources={"d": source}) as first,
        DataContext(sources={"d": source}, env=SourceEnvironment(base_dir=Path("/"))) as second,
    ):
        assert first.get("d").to_dicts() == second.get("d").to_dicts() == [{"x": 1}]
        assert first._catalog.sources["d"] is not second._catalog.sources["d"]
        assert first._catalog.sources["d"].env.connection_budget is first._resources
        assert second._catalog.sources["d"].env.connection_budget is second._resources
    assert source.env is env and not source.load_notes and not hasattr(source, "_manifest")
    pl.DataFrame({"x": [2]}).write_csv(tmp_path / "b.csv")
    with DataContext(sources={"d": source}) as third:
        assert third.get("d")["x"].to_list() == [1, 2]


def test_parallel_bindings_keep_their_own_lifecycle(tmp_path):
    pl.DataFrame({"x": [1]}).write_csv(tmp_path / "data.csv")
    source = FileSource(FileParams(path="data.csv"), SourceEnvironment(base_dir=tmp_path))
    barrier = threading.Barrier(2)
    seen = []
    errors = []

    def run():
        try:
            with DataContext(sources={"d": source}) as ctx:
                barrier.wait(timeout=5)
                seen.append((ctx.get("d").item(), id(ctx._catalog.sources["d"])))
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=run) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not errors and len(seen) == 2
    assert [value for value, _ in seen] == [1, 1]
    assert len({identity for _, identity in seen}) == 2
    assert source.env.connection_budget is None


def test_stateful_plugin_must_define_binding_without_implicit_copy():
    class Stateful(DataSource):
        kind = "stateful"
        Params = SourceParams

        def __init__(self):
            super().__init__(SourceParams(type=self.kind), SourceEnvironment())

        def load(self):
            return pl.DataFrame({"x": [1]})

    with pytest.raises(SourceError, match=r"implement bind\(env\)"):
        Stateful().bind(SourceEnvironment())


def test_binding_preserves_parameter_aliases():
    source = DatabaseSource(
        DatabaseParams(connection="w", table="readings", schema="public"), SourceEnvironment()
    )
    bound = source.bind(SourceEnvironment())
    assert bound is not source and bound.params.db_schema == "public"


def test_changed_native_file_invalidates_identity_instead_of_reusing_hash(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    with DataContext(sources={"d": source}, hash_inputs=True) as ctx:
        assert ctx.sql("SELECT sum(x) FROM d").item() == 1
        assert ctx.describe()[0].fingerprint["identity"] == "content"
        path.write_text("x\n1234\n")
        with pytest.raises(SourceIntegrityError, match="changed"):
            ctx.sql("SELECT sum(x) FROM d")


def test_identity_is_verified_after_operation_even_without_native_reader(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())

    def changed():
        path.write_text("x\n1234\n")
        return "would otherwise pass"

    with pytest.raises(SourceIntegrityError, match="changed"):
        execute_with_source_policy([source], changed)


def test_native_and_eager_transient_retries_share_policy(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    params = FileParams(path=str(path), retries=2, retry_delay_s=0)
    with DataContext(sources={"d": FileSource(params, SourceEnvironment())}) as ctx:
        con = ctx._connection([])
        native_calls = []

        class FlakyBinding:
            def __getattr__(self, name):
                return getattr(con, name)

            def sql(self, *args, **kwargs):
                native_calls.append(1)
                if len(native_calls) == 1:
                    raise duckdb.HTTPException("HTTP Error: 503 Service Unavailable")
                return con.sql(*args, **kwargs)

        with patch.object(ctx, "_connection", return_value=FlakyBinding()):
            assert ctx.has("d")
        assert len(native_calls) == 2
        assert ctx._catalog.sources["d"].load_notes["retries_used"] == 1
    with DataContext(sources={"d": FileSource(params, SourceEnvironment())}) as ctx:
        source = ctx._catalog.sources["d"]
        original = source.load
        eager_calls = []

        def flaky():
            eager_calls.append(1)
            if len(eager_calls) == 1:
                raise SourceError("temporary", retryable=True)
            return original()

        with patch.object(source, "load", side_effect=flaky):
            assert ctx.get("d").height == 1
        assert len(eager_calls) == 2 and source.load_notes["retries_used"] == 1


@pytest.mark.parametrize("error", [
    SourceError("permanent"), SourceNotFoundError("absent", retryable=True),
])
def test_permanent_and_missing_source_failures_do_not_retry(error):
    source = QuerySource(
        QueryParams(sql="SELECT 1", retries=2, retry_delay_s=0), SourceEnvironment()
    )
    calls = []

    def fail():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        execute_with_source_policy([source], fail)
    assert calls == [1]


def test_connection_lease_registers_and_releases_on_registration_error():
    class Owner:
        def __init__(self):
            self.events = []

        def acquire(self):
            return "token", 0, None

        def register_connection(self, token, connection):
            self.events.append(("register", token))
            self.connection = connection
            raise RuntimeError("cancelled during registration")

        def unregister_connection(self, token):
            self.events.append(("unregister", token))

        def release(self, token):
            self.events.append(("release", token))

    owner = Owner()
    with pytest.raises(RuntimeError, match="cancelled"):
        connect(SourceEnvironment(connection_budget=owner))
    assert owner.events == [("register", "token"), ("unregister", "token"), ("release", "token")]
    assert owner.connection._closed


@pytest.mark.parametrize("dtype", [pl.Duration("ns"), pl.List(pl.Duration("us"))])
def test_parquet_duration_preserves_logical_type_and_projection(dtype, tmp_path):
    values = [1, 2] if dtype == pl.Duration("ns") else [[1, 2], [3]]
    expected = pl.DataFrame({"d": pl.Series(values, dtype=dtype), "unused": [7, 8]})
    path = tmp_path / "duration.parquet"
    expected.write_parquet(path)
    source = FileSource(FileParams(path=str(path), columns=["d"]), SourceEnvironment())
    assert source.scan() is None
    assert source.load().equals(expected.select("d"))
    limited = FileSource(FileParams(path=str(path), row_budget=1), SourceEnvironment())
    with pytest.raises(SourceError, match="row_budget"):
        limited.load()


def test_failed_source_module_does_not_publish_earlier_classes():
    module = ModuleType("failed_source_module")
    good = type("Good", (QuerySource,), {"__module__": module.__name__, "kind": "fresh"})
    collision = type("Collision", (QuerySource,), {"__module__": module.__name__})
    module.Good = good
    module.Collision = collision
    registry = SourceRegistry()
    registry.register(QuerySource)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register_module(module)
    assert "fresh" not in registry


def test_retry_unsafe_source_never_replays_and_cancellation_stops_backoff():
    source = QuerySource(QueryParams(sql="SELECT 1", retries=2), SourceEnvironment())
    calls = []

    def operation():
        calls.append(1)
        raise SourceError("temporary", retryable=True)

    source.retry_safe = False
    with pytest.raises(SourceError, match="temporary"):
        execute_with_source_policy([source], operation)
    assert calls == [1]
    source.retry_safe = True
    checks = []

    def cancelled():
        checks.append(1)
        if len(checks) > 1:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        execute_with_source_policy([source], operation, check_cancelled=cancelled)
    assert calls == [1, 1]


def test_old_cancellation_cannot_cross_reused_worker_invocation():
    session = RunSession()
    ready, advance, advanced = threading.Event(), threading.Event(), threading.Event()
    interrupting, interrupted, finished = threading.Event(), threading.Event(), threading.Event()
    invocations = []

    class PausedInterrupt:
        def interrupt(self):
            interrupting.set()
            assert interrupted.wait(5)

    def worker():
        invocations.append(session.begin())
        session.register_connection("reusable", PausedInterrupt())
        ready.set()
        assert advance.wait(5)
        session.end()
        invocations.append(session.begin())
        advanced.set()
        assert finished.wait(5)
        session.end()

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(5)
    cancellation = threading.Thread(target=session.cancel, args=(invocations[0],))
    cancellation.start()
    try:
        assert interrupting.wait(5)
        advance.set()
        # A pending old interrupt must finish before this worker can start another call.
        assert not advanced.wait(0.05)
    finally:
        interrupted.set()
        cancellation.join(5)
        assert advanced.wait(5)
        finished.set()
        thread.join(5)
    assert len(invocations) == 2 and not invocations[1].cancelled.is_set()
