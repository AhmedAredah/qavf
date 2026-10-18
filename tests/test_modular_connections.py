"""SQL ownership remains bounded and isolated after extraction from DataContext."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import duckdb
import polars as pl
import pytest

from qavf import DataContext
from qavf.engine.connections import merge_connection_binding
from qavf.settings import ConnectionSpec
from qavf.sources.base import SourceEnvironment, SourceIntegrityError
from qavf.sources.parameters import QueryParams
from qavf.sources.query import QuerySource


@pytest.mark.parametrize("sample", [False, True])
def test_provenance_failure_after_native_binding_releases_its_view(monkeypatch, sample):
    source = QuerySource(QueryParams(sql="SELECT 1 AS x"), SourceEnvironment())

    def fail(*_):
        raise SourceIntegrityError("identity changed during provenance")

    with DataContext(sources={"data": source}) as ctx:
        monkeypatch.setattr(ctx, "_fingerprint", fail)
        with pytest.raises(SourceIntegrityError, match="identity changed"):
            if sample:
                ctx.sql_sample("SELECT * FROM data", 1)
            else:
                ctx.sql("SELECT * FROM data")
        with pytest.raises(duckdb.CatalogException):
            ctx._connection([]).execute("SELECT * FROM data")
        assert ctx.resource_usage()["active_sql_registration_bytes_upper_bound"] == 0


def test_same_attachment_alias_rebinds_when_source_environment_changes(tmp_path):
    roots = [tmp_path / name for name in ("first", "second")]
    for value, root in enumerate(roots, start=1):
        root.mkdir()
        with duckdb.connect(str(root / "data.duckdb")) as con:
            con.execute(f"CREATE TABLE data AS SELECT {value} AS x")
    spec = ConnectionSpec(type="duckdb", path="data.duckdb")
    with DataContext() as ctx:
        handles = []
        for expected, root in enumerate(roots, start=1):
            bindings = {}
            merge_connection_binding(
                bindings,
                "external",
                SourceEnvironment(base_dir=root, connections={"external": spec}),
            )
            con = ctx._connection(["external"], bindings)
            handles.append(con)
            assert con.execute("SELECT x FROM external.data").fetchone() == (expected,)
            assert ctx._connection(["external"], bindings) is con
            assert ctx._sql_connections.open_count == 1
        with pytest.raises(duckdb.ConnectionException):
            handles[0].execute("SELECT 1")
    assert ctx._sql_connections.open_count == 0


def test_one_worker_cleanup_keeps_another_workers_registration():
    bound = threading.Barrier(2)
    finished = threading.Event()
    with DataContext() as ctx:
        owner = ctx._sql_connections

        def first():
            with owner.operation():
                con = owner.connection([])
                owner.register_frame(con, "data", pl.DataFrame({"x": [1]}))
                bound.wait(timeout=5)
            ctx.finish_worker()
            finished.set()

        def second():
            with owner.operation():
                con = owner.connection([])
                owner.register_frame(con, "data", pl.DataFrame({"x": [2]}))
                bound.wait(timeout=5)
                assert finished.wait(5)
                assert con.execute("SELECT x FROM data").fetchone() == (2,)
                assert owner.open_count == 1
            ctx.finish_worker()

        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(first), pool.submit(second)]
            for job in jobs:
                job.result(timeout=10)
        assert owner.open_count == 0
        assert owner.usage()["active_sql_registration_bytes_upper_bound"] == 0
        assert owner.usage()["sql_frame_registrations"] == 2
