"""Native projection must survive binding without changing volatile sources."""

from __future__ import annotations

import sys
from unittest.mock import Mock

import duckdb
import polars as pl
import pytest

from qavf import DataContext
from qavf.engine import connections, context, sql_read
from qavf.engine.relations import bind_scan
from qavf.sources.base import ScanPlan, SourceEnvironment, SourceError
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource


def native_read(ctx, backend, query):
    if backend == "context":
        return ctx.sql(query, datasets=["s"])
    return ctx.consume_sql(lambda: lambda session: session.frame(query), datasets=["s"])


def file_source(path, *, row_budget=None):
    return FileSource(
        FileParams(path=str(path), row_budget=row_budget), SourceEnvironment(threads=1)
    )


@pytest.mark.parametrize("backend", ["context", "session"])
def test_file_projection_reaches_native_parquet_scan(tmp_path, monkeypatch, backend):
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"x": [1, 2, 3], "payload": ["large" * 1000] * 3}).write_parquet(path)

    def forbid_load(*args, **kwargs):
        raise AssertionError("a native projection must not load the full Python frame")

    plans = []

    def observed_binding(*args):
        relation = bind_scan(*args)
        plans.append(relation.project("x").filter("x > 1").explain())
        return relation

    monkeypatch.setattr(FileSource, "load", forbid_load)
    monkeypatch.setattr(
        context if backend == "context" else sql_read, "bind_scan", observed_binding
    )
    with DataContext(sources={"s": file_source(path)}) as ctx:
        assert native_read(ctx, backend, "SELECT x FROM s WHERE x > 1 ORDER BY x").rows() == [
            (2,),
            (3,),
        ]
        assert len(plans) == 1
        plan = plans[0]
        assert "READ_PARQUET" in plan or "PARQUET_SCAN" in plan
        assert "COLUMN_DATA_SCAN" not in plan
        assert "payload" not in plan
        assert "x>1" in plan.replace(" ", "")
        assert not ctx.is_materialized("s")


def test_local_relation_binding_and_schema_do_not_decode_discarded_values(tmp_path):
    # A side-effectful expression makes eager execution observable without a
    # timing threshold or dependence on OS file-cache accounting.
    calls = []

    def observed(value):
        calls.append(value)
        return value

    with duckdb.connect(config={"threads": 1}) as con:
        con.create_function("observed", observed, ["BIGINT"], "BIGINT", side_effects=True)
        relation = bind_scan(
            con,
            file_source(tmp_path / "local.parquet"),
            ScanPlan("SELECT i AS x, observed(i) AS payload FROM range(5) t(i)"),
        )
        assert calls == []
        assert relation.limit(0).pl().columns == ["x", "payload"]
        assert relation.project("x").fetchall() == [(i,) for i in range(5)]
        assert calls == []
        assert relation.project("payload").fetchall() == [(i,) for i in range(5)]
        assert calls == list(range(5))


class CustomFileSource(FileSource):
    """A custom reader cannot inherit the built-in repeatability guarantee."""


class PluginQuerySource(QuerySource):
    """A representative plugin source with its own query semantics."""


@pytest.mark.parametrize("kind", ["query", "plugin", "subclass", "remote", "parameters"])
def test_other_native_bindings_preserve_existing_parameter_semantics(tmp_path, kind):
    env = SourceEnvironment(threads=1)
    source = {
        "query": QuerySource(QueryParams(sql="SELECT 1"), env),
        "plugin": PluginQuerySource(QueryParams(sql="SELECT 1"), env),
        "subclass": CustomFileSource(FileParams(path=str(tmp_path / "local.parquet")), env),
        "remote": FileSource(FileParams(path="https://example.invalid/data.parquet"), env),
        "parameters": file_source(tmp_path / "local.parquet"),
    }[kind]
    plan = (
        ScanPlan("SELECT nextval('observation') + $offset AS x", {"offset": 2})
        if kind == "parameters"
        else ScanPlan("SELECT nextval('observation') AS x")
    )
    results = []
    for legacy in (True, False):
        with duckdb.connect(config={"threads": 1}) as con:
            con.execute("CREATE SEQUENCE observation")
            proxy = Mock(wraps=con)
            relation = (
                proxy.sql(plan.sql, params=dict(plan.parameters))
                if legacy
                else bind_scan(proxy, source, plan)
            )
            proxy.sql.assert_called_once_with(plan.sql, params=dict(plan.parameters))
            results.append(
                (
                    relation.fetchall(),
                    relation.fetchall(),
                    con.execute("SELECT nextval('observation')").fetchone(),
                )
            )
    assert results[0] == results[1]
    if kind == "parameters":
        assert results[1] == ([(3,)], [(3,)], (2,))


def legacy_empty_binding_materializes():
    # DuckDB 1.2 materializes params={}; newer versions can keep it lazy.
    # Preserve the installed driver's semantics rather than inventing a
    # cross-version snapshot guarantee for arbitrary query sources.
    with duckdb.connect(config={"threads": 1}) as con:
        con.execute("CREATE SEQUENCE observation")
        relation = con.sql("SELECT nextval('observation') AS x", params={})
        return relation.fetchall() == relation.fetchall()


@pytest.mark.parametrize("backend", ["context", "session"])
def test_volatile_query_budget_schema_and_reads_preserve_driver_semantics(monkeypatch, backend):
    materialized = legacy_empty_binding_materializes()
    connect = connections.connect

    def with_sequence(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.execute("CREATE SEQUENCE observation")
        return connection

    monkeypatch.setattr(connections, "connect", with_sequence)
    source = QuerySource(
        QueryParams(
            sql="SELECT i AS x, nextval('observation') AS value FROM range(5) t(i)",
            row_budget=5,
        ),
        SourceEnvironment(threads=1),
    )
    query = "SELECT a.value, b.value FROM s a JOIN s b ON a.x = b.x ORDER BY a.x"

    def check_observations(frame):
        values = sorted(value for row in frame.rows() for value in row)
        assert values == (sorted(list(range(1, 6)) * 2) if materialized else list(range(1, 11)))

    with DataContext(sources={"s": source}) as ctx:
        if backend == "context":
            check_observations(native_read(ctx, backend, query))
        else:

            def consume(session):
                assert session.schema("SELECT value FROM s") == {"value": pl.Int64}
                first = session.frame(query)
                check_observations(first)
                assert session.frame(query).equals(first) is materialized
                assert session.frame("SELECT nextval('observation')").item() == (
                    6 if materialized else 21
                )
                return first

            check_observations(ctx.consume_sql(lambda: consume, datasets=["s"]))


@pytest.mark.parametrize("backend", ["context", "session"])
@pytest.mark.parametrize("format", ["csv", "parquet"])
@pytest.mark.parametrize(
    "name",
    [
        "quote's",
        "close'); SELECT 99; --",
        "literal[1]",
        "日本語 file",
        *(
            pytest.param(
                name,
                marks=pytest.mark.skipif(
                    sys.platform == "win32",
                    reason="Windows forbids '?', '*' and newlines in file names",
                ),
            )
            for name in ("literal[1]?*", "日本語\nfile")
        ),
    ],
)
def test_generated_reader_literals_preserve_hostile_and_glob_filenames(
    tmp_path, backend, format, name
):
    path = tmp_path / f"{name}.{format}"
    frame = pl.DataFrame({"x": [7, 11]})
    getattr(frame, f"write_{format}")(path)
    with DataContext(sources={"s": file_source(path)}) as ctx:
        assert native_read(ctx, backend, "SELECT sum(x) AS total FROM s").item() == 18


@pytest.mark.parametrize("backend", ["context", "session"])
@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_generated_reader_literal_list_preserves_glob_manifest(tmp_path, backend, format):
    for number, name in enumerate(["part'1", "part[2]"]):
        frame = pl.DataFrame({"x": [number]})
        getattr(frame, f"write_{format}")(tmp_path / f"{name}.{format}")
    source = file_source(tmp_path / f"part*.{format}")
    with DataContext(sources={"s": source}) as ctx:
        assert native_read(ctx, backend, "SELECT x FROM s ORDER BY x").rows() == [(0,), (1,)]


@pytest.mark.parametrize("backend", ["context", "session"])
def test_complete_source_budget_precedes_projected_filtered_result(tmp_path, backend):
    path = tmp_path / "input.parquet"
    pl.DataFrame({"x": list(range(5)), "payload": ["unused"] * 5}).write_parquet(path)
    with (
        DataContext(sources={"s": file_source(path, row_budget=4)}) as ctx,
        pytest.raises(SourceError, match="row_budget"),
    ):
        native_read(ctx, backend, "SELECT count(*) FROM s WHERE x < 0")
    assert ctx._resources.usage()["active_connection_reservations"] == 0
    with DataContext(sources={"s": file_source(path, row_budget=5)}) as ctx:
        assert native_read(ctx, backend, "SELECT count(*) FROM s WHERE x < 0").item() == 0


def test_only_builtin_local_file_plans_embed_generated_paths(tmp_path):
    path = tmp_path / "data.csv"
    pl.DataFrame({"x": [1]}).write_csv(path)
    local = file_source(path)._plan()[0]
    subclass = CustomFileSource(FileParams(path=str(path)), SourceEnvironment())._plan()[0]
    remote = file_source("https://example.invalid/data.csv")._plan()[0]
    assert not local.parameters and "$path" not in local.sql
    assert subclass.parameters["path"] == [str(path)] and "$path" in subclass.sql
    assert remote.parameters["path"] == "https://example.invalid/data.csv" and "$path" in remote.sql
