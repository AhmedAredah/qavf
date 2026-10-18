"""Source registration and connection ownership across the modular SQL family."""

from __future__ import annotations

from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from qavf.settings import ConnectionSpec
from qavf.sources import (
    DatabaseParams,
    DatabaseSource,
    FileParams,
    FileSource,
    PythonParams,
    PythonSource,
    QueryParams,
    QuerySource,
    SourceEnvironment,
    SourceError,
    SourceRegistry,
)
from qavf.sources.native import connect


@pytest.fixture
def environment(tmp_path):
    path = tmp_path / "source.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE readings AS SELECT * FROM range(3)")
    return SourceEnvironment(
        base_dir=tmp_path,
        connections={
            "local": ConnectionSpec(type="duckdb", path=str(path), snapshot=True),
        },
    )


def sql_source(family, env, budget=None):
    if family == "query":
        return QuerySource(QueryParams(sql="SELECT * FROM local.readings", row_budget=budget), env)
    return DatabaseSource(
        DatabaseParams(connection="local", table="readings", row_budget=budget), env
    )


def test_discovery_builds_each_public_builtin_once(tmp_path):
    registry = SourceRegistry().discover(use_entry_points=False)
    expected = {
        "file": (FileSource, FileParams, {"path": "readings.csv"}),
        "query": (QuerySource, QueryParams, {"sql": "SELECT 1"}),
        "database": (
            DatabaseSource,
            DatabaseParams,
            {"connection": "local", "table": "readings"},
        ),
        "python": (PythonSource, PythonParams, {"callable": "package.module:read"}),
    }
    assert len(registry) == len(expected)
    for name, (source_type, parameter_type, config) in expected.items():
        assert registry.get(name) is source_type
        source = registry.build({"type": name, **config}, SourceEnvironment(base_dir=tmp_path))
        assert type(source) is source_type
        assert type(source.params) is parameter_type
        rebound = source.bind(SourceEnvironment(threads=1))
        assert type(rebound) is source_type
        assert rebound is not source
        assert rebound.env.base_dir == tmp_path


def test_builtin_registration_conflict_leaves_registry_unchanged():
    class AlternateQuerySource(QuerySource):
        pass

    registry = SourceRegistry()
    registry.register(AlternateQuerySource)
    with pytest.raises(ValueError, match="duplicate source type"):
        registry.discover(use_entry_points=False)
    assert list(registry) == [AlternateQuerySource]


@pytest.mark.parametrize("family", ["query", "database"])
@pytest.mark.parametrize("budget", [None, 1])
def test_sql_load_closes_owned_connection_on_success_or_budget_failure(family, budget, environment):
    source = sql_source(family, environment, budget)
    connections = []

    def tracked(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    with patch("qavf.sources.query.connect", tracked):
        if budget is None:
            assert source.load().equals(pl.DataFrame({"range": [0, 1, 2]}))
        else:
            with pytest.raises(SourceError, match="row_budget"):
                source.load()
    assert len(connections) == 1
    with pytest.raises(duckdb.ConnectionException):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("family", ["query", "database"])
@pytest.mark.parametrize("budget", [None, 1])
def test_sql_snapshot_read_keeps_borrowed_connection_open(family, budget, environment):
    source = sql_source(family, environment, budget)
    assert source.snapshot_connection() == "local"
    with connect(environment, ["local"]) as connection:
        if budget is None:
            assert source.read(connection).height == 3
        else:
            with pytest.raises(SourceError, match="row_budget"):
                source.read(connection)
        assert connection.execute("SELECT 7").fetchone() == (7,)


@pytest.mark.parametrize("family", ["query", "database"])
def test_invalid_sql_is_rejected_before_allocating_connection(family, environment):
    sql = "SELECT 1; SELECT 2"
    source = (
        QuerySource(QueryParams(sql=sql), environment)
        if family == "query"
        else DatabaseSource(DatabaseParams(connection="local", query=sql), environment)
    )
    with (
        patch("qavf.sources.query.connect", side_effect=AssertionError("unexpected allocation")),
        pytest.raises(SourceError),
    ):
        source.load()
