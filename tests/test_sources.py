"""Data sources: files, queries, databases, Python callables and plugins."""

from __future__ import annotations

import sqlite3
import sys
import textwrap
from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.runner import ConfigurationError, QARunner
from qavf.io import read_table, write_table
from qavf.settings import ConnectionSpec
from qavf.sources import (
    SourceEnvironment,
    SourceError,
    SourceNotFoundError,
    SourceRegistry,
    infer_format,
)
from qavf.testing import make_settings

FRAME = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "value": [1.5, 2.5, None],
        "day": [date(2024, 1, d) for d in (1, 2, 3)],
    }
)


@pytest.fixture(scope="module")
def registry() -> SourceRegistry:
    return SourceRegistry().discover(use_entry_points=False)


def _load(registry: SourceRegistry, config: dict, base: Path, **connections) -> pl.DataFrame:
    env = SourceEnvironment(base_dir=base, connections=connections)
    return registry.build(config, env).load()


# ----------------------------------------------------------------------- files
@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".json", ".ndjson"])
def test_file_formats_round_trip(tmp_path, registry, suffix: str) -> None:
    write_table(FRAME, tmp_path / f"data{suffix}")
    loaded = _load(registry, {"type": "file", "path": f"data{suffix}"}, tmp_path)
    assert loaded.height == 3 and loaded["id"].to_list() == [1, 2, 3]
    assert loaded["value"].null_count() == 1


def test_excel_files(tmp_path, registry) -> None:
    target = tmp_path / "book.xlsx"
    with duckdb.connect() as con:
        try:
            con.execute("INSTALL excel; LOAD excel")
        except (duckdb.IOException, duckdb.HTTPException) as exc:  # pragma: no cover
            # Only a failed extension download is an environment limitation; a missing
            # function or format below is a real failure (DuckDB older than 1.2).
            pytest.skip(f"DuckDB excel extension could not be downloaded: {exc}")
        con.register("f", FRAME.select("id", "name"))
        con.execute(f"COPY f TO '{target.as_posix()}' (FORMAT xlsx, HEADER true)")
    loaded = _load(registry, {"type": "file", "path": "book.xlsx"}, tmp_path)
    assert loaded["name"].to_list() == ["a", "b", "c"]


def test_globs_options_and_column_selection(tmp_path, registry) -> None:
    for part in (1, 2):
        (tmp_path / f"part{part}.txt").write_text(f"id;label\n{part};x{part}\n")
    loaded = _load(
        registry,
        {
            "type": "file",
            "path": "part*.txt",
            "format": "csv",
            "options": {"delim": ";"},
            "columns": ["id"],
        },
        tmp_path,
    )
    assert loaded.columns == ["id"] and sorted(loaded["id"].to_list()) == [1, 2]


def test_missing_and_broken_files(tmp_path, registry) -> None:
    with pytest.raises(SourceNotFoundError):
        _load(registry, {"type": "file", "path": "absent.csv"}, tmp_path)
    with pytest.raises(SourceNotFoundError):
        _load(registry, {"type": "file", "path": "absent_*.csv"}, tmp_path)
    (tmp_path / "bad.parquet").write_text("not parquet")
    with pytest.raises(SourceError):
        _load(registry, {"type": "file", "path": "bad.parquet"}, tmp_path)
    with pytest.raises(ValueError, match="cannot infer"):
        infer_format("data.unknown")
    assert infer_format("x.csv.gz") == "csv"


# ----------------------------------------------------------------------- query
def test_query_resolves_paths_against_configuration_dir(tmp_path, registry) -> None:
    write_table(FRAME, tmp_path / "data.parquet")
    sql = (
        "SELECT name, value * 2 AS doubled FROM read_parquet('data.parquet') "
        "WHERE value IS NOT NULL"
    )
    loaded = _load(registry, {"type": "query", "sql": sql}, tmp_path)
    assert loaded.to_dicts() == [{"name": "a", "doubled": 3.0}, {"name": "b", "doubled": 5.0}]
    with pytest.raises(SourceError):
        _load(registry, {"type": "query", "sql": "SELEC nonsense"}, tmp_path)


# -------------------------------------------------------------------- database
def test_duckdb_database_table_and_query(tmp_path, registry) -> None:
    with duckdb.connect(str(tmp_path / "w.duckdb")) as con:
        con.register("f", FRAME)
        con.execute("CREATE SCHEMA s; CREATE TABLE s.items AS SELECT * FROM f")
    connection = ConnectionSpec(type="duckdb", path="w.duckdb")
    table = _load(
        registry,
        {"type": "database", "connection": "w", "table": "items", "schema": "s"},
        tmp_path,
        w=connection,
    )
    assert table.height == 3
    query = _load(
        registry,
        {"type": "database", "connection": "w", "query": "SELECT count(*) AS n FROM w.s.items"},
        tmp_path,
        w=connection,
    )
    assert query["n"].to_list() == [3]
    joined = _load(
        registry,
        {"type": "query", "sql": "SELECT max(id) AS m FROM w.s.items"},
        tmp_path,
        w=connection,
    )
    assert joined["m"].to_list() == [3]


def test_sqlite_database(tmp_path, registry) -> None:
    with sqlite3.connect(tmp_path / "s.db") as con:
        con.execute("CREATE TABLE t (id INTEGER, v REAL)")
        con.executemany("INSERT INTO t VALUES (?, ?)", [(1, 0.5), (2, 1.5)])
    connection = ConnectionSpec(type="sqlite", path="s.db")
    try:
        loaded = _load(
            registry,
            {"type": "database", "connection": "lite", "table": "t"},
            tmp_path,
            lite=connection,
        )
    except SourceError as exc:  # pragma: no cover - needs the sqlite extension
        pytest.skip(f"DuckDB sqlite extension unavailable: {exc}")
    assert loaded["v"].to_list() == [0.5, 1.5]


def test_database_params_and_connections_are_validated(tmp_path, registry) -> None:
    with pytest.raises(ValueError, match="exactly one of 'table' or 'query'"):
        registry.build({"type": "database", "connection": "x"}, SourceEnvironment())
    with pytest.raises(SourceError, match="unknown connection"):
        _load(registry, {"type": "database", "connection": "nope", "table": "t"}, tmp_path)
    with pytest.raises(ValueError, match="needs 'path'"):
        ConnectionSpec(type="sqlite")
    with pytest.raises(ValueError, match="needs 'uri'"):
        ConnectionSpec(type="postgres")
    sql = ConnectionSpec(type="mysql", uri="host=h user=u").attach_sql("m", SourceEnvironment())
    assert sql == "ATTACH 'host=h user=u' AS \"m\" (TYPE mysql, READ_ONLY)"


# ---------------------------------------------------------------------- python
def test_python_callable_source(tmp_path, registry, monkeypatch) -> None:
    (tmp_path / "loaders.py").write_text(
        textwrap.dedent(
            """
            import pandas as pd

            def make(n):
                return pd.DataFrame({"x": list(range(n))})

            def broken():
                raise RuntimeError("nope")
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    loaded = _load(
        registry, {"type": "python", "callable": "loaders:make", "kwargs": {"n": 4}}, tmp_path
    )
    assert loaded["x"].to_list() == [0, 1, 2, 3]
    with pytest.raises(SourceError, match="nope"):
        _load(registry, {"type": "python", "callable": "loaders:broken"}, tmp_path)
    with pytest.raises(SourceError, match=r"package.module:function"):
        _load(registry, {"type": "python", "callable": "loaders"}, tmp_path)
    sys.modules.pop("loaders", None)


# --------------------------------------------------------------------- plugins
def test_custom_source_via_plugin_module(tmp_path, monkeypatch) -> None:
    (tmp_path / "my_sources.py").write_text(
        textwrap.dedent(
            """
            from typing import ClassVar, Literal
            import polars as pl
            from qavf.sources import DataSource, SourceParams

            class ConstantParams(SourceParams):
                type: Literal["constant"] = "constant"
                rows: int

            class ConstantSource(DataSource):
                kind: ClassVar[str] = "constant"
                Params: ClassVar[type[SourceParams]] = ConstantParams

                def load(self):
                    return pl.DataFrame({"x": [1] * self.params.rows})
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    settings = make_settings(datasets={"c": {"source": {"type": "constant", "rows": 2}}})
    settings.plugins.modules = ["my_sources"]
    settings.plugins.entry_points = False
    runner = QARunner(settings)
    assert runner.context().get("c").height == 2
    sys.modules.pop("my_sources", None)


def test_unknown_or_invalid_sources_are_configuration_errors(registry) -> None:
    with pytest.raises(KeyError, match="unknown source type"):
        registry.build({"type": "ftp"}, SourceEnvironment())
    with pytest.raises(ValueError, match="needs a 'type'"):
        registry.build({"path": "x.csv"}, SourceEnvironment())
    with pytest.raises(ValueError, match="invalid 'file' source"):
        registry.build({"type": "file", "path": "x.csv", "colour": "red"}, SourceEnvironment())
    settings = make_settings(datasets={"d": {"source": {"type": "ftp"}}})
    with pytest.raises(ConfigurationError, match="dataset 'd'"):
        QARunner(settings).context()


# ------------------------------------------------------------- context wiring
def test_context_loads_lazily_and_records_failures(tmp_path) -> None:
    write_table(FRAME, tmp_path / "ok.csv")
    settings = make_settings(
        datasets={
            "ok": {"path": str(tmp_path / "ok.csv")},
            "missing": {"path": str(tmp_path / "missing.csv")},
            "broken": {"source": {"type": "query", "sql": "SELECT * FROM nowhere_table"}},
        }
    )
    ctx = DataContext.from_settings(
        settings, source_registry=SourceRegistry().discover(use_entry_points=False)
    )
    assert ctx.names() == ["broken", "missing", "ok"]
    assert ctx.has("ok") and ctx.get("ok").height == 3
    assert isinstance(ctx.load_error("missing"), SourceNotFoundError)
    assert "nowhere_table" in str(ctx.unavailable_reason("broken"))
    assert [info.name for info in ctx.describe()] == ["ok"]
    assert read_table(tmp_path / "ok.csv").height == 3
