"""Writers, source formats, credential redaction and README contracts.

Writers that honour their file names, format inference from URLs,
credentials kept out of reports and logs, SQL-aware attachment of database
connections, Excel support at the minimum DuckDB version and a
README that matches the code.
"""

from __future__ import annotations

import gzip
import logging
import re
from importlib.metadata import requires
from pathlib import Path
from typing import ClassVar

import duckdb
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from qavf.cli import _parser as build_parser
from qavf.engine.runner import QARunner
from qavf.io import read_table, write_table
from qavf.reports.html import render_html
from qavf.settings import ConnectionSpec, QAVFSettings
from qavf.sources import sql as source_sql
from qavf.sources.base import DataSource, SourceEnvironment, SourceError, SourceParams
from qavf.sources.formats import infer_format
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource
from qavf.testing import discovered_rules

ROOT = Path(__file__).resolve().parents[1]
FRAME = pl.DataFrame({"x": [1, 2], "y": ["a, b", "c"]})


# =========================================================================
# Written files match their names and read back
# =========================================================================
@pytest.mark.parametrize(
    "name",
    [
        "out.csv",
        "out.tsv",
        "out.txt",
        "out.csv.gz",
        "out.tsv.gz",
        "out.json",
        "out.json.gz",
        "out.ndjson",
        "out.ndjson.gz",
        "out.parquet",
    ],
)
def test_written_tables_round_trip(tmp_path, name) -> None:
    target = write_table(FRAME, tmp_path / name)
    assert read_table(target).equals(FRAME)


def test_tsv_is_tab_separated(tmp_path) -> None:
    raw = write_table(FRAME, tmp_path / "out.tsv").read_bytes()
    assert raw.splitlines()[0] == b"x\ty"


def test_gzip_output_is_really_gzip(tmp_path) -> None:
    raw = write_table(FRAME, tmp_path / "out.csv.gz").read_bytes()
    assert gzip.decompress(raw).splitlines()[0] == b"x,y"


@pytest.mark.parametrize("name", ["out.csv.zst", "out.parquet.gz", "out.xlsx", "out.unknown"])
def test_unwritable_names_are_rejected_before_writing(tmp_path, name) -> None:
    with pytest.raises(ValueError, match="write"):
        write_table(FRAME, tmp_path / name)
    assert not (tmp_path / name).exists()


# =========================================================================
# Formats are inferred from a URL's path
# =========================================================================
@pytest.mark.parametrize(
    ("location", "fmt"),
    [
        ("https://example.test/data.csv?signature=test", "csv"),
        ("https://example.test/data.parquet#part", "parquet"),
        ("https://example.test/my%20data.csv", "csv"),
        ("https://example.test/data%2Ecsv", "csv"),
        ("s3://bucket/key/data.ndjson.gz?versionId=3", "ndjson"),
        ("data/file?.csv", "csv"),  # a local glob: '?' is a wildcard, not a query
        ("data/part-[0-9].parquet", "parquet"),
    ],
)
def test_format_inference(location, fmt) -> None:
    assert infer_format(location) == fmt


def test_uninferable_format_is_a_configuration_error() -> None:
    with pytest.raises(ValidationError, match="format"):
        FileParams.model_validate({"type": "file", "path": "https://example.test/download?file=x"})
    FileParams.model_validate(
        {"type": "file", "path": "https://example.test/download", "format": "csv"}
    )


# =========================================================================
# Credentials never reach reports or logs
# =========================================================================
SECRET_URL = "https://user:pa55word@data.example/data.csv?access_token=t0ken&X-Amz-Signature=s1g"
SECRETS = ("pa55word", "t0ken", "s1g", "hunter2")


class _RemoteFailure(DataSource):
    """Fails the way a signed-URL read does, quoting the URL (no network involved)."""

    kind: ClassVar[str] = "remote-failure"
    Params: ClassVar[type[SourceParams]] = SourceParams

    def load(self) -> pl.DataFrame:
        raise duckdb.IOException(f"HTTP GET error on '{SECRET_URL}' (HTTP 403)")

    def describe(self) -> str:
        return SECRET_URL


def _leaky_attach(self, name, env) -> str:
    # A local ATTACH that fails and whose DuckDB error quotes the connection string,
    # password included, exercising the real attach path without an extension download.
    return f"ATTACH '{env.base_dir.as_posix()}/missing/host=db password=hunter2.db' AS \"{name}\""


def test_credentials_are_masked_in_reports_and_logs(caplog, tmp_path, monkeypatch) -> None:
    from qavf.engine.context import DataContext

    monkeypatch.setattr(ConnectionSpec, "attach_sql", _leaky_attach)
    settings = QAVFSettings.model_validate(
        {
            "datasets": {"remote": {}},
            "connections": {"pg": {"type": "duckdb", "path": "unused.duckdb"}},
            "assertions": [{"name": "remote db", "sql": "SELECT * FROM pg.main.t"}],
            "plugins": {"entry_points": False},
        }
    )
    env = SourceEnvironment(base_dir=tmp_path, connections=settings.connections)
    ctx = DataContext(
        specs=settings.datasets,
        sources={"remote": _RemoteFailure(SourceParams(type="remote-failure"), env)},
        env=env,
        assertions=settings.assertions,
    )
    with caplog.at_level(logging.DEBUG, logger="qavf"):
        report = QARunner(settings, discovered_rules(entry_points=False)).run(ctx=ctx)
    outputs = {
        "json": report.model_dump_json(),
        "html": render_html(report),
        "logs": caplog.text,
    }
    for where, text in outputs.items():
        for secret in SECRETS:
            assert secret not in text, (where, secret)
    assert report.get("INP-001").failed and "data.example" in report.get("INP-001").message
    assert report.get("INP-007").status.value == "ERROR"  # the attach failure, masked


def test_source_errors_are_redacted_without_the_original_chain(tmp_path) -> None:
    from qavf.engine.context import DataContext

    source = _RemoteFailure(
        SourceParams(type="remote-failure"), SourceEnvironment(base_dir=tmp_path)
    )
    ctx = DataContext(sources={"remote": source})
    error = ctx.load_error("remote")
    assert isinstance(error, SourceError)
    chain = [error, error.__cause__, error.__context__]
    assert not any(secret in str(item) for item in chain if item for secret in SECRETS)


# =========================================================================
# Connections are attached by what the SQL references
# =========================================================================
@pytest.mark.parametrize(
    ("sql", "catalogs"),
    [
        ('SELECT * FROM "w".items', {"w"}),
        ("SELECT * FROM W.items", {"W"}),
        ("SELECT * FROM w.main.items", {"w"}),
        ("SELECT * FROM t WHERE id IN (SELECT id FROM w.items)", {"w"}),
        ("SELECT w.x FROM (SELECT 1 AS x) AS w", set()),
        ("SELECT 1 AS x -- old query read w.items", set()),
        ("SELECT 'w.items' AS s", set()),
        ("WITH w AS (SELECT 1 AS x) SELECT * FROM w", set()),
    ],
)
def test_sql_catalog_references(sql, catalogs) -> None:
    assert source_sql.sql_catalogs(sql) == catalogs


@pytest.fixture
def warehouse(tmp_path) -> SourceEnvironment:
    path = tmp_path / "w.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE items AS SELECT 1 AS id")
    return SourceEnvironment(
        base_dir=tmp_path, connections={"w": ConnectionSpec(type="duckdb", path=str(path))}
    )


@pytest.mark.parametrize(
    "sql", ['SELECT * FROM "w".items', "SELECT * FROM W.items", "SELECT * FROM w.main.items"]
)
def test_referenced_connections_are_attached(warehouse, sql) -> None:
    source = QuerySource(QueryParams(type="query", sql=sql), warehouse)
    assert source.load().height == 1


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT w.x FROM (SELECT 1 AS x) AS w",
        "SELECT 1 AS x -- old query read w.items",
        "SELECT 'w.items' AS s",
    ],
)
def test_unreferenced_connections_are_not_attached(warehouse, sql) -> None:
    source = QuerySource(QueryParams(type="query", sql=sql), warehouse)
    assert source._attachments() == []


def test_context_sql_attaches_referenced_connections(warehouse) -> None:
    from qavf.engine.context import DataContext

    ctx = DataContext(env=warehouse)
    assert ctx.sql('SELECT * FROM "W".items').height == 1


# =========================================================================
# Excel works at the minimum supported DuckDB
# =========================================================================
def test_minimum_duckdb_supports_excel() -> None:
    [spec] = [r for r in requires("qavf") or [] if re.match(r"duckdb\b", r)]
    minimum = tuple(int(p) for p in re.search(r">=\s*([\d.]+)", spec).group(1).split("."))
    assert minimum >= (1, 2), spec  # read_xlsx / COPY ... (FORMAT xlsx) arrived in DuckDB 1.2


# =========================================================================
# The README describes this package
# =========================================================================
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _section(title: str) -> str:
    start = README.index(f"## {title}\n")
    end = README.find("\n## ", start + 1)
    return README[start : end if end != -1 else None]


def test_readme_rule_table_matches_the_registry() -> None:
    rows = re.findall(r"^\| ([A-Z]+-\d{3}) \| (\w+)\*? \|", _section("Rules"), flags=re.MULTILINE)
    registry = discovered_rules(entry_points=False)
    assert {rule_id for rule_id, _ in rows} == {rule.rule_id for rule in registry}
    for rule_id, severity in rows:
        assert registry.get(rule_id).default_severity.value == severity, rule_id


def test_readme_mentions_only_existing_rule_ids() -> None:
    mentioned = set(re.findall(r"\b(?!SHA-)[A-Z]{3}-\d{3}\b", README))
    known = {rule.rule_id for rule in discovered_rules(entry_points=False)}
    assert mentioned <= known, mentioned - known


def test_readme_cli_table_matches_the_parser() -> None:
    documented = set(re.findall(r"^\| `qavf ([a-z-]+)", _section("CLI"), flags=re.MULTILINE))
    parser = build_parser()
    commands = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    assert documented == set(commands.choices)


def test_readme_local_links_exist() -> None:
    for target in re.findall(r"\]\(((?!https?:|#)[^)\s]+)\)", README):
        assert (ROOT / target.split("#")[0]).exists(), target


def test_readme_configuration_examples_validate() -> None:
    blocks = re.findall(r"```yaml\n(.*?)```", README, flags=re.DOTALL)
    fields = set(QAVFSettings.model_fields) | {"schema"}
    checked = 0
    for block in blocks:
        data = yaml.safe_load(block)
        if isinstance(data, dict) and set(data) <= set(QAVFSettings.model_fields):
            QAVFSettings.model_validate(data)
            checked += 1
    assert checked >= 2, fields


def test_readme_quick_start_runs(tmp_path) -> None:
    from qavf.cli import main

    assert main(["demo", str(tmp_path / "my-demo")]) == 0
    assert main(["run", "-c", str(tmp_path / "my-demo" / "qavf.yaml")]) in (0, 1)
    assert (tmp_path / "my-demo" / "qa_output" / "qa_report.html").exists()
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame({"site": ["a", "b"], "count": [1, 2]}).write_csv(data / "counts.csv")
    pl.DataFrame({"site": ["a"], "crashes": [3]}).write_parquet(data / "crashes.parquet")
    out = tmp_path / "qavf.yaml"
    args = [
        "init",
        "-o",
        str(out),
        "-d",
        str(data / "counts.csv"),
        "-d",
        f"crashes={data / 'crashes.parquet'}",
        "--margin",
        "0.1",
    ]
    assert main(args) == 0 and out.exists()
