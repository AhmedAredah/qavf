"""Source boundary, provenance, resource cleanup, native scan and CLI contracts."""

from pathlib import Path
from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from qavf.cli import _parse_dataset, main
from qavf.engine.runner import build_registry
from qavf.profiling import ProfileResult
from qavf.scaffold import profile_frame, render_config
from qavf.settings import QAVFSettings, load_settings
from qavf.sources.base import SourceEnvironment, SourceError
from qavf.sources.file import FileSource
from qavf.sources.native import connect
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource
from qavf.sources.sql import resolve_sql_paths, validate_query


def _csv(tmp_path, name="data.csv", contents="x\n1\nbad\n2\n"):
    path = tmp_path / name
    path.write_text(contents)
    return path


def test_sql_paths_ignore_comments_and_quoted_brackets(tmp_path, monkeypatch):
    base, cwd = tmp_path / "config", tmp_path / "cwd"
    base.mkdir()
    cwd.mkdir()
    for folder, value in ((base, 1), (cwd, 999)):
        pl.DataFrame({"x": [value]}).write_csv(folder / "data.csv")
        pl.DataFrame({"x": [value]}).write_parquet(folder / "a]b.parquet")
    monkeypatch.chdir(cwd)
    for query in (
        "SELECT * FROM read_csv(/* comment */ 'data.csv')",
        "SELECT * FROM read_parquet(['a]b.parquet'])",
    ):
        source = QuerySource(QueryParams(sql=query), SourceEnvironment(base_dir=base))
        assert source.load()["x"].to_list() == [1]
    literal = "SELECT $$read_csv('data.csv')$$ AS x"
    assert resolve_sql_paths(literal, base) == literal
    with pytest.raises(SourceError, match="dynamic"):
        resolve_sql_paths("SELECT * FROM read_csv('data' || '.csv')", base)


@pytest.mark.parametrize("value", ["true", "false", 1, 0])
def test_lossy_options_are_strictly_typed(value):
    with pytest.raises(ValueError, match="Boolean"):
        FileParams(path="data.csv", options={"ignore_errors": value})


def test_option_names_normalize_before_loss_policy():
    with pytest.raises(ValueError, match="acknowledge_lossy"):
        FileParams(path="data.csv", options={"IGNORE_ERRORS": True})
    with pytest.raises(ValueError, match="repeated"):
        FileParams(path="data.csv", options={"header": True, "HEADER": False})


def test_false_does_not_discard_bad_rows(tmp_path):
    _csv(tmp_path)
    source = FileSource(
        FileParams(
            path="data.csv",
            options={"columns": {"x": "INTEGER"}, "header": True, "ignore_errors": False},
        ),
        SourceEnvironment(base_dir=tmp_path),
    )
    with pytest.raises(SourceError):
        source.load()


def test_rejected_rows_count_file_identity_and_configured_table(tmp_path):
    for name in ("a.csv", "b.csv"):
        _csv(tmp_path, name)
    options = {
        "columns": {"x": "INTEGER"},
        "header": True,
        "store_rejects": True,
        "rejects_table": "custom_rejects",
    }
    source = FileSource(
        FileParams(path="*.csv", options=options, acknowledge_lossy=True),
        SourceEnvironment(base_dir=tmp_path),
    )
    assert source.load().height == 4
    assert source.load_notes["rejected_rows"] == 2
    assert source.load_notes["rejected_rows_exact"] is True


def test_capped_reject_count_is_never_presented_as_exact(tmp_path):
    _csv(tmp_path, contents="x\n1\nbad\nworse\n2\n")
    source = FileSource(
        FileParams(
            path="data.csv",
            options={
                "columns": {"x": "INTEGER"},
                "header": True,
                "store_rejects": True,
                "rejects_limit": 1,
            },
            acknowledge_lossy=True,
        ),
        SourceEnvironment(base_dir=tmp_path),
    )
    source.load()
    assert source.load_notes["rejected_rows_exact"] is False
    assert source.load_notes["rejected_rows_lower_bound"] == 1
    assert "rejected_rows" not in source.load_notes


def test_manifest_is_shared_by_load_count_and_fingerprint(tmp_path):
    _csv(tmp_path, "data.csv", "x\n1\n")
    _csv(tmp_path, ".hidden.csv", "x\n999\n")
    source = FileSource(FileParams(path="*.csv", min_files=2), SourceEnvironment(base_dir=tmp_path))
    assert sorted(source.load()["x"].to_list()) == [1, 999]
    fingerprint = source.fingerprint(True)
    assert fingerprint["files"] == source.load_notes["files_matched"] == 2
    assert {Path(p).name for p in fingerprint["paths"]} == {"data.csv", ".hidden.csv"}
    _csv(tmp_path, "new.csv", "x\n5\n")
    assert source.fingerprint(True) == fingerprint
    _csv(tmp_path, ".hidden.csv", "x\n9999\n")
    with pytest.raises(SourceError, match="changed"):
        source.fingerprint(True)


def test_rejection_connection_closed_on_budget_failure(tmp_path):
    _csv(tmp_path)
    handles = []

    def tracked(env):
        result = connect(env)
        handles.append(result)
        return result

    source = FileSource(
        FileParams(
            path="data.csv",
            row_budget=1,
            options={"header": True, "columns": {"x": "INTEGER"}, "ignore_errors": True},
            acknowledge_lossy=True,
        ),
        SourceEnvironment(base_dir=tmp_path),
    )
    with (
        patch("qavf.sources.file.connect", tracked),
        pytest.raises(SourceError, match="row_budget"),
    ):
        source.load()
    with pytest.raises(duckdb.ConnectionException):
        handles[0].execute("SELECT 1")


def test_budget_allows_single_query_terminator_and_rejects_writes(tmp_path):
    env = SourceEnvironment(base_dir=tmp_path)
    source = QuerySource(QueryParams(sql="SELECT ';' AS x; -- final comment", row_budget=1), env)
    assert source.load()["x"].to_list() == [";"]
    for sql in ("CREATE TABLE x AS SELECT 1", "SELECT 1; SELECT 2"):
        with pytest.raises(SourceError):
            validate_query(sql)


def test_native_file_scan_does_not_load_the_frame(tmp_path):
    _csv(tmp_path, contents="x\n1\n2\n")
    source = FileSource(FileParams(path="data.csv"), SourceEnvironment(base_dir=tmp_path))
    plan = source.scan()
    assert plan is not None and not plan.parameters
    with connect(source.env) as con:
        relation = con.sql(plan.sql, params=dict(plan.parameters))
        assert relation.aggregate("sum(x)").fetchone() == (3,)


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["abc", "def"])])
def test_categorical_profile_preserves_string_constraints(dtype):
    result = profile_frame(pl.DataFrame({"x": pl.Series(["abc", "def"], dtype=dtype)}))
    assert result["columns"]["x"]["dtype"] == "str"
    assert result["columns"]["x"]["min_length"] == 3


@pytest.mark.parametrize("name", ["yes", "123", "my: dataset"])
def test_scaffold_yaml_quotes_dataset_names(tmp_path, name):
    registry = build_registry(QAVFSettings(plugins={"entry_points": False}))
    path = tmp_path / "config.yaml"
    path.write_text(render_config(registry, {name: {"path": "data.csv"}}), encoding="utf-8")
    assert name in load_settings(path).datasets


def test_cli_rejects_duplicate_names_before_reading(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("qavf.cli.profile_input", lambda path, **kwargs: calls.append(path))
    output = tmp_path / "qavf.yaml"
    assert main(["init", "-o", str(output), "-d", "a=one.csv", "-d", "a=two.csv"]) == 2
    assert not calls and not output.exists()


def test_cli_uri_parsing_and_output_preserve_signed_parameters(tmp_path, monkeypatch):
    uri = "https://example.org/records.csv?signature=abc&other=def"
    assert _parse_dataset(uri) == ("records", uri)
    assert _parse_dataset("sample=" + uri) == ("sample", uri)

    def profile(path, **kwargs):
        assert path == uri
        return ProfileResult(profile_frame(pl.DataFrame({"x": [1]})), rows=1, columns=1)

    monkeypatch.setattr("qavf.cli.profile_input", profile)
    output = tmp_path / "qavf.yaml"
    assert main(["init", "-o", str(output), "-d", uri, "--no-schema"]) == 0
    assert load_settings(output).datasets["records"].path == uri


def test_sql_grammar_literals_qualified_functions_and_unicode(tmp_path):
    _csv(tmp_path, contents="x\n1\n")
    env = SourceEnvironment(base_dir=tmp_path)
    sql = "SELECT extract(year FROM '2020-01-01'::DATE) AS y"
    assert QuerySource(QueryParams(sql=sql), env).load()["y"].to_list() == [2020]
    sql = "SELECT 'é' AS label, a.x FROM 'data.csv' a, 'data.csv' b"
    assert QuerySource(QueryParams(sql=sql), env).load()["x"].to_list() == [1]
    sql = "SELECT * FROM main.read_csv('data.csv')"
    assert QuerySource(QueryParams(sql=sql), env).load()["x"].to_list() == [1]


def test_connection_reservation_released_on_close_and_initialization_error(tmp_path):
    class Budget:
        def __init__(self):
            self.acquired = 0
            self.released = []

        def acquire(self):
            self.acquired += 1
            return self.acquired, 0, None

        def release(self, token):
            self.released.append(token)

    budget = Budget()
    env = SourceEnvironment(connection_budget=budget, temp_directory=tmp_path)
    con = connect(env)
    assert con.execute("SELECT 1").fetchone() == (1,)
    con.close()
    con.close()
    assert budget.released == [1]
    with pytest.raises(duckdb.Error):
        connect(SourceEnvironment(connection_budget=budget, memory_limit="invalid"))
    assert budget.released == [1, 2]


def test_excel_glob_trims_each_workbook_before_result_budget(tmp_path):
    with duckdb.connect() as con:
        try:
            con.execute("LOAD excel")
        except duckdb.Error as exc:
            pytest.skip(f"Excel extension is not installed locally: {exc}")
        for index in (1, 2):
            con.register("data", pl.DataFrame({"x": [index, None, None]}))
            path = tmp_path / f"book{index}.xlsx"
            con.execute(f"COPY data TO '{path}' (FORMAT xlsx, HEADER true)")
    source = FileSource(
        FileParams(path="book*.xlsx", row_budget=2), SourceEnvironment(base_dir=tmp_path)
    )
    assert source.scan() is None
    assert source.load()["x"].to_list() == [1, 2]
    with pytest.raises(SourceError, match="row_budget"):
        FileSource(
            FileParams(path="book*.xlsx", row_budget=1), SourceEnvironment(base_dir=tmp_path)
        ).load()


_CREDENTIAL_MESSAGE = (
    "password=do-not-emit-password "
    "https://example.org/data.csv?signature=do-not-emit-signature&batch=4"
)


def test_source_plugin_logging_and_status_hide_credentials(caplog, monkeypatch):
    import logging

    from qavf.sources.base import SourceRegistry

    class Entry:
        name = "broken-source"

        def load(self):
            raise RuntimeError(_CREDENTIAL_MESSAGE)

    monkeypatch.setattr("qavf.sources.base.entry_points", lambda group: [Entry()])
    with caplog.at_level(logging.DEBUG):
        registry = SourceRegistry().discover()
    observed = caplog.text + registry.plugins["broken-source"]
    assert "do-not-emit-password" not in observed
    assert "do-not-emit-signature" not in observed
    assert "RuntimeError" in observed and "test_source_boundaries.py:" in observed
    assert "batch=4" in observed
    assert all(record.exc_info is None for record in caplog.records)


def test_source_validation_and_constructor_errors_have_safe_diagnostics(monkeypatch):
    import traceback

    from qavf.sources.base import SourceRegistry

    registry = SourceRegistry().discover(use_entry_points=False)
    with pytest.raises(ValueError) as caught:
        registry.build(
            {
                "type": "file",
                "path": "https://example.org/data.csv?signature=do-not-emit-signature",
                "row_budget": "unlabelled-input-secret",
            },
            SourceEnvironment(),
        )
    message = "".join(traceback.format_exception(caught.value))
    assert "row_budget" in message and "ValidationError" in message
    assert "do-not-emit-signature" not in message and "unlabelled-input-secret" not in message

    def broken_constructor(self, params, env):
        raise RuntimeError(_CREDENTIAL_MESSAGE)

    monkeypatch.setattr(FileSource, "__init__", broken_constructor)
    with pytest.raises(ValueError) as caught:
        registry.build({"type": "file", "path": "data.csv"}, SourceEnvironment())
    message = "".join(traceback.format_exception(caught.value))
    assert "RuntimeError" in message
    assert "do-not-emit-password" not in message and "do-not-emit-signature" not in message


def test_verbose_cli_error_logging_and_stderr_are_redacted(caplog, capsys, monkeypatch):
    import logging

    def command(args):
        raise ValueError(_CREDENTIAL_MESSAGE)

    monkeypatch.setitem(
        __import__("qavf.cli", fromlist=["_COMMANDS"])._COMMANDS, "profile", command
    )
    with caplog.at_level(logging.DEBUG):
        assert main(["-vv", "profile", "unused.csv"]) == 2
    observed = caplog.text + capsys.readouterr().err
    assert "do-not-emit-password" not in observed and "do-not-emit-signature" not in observed
    assert "ValueError" in observed and "test_source_boundaries.py:" in observed
    assert all(record.exc_info is None for record in caplog.records)
