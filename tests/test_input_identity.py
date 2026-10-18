"""Exact input identity, typed file boundaries and bounded metadata planning."""

import os
from unittest.mock import patch

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from qavf import Status, validate
from qavf.io import read_table, write_table
from qavf.sources import SourceEnvironment, SourceError, SourceIntegrityError
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource
from qavf.sources.sql import normalize_sql_paths, resolve_sql_paths


@pytest.mark.parametrize("suffix", ["csv", "json", "parquet"])
@pytest.mark.parametrize(
    "name",
    [
        "a[1]",
        pytest.param(
            "a*",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="Windows forbids '*' in literal file names"
            ),
        ),
        pytest.param(
            "a?",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="Windows forbids '?' in literal file names"
            ),
        ),
        "a]1",
    ],
)
def test_expanded_manifest_names_are_literal(tmp_path, suffix, name):
    target = tmp_path / f"{name}.{suffix}"
    write_table(pl.DataFrame({"x": [-7]}), target)
    write_table(pl.DataFrame({"x": [999]}), tmp_path / f"a1.{suffix}")
    source = FileSource(FileParams(path=str(target)), SourceEnvironment())
    assert source.load()["x"].to_list() == [-7]
    plan = source.scan()
    assert plan is not None
    with duckdb.connect() as con:
        assert con.sql(plan.sql, params=dict(plan.parameters)).fetchall() == [(-7,)]
    assert source._files() == (target,)


def test_literal_directory_and_glob_manifest_do_not_duplicate_files(tmp_path):
    folder = tmp_path / "site[1]"
    folder.mkdir()
    for name, value in [("a[1].csv", 7), ("a1.csv", 999)]:
        write_table(pl.DataFrame({"x": [value]}), folder / name)
    # The directory is literal in an explicitly provided file path.
    assert read_table(folder / "a[1].csv")["x"].to_list() == [7]
    # A genuine pattern follows explicit glob syntax (escape the literal directory).
    import glob

    source = FileSource(FileParams(path=glob.escape(str(folder)) + "/*.csv"), SourceEnvironment())
    assert sorted(source.load()["x"].to_list()) == [7, 999]
    assert source.fingerprint()["files"] == 2


def test_public_validation_assesses_the_fingerprinted_literal_file(tmp_path):
    bad = tmp_path / "sample[1].csv"
    write_table(pl.DataFrame({"x": [-1]}), bad)
    write_table(pl.DataFrame({"x": [1]}), tmp_path / "sample1.csv")
    report = validate(bad, schema={"columns": {"x": {"ge": 0}}})
    assert any(r.rule_id == "INP-003" and r.status is Status.FAIL for r in report.results)
    assert report.metadata.datasets[0].fingerprint["paths"] == [str(bad)]


TYPED = [
    (pl.Datetime("ns", "UTC"), [1, 999]),
    (pl.Datetime("ns", "America/New_York"), [1, 999]),
    (pl.Datetime("ms"), [1, 2]),
    (pl.Duration("ns"), [1, 999]),
    (pl.Int128, [2**100 + 1, 2**100 + 2]),
    (pl.Time, [1, 999]),
    (pl.Array(pl.Int64, 2), [[1, 2], [3, 4]]),
    (pl.Enum(["a", "b", "unused"]), ["a", "b"]),
    (pl.List(pl.Datetime("ns", "UTC")), [[1, 999], [1001]]),
    (pl.List(pl.Int128), [[2**100 + 1], [2**100 + 2]]),
    (pl.List(pl.Array(pl.Int64, 2)), [[[1, 2]], [[3, 4]]]),
    (pl.Struct({"at": pl.Datetime("ns", "UTC")}), [{"at": 1}, {"at": 999}]),
    (pl.Struct({"tag": pl.Enum(["a", "b"])}), [{"tag": "a"}, {"tag": "b"}]),
]


@pytest.mark.parametrize("dtype,values", TYPED)
@pytest.mark.parametrize("empty", [False, True])
def test_recursive_parquet_logical_types_round_trip(dtype, values, empty, tmp_path):
    expected = pl.DataFrame({"v": pl.Series(values, dtype=dtype)})
    if empty:
        expected = expected.head(0)
    path = tmp_path / "typed.parquet"
    write_table(expected, path)
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    assert source.scan() is None
    plan = source.scan_polars()
    assert isinstance(plan, pl.LazyFrame)
    assert_frame_equal(plan.collect(), expected)
    assert_frame_equal(read_table(path), expected)


def test_typed_plan_keeps_complete_rows_and_load_enforces_budget(tmp_path):
    path = tmp_path / "typed.parquet"
    pl.DataFrame({"at": pl.Series([1, 2, 3], dtype=pl.Datetime("ns", "UTC"))}).write_parquet(path)
    source = FileSource(FileParams(path=str(path), row_budget=1), SourceEnvironment())
    assert source.scan_polars().select(pl.len()).collect().item() == 3
    with pytest.raises(SourceError, match="row_budget"):
        source.load()


def test_parquet_capability_decision_is_cached_without_retaining_every_schema(tmp_path):
    for n in range(4):
        pl.DataFrame({"x": [n]}).write_parquet(tmp_path / f"{n}.parquet")
    source = FileSource(FileParams(path=str(tmp_path / "*.parquet")), SourceEnvironment())
    original = pl.scan_parquet
    with patch("qavf.sources.file.pl.scan_parquet", wraps=original) as read:
        for _ in range(4):
            assert source.scan() is not None
        assert read.call_count == 4
    assert not hasattr(source, "_parquet_schemas")
    assert source._parquet_union is None


def test_typed_capability_validates_every_file_before_caching(tmp_path):
    for n in range(4):
        pl.DataFrame({"d": pl.Series([n], dtype=pl.Duration("ns"))}).write_parquet(
            tmp_path / f"{n}.parquet"
        )
    source = FileSource(FileParams(path=str(tmp_path / "*.parquet")), SourceEnvironment())
    original = pl.scan_parquet
    with patch("qavf.sources.file.pl.scan_parquet", wraps=original) as read:
        assert source.scan() is None
        assert source.scan() is None
        assert read.call_count == 4


def test_typed_union_retains_one_schema_and_missing_columns(tmp_path):
    pl.DataFrame({"d": pl.Series([1], dtype=pl.Duration("ns")), "a": [7]}).write_parquet(
        tmp_path / "a.parquet"
    )
    pl.DataFrame({"d": pl.Series([2], dtype=pl.Duration("ns")), "b": [8]}).write_parquet(
        tmp_path / "b.parquet"
    )
    source = FileSource(
        FileParams(path=str(tmp_path / "*.parquet"), options={"union_by_name": True}),
        SourceEnvironment(),
    )
    frame = source.load()
    assert frame.schema == {"d": pl.Duration("ns"), "a": pl.Int64, "b": pl.Int64}
    assert frame["a"].to_list() == [7, None]
    assert frame["b"].to_list() == [None, 8]
    assert not hasattr(source, "_parquet_schemas")


def test_hash_reuse_does_not_skip_content_verification(tmp_path):
    import qavf.sources.file as file_sources

    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    with patch("qavf.sources.file._sha256", wraps=file_sources._sha256) as hashed:
        first = source.fingerprint(True)
        assert hashed.call_count == 1
        assert source.fingerprint(True) == first
        assert hashed.call_count == 2  # one fresh verification, no redundant second hash
    original = source._file_stat(path)
    path.write_text("x\n2\n")
    # Even a writer preserving all observable metadata must fail content identity.
    with (
        patch.object(source, "_file_stat", return_value=original),
        pytest.raises(SourceIntegrityError, match="content changed"),
    ):
        source.verify_identity()


@pytest.mark.parametrize(
    "sql",
    [
        'SELECT * FROM "data.csv"',
        'SELECT * FROM main."data.csv"',
        "SELECT * FROM query_table('data.csv')",
        "SELECT * FROM query_table($name)",
    ],
)
def test_ambiguous_sql_filename_does_not_fall_back_to_cwd(tmp_path, sql):
    with pytest.raises(SourceError, match=r"ambiguous.*explicit file reader"):
        normalize_sql_paths(sql, tmp_path, {"name": "data.csv"} if "$" in sql else None)


def test_sql_real_quoted_tables_and_ctes_remain_identifiers(tmp_path):
    for sql in [
        'SELECT * FROM "my table"',
        'WITH "data.csv" AS (SELECT 1) SELECT * FROM "data.csv"',
    ]:
        assert resolve_sql_paths(sql, tmp_path) == sql
    sql = 'SELECT * FROM "data.csv"'
    assert resolve_sql_paths(sql, tmp_path, known_tables=["data.csv"]) == sql


def test_self_named_cte_cannot_bypass_quoted_file_guard(tmp_path):
    sql = 'WITH "data.csv" AS (SELECT * FROM "data.csv") SELECT * FROM "data.csv"'
    with pytest.raises(SourceError, match="ambiguous"):
        resolve_sql_paths(sql, tmp_path)


@pytest.mark.parametrize("syntax", ["read_csv('a[1].csv')", "'a[1].csv'", "read_csv($path)"])
def test_supported_sql_file_forms_resolve_exact_config_path(tmp_path, monkeypatch, syntax):
    base, cwd = tmp_path / "config", tmp_path / "cwd"
    base.mkdir()
    cwd.mkdir()
    for folder, wanted, neighbor in [(base, 7, 8), (cwd, 9, 10)]:
        (folder / "a[1].csv").write_text(f"x\n{wanted}\n")
        (folder / "a1.csv").write_text(f"x\n{neighbor}\n")
    monkeypatch.chdir(cwd)
    sql, params = normalize_sql_paths(
        f"SELECT * FROM {syntax}", base, {"path": "a[1].csv"} if "$" in syntax else None
    )
    with duckdb.connect() as con:
        assert con.sql(sql, params=params).fetchall() == [(7,)]


def test_literal_only_excel_reader_does_not_receive_glob_escapes(tmp_path):
    path = tmp_path / "book[1].xlsx"
    path.touch()
    assert str(path) in resolve_sql_paths("SELECT * FROM read_xlsx('book[1].xlsx')", tmp_path)


def test_query_table_regular_table_argument_stays_unchanged(tmp_path):
    sql = "WITH t AS (SELECT 7 AS x) SELECT * FROM query_table('t')"
    assert resolve_sql_paths(sql, tmp_path) == sql
    assert QuerySource(QueryParams(sql=sql), SourceEnvironment()).load().item() == 7


def test_explicit_database_table_can_have_a_filename_like_identifier(tmp_path):
    from qavf.settings import ConnectionSpec
    from qavf.sources.parameters import DatabaseParams
    from qavf.sources.query import DatabaseSource

    path = tmp_path / "w.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute('CREATE TABLE "data.csv" AS SELECT 7 AS x')
    source = DatabaseSource(
        DatabaseParams(connection="w", table="data.csv"),
        SourceEnvironment(connections={"w": ConnectionSpec(type="duckdb", path=str(path))}),
    )
    assert source.load().item() == 7
