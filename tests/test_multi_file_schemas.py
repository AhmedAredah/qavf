"""Lossless multi-file boundaries and native attachment retry classification."""

from unittest.mock import patch

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from qavf import Status, validate
from qavf.settings import ConnectionSpec
from qavf.sources import SourceEnvironment, SourceError
from qavf.sources.execution import execute_with_source_policy
from qavf.sources.file import FileSource
from qavf.sources.native import connect
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource


@pytest.mark.parametrize("union_by_name", [False, True])
@pytest.mark.parametrize("first", [1, 2**53 + 1])
def test_mixed_numeric_parquet_never_passes_after_implicit_cast(
    tmp_path,
    first,
    union_by_name,
):
    pl.DataFrame({"x": pl.Series([first], dtype=pl.Int64)}).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame({"x": [1.25]}).write_parquet(tmp_path / "b.parquet")
    source = {
        "type": "file",
        "path": str(tmp_path / "*.parquet"),
        "options": {"union_by_name": union_by_name},
    }
    report = validate(
        config={
            "datasets": {"data": {"source": source, "schema": {"columns": {"x": {"le": 1}}}}},
            "plugins": {"entry_points": False},
            "engine": {"parallel": False},
        }
    )
    assert report.exit_code() == 1
    assert any(
        result.status is Status.ERROR and "matching logical types" in result.message
        for result in report.results
    )
    assert not any(
        result.rule_id == "INP-003" and result.status is Status.PASS for result in report.results
    )


@pytest.mark.parametrize("union_by_name", [False, True])
def test_typed_first_file_does_not_hide_later_schema_conflict(tmp_path, union_by_name):
    for name, unit in (("a", "ns"), ("b", "us")):
        pl.DataFrame({"x": pl.Series([1], dtype=pl.Duration(unit))}).write_parquet(
            tmp_path / f"{name}.parquet"
        )
    source = FileSource(
        FileParams(path=str(tmp_path / "*.parquet"), options={"union_by_name": union_by_name}),
        SourceEnvironment(),
    )
    with pytest.raises(SourceError, match="matching logical types"):
        source.scan()


def test_parquet_extra_columns_require_explicit_union(tmp_path):
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame({"x": [2], "new": [3]}).write_parquet(tmp_path / "b.parquet")
    with pytest.raises(SourceError, match="matching column names"):
        FileSource(FileParams(path=str(tmp_path / "*.parquet")), SourceEnvironment()).load()
    combined = FileSource(
        FileParams(path=str(tmp_path / "*.parquet"), options={"union_by_name": True}),
        SourceEnvironment(),
    ).load()
    assert_frame_equal(combined, pl.DataFrame({"x": [1, 2], "new": [None, 3]}))


def test_matching_parquet_schemas_preserve_values_without_repeated_footer_reads(tmp_path):
    expected = pl.DataFrame({"x": [2**53 + 1, 2**53 + 3], "y": ["a", "b"]})
    expected.head(1).write_parquet(tmp_path / "a.parquet")
    expected.tail(1).select("y", "x").write_parquet(tmp_path / "b.parquet")
    source = FileSource(FileParams(path=str(tmp_path / "*.parquet")), SourceEnvironment())
    with patch("qavf.sources.file.pl.scan_parquet", wraps=pl.scan_parquet) as read:
        assert_frame_equal(source.load(), expected)
        assert source.scan() is not None
        assert read.call_count == 2


def test_parquet_union_preserves_distinct_case_sensitive_columns(tmp_path):
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame({"X": [2]}).write_parquet(tmp_path / "b.parquet")
    source = FileSource(
        FileParams(path=str(tmp_path / "*.parquet"), options={"union_by_name": True}),
        SourceEnvironment(),
    )
    assert source.scan() is None
    assert_frame_equal(source.load(), pl.DataFrame({"x": [1, None], "X": [None, 2]}))
    report = validate(
        source,
        schema={"columns": {"x": {"nullable": False}}},
        config={
            "plugins": {"entry_points": False},
            "engine": {"parallel": False},
        },
    )
    assert report.exit_code() == 1
    assert any(
        result.rule_id == "INP-003" and result.status is Status.FAIL for result in report.results
    )


def test_single_parquet_preserves_distinct_case_sensitive_columns(tmp_path):
    path = tmp_path / "values.parquet"
    expected = pl.DataFrame({"x": [1], "X": [2]})
    expected.write_parquet(path)
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    assert source.scan() is None
    assert_frame_equal(source.load(), expected)


@pytest.mark.parametrize("status,expected_attempts", [(503, 3), (401, 1)])
def test_attachment_errors_keep_native_retry_classification(status, expected_attempts):
    created = []

    class NativeConnection:
        closed = False

        def __init__(self):
            created.append(self)

        def execute(self, sql):
            assert sql.startswith("ATTACH ")
            if len(created) < 3:
                raise duckdb.HTTPException(f"HTTP Error: {status} attachment failure")
            return self

        def close(self):
            self.closed = True

    env = SourceEnvironment(
        connections={
            "warehouse": ConnectionSpec(type="postgres", uri="host=example.org dbname=qa"),
        }
    )
    source = QuerySource(
        QueryParams(sql="SELECT * FROM warehouse.t", retries=2, retry_delay_s=0),
        env,
    )
    with patch("qavf.sources.native.duckdb.connect", side_effect=NativeConnection):
        if status == 503:
            connection = execute_with_source_policy([source], lambda: connect(env, ["warehouse"]))
            connection.close()
        else:
            with pytest.raises(SourceError, match="could not attach connection") as failed:
                execute_with_source_policy([source], lambda: connect(env, ["warehouse"]))
            assert not failed.value.retryable
    assert len(created) == expected_attempts
    assert all(connection.closed for connection in created)
