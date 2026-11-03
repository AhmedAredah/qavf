"""DuckDB results are read through relations, without DuckDB 2.0's copy warning.

DuckDB 2.0 warns ("Arrow output on a DB-API result makes an extra copy of the
rows") when Arrow output is taken from the result of ``con.execute``. QAVF reads
from relations instead, with or without query parameters, on DuckDB 1.3 and 2.x.
These tests turn every warning into an error around those reads.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import pytest

from qavf import Dataset, Query, validate
from qavf.duckdb_results import query_frame, record_batch_reader, relation
from qavf.sources.native import fetch


@pytest.fixture
def strict_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.mark.usefixtures("strict_warnings")
def test_parameterized_queries_read_without_warnings():
    con = duckdb.connect()
    try:
        named = query_frame(con, "SELECT $x AS a, 'q' AS b", {"x": 2})
        positional = query_frame(con, "SELECT ? AS a", [3])
        plain = query_frame(con, "SELECT 4 AS a")
        assert named is not None and named.to_dicts() == [{"a": 2, "b": "q"}]
        assert positional is not None and positional["a"].to_list() == [3]
        assert plain is not None and plain["a"].to_list() == [4]
    finally:
        con.close()


@pytest.mark.usefixtures("strict_warnings")
def test_batches_stream_from_a_relation_without_warnings():
    con = duckdb.connect()
    try:
        reader = record_batch_reader(relation(con, "SELECT * FROM range(10)"), 4)
        assert sum(batch.num_rows for batch in reader) == 10
    finally:
        con.close()


@pytest.mark.usefixtures("strict_warnings")
def test_native_fetch_binds_parameters_without_warnings():
    frame = fetch(duckdb.connect(), "SELECT $n AS n", {"n": 7})
    assert frame["n"].to_list() == [7]


@pytest.mark.usefixtures("strict_warnings")
@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_validation_of_query_sources_raises_no_warning(tmp_path: Path, backend: str):
    data = tmp_path / "rows.csv"
    data.write_text("id,x\n1,2\n2,-1\n", encoding="utf-8")
    config = {
        "datasets": {
            "rows": {
                "source": {"type": "query", "sql": f"SELECT * FROM read_csv('{data.as_posix()}')"},
                "schema": {"columns": {"x": {"dtype": "int", "ge": 0}}, "primary_key": ["id"]},
            }
        },
        "checks": [{"check": "sql", "name": "negative x", "sql": "SELECT * FROM rows WHERE x < 0"}],
    }
    report = validate(config=config, backend=backend)
    assert report.get("INP-007").affected_count == 1


@pytest.mark.usefixtures("strict_warnings")
@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_parameterized_query_inputs_raise_no_warning(backend: str):
    query = Query("SELECT * FROM range($n) t(id)", parameters={"n": 3})
    report = validate(Dataset(query, schema={"primary_key": ["id"]}), backend=backend)
    assert report.verdict() == "passed"
