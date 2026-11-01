"""Syntax reuse must never cache bound objects or filesystem observations."""

import pytest

from qavf.sources.base import SourceError
from qavf.sources.sql import _sql_syntax, normalize_sql_paths


def test_repeated_generated_query_reuses_syntax(tmp_path):
    _sql_syntax.cache_clear()
    query = "SELECT sum(x) FROM records WHERE x >= 0"
    assert normalize_sql_paths(query, tmp_path, known_tables=("records",)) == (query, {})
    first = _sql_syntax.cache_info()
    normalize_sql_paths(query, tmp_path, known_tables=("records",))
    assert _sql_syntax.cache_info().hits == first.hits + 1


def test_cached_syntax_rechecks_new_glob_named_literal_file(tmp_path):
    query = "SELECT * FROM read_csv('part[1].csv')"
    before, _ = normalize_sql_paths(query, tmp_path)
    (tmp_path / "part[1].csv").write_text("x\n7\n")
    after, _ = normalize_sql_paths(query, tmp_path)
    assert "part[1].csv" in before
    assert "part[[]1].csv" in after


def test_cached_syntax_keeps_parameter_objects_independent(tmp_path):
    query = "SELECT * FROM read_csv($file)"
    first = {"file": "a.csv"}
    second = {"file": "b.csv"}
    assert normalize_sql_paths(query, tmp_path, first)[1]["file"].endswith("a.csv")
    assert normalize_sql_paths(query, tmp_path, second)[1]["file"].endswith("b.csv")
    assert first == {"file": "a.csv"} and second == {"file": "b.csv"}


def test_cached_syntax_does_not_cache_known_table_authority(tmp_path):
    query = 'SELECT * FROM "records.csv"'
    assert normalize_sql_paths(query, tmp_path, known_tables=("records.csv",))[0] == query
    with pytest.raises(SourceError, match="ambiguous quoted"):
        normalize_sql_paths(query, tmp_path)
