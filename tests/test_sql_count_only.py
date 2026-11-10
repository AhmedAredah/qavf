"""A SQL check counts one evaluation exactly and keeps only bounded evidence (EP-07)."""

from __future__ import annotations

from typing import Any

import duckdb
import pytest

import qavf
from qavf.core.runner import NativeRunner
from qavf.ordered_sample import OrderedSample
from qavf.settings import QAVFSettings

FAILING = "SELECT v FROM range(10) AS t(v) WHERE v >= 4"


class RecordingConnection:
    """A DuckDB connection that records every statement it executes."""

    def __init__(self) -> None:
        self.connection = duckdb.connect()
        self.statements: list[str] = []

    def execute(self, sql: str) -> Any:
        self.statements.append(sql)
        return self.connection.execute(sql)

    def sql(self, sql: str) -> Any:
        """Bind without running, as :meth:`duckdb.DuckDBPyConnection.sql` does."""
        return self.connection.sql(sql)


class RecordingOutcome:
    """The part of the native result collector an assertion writes to."""

    def __init__(self) -> None:
        self.connection = RecordingConnection()
        self.reports: list[dict[str, Any]] = []

    def report(self, *args: Any, **kwargs: Any) -> None:
        self.reports.append({"args": args, **kwargs})

    def passed(self, *args: Any, **kwargs: Any) -> None:
        self.reports.append({"args": args, "passed": True})

    def records(self, sql: str) -> list[dict[str, Any]]:
        frame = self.connection.execute(sql).fetchall()
        return [{"v": row[0]} for row in frame]


def _settings(limit: int) -> QAVFSettings:
    return QAVFSettings.model_validate(
        {
            "datasets": {"t": {"path": "t.csv"}},
            "checks": [{"check": "sql", "name": "big", "sql": FAILING}],
            "engine": {"max_affected_records": limit},
        }
    )


def _run(limit: int) -> RecordingOutcome:
    settings = _settings(limit)
    out = RecordingOutcome()
    NativeRunner(settings, {})._assertion(out, 0, settings.assertions[0], "__snap")
    return out


def test_no_evidence_counts_without_a_temporary_table() -> None:
    out = _run(0)
    assert not any("CREATE TEMP TABLE" in sql for sql in out.connection.statements)
    counts = [sql for sql in out.connection.statements if "count(*)" in sql]
    assert len(counts) == 1
    assert out.reports[0]["count"] == 6 and out.reports[0]["records"] is None


def test_evidence_still_comes_from_one_snapshot() -> None:
    out = _run(3)
    assert sum("CREATE TEMP TABLE" in sql for sql in out.connection.statements) == 1
    assert out.reports[0]["count"] == 6 and len(out.reports[0]["records"]) == 3


@pytest.mark.parametrize("limit", [0, 5])
def test_native_sql_check_counts_failing_rows_with_and_without_evidence(limit: int) -> None:
    report = qavf.validate(
        {"t": qavf.Dataset("unused.csv")},
        config={
            "checks": [{"check": "sql", "name": "big", "sql": FAILING}],
            "engine": {"max_affected_records": limit},
        },
        backend="duckdb",
        only="big",
    )
    (result,) = [item for item in report.results if item.check == "big"]
    assert result.affected_count == 6
    assert len(result.affected_records) == min(limit, 6)


def test_the_snapshot_holds_only_the_count_and_a_bounded_sample() -> None:
    out = _run(3)
    (created,) = [sql for sql in out.connection.statements if "CREATE TEMP TABLE" in sql]
    assert "count(*) AS __count" in created and ", 3) AS __sample" in created
    assert [row["v"] for row in out.reports[0]["records"]] == [4, 5, 6]


def _unbindable(self: OrderedSample, connection: Any, query: str) -> None:
    """Rows the reducer cannot order: the assertion is staged instead."""
    return None


def test_rows_the_reducer_cannot_order_are_staged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OrderedSample, "bound", _unbindable)
    out = _run(3)
    (created,) = [sql for sql in out.connection.statements if "CREATE TEMP TABLE" in sql]
    assert "__count" not in created
    assert out.reports[0]["count"] == 6
    assert [row["v"] for row in out.reports[0]["records"]] == [4, 5, 6]


ORDERED = (
    "SELECT v % 3 AS a, CASE WHEN v % 4 = 0 THEN NULL ELSE v % 5 END AS b, "
    "CASE WHEN v % 7 = 0 THEN 'nan'::DOUBLE ELSE v / 10 END AS c, 'x' AS a "
    "FROM range(500) AS t(v)"
)


def test_the_bounded_sample_is_the_first_rows_of_a_full_sort() -> None:
    """Repeated values, NULLs, NaN and duplicate names keep ``ORDER BY ALL`` order."""
    with duckdb.connect() as con:
        expected = con.execute(f"SELECT * FROM ({ORDERED}) ORDER BY ALL LIMIT 9").fetchall()
        sample = OrderedSample(9)
        statement = sample.bound(con, ORDERED)
        assert statement is not None
        con.execute(f"CREATE TEMP TABLE kept AS {statement}")
        assert con.execute("SELECT __count FROM kept").fetchall() == [(500,)]
        relation = con.sql(sample.rows("kept"))
        assert relation.columns == ["a", "b", "c", "a_1"]
        assert relation.fetchall() == expected


def test_a_volatile_assertion_is_counted_and_sampled_from_one_evaluation() -> None:
    with duckdb.connect() as con:
        statement = OrderedSample(4).statement("SELECT random() AS r FROM range(1000)")
        ((count, kept),) = con.execute(statement).fetchall()
    values = [row["r"] for row in kept]
    assert count == 1000 and values == sorted(values) and len(values) == 4


def test_a_zero_limit_keeps_a_typed_empty_sample() -> None:
    with duckdb.connect() as con:
        ((count, kept),) = con.execute(OrderedSample(0).statement("SELECT 1 AS v")).fetchall()
    assert (count, kept) == (1, None)
    with pytest.raises(ValueError, match="non-negative"):
        OrderedSample(-1)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_both_backends_count_exactly_and_keep_the_same_bounded_evidence(backend: str) -> None:
    report = qavf.validate(
        {"t": qavf.Dataset("unused.csv")},
        config={
            "checks": [{"check": "sql", "name": "wide", "sql": ORDERED.replace(", 'x' AS a", "")}],
            "engine": {"max_affected_records": 7},
        },
        backend=backend,
        only="wide",
    )
    (result,) = [item for item in report.results if item.check == "wide"]
    assert result.affected_count == 500
    rows = [(row["a"], row["b"]) for row in result.affected_records]
    assert rows == [(0, 0)] * 7
