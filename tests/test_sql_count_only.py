"""A SQL check that keeps no evidence counts in one evaluation, without a snapshot (EP-07)."""

from __future__ import annotations

from typing import Any

import duckdb
import pytest

import qavf
from qavf.core.runner import NativeRunner
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
