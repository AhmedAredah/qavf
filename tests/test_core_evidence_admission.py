"""Database evidence preserves distinguishing values before bounded transfer."""

from datetime import date, datetime

import duckdb
import pytest

from qavf import Query, validate
from qavf.audit_codec import evidence_json_size
from qavf.core.results import ResultCollector
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings


@pytest.fixture
def collector():
    with duckdb.connect() as connection:
        yield ResultCollector(connection, QAVFSettings())


@pytest.mark.parametrize(
    "stamp",
    ["2024-01-01 00:00:00.000000001", "1969-12-31 23:59:59.999999999"],
)
def test_nanosecond_evidence_is_exact_at_every_container_depth(collector, stamp):
    value = f"TIMESTAMP_NS '{stamp}'"
    records = collector.records(
        f"SELECT {value} AS scalar, [{value}, NULL] AS items, "
        f"{{'inner': {{'stamp': {value}}}}} AS nested, "
        f"map(['stamp'], [{value}]) AS mapping, "
        f"map([{value}], [1]) AS temporal_keys, "
        "NULL::STRUCT(stamp TIMESTAMP_NS) AS absent"
    )
    expected = stamp.replace(" ", "T")
    assert records == [
        {
            "scalar": expected,
            "items": [expected, None],
            "nested": {"inner": {"stamp": expected}},
            "mapping": {"stamp": expected},
            "temporal_keys": {"key": [expected], "value": [1]},
            "absent": None,
        }
    ]
    assert not records.truncated


def test_temporal_extremes_and_calendar_intervals_are_not_finite_substitutes(collector):
    records = collector.records(
        "SELECT DATE 'infinity' AS high_date, TIMESTAMP '-infinity' AS low_timestamp, "
        "DATE '10000-01-01' AS wide_date, TIMESTAMP '10000-01-01' AS wide_timestamp, "
        "INTERVAL '1 month' AS calendar_interval, "
        "DATE '2024-01-01' AS ordinary_date, "
        "TIMESTAMP '2024-01-01 01:02:03.123456' AS ordinary_stamp"
    )
    assert records == [
        {
            "high_date": "infinity",
            "low_timestamp": "-infinity",
            "wide_date": "10000-01-01",
            "wide_timestamp": "10000-01-01T00:00:00",
            "calendar_interval": "1 month",
            "ordinary_date": date(2024, 1, 1),
            "ordinary_stamp": datetime(2024, 1, 1, 1, 2, 3, 123456),
        }
    ]


def test_native_membership_counterexample_retains_the_failing_nanosecond():
    report = validate(
        Query("SELECT TIMESTAMP_NS '2024-01-01 00:00:00.000000001' AS stamp"),
        schema={"columns": {"stamp": {"isin": [datetime(2024, 1, 1)]}}},
        backend="duckdb",
    )
    failed = [result for result in report.results if result.status == Status.FAIL]
    assert len(failed) == 1
    assert failed[0].affected_records[0]["stamp"] == "2024-01-01T00:00:00.000000001"


class ObservedConnection:
    """Observe objects actually crossing the driver boundary, before admission."""

    def __init__(self, connection):
        self.connection = connection
        self.transferred = []

    def sql(self, query):
        return self.connection.sql(query)

    def execute(self, query):
        cursor = self.connection.execute(query)
        observed = self.transferred

        class Cursor:
            def fetchone(self):
                row = cursor.fetchone()
                if row is not None:
                    observed.append(row)
                return row

        return Cursor()


@pytest.mark.parametrize("value", ["repeat('x', 262144)", "CAST(repeat('x', 262144) AS BLOB)"])
def test_oversized_cells_never_cross_the_python_driver_boundary(value):
    with duckdb.connect() as connection:
        observed = ObservedConnection(connection)
        collector = ResultCollector(observed, QAVFSettings(engine={"max_evidence_bytes": 1024}))
        records = collector.records(f"SELECT i AS id, {value} AS payload FROM range(50) t(i)")
        assert records == [] and records.truncated
        assert observed.transferred == [(None, True)]
        collector.add("INP-003", "le", "data", ["id"], count=50, total=50, records=records)
        result = collector.results[0]
        assert result.affected_count == 50 and result.affected_count_exact
        assert result.evidence_truncated and result.evidence_complete is False
        assert result.affected_records == []


def test_byte_admission_keeps_a_deterministic_prefix_and_exact_count(collector):
    collector.settings.engine.max_evidence_bytes = 1024
    records = collector.records(
        "SELECT i, repeat('x', CASE WHEN i = 2 THEN 262144 ELSE 64 END) AS payload "
        "FROM range(5) t(i) ORDER BY i"
    )
    assert [record["i"] for record in records] == [0, 1]
    assert records.truncated
    collector.add("INP-003", "le", "data", ["i"], count=5, records=records)
    result = collector.results[0]
    assert result.affected_count == 5 and result.affected_count_exact
    assert result.evidence_truncated and not result.evidence_complete
    assert evidence_json_size(result.affected_records) <= 1024


@pytest.mark.parametrize("status", [Status.ERROR, Status.SKIPPED])
def test_unassessed_outcomes_do_not_claim_exact_zero_or_passed_message(collector, status):
    collector.add("INP-003", "sum", "data", ["x"], status=status)
    result = collector.results[0]
    assert "passed" not in result.message.lower()
    assert not result.affected_count_exact
    assert result.evidence_complete is None and result.affected_rows is None


def test_explicit_zero_count_failure_is_not_reported_as_complete_success(collector):
    collector.add("INP-004", "missing key", "data", ["absent"], status=Status.FAIL)
    result = collector.results[0]
    assert result.message == "Check failed"
    assert not result.affected_count_exact and result.evidence_complete is False
