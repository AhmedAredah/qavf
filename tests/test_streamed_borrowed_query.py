"""A borrowed DuckDB query crosses into the native backend in bounded Arrow batches."""

import duckdb
import pytest

from qavf import Query, explain, validate
from qavf.bounded_batches import BoundedBatches
from qavf.core.sources import InputReader
from qavf.duckdb_results import arrow_table
from qavf.run_control import CANCELLED_MESSAGE, RunControl
from qavf.settings import DatasetSpec, QAVFSettings
from qavf.sources.base import SourceError

ROWS = 150_000
TYPED = (
    "SELECT range AS id, (range / 7)::DECIMAL(18, 3) AS amount, "
    "TIMESTAMPTZ '2024-01-01 00:00:00+00' + to_seconds(range) AS zoned, "
    "[range, range + 1] AS pair, {'k': range % 5, 's': range::VARCHAR} AS record, "
    "range::HUGEINT AS huge, (['low', 'mid', 'high'][(range % 3) + 1])::level AS level "
    "FROM range({rows})"
)


class CancelAt:
    """A scalar UDF that cancels the run once it sees a chosen value."""

    def __init__(self, control: RunControl, at: int) -> None:
        self.control, self.at = control, at

    def __call__(self, value: int) -> int:
        if value >= self.at:
            self.control.token.cancel()
        return value


def _owned_objects(connection):
    return connection.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE '__qavf_input_%'"
    ).fetchall()


@pytest.fixture
def owned():
    with duckdb.connect(config={"threads": 2}) as connection:
        yield connection


@pytest.fixture
def borrowed():
    with duckdb.connect(config={"threads": 2}) as connection:
        connection.execute("CREATE TYPE level AS ENUM ('low', 'mid', 'high')")
        yield connection


def _reader(connection, control=None):
    settings = QAVFSettings.model_validate({"datasets": {"data": {}}})
    return InputReader(connection, settings, control=control)


def test_streamed_copy_keeps_order_and_types_across_batches(owned, borrowed):
    sql = TYPED.replace("{rows}", str(ROWS))
    loaded = _reader(owned).load("data", Query(sql, connection=borrowed), DatasetSpec())
    transfer = loaded.metadata["transfer"]
    assert transfer["kind"] == "bounded_arrow_batches"
    assert transfer["batches"] > 1
    assert loaded.rows == ROWS
    columns = ", ".join(loaded.columns)
    copied = owned.sql(f"SELECT {columns} FROM {loaded.table} ORDER BY {loaded.row_column}")
    expected = arrow_table(borrowed.sql(sql))
    reference = duckdb.connect()
    reference.register("complete", expected)
    # The types and values a complete Arrow copy would have produced.
    assert copied.types == reference.sql("SELECT * FROM complete").types
    assert arrow_table(copied).equals(arrow_table(reference.sql("SELECT * FROM complete")))
    ids = owned.sql(f"SELECT {loaded.row_column} = id FROM {loaded.table}").fetchall()
    assert all(row[0] for row in ids)


def test_exact_row_budget_across_batches(owned, borrowed):
    query = Query(f"SELECT range AS id FROM range({ROWS})", connection=borrowed)
    loaded = _reader(owned).load("data", query, DatasetSpec(load_policy={"row_budget": ROWS}))
    assert loaded.rows == ROWS
    with pytest.raises(SourceError, match=f"more than {ROWS - 1} rows"):
        _reader(owned).load("data", query, DatasetSpec(load_policy={"row_budget": ROWS - 1}))
    assert [name for (name,) in _owned_objects(owned)] == [loaded.table]
    assert borrowed.execute("SELECT 1").fetchone() == (1,)


def test_cancellation_between_batches_leaves_nothing_owned(owned, borrowed):
    control = RunControl()
    borrowed.create_function("cancel_at", CancelAt(control, 70_000), ["BIGINT"], "BIGINT")
    query = Query(f"SELECT cancel_at(range) AS id FROM range({ROWS})", connection=borrowed)
    with pytest.raises(SourceError, match=CANCELLED_MESSAGE):
        _reader(owned, control).load("data", query, DatasetSpec())
    assert _owned_objects(owned) == []
    assert borrowed.execute("SELECT 42").fetchone() == (42,)


def test_caller_relation_stays_usable(owned, borrowed):
    borrowed.execute(f"CREATE TABLE source AS SELECT range AS id FROM range({ROWS})")
    relation = borrowed.sql("SELECT count(*) FROM source")
    loaded = _reader(owned).load(
        "data", Query("SELECT * FROM source", connection=borrowed), DatasetSpec()
    )
    assert loaded.rows == ROWS
    assert relation.fetchall() == [(ROWS,)]
    assert borrowed.sql("SELECT max(id) FROM source").fetchall() == [(ROWS - 1,)]


def test_reader_closes_part_way_without_reading_the_rest(borrowed):
    relation = borrowed.sql(f"SELECT range AS id FROM range({ROWS})")
    seen = []
    with BoundedBatches(relation, seen.append, int, batch_rows=2048) as stream:
        next(iter(stream))
    assert seen == [stream.rows] and stream.rows < ROWS
    assert borrowed.execute("SELECT 7").fetchone() == (7,)


def test_both_backends_agree_on_a_streamed_borrowed_query(borrowed):
    borrowed.execute(
        f"CREATE TABLE source AS SELECT range AS id, range % 10 AS v FROM range({ROWS})"
    )
    schema = {"columns": {"id": {"dtype": "int"}, "v": {"dtype": "int", "le": 8}}}
    reports = {
        backend: validate(
            {"data": Query("SELECT * FROM source", connection=borrowed)},
            config={"datasets": {"data": {"schema": schema}}},
            backend=backend,
        )
        for backend in ("duckdb", "polars")
    }
    outcomes = {
        backend: sorted(
            (r.rule_id, r.status.value, r.metrics.get("failed_rows")) for r in report.results
        )
        for backend, report in reports.items()
    }
    assert outcomes["duckdb"] == outcomes["polars"]
    assert reports["duckdb"].verdict() == "failed"


def test_explain_names_the_bounded_copy(borrowed):
    plan = explain(
        {"data": Query("SELECT 1 AS x", connection=borrowed)},
        config={"datasets": {"data": {"schema": {"columns": {"x": {"dtype": "int"}}}}}},
        backend="duckdb",
    )
    assert plan.datasets[0].staging == "bounded_batch_copy"
    assert plan.datasets[0].component  # The native description is still present.
