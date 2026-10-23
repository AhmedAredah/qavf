"""Reproductions for coherent reads, lock ordering and bounded source conversion."""

from __future__ import annotations

import contextlib
import threading

import duckdb
import polars as pl
import pytest

from qavf.engine.context import DataContext, DatasetNotAvailableError
from qavf.engine.resources import ResourceLimitError
from qavf.engine.session import KeyedLocks
from qavf.engine.sql import sql_table_names
from qavf.settings import ConnectionSpec, DatasetSpec
from qavf.sources.base import DataSource, SourceEnvironment, SourceError, SourceParams
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams


def source(path):
    return FileSource(FileParams(path=str(path)), SourceEnvironment())


def test_serialized_native_callback_and_pandas_share_one_lock_order(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_text("id\n1\n2\n")
    with DataContext(sources={"data": source(path)}, concurrent_loads=False) as ctx:
        consumer_entered, competing_entered = threading.Event(), threading.Event()
        guard = ctx._operations.guard
        errors, results = [], []

        @contextlib.contextmanager
        def signal_guard():
            if threading.current_thread().name == "competing-pandas":
                competing_entered.set()
            with guard():
                yield

        monkeypatch.setattr(ctx._operations, "guard", signal_guard)

        def native():
            try:

                def consume(session):
                    consumer_entered.set()
                    assert competing_entered.wait(5)
                    results.append(ctx.pandas("data")["id"].tolist())

                ctx.consume_sql(lambda: consume, datasets=["data"])
            except BaseException as exc:
                errors.append(exc)

        def pandas():
            try:
                assert consumer_entered.wait(5)
                results.append(ctx.pandas("data")["id"].tolist())
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=native, daemon=True),
            threading.Thread(target=pandas, name="competing-pandas", daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        assert results == [[1, 2], [1, 2]]


def test_key_waits_check_cancellation_and_release_waiter_reference():
    locks, cancelled, waiting = KeyedLocks(), threading.Event(), threading.Event()
    errors = []

    def check():
        waiting.set()
        if cancelled.is_set():
            raise ResourceLimitError("cancelled")

    def wait():
        try:
            with locks.hold("key", check_cancelled=check):
                pytest.fail("cancelled waiter acquired the key")
        except ResourceLimitError as exc:
            errors.append(str(exc))

    with locks.hold("key"):
        thread = threading.Thread(target=wait, daemon=True)
        thread.start()
        assert waiting.wait(5)
        cancelled.set()
        thread.join(5)
        assert not thread.is_alive()
        assert locks.entries["key"][1] == 1
    assert errors == ["cancelled"]
    assert locks.entries == {}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM main.items",
        'SELECT * FROM "main"."items"',
        "WITH items AS (SELECT 99 AS id) SELECT * FROM main.items",
        'WITH "ItEmS" AS (SELECT 99 AS id) SELECT * FROM "MaIn"."items"',
        "WITH items AS (SELECT 99 AS id), x AS (SELECT * FROM main.items) SELECT * FROM x",
        "WITH items AS (SELECT 99 AS id) SELECT * FROM (SELECT * FROM main.items) x",
        "WITH x AS (SELECT * FROM items), items AS (SELECT 99 AS id) SELECT * FROM x",
        "WITH items AS (SELECT * FROM items) SELECT * FROM items",
    ],
)
def test_sql_lexical_dependencies_preserve_physical_dataset(query):
    frame = pl.DataFrame({"id": [1, 2]})
    # DuckDB 1.2 rejects self-shadowing CTEs; supported syntax must follow the
    # installed engine's own binding semantics without losing physical inputs.
    with duckdb.connect(config={"threads": 1}) as native:
        native.register("items", frame.to_arrow())
        try:
            expected = native.execute(query).pl()
        except duckdb.Error:
            expected = None
    assert sql_table_names(query) == {"items"}
    with DataContext(frames={"items": frame}) as ctx:
        if expected is None:
            with pytest.raises(SourceError, match="Circular reference"):
                ctx.sql(query)
        else:
            assert ctx.sql(query).equals(expected)


def test_recursive_cte_reference_is_not_a_physical_dataset():
    query = (
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n<3) SELECT * FROM r"
    )
    assert sql_table_names(query) == set()
    with DataContext() as ctx:
        assert ctx.sql(query)["n"].to_list() == [1, 2, 3]


@pytest.mark.parametrize(
    ("query", "dependencies"),
    [
        ("WITH items AS (SELECT 99 AS id) SELECT * FROM items", set()),
        (
            "WITH items AS (SELECT 99 AS id), x AS (SELECT * FROM items) SELECT * FROM x",
            set(),
        ),
        ("SELECT * FROM external.items", set()),
        ("SELECT * FROM attached.main.items", set()),
        (
            "SELECT * FROM items WHERE id IN "
            "(WITH items AS (SELECT 99 AS id) SELECT id FROM items)",
            {"items"},
        ),
        (
            "(WITH items AS (SELECT 99 AS id) SELECT * FROM items) UNION ALL SELECT * FROM items",
            {"items"},
        ),
    ],
)
def test_sql_dependencies_keep_cte_and_catalog_scopes_separate(query, dependencies):
    assert sql_table_names(query) == dependencies


def test_unlimited_materialization_never_serializes_nested_values(monkeypatch):
    frame = pl.DataFrame({"nested": [[1, 2, 3]] * 100})

    def forbidden(*args, **kwargs):
        pytest.fail("unlimited materialization performed size serialization")

    with DataContext() as ctx:
        monkeypatch.setattr(pl.DataFrame, "write_ipc", forbidden)
        ctx._check_materialization("nested", frame)


def test_resolved_native_capability_adds_no_content_hash_passes(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("x\n1\n2\n")
    with DataContext(sources={"data": source(path)}, hash_inputs=True) as ctx:
        ctx.schema("data")
        bound = ctx._catalog.sources["data"]
        before = bound.identity_telemetry()["hash_passes"]
        for _ in range(5):
            assert ctx.execution_kind("data") == "native_sql"
        assert bound.identity_telemetry()["hash_passes"] == before
        assert ctx.sql("SELECT sum(x) FROM data").item() == 3
        assert bound.identity_telemetry()["hash_passes"] == before + 2
        assert ctx.verify_inputs() == {}
        assert bound.identity_telemetry()["hash_passes"] == before + 3


def test_snapshot_lazy_result_limit_precedes_collection(tmp_path, monkeypatch):
    import qavf.sources.collection as collection

    seen, collected = [], []
    original = collection.as_polars

    def record(frame):
        collected.append(frame.height)
        return original(frame)

    monkeypatch.setattr(collection, "as_polars", record)

    class LazySnapshot(DataSource):
        kind, Params = "lazy_snapshot", SourceParams

        def load(self):
            raise AssertionError("expected transactional reader")

        def snapshot_connection(self):
            return "db"

        def read(self, con):
            return (
                pl.DataFrame({"x": range(1000)})
                .lazy()
                .map_batches(
                    lambda frame: (seen.append(frame.height), frame)[1],
                    schema={"x": pl.Int64},
                    predicate_pushdown=False,
                    projection_pushdown=False,
                    slice_pushdown=True,
                )
            )

    path = tmp_path / "input.duckdb"
    duckdb.connect(str(path)).close()
    env = SourceEnvironment(
        connections={"db": ConnectionSpec(type="duckdb", path=str(path), snapshot=True)}
    )
    supplied = LazySnapshot(SourceParams(type="lazy_snapshot", row_budget=3), env)
    with DataContext(sources={"data": supplied}) as ctx:
        with pytest.raises(DatasetNotAvailableError, match="more than 3 rows"):
            ctx.get("data")
        assert len(seen) == 1  # Upstream UDF work is not bounded by an output-row cap.
        assert collected == [4]
        assert not ctx.is_materialized("data")
        assert ctx.resource_usage()["active_connection_reservations"] == 0


def test_observation_replays_one_volatile_lazy_evaluation():
    calls = []

    def change(frame):
        calls.append(1)
        return pl.DataFrame({"id": [len(calls), len(calls) + 1]})

    frame = pl.DataFrame({"id": [0, 0]}).lazy().map_batches(change, schema={"id": pl.Int64})
    with DataContext(frames={"data": frame}) as ctx:
        with ctx.observe("data", "id") as observation:
            assert observation.reduce(lambda plan: plan.select(pl.sum("id"))).item() == 3
            assert [batch.item() for batch in observation.batches(1)] == [1, 2]
            assert observation.reduce(lambda plan: plan.select(pl.sum("id"))).item() == 3
            assert len(calls) == 1
        assert ctx.resource_usage()["owned_spill_bytes"] == 0
        with pytest.raises(RuntimeError, match="ended"):
            observation.scan()


def test_observation_lazy_row_budget_is_checked_on_same_capped_evaluation(monkeypatch):
    import qavf.engine.observation as observations

    calls, staged_rows = [], []
    original = observations._observe_plan

    def record(ctx, plan):
        observed = original(ctx, plan)
        staged_rows.append(observed.reduce(lambda data: data.select(pl.len())).item())
        return observed

    monkeypatch.setattr(observations, "_observe_plan", record)
    plan = (
        pl.DataFrame({"id": range(100)})
        .lazy()
        .map_batches(
            lambda frame: (calls.append(frame.height), frame)[1],
            schema={"id": pl.Int64},
            slice_pushdown=True,
        )
    )
    spec = DatasetSpec(load_policy={"row_budget": 3})
    with DataContext(frames={"data": plan}, specs={"data": spec}) as ctx:
        with pytest.raises(SourceError, match="more than 3 rows"), ctx.observe("data"):
            pytest.fail("over-budget observation was admitted")
        assert len(calls) == 1
        assert staged_rows == [4]
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


def test_existing_spill_observation_survives_source_release_without_loading(tmp_path):
    original = pl.DataFrame({"id": range(10), "payload": ["x" * 10000] * 10})
    with DataContext(frames={"data": original}, cache_max_bytes=0, temp_directory=tmp_path) as ctx:
        before = ctx.resource_usage()["owned_spill_bytes"]
        with ctx.observe("data", ["id"]) as observation:
            assert ctx.resource_usage()["owned_spill_bytes"] == before
            ctx.release(["data"])
            assert ctx.resource_usage()["owned_spill_bytes"] == before
            assert observation.reduce(lambda plan: plan.select(pl.sum("id"))).item() == 45
            assert ctx.resource_usage()["frame_spill_read_count"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


def test_observation_preserves_nested_int128_enum_and_nanosecond_types():
    frame = pl.DataFrame(
        {
            "wide": pl.Series([2**90, -(2**90)], dtype=pl.Int128),
            "nested": pl.Series([[2**90], [None]], dtype=pl.List(pl.Int128)),
            "category": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b", "c"])),
            "time": pl.Series([1, 1001], dtype=pl.Datetime("ns")),
        }
    )
    with DataContext(frames={"data": frame.lazy()}) as ctx:
        with ctx.observe("data") as observation:
            actual = pl.concat(list(observation.batches(1)))
            assert actual.equals(frame)
            reverse = pl.concat(list(observation.batches(1, build=lambda plan: plan.reverse())))
            assert reverse.equals(frame.reverse())
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


@pytest.mark.parametrize("lazy", [False, True])
def test_observation_batch_budget_measures_detached_slice(lazy):
    from qavf.engine.cache_ownership import retained_size

    frame = pl.DataFrame({"id": range(100)})
    with DataContext(
        frames={"data": frame.lazy() if lazy else frame},
        # An eager input must itself pass admission; detached output is still
        # one bounded row even when its source was already held in memory.
        memory_budget_bytes=16 if lazy else frame.estimated_size(),
    ) as ctx:
        with ctx.observe("data") as observation:
            batches = list(observation.batches(1))
            assert [batch.item() for batch in batches] == list(range(100))
            assert all(retained_size(batch) <= 16 for batch in batches)
            assert observation.reduce(lambda plan: plan.slice(99, 1)).item() == 99
        assert pl.concat(batches).equals(frame)
        with ctx.projected_observations({"data": None}) as observations:
            result = observations.collect(observations["data"].scan().slice(99, 1))
            assert result.item() == 99
            assert retained_size(result) <= 16


def test_observation_spill_failure_removes_partial_files(tmp_path):
    plan = pl.DataFrame({"x": range(10000)}).lazy()
    with DataContext(
        frames={"data": plan}, cache_max_bytes=0, spill_max_bytes=1000, temp_directory=tmp_path
    ) as ctx:
        with pytest.raises(ResourceLimitError, match="spill budget"), ctx.observe("data"):
            pytest.fail("oversized observation admitted")
        assert ctx.resource_usage()["owned_spill_bytes"] == 0
        assert not list(ctx._resources.directory.rglob("*.arrow"))


def test_failed_observation_postcondition_releases_all_staged_bytes():
    from qavf.sources.base import SourceIntegrityError

    class ChangingSource(DataSource):
        kind, Params = "changing_observation", SourceParams

        def load(self):
            raise AssertionError("use typed plan")

        def scan_polars(self):
            def read(frame):
                self.load_notes["changed"] = True
                return frame

            return pl.DataFrame({"id": [1, 2]}).lazy().map_batches(read, schema={"id": pl.Int64})

        def verify_identity(self):
            if self.load_notes.get("changed"):
                raise SourceIntegrityError("source changed during observation")

    source = ChangingSource(SourceParams(type="changing_observation"), SourceEnvironment())
    with DataContext(sources={"data": source}) as ctx:
        with pytest.raises(SourceIntegrityError, match="changed"), ctx.observe("data"):
            pytest.fail("failed source identity was published")
        assert ctx.resource_usage()["owned_spill_bytes"] == 0
        assert not list(ctx._resources.directory.rglob("*.arrow"))
