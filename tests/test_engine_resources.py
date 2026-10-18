"""Engine resource, SQL and lifecycle contracts."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.resources import ResourceLimitError, RunResources
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import ConnectionSpec, QAVFSettings
from qavf.sources.base import DataSource, SourceEnvironment, SourceError, SourceParams
from qavf.sources.file import FileSource


def settings(**engine):
    return QAVFSettings.model_validate({"plugins": {"entry_points": False}, "engine": engine})


def test_sql_count_and_sample_share_one_evaluation():
    with DataContext() as ctx:
        ctx._connection([]).execute("CREATE SEQUENCE seq START 1")
        sample, count = ctx.sql_sample("SELECT 42 AS bad WHERE nextval('seq') % 2 = 0", 5)
        assert count == 0 and sample.is_empty()
        assert ctx._connection([]).execute("SELECT nextval('seq')").fetchone() == (2,)


def test_namespace_binding_and_using_join(tmp_path):
    database = tmp_path / "live.duckdb"
    with duckdb.connect(str(database)) as con:
        con.execute("CREATE TABLE t AS SELECT 7 AS x")
    cfg = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "connections": {"live": {"type": "duckdb", "path": str(database)}},
            "datasets": {"t": {}},
        }
    )
    with DataContext.from_settings(cfg) as ctx:
        assert ctx.sql("SELECT * FROM live.t").to_dicts() == [{"x": 7}]
        assert ctx.referenced("SELECT * FROM live.t") == []
    with DataContext(frames={"a": pl.DataFrame({"x": [1]}), "b": pl.DataFrame({"x": [1]})}) as ctx:
        assert ctx.sql("SELECT * FROM a JOIN b USING(x)").height == 1


def test_native_sql_projection_never_calls_file_load(tmp_path, monkeypatch):
    path = tmp_path / "wide.parquet"
    pl.DataFrame({"x": range(1000), "large": ["x" * 1000] * 1000}).write_parquet(path)
    cfg = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "datasets": {"d": {"path": str(path)}}}
    )

    def forbidden(self):
        raise AssertionError("full source collection is forbidden in this test")

    monkeypatch.setattr(FileSource, "load", forbidden)
    with DataContext.from_settings(cfg) as ctx:
        assert ctx.has("d")
        assert ctx.sql("SELECT sum(x) AS total FROM d").item() == 499500
        assert not ctx.is_materialized("d")
        assert ctx.describe()[0].rows is None
        assert ctx.query_plans()[0]["execution"] == "native_scan"


def test_cache_spill_is_exact_bounded_and_owned(tmp_path):
    ctx = DataContext(
        frames={"a": pl.DataFrame({"x": range(1000)}), "b": pl.DataFrame({"x": range(1000)})},
        cache_max_bytes=100,
        spill_max_bytes=100000,
        temp_directory=tmp_path,
    )
    directory = ctx._resources.directory
    assert ctx.get("a")["x"].sum() == 499500
    usage = ctx.resource_usage()
    assert usage["retained_cache_bytes"] <= 100
    assert 0 < usage["frame_spill_bytes"] <= 50000
    assert len(ctx.describe()) == 2
    ctx.close()
    assert not directory.exists()
    assert len(ctx.describe()) == 2


def test_zero_spill_budget_fails_without_creating_unbounded_files(tmp_path):
    with pytest.raises(ResourceLimitError):
        DataContext(
            frames={"d": pl.DataFrame({"x": range(100)})},
            cache_max_bytes=0,
            spill_max_bytes=0,
            temp_directory=tmp_path,
        )
    assert not list(tmp_path.rglob("*.arrow"))


def test_native_connection_reservations_are_aggregate(tmp_path):
    resources = RunResources(1000, 10000, 1000000, tmp_path, connection_slots=2)
    first = resources.acquire()
    second = resources.acquire()
    assert first[1] + second[1] <= 5000
    assert first[2] + second[2] <= 500000
    with pytest.raises(ResourceLimitError, match="capacity"):
        resources.acquire()
    resources.release(first[0])
    resources.acquire()
    resources.close()


class Params(SourceParams):
    type: str = "audit"


class RetrySource(DataSource):
    kind = "audit"
    Params = Params

    def __init__(self):
        super().__init__(Params(retries=1, retry_delay_s=0, row_budget=1), SourceEnvironment())
        self.calls = 0

    def load(self):
        self.calls += 1
        if self.calls == 1:
            raise SourceError("temporary", retryable=True)
        return pl.DataFrame({"x": [1]})


def test_ad_hoc_loading_uses_retry_policy():
    source = RetrySource()

    class Registry:
        def build(self, config, env):
            return source

    with DataContext(source_registry=Registry()) as ctx:
        assert ctx.load_source({"type": "audit"}).height == 1
        assert source.calls == 2


def test_snapshot_execution_failure_does_not_publish_partial_data(tmp_path):
    database = tmp_path / "live.duckdb"
    with duckdb.connect(str(database)) as con:
        con.execute("CREATE TABLE t AS SELECT 'x' AS v")
    cfg = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "connections": {"live": {"type": "duckdb", "path": str(database), "snapshot": True}},
            "datasets": {
                "a_ok": {"source": {"type": "database", "connection": "live", "table": "t"}},
                "z_bad": {
                    "source": {"type": "query", "sql": "SELECT CAST(v AS BIGINT) FROM live.t"}
                },
            },
        }
    )
    with DataContext.from_settings(cfg) as ctx:
        assert not ctx.has("a_ok") and not ctx.has("z_bad")
        assert ctx.describe() == []


def test_snapshot_groups_obey_serial_loads():
    active = 0
    peak = 0
    lock = threading.Lock()

    class Snapshot(RetrySource):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.env = SourceEnvironment(connections={
                name: ConnectionSpec(type="duckdb", path="mock.duckdb")
            })

        def bind(self, env):
            bound = type(self)(self.name)
            bound.env = self.bound_environment(env)
            return bound

        def snapshot_connection(self):
            return self.name

        def read(self, con):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            time.sleep(0.02)
            with lock:
                active -= 1
            return pl.DataFrame({"x": [1]})

    class Connection:
        def execute(self, q):
            return self

        def close(self):
            pass

    with (
        DataContext(
            sources={"a": Snapshot("a"), "b": Snapshot("b")}, concurrent_loads=False
        ) as ctx,
        patch("qavf.engine.connections.connect", return_value=Connection()),
        ThreadPoolExecutor(2) as pool,
    ):
        assert all(pool.map(ctx.has, ["a", "b"]))
    assert peak == 1


def test_timeout_never_replaces_occupied_worker_or_accumulates_next_run():
    gate = threading.Event()
    entered = threading.Event()
    active = 0
    peak = 0
    lock = threading.Lock()

    class Slow(Rule):
        rule_id = "S"
        name = "slow"
        description = "audit"
        category = "audit"

        def evaluate(self, ctx):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            entered.set()
            gate.wait(2)
            with lock:
                active -= 1
            return Finding.ok("done")

    registry = RuleRegistry()
    for i in range(4):
        registry.register(type(f"Slow{i}", (Slow,), {"rule_id": f"S-{i}"}))
    cfg = settings(max_workers=1, rule_timeout_s=0.02, timeout_grace_s=0)
    try:
        report = QARunner(cfg, registry).run()
        second = QARunner(cfg, registry).run()
        assert peak == 1 and entered.is_set()
        assert all(result.status is Status.ERROR for result in report.results + second.results)
    finally:
        gate.set()
    deadline = time.monotonic() + 2
    while active and time.monotonic() < deadline:
        time.sleep(0.005)


def test_native_timeout_cleanup_does_not_wait_for_udf():
    registry = RuleRegistry()

    class Native(Rule):
        rule_id = "NATIVE"
        name = "native"
        description = "audit"
        category = "audit"

        def evaluate(self, ctx):
            con = ctx._connection([])
            con.create_function(
                "slow_udf", lambda: time.sleep(0.4) or 1, return_type="INTEGER", side_effects=True
            )
            return Finding.ok(str(ctx.sql("SELECT slow_udf()").item()))

    registry.register(Native)
    started = time.monotonic()
    report = QARunner(settings(rule_timeout_s=0.03, timeout_grace_s=0), registry).run()
    assert time.monotonic() - started < 0.3
    assert report.get("NATIVE").status is Status.ERROR
    time.sleep(0.45)


def test_evidence_budget_truncates_without_changing_exact_count():
    registry = RuleRegistry()

    class Large(Rule):
        rule_id = "LARGE"
        name = "large"
        description = "audit"
        category = "audit"

        def evaluate(self, ctx):
            return Finding.fail(
                "bad",
                affected=[{"v": "x" * 10000}],
                affected_count=999,
                metrics={"giant": "y" * 10000},
            )

    registry.register(Large)
    report = QARunner(settings(max_evidence_bytes=200), registry).run()
    result = report.get("LARGE")
    assert result.affected_count == 999 and result.status is Status.FAIL
    assert result.affected_records == [] and result.metrics == {}
    assert "truncated" in result.message


def test_zero_evidence_bytes_preserves_coverage_outcomes():
    registry = RuleRegistry()

    class Covered(Rule):
        rule_id = "COVER"
        name = "coverage"
        description = "audit"
        category = "audit"

        def evaluate(self, ctx):
            return Finding.ok(
                "assessed", metrics={"passed_checks": ["one", "two"], "detail": "x" * 10000}
            )

    registry.register(Covered)
    cfg = settings(max_evidence_bytes=0, require=["COVER[one]"])
    report = QARunner(cfg, registry).run()
    assert report.get("COVER").status is Status.PASS
    assert report.get("COVER").metrics["passed_checks"] == ["one", "two"]
    assert report.verdict() == "passed"


def test_complete_report_budget_trims_optional_evidence_and_fails_closed():
    registry = RuleRegistry()

    class Large(Rule):
        rule_id = "BIG"
        name = "large"
        description = "audit"
        category = "audit"

        def evaluate(self, ctx):
            return Finding.fail("bad", affected=[{"value": "x" * 20000}], affected_count=20)

    registry.register(Large)
    report = QARunner(settings(max_report_bytes=10000), registry).run()
    assert report.get("BIG").affected_count == 20 and report.get("BIG").status is Status.FAIL
    assert report.get("BIG").affected_records == []
    assert len(report.model_dump_json().encode()) <= 10000
    with pytest.raises(ResourceLimitError, match="outcome ledger"):
        QARunner(settings(max_report_bytes=1), registry).run()


def test_plugin_error_log_redacts_credentials(caplog):
    registry = RuleRegistry()

    class Crash(Rule):
        rule_id = "CRASH"
        name = "crash"
        description = "audit"
        category = "audit"

        def evaluate(self, ctx):
            raise RuntimeError("password=secret")

    registry.register(Crash)
    report = QARunner(settings(), registry).run()
    assert report.get("CRASH").status is Status.ERROR
    assert "secret" not in caplog.text


def test_temporary_directory_resolves_against_config_location(tmp_path):
    from pathlib import Path

    cfg = settings(temp_directory=Path("temporary"))
    cfg.source_path = tmp_path / "config.yml"
    with DataContext.from_settings(cfg) as ctx:
        assert ctx._resources.directory.parent == tmp_path / "temporary"
        assert ctx.env.temp_directory.parent == ctx._resources.directory
