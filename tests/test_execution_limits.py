"""Execution limits and telemetry: rule deadlines, load retries, row budgets, DuckDB
memory limits and per-rule load time."""

from __future__ import annotations

import threading
import time
from typing import ClassVar

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings
from qavf.sources.base import (
    DataSource,
    SourceEnvironment,
    SourceError,
    SourceNotFoundError,
    SourceParams,
)

RELEASE = threading.Event()


class _Hangs(Rule):
    rule_id: ClassVar[str] = "SLOW-1"
    name: ClassVar[str] = "Hangs"
    description: ClassVar[str] = "Blocks until released."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx):
        RELEASE.wait(30)
        return Finding.ok("finally")


class _Quick(Rule):
    rule_id: ClassVar[str] = "FAST-1"
    name: ClassVar[str] = "Quick"
    description: ClassVar[str] = "Returns at once."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx):
        return Finding.ok("quick")


def _registry(*rules) -> RuleRegistry:
    registry = RuleRegistry()
    for rule in rules:
        registry.register(rule)
    return registry


@pytest.mark.parametrize("parallel", [False, True])
def test_a_rule_past_its_deadline_is_unassessed_and_the_report_is_produced(parallel) -> None:
    RELEASE.clear()
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"parallel": parallel, "max_workers": 2, "rule_timeout_s": 0.3},
            "rules": {"SLOW-1": {}, "FAST-1": {}},
        }
    )
    started = time.monotonic()
    report = QARunner(settings, _registry(_Hangs, _Quick)).run()
    elapsed = time.monotonic() - started
    RELEASE.set()  # let the abandoned daemon thread finish
    assert elapsed < 5
    slow = report.get("SLOW-1")
    assert slow.status is Status.ERROR and "did not finish within 0.3s" in slow.message
    assert report.get("FAST-1").status is Status.PASS
    assert report.verdict() == "incomplete"


def test_the_deadline_does_not_affect_rules_that_finish(registry) -> None:
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"rule_timeout_s": 30},
            "rules": {"FAST-1": {}},
        }
    )
    assert QARunner(settings, _registry(_Quick)).run().get("FAST-1").status is Status.PASS


# ------------------------------------------------------------------ retries
class _FlakyParams(SourceParams):
    type: str = "flaky"
    failures: int = 0
    missing: bool = False


class _Flaky(DataSource):
    kind: ClassVar[str] = "flaky"
    Params: ClassVar[type[SourceParams]] = _FlakyParams

    def __init__(self, params, env):
        super().__init__(params, env)
        self.calls = 0

    def bind(self, env):
        return type(self)(self.params, self.bound_environment(env))

    def load(self) -> pl.DataFrame:
        self.calls += 1
        if self.params.missing:
            raise SourceNotFoundError("not there")
        if self.calls <= self.params.failures:
            raise SourceError("connection reset", retryable=True)
        return pl.DataFrame({"a": [1, 2, 3]})


def _ctx(**params) -> tuple[DataContext, _Flaky]:
    source = _Flaky(_FlakyParams(**params), SourceEnvironment())
    context = DataContext(sources={"d": source})
    # Calls are run state; inspect the explicit bound session, not the reusable input.
    return context, context._catalog.sources["d"]


def test_transient_failures_are_retried() -> None:
    ctx, source = _ctx(failures=2, retries=2, retry_delay_s=0)
    assert ctx.get("d").height == 3 and source.calls == 3
    assert ctx.describe()[0].notes["retries_used"] == 2


def test_retries_are_bounded() -> None:
    ctx, source = _ctx(failures=5, retries=1, retry_delay_s=0)
    assert not ctx.has("d") and source.calls == 2


def test_missing_data_is_not_retried() -> None:
    ctx, source = _ctx(missing=True, retries=3, retry_delay_s=0)
    assert not ctx.has("d") and source.calls == 1


# ------------------------------------------------------------------ row budgets
def test_row_budgets_for_files_queries_and_other_sources(tmp_path) -> None:
    path = tmp_path / "d.parquet"
    pl.DataFrame({"a": list(range(100))}).write_parquet(path)
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "datasets": {
                "file": {"source": {"type": "file", "path": str(path), "row_budget": 10}},
                "ok": {"source": {"type": "file", "path": str(path), "row_budget": 100}},
                "query": {
                    "source": {
                        "type": "query",
                        "sql": f"SELECT * FROM '{path.as_posix()}'",
                        "row_budget": 5,
                    }
                },
            },
        }
    )
    ctx = DataContext.from_settings(settings)
    assert "more than 10 rows" in str(ctx.load_error("file"))
    assert ctx.get("ok").height == 100
    assert "more than 5 rows" in str(ctx.load_error("query"))
    flaky, _ = _ctx(row_budget=2)
    assert "more than 2 rows" in str(flaky.load_error("d"))


def test_duckdb_memory_limit_is_applied() -> None:
    settings = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "engine": {"duckdb_memory_limit": "512MB"}}
    )
    ctx = DataContext.from_settings(settings)
    limit = ctx.sql("SELECT current_setting('memory_limit') AS m")["m"][0]
    assert limit.replace(" ", "").lower().startswith(("512", "488"))  # MB vs MiB display


# ------------------------------------------------------------------ telemetry
class _SlowParams(SourceParams):
    type: str = "slow"


class _Slow(DataSource):
    kind: ClassVar[str] = "slow"
    Params: ClassVar[type[SourceParams]] = _SlowParams

    def load(self) -> pl.DataFrame:
        time.sleep(0.2)
        return pl.DataFrame({"x": [1]})


class _Reads(Rule):
    rule_id: ClassVar[str] = "READ-1"
    name: ClassVar[str] = "Reads"
    description: ClassVar[str] = "Reads a dataset."
    category: ClassVar[str] = "test"
    requires: ClassVar[tuple[str, ...]] = ("slow",)

    def evaluate(self, ctx):
        return Finding.ok("read")


def test_rule_timing_includes_and_reports_data_loading() -> None:
    settings = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "rules": {"READ-1": {}}}
    )
    ctx = DataContext(sources={"slow": _Slow(_SlowParams(), SourceEnvironment())})
    result = QARunner(settings, _registry(_Reads)).run(ctx=ctx).get("READ-1")
    assert result.load_time_s >= 0.19 and result.execution_time_s >= result.load_time_s
