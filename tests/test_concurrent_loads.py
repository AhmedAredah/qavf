"""Concurrent loading, shared work and tolerance settling.

Concurrent, once-only dataset loading, shared aggregates, reused connections
and a thread budget, and columnar settling of within-tolerance rows that keeps
the comparison's exact semantics.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import numpy as np
import polars as pl
import pytest

import qavf.regression  # noqa: F401  (the package re-exports a function of the module's name)
from qavf.engine.context import DataContext
from qavf.engine.runner import QARunner
from qavf.settings import QAVFSettings
from qavf.sources.base import DataSource, SourceEnvironment, SourceParams
from qavf.testing import discovered_rules

compare_module = sys.modules["qavf.regression.compare_outputs"]
comparison_execution = sys.modules["qavf.regression._comparison"]
REGISTRY = discovered_rules(entry_points=False)


class _Params(SourceParams):
    type: str = "test"


class _BarrierSource(DataSource):
    """Loads only if another load reaches the barrier at the same time."""

    kind: ClassVar[str] = "barrier"
    Params: ClassVar[type[SourceParams]] = _Params

    def __init__(self, barrier: threading.Barrier, counter: list[int]) -> None:
        super().__init__(_Params(), SourceEnvironment())
        self.barrier, self.counter = barrier, counter

    def bind(self, env):
        # Synchronization and observation handles are intentionally shared; source
        # state and its environment belong to the new validation session.
        bound = type(self)(self.barrier, self.counter)
        bound.env = self.bound_environment(env)
        return bound

    def load(self) -> pl.DataFrame:
        self.counter.append(1)
        self.barrier.wait(timeout=10)
        return pl.DataFrame({"a": [1]})


class _GatedSource(DataSource):
    kind: ClassVar[str] = "gated"
    Params: ClassVar[type[SourceParams]] = _Params

    def __init__(self, gate: threading.Event, counter: list[int]) -> None:
        super().__init__(_Params(), SourceEnvironment())
        self.gate, self.counter = gate, counter

    def bind(self, env):
        bound = type(self)(self.gate, self.counter)
        bound.env = self.bound_environment(env)
        return bound

    def load(self) -> pl.DataFrame:
        self.counter.append(1)
        assert self.gate.wait(timeout=10)
        return pl.DataFrame({"a": [1]})


# =========================================================================
# Independent datasets load concurrently, each exactly once
# =========================================================================
def test_independent_loads_run_concurrently() -> None:
    barrier, counter = threading.Barrier(2), []
    ctx = DataContext(
        sources={"a": _BarrierSource(barrier, counter), "b": _BarrierSource(barrier, counter)}
    )
    with ThreadPoolExecutor(2) as pool:
        frames = list(pool.map(ctx.get, ["a", "b"]))
    assert [f.height for f in frames] == [1, 1] and len(counter) == 2


def test_serial_loading_can_be_requested() -> None:
    barrier, counter = threading.Barrier(2), []
    ctx = DataContext(
        sources={"a": _BarrierSource(barrier, counter), "b": _BarrierSource(barrier, counter)},
        concurrent_loads=False,
    )
    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(ctx.has, ["a", "b"]))
    # The loads never overlapped, so the barrier could not be passed.
    assert outcomes == [False, False]


def test_concurrent_requests_for_one_dataset_load_it_once() -> None:
    gate, counter = threading.Event(), []
    ctx = DataContext(sources={"a": _GatedSource(gate, counter)})
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(ctx.get, "a") for _ in range(8)]
        gate.set()
        heights = [f.result(timeout=10).height for f in futures]
    assert heights == [1] * 8 and len(counter) == 1


def test_a_slow_load_does_not_block_cached_datasets() -> None:
    gate, counter = threading.Event(), []
    ctx = DataContext(
        frames={"ready": pl.DataFrame({"x": [1]})}, sources={"slow": _GatedSource(gate, counter)}
    )
    with ThreadPoolExecutor(2) as pool:
        slow = pool.submit(ctx.get, "slow")
        while not counter:  # the slow load has started and holds its own lock
            pass
        assert pool.submit(ctx.get, "ready").result(timeout=5).height == 1
        gate.set()
        assert slow.result(timeout=10).height == 1


def test_memo_runs_each_key_once_and_keys_in_parallel() -> None:
    ctx = DataContext()
    barrier, calls = threading.Barrier(2), []

    def compute(key: str):
        def run():
            calls.append(key)
            barrier.wait(timeout=10)  # passes only if both keys compute at once
            return key.upper()

        return run

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda k: ctx.memo(k, compute(k)), ["x", "y", "x", "y"]))
    assert results == ["X", "Y", "X", "Y"] and sorted(calls) == ["x", "y"]


# =========================================================================
# Repeated work is shared; connections are reused; threads are budgeted
# =========================================================================
def test_temporal_rules_share_one_aggregate_per_series(monkeypatch) -> None:
    from qavf.rules import temporal_checks

    calls = []
    real = temporal_checks._aggregate_raw

    def counting(ctx, spec, blocked):
        calls.append(spec.label)
        return real(ctx, spec, blocked)

    monkeypatch.setattr(temporal_checks, "_aggregate_raw", counting)
    series = [{"dataset": "s", "value": "v", "group_by": ["g"]}]
    frame = pl.DataFrame(
        {"g": ["a"] * 4, "year": [2020, 2021, 2022, 2023], "v": [1.0, 1.1, 1.2, 1.3]}
    )
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "datasets": {"s": {}},
            "engine": {"run_unconfigured_rules": False, "parallel": True},
            "rules": {
                "TMP-001": {"params": {"series": series}},
                "TMP-004": {"params": {"series": series}},
            },
        }
    )
    report = QARunner(settings, REGISTRY).run(frames={"s": frame})
    assert all(not r.failed for r in report.results)
    assert calls == ["s.v"]


def test_unusable_shared_aggregates_are_reported_by_every_rule() -> None:
    series = [{"dataset": "s", "value": "missing_column"}]
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "datasets": {"s": {}},
            "engine": {"run_unconfigured_rules": False},
            "rules": {
                "TMP-001": {"params": {"series": series}},
                "TMP-004": {"params": {"series": series}},
            },
        }
    )
    report = QARunner(settings, REGISTRY).run(frames={"s": pl.DataFrame({"year": [2020]})})
    blocked = {r.rule_id for r in report.results if "missing_column" in r.message}
    assert blocked == {"TMP-001", "TMP-004"}


def test_sql_connections_are_reused_per_thread_and_closed() -> None:
    ctx = DataContext(frames={"t": pl.DataFrame({"a": [1, 2, 3]})})
    for _ in range(5):
        assert ctx.sql("SELECT count(*) AS n FROM t")["n"][0] == 3
    assert ctx._sql_connections.open_count == 1
    ctx.close()
    assert ctx._sql_connections.open_count == 0


def test_parallel_runs_budget_duckdb_threads(monkeypatch) -> None:
    import qavf.engine.runner as runner_module

    monkeypatch.setattr(runner_module.os, "cpu_count", lambda: 8)
    seen = {}
    real = DataContext.from_settings

    def spy(settings, frames=None, source_registry=None, threads=None, *, execution_workers=None):
        seen["threads"] = threads
        seen["execution_workers"] = execution_workers
        result = real(
            settings, frames, source_registry, threads, execution_workers=execution_workers
        )
        seen["connection_slots"] = result._resources.connection_slots
        return result

    monkeypatch.setattr(DataContext, "from_settings", staticmethod(spy))
    settings = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "engine": {"parallel": True, "max_workers": 4}}
    )
    QARunner(settings, REGISTRY).run()
    assert seen["threads"] == 2
    assert seen["execution_workers"] == 4
    assert seen["connection_slots"] == 8


# =========================================================================
# Within-tolerance rows are settled columnarly, with identical results
# =========================================================================
def _summary(result):
    return (
        result.equal,
        result.values_changed,
        result.rows_changed,
        result.rows_added,
        result.rows_removed,
        result.nonfinite_changes,
        result.max_abs_diff,
        result.max_rel_diff,
        # JSON, because NaN != NaN would make identical records compare unequal.
        json.dumps([d.model_dump() for d in result.differences], sort_keys=True, default=str),
    )


@pytest.mark.parametrize("seed", range(6))
def test_prefilter_never_changes_the_result(monkeypatch, seed) -> None:
    rng = np.random.default_rng(seed)
    n = 3000
    base = pl.DataFrame(
        {
            "k": np.arange(n),
            "f": rng.normal(size=n) * 10.0 ** rng.integers(-5, 5, n),
            "i": rng.integers(-5, 5, n),
            "s": pl.Series([["a", "b", None][j] for j in rng.integers(0, 3, n)], dtype=pl.String),
        }
    )
    noise = rng.choice([0.0, 1e-9, 1e-7, 1e-5, 1e-3], n)
    special = [[None, math.nan, math.inf, -math.inf, 0.0][j] for j in rng.integers(0, 5, n)]
    cand = base.with_columns(
        (pl.col("f") * (1 + pl.Series(noise))).alias("f"),
        pl.when(pl.Series(rng.random(n) < 0.02))
        .then(pl.col("i") + 1)
        .otherwise(pl.col("i"))
        .alias("i"),
    ).with_columns(
        pl.when(pl.Series(rng.random(n) < 0.03))
        .then(pl.Series(special, dtype=pl.Float64))
        .otherwise(pl.col("f"))
        .alias("f")
    )
    results = {}
    for flag in (True, False):
        monkeypatch.setattr(comparison_execution, "_TOLERANCE_PREFILTER", flag)
        results[flag] = _summary(
            compare_module.compare_frames(
                base, cand, ["k"], rtol=1e-6, atol=1e-12, max_differences=10_000
            )
        )
    assert results[True] == results[False]


def test_within_tolerance_rows_never_reach_python_comparison(monkeypatch) -> None:
    sizes = []
    real = comparison_execution._cell_change

    def spy(*args, **kwargs):
        sizes.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(comparison_execution, "_cell_change", spy)
    n = 50_000
    base = pl.DataFrame({"k": np.arange(n), "v": np.linspace(1, 100, n)})
    cand = base.with_columns(pl.col("v") * (1 + 1e-12))
    assert compare_module.compare_frames(base, cand, ["k"]).equal
    # Matching scalar values never enter Python cell traversal.
    assert sizes == []
