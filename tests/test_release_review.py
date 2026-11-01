"""Defects found reviewing the imported 0.6.0 changes, kept as regression tests."""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import polars as pl

from qavf.compare.evidence import bound_check
from qavf.compare.results import CheckResult
from qavf.engine.resources import FrameStore, RunResources
from qavf.engine.spill import SpillBudget
from qavf.engine.sqlite_store import Store


class _Opened:
    """A store opened on another thread, as an abandoned timed-out rule leaves it."""

    def __init__(self, resources: RunResources) -> None:
        self.resources, self.store = resources, None

    def __call__(self) -> None:
        self.store = Store(self.resources)


def test_run_cleanup_closes_a_store_opened_on_another_thread(tmp_path: Path) -> None:
    resources = RunResources(1 << 20, 100_000_000, None, tmp_path)
    opened = _Opened(resources)
    worker = threading.Thread(target=opened)
    worker.start()
    worker.join()
    assert opened.store is not None
    opened.store.execute("CREATE TABLE t (a INTEGER PRIMARY KEY)")
    resources.close()
    assert not resources.directory.exists()


class _Reserve:
    """Reserve spill bytes on another thread and record that it returned."""

    def __init__(self, budget: SpillBudget, size: int) -> None:
        self.budget, self.size, self.done = budget, size, threading.Event()

    def __call__(self) -> None:
        self.budget.reserve(self.size)
        self.done.set()


def _never_cancelled() -> None:
    """A cancellation check that never cancels."""


def test_a_spill_write_waits_for_space_lent_to_a_native_write() -> None:
    budget = SpillBudget(1_000)
    credit = budget.acquire_growth(_never_cancelled)
    reserve = _Reserve(budget, 10)
    writer = threading.Thread(target=reserve)
    writer.start()
    assert not reserve.done.wait(0.3)  # waits instead of failing
    budget.settle_growth(credit, 0)
    assert reserve.done.wait(5) and budget.used == 10
    writer.join()


def _publish(store: FrameStore, key: str, frame: pl.DataFrame) -> None:
    store.publish({key: frame}, _no_metadata, contextlib.nullcontext())


def _no_metadata() -> None:
    """Publication metadata that records nothing."""


def test_new_data_replaces_the_least_recently_used_in_memory(tmp_path: Path) -> None:
    frame = pl.DataFrame({"x": range(100_000)})
    store = FrameStore(int(frame.estimated_size() * 1.5), 1_000_000_000, tmp_path)
    try:
        _publish(store, "old", frame)
        _publish(store, "new", frame)
        assert "new" in store.data and "old" in store.paths
    finally:
        store.clear()


def test_evidence_bounds_keep_metrics_before_sample_records() -> None:
    metrics = {"test": "poisson", "family_size": 5, "strata": ["s" * 40] * 8, "change": "7/3"}
    check = CheckResult(
        check="c", passed=False, message="m", affected_count=50,
        affected_records=[{"group": i, "detail": "x" * 80} for i in range(50)], metrics=metrics,
    )  # fmt: skip
    bounded = bound_check(check, 50, 2_000)
    assert bounded.metrics == metrics and bounded.truncated
    assert 0 < len(bounded.affected_records) < 50


def test_a_lower_bound_reads_at_least_on_the_failure_card() -> None:
    from qavf.reports.html import _affected
    from qavf.reports.models import RuleResult

    result = RuleResult(
        rule_id="INP-002", name="n", description="d", category="input", severity="ERROR",
        status="FAIL", message="m", affected_count=5, affected_count_exact=False,
        total_records=10, unit="rows",
    )  # fmt: skip
    assert _affected(result) == "at least 5 rows of 10"


def test_native_csv_missing_a_declared_column_still_checks_the_others(tmp_path: Path) -> None:
    from qavf import validate
    from qavf.engine.severity import Status

    path = tmp_path / "x.csv"
    path.write_bytes(b"a,b\n1,2\n-3,4\n")
    schema = {"columns": {"a": {"dtype": "int", "ge": 0}, "missing": {"dtype": "int"}}}
    report = validate(str(path), schema=schema, backend="duckdb")
    assert all(r.status is Status.PASS for r in report.results_for("INP-001"))
    assert any(r.status is Status.FAIL for r in report.results_for("INP-002"))
    [limit] = [r for r in report.results_for("INP-003") if r.status is Status.FAIL]
    assert limit.affected_count == 1
