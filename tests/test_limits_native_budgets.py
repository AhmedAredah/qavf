"""Resource composition must never increase an independently configured cap."""

from __future__ import annotations

from dataclasses import replace

import duckdb
import pytest

from qavf.engine.resources import ResourceLimitError, RunResources
from qavf.sources.base import SourceEnvironment, SourceError
from qavf.sources.native import connect


@pytest.mark.parametrize("memory,slots", [(1, 16), (2, 16), (31, 16), (1, 1)])
def test_tiny_targets_reject_native_work_without_exceeding_the_target(tmp_path, memory, slots):
    resources = RunResources(1024, 0, memory, tmp_path, connection_slots=slots)
    try:
        usage = resources.usage()
        assert usage["native_memory_budget_bytes"] == 0
        assert usage["native_memory_bytes_per_connection"] == 0
        assert resources.frames.limit + resources.pandas.limit + resources.memo.limit <= memory // 2
        with pytest.raises(ResourceLimitError, match=r"too small.*native byte"):
            connect(SourceEnvironment(connection_budget=resources))
        assert resources.usage()["active_connection_reservations"] == 0
        assert resources.usage()["peak_connection_reservations"] == 0
    finally:
        resources.close()


@pytest.mark.parametrize("memory,slots", [(32, 16), (33, 16), (101, 3), (1024, 16)])
def test_memory_partition_never_exceeds_the_target(tmp_path, memory, slots):
    resources = RunResources(10_000, 0, memory, tmp_path, connection_slots=slots)
    try:
        tokens = [resources.acquire()[0] for _ in range(slots)]
        native = resources.usage()["native_memory_budget_bytes"]
        cache = resources.frames.limit + resources.pandas.limit + resources.memo.limit
        assert native <= memory // 2
        assert native + cache <= memory
        with pytest.raises(ResourceLimitError, match="capacity exhausted"):
            resources.acquire()
        for token in tokens:
            resources.release(token)
    finally:
        resources.close()


@pytest.mark.parametrize(
    "configured",
    [
        "1MB", "1000000 bytes", "1e6B", "1e3KB", "1000 kilobytes",
        "1 megabyte", "1 megabytes", "0.001 gigabyte", "0.000001 terabytes",
        "1000k", "1m", "0.001g", "0.000001t", "976.5625KiB", "1MiB",
        "0.001GiB", "0.000001TiB", " 1 MB ", "1.0000001MB",
    ],
)
def test_explicit_lower_native_limit_keeps_its_original_exact_setting(tmp_path, configured):
    resources = RunResources(0, 0, 64 * 1024**2, tmp_path, connection_slots=1)
    try:
        with duckdb.connect() as expected:
            expected.execute("SET memory_limit = ?", [configured])
            original = expected.execute("SELECT current_setting('memory_limit')").fetchone()
        env = SourceEnvironment(connection_budget=resources, memory_limit=configured, threads=1)
        with connect(env) as actual:
            assert actual.execute("SELECT current_setting('memory_limit')").fetchone() == original
        assert resources.usage()["active_connection_reservations"] == 0
    finally:
        resources.close()


@pytest.mark.parametrize(
    "configured", [None, "128MB", "1GB", "-1", "-2MB", " -2MB", "none", "null"]
)
def test_reservation_tightens_larger_or_unlimited_native_setting(tmp_path, configured):
    resources = RunResources(0, 0, 64 * 1024**2, tmp_path, connection_slots=1)
    try:
        env = SourceEnvironment(connection_budget=resources, memory_limit=configured, threads=1)
        with connect(env) as actual:
            assert actual.execute("SELECT current_setting('memory_limit')").fetchone() == (
                "32.0 MiB",
            )
    finally:
        resources.close()


def test_invalid_explicit_cap_is_not_hidden_by_the_run_reservation(tmp_path):
    resources = RunResources(0, 0, 64 * 1024**2, tmp_path, connection_slots=1)
    try:
        env = SourceEnvironment(connection_budget=resources, memory_limit="invalid")
        with pytest.raises(duckdb.Error):
            connect(env)
        assert resources.usage()["active_connection_reservations"] == 0
        with connect(replace(env, memory_limit="1MB")) as actual:
            assert actual.execute("SELECT 1").fetchone() == (1,)
    finally:
        resources.close()


def test_custom_zero_reservation_rejects_before_native_creation_and_releases(monkeypatch):
    class ZeroBudget:
        def __init__(self):
            self.released = []

        def acquire(self):
            return "empty", 0, 0

        def release(self, token):
            self.released.append(token)

    budget = ZeroBudget()

    def unexpected_connect():
        pytest.fail("a zero native allocation must not create a connection")

    monkeypatch.setattr(duckdb, "connect", unexpected_connect)
    with pytest.raises(SourceError, match="no capacity"):
        connect(SourceEnvironment(connection_budget=budget))
    assert budget.released == ["empty"]
