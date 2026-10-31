"""Regression contracts for cleanup, cache concurrency and evidence admission."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from qavf.engine import resources as resources_module
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.registry import Finding
from qavf.engine.resources import FrameStore, ResourceLimitError, RunResources
from qavf.engine.row_locator import FileIndex, RowLocator
from qavf.engine.runner import QARunner
from qavf.engine.session import RunSession
from qavf.engine.spill import SpillFile
from qavf.engine.sqlite_store import Store
from qavf.settings import QAVFSettings
from qavf.sources.native import OwnedConnection


def test_finding_new_metadata_preserves_positional_plugin_contract():
    finding = Finding(False, "bad", None, 2, 10, "rows", 2)
    assert finding.affected_count == 2 and finding.total_records == 10
    assert finding.unit == "rows" and finding.affected_rows == 2
    assert finding.affected_count_exact and finding.row_column == "_row"


def _unlink_gate(monkeypatch, directory):
    original = Path.unlink
    blocked = Event()
    blocked.set()

    def unlink(path, *args, **kwargs):
        if blocked.is_set() and directory in path.parents:
            raise PermissionError("injected sharing violation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    return blocked


def test_run_close_attempts_memory_after_file_failure_and_retries(tmp_path, monkeypatch):
    resources = RunResources(32, 50_000, None, tmp_path)
    resources.frames["large"] = pl.DataFrame({"x": range(100)})
    resources.frames["small"] = pl.DataFrame({"x": [1]})
    resources.memo["answer"] = 1
    gate = _unlink_gate(monkeypatch, resources.directory)
    with pytest.raises((PermissionError, ExceptionGroup)):
        resources.close()
    assert resources.closed and not resources.cleanup_complete
    assert not resources.frames.data
    assert resources.retained_budget.used == 0
    assert resources.spill_budget.used > 0
    assert resources.directory.exists()
    with pytest.raises(ResourceLimitError, match="closed"):
        resources.frames["late"] = pl.DataFrame({"x": [1]})
    gate.clear()
    resources.close()
    resources.close()
    assert resources.cleanup_complete
    assert resources.spill_budget.used == resources.retained_budget.used == 0
    assert not resources.directory.exists()


def test_frame_clear_retains_failed_removals_for_retry(tmp_path, monkeypatch):
    frames = FrameStore(8, 20_000, tmp_path)
    frames["spill"] = pl.DataFrame({"x": range(10)})
    frames["memory"] = pl.DataFrame({"x": [1]})
    gate = _unlink_gate(monkeypatch, frames.directory)
    with pytest.raises(PermissionError):
        frames.clear()
    assert frames.bytes == 0 and not frames.data
    assert frames.spill_budget.used > 0
    gate.clear()
    frames.clear()
    assert frames.spill_budget.used == 0
    assert not list(frames.directory.iterdir())


def test_run_directory_survives_live_spill_lease(tmp_path):
    resources = RunResources(0, 20_000, None, tmp_path)
    resources.frames["d"] = pl.DataFrame({"x": [7]})
    _, lease = resources.frames.observe("d")
    resources.close()
    assert resources.directory.exists() and not resources.cleanup_complete
    assert pl.read_ipc(lease.path)["x"].to_list() == [7]
    lease.close()
    resources.close()
    assert resources.cleanup_complete and not resources.directory.exists()


def test_empty_file_lease_keeps_directory_until_final_release(tmp_path):
    resources = RunResources(0, 20_000, None, tmp_path)
    owned = SpillFile(resources.directory / "empty", resources.spill_budget)
    with owned.writer():
        pass
    lease = owned.borrow()
    resources.close()
    assert resources.spill_budget.used == 0 and resources.spill_budget.has_files
    assert lease.path.exists() and not resources.cleanup_complete
    lease.close()
    resources.close()
    assert resources.cleanup_complete


def test_live_memory_lease_keeps_cleanup_incomplete(tmp_path):
    resources = RunResources(32, 20_000, None, tmp_path)
    resources.frames["d"] = pl.DataFrame({"x": [7]})
    frame, lease = resources.frames.observe("d")
    resources.close()
    assert resources.retained_budget.used == 8
    assert frame["x"].to_list() == [7] and not resources.cleanup_complete
    lease.release()
    resources.close()
    assert resources.cleanup_complete and resources.retained_budget.used == 0


def test_group_publication_stages_io_then_publishes_atomic_metadata(tmp_path, monkeypatch):
    frames = FrameStore(8, 50_000, tmp_path)
    frames["old"] = pl.DataFrame({"x": [99]})
    entered, release = Event(), Event()
    original = FrameStore._write_spill
    publications = []

    def slow(store, frame):
        entered.set()
        assert release.wait(5)
        return original(store, frame)

    def metadata():
        publications.append(set(frames))

    monkeypatch.setattr(FrameStore, "_write_spill", slow)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = workers.submit(
            frames.publish,
            {"a": pl.DataFrame({"x": range(10)}), "b": pl.DataFrame({"x": range(10)})},
            metadata,
            RLock(),
        )
        try:
            assert entered.wait(5)
            assert set(frames) == {"old"} and not publications
            read = workers.submit(lambda: frames["old"]["x"].item())
            assert read.result(timeout=2) == 99
        finally:
            release.set()
        pending.result(timeout=5)
    assert publications == [{"old", "a", "b"}]
    frames.close()
    assert frames.spill_budget.used == frames.retained_budget.used == 0


def test_failed_group_staging_does_not_publish_partial_frames(tmp_path, monkeypatch):
    frames = FrameStore(8, 50_000, tmp_path)
    frames["old"] = pl.DataFrame({"x": [99]})
    original = FrameStore._write_spill
    calls = 0
    metadata = Mock()

    def write(store, frame):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second write")
        return original(store, frame)

    monkeypatch.setattr(FrameStore, "_write_spill", write)
    with pytest.raises(OSError, match="second write"):
        frames.publish(
            {"a": pl.DataFrame({"x": range(10)}), "b": pl.DataFrame({"x": range(10)})},
            metadata,
            RLock(),
        )
    assert set(frames) == {"old"} and frames["old"]["x"].item() == 99
    assert not metadata.called and frames.spill_budget.used == 0
    frames.close()


def test_release_during_staged_load_keeps_catalog_responsive(tmp_path, monkeypatch):
    ctx = DataContext(
        frames={"d": pl.DataFrame({"x": range(10)}).lazy()},
        cache_max_bytes=0,
        temp_directory=tmp_path,
    )
    entered, proceed, releasing = Event(), Event(), Event()
    original_write = FrameStore._write_spill
    original_discard = ctx._frames.discard

    def write(store, frame):
        entered.set()
        assert proceed.wait(5)
        return original_write(store, frame)

    def discard(name):
        releasing.set()
        return original_discard(name)

    monkeypatch.setattr(FrameStore, "_write_spill", write)
    monkeypatch.setattr(ctx._frames, "discard", discard)
    with ThreadPoolExecutor(max_workers=3) as workers:
        loading = workers.submit(ctx.get, "d")
        try:
            assert entered.wait(5)
            release = workers.submit(ctx.release, ["d"])
            assert releasing.wait(5)
            assert workers.submit(ctx.names).result(timeout=2) == []
        finally:
            proceed.set()
        with pytest.raises(Exception, match="released"):
            loading.result(timeout=5)
        release.result(timeout=5)
    assert "d" not in ctx._frames
    assert ctx._resources.spill_budget.used == 0
    ctx.close()


def test_observation_attempts_all_owners_and_retries_failure():
    first, second = Mock(), Mock()
    first.close.side_effect = [PermissionError("temporary"), None]
    observation = Observation(None, files=[first, second])
    with pytest.raises(PermissionError):
        observation.close()
    assert first.close.call_count == second.close.call_count == 1
    assert observation._files == (first,)
    observation.close()
    observation.close()
    assert first.close.call_count == 2 and second.close.call_count == 1
    assert observation._files == ()


def test_partial_observation_borrow_releases_acquired_leases():
    first, second, lease = Mock(), Mock(), Mock()
    first.borrow.return_value = lease
    second.borrow.side_effect = OSError("injected borrow failure")
    ctx = SimpleNamespace(_check_cancelled=lambda: None)
    observation = Observation(ctx, files=[first, second])
    with pytest.raises(OSError, match="borrow failure"):
        observation.borrow()
    lease.close.assert_called_once_with()
    observation.close()
    first.close.assert_called_once_with()
    second.close.assert_called_once_with()


def test_failed_publication_rolls_back_all_keys_after_unlink_error(tmp_path, monkeypatch):
    frames = FrameStore(0, 50_000, tmp_path)
    gate = _unlink_gate(monkeypatch, frames.directory)

    def metadata():
        raise ValueError("injected metadata rejection")

    with pytest.raises((PermissionError, ExceptionGroup)):
        frames.publish(
            {"a": pl.DataFrame({"x": [1]}), "b": pl.DataFrame({"x": [2]})},
            metadata,
            RLock(),
        )
    assert not list(frames) and len(frames._retired) == 2
    gate.clear()
    frames.close()
    assert frames.spill_budget.used == 0


def test_unpublished_spill_cleanup_remains_reachable(tmp_path, monkeypatch):
    resources = RunResources(0, 20_000, None, tmp_path)
    owned = SpillFile(resources.directory / "partial.arrow", resources.spill_budget)
    with owned.writer() as writer:
        writer.write(b"partial")
    gate = _unlink_gate(monkeypatch, resources.directory)
    with pytest.raises(PermissionError):
        owned.close()
    del owned
    gate.clear()
    resources.close()
    assert resources.spill_budget.used == 0 and resources.cleanup_complete


def test_work_store_failed_delete_is_retried_by_run_owner(tmp_path, monkeypatch):
    resources = RunResources(32, 100_000, None, tmp_path)
    store = Store(resources)
    store.execute("CREATE TABLE t (value INTEGER)")
    gate = _unlink_gate(monkeypatch, resources.directory)
    with pytest.raises(PermissionError):
        store.close()
    assert resources.spill_budget.used > 0
    gate.clear()
    resources.close()
    assert resources.spill_budget.used == 0 and resources.cleanup_complete


def test_native_close_failure_keeps_connection_and_reservation(tmp_path):
    resources = RunResources(0, 1000, None, tmp_path)
    token, _, _ = resources.acquire()
    native = Mock()
    native.close.side_effect = [OSError("busy"), None]
    connection = OwnedConnection(native, resources, token)
    with pytest.raises(OSError, match="busy"):
        connection.close()
    assert token in resources.reservations and token in resources.session.connections
    connection.close()
    connection.close()
    assert native.close.call_count == 2
    assert token not in resources.reservations and token not in resources.session.connections
    resources.close()


def test_session_closes_independent_connections_after_failure():
    session = RunSession()
    first, second = Mock(), Mock()
    first.close.side_effect = [OSError("busy"), None]
    session.register_connection("first", first)
    session.register_connection("second", second)
    with pytest.raises(OSError):
        session.close_connections()
    assert list(session.connections) == ["first"]
    assert second.close.call_count == 1
    session.close_connections()
    assert not session.connections and first.close.call_count == 2


def test_projection_leases_file_without_blocking_unrelated_cache_hit(tmp_path, monkeypatch):
    frames = FrameStore(8, 20_000, tmp_path)
    frames["a"] = pl.DataFrame({"x": range(10)})
    frames["b"] = pl.DataFrame({"x": [99]})
    entered, release = Event(), Event()
    original = resources_module.project_ipc

    def slow(path, columns):
        entered.set()
        assert release.wait(5)
        return original(path, columns)

    monkeypatch.setattr(resources_module, "project_ipc", slow)
    with ThreadPoolExecutor(2) as pool:
        projection = pool.submit(frames.project, "a", ["x"])
        assert entered.wait(5)
        try:
            assert pool.submit(lambda: frames["b"].item()).result(timeout=1) == 99
            path = frames.paths["a"]
            del frames["a"]
            assert path.exists()
        finally:
            release.set()
        assert projection.result().height == 10
    assert not path.exists()
    frames.clear()


def test_spill_write_does_not_hold_metadata_lock(tmp_path, monkeypatch):
    frames = FrameStore(8, 20_000, tmp_path)
    frames["a"] = pl.DataFrame({"x": [1]})
    entered, release = Event(), Event()
    original = pl.DataFrame.write_ipc

    def slow(frame, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_ipc", slow)
    with ThreadPoolExecutor(2) as pool:
        insertion = pool.submit(frames.__setitem__, "b", pl.DataFrame({"x": [2]}))
        assert entered.wait(5)
        try:
            assert pool.submit(lambda: frames["a"].item()).result(timeout=1) == 1
        finally:
            release.set()
        insertion.result()
    assert frames["b"].item() == 2
    frames.clear()


def test_spill_promotion_adopts_detached_payload_without_second_compaction(tmp_path, monkeypatch):
    frames = FrameStore(8, 20_000, tmp_path)
    frames["a"] = pl.DataFrame({"x": [1]})
    frames["b"] = pl.DataFrame({"x": [2]})
    del frames["b"]
    monkeypatch.setattr(
        resources_module,
        "compact_polars_with_size",
        Mock(side_effect=AssertionError("second copy")),
    )
    result = frames["a"]
    assert result.item() == 1 and frames.spill_promotions == 1
    result.replace_column(0, pl.Series("x", [9]))
    assert frames["a"].item() == 1
    frames.clear()


@pytest.mark.parametrize("rows,bytes_", [(0, 10_000), (1, 0), (1, 10_000)])
def test_location_only_receives_admitted_sample(rows, bytes_):
    settings = QAVFSettings.model_validate(
        {"engine": {"max_affected_records": rows, "max_evidence_bytes": bytes_}}
    )
    runner = QARunner(settings)
    source = SimpleNamespace(file_rows=Mock(return_value=[("part.csv", 200_000)]))
    locator = RowLocator(SimpleNamespace(source=lambda _: source))
    finding = Finding.fail(
        "bad", affected=pl.DataFrame({"_row": range(200_000)}), datasets=("d",), unit="rows"
    )
    result = runner._locate_evidence(finding, locator)
    assert result.affected_count == 200_000
    assert len(result.affected) == (1 if rows and bytes_ else 0)
    assert source.file_rows.call_count == (1 if rows and bytes_ else 0)
    assert finding.affected.height == 200_000


@pytest.mark.parametrize("as_frame", [False, True])
def test_location_roles_preserve_metadata_named_user_columns(as_frame):
    rows = [{"_row": 900, "position": 1, "_file": "user", "_file_row": "user-row"}]
    affected = pl.DataFrame(rows) if as_frame else rows
    located = FileIndex([("a.csv", 3)]).annotate(affected, "position")
    row = located.to_dicts()[0] if as_frame else located[0]
    assert row["_row"] == 900 and row["_file"] == "user" and row["_file_row"] == "user-row"
    assert row["qavf__file"] == "a.csv" and row["qavf__file_row"] == 1


def test_context_retries_failed_budget_observation_and_still_closes_caches(tmp_path):
    ctx = DataContext(frames={"d": pl.DataFrame({"x": [1]})}, temp_directory=tmp_path)
    observed = Mock()
    observed.close.side_effect = [PermissionError("busy"), None]
    ctx._budget_observations["held"] = observed
    with pytest.raises(PermissionError):
        ctx.close()
    assert not ctx._frames.data
    assert "held" in ctx._budget_observations
    ctx.close()
    assert observed.close.call_count == 2
    assert not ctx._budget_observations
