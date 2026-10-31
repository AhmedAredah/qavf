"""Regression cases for the independently reproduced 0.7.1 review findings."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager, suppress
from threading import Event, Thread
from types import ModuleType

import polars as pl
import pyarrow as pa
import pytest

from qavf import BatchSource, ResourceLimits
from qavf.cli import main
from qavf.engine.context import DataContext
from qavf.engine.resources import ResourceLimitError, RunResources
from qavf.engine.spill import SpillBudget, SpillFile
from qavf.engine.sqlite_store import Store
from qavf.inputs.normalize import prepare
from qavf.io import ProtectedInputs, atomic_stream
from qavf.sources.frames import as_polars, known_rows, supports_frame


@pytest.mark.parametrize("command", ["run", "compare-datasets"])
@pytest.mark.parametrize(
    "name,pattern",
    [
        ("input[0].csv", None),
        ("input?.csv", None),
        ("input*.csv", None),
        (".input.csv", "*.csv"),
        ("input.csv", "*.csv"),
    ],
)
def test_cli_never_replaces_selected_input(tmp_path, command, name, pattern):
    if sys.platform == "win32" and any(c in name for c in "?*"):
        pytest.skip("Windows filenames forbid these characters")
    source = tmp_path / name
    source.write_text("x\n1\n2\n")
    before = source.read_bytes()
    selected = str(tmp_path / pattern) if pattern else str(source)
    config = tmp_path / "config[0].json"
    config.write_text(
        json.dumps(
            {
                "plugins": {"entry_points": False},
                "engine": {"parallel": False},
                "datasets": {"data": {"path": selected, "schema": {"columns": {"x": {"ge": 0}}}}},
                "report": {"html": False, "json": False},
            }
        )
    )
    args = (
        ["run", "--config", str(config)]
        if command == "run"
        else ["compare-datasets", selected, selected, "--no-plugins"]
    )
    assert main([*args, "--json", str(source)]) == 2
    assert source.read_bytes() == before


def test_guard_rechecks_new_glob_match_at_atomic_commit(tmp_path):
    target = tmp_path / ".new.csv"
    guard = ProtectedInputs([tmp_path / "*.csv"])
    with (
        pytest.raises(ValueError, match="replace the input"),
        guard.publication(),
        atomic_stream(target) as writer,
    ):
        writer.write(b"report")
        target.write_bytes(b"new input")
    assert target.read_bytes() == b"new input"
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_protection_preserves_file_aliases(tmp_path, alias):
    source, destination = tmp_path / "source.csv", tmp_path / "alias.csv"
    source.write_bytes(b"x\n1\n")
    try:
        if alias == "symlink":
            destination.symlink_to(source)
        else:
            destination.hardlink_to(source)
    except OSError as exc:
        if (
            sys.platform == "win32"
            and alias == "symlink"
            and getattr(exc, "winerror", None) == 1314
        ):
            pytest.skip("Windows symlink privileges unavailable")
        raise
    with pytest.raises(ValueError, match="replace the input"):
        ProtectedInputs([source]).check(destination)
    assert source.read_bytes() == b"x\n1\n"


def test_writer_rejects_closed_and_repeated_acquisition(tmp_path):
    budget = SpillBudget(100)
    owner = SpillFile(tmp_path / "closed", budget)
    owner.close()
    with pytest.raises(RuntimeError), owner.writer():
        pass
    assert not owner.path.exists() and not budget.has_files and budget.used == 0
    owner = SpillFile(tmp_path / "once", budget)
    with owner.writer() as writer:
        writer.write(b"ok")
        with pytest.raises(RuntimeError), owner.writer():
            pass
    with pytest.raises(RuntimeError), owner.writer():
        pass
    assert owner.path.read_bytes() == b"ok"
    owner.close()
    assert budget.used == 0


def test_suppressed_write_failure_cannot_finish_successfully(tmp_path):
    budget = SpillBudget(3)
    with (
        SpillFile(tmp_path / "partial", budget) as owner,
        pytest.raises(ResourceLimitError),
        owner.writer() as writer,
    ):
        writer.write(b"ok")
        with suppress(ResourceLimitError):
            writer.write(b"too much")
    assert budget.used == 0 and not budget.has_files and not owner.path.exists()


def test_writer_lease_keeps_file_owned_until_handle_is_closed(tmp_path):
    budget = SpillBudget(100)
    owner = SpillFile(tmp_path / "active", budget)
    with pytest.raises(RuntimeError, match="closed during writing"), owner.writer() as writer:
        writer.write(b"ok")
        owner.close()
        assert owner.path.exists() and budget.has_files and budget.used == 2
    assert not owner.path.exists() and budget.used == 0 and not budget.has_files


def test_file_count_admission_includes_pending_leases(tmp_path):
    budget = SpillBudget(100, max_files=1)
    owner = SpillFile(tmp_path / "first", budget)
    lease = owner.borrow()
    owner.close()
    with pytest.raises(ResourceLimitError, match="file-count"):
        SpillFile(tmp_path / "second", budget)
    lease.close()
    SpillFile(tmp_path / "second", budget).close()
    assert not budget.has_files


@pytest.mark.parametrize("size", [1, 7, 1000])
def test_small_batch_partitions_share_one_exact_spill(tmp_path, size):
    original = pl.DataFrame({"x": range(1000), "label": ["a"] * 1000})

    @contextmanager
    def batches():
        yield original.iter_slices(size)

    settings, frames = prepare(
        BatchSource(batches, schema=original.schema),
        schema={},
        config=None,
        resources=ResourceLimits(
            cache_memory=0, target_memory="4MiB", max_spill="32MiB", temp_dir=tmp_path
        ),
    )
    with DataContext.from_settings(settings, frames) as ctx, ctx.observe("data") as observation:
        assert observation.collect(observation.scan()).equals(original)
        usage = ctx.resource_usage()
        assert usage["owned_spill_files"] == 1
        assert usage["spool_working_bytes"] == 0
        assert usage["peak_spool_working_bytes"] <= usage["spool_working_budget_bytes"]
    assert not list(tmp_path.iterdir())


def test_partial_optional_module_does_not_block_arrow(monkeypatch):
    table = pa.table({"x": [1]})
    monkeypatch.setitem(sys.modules, "pandas", ModuleType("pandas"))
    assert supports_frame(table) and known_rows(table) == 1
    assert as_polars(table).to_dict(as_series=False) == {"x": [1]}


def test_real_structural_interchange_is_supported():
    class Provider:
        def __dataframe__(self, nan_as_null=False, allow_copy=True):
            return pa.table({"x": [1, 2]}).__dataframe__(nan_as_null, allow_copy)

    assert supports_frame(Provider())
    assert as_polars(Provider()).to_dict(as_series=False) == {"x": [1, 2]}


def test_unrelated_sqlite_read_does_not_wait_for_another_query(tmp_path):
    resources = RunResources(0, 1 << 20, 1 << 20, tmp_path)
    entered, release, completed = Event(), Event(), Event()
    errors = []

    def slow():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test barrier expired")
        return 1

    def run(first):
        try:
            with Store(resources) as store:
                if first:
                    store.connection.create_function("slow", 0, slow)
                    store.execute("SELECT slow()").fetchone()
                else:
                    assert store.execute("SELECT 1").fetchone() == (1,)
                    completed.set()
        except BaseException as exc:
            errors.append(exc)

    first, second = Thread(target=run, args=(True,)), Thread(target=run, args=(False,))
    first.start()
    try:
        assert entered.wait(3)
        second.start()
        assert completed.wait(3), "independent SELECT blocked on the allocation lock"
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
        resources.close()
    assert not errors and not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["short", "partial", "flush"])
def test_writer_failure_accounts_actual_bytes_and_latches(tmp_path, monkeypatch, failure):
    from pathlib import Path

    path = tmp_path / "partial"
    original_open = Path.open

    class BrokenFile:
        def __init__(self):
            self.handle = original_open(path, "wb", buffering=0)

        @property
        def closed(self):
            return self.handle.closed

        def tell(self):
            return self.handle.tell()

        def write(self, data):
            if failure == "flush":
                return self.handle.write(data)
            count = self.handle.write(data[:2])
            if failure == "partial":
                raise OSError("partial write")
            return count

        def flush(self):
            if failure == "flush":
                raise OSError("failed flush")
            self.handle.flush()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.handle.close()

    monkeypatch.setattr(
        Path,
        "open",
        lambda self, *a, **kw: BrokenFile() if self == path else original_open(self, *a, **kw),
    )
    budget = SpillBudget(100)
    owner = SpillFile(path, budget)
    try:
        with pytest.raises(OSError), owner.writer() as writer, suppress(OSError):
            writer.write(b"abcdef")
            writer.flush()
        assert owner.size == budget.used == path.stat().st_size
        assert owner.size == (6 if failure == "flush" else 2)
    finally:
        owner.close()
    assert budget.used == 0 and not budget.has_files and not path.exists()


def test_sqlite_concurrent_growth_never_spends_shared_credit_twice(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    resources = RunResources(0, 131072, None, temp_directory=tmp_path)
    finished = Barrier(3)
    release = Event()

    def writing():
        try:
            with Store(resources) as store:
                try:
                    store.execute("CREATE TABLE values_(x BLOB)")
                    store.executemany("INSERT INTO values_ VALUES (?)", [(b"x" * 6000,)] * 20)
                except ResourceLimitError:
                    pass
                finally:
                    finished.wait(timeout=15)
                    assert release.wait(15)
        finally:
            release.set()

    try:
        with ThreadPoolExecutor(2) as executor:
            futures = [executor.submit(writing) for _ in range(2)]
            try:
                finished.wait(timeout=15)
                actual = sum(path.stat().st_size for path in resources.directory.glob("*.sqlite"))
                assert actual == resources.spill_budget.used <= 65536
                assert resources.spill_budget.peak <= 65536
            finally:
                release.set()
            for future in futures:
                future.result()
    finally:
        resources.close()
    assert not list(tmp_path.iterdir())


def test_sqlite_completed_writes_and_readonly_ctes_at_full_quota(tmp_path):
    resources = RunResources(0, 16384, None, temp_directory=tmp_path)
    try:
        with Store(resources) as store:
            store.execute("CREATE TABLE t(x)")
            assert resources.spill_budget.remaining == 0
            assert store.execute("WITH x(v) AS (SELECT 7) SELECT v FROM x").fetchone() == (7,)
            with pytest.raises(ValueError, match="must not return"):
                store.execute("INSERT INTO t VALUES (1) RETURNING x")
            assert store.execute("SELECT count(*) FROM t").fetchone() == (0,)
    finally:
        resources.close()


def test_real_partial_pandas_import_preserves_arrow_admission():
    import importlib.util
    import subprocess
    from pathlib import Path

    if importlib.util.find_spec("pandas") is None:
        # Core-only installations still test partial-module admission above.
        return
    code = (Path(__file__).parent / "helpers" / "import_race_probe.py").read_text()
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, timeout=45, check=True
    )
    result = json.loads(completed.stdout)
    assert result == {
        "accepted_before": True,
        "during_import": {"accepted": True},
        "accepted_after": True,
        "import_errors": [],
    }
