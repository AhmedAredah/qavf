"""Operation-scoped hash reuse retains strict content and failure boundaries."""

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import polars as pl
import pytest

from qavf.engine.context import DataContext, DatasetNotAvailableError
from qavf.sources import SourceEnvironment, SourceError, SourceIntegrityError
from qavf.sources.base import DataSource, SourceParams
from qavf.sources.errors import verify_postcondition
from qavf.sources.execution import execute_with_source_policy
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams


def file_source(tmp_path, *, name="data.csv", retries=0, content="x\n1\n"):
    path = tmp_path / name
    path.write_bytes(content.encode())
    return path, FileSource(
        FileParams(path=str(path), retries=retries, retry_delay_s=0), SourceEnvironment()
    )


def passes(source):
    return source.identity_telemetry()["hash_passes"]


def test_nested_reads_and_provenance_share_only_the_outer_hash_boundaries(tmp_path):
    path, source = file_source(tmp_path)
    fingerprint = source.fingerprint(True)

    def nested():
        source.verify_identity()
        assert source.scan() is not None
        assert source.fingerprint(True) == fingerprint
        execute_with_source_policy([source], source.verify_identity)

    execute_with_source_policy([source], nested)
    counters = source.identity_telemetry()
    assert counters["hash_passes"] == counters["hash_passes_completed"] == 3
    assert counters["bytes_hashed"] == 3 * path.stat().st_size
    assert counters["coalesced_identity_checks"] == 6
    source.fingerprint(True)
    source.verify_identity()
    assert passes(source) == 5  # Never reuse solely because metadata still matches.


@pytest.mark.parametrize("boundary", ["entry", "exit", "initial_digest"])
def test_same_size_restored_mtime_changes_still_fail_content_identity(tmp_path, boundary):
    path, source = file_source(tmp_path)
    if boundary != "initial_digest":
        source.fingerprint(True)
    source._files()
    original = source._file_stat(path)
    stat = path.stat()

    def mutate():
        path.write_text("x\n2\n")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def read():
        if boundary == "initial_digest":
            source.fingerprint(True)
        if boundary != "entry":
            mutate()
        return "must not publish"

    if boundary == "entry":
        mutate()
    # Also hide ctime/inode differences: content checks cannot trust metadata.
    with (
        patch.object(source, "_file_stat", return_value=original),
        pytest.raises(SourceIntegrityError, match="content changed"),
    ):
        execute_with_source_policy([source], read)
    assert not source._identity_session.active


def test_failed_read_checks_integrity_before_deciding_to_retry(tmp_path):
    path, source = file_source(tmp_path, retries=2)
    source.fingerprint(True)
    original = source._file_stat(path)
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        path.write_text("x\n2\n")
        raise SourceError("temporary reader failure", retryable=True)

    with (
        patch.object(source, "_file_stat", return_value=original),
        pytest.raises(SourceIntegrityError, match="content changed"),
    ):
        execute_with_source_policy([source], fail)
    assert calls == 1
    assert passes(source) == 3
    assert not source._identity_session.active


def test_each_retry_has_fresh_entry_and_exit_hashes(tmp_path):
    _, source = file_source(tmp_path, retries=2)
    source.fingerprint(True)
    calls = 0

    def transient():
        nonlocal calls
        calls += 1
        source.fingerprint(True)
        if calls == 1:
            raise SourceError("temporary reader failure", retryable=True)
        return 7

    assert execute_with_source_policy([source], transient) == 7
    assert passes(source) == 5
    assert source.load_notes["retries_used"] == 1


def test_new_nested_participant_gets_its_own_verified_boundaries(tmp_path):
    _, first = file_source(tmp_path, name="first.csv")
    _, second = file_source(tmp_path, name="second.csv")
    for source in (first, second):
        source.fingerprint(True)
    execute_with_source_policy(
        [first],
        lambda: execute_with_source_policy([first, second], lambda: second.fingerprint(True)),
    )
    assert passes(first) == passes(second) == 3


def test_concurrent_threads_never_share_an_active_verification_scope(tmp_path):
    _, source = file_source(tmp_path)
    source.fingerprint(True)
    rendezvous = threading.Barrier(2)

    def read():
        rendezvous.wait(timeout=10)
        source.verify_identity()
        return source.fingerprint(True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute_with_source_policy, [source], read) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert results[0] == results[1]
    assert passes(source) == 5
    assert not source._identity_session.active


@pytest.mark.parametrize("inside_operation", [False, True])
def test_concurrent_first_fingerprints_cannot_replace_the_accepted_content_identity(
    tmp_path, inside_operation
):
    import qavf.sources.file as file_sources

    path, source = file_source(tmp_path)
    source._files()
    original = source._file_stat(path)
    first_hashed = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_hashed = threading.Event()
    native_hash = file_sources._sha256
    calls = 0
    count_lock = threading.Lock()

    def controlled_hash(path, **kwargs):
        nonlocal calls
        digest = native_hash(path, **kwargs)
        with count_lock:
            calls += 1
            first = calls == 1
        if first:
            first_hashed.set()
            assert release_first.wait(timeout=10)
        else:
            second_hashed.set()
        return digest

    def fingerprint():
        if inside_operation:
            def read():
                source.prepare_identity(True)
                return path.read_text()

            return execute_with_source_policy([source], read)
        return source.fingerprint(True)

    def second_fingerprint():
        second_started.set()
        return fingerprint()

    with (
        patch.object(source, "_file_stat", return_value=original),
        patch("qavf.sources.file._sha256", controlled_hash),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(fingerprint)
        try:
            assert first_hashed.wait(timeout=10)
            path.write_text("x\n2\n")
            second = pool.submit(second_fingerprint)
            assert second_started.wait(timeout=10)
            # Give an incorrectly unlocked second hasher the chance to publish
            # before the paused first hasher. Correct code holds the identity lock.
            second_hashed.wait(timeout=0.2)
        finally:
            release_first.set()
        if inside_operation:
            with pytest.raises(SourceIntegrityError, match="content changed"):
                first.result(timeout=10)
        else:
            assert first.result(timeout=10)["sha256"] == hashlib.sha256(b"x\n1\n").hexdigest()
        with pytest.raises(SourceIntegrityError, match="content changed"):
            second.result(timeout=10)
    assert source._content_hashes[path] == hashlib.sha256(b"x\n1\n").hexdigest()


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        (KeyboardInterrupt(), SourceIntegrityError("changed"), KeyboardInterrupt),
        (RuntimeError("read failed"), KeyboardInterrupt(), KeyboardInterrupt),
        (SourceError("read failed"), SystemExit(7), SystemExit),
    ],
)
def test_postcheck_error_precedence_preserves_process_interruptions(previous, current, expected):
    def verify():
        raise current

    with pytest.raises(expected):
        verify_postcondition(verify, previous)


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_exception_unwinding_clears_scope_and_verifies_before_next_operation(tmp_path, failure):
    _, source = file_source(tmp_path)
    source.fingerprint(True)

    def fail():
        raise failure

    with pytest.raises(type(failure)):
        execute_with_source_policy([source], fail)
    assert not source._identity_session.active
    execute_with_source_policy([source], lambda: None)
    assert passes(source) == 5


@pytest.mark.parametrize("source_kind", ["file", "plugin"])
@pytest.mark.parametrize(
    "original",
    [
        RuntimeError("unknown read failure"),
        KeyboardInterrupt(),
        SourceError("permanent failure"),
        SourceIntegrityError("changed", retryable=True),
    ],
)
def test_postcheck_failure_never_converts_a_terminal_read_error_into_a_retry(
    tmp_path, source_kind, original
):
    class Plugin(DataSource):
        kind = "plugin"
        Params = SourceParams

        def load(self):
            return pl.DataFrame()

    if source_kind == "file":
        _, source = file_source(tmp_path, retries=1)
    else:
        source = Plugin(
            SourceParams(type="plugin", retries=1, retry_delay_s=0), SourceEnvironment()
        )
    attempts = 0
    checks = 0

    def verify():
        nonlocal checks
        checks += 1
        if checks % 2 == 0:
            raise SourceError("temporary postcheck failure", retryable=True)

    def fail():
        nonlocal attempts
        attempts += 1
        raise original

    with patch.object(source, "verify_identity", verify), pytest.raises(type(original)) as raised:
        execute_with_source_policy([source], fail)
    assert raised.value is original
    assert attempts == 1
    assert checks == 2


def test_unknown_postcheck_failure_is_not_hidden_by_a_retryable_read_error(tmp_path):
    _, source = file_source(tmp_path, retries=1)
    checks = 0
    attempts = 0

    def verify():
        nonlocal checks
        checks += 1
        if checks % 2 == 0:
            raise RuntimeError("unknown postcheck failure")

    def fail():
        nonlocal attempts
        attempts += 1
        raise SourceError("temporary read failure", retryable=True)

    with (
        patch.object(source, "verify_identity", verify),
        pytest.raises(RuntimeError, match="unknown"),
    ):
        execute_with_source_policy([source], fail)
    assert attempts == 1


def test_nested_checks_still_detect_metadata_changes_immediately(tmp_path):
    path, source = file_source(tmp_path)
    source.fingerprint(True)
    reached = []

    def read():
        path.write_text("x\n12345\n")
        source.verify_identity()
        reached.append("after mutation")

    with pytest.raises(SourceIntegrityError, match="changed"):
        execute_with_source_policy([source], read)
    assert not reached
    assert not source._identity_session.active


@pytest.mark.parametrize("size", [0, 17, 2 * (1 << 20) + 7])
def test_hash_telemetry_measures_streamed_bytes_and_passes(tmp_path, size):
    path, source = file_source(tmp_path, content="x" * size)
    assert source.fingerprint(True)["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    source.verify_identity()
    counters = source.identity_telemetry()
    assert counters["bytes_hashed"] == size * 2
    assert counters["hash_passes"] == counters["hash_passes_completed"] == 2


def test_incomplete_hash_counts_attempt_and_bytes_without_claiming_completion(tmp_path):
    path, source = file_source(tmp_path)
    source.fingerprint(True)

    def interrupted(path, *, record_bytes):
        with path.open("rb") as handle:
            record_bytes(len(handle.read(2)))
        raise OSError("interrupted file read")

    with (
        patch("qavf.sources.file._sha256", interrupted),
        pytest.raises(SourceIntegrityError, match="unavailable"),
    ):
        source.verify_identity()
    assert source.identity_telemetry() == {
        "hash_passes": 2,
        "hash_passes_completed": 1,
        "bytes_hashed": path.stat().st_size + 2,
        "coalesced_identity_checks": 0,
    }


def test_context_initial_fingerprint_precedes_eager_read_and_is_checked_after(tmp_path):
    path, source = file_source(tmp_path)
    with DataContext(sources={"data": source}, hash_inputs=True) as ctx:
        bound = ctx._catalog.sources["data"]
        bound._files()
        original = bound._file_stat(path)

        def changed_load():
            assert bound._content_hashes
            path.write_text("x\n2\n")
            return pl.DataFrame({"x": [2]})

        with (
            patch.object(bound, "_file_stat", return_value=original),
            patch.object(bound, "load", changed_load),
            pytest.raises(DatasetNotAvailableError, match="content changed"),
        ):
            ctx.get("data")
        assert not ctx.is_materialized("data")


def test_context_initial_native_read_is_anchored_to_the_content_identity(tmp_path):
    path, source = file_source(tmp_path)
    with DataContext(sources={"data": source}, hash_inputs=True) as ctx:
        bound = ctx._catalog.sources["data"]
        bound._files()
        original = bound._file_stat(path)
        scan = bound.scan

        def changed_scan():
            assert bound._content_hashes
            path.write_text("x\n2\n")
            return scan()

        with (
            patch.object(bound, "_file_stat", return_value=original),
            patch.object(bound, "scan", changed_scan),
            pytest.raises(SourceIntegrityError, match="content changed"),
        ):
            ctx.sql("SELECT sum(x) FROM data")


def test_eager_plugin_fingerprint_can_still_depend_on_completed_load():
    class AfterLoad(DataSource):
        kind = "after_load"
        Params = SourceParams

        def load(self):
            self.load_notes["loaded"] = True
            return pl.DataFrame({"x": [7]})

        def fingerprint(self, hash_contents=False):
            assert self.load_notes.get("loaded")
            return {"identity": "plugin", "version": "loaded"}

    with DataContext(
        sources={"data": AfterLoad(SourceParams(type="after_load"), SourceEnvironment())},
        hash_inputs=True,
    ) as ctx:
        assert ctx.get("data").item() == 7
        assert ctx.describe()[0].fingerprint == {"identity": "plugin", "version": "loaded"}


def test_repeated_lazy_schema_uses_metadata_while_collection_rechecks_content(tmp_path):
    path = tmp_path / "typed.parquet"
    pl.DataFrame({"t": pl.Series([1, 2], dtype=pl.Datetime("ns"))}).write_parquet(path)
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    with DataContext(sources={"data": source}, hash_inputs=True) as ctx:
        expected = ctx.schema("data")
        bound = ctx._catalog.sources["data"]
        previous = passes(bound)
        with patch.object(pl.LazyFrame, "collect_schema", side_effect=AssertionError("rescan")):
            for _ in range(3):
                returned = ctx.schema("data")
                assert returned == expected
                returned.clear()  # Never expose the shared schema dictionary.
            assert passes(bound) == previous
        plan = ctx.scan("data").select(pl.len())
        before_collect = passes(bound)
        assert ctx.collect("data", plan).item() == 2
        assert passes(bound) == before_collect + 2
        assert ctx.describe()[0].notes["identity_work"] == bound.identity_telemetry()
        before_final = passes(bound)
        assert ctx.verify_inputs() == {}
        assert passes(bound) == before_final + 1
