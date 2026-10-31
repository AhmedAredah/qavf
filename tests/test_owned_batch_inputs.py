"""Owned incremental inputs, coherent snapshots, admission, retries and close paths."""

from contextlib import contextmanager

import polars as pl
import pyarrow as pa
import pytest

from qavf import BatchSource, Dataset, Loader, ResourceLimits, validate
from qavf.engine.context import DataContext
from qavf.inputs.normalize import prepare, prepare_configuration
from qavf.sources.base import SourceEnvironment, SourceError
from qavf.sources.parameters import PythonParams
from qavf.sources.python import PythonSource

CONFIG = {"plugins": {"entry_points": False}, "engine": {"parallel": False}}


def _context(value, **resources):
    settings, frames = prepare(
        value,
        schema={"columns": {}},
        config=CONFIG,
        resources=ResourceLimits(**resources),
    )
    return DataContext.from_settings(settings, frames)


def test_batch_descriptor_and_metadata_do_not_open_factory():
    def forbidden():
        pytest.fail("schema inspection opened factory")

    schema = {"x": pl.Int64}
    descriptor = BatchSource(forbidden, schema)
    schema["x"] = pl.String
    settings, raw = prepare_configuration(
        descriptor, schema={"columns": {}}, config=CONFIG, resources=None
    )
    assert raw["data"] is descriptor
    assert settings.datasets["data"].schema_spec is not None
    with _context(descriptor) as ctx:
        assert ctx.schema("data") == {"x": pl.Int64}


def test_complete_stream_is_observed_once_for_all_checks_and_repeated_reads():
    events = []

    @contextmanager
    def factory():
        events.append("open")
        try:
            yield iter([pa.record_batch({"x": [1, 2]}), pl.DataFrame({"x": [-1, 3]})])
        finally:
            events.append("close")

    source = BatchSource(factory, {"x": pl.Int64})
    report = validate(source, schema={"columns": {"x": {"ge": 0}}}, config=CONFIG)
    assert report.get("INP-003", "data.x: ge 0").affected_count == 1
    assert report.verdict() == "failed"
    assert events == ["open", "close"]
    events.clear()
    with _context(source) as ctx:
        with ctx.observe("data", ["x"]) as first:
            assert first.reduce(lambda lf: lf.select(pl.len())).item() == 4
        with ctx.observe("data", ["x"]) as second:
            assert second.reduce(lambda lf: lf.select(pl.col("x").sum())).item() == 5
        assert ctx.get("data")["x"].to_list() == [1, 2, -1, 3]
    assert events == ["open", "close"]


def test_empty_owned_stream_keeps_declared_schema_and_closes():
    events = []

    @contextmanager
    def factory():
        try:
            yield iter(())
        finally:
            events.append("closed")

    with _context(BatchSource(factory, {"x": pl.Int64})) as ctx:
        assert ctx.get("data").schema == {"x": pl.Int64}
        assert ctx.get("data").height == 0
    assert events == ["closed"]


@pytest.mark.parametrize("bad", [pl.DataFrame({"x": ["bad"]}), pl.DataFrame({"y": [1]})])
def test_schema_drift_is_unassessed_and_closes(bad):
    events = []

    @contextmanager
    def factory():
        try:
            yield iter([pl.DataFrame({"x": [1]}), bad])
        finally:
            events.append("closed")

    report = validate(
        BatchSource(factory, {"x": pl.Int64}), schema={"columns": {"x": {"ge": 0}}}, config=CONFIG
    )
    assert report.incomplete and report.verdict() != "passed"
    assert events == ["closed"]


def test_cleanup_suppression_cannot_accept_a_partial_stream():
    class SuppressingReader:
        def __enter__(self):
            return iter([pl.DataFrame({"x": [1]}), pl.DataFrame({"x": ["wrong type"]})])

        def __exit__(self, *error):
            return True

    report = validate(
        BatchSource(SuppressingReader, {"x": pl.Int64}),
        schema={"columns": {"x": {"ge": 0}}},
        config=CONFIG,
    )
    assert report.incomplete and report.verdict() != "passed"


def test_batch_rename_is_applied_once_and_execution_is_disclosed():
    @contextmanager
    def factory():
        yield iter([pl.DataFrame({"raw": [1, 2]})])

    source = Dataset(BatchSource(factory, {"raw": pl.Int64}), rename={"raw": "x"})
    report = validate(source, schema={"columns": {"x": {"ge": 0}}}, config=CONFIG)
    assert report.verdict() == "passed"
    assert report.metadata.datasets[0].columns == ["x"]
    assert report.metadata.datasets[0].notes["execution"] == "owned_batch_snapshot"


def test_spill_failure_closes_reader_and_does_not_publish_partial_data(tmp_path):
    events = []

    @contextmanager
    def factory():
        try:
            yield iter([pl.DataFrame({"x": list(range(100))}) for _ in range(10)])
        finally:
            events.append("closed")

    report = validate(
        BatchSource(factory, {"x": pl.Int64}),
        schema={"columns": {"x": {"ge": 0}}},
        config=CONFIG,
        resources=ResourceLimits(
            target_memory=2048, cache_memory=0, max_spill=0, temp_dir=tmp_path
        ),
    )
    assert report.incomplete and report.verdict() != "passed"
    assert events == ["closed"]
    assert list(tmp_path.iterdir()) == []


def test_row_budget_is_for_the_complete_stream_not_individual_batches():
    events = []

    @contextmanager
    def factory():
        try:
            yield iter([pl.DataFrame({"x": [1, 2]}), pl.DataFrame({"x": [3, 4]})])
        finally:
            events.append("closed")

    source = Dataset(BatchSource(factory, {"x": pl.Int64}), load_policy={"row_budget": 3})
    report = validate(source, schema={"columns": {"x": {"ge": 0}}}, config=CONFIG)
    assert report.incomplete
    assert any("row_budget" in result.message for result in report.results)
    assert events == ["closed"]


def test_explicit_retry_discards_the_entire_partial_attempt():
    events = []

    @contextmanager
    def factory():
        attempt = events.count("open")
        events.append("open")

        def batches():
            if attempt == 0:
                yield pl.DataFrame({"x": [-99]})
                raise SourceError("temporary", retryable=True)
            yield pl.DataFrame({"x": [1, 2]})

        try:
            yield batches()
        finally:
            events.append("close")

    source = Dataset(
        BatchSource(factory, {"x": pl.Int64}, replayable=True, retry_safe=True),
        load_policy={"retries": 1, "retry_delay_s": 0},
    )
    report = validate(source, schema={"columns": {"x": {"ge": 0}}}, config=CONFIG)
    assert report.verdict() == "passed"
    assert events == ["open", "close", "open", "close"]


def test_non_retry_safe_batch_factory_is_not_reopened():
    events = []

    @contextmanager
    def factory():
        events.append("open")
        raise SourceError("temporary", retryable=True)
        yield  # pragma: no cover

    source = Dataset(
        BatchSource(factory, {"x": pl.Int64}), load_policy={"retries": 2, "retry_delay_s": 0}
    )
    report = validate(source, schema={"columns": {"x": {"ge": 0}}}, config=CONFIG)
    assert report.incomplete
    assert events == ["open"]


def test_batches_spill_under_same_owner_and_cleanup(tmp_path):
    events = []

    @contextmanager
    def factory():
        try:
            yield (pl.DataFrame({"x": [n] * 100}) for n in range(10))
        finally:
            events.append("closed")

    with _context(
        BatchSource(factory, {"x": pl.Int64}),
        target_memory=2048,
        cache_memory=0,
        max_spill="1MiB",
        temp_dir=tmp_path,
    ) as ctx:
        with ctx.observe("data") as observed:
            assert observed.reduce(lambda lf: lf.select(pl.len())).item() == 1000
        assert ctx.resource_usage()["peak_owned_spill_bytes"] > 0
        directory = ctx._resources.directory
    assert not directory.exists()
    assert events == ["closed"]


def test_cancellation_between_batches_closes_owned_stream():
    events = []
    holder = {}

    @contextmanager
    def factory():
        def batches():
            yield pl.DataFrame({"x": [1]})
            holder["ctx"].cancel_rule(holder["invocation"])
            yield pl.DataFrame({"x": [2]})

        try:
            yield batches()
        finally:
            events.append("closed")

    with _context(BatchSource(factory, {"x": pl.Int64})) as ctx:
        holder.update(ctx=ctx, invocation=ctx.begin_rule())
        try:
            with pytest.raises(Exception, match=r"cancel|deadline"), ctx.observe("data"):
                pass
        finally:
            ctx.end_rule()
    assert events == ["closed"]


@pytest.mark.parametrize("kind", ["polars", "arrow", "pandas"])
def test_loader_row_budget_rejects_before_conversion(monkeypatch, kind):
    if kind == "polars":
        frame = pl.DataFrame({"x": [1, 2]})
    elif kind == "arrow":
        frame = pa.table({"x": [1, 2]})
    else:
        frame = pytest.importorskip("pandas").DataFrame({"x": [1, 2]})

    def forbidden(value):
        pytest.fail("oversized loader result was converted")

    monkeypatch.setattr("qavf.sources.frames.as_polars", forbidden)
    report = validate(
        Dataset(Loader(lambda: frame), load_policy={"row_budget": 1}),
        schema={"columns": {"x": {"ge": 0}}},
        config=CONFIG,
    )
    assert report.incomplete
    assert any("row_budget" in result.message for result in report.results)


def test_configured_python_source_rejects_known_rows_before_conversion(monkeypatch):
    frame = pa.table({"x": [1, 2]})
    monkeypatch.setattr("qavf.sources.frames.as_polars", lambda value: pytest.fail("conversion"))
    monkeypatch.setattr(
        "qavf.sources.python.importlib.import_module",
        lambda module: type("M", (), {"load": staticmethod(lambda: frame)}),
    )
    source = PythonSource(PythonParams(callable="example:load", row_budget=1), SourceEnvironment())
    with pytest.raises(SourceError, match="row_budget"):
        source.load()


def test_batch_factory_is_not_opened_for_invalid_sibling_or_isolated_execution():
    def forbidden():
        pytest.fail("factory opened")

    source = BatchSource(forbidden, {"x": pl.Int64})
    with pytest.raises(TypeError, match="unsupported input"):
        validate({"first": Dataset(source, schema={"columns": {}}), "bad": object()})
    with pytest.raises(ValueError, match="isolated execution cannot transport"):
        validate(source, schema={"columns": {}}, resources=ResourceLimits(isolate=True))
