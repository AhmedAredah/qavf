"""The engine, rather than a plugin's cleanup manager, owns completion decisions."""

from contextlib import contextmanager
from typing import ClassVar

import polars as pl
import pytest

from qavf import ResourceLimits, validate
from qavf.engine.context import DataContext
from qavf.engine.resources import ResourceLimitError
from qavf.inputs.normalize import prepare
from qavf.sources.base import DataSource, SourceEnvironment, SourceError
from qavf.sources.parameters import SourceParams

CONFIG = {"plugins": {"entry_points": False}, "engine": {"parallel": False}}
SCHEMA = {"columns": {"x": {"dtype": "int", "ge": 0}}}


def custom_source(fault, events):
    class CustomBatchSource(DataSource[SourceParams]):
        kind: ClassVar[str] = "custom_batch"
        Params = SourceParams

        def batch_schema(self):
            return {"x": pl.Int64}

        @contextmanager
        def open_batches(self):
            def batches():
                yield pl.DataFrame({"x": [1]})
                if fault == "iteration":
                    raise SourceError("interrupted reader")
                if fault == "interrupt":
                    raise KeyboardInterrupt("interrupted reader")
                if fault == "schema":
                    yield pl.DataFrame({"x": ["bad"]})
                else:
                    yield pl.DataFrame({"x": [2]})

            events.append("open")
            try:
                yield batches()
            except BaseException:
                events.append("suppressed")
            finally:
                events.append("close")

        def load(self):
            pytest.fail("batch source used eager loading")

    params = SourceParams(type="custom_batch", row_budget=1 if fault == "row_budget" else None)
    return CustomBatchSource(params, SourceEnvironment())


@pytest.mark.parametrize("fault", ["schema", "row_budget", "iteration"])
def test_custom_cleanup_cannot_accept_an_incomplete_population(tmp_path, fault):
    events = []
    report = validate(
        custom_source(fault, events),
        schema=SCHEMA,
        config=CONFIG,
        resources=ResourceLimits(temp_dir=tmp_path),
    )
    assert report.incomplete and report.verdict() != "passed"
    assert "suppressed" in events
    assert events.count("open") == events.count("close")
    assert list(tmp_path.iterdir()) == []


def test_successful_custom_stream_still_publishes_every_row():
    events = []
    report = validate(
        custom_source("success", events),
        schema=SCHEMA | {"min_rows": 2, "max_rows": 2},
        config=CONFIG,
    )
    assert report.verdict() == "passed"
    assert events == ["open", "close"]


@pytest.mark.parametrize("fault", ["interrupt", "admission"])
def test_suppressed_failures_unwind_builder_and_preserve_original_exception(
    tmp_path, monkeypatch, fault
):
    from qavf.engine.observation import _ObservationBuilder

    events = []
    settings, frames = prepare(
        custom_source(fault, events),
        schema=SCHEMA,
        config=CONFIG,
        resources=ResourceLimits(temp_dir=tmp_path),
    )
    error = KeyboardInterrupt if fault == "interrupt" else ResourceLimitError
    if fault == "admission":
        original = _ObservationBuilder.append
        calls = 0

        def append(self, frame):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ResourceLimitError("admission refused")
            original(self, frame)

        monkeypatch.setattr(_ObservationBuilder, "append", append)
    with DataContext.from_settings(settings, frames) as ctx:
        with (
            pytest.raises(error, match=r"interrupted reader|admission refused"),
            ctx.observe("data"),
        ):
            pytest.fail("partial batch population was published")
        usage = ctx.resource_usage()
        assert usage["owned_spill_bytes"] == 0
        assert usage["retained_total_bytes"] == 0
    assert events == ["open", "suppressed", "close"]
    assert list(tmp_path.iterdir()) == []
