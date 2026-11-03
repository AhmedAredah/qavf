"""Progress events and caller cancellation, on both backends."""

from __future__ import annotations

import json
import threading
import time

import pytest

from qavf import CancellationToken, RunEvent, validate
from qavf.cli import _InterruptCancels, main
from qavf.run_control import CANCELLED_MESSAGE, RunControl

BACKENDS = ["polars", "duckdb"]
# Far too slow to finish: only an interrupt ends it within the test's bound.
SLOW_SQL = (
    "SELECT * FROM (SELECT count(*) AS c FROM range(100000000000) t(x) WHERE x % 7 = 3) WHERE c < 0"
)


class EventLog:
    """Collects events; optionally cancels after a number of finished steps."""

    def __init__(self, cancel_after: int | None = None, token: CancellationToken | None = None):
        self.events: list[RunEvent] = []
        self.cancel_after = cancel_after
        self.token = token

    def __call__(self, event: RunEvent) -> None:
        self.events.append(event)
        if (
            self.token is not None
            and event.kind == "step_finished"
            and event.completed == self.cancel_after
        ):
            self.token.cancel()

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]


class FailingCallback:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, event: RunEvent) -> None:
        self.calls += 1
        raise RuntimeError("callback bug")


class CancelLater:
    """Cancel ``token`` after ``delay`` seconds on a background thread."""

    def __init__(self, token: CancellationToken, delay: float) -> None:
        self.token = token
        self.timer = threading.Timer(delay, token.cancel)

    def __enter__(self) -> CancelLater:
        self.timer.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.timer.cancel()


def _config(tmp_path) -> dict:
    data = tmp_path / "data.csv"
    data.write_text("id,v\n1,1\n2,-1\n3,3\n")
    other = tmp_path / "other.csv"
    other.write_text("id\n1\n2\n")
    return {
        "datasets": {
            "data": {
                "path": str(data),
                "schema": {
                    "columns": {"id": {"dtype": "int"}, "v": {"dtype": "int", "ge": 0}},
                    "primary_key": ["id"],
                },
            },
            "other": {"path": str(other), "schema": {"columns": {"id": {"dtype": "int"}}}},
        }
    }


def _slow_config(tmp_path) -> dict:
    return _config(tmp_path) | {"checks": [{"check": "sql", "name": "slow", "sql": SLOW_SQL}]}


@pytest.mark.parametrize("backend", BACKENDS)
def test_events_bracket_every_step_in_order(tmp_path, backend):
    log = EventLog()
    report = validate(config=_config(tmp_path), backend=backend, on_event=log)
    kinds = log.kinds()
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    finished = [event for event in log.events if event.kind == "step_finished"]
    assert finished and [event.completed for event in finished] == list(range(1, len(finished) + 1))
    assert kinds.count("step_started") == len(finished)
    assert log.events[-1].report is report and not report.metadata.cancelled
    if backend == "polars":
        assert log.events[0].total == len(finished)
    else:
        assert log.events[0].total is None
        assert {event.dataset for event in finished} >= {"data", "other"}
    failed = [result for event in finished for result in event.results]
    assert any(result.status.value == "FAIL" for result in failed)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancelled_before_start_assesses_nothing(tmp_path, backend):
    token = CancellationToken()
    token.cancel()
    report = validate(config=_config(tmp_path), backend=backend, cancel=token)
    assert report.metadata.cancelled
    assert report.verdict() == "incomplete"
    cancelled = [result for result in report.results if result.message == CANCELLED_MESSAGE]
    assert cancelled
    assert not [result for result in report.results if result.status.value == "FAIL"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancelling_mid_run_keeps_finished_steps(tmp_path, backend):
    token = CancellationToken()
    log = EventLog(cancel_after=1, token=token)
    report = validate(config=_config(tmp_path), backend=backend, on_event=log, cancel=token)
    assert report.metadata.cancelled and report.verdict() == "incomplete"
    first = next(event for event in log.events if event.kind == "step_finished")
    assert not first.cancelled
    later = [e for e in log.events if e.kind == "step_finished" and e.completed > 1]
    assert later and all(event.cancelled for event in later)
    assert any(result.message == CANCELLED_MESSAGE for result in report.results)


def test_cancel_interrupts_a_running_native_query(tmp_path):
    token = CancellationToken()
    started = time.monotonic()
    with CancelLater(token, 0.5):
        report = validate(
            config=_slow_config(tmp_path),
            backend="duckdb",
            cancel=token,
        )
    assert time.monotonic() - started < 30
    assert report.metadata.cancelled
    slow = [result for result in report.results if result.rule_id == "INP-007"]
    assert slow and all(result.message == CANCELLED_MESSAGE for result in slow)


def test_cancel_interrupts_a_running_full_backend_rule(tmp_path):
    token = CancellationToken()
    started = time.monotonic()
    with CancelLater(token, 0.5):
        report = validate(
            config=_slow_config(tmp_path),
            backend="polars",
            cancel=token,
        )
    assert time.monotonic() - started < 30
    assert report.metadata.cancelled
    slow = [result for result in report.results if result.rule_id == "INP-007"]
    assert slow and all(result.message == CANCELLED_MESSAGE for result in slow)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_failing_callback_never_stops_the_run(tmp_path, backend):
    callback = FailingCallback()
    report = validate(config=_config(tmp_path), backend=backend, on_event=callback)
    assert callback.calls > 2
    assert not report.metadata.cancelled
    assert any(result.status.value == "FAIL" for result in report.results)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hooks_require_in_process_execution(tmp_path, backend):
    config = _config(tmp_path) | {"engine": {"execution_mode": "isolated"}}
    with pytest.raises(ValueError, match="in_process"):
        validate(config=config, backend=backend, cancel=CancellationToken())


def test_token_listeners_run_once_and_late_listeners_run_at_once():
    token = CancellationToken()
    log = EventLog()
    control = RunControl(log, token)
    calls: list[str] = []
    token.add_listener(calls.clear)
    token.cancel()
    token.cancel()
    assert control.cancelled
    late = threading.Event()
    token.add_listener(late.set)
    assert late.is_set()


def test_event_summary_is_json_ready(tmp_path):
    log = EventLog()
    validate(config=_config(tmp_path), backend="duckdb", on_event=log)
    records = [json.loads(json.dumps(event.to_dict())) for event in log.events]
    assert records[0]["event"] == "run_started"
    assert records[-1]["verdict"] in {"passed", "failed", "incomplete"}
    assert any("statuses" in record for record in records)


def test_cli_progress_writes_json_lines(tmp_path, capsys):
    config = tmp_path / "qavf.yaml"
    config.write_text(json.dumps(_config(tmp_path)))
    code = main(
        [
            "run",
            "-c",
            str(config),
            "--backend",
            "duckdb",
            "--progress",
            "--json",
            str(tmp_path / "r.json"),
            "--html",
            str(tmp_path / "r.html"),
        ]
    )
    assert code == 1
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert lines[0]["event"] == "run_started" and lines[-1]["event"] == "run_finished"


def test_first_interrupt_cancels_and_second_stops():
    token = CancellationToken()
    handler = _InterruptCancels(token)
    handler.handle(2, None)
    assert token.cancelled
    with pytest.raises(KeyboardInterrupt):
        handler.handle(2, None)
