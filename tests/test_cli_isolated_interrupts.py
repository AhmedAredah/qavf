"""Interrupts and progress for ``qavf run`` in ``execution_mode: isolated`` (CLI-02).

An isolated worker has no cooperative cancellation and reports no progress, so
``--progress`` is refused before any input is read, and an interrupt stops the
worker (the supervisor's cleanup runs) and reports the run cancelled with a
nonzero status, never QA PASSED.

The real-signal test sends Ctrl-Break on Windows (a console process group) and
SIGTERM/SIGINT elsewhere. Windows cannot deliver SIGTERM to a handler (it
terminates the process outright); the Windows job object then ends the worker.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qavf import cli

SLOW_SQL = "SELECT sum(hash(a.range + b.range)) AS s FROM range(200000) a, range(200000) b"
"""A query that runs for minutes: the worker is still busy when the interrupt comes."""


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "isolated.yaml"
    config.write_text(
        "datasets:\n"
        f"  slow:\n    source: {{type: query, sql: '{SLOW_SQL}'}}\n"
        "engine:\n  execution_mode: isolated\n  run_timeout_s: 120\n"
        f"  temp_directory: {(tmp_path / 'work').as_posix()}\n"
        "report: {html: false, json: false}\n",
        encoding="utf-8",
    )
    return config


def test_progress_is_refused_for_an_isolated_run_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("qavf.api.validate", _never_called)
    arguments = ["run", "-c", str(_config(tmp_path)), "--backend", "duckdb", "--progress"]
    assert cli.main(arguments) == cli.EXIT_USAGE
    assert "--progress needs engine.execution_mode: in_process" in capsys.readouterr().err
    assert not (tmp_path / "work").exists()


def _never_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("the run started")


def test_an_interrupt_stops_an_isolated_run_once() -> None:
    handler = cli._InterruptStops()
    with pytest.raises(cli.RunInterrupted) as raised:
        handler.handle(signal.SIGINT, None)
    assert raised.value.signal_name == "SIGINT"
    handler.handle(signal.SIGINT, None)  # cleanup is running: later interrupts wait


def test_an_isolated_run_does_not_install_cooperative_cancellation() -> None:
    assert type(cli._interrupts(False, cli.CancellationToken())) is cli._InterruptStops
    assert type(cli._interrupts(True, cli.CancellationToken())) is cli._InterruptCancels


def _interrupts() -> list[int]:
    if sys.platform == "win32":
        return [signal.CTRL_BREAK_EVENT]
    return [signal.SIGTERM, signal.SIGINT]


@pytest.mark.parametrize("number", _interrupts())
def test_a_real_interrupt_stops_the_worker_and_reports_cancelled(
    tmp_path: Path, number: int
) -> None:
    config = _config(tmp_path)
    work = tmp_path / "work"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    process = subprocess.Popen(
        [sys.executable, "-m", "qavf", "run", "-c", str(config), "--backend", "duckdb"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=flags,
    )
    try:
        deadline = time.monotonic() + 60
        # The worker's control file is written just before the worker starts.
        while not any(work.glob("qavf-isolated-*/control.json")):
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "the isolated worker never started"
            time.sleep(0.05)
        time.sleep(0.5)
        process.send_signal(number)
        out, err = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == cli.EXIT_QA_FAILED, (out, err)
    assert "QA CANCELLED" in out and "stopped the isolated worker" in out
    assert "QA PASSED" not in out and "Cancelling:" not in err
    # The supervisor removes its run directory only after the worker is reaped.
    assert not any(work.glob("qavf-isolated-*"))
