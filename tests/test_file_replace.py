"""Atomic replacement tolerates a target Windows holds for a moment (qavf.file_replace)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest

from qavf import file_replace
from qavf.execution.protocol import write_result
from qavf.file_replace import replace_file
from qavf.io import atomic_write
from qavf.regression import accept_baseline, verify_baseline
from qavf.regression.baselines import manifest_path


class HeldTarget:
    """``os.replace`` failing like a sharing violation for the first ``failures`` calls."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.real = os.replace

    def __call__(self, source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise PermissionError(13, "The process cannot access the file", str(target))
        self.real(source, target)


class NoSleep:
    """Records the delays a retry would wait, without waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _held(monkeypatch: pytest.MonkeyPatch, failures: int, platform: str = "win32") -> HeldTarget:
    held = HeldTarget(failures)
    monkeypatch.setattr(file_replace.sys, "platform", platform)
    monkeypatch.setattr(file_replace.os, "replace", held)
    return held


def test_a_briefly_held_target_is_retried_on_windows(tmp_path, monkeypatch) -> None:
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_text("new")
    target.write_text("old")
    held = _held(monkeypatch, failures=3)
    assert replace_file(source, target) == target
    assert target.read_text() == "new" and not source.exists()
    assert held.calls == 4


def test_the_retry_backs_off_and_gives_up_after_its_tolerance(tmp_path, monkeypatch) -> None:
    source = tmp_path / "new"
    source.write_text("new")
    held = _held(monkeypatch, failures=10**6)
    clock = iter(float(step) / 10 for step in range(10**6))
    monkeypatch.setattr(file_replace.time, "monotonic", clock.__next__)
    sleeps = NoSleep()
    monkeypatch.setattr(file_replace.time, "sleep", sleeps)
    with pytest.raises(PermissionError):
        replace_file(source, tmp_path / "old", tolerance_s=2.0)
    assert 1 < held.calls < 40
    assert sleeps.delays[:3] == [0.01, 0.02, 0.04] and max(sleeps.delays) <= 0.2
    assert source.exists()


def test_a_held_target_fails_at_once_elsewhere(tmp_path, monkeypatch) -> None:
    source = tmp_path / "new"
    source.write_text("new")
    held = _held(monkeypatch, failures=1, platform="linux")
    with pytest.raises(PermissionError):
        replace_file(source, tmp_path / "old")
    assert held.calls == 1


def test_other_errors_are_not_retried(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(file_replace.sys, "platform", "win32")
    with pytest.raises(FileNotFoundError):
        replace_file(tmp_path / "missing", tmp_path / "old")


def test_atomic_writes_and_worker_results_wait_out_a_held_target(tmp_path, monkeypatch) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}")
    held = _held(monkeypatch, failures=2)
    atomic_write(report, '{"status": "passed"}')
    assert json.loads(report.read_text()) == {"status": "passed"} and held.calls == 3
    result = tmp_path / "result.bin"
    result.write_bytes(b"stale")
    held = _held(monkeypatch, failures=2)
    write_result(result, [b"done"], limit=1024)
    assert result.read_bytes() == b"done" and held.calls == 3
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.partial"))


def test_accepting_a_baseline_waits_out_a_held_journal_or_target(tmp_path, monkeypatch) -> None:
    """Accepting writes the journal (state.json) twice and replaces data and manifest;
    each of those renames may meet a briefly held target on Windows."""
    frame = pl.DataFrame({"k": [2, 1], "v": [2.0, 1.0]})
    path = tmp_path / "baseline.csv"
    accept_baseline(frame, path, ["k"])
    held = _held(monkeypatch, failures=0)
    flaky = FlakyEveryOther(held)
    monkeypatch.setattr(file_replace.os, "replace", flaky)
    changed = frame.with_columns(pl.col("v") * 10)
    accept_baseline(changed, path, ["k"], reason="re-accepted", replace=True)
    assert flaky.failed >= 4
    assert verify_baseline(path, ["k"]) == []
    assert json.loads(manifest_path(path).read_text())["reason"] == "re-accepted"
    assert pl.read_csv(path)["v"].to_list() == [10.0, 20.0]
    assert not Path(path.with_name(f".{path.name}.transaction")).exists()


class FlakyEveryOther:
    """Fails the first attempt of every rename once, as a scanner briefly holding each
    freshly written file would."""

    def __init__(self, real: HeldTarget) -> None:
        self.real = real
        self.failed = 0
        self._retrying = False

    def __call__(self, source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if not self._retrying:
            self._retrying = True
            self.failed += 1
            raise PermissionError(13, "The process cannot access the file", str(target))
        self._retrying = False
        self.real(source, target)
