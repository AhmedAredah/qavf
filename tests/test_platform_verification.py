"""Verification orchestration must fail closed without genuine host evidence."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def platform_checks(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("release_checks.platform")


@pytest.mark.parametrize("label", ["linux-3.12", "windows-3.12", "macos-3.12"])
def test_every_host_runs_full_source_and_wheel_modes(platform_checks, tmp_path, label):
    plan = dict(platform_checks.command_plan(tmp_path, tmp_path / "evidence", label))
    assert "--optimized" in plan["source-optimized"]
    # Inspect arguments, not substrings of caller-owned absolute paths.
    expected = ["test", "--id", f"{label}-source", "--output",
                str(tmp_path / "evidence" / f"{label}-source.json")]
    if label == "linux-3.12":
        expected.append("--coverage")
    assert plan["source"][2:] == expected
    assert plan["wheel"][-3] == f"{label}-wheel"
    assert "--mode" in plan["wheel"] and "locked" in plan["wheel"]
    assert all(isinstance(argv, list) for argv in plan.values())


@pytest.mark.parametrize("system,minor", [("FreeBSD", 12), ("Windows", 13),
                                         ("Darwin", 14), ("Linux", 11)])
def test_unsupported_hosts_are_rejected(platform_checks, monkeypatch, system, minor):
    monkeypatch.setattr(platform_checks.platform, "system", lambda: system)
    monkeypatch.setattr(sys, "version_info", SimpleNamespace(major=3, minor=minor))
    with pytest.raises(ValueError):
        platform_checks.host_label()


def test_failed_build_blocks_stale_artifacts_and_keeps_source_checks(
    platform_checks, monkeypatch, tmp_path,
):
    calls = []
    monkeypatch.setattr(platform_checks, "host_label", lambda: "linux-3.12")
    monkeypatch.setattr(platform_checks.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(platform_checks, "repository_identity", lambda _: {})

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if "build" in argv else 0)

    monkeypatch.setattr(platform_checks.subprocess, "run", run)
    output = tmp_path / "attempt"
    assert platform_checks.verify_platform(tmp_path, output) == 1
    assert sum("test" in argv for argv in calls) == 2
    assert not any("installed" in argv for argv in calls)
    summary = json.loads((output / "platform-summary.json").read_text())
    assert summary["status"] == "blocked" and not summary["release_ready"]
    assert summary["errors"]
    with pytest.raises(ValueError, match="empty"):
        platform_checks.verify_platform(tmp_path, output)


def test_evidence_cannot_overwrite_source(platform_checks, monkeypatch, tmp_path):
    monkeypatch.setattr(platform_checks, "host_label", lambda: "linux-3.12")
    monkeypatch.setattr(platform_checks.shutil, "which", lambda _: "uv")
    for name in ("qavf", "tests", "scripts", "docs", "config", ".github", "dist"):
        with pytest.raises(ValueError, match="outside"):
            platform_checks.verify_platform(tmp_path, tmp_path / name / "attempt")


@pytest.mark.parametrize("interrupt", [False, True])
def test_exit_codes_without_records_never_qualify(
    platform_checks, monkeypatch, tmp_path, interrupt,
):
    monkeypatch.setattr(platform_checks, "host_label", lambda: "linux-3.12")
    monkeypatch.setattr(platform_checks.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(platform_checks, "repository_identity", lambda _: {})

    def run(*args, **kwargs):
        if interrupt:
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(platform_checks.subprocess, "run", run)
    output = tmp_path / "attempt"
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            platform_checks.verify_platform(tmp_path, output)
    else:
        assert platform_checks.verify_platform(tmp_path, output) == 1
    summary = json.loads((output / "platform-summary.json").read_text())
    assert summary["status"] == "blocked"
    assert summary["interrupted"] is interrupt
    assert not summary["release_ready"]


def test_unicode_evidence_directory_with_legacy_redirected_output(
    platform_checks, monkeypatch, tmp_path,
):
    import io

    monkeypatch.setattr(platform_checks, "host_label", lambda: "windows-3.12")
    monkeypatch.setattr(platform_checks.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(platform_checks, "repository_identity", lambda _: {})
    monkeypatch.setattr(platform_checks.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=1))
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    output = tmp_path / "résultats العربية"
    assert platform_checks.verify_platform(tmp_path, output) == 1
    stream.flush()
    assert "Running" in buffer.getvalue().decode("ascii")
    assert (output / "platform-summary.json").is_file()
