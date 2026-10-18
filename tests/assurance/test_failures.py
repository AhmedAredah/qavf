"""Fault injection at public execution and durable audit boundaries."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest

from qavf import Finding, Loader, QARunner, QAVFSettings, Rule, RuleRegistry, Status, validate
from qavf.reports.models import QAReport
from qavf.sources.base import SourceEnvironment, SourceError
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams

from .support.adapters import BACKENDS, adapted_frames


class _HealthyRule(Rule):
    rule_id = "ASSURANCE-HEALTHY"
    name = "Independent successful assessment"
    description = "A passing sibling must not conceal a blocked assessment"
    category = "assurance"

    def evaluate(self, ctx):
        return Finding.ok("independent check completed")


def _runner(evaluate: Callable[..., Any], *, hash_inputs: bool = False, **engine: Any) -> QARunner:
    """Run an injected plugin beside a healthy one with an explicit requirement."""

    class Probe(Rule):
        rule_id = "ASSURANCE-PROBE"
        name = "Injected plugin"
        description = "Exercise the public finding contract"
        category = "assurance"

        def evaluate(self, ctx):
            return evaluate(ctx)

    registry = RuleRegistry()
    registry.register(Probe)
    registry.register(_HealthyRule)
    return QARunner(
        QAVFSettings(
            plugins={"entry_points": False},
            engine={"parallel": False, "require": [Probe.rule_id], **engine},
            rules={Probe.rule_id: {"severity": "INFO"}, _HealthyRule.rule_id: {}},
            report={"hash_inputs": hash_inputs},
        ),
        registry,
    )


def _assert_incomplete(report: QAReport) -> None:
    """The in-memory and persisted acceptance decisions must agree."""
    restored = QAReport.model_validate_json(b"".join(report.iter_json_chunks()))
    for snapshot in (report, restored):
        assert snapshot.incomplete
        assert snapshot.verdict() != "passed"
        assert snapshot.exit_code() == 1
        assert any(result.status is Status.ERROR for result in snapshot.results)


@pytest.mark.parametrize("field", ["passed", "skipped", "assessed"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", []])
def test_malformed_plugin_flags_cannot_be_interpreted_by_truthiness(field, value):
    def evaluate(ctx):
        return Finding(**{"passed": True, "message": "invalid flag", field: value})

    report = _runner(evaluate).run()
    _assert_incomplete(report)
    assert report.get("ASSURANCE-PROBE").status is Status.ERROR
    assert f"Finding.{field}" in report.get("ASSURANCE-PROBE").message


@pytest.mark.parametrize("field", ["affected_count", "total_records", "affected_rows"])
@pytest.mark.parametrize("value", [-1, True, 1.25, "1", float("nan")])
def test_malformed_plugin_counts_cannot_be_rounded_or_enter_the_ledger(field, value):
    report = _runner(lambda ctx: Finding.ok("invalid count", **{field: value})).run()
    _assert_incomplete(report)
    assert f"Finding.{field}" in report.get("ASSURANCE-PROBE").message


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("failure", ["raise", "iterator", "malformed", "unassessed", "empty"])
def test_partial_plugin_execution_never_certifies_a_required_assessment(failure, parallel):
    def interrupted():
        yield Finding.ok("first check completed", check="first")
        raise RuntimeError("injected failure after a partial result")

    def evaluate(ctx):
        match failure:
            case "raise":
                raise RuntimeError("injected evaluation failure")
            case "iterator":
                return interrupted()
            case "malformed":
                return [Finding.ok("first check completed"), None]
            case "unassessed":
                return [Finding.ok("first", check="first"), Finding.unassessed("second")]
            case _:
                return []

    _assert_incomplete(_runner(evaluate, parallel=parallel).run())


@pytest.mark.parametrize(
    "retryable,replayable,expected_calls", [(False, True, 1), (True, False, 1), (True, True, 3)]
)
def test_exhausted_or_unsafe_loader_retries_never_validate_a_good_sibling(
    retryable,
    replayable,
    expected_calls,
):
    calls = []

    def unavailable():
        calls.append(1)
        raise SourceError("injected transport failure", retryable=retryable)

    report = validate(
        {
            "good": pl.DataFrame({"x": [1]}),
            "unavailable": Loader(unavailable, replayable=replayable, retry_safe=replayable),
        },
        config={
            "datasets": {
                name: {
                    "schema": {"columns": {"x": {"ge": 0}}},
                    "load_policy": {"retries": 2, "retry_delay_s": 0},
                }
                for name in ("good", "unavailable")
            },
            "plugins": {"entry_points": False},
            "engine": {"parallel": True, "max_workers": 2},
        },
    )
    assert len(calls) == expected_calls
    _assert_incomplete(report)
    assert "unavailable" in report.metadata.unavailable_datasets
    assert any(result.status is Status.PASS for result in report.results)


@pytest.mark.parametrize("unsupported", [object(), {"x": [1]}, [{"x": 1}]])
def test_loader_conversion_failure_is_reported_without_assessing_a_prefix(unsupported):
    report = validate(
        Loader(lambda: unsupported),
        schema={"columns": {"x": {"ge": 0}}},
        config={"plugins": {"entry_points": False}, "engine": {"parallel": False}},
    )
    _assert_incomplete(report)
    assert not any(result.status is Status.PASS for result in report.results_for("INP-003"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_unsupported_numeric_constraint_remains_unassessed_beside_valid_checks(backend):
    frame = pl.DataFrame({"numeric": [1, 2], "text": ["1", "not-a-number"]})
    with adapted_frames({"data": frame}, backend) as inputs:
        report = validate(
            inputs,
            config={
                "datasets": {
                    "data": {"schema": {"columns": {"numeric": {"ge": 0}, "text": {"ge": 0}}}}
                },
                "plugins": {"entry_points": False},
                "engine": {"parallel": False},
            },
        )
    _assert_incomplete(report)
    checks = report.results_for("INP-003")
    assert any(result.status is Status.PASS for result in checks)
    assert any(result.status is Status.ERROR and "text" in result.message for result in checks)


def test_content_hash_detects_same_size_same_timestamp_rewrite_after_a_passing_check(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    before = path.stat()

    def evaluate(ctx):
        assert ctx.sql("SELECT x FROM data").item() == 1
        path.write_text("x\n9\n", encoding="utf-8")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_size == before.st_size
        assert path.stat().st_mtime_ns == before.st_mtime_ns
        return Finding.ok("assessed original snapshot")

    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    report = _runner(evaluate, hash_inputs=True).run(frames={"data": source})
    _assert_incomplete(report)
    assert report.get("ASSURANCE-PROBE").status is Status.PASS
    assert report.get("QAVF-INPUT-INTEGRITY").status is Status.ERROR
    assert "data" in report.metadata.unavailable_datasets


@pytest.mark.parametrize("boundary", ["serialization", "fsync", "replace"])
def test_failed_report_publication_preserves_last_complete_evidence(
    tmp_path, monkeypatch, boundary
):
    report = _runner(lambda ctx: Finding.fail("complete failure evidence", affected_count=7)).run()
    target = report.write_json(tmp_path / "report.json")
    original = target.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    if boundary == "serialization":

        def broken_chunks(self):
            yield b'{"partial":'
            raise OSError("injected publication failure")

        monkeypatch.setattr(QAReport, "iter_json_chunks", broken_chunks)
    else:
        monkeypatch.setattr(f"qavf.io.os.{boundary}", fail)
    with pytest.raises(OSError, match="publication failure"):
        report.write_json(target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
    assert QAReport.model_validate_json(original).get("ASSURANCE-PROBE").affected_count == 7
