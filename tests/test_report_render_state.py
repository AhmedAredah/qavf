"""Document ownership and occurrence identity across streaming report renders."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest

from qavf.compare.results import CheckResult, DatasetComparison, LevelResult
from qavf.engine.severity import Severity, Status
from qavf.reports import charts, comparison
from qavf.reports.html import render_html
from qavf.reports.models import QAReport, RuleResult, RunMetadata


def _report(*results: RuleResult) -> QAReport:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return QAReport(
        metadata=RunMetadata(
            run_id="render-state",
            project="project",
            qavf_version="test",
            started_at=now,
            finished_at=now,
        ),
        results=list(results),
    )


def _failure() -> RuleResult:
    return RuleResult(
        rule_id="EX-001",
        name="Repeated",
        description="Same immutable result",
        category="test",
        check="same label",
        status=Status.FAIL,
        severity=Severity.WARNING,
        message="failed",
        recommendation="repair",
        metrics={"value": 3},
    )


def test_repeated_result_object_has_distinct_occurrence_navigation() -> None:
    result = _failure()
    report = _report(result, result)
    assert report.results[0] is report.results[1]
    html = render_html(report)
    ids = re.findall(r'\bid="([^"]+)"', html)
    links = re.findall(r'href="#([^"]+)"', html)
    assert len(ids) == len(set(ids))
    assert set(links) <= set(ids)
    assert 'id="EX-001-same-label"' in html
    assert 'id="EX-001-same-label-2"' in html
    assert 'href="#EX-001-same-label-2"' in html
    assert render_html(report) == html


def test_interleaved_comparison_streams_have_independent_ids(monkeypatch) -> None:
    def builders(level, budget):
        return [
            budget.chart(
                level.name,
                1,
                lambda: charts.deltas([level.name], [-1.0], level.name, state=budget.state),
            )
        ]

    monkeypatch.setitem(comparison._CHARTS, "render_probe", builders)
    result = DatasetComparison(
        name="interleaved",
        baseline="before",
        candidate="after",
        levels=[
            LevelResult(
                level="render_probe",
                name=str(index),
                checks=[CheckResult(check="probe", passed=True, message="checked")],
            )
            for index in range(3)
        ],
    )
    expected = b"".join(comparison.iter_comparison(result))
    first = iter(comparison.iter_comparison(result))
    prefix = []
    while b"hatch-1" not in b"".join(prefix):
        prefix.append(next(first))
    second = b"".join(comparison.iter_comparison(result))
    resumed = b"".join(prefix) + b"".join(first)
    assert resumed == second == expected
    assert re.findall(rb'id="hatch-(\d+)"', resumed) == [b"1", b"2", b"3"]


def test_standalone_charts_accept_an_explicit_document_state() -> None:
    state = charts.RenderState()
    first = charts.deltas(["a"], [-1.0], "first", state=state)
    second = charts.paired_bars(["a"], [0.2], [0.3], "second", state=state)
    assert 'id="hatch-1"' in first
    assert 'id="hatch-2"' in second
    assert "url(#hatch-2)" in second


def test_analysis_policy_and_recommendations_are_local() -> None:
    report = _report(_failure())
    permissive = report.analyze(Severity.ERROR)
    strict = report.analyze(Severity.WARNING)
    assert permissive.acceptance.verdict == "passed"
    assert strict.acceptance.verdict == "failed"
    assert permissive.recommendations is permissive.recommendations
    assert list(permissive.recommendations) == report.recommendations()
    assert report.acceptance == report.analyze().acceptance
    assert json.loads(b"".join(report.iter_json_chunks())) == json.loads(report.model_dump_json())


@pytest.mark.parametrize("operation", ["html", "json"])
def test_each_presentation_computes_summary_once(monkeypatch, operation) -> None:
    report = _report(_failure())
    original = QAReport.summary.fget
    calls = []

    def count(instance):
        calls.append(instance)
        return original(instance)

    monkeypatch.setattr(QAReport, "summary", property(count))
    if operation == "html":
        render_html(report)
    else:
        list(report.iter_json_chunks())
    assert calls == [report]


def test_lower_bound_count_survives_json_and_is_explicit_in_html() -> None:
    result = _failure().model_copy(update={"affected_count": 2, "affected_count_exact": False})
    report = _report(result)
    payload = json.loads(b"".join(report.iter_json_chunks()))
    assert payload["results"][0]["affected_count_exact"] is False
    restored = QAReport.model_validate(payload)
    assert not restored.results[0].affected_count_exact
    assert "at least 2" in render_html(restored)
    payload["results"][0].pop("affected_count_exact")
    assert QAReport.model_validate(payload).results[0].affected_count_exact


def test_concurrent_closed_and_failed_generators_cannot_change_another_render(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    fail = False

    def builders(level, budget):
        if fail and level.name == "second":
            raise RuntimeError("interrupted chart")
        return [
            budget.chart(
                level.name,
                1,
                lambda: charts.deltas([level.name], [-1.0], level.name, state=budget.state),
            )
        ]

    monkeypatch.setitem(comparison._CHARTS, "state_lifetime", builders)
    result = DatasetComparison(
        name="lifetime",
        baseline="before",
        candidate="after",
        levels=[
            LevelResult(
                level="state_lifetime",
                name=name,
                checks=[CheckResult(check="probe", passed=True, message="checked")],
            )
            for name in ("first", "second")
        ],
    )

    def render(_=None):
        return b"".join(comparison.iter_comparison(result))

    expected = render()
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(render, range(8))) == [expected] * 8
    stream = comparison.iter_comparison(result)
    while b"hatch-1" not in next(stream):
        pass
    stream.close()
    assert render() == expected
    fail = True
    with pytest.raises(RuntimeError, match="interrupted chart"):
        render()
    fail = False
    assert render() == expected
