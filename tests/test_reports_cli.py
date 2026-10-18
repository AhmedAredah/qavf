"""Result models, HTML rendering, the CLI and the demo project end to end."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime

import pytest
import yaml

from qavf.cli import EXIT_OK, EXIT_QA_FAILED, EXIT_USAGE, main
from qavf.demo import DEFECTS, inject_defects, write_demo_project
from qavf.engine.runner import QARunner
from qavf.engine.severity import Severity, Status
from qavf.reports.charts import BarRow, Segment, horizontal_bars
from qavf.reports.html import render_html, write_html
from qavf.reports.models import QAReport, RuleResult, RunMetadata
from qavf.settings import load_settings


def _result(rule_id: str, status: Status, severity: Severity, **extra) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        name=f"Rule {rule_id}",
        description="A rule.",
        category=extra.pop("category", "input"),
        status=status,
        severity=severity,
        message=extra.pop("message", "message"),
        **extra,
    )


def _report(*results: RuleResult, **metadata) -> QAReport:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    meta = RunMetadata(
        run_id="abc", project="P", qavf_version="0", started_at=now, finished_at=now, **metadata
    )
    return QAReport(metadata=meta, results=list(results))


# --------------------------------------------------------------------- models
def test_summary_counts_and_ordering() -> None:
    report = _report(
        _result("A", Status.PASS, Severity.CRITICAL),
        _result(
            "B",
            Status.FAIL,
            Severity.WARNING,
            recommendation="fix b",
            affected_count=5,
            total_records=20,
        ),
        _result("C", Status.FAIL, Severity.CRITICAL, recommendation="fix c", category="moves"),
        _result("D", Status.ERROR, Severity.ERROR),
        _result("E", Status.SKIPPED, Severity.ERROR),
    )
    summary = report.summary
    assert (summary.total, summary.passed, summary.failed, summary.errored, summary.skipped) == (
        5,
        1,
        2,
        1,
        1,
    )
    assert summary.highest_severity is Severity.CRITICAL
    assert summary.pass_rate == pytest.approx(1 / 3)
    assert [r.rule_id for r in report.failures()] == ["C", "D", "B"]
    assert [r.rule_id for r in report.failures(Severity.ERROR)] == ["C", "D"]
    assert [(r.rule_id, r.text) for r in report.recommendations()] == [
        ("C", "fix c"),
        ("B", "fix b"),
    ]
    assert report.get("B").affected_fraction == 0.25
    with pytest.raises(KeyError):
        report.get("Z")
    assert _report().summary.pass_rate is None


def test_json_round_trip(tmp_path, demo) -> None:
    from qavf.testing import run_rules

    report = run_rules(
        inject_defects(demo, ["negative_aadt"]),
        datasets={"counts": {"schema": {"columns": {"aadt": {"ge": 0}}}}},
    )
    path = report.write_json(tmp_path / "out" / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["failed"] == report.summary.failed == 1
    restored = QAReport.model_validate({k: v for k, v in payload.items() if k != "summary"})
    assert [r.check for r in restored.results] == [r.check for r in report.results]


# ----------------------------------------------------------------------- html
def test_html_report_content(tmp_path) -> None:
    config = write_demo_project(tmp_path / "demo")
    report = QARunner(load_settings(config)).run()
    html = render_html(report, title="My QA", fail_on=Severity.ERROR)
    assert "<title>My QA" in html and "QA failed" in html
    assert html.count("<svg") == 3
    assert "counts.aadt: ge 0" in html and 'id="INP-003-counts-aadt-ge-0"' in html
    assert "http://" not in html and "https://" not in html  # self-contained
    target = write_html(report, tmp_path / "nested" / "report.html")
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_html_verdicts_escaping_and_extensions() -> None:
    hostile = _result(
        "X",
        Status.FAIL,
        Severity.WARNING,
        message="<script>alert(1)</script>",
        check="<b>bold</b>",
        affected_records=[{"col<b>": "<img src=x>", "n": 1234567.0, "none": None, "lst": [1, 2]}],
        affected_count=1,
    )
    html = render_html(
        _report(hostile, extensions={"moves": {"version": "MOVES5"}}), fail_on=Severity.ERROR
    )
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "<img src=x>" not in html and "<b>bold</b>" not in html
    assert "1,234,567" in html and "1, 2" in html
    assert "QA passed with findings" in html and "moves version=MOVES5" in html
    assert "QA failed" in render_html(_report(hostile), fail_on=Severity.WARNING)
    assert "No rule failed." in render_html(_report(_result("A", Status.PASS, Severity.ERROR)))


def test_chart_rendering() -> None:
    svg = horizontal_bars(
        [BarRow("a <b>", [Segment("pass", 3, "s-pass"), Segment("fail", 0, "s-fail")])], "x"
    )
    assert svg.count("<rect") == 1 and "&lt;b&gt;" in svg
    assert "Nothing to show" in horizontal_bars([], "empty")


# ------------------------------------------------------------------------ cli
def test_demo_project_clean_and_defective(tmp_path, capsys) -> None:
    assert main(["demo", str(tmp_path / "clean"), "--clean"]) == EXIT_OK
    assert (tmp_path / "clean" / "qavf.schema.json").is_file()
    assert main(["run", "-c", str(tmp_path / "clean" / "qavf.yaml")]) == EXIT_OK
    assert "QA PASSED" in capsys.readouterr().out

    assert main(["demo", str(tmp_path / "bad")]) == EXIT_OK
    assert (
        main(["run", "-c", str(tmp_path / "bad" / "qavf.yaml"), "--no-parallel"]) == EXIT_QA_FAILED
    )
    out = capsys.readouterr().out
    assert "INP-003 [counts.aadt: ge 0]" in out and "QA FAILED" in out
    assert (tmp_path / "bad" / "qa_output" / "qa_report.html").is_file()
    assert main(["demo", str(tmp_path / "bad")]) == EXIT_USAGE  # not empty
    assert main(["demo", str(tmp_path / "x"), "--defects", "nope"]) == EXIT_USAGE


#: Rules each demo defect must trip (a lower bound: knock-on failures are fine).
EXPECTED = {
    "negative_aadt": {"INP-003", "REG-001"},
    "truck_share_percent": {"INP-003"},
    "missing_count": {"TMP-004", "SPA-003", "REG-001"},
    "aadt_spike": {"TMP-001", "TMP-003", "SPA-001", "SPA-002", "REG-001"},
    "fatal_exceeds": {"INP-006"},
    "unknown_site": {"INP-005"},
    "duplicate_count": {"INP-004"},
    "bad_site_id": {"INP-003", "INP-005"},
    "future_date": {"INP-003"},
    "declining_site": {"TMP-002"},
}


def test_every_defect_has_an_expectation() -> None:
    assert set(EXPECTED) == set(DEFECTS)


HAS_STATS = (
    importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("statsmodels") is not None
)


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_demo_defect_is_detected(defect: str, tmp_path) -> None:
    report = QARunner(load_settings(write_demo_project(tmp_path, [defect]))).run()
    failed = {r.rule_id for r in report.results if r.status is Status.FAIL}
    expected = EXPECTED[defect] if HAS_STATS else EXPECTED[defect] - {"TMP-002", "TMP-003"}
    unassessed = {r.label: r.message for r in report.results if r.status is Status.ERROR}
    assert set(unassessed) == UNNORMALISABLE.get(defect, set()), f"{defect}: {unassessed}"
    assert expected <= failed, f"{defect}: expected {expected}, failed {failed}"


#: Defects that leave crash rows without a usable AADT denominator: the crash-rate
#: comparisons must report those entities as not assessable rather than drop them.
_RATE = "crashes.crashes per counts.aadt: unusable values"
UNNORMALISABLE = {
    defect: {f"SPA-001 [{_RATE}]", f"SPA-002 [{_RATE}]"}
    for defect in ("negative_aadt", "missing_count", "bad_site_id", "unknown_site")
}
# A malformed site creates a singleton temporal group; partial scope is explicit.
UNNORMALISABLE["bad_site_id"].update(
    f"{rule} [incomplete series assessment]" for rule in ("TMP-001", "TMP-002", "TMP-003")
)
# Duplicate candidate keys prevent an unambiguous baseline comparison.
UNNORMALISABLE["duplicate_count"] = {"REG-001 [counts]"}


def test_cli_overrides_and_errors(tmp_path, capsys) -> None:
    config = write_demo_project(tmp_path, ["declining_site"])
    settings = yaml.safe_load(config.read_text())
    settings["rules"]["REG-001"]["enabled"] = False  # leaves only WARNING findings
    config.write_text(yaml.safe_dump(settings))
    assert main(["run", "-c", str(config)]) == EXIT_OK
    assert main(["run", "-c", str(config), "--fail-on", "WARNING"]) == EXIT_QA_FAILED
    assert main(["run", "-c", str(tmp_path / "absent.yaml")]) == EXIT_USAGE
    settings = yaml.safe_load(config.read_text())
    settings["rules"]["NOPE-1"] = {}
    config.write_text(yaml.safe_dump(settings))
    assert main(["run", "-c", str(config)]) == EXIT_USAGE
    assert "unknown rule id" in capsys.readouterr().err


def test_cli_list_compare_baseline_profile(tmp_path, capsys) -> None:
    for fmt in ("table", "markdown", "json"):
        assert main(["list-rules", "--format", fmt]) == EXIT_OK
    out = capsys.readouterr().out
    assert "INP-003" in out and "| INP-003 |" in out and '"rule_id": "INP-003"' in out

    write_demo_project(tmp_path / "a", clean=True)
    write_demo_project(tmp_path / "b", ["aadt_spike"])
    counts_a, counts_b = (
        tmp_path / "a" / "data" / "counts.csv",
        tmp_path / "b" / "data" / "counts.csv",
    )
    baseline = tmp_path / "baseline.csv"
    assert main(["baseline", str(counts_a), str(baseline), "-k", "site_id", "year"]) == EXIT_OK
    assert main(["compare", str(baseline), str(counts_a), "-k", "site_id", "year"]) == EXIT_OK
    assert (
        main(["compare", str(baseline), str(counts_b), "-k", "site_id", "year"]) == EXIT_QA_FAILED
    )
    assert "site_id=S03, year=2021" in capsys.readouterr().out

    assert main(["profile", str(counts_a)]) == EXIT_OK
    profile = yaml.safe_load(capsys.readouterr().out)
    assert profile["schema"]["columns"]["count_type"]["isin"] == ["permanent", "portable"]
