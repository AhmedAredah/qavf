"""Record test outcomes and make unexpected skip/xfail/empty runs fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from .policy import accepted_skip, unexpected_outcomes
from .provenance import environment, timestamp, write_json


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--release-results", type=Path, help="Write strict release test evidence JSON")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("release_results"):
        config.pluginmanager.register(ReleaseResults(config), "qavf-release-results")


class ReleaseResults:
    def __init__(self, config: pytest.Config) -> None:
        self.path = config.getoption("release_results")
        self.started = timestamp()
        self.outcomes: list[dict] = []
        self.collection_errors: list[str] = []
        self.selected: list[str] = []
        self.deselected: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.selected = [item.nodeid for item in session.items]

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed or report.skipped:
            self.collection_errors.append(f"{report.nodeid}: {report.outcome}: {report.longrepr}")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call" and report.passed:
            return
        reason = (
            str(report.longrepr[2]) if isinstance(report.longrepr, tuple) else str(report.longrepr)
        )
        outcome: str = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        self.outcomes.append(
            {
                "nodeid": report.nodeid,
                "phase": report.when,
                "outcome": outcome,
                "reason": reason if not report.passed else "",
                "accepted": outcome == "skipped" and accepted_skip(report.nodeid, reason),
                "duration_seconds": report.duration,
            }
        )

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        violations = unexpected_outcomes(self.outcomes)
        passed = sum(item["outcome"] == "passed" for item in self.outcomes)
        complete = {item["nodeid"] for item in self.outcomes} >= set(self.selected)
        if violations or self.collection_errors or not passed or not complete:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        write_json(
            self.path,
            {
                "schema_version": 1,
                "status": "passed" if session.exitstatus == 0 else "failed",
                "started": self.started,
                "finished": timestamp(),
                "environment": environment(),
                "selected": self.selected,
                "deselected": self.deselected,
                "outcomes": self.outcomes,
                "collection_errors": self.collection_errors,
                "unexpected_outcomes": violations,
                "passed": passed,
                "exit_code": int(session.exitstatus),
            },
        )
