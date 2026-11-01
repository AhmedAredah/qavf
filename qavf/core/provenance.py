"""Small provenance and acceptance services independent of execution backend."""

from __future__ import annotations

import contextlib
import functools
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from qavf._version import __version__
from qavf.engine.severity import Severity, Status
from qavf.reports.models import RuleResult


@functools.cache
def _environment():
    result = {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "qavf": __version__,
    }
    for package in ("duckdb", "pyarrow", "pydantic"):
        with contextlib.suppress(PackageNotFoundError):
            result[package] = version(package)
    return tuple(result.items())


def requirements(results, entries):
    """Assess explicit acceptance requirements against the complete outcome ledger."""
    added = []
    for entry in entries:
        rule_id = entry.split("[", 1)[0].strip()
        check = (
            entry.split("[", 1)[1].rstrip()[:-1]
            if "[" in entry and entry.rstrip().endswith("]")
            else None
        )
        matching = [
            r
            for r in results
            if r.rule_id == rule_id
            and (
                check is None or r.check == check or check in (r.metrics.get("passed_checks") or ())
            )
        ]
        if any(r.status in (Status.PASS, Status.FAIL) for r in matching) and not any(
            r.status is Status.ERROR for r in matching
        ):
            continue
        added.append(
            RuleResult(
                rule_id=rule_id,
                name="required check",
                description="Required acceptance contract",
                category="input",
                check=f"required: {entry}",
                check_id="acceptance.required",
                parameters={"required": entry},
                status=Status.ERROR,
                severity=Severity.ERROR,
                message=f"Not assessed: required check {entry!r} was not assessed.",
                affected_count_exact=False,
                evidence_complete=None,
            )
        )
    return [*results, *added]


def environment():
    """Detached process-stable environment, without repeated package metadata I/O."""
    return dict(_environment())
