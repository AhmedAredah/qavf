"""Small provenance and acceptance services independent of execution backend."""

from __future__ import annotations

import contextlib
import functools
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from qavf._version import __version__
from qavf.engine.severity import Severity, Status
from qavf.input_rules import INPUT_RULES, required_not_assessed
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


def requirements(results, entries, excluded=None):
    """Assess explicit acceptance requirements against the complete outcome ledger.

    ``excluded`` gives the reason each rule that did not run was left out.
    """
    excluded = excluded or {}
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
        added.append(_required(rule_id, entry, _reason(rule_id, matching, excluded)))
    return [*results, *added]


def _reason(rule_id, matching, excluded):
    """Why a required check has no assessed result, in the full backend's words."""
    if rule_id in excluded:
        return f"the rule was excluded: {excluded[rule_id]}"
    if not matching:
        return "no result was produced for it"
    if any(r.status is Status.ERROR for r in matching):
        return "it could not be assessed"
    return "it was skipped"


def _required(rule_id, entry, reason):
    rule = INPUT_RULES.get(rule_id)
    return RuleResult(
        rule_id=rule_id,
        name=rule.name if rule else "required check",
        description=rule.description if rule else "Required acceptance contract",
        category=rule.category if rule else "input",
        check=f"required: {entry}",
        status=Status.ERROR,
        severity=max(rule.default_severity if rule else Severity.ERROR, Severity.ERROR),
        message=required_not_assessed(entry, reason),
        affected_count_exact=False,
        evidence_complete=None,
        recommendation=rule.recommendation if rule else "",
    )


def environment():
    """Detached process-stable environment, without repeated package metadata I/O."""
    return dict(_environment())
