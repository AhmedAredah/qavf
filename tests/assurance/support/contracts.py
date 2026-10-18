"""Public-entry-point execution and order-independent result comparisons."""

from collections.abc import Mapping, Sequence
from typing import Any

from qavf import Status, validate
from qavf.reports.models import RuleResult


def run_checks(
    inputs: Mapping[str, object],
    rule_id: str,
    *,
    datasets: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    evidence_rows: int = 4,
    assessed_only: bool = True,
) -> list[RuleResult]:
    """Run one configured rule, failing loudly on unexpected unassessed results."""
    report = validate(
        inputs,
        config={
            "datasets": dict(datasets or {name: {} for name in inputs}),
            "plugins": {"entry_points": False},
            "engine": {
                "parallel": False,
                "run_unconfigured_rules": False,
                "max_affected_records": evidence_rows,
            },
            "rules": {rule_id: {"params": dict(params or {})}},
        },
    )
    results = report.results_for(rule_id)
    assert results, [result.message for result in report.results]
    if assessed_only:
        assert all(result.status in (Status.PASS, Status.FAIL) for result in report.results), [
            result.message for result in report.results
        ]
    return results


def signatures(results: Sequence[RuleResult]) -> list[tuple[Any, ...]]:
    """Compare verdicts and exact counts while allowing legitimate evidence ordering."""
    return sorted(
        (
            result.check or "",
            result.status.value,
            result.affected_count,
            result.total_records,
            result.unit,
            result.affected_rows,
        )
        for result in results
    )
