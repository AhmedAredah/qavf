"""Findings of checks relating many datasets: member roles and keys without consensus.

A check with more than two datasets (or a strategy other than ``reference``)
evaluates one list entry per pair. :class:`GroupedFindings` then says which
member each finding is about (from the pair its label belongs to and, for
``compare``, its level) and, for a consensus, reports the keys that have none as
not assessed: they were left out of every pair rather than guessed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import polars as pl

from qavf.compare.consensus import NO_CONSENSUS
from qavf.compare.pairing import GroupPlan
from qavf.engine.ports import FindingReads
from qavf.engine.registry import Finding

COMPARISON_IDENTITY = "_qavf_comparison_identity"
NO_CONSENSUS_FIELD = "no_consensus"
"""Evidence field giving why a key has no consensus."""


class GroupedFindings:
    """Completes the findings of one check relating many datasets."""

    def __init__(self, plan: GroupPlan) -> None:
        self.plan = plan

    def requirements(self) -> frozenset[str]:
        """Datasets read besides the pairs' own (the consensus with its gaps)."""
        return frozenset([self.plan.consensus_dataset] if self.plan.consensus_dataset else [])

    def finish(self, ctx: FindingReads, findings: Sequence[Finding]) -> list[Finding]:
        """``findings`` with member roles, plus the consensus report."""
        completed = [
            self.assign(finding) for summary in findings for finding in self.split(summary)
        ]
        if self.plan.consensus_dataset is not None:
            completed.append(self.consensus(ctx, self.plan.consensus_dataset))
        return completed

    def split(self, finding: Finding) -> list[Finding]:
        """A passing summary of several pairs as one passing finding per pair, so each
        member's result is reported."""
        passed = finding.metrics.get("passed_checks")
        if finding.check is not None or finding.skipped or not finding.passed:
            return [finding]
        if not isinstance(passed, list) or not passed:
            return [finding]
        split = []
        for label in passed:
            pair = self.plan.role_for(label)
            datasets = (pair.reference, pair.other) if pair is not None else ()
            split.append(
                Finding.ok(
                    f"{label}: passed",
                    check=label,
                    datasets=tuple(name for name in datasets if name in self.plan.members),
                    check_id=finding.check_id,
                )
            )
        return split

    def assign(self, finding: Finding) -> Finding:
        """``finding`` with the role of the pair (and level) it belongs to."""
        if finding.group is not None:
            return finding
        identity = finding.metrics.get(COMPARISON_IDENTITY)
        level = identity.get("level") if isinstance(identity, dict) else None
        pair = self.plan.role_for(finding.check)
        role = self.plan.pair_role(pair, level if isinstance(level, str) else None)
        return dataclasses.replace(finding, group=role)

    def consensus(self, ctx: FindingReads, dataset: str) -> Finding:
        """The keys without a consensus value: not assessed for those keys."""
        label = f"{self.plan.name}: consensus"
        common = {
            "check": label,
            "datasets": self.plan.members,
            "unit": "keys",
            "group": self.plan.pair_role(None, "consensus"),
        }
        if not ctx.has(dataset):
            reason = ctx.unavailable_reason(dataset) or "it could not be built"
            return Finding.unassessed(f"{label}: no consensus was built ({reason})", **common)
        frame = ctx.get(dataset)
        missing = frame.filter(pl.col(NO_CONSENSUS).is_not_null())
        total = frame.height
        if not missing.height:
            return Finding.ok(
                f"{label}: every one of {total:,} key(s) has a consensus value",
                total_records=total,
                affected_count=0,
                **common,
            )
        evidence = (
            missing.head(ctx.evidence_limit)
            .select([*self.plan.keys, NO_CONSENSUS])
            .rename({NO_CONSENSUS: NO_CONSENSUS_FIELD})
        )
        return Finding.unassessed(
            f"{label}: {missing.height:,} of {total:,} key(s) have no consensus value (ties or "
            "too little agreement among the members) and are not assessed",
            affected=evidence.to_dicts(),
            affected_count=missing.height,
            total_records=total,
            row_column=None,
            **common,
        )


__all__ = ["NO_CONSENSUS_FIELD", "GroupedFindings"]
