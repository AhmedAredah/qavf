"""``run.require``: checks that must actually be assessed, for either backend.

An entry names check items (their ``name``, or the default ``<check> on
<dataset>``) or a dataset, which stands for every check on that dataset and its
schema contracts. Results are matched to items by instance id, never by name,
and every item an entry names must be assessed: a requirement is met when each
of them produced a PASS or FAIL and none could not be assessed; otherwise the run
gains an ERROR result explaining why. Items a run-time selection leaves out are
not required.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qavf.checks.placement import placed_items
from qavf.engine.severity import Severity, Status
from qavf.input_rules import INPUT_RULES, required_not_assessed
from qavf.reports.models import RuleResult

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings


@dataclass(frozen=True, slots=True)
class RuleDescription:
    """What a requirement result says about the rule it stands for."""

    rule_id: str
    check_name: str
    name: str
    description: str
    category: str
    default_severity: Severity
    recommendation: str


def input_rule_description(rule_id: str) -> RuleDescription:
    """Describe an input rule from the backend-independent input-rule catalog."""
    rule = INPUT_RULES[rule_id]
    return RuleDescription(
        rule.rule_id,
        rule.check_name,
        rule.name,
        rule.description,
        rule.category,
        rule.default_severity,
        rule.recommendation,
    )


@dataclass(frozen=True, slots=True)
class _Item:
    item_id: str
    """The item's instance id (its location), which its results carry."""
    name: str
    check: str
    datasets: tuple[str, ...]
    enabled: bool


def _items(settings: QAVFSettings) -> list[_Item]:
    """Every check item, except those a run-time selection left out: a selection
    narrows the requirements to the items it runs."""
    applied = settings.selection
    left_out = applied.not_selected if applied is not None else frozenset()
    return [
        _Item(placed.path, placed.name, placed.item.check, placed.datasets, placed.item.enabled)
        for placed in placed_items(settings)
        if placed.path not in left_out
    ]


class Requirements:
    """Assess ``run.require`` against a run's complete results."""

    def __init__(
        self,
        settings: QAVFSettings,
        describe: Callable[[str], RuleDescription | None],
        check_rules: Mapping[str, str],
    ) -> None:
        """``describe`` gives a rule's description by rule id; ``check_rules`` maps
        check names to rule ids."""
        self.settings = settings
        self.describe = describe
        self.check_rules = dict(check_rules)
        self.items = _items(settings)

    def apply(self, results: Sequence[RuleResult]) -> list[RuleResult]:
        """``results`` plus an ERROR for each requirement that was not assessed."""
        added = []
        for entry in self.settings.run.require:
            reason = self._unmet(entry, results)
            if reason is not None:
                added.append(self._result(entry, reason))
        return [*results, *added]

    def _unmet(self, entry: str, results: Sequence[RuleResult]) -> str | None:
        """Why ``entry`` is not met, or ``None``. A name stands for every item that
        has it, and each of them must be assessed."""
        named = [item for item in self.items if item.name == entry]
        if named:
            return self._unmet_items(named, results)
        if entry in self.settings.datasets:
            on_it = [item for item in self.items if entry in item.datasets]
            disabled = [item.name for item in on_it if not item.enabled]
            if disabled:
                return f"check {disabled[0]!r} on it is disabled in the configuration"
            for name in dict.fromkeys(item.name for item in on_it):
                reason = self._unmet_items([i for i in on_it if i.name == name], results)
                if reason is not None:
                    return f"check {name!r} on it: {reason}"
            blocked = [r for r in results if entry in r.datasets and r.status is Status.ERROR]
            return "it could not be assessed" if blocked else None
        return "no check or dataset has this name"

    @staticmethod
    def _unmet_items(items: Sequence[_Item], results: Sequence[RuleResult]) -> str | None:
        """Why one of ``items`` is not assessed (naming it when there are several)."""
        for item in items:
            if not item.enabled:
                reason: str | None = "the check is disabled in the configuration"
            else:
                reason = _assessed([r for r in results if item.item_id in r.item_ids])
            if reason is not None:
                return f"its item {item.item_id}: {reason}" if len(items) > 1 else reason
        return None

    def _result(self, entry: str, reason: str) -> RuleResult:
        item = next((item for item in self.items if item.name == entry), None)
        rule_id = self.check_rules.get(item.check, "INP-001") if item else "INP-001"
        description = self.describe(rule_id) or input_rule_description("INP-001")
        return RuleResult(
            rule_id=description.rule_id,
            name=description.name,
            description=description.description,
            category=description.category,
            check=f"required: {entry}",
            check_name=item.check if item else None,
            item_name=entry if item else None,
            item_ids=[named.item_id for named in self.items if named.name == entry],
            status=Status.ERROR,
            severity=max(description.default_severity, Severity.ERROR),
            message=required_not_assessed(entry, reason),
            affected_count_exact=False,
            evidence_complete=None,
            recommendation=description.recommendation,
            datasets=list(item.datasets) if item else [entry],
        )


def _assessed(matching: Sequence[RuleResult]) -> str | None:
    """Why ``matching`` results do not assess a requirement, or ``None`` when they do."""
    if not matching:
        return "no result was produced for it"
    if any(r.status is Status.ERROR for r in matching):
        return "it could not be assessed"
    if not any(r.status in (Status.PASS, Status.FAIL) for r in matching):
        return "it was skipped"
    return None


__all__ = ["Requirements", "RuleDescription", "input_rule_description"]
