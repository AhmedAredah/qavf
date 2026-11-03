"""``run.require``: checks that must actually be assessed, for either backend.

An entry names a check item (its ``name``, or the default ``<check> on
<dataset>``) or a dataset, which stands for every check on that dataset and its
schema contracts. A requirement is met when every result for it was assessed
(PASS or FAIL) and none could not be assessed; otherwise the run gains an
ERROR result explaining why.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qavf.checks.items import CheckItem
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
    name: str
    check: str
    datasets: tuple[str, ...]
    enabled: bool
    passed_label: str | None
    """How a passing ``expression``/``sql`` item is listed in its rule's summary."""


def _items(settings: QAVFSettings) -> list[_Item]:
    items: list[_Item] = []
    for dataset, spec in settings.datasets.items():
        items.extend(_item(item, (dataset,)) for item in spec.checks)
    for item in settings.checks:
        named = item.settings.get("datasets")
        datasets = tuple(named) if isinstance(named, list | tuple) else ()
        items.append(_item(item, datasets))
    return items


def _item(item: CheckItem, datasets: tuple[str, ...]) -> _Item:
    label = None
    if item.check == "expression" and datasets:
        label = f"{datasets[0]}: {item.name or item.settings.get('expr')}"
    elif item.check == "sql":
        label = item.name
    return _Item(item.effective_name(datasets), item.check, datasets, item.enabled, label)


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
        named = [item for item in self.items if item.name == entry]
        if named:
            if not any(item.enabled for item in named):
                return "the check is disabled in the configuration"
            labels = {item.passed_label for item in named} - {None}
            matching = [
                r
                for r in results
                if r.item_name == entry or labels & set(r.metrics.get("passed_checks") or ())
            ]
            return _assessed(matching)
        if entry in self.settings.datasets:
            disabled = [
                item.name for item in self.items if entry in item.datasets and not item.enabled
            ]
            if disabled:
                return f"check {disabled[0]!r} on it is disabled in the configuration"
            for name in dict.fromkeys(item.name for item in self.items if entry in item.datasets):
                reason = self._unmet(name, results)
                if reason is not None:
                    return f"check {name!r} on it: {reason}"
            blocked = [r for r in results if entry in r.datasets and r.status is Status.ERROR]
            return "it could not be assessed" if blocked else None
        return "no check or dataset has this name"

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
