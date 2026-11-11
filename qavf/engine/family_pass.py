"""The run's inference families: measured with every other rule, judged together after.

A rule whose check item names an inference family (``inference: {family: ...}``)
is measured when the runner reaches it: its raw tests are kept (with the datasets
they read, which are not released yet) and it reports nothing. Once every rule
has run, :meth:`FamilyStage.correct` corrects each family's p-values together,
and the runner invokes each waiting member again to judge its tests with its
share of the correction (:meth:`FamilyStage.outcome`). Every member's results
then carry the family (:class:`~qavf.reports.models.InferenceFamily`).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from qavf.checks.families import FamilyPlan, FamilyPlans
from qavf.engine.context import DataContext
from qavf.engine.families import CorrectedRule, FamilyCorrection, FamilyTests, PooledFamily
from qavf.engine.registry import Finding, Rule
from qavf.reports.models import InferenceFamily, RuleResult
from qavf.settings import QAVFSettings


@dataclass(slots=True)
class _Member:
    """One member rule of a family and what the run knows about it so far."""

    rule: CorrectedRule
    family: str
    tests: FamilyTests | None = None
    correction: FamilyCorrection | None = None
    problem: str | None = None
    """Why the member's tests cannot be judged with its family."""
    finished: bool = False
    seconds: tuple[float, float] = (0.0, 0.0)
    """Execution and load time of the measuring invocation."""
    notes: list[str] = field(default_factory=list)

    @property
    def waiting(self) -> bool:
        """Measured, not yet judged: its datasets stay loaded."""
        return self.tests is not None and not self.finished


class FamilyStage:
    """Measures the members of the run's inference families, corrects each family
    over all of its members' p-values, then judges every member."""

    def __init__(self, rules: Sequence[Rule], settings: QAVFSettings) -> None:
        self._plans = FamilyPlans(settings).plans
        self._members: dict[str, _Member] = {}
        for rule in rules:
            family = rule.binding.inference_family if rule.binding is not None else None
            if family is not None and isinstance(rule, CorrectedRule):
                self._members[rule.key] = _Member(rule, family)
        self._pooled: dict[str, PooledFamily] = {}
        self._limit = settings.engine.max_affected_records
        self._lock = threading.Lock()

    @classmethod
    def of(cls, rules: Sequence[Rule], settings: QAVFSettings) -> FamilyStage | None:
        """The stage of ``rules``; ``None`` when no rule belongs to a family."""
        stage = cls(rules, settings)
        return stage if stage._members else None

    def outcome(self, rule: Rule, ctx: DataContext) -> Finding | Sequence[Finding] | None:
        """What ``rule`` reports now: its findings, or ``None`` while its measured tests
        wait for their family's correction."""
        member = self._members.get(rule.key)
        if member is None:
            return rule.evaluate(ctx)
        with self._lock:
            tests, correction, problem = member.tests, member.correction, member.problem
        if tests is not None:
            with self._lock:
                member.finished = True
            with tests:
                if problem is not None:
                    raise ValueError(problem)
                if correction is None:
                    raise RuntimeError("the inference family was not corrected")
                return tests.finish(ctx, correction)
        measured = member.rule.measure(ctx)
        if not isinstance(measured, FamilyTests):
            return measured
        with self._lock:
            member.tests = measured
        return None

    def holds(self, rule: Rule) -> bool:
        """Whether ``rule``'s datasets must stay loaded until its family is judged."""
        member = self._members.get(rule.key)
        with self._lock:
            return member is not None and member.waiting

    def timed(self, rule: Rule, seconds: float, load: float) -> tuple[float, float]:
        """The time to record for ``rule``: a measured member's is kept until it is
        judged, then reported with the judging time."""
        member = self._members.get(rule.key)
        if member is None:
            return seconds, load
        with self._lock:
            if member.waiting:
                member.seconds = (seconds, load)
                return seconds, load
            measured, loaded = member.seconds
            return seconds + measured, load + loaded

    def waiting(self, rules: Sequence[Rule]) -> list[int]:
        """Positions in ``rules`` of the members waiting to be judged."""
        return [
            index
            for index, rule in enumerate(rules)
            if (member := self._members.get(rule.key)) is not None and member.waiting
        ]

    def correct(self) -> None:
        """Correct each family over the p-values of its measured members."""
        families: dict[str, list[_Member]] = {}
        for member in self._members.values():
            families.setdefault(member.family, []).append(member)
        for name, members in families.items():
            method = members[0].rule.family_method()
            measured: list[tuple[str, FamilyTests]] = []
            for member in members:
                tests = member.tests
                if tests is None:
                    continue
                applied = sorted(tests.methods - {method})
                if applied:
                    member.problem = (
                        f"inference family {name!r} is corrected with {method}, but this "
                        f"item's tests applied {', '.join(applied)}"
                    )
                    continue
                measured.append((member.rule.key, tests))
            pooled, corrections = PooledFamily.correct(name, method, measured)
            self._pooled[name] = pooled
            for member in members:
                member.correction = corrections.get(member.rule.key)

    def annotate(self, rule: Rule, results: list[RuleResult]) -> list[RuleResult]:
        """``results`` of ``rule`` with the family its tests were corrected with."""
        member = self._members.get(rule.key)
        if member is None:
            return results
        family = self._family(member)
        return [result.model_copy(update={"inference": family}) for result in results]

    def close(self) -> None:
        """Release the tests of members that were never judged (a cancelled run)."""
        for member in self._members.values():
            if member.tests is not None:
                member.tests.close()

    def _family(self, member: _Member) -> InferenceFamily:
        pooled = self._pooled.get(member.family)
        plan = self._plans.get(member.family)
        names = [m.rule.key for m in self._members.values() if m.family == member.family]
        corrected = list(pooled.members) if pooled is not None else []
        correction = member.correction
        adjusted = [] if correction is None else correction.adjusted[: self._limit].tolist()
        return InferenceFamily(
            family=member.family,
            method=pooled.method if pooled is not None else member.rule.family_method(),
            family_size=pooled.family_size if pooled is not None else 0,
            members=corrected,
            configured_members=len(plan.configured) if plan is not None else len(names),
            tests=pooled.tests.get(member.rule.key, 0) if pooled is not None else 0,
            adjusted=[None if math.isnan(value) else value for value in adjusted],
            notes=self._notes(plan, names, corrected, pooled),
        )

    @staticmethod
    def _notes(
        plan: FamilyPlan | None,
        names: list[str],
        corrected: list[str],
        pooled: PooledFamily | None,
    ) -> list[str]:
        notes = []
        if plan is not None and plan.partial:
            notes.append(f"{plan.selected()} (not run: {', '.join(plan.left_out)})")
        left_out = [name for name in names if name not in corrected]
        if left_out:
            notes.append(f"not corrected with the family (not measured): {', '.join(left_out)}")
        if pooled is not None and pooled.non_finite:
            notes.append(f"{pooled.non_finite} non-finite p-value(s) left out of the family")
        return notes


__all__ = ["FamilyStage"]
