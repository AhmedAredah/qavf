"""Comparison rule: two configured datasets agree at the configured levels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from qavf.compare.engine import compare_in_context
from qavf.compare.group_levels import group_levels
from qavf.compare.pairing import GroupPlan
from qavf.compare.registry import entry_point_scope, entry_points_enabled
from qavf.compare.results import DatasetComparison
from qavf.compare.significance import Inference
from qavf.compare.spec import ComparisonSpec, InferenceParams
from qavf.engine.context import DataContext
from qavf.engine.families import CorrectedRule, FamilyCorrection, FamilyTests
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.inference.ledger import FamilyReplayError, RecordingLedger, ReplayingLedger
from qavf.inference.results import Correction
from qavf.rules._groups import GroupedFindings
from qavf.settings import QAVFSettings

CATEGORY = "comparison"


class ComparisonParams(RuleParams):
    """Parameters for :class:`DatasetComparisonRule`."""

    comparisons: tuple[ComparisonSpec, ...] = ()
    """Pairs of configured datasets and how to compare them."""
    group: GroupPlan | None = None
    """For a check relating many datasets: which member each comparison is about, the
    consensus and the levels assessed over the whole group."""


def comparison_findings(
    result: DatasetComparison, spec: ComparisonSpec, byte_limit: int | None = None
) -> list[Finding]:
    """One finding per check of every level, labelled ``comparison: level: check``.

    With ``byte_limit`` (the run's per-result evidence bytes), the first finding also
    keeps the typed comparison in ``metrics["comparison"]`` for the report's
    comparison section, within what that result may hold (detail left out is
    disclosed in its ``omitted``); a comparison that cannot fit is left out.
    """
    findings = _findings(result, spec)
    if byte_limit is None or not findings:
        return findings
    from qavf.audit_codec import evidence_json_size
    from qavf.reports.comparison import COMPARISON_METRIC

    first = findings[0]
    records = first.affected if isinstance(first.affected, list) else []
    used = evidence_json_size(first.metrics) + (evidence_json_size(records) if records else 0)
    allowance = byte_limit - used - 4_096
    if allowance <= 0:
        return findings
    bounded = result.bounded(allowance)
    payload = bounded.model_dump()
    if evidence_json_size(payload) > allowance:
        return findings
    first.metrics[COMPARISON_METRIC] = payload
    return findings


def _findings(result: DatasetComparison, spec: ComparisonSpec) -> list[Finding]:
    severities = {config.label: config.severity for config in spec.selected_levels()}
    datasets = tuple(dict.fromkeys((spec.baseline, spec.candidate)))
    findings = []
    for level in result.levels:
        for check in level.checks:
            label = f"{result.name}: {level.name}: {check.check}"
            metrics = dict(check.metrics)
            identity: dict[str, Any] = {
                "name": result.name,
                "level": level.name,
                "check": check.check,
            }
            # Labels remain human-readable; structured identity survives names
            # containing the separator and preserves any same-key plugin metric.
            marker = "_qavf_comparison_identity"
            if marker in metrics:
                identity["metric"] = metrics[marker]
            metrics[marker] = identity
            common = {
                "check": label,
                "datasets": datasets,
                "severity": severities.get(level.name),
                "metrics": metrics,
            }
            if not check.assessed:
                findings.append(Finding.unassessed(f"{label}: {check.message}", **common))
            elif check.passed:
                findings.append(
                    Finding.ok(
                        f"{label}: {check.message}",
                        total_records=check.total,
                        unit=check.unit,
                        affected=[dict(record) for record in check.affected_records] or None,
                        affected_count=check.affected_count,
                        **common,
                    )
                )
            else:
                findings.append(
                    Finding.fail(
                        f"{label}: {check.message}",
                        affected=[dict(record) for record in check.affected_records],
                        affected_count=check.affected_count,
                        total_records=check.total,
                        unit=check.unit,
                        **common,
                    )
                )
    return findings


class _ComparisonTests(FamilyTests):
    """The corrections a comparison made while measured, replayed when it is judged.

    A comparison corrects many small families inside its levels; their raw
    p-values join the inference family in the order they were corrected. Judging
    evaluates the comparison again with each correction handed its share of the
    family's adjusted p-values (comparisons are deterministic).
    """

    def __init__(self, rule: DatasetComparisonRule, ledger: RecordingLedger) -> None:
        self._rule = rule
        self._ledger = ledger

    @property
    def p_values(self) -> np.ndarray:
        return self._ledger.p_values

    @property
    def family_size(self) -> int:
        return self._ledger.family_size

    @property
    def methods(self) -> frozenset[str]:
        return self._ledger.methods

    def finish(self, ctx: DataContext, correction: FamilyCorrection) -> list[Finding]:
        replay = ReplayingLedger(
            self._ledger.recorded, correction.adjusted, correction.method, correction.family_size
        )
        with replay.active():
            findings = self._rule.compared(ctx)
        if not replay.complete():
            raise FamilyReplayError(
                "the comparison corrected fewer families when judged than when measured; "
                "its inputs must not change during the run"
            )
        return findings


class DatasetComparisonRule(CorrectedRule):
    """Datasets compare within tolerance at each configured level."""

    rule_id: ClassVar[str] = "CMP-001"
    check_name: ClassVar[str] = "compare"
    name: ClassVar[str] = "Datasets compare within tolerance"
    description: ClassVar[str] = (
        "Each configured pair of datasets (for example two versions of an output) agrees "
        "after alignment (column mapping, lossless casts, unit conversion, recoding, "
        "derived columns, filters) at each configured level: structure, key coverage, "
        "keyed records, grouped measures, time, space, distributions, relationships and "
        "contributions to the change."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Decide whether each difference is intended. Start with structure and coverage "
        "(renamed, missing or new columns and keys), then the largest record and group "
        "changes. Map, convert or recode intended differences in the comparison."
    )
    Params: ClassVar[type[RuleParams]] = ComparisonParams

    def __init__(
        self,
        params: ComparisonParams | Mapping[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> None:
        super().__init__(params, severity)
        # Rules evaluate on worker threads, outside the compile-time context:
        # keep the run's plugin choice to resolve the same components there.
        self._installed_plugins = entry_points_enabled()

    def configuration_errors(self, settings: QAVFSettings) -> list[str]:
        problems = []
        for spec in self.params.comparisons:
            for dataset in spec.datasets():
                if not settings.knows_dataset(dataset):
                    problems.append(f"comparison {spec.label!r} names unknown dataset {dataset!r}")
        counts = Counter(spec.label for spec in self.params.comparisons)
        problems.extend(
            f"comparison name {label!r} is used {count} times; give each a distinct 'name'"
            for label, count in counts.items()
            if count > 1
        )
        return problems

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        datasets = {dataset for spec in self.params.comparisons for dataset in spec.datasets()}
        plan = self.params.group
        if plan is not None:
            datasets |= GroupedFindings(plan).requirements()
            if group_levels(plan):
                datasets |= set(plan.members)
        return DataRequirements(frozenset(datasets))

    def family_corrections(self) -> frozenset[str]:
        """The corrections of the levels that test significance (``decision`` other
        than ``material``), each level's own or its comparison's."""
        corrections: set[str] = set()
        from qavf.compare.levels import LEVELS

        for spec in self.params.comparisons:
            for level in spec.selected_levels():
                params = LEVELS.create(level.level, level.params).params
                if not isinstance(params, InferenceParams):
                    continue
                inference = Inference.of(spec, params)
                if inference.tests:
                    corrections.add(inference.correction)
        return frozenset(corrections)

    def family_method(self) -> Correction:
        methods = sorted(self.family_corrections())
        method: Correction = methods[0] if len(methods) == 1 else "none"  # type: ignore[assignment]
        return method

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        """Each level's families corrected where they are judged."""
        return self.compared(ctx)

    def measure(self, ctx: DataContext) -> FamilyTests | list[Finding]:
        """Evaluate once, recording every correction's raw p-values; a comparison that
        corrected nothing is final."""
        ledger = RecordingLedger()
        with ledger.active():
            findings = self.compared(ctx)
        if not ledger.recorded:
            return findings
        return _ComparisonTests(self, ledger)

    def compared(self, ctx: DataContext) -> list[Finding]:
        """Every comparison's findings (and the group's)."""
        plan = self.params.group
        if not self.params.comparisons and plan is None:
            return [Finding.skip("No comparisons are configured.")]
        findings: list[Finding] = []
        if plan is not None:
            for level in group_levels(plan):
                findings.extend(level.evaluate(ctx))
        with entry_point_scope(self._installed_plugins):
            for spec in self.params.comparisons:
                found = comparison_findings(compare_in_context(ctx, spec), spec, ctx.evidence_bytes)
                findings.extend(found)
        if plan is not None:
            return GroupedFindings(plan).finish(ctx, findings)
        return findings
