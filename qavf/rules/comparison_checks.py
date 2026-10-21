"""Comparison rule: two configured datasets agree at the configured levels."""

from __future__ import annotations

from collections import Counter
from typing import ClassVar

from qavf.compare.engine import compare_in_context
from qavf.compare.results import DatasetComparison
from qavf.compare.spec import ComparisonSpec
from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.settings import QAVFSettings

CATEGORY = "comparison"


class ComparisonParams(RuleParams):
    """Parameters for :class:`DatasetComparisonRule`."""

    comparisons: tuple[ComparisonSpec, ...] = ()
    """Pairs of configured datasets and how to compare them."""


def comparison_findings(result: DatasetComparison, spec: ComparisonSpec) -> list[Finding]:
    """One finding per check of every level, labelled ``comparison: level: check``."""
    severities = {config.label: config.severity for config in spec.selected_levels()}
    datasets = tuple(dict.fromkeys((spec.baseline, spec.candidate)))
    findings = []
    for level in result.levels:
        for check in level.checks:
            label = f"{result.name}: {level.name}: {check.check}"
            common = {
                "check": label,
                "datasets": datasets,
                "severity": severities.get(level.name),
                "metrics": dict(check.metrics),
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


class DatasetComparisonRule(Rule):
    """Datasets compare within tolerance at each configured level."""

    rule_id: ClassVar[str] = "CMP-001"
    name: ClassVar[str] = "Datasets compare within tolerance"
    description: ClassVar[str] = (
        "Each configured pair of datasets (for example two versions of an output) agrees "
        "after alignment (column mapping, lossless casts, unit conversion, recoding, "
        "derived columns, filters) at each configured level: structure, key coverage, "
        "keyed records and grouped measures."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Decide whether each difference is intended. Start with structure and coverage "
        "(renamed, missing or new columns and keys), then the largest record and group "
        "changes. Map, convert or recode intended differences in the comparison."
    )
    Params: ClassVar[type[RuleParams]] = ComparisonParams

    def configuration_errors(self, settings: QAVFSettings) -> list[str]:
        problems = []
        for spec in self.params.comparisons:
            for dataset in (spec.baseline, spec.candidate):
                if dataset not in settings.datasets:
                    problems.append(f"comparison {spec.label!r} names unknown dataset {dataset!r}")
        counts = Counter(spec.label for spec in self.params.comparisons)
        problems.extend(
            f"comparison name {label!r} is used {count} times; give each a distinct 'name'"
            for label, count in counts.items()
            if count > 1
        )
        return problems

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset(
                dataset
                for spec in self.params.comparisons
                for dataset in (spec.baseline, spec.candidate)
            )
        )

    def evaluate(self, ctx: DataContext) -> Finding | list[Finding]:
        if not self.params.comparisons:
            return Finding.skip("No comparisons are configured.")
        findings: list[Finding] = []
        for spec in self.params.comparisons:
            findings.extend(comparison_findings(compare_in_context(ctx, spec), spec))
        return findings
