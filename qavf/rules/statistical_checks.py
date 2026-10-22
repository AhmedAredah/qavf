"""Generic statistical QA/QC: configurable outlier scans over any numeric column."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import polars as pl
from pydantic import Field

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.inference._support import scipy_stats
from qavf.inference.multiple import adjust
from qavf.inference.results import Correction
from qavf.rules._contracts import isolate_contract
from qavf.rules.helpers import (
    MetricSpec,
    annotate,
    normalize_missing,
    plural,
)
from qavf.settings import QAVFSettings
from qavf.stats import (
    DEFAULT_THRESHOLDS,
    OutlierMethod,
    minimum_samples,
)

if TYPE_CHECKING:
    from qavf.engine.observation import Observation
    from qavf.engine.observation_plan import ObservationView

CATEGORY = "statistical"


class OutlierTarget(MetricSpec):
    """A column to scan row-by-row for outliers within each ``group_by`` group."""

    method: OutlierMethod | None = None
    """Overrides the rule-level method for this target."""
    threshold: float | None = Field(default=None, gt=0)
    """Overrides the rule-level threshold for this target."""
    direction: Literal["both", "high", "low"] = "both"
    """Flag only unusually high (or low) values, e.g. spikes in a count."""
    min_relative_deviation: float = Field(default=0.0, ge=0)
    """Also require the value to differ from its group's median by at least this
    fraction (0.1 = 10 %), so statistically extreme but immaterial values pass."""
    transform: Literal["none", "log"] = "none"
    """Scan ``log(value)``: for skewed, multiplicative data (concentrations, flows).
    Non-positive values cannot be transformed and are reported as not tested."""


class ColumnOutlierParams(RuleParams):
    """Parameters for :class:`ColumnOutlierRule`."""

    targets: tuple[OutlierTarget, ...] = ()
    """Columns to scan; the rule skips until at least one is configured."""
    method: OutlierMethod = OutlierMethod.IQR
    threshold: float | None = Field(default=None, gt=0)
    """Method threshold; the conventional default for the method when omitted
    (3.0 for z-score, 3.5 for modified z-score, 1.5 for IQR)."""
    min_group_size: int = Field(default=8, ge=3)
    """Groups with fewer non-null values are not scanned."""


def _needed(method: OutlierMethod, threshold: float | None, configured: int) -> str:
    """The group size a test needs, explaining when the method itself requires more."""
    limit = DEFAULT_THRESHOLDS[method] if threshold is None else threshold
    required = minimum_samples(method, limit)
    if required <= configured:
        return str(configured)
    return (
        f"{required} (a z-score with threshold {limit:g} cannot flag any value in fewer: "
        f"the largest possible |z| in n values is (n-1)/sqrt(n))"
    )


class ColumnOutlierRule(Rule):
    """Individual values are not statistical outliers within their group."""

    rule_id: ClassVar[str] = "STA-001"
    name: ClassVar[str] = "Statistical outliers in values"
    description: ClassVar[str] = (
        "Within each configured group, no individual value is an outlier by the configured "
        "method (z-score, modified z-score or IQR) and threshold."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "Outliers are prompts for review, not proof of error. Inspect the flagged rows; if "
        "they are legitimate, group more finely or raise the threshold for that target so "
        "the check stays informative."
    )
    Params: ClassVar[type[RuleParams]] = ColumnOutlierParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(frozenset(target.dataset for target in self.params.targets))

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        p = self.params
        if not p.targets:
            return [Finding.skip("No outlier targets are configured.")]
        records: list[dict[str, Any]] = []
        blocked: list[Finding] = []
        used: list[OutlierTarget] = []
        examined = 0
        outliers = 0
        from qavf.rules._statistics_plan import outlier_targets

        targets = outlier_targets(ctx, p.targets, blocked)
        requests = [{t.dataset: [*t.group_by, t.value]} for t in targets]
        with ctx.observation_plan(requests) as plan:
            for index, target in enumerate(targets):
                with (
                    isolate_contract(target.label, (target.dataset,), blocked),
                    plan.contract(index) as observations,
                ):
                    method = target.method or p.method
                    from qavf.rules._outlier_batches import scan_target

                    try:
                        scan = scan_target(
                            ctx,
                            target,
                            method,
                            target.threshold or p.threshold,
                            p.min_group_size,
                            max(0, ctx.evidence_limit - len(records)),
                            normalize_missing,
                            observations[target.dataset],
                        )
                    except ValueError as exc:
                        blocked.append(
                            Finding.unassessed(
                                f"Not assessed: {target.label}: {exc}.",
                                check=target.label,
                                datasets=(target.dataset,),
                            )
                        )
                        continue
                    if scan.invalid:
                        blocked.append(
                            Finding.unassessed(
                                f"Not assessed: {scan.invalid} missing or non-finite values "
                                f"of {target.label}.",
                                check=f"{target.label}: invalid observations",
                                affected=pl.DataFrame(scan.invalid_records)
                                if scan.invalid_records
                                else [],
                                affected_count=scan.invalid,
                                unit="values",
                                datasets=(target.dataset,),
                            )
                        )
                    if scan.nonpositive:
                        blocked.append(
                            Finding.unassessed(
                                f"Not assessed: {plural(scan.nonpositive, 'value')} "
                                f"of {target.label} "
                                "are zero or negative and cannot be log-transformed; "
                                "the other values were tested.",
                                check=f"{target.label}: log transform",
                                affected_count=scan.nonpositive,
                                unit="values",
                                datasets=(target.dataset,),
                            )
                        )
                    if scan.tested and scan.untested:
                        blocked.append(
                            Finding.unassessed(
                                f"Not assessed: {scan.untested:,} values of "
                                f"{target.label} belong to "
                                "groups too small or numerically untestable.",
                                check=f"{target.label}: incomplete assessment",
                                affected_count=scan.untested,
                                affected=scan.untested_records,
                                unit="values",
                                datasets=(target.dataset,),
                            )
                        )
                    if not scan.tested:
                        needed = _needed(method, target.threshold or p.threshold, p.min_group_size)
                        blocked.append(
                            Finding.unassessed(
                                f"Not assessed: target {target.label} has no group with at least "
                                f"{needed} "
                                "finite values to test.",
                                check=target.label,
                                total_records=0,
                                unit="values",
                                datasets=(target.dataset,),
                            )
                        )
                        continue
                    used.append(target)
                    examined += scan.tested
                    outliers += scan.outliers
                    records.extend(
                        annotate(row, target=target.label, method=method.value)
                        for row in scan.records
                    )
        if not used:
            return blocked
        datasets = tuple(dict.fromkeys(t.dataset for t in used))
        metrics = {"targets": [t.label for t in used], "examined": examined}
        if outliers:
            return [
                *blocked,
                Finding.fail(
                    f"{plural(outliers, 'outlier')} among {examined:,} values across "
                    f"{plural(len(used), 'target')}.",
                    affected=records,
                    affected_count=outliers,
                    total_records=examined,
                    unit="values",
                    metrics=metrics,
                    datasets=datasets,
                ),
            ]
        return [
            *blocked,
            Finding.ok(
                f"No outliers among {examined:,} values across {plural(len(used), 'target')}.",
                total_records=examined,
                unit="values",
                metrics=metrics,
                datasets=datasets,
            ),
        ]


# ----------------------------------------------------------------- STA-002
class DriftComparison(RuleParams):
    """A column whose distribution must stay close to that of a reference dataset."""

    name: str | None = None
    dataset: str
    reference: str
    """The reference dataset (e.g. last accepted run), with the same column."""
    column: str
    method: Literal["psi", "ks"] = "psi"
    """``psi``: population stability index over the reference's quantile bins (numbers)
    or categories (text). ``ks``: two-sample Kolmogorov-Smirnov test (numbers; needs the
    stats extra)."""
    bins: int = Field(default=10, ge=2, le=100)
    max_psi: float = Field(default=0.2, gt=0)
    """PSI above which the distribution has drifted (0.1 small, 0.2 notable, 0.25 large)."""
    alpha: float = Field(default=0.01, gt=0, lt=1)
    """KS significance level before the multiple-testing correction."""
    min_effect: float = Field(default=0.1, ge=0, le=1)
    """KS statistic below which a significant difference is still immaterial (large
    samples make tiny differences significant)."""
    min_samples: int = Field(default=50, ge=2)
    """Fewest usable values on each side for a comparison to be judged."""

    def label(self) -> str:
        return self.name or f"{self.dataset}.{self.column} vs {self.reference}"


class DriftParams(RuleParams):
    """Parameters for :class:`DistributionDriftRule`."""

    comparisons: tuple[DriftComparison, ...] = ()
    correction: Correction = "bonferroni"
    """Multiple-testing correction across the KS comparisons (``holm``, ``bh``,
    ``by``, ``none``); a comparison drifts when its adjusted p-value is below its
    ``alpha``."""


@dataclass(slots=True)
class _PendingKS:
    """A KS comparison measured, awaiting the family's correction."""

    comparison: DriftComparison
    statistic: float
    p_value: float
    metrics: dict[str, Any]


class DistributionDriftRule(Rule):
    """A column's distribution has not drifted from its reference."""

    rule_id: ClassVar[str] = "STA-002"
    name: ClassVar[str] = "Distribution drift"
    description: ClassVar[str] = (
        "Each configured column's distribution stays close to a reference dataset's, by "
        "population stability index or a two-sample KS test (with a minimum sample size, "
        "an effect-size floor and a Bonferroni correction across KS comparisons)."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "Drift is not an error by itself: the population may really have changed. Check "
        "what changed upstream (inputs, coverage, units) before accepting the new run."
    )
    Params: ClassVar[type[RuleParams]] = DriftParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset(
                name
                for comparison in self.params.comparisons
                for name in (comparison.dataset, comparison.reference)
            )
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        comparisons = self.params.comparisons
        tests = max(sum(c.method == "ks" for c in comparisons), 1)
        findings: list[Finding] = []
        measured: dict[int, _PendingKS] = {}
        """KS comparisons awaiting the correction, by their place in ``findings``."""
        from qavf.rules._statistics_plan import drift_comparisons

        admitted = drift_comparisons(ctx, comparisons, findings)
        requests = [{name: [c.column] for name in (c.dataset, c.reference)} for c in admitted]
        with ctx.observation_plan(requests) as plan:
            for index, comparison in enumerate(admitted):
                with (
                    isolate_contract(
                        comparison.label(), (comparison.dataset, comparison.reference), findings
                    ),
                    plan.contract(index) as observations,
                ):
                    finding = self._compare(ctx, comparison, tests, observations)
                    excluded = finding.metrics.get("reference_excluded", 0) + finding.metrics.get(
                        "excluded", 0
                    )
                    if excluded:
                        findings.append(
                            Finding.unassessed(
                                f"Not assessed: {excluded} missing or non-finite observations "
                                "were excluded "
                                f"from {comparison.label()}.",
                                check=f"{comparison.label()}: incomplete assessment",
                                affected_count=excluded,
                                unit="values",
                                datasets=(comparison.dataset, comparison.reference),
                            )
                        )
                    if isinstance(finding, _PendingKS):
                        measured[len(findings)] = finding
                        findings.append(Finding.skip(comparison.label()))  # replaced below
                    else:
                        findings.append(finding)
        # Untested KS comparisons count as p = 1: Bonferroni divides by every one.
        pending = [measured[i].p_value for i in measured]
        adjusted = adjust(pending, self.params.correction, family_size=tests)
        for i, q in zip(measured, adjusted, strict=True):
            findings[i] = self._ks_finding(measured[i], float(q), tests)
        return findings or [Finding.skip("No drift comparisons are configured.")]

    def _ks_finding(self, measured: _PendingKS, adjusted: float, tests: int) -> Finding:
        """Judge one KS comparison once the family is corrected."""
        c, correction = measured.comparison, self.params.correction
        metrics = dict(measured.metrics)
        statistic, probability = measured.statistic, measured.p_value
        if correction == "bonferroni":
            alpha = c.alpha / tests
            metrics |= {"alpha_bonferroni": alpha}
            significant = probability < alpha
            detail = f"KS statistic {statistic:.3f}, p = {probability:.2g} < {alpha:.2g}"
        else:
            significant = adjusted < c.alpha
            detail = (
                f"KS statistic {statistic:.3f}, adjusted p = {adjusted:.2g} < {c.alpha:.2g} "
                f"({correction})"
            )
        metrics |= {"adjusted_p": adjusted, "correction": correction, "effect": statistic}
        return self._judged(c, metrics, significant and statistic >= c.min_effect, detail)

    def _compare(
        self,
        ctx: DataContext,
        c: DriftComparison,
        tests: int,
        observations: Mapping[str, Observation | ObservationView],
    ) -> Finding | _PendingKS:
        from qavf.rules._distribution import observed_distribution

        label = c.label()
        try:
            with observed_distribution(
                ctx, c.reference, c.dataset, c.column, normalize_missing, observations
            ) as distribution:
                if min(distribution.sizes) < c.min_samples:
                    return Finding.unassessed(
                        f"Not assessed: {label}: {min(distribution.sizes)} usable values, "
                        "fewer than "
                        f"min_samples {c.min_samples}.",
                        check=label,
                    )
                metrics = distribution.metrics()
                if c.method == "ks":
                    if not distribution.numeric:
                        return Finding.unassessed(
                            f"Not assessed: {label}: the KS test needs numeric values; "
                            "use method: psi.",
                            check=label,
                        )
                    if scipy_stats() is None:
                        return Finding.unassessed(
                            f"Not assessed: {label}: the KS test needs the stats extra "
                            "(pip install 'qavf[stats]').",
                            check=label,
                        )
                    statistic, probability, policy = distribution.ks()
                    metrics |= {
                        "ks_statistic": statistic,
                        "p_value": probability,
                        "ks_policy": policy,
                    }
                    return _PendingKS(c, statistic, probability, metrics)
                else:
                    psi, bins, policy = distribution.psi(c.bins)
                    metrics |= {"psi": round(psi, 6), "bins": bins, "bin_policy": policy}
                    drifted = psi > c.max_psi
                    detail = f"PSI {psi:.3f} > {c.max_psi:g}"
        except ValueError as exc:
            return Finding.unassessed(
                f"Not assessed: {label}: {exc}.",
                check=label,
                datasets=(c.dataset, c.reference),
            )
        return self._judged(c, metrics, drifted, detail)

    @staticmethod
    def _judged(c: DriftComparison, metrics: dict[str, Any], drifted: bool, detail: str) -> Finding:
        label = c.label()
        if not drifted:
            return Finding.ok(
                f"{label}: distribution stable.",
                check=label,
                metrics=metrics,
                datasets=(c.dataset, c.reference),
                unit="columns",
                total_records=1,
            )
        return Finding.fail(
            f"{label}: the distribution has drifted ({detail}).",
            check=label,
            affected=[
                {"dataset": c.dataset, "reference": c.reference, "column": c.column, **metrics}
            ],
            unit="columns",
            metrics=metrics,
            datasets=(c.dataset, c.reference),
        )
