"""Structured immutable audit snapshots produced by the engine and reporters."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    Field,
    computed_field,
    model_validator,
)
from pydantic_core import to_json

from qavf.audit_codec import (
    evidence_json_size as evidence_json_size,
)
from qavf.audit_codec import (
    iter_audit_json,
    iter_json_string,
)
from qavf.audit_codec import (
    normalize_evidence as normalize_evidence,
)
from qavf.engine.severity import Severity, Status, highest_severity, meets_threshold
from qavf.snapshots import SnapshotModel as SnapshotModel
from qavf.snapshots import decode_non_finite as decode_non_finite
from qavf.snapshots import encode_non_finite as encode_non_finite

# Fields with custom audit serializers; all other model fields use ordinary JSON
# scalars and recursively streamed model/container values.
_AUDIT_JSON_FIELDS = frozenset(
    {
        "affected_records",
        "metrics",
        "notes",
        "fingerprint",
        "params",
        "parameters",
        "extensions",
        "config",
        "resource_usage",
        "query_plans",
        "key",
        "baseline",
        "candidate",
        "abs_diff",
        "rel_diff",
        "max_abs_diff",
        "max_rel_diff",
        "leaf",
    }
)


@functools.cache
def _serialized_fields(model: type[SnapshotModel]) -> frozenset[str]:
    """Fields of ``model`` with a declared field serializer, built-in or a plugin's."""
    decorators = model.__pydantic_decorators__.field_serializers.values()
    return frozenset(name for decorator in decorators for name in decorator.info.fields)


@functools.cache
def _excluded_fields(model: type[SnapshotModel]) -> frozenset[str]:
    return frozenset(name for name, field in model.model_fields.items() if field.exclude)


def _iter_snapshot_json(value: Any) -> Iterator[bytes]:
    """Serialize one validated snapshot without copying its complete object graph.

    The output equals Pydantic's ``model_dump_json``. Audit fields stream through
    their bounded encoders; any other field with a declared serializer (including
    a plugin result's) is serialized by Pydantic, for that field alone.
    """
    if isinstance(value, SnapshotModel):
        analysis = value.analyze() if isinstance(value, QAReport) else None
        derived = (
            {"summary": analysis.summary, "acceptance": analysis.acceptance}
            if analysis is not None
            else {}
        )
        model = type(value)
        audited = model.audit_fields
        serialized = _serialized_fields(model)
        excluded = _excluded_fields(model)
        names = [
            name
            for name in (*model.model_fields, *model.model_computed_fields)
            if name not in excluded
        ]
        yield b"{"
        for index, name in enumerate(names):
            if index:
                yield b","
            yield from iter_json_string(name)
            yield b":"
            if name in derived:
                yield from _iter_snapshot_json(derived[name])
            elif name in _AUDIT_JSON_FIELDS or name in audited:
                yield from iter_audit_json(getattr(value, name))
            elif name in serialized:
                yield _json(value.model_dump(mode="json", include={name})[name])
            else:
                yield from _iter_snapshot_json(getattr(value, name))
        yield b"}"
    elif isinstance(value, Enum):
        yield from _iter_snapshot_json(value.value)
    elif isinstance(value, Mapping):
        yield b"{"
        for index, (key, child) in enumerate(value.items()):
            if index:
                yield b","
            yield from iter_json_string(str(key.value if isinstance(key, Enum) else key))
            yield b":"
            yield from _iter_snapshot_json(child)
        yield b"}"
    elif isinstance(value, (list, tuple)):
        yield b"["
        for index, child in enumerate(value):
            if index:
                yield b","
            yield from _iter_snapshot_json(child)
        yield b"]"
    elif isinstance(value, str):
        yield from iter_json_string(value)
    elif value is None or isinstance(value, bool | int | float):
        yield _json(value)
    else:
        # Dates, times, Decimals, UUIDs and other scalars, written as Pydantic writes them.
        yield to_json(value)


def _json(value: Any) -> bytes:
    """Compact JSON; a non-finite float is refused, never written as invalid JSON."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


class CheckIdentity(SnapshotModel):
    """Immutable check contract; display labels and evidence are not identifiers.

    ``check_id`` names the operation. The complete identity includes its rule,
    dataset/column scope and effective parameters; no label parsing is needed.
    Empty columns explicitly denotes a dataset, rule, or run-level check.
    """

    audit_fields: ClassVar[tuple[str, ...]] = ("parameters",)
    rule_id: str
    check_id: str
    scope: str = "rule"
    datasets: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


def check_identity(
    rule_id: str,
    *,
    check_id: str | None = None,
    datasets: Sequence[str] = (),
    columns: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
    scope: str | None = None,
) -> CheckIdentity:
    """Build a scoped contract for engines and third-party rule producers."""
    return CheckIdentity(
        rule_id=rule_id,
        check_id=check_id or rule_id,
        scope=scope
        or (
            "column"
            if len(columns) == 1
            else "columns"
            if columns
            else "dataset"
            if datasets
            else "rule"
        ),
        datasets=list(datasets),
        columns=list(columns),
        parameters=dict(parameters or {}),
    )


def exact_count(status: Status, count: int, declared: bool = True) -> bool:
    """Whether a result's affected count is an exact total.

    A check that was not assessed (ERROR, SKIPPED) has no exact total, and a
    failure without any counted item is never an exact zero.
    """
    assessed = status in (Status.PASS, Status.FAIL)
    return declared and assessed and not (status == Status.FAIL and not count)


def _fails_under(result: RuleResult, limit: Severity, policies: Mapping[str, str]) -> bool:
    """Whether a failed result fails the run: the strictest policy of its tags (its
    own and its family) applies, else ``limit``."""
    applicable = [policies[tag] for tag in {*result.tags, result.category} if tag in policies]
    if not applicable:
        return meets_threshold(result.severity, limit)
    thresholds = [Severity(value) for value in applicable if value != "never"]
    if not thresholds:
        return False
    return meets_threshold(result.severity, min(thresholds))


class GroupRole(SnapshotModel):
    """Which member of a check relating many datasets a result is about.

    Set on the results of a check with more than two datasets, or a strategy other
    than ``reference`` (two datasets with the default strategy report exactly as a
    single comparison does, without a role). Reports roll these up per member
    (:attr:`QAReport.groups`).
    """

    group: str
    """The check item's name."""
    check: str
    """The check's name (``compare``, ``totals_match``, ``key_sets_match``, ``drift``)."""
    strategy: str
    """How the members are paired (``reference``, ``sequential``, ``all_pairs``,
    ``consensus``)."""
    members: list[str] = Field(default_factory=list)
    """The members the result is about (both members of an ``all_pairs`` pair; none
    for a result about the whole group, such as keys without a consensus)."""
    group_members: list[str] = Field(default_factory=list)
    """Every member of the check, in its order."""
    pair: list[str] = Field(default_factory=list)
    """The pair compared, reference first, as reports show it (a member,
    ``consensus`` or ``pooled``); empty for a whole-group level."""
    level: str | None = None
    """The comparison level the result belongs to (``compare``); ``None`` for the
    check as a whole."""
    whole_group: bool = False
    """True for a level assessed over all members at once rather than pair by pair."""


class MemberRow(SnapshotModel):
    """One member's row of the matrix."""

    member: str
    cells: dict[str, Status] = Field(default_factory=dict)
    """The worst status of the member's results at each level (levels without results
    are absent)."""


class PairSummary(SnapshotModel):
    """One pair of members compared (or a member against the consensus or pool)."""

    reference: str
    other: str
    comparison: str
    """The pair's label: its findings' prefix and its comparison's name."""
    status: Status
    """The worst status of the pair's results."""
    results: int
    failed: int
    not_assessed: int
    whole_group_level: str | None = None
    """For a whole-group level run as pairs (``distribution`` against the pool)."""


class MemberRank(SnapshotModel):
    """One member's place in the ranking."""

    member: str
    rank: int
    """1 for the member with the most problems."""
    status: Status
    """The worst status of the member's results."""
    failed: int
    """Failed results about this member."""
    not_assessed: int
    """Results about this member that could not be assessed."""
    changed_records: int
    """Records, rows or keys counted by the member's failed results."""


class GroupReport(SnapshotModel):
    """The roll-up of one check relating many datasets."""

    name: str
    """The check item's name."""
    check: str
    rule_id: str
    strategy: str
    members: list[str]
    levels: list[str]
    """The matrix columns, in the order results report them."""
    whole_group_levels: list[str] = Field(default_factory=list)
    """Levels assessed over every member at once."""
    matrix: list[MemberRow] = Field(default_factory=list)
    pairs: list[PairSummary] = Field(default_factory=list)
    ranking: list[MemberRank] = Field(default_factory=list)
    group_results: list[str] = Field(default_factory=list)
    """Checks about the group as a whole (such as keys without a consensus), with
    their status: ``<check>: <status>``."""


class RuleResult(SnapshotModel):
    """Outcome of one rule execution."""

    audit_fields: ClassVar[tuple[str, ...]] = ("affected_records", "metrics", "parameters")

    rule_id: str
    name: str
    description: str
    category: str
    check: str | None = None
    """Which of a rule's separately graded checks this is (``None`` for single-check rules)."""
    check_id: str = ""
    scope: str = "rule"
    check_name: str | None = None
    """The check's name in configurations, e.g. ``missing_periods`` (``None`` in reports
    written before QAVF 0.7)."""
    item_name: str | None = None
    """The configured check item's name (its default ``<check> on <dataset>`` when the
    configuration gives none); ``None`` for schema contracts."""
    tags: list[str] = Field(default_factory=list)
    """The check item's own tags (its family is :attr:`category`)."""
    group: GroupRole | None = None
    """For a check relating many datasets: which member the result is about (``None``
    otherwise, and in reports written before QAVF 0.7)."""
    columns: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    checks: list[CheckIdentity] = Field(default_factory=list)
    """Complete constituent identities; retained when optional evidence is removed."""
    status: Status
    severity: Severity
    message: str
    affected_records: list[dict[str, Any]] = Field(default_factory=list)
    """Sample of offending records, capped by ``engine.max_affected_records``."""
    affected_count: int = 0
    """Total number of offending items, counted in :attr:`unit` (may exceed
    ``len(affected_records)``, which is a sample)."""
    affected_count_exact: bool = True
    """False when ``affected_count`` is a lower bound or the total is unassessed.
    This decision-critical field is retained independently of optional evidence.
    """
    evidence_complete: bool | None = None
    """Whether all offending items/positions are represented, when known."""
    evidence_truncated: bool = False
    """Optional evidence was sampled, capped, or removed; exact totals are independent."""
    total_records: int | None = None
    """Number of items the rule examined, in the same :attr:`unit`, when meaningful.
    ``None`` when there is no meaningful denominator (e.g. column-level problems)."""
    unit: str | None = None
    """What :attr:`affected_count` and :attr:`total_records` count: ``rows``, ``keys``,
    ``problems``, ``steps``, ... ``None`` when the rule does not say."""
    affected_rows: int | None = None
    """Data rows involved in the problem, when the rule knows it, independently of
    :attr:`unit` (e.g. all rows sharing a duplicated key)."""
    execution_time_s: float = 0.0
    """Wall time of the rule (recorded on its first result), including dataset loads."""
    load_time_s: float = 0.0
    """The part of ``execution_time_s`` spent loading datasets for this rule."""
    recommendation: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    datasets: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def complete_identity(cls, value: Any) -> Any:
        """Older result constructors/JSON gain a truthful rule-level identity."""
        if not isinstance(value, Mapping) or "rule_id" not in value:
            return value
        value = dict(value)
        identity = check_identity(
            value["rule_id"],
            check_id=value.get("check_id"),
            datasets=value.get("datasets", ()),
            columns=value.get("columns", ()),
            parameters=value.get("parameters"),
            scope=value.get("scope"),
        )
        value["check_id"] = identity.check_id
        value["scope"] = identity.scope
        return value

    @property
    def identity(self) -> CheckIdentity:
        """This result's own identity, without duplicating it in persisted JSON."""
        return check_identity(
            self.rule_id,
            check_id=self.check_id,
            scope=self.scope,
            datasets=self.datasets,
            columns=self.columns,
            parameters=self.parameters,
        )

    @property
    def label(self) -> str:
        """Check name, dataset, item name and rule code, plus the separately graded
        check when there is one: ``missing_periods · counts · no missing years
        (TMP-004)``. Reports written before QAVF 0.7 keep ``TMP-004 [check]``."""
        if self.check_name is None:
            return f"{self.rule_id} [{self.check}]" if self.check else self.rule_id
        parts = [self.check_name]
        default_prefix = f"{self.check_name} on "
        if self.datasets:
            parts.append(", ".join(self.datasets))
        elif self.item_name and self.item_name.startswith(default_prefix):
            parts.append(self.item_name.removeprefix(default_prefix).replace(" and ", ", "))
        name = self.item_name
        if name and name != self.check_name and not name.startswith(default_prefix):
            parts.append(name)
        label = f"{' · '.join(parts)} ({self.rule_id})"
        if self.check and self.check != name and not self.check.endswith(f": {name}"):
            label += f" [{self.check}]"
        return label

    @property
    def failed(self) -> bool:
        """True when the rule found a problem or could not run."""
        return self.status in (Status.FAIL, Status.ERROR)

    @property
    def affected_fraction(self) -> float | None:
        """Share of examined items that were flagged (both counted in :attr:`unit`), when
        there is a denominator."""
        if not self.total_records:
            return None
        return self.affected_count / self.total_records


class DatasetInfo(SnapshotModel):
    """Shape of a dataset that took part in a run."""

    audit_fields: ClassVar[tuple[str, ...]] = ("notes", "fingerprint")

    name: str
    rows: int | None
    columns: list[str]
    source: str | None = None
    """Where the data came from (credentials and access tokens redacted)."""
    dtypes: dict[str, str] = Field(default_factory=dict)
    """Column types as loaded, after renaming."""
    renamed: dict[str, str] = Field(default_factory=dict)
    """Columns renamed on load (``original: new``), by ``rename`` or ``canonical_columns``."""
    notes: dict[str, Any] = Field(default_factory=dict)
    """What the load disclosed: reader options, columns loaded, rows rejected, files
    matched."""
    loaded_at: datetime | None = None
    """When the data was read (UTC): in snapshot mode, the start of the shared snapshot."""
    load_time_s: float | None = None
    """Seconds spent loading (and renaming) the dataset; ``None`` for in-memory frames."""
    fingerprint: dict[str, Any] | None = None
    """Identity of the underlying data when it was loaded (for files: size, modification
    time and, with ``report.hash_inputs``, a SHA-256 digest)."""


class RuleInfo(SnapshotModel):
    """A rule that took part in a run, with its effective settings."""

    audit_fields: ClassVar[tuple[str, ...]] = ("params",)

    rule_id: str
    check_name: str | None = None
    """The check's name in configurations (``None`` in reports before QAVF 0.7)."""
    item_name: str | None = None
    """The configured check item this rule instance ran for, if any."""
    severity: Severity
    params: dict[str, Any] = Field(default_factory=dict)
    """Effective parameters, defaults included."""
    module: str
    """Module defining the rule."""
    distribution: str | None = None
    """Installed distribution providing the rule, with its version (``name==version``)."""


class RunMetadata(SnapshotModel):
    """Provenance of a QA run."""

    audit_fields: ClassVar[tuple[str, ...]] = (
        "extensions",
        "config",
        "resource_usage",
        "query_plans",
    )

    run_id: str
    project: str
    qavf_version: str
    started_at: datetime
    finished_at: datetime
    config_path: str | None = None
    datasets: list[DatasetInfo] = Field(default_factory=list)
    parallel: bool = False
    max_workers: int = 1
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Plugin settings in effect (e.g. the MOVES version)."""
    fail_on: Severity = Severity.ERROR
    """Lowest failure severity that fails the run."""
    fail_on_incomplete: bool = True
    """Whether an incomplete run (unassessed checks, or nothing assessed) fails."""
    tag_fail_on: dict[str, str] = Field(default_factory=dict)
    """Per-tag ``fail_on`` (a severity, or ``never``) from ``run.tags``."""
    cancelled: bool = False
    """Whether the caller cancelled the run; steps it did not finish are not assessed."""
    plugins: dict[str, str] = Field(default_factory=dict)
    """Entry-point plugins tried (``rules:name``/``sources:name``): ``loaded`` or the error."""
    excluded_rules: dict[str, str] = Field(default_factory=dict)
    """Discovered rules that were not run, with the reason (disabled, filtered out)."""
    unused_datasets: list[str] = Field(default_factory=list)
    """Datasets with a source that no rule needed, so they were never loaded."""
    unavailable_datasets: dict[str, str] = Field(default_factory=dict)
    """Datasets that could not be loaded, with the reason."""
    config: dict[str, Any] = Field(default_factory=dict)
    """The effective configuration, defaults included, with secrets redacted."""
    config_digest: str | None = None
    """SHA-256 of :attr:`config` in canonical JSON form; equal digests mean equal settings."""
    rules: list[RuleInfo] = Field(default_factory=list)
    """Rules that ran, with effective severities and parameters, and their provenance."""
    plan_digest: str | None = None
    """SHA-256 of the executed plan (rule ids, versions, severities and parameters):
    equal digests mean the same checks were run the same way."""
    resource_usage: dict[str, Any] = Field(default_factory=dict)
    query_plans: list[dict[str, Any]] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    """Python, platform and versions of QAVF and the libraries that compute results."""

    @property
    def duration_s(self) -> float:
        """Wall-clock duration of the run."""
        return (self.finished_at - self.started_at).total_seconds()


class Acceptance(SnapshotModel):
    """The acceptance decision of a run, with the policy that produced it."""

    verdict: Literal["passed", "failed", "incomplete"]
    exit_code: int
    fail_on: Severity
    fail_on_incomplete: bool


class ReportSummary(SnapshotModel):
    """Roll-up counts for a report."""

    total: int
    passed: int
    failed: int
    skipped: int
    errored: int
    failures_by_severity: dict[Severity, int]
    highest_severity: Severity | None
    by_category: dict[str, dict[Status, int]]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def assessed(self) -> int:
        """Results that actually assessed data (passed or failed)."""
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float | None:
        """Passed results over results that actually assessed data."""
        return self.passed / self.assessed if self.assessed else None


class Recommendation(SnapshotModel):
    """Actionable advice derived from a failed rule."""

    rule_id: str
    name: str
    severity: Severity
    text: str
    checks: list[str] = Field(default_factory=list)
    """Failed checks of the rule this advice covers."""


@dataclass(frozen=True)
class ReportAnalysis:
    """A presentation's policy and roll-up, shared without caching mutable drafts.

    The source results already belong to an immutable QAReport. Recommendations
    are computed only if requested, and cached on this analysis, not globally.
    """

    summary: ReportSummary
    acceptance: Acceptance
    results: Sequence[RuleResult]

    @functools.cached_property
    def failure_indices(self) -> tuple[int, ...]:
        """Severity order preserves each occurrence, including repeated objects."""
        return tuple(
            sorted(
                (index for index, result in enumerate(self.results) if result.failed),
                key=self._failure_order,
            )
        )

    def _failure_order(self, index: int) -> tuple[int, str]:
        """Most severe first, then by rule id, for the result at ``index``."""
        result = self.results[index]
        return -result.severity.rank, result.rule_id

    @functools.cached_property
    def failures(self) -> tuple[RuleResult, ...]:
        return tuple(self.results[index] for index in self.failure_indices)

    @functools.cached_property
    def recommendations(self) -> tuple[Recommendation, ...]:
        advice: dict[str, RuleResult] = {}
        checks: dict[str, list[str]] = {}
        for index in self.failure_indices:
            result = self.results[index]
            if not result.recommendation:
                continue
            advice.setdefault(result.rule_id, result)
            if result.check:
                checks.setdefault(result.rule_id, []).append(result.check)
        return tuple(
            Recommendation(
                rule_id=rule_id,
                name=result.name,
                severity=result.severity,
                text=result.recommendation,
                checks=checks.get(rule_id, []),
            )
            for rule_id, result in advice.items()
        )


#: Version of the JSON report layout; incremented when fields change meaning or are removed.
REPORT_FORMAT_VERSION = 5


class QAReport(SnapshotModel):
    """Complete outcome of a QA run."""

    format_version: int = REPORT_FORMAT_VERSION
    """Layout version of this report (:data:`REPORT_FORMAT_VERSION` when written)."""
    metadata: RunMetadata
    results: list[RuleResult]

    @model_validator(mode="after")
    def _supported_version(self) -> Self:
        if self.format_version != REPORT_FORMAT_VERSION:
            raise ValueError(f"unsupported QA report version {self.format_version}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> ReportSummary:
        """Counts by status, severity and category."""
        by_status = {status: 0 for status in Status}
        by_severity = {severity: 0 for severity in Severity}
        by_category: dict[str, dict[Status, int]] = {}
        for result in self.results:
            by_status[result.status] += 1
            by_category.setdefault(result.category, {s: 0 for s in Status})[result.status] += 1
            if result.failed:
                by_severity[result.severity] += 1
        return ReportSummary(
            total=len(self.results),
            passed=by_status[Status.PASS],
            failed=by_status[Status.FAIL],
            skipped=by_status[Status.SKIPPED],
            errored=by_status[Status.ERROR],
            failures_by_severity=by_severity,
            highest_severity=highest_severity(s for s, count in by_severity.items() if count),
            by_category=by_category,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def acceptance(self) -> Acceptance:
        """Verdict and exit code under the run's own policy (``fail_on``,
        ``fail_on_incomplete``), recorded so the JSON is self-describing."""
        return self.analyze().acceptance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def groups(self) -> list[GroupReport]:
        """Each check relating many datasets rolled up per member: a matrix of members
        by levels, the pairs compared and a ranking of the members
        (:class:`~qavf.reports.groups.GroupReport`). Derived from the results, so
        reports written before QAVF 0.7 have none."""
        from qavf.reports.groups import group_reports

        return group_reports(self.results)

    def analyze(self, threshold: Severity | None = None) -> ReportAnalysis:
        """Derive one immutable view under this operation's acceptance policy."""
        effective = self.metadata.fail_on if threshold is None else threshold
        summary = self.summary
        verdict: Literal["passed", "failed", "incomplete"] = (
            "failed"
            if self.should_fail(effective)
            else "incomplete"
            if summary.errored or not summary.assessed
            else "passed"
        )
        incomplete_fails = self.metadata.fail_on_incomplete
        acceptance = Acceptance(
            verdict=verdict,
            exit_code=int(verdict == "failed" or (verdict == "incomplete" and incomplete_fails)),
            fail_on=effective,
            fail_on_incomplete=incomplete_fails,
        )
        return ReportAnalysis(summary, acceptance, self.results)

    def failures(self, min_severity: Severity = Severity.INFO) -> list[RuleResult]:
        """Failed or errored results at or above ``min_severity``, most severe first."""
        hits = [r for r in self.results if r.failed and r.severity >= min_severity]
        return sorted(hits, key=lambda r: (-r.severity.rank, r.rule_id))

    def recommendations(self) -> list[Recommendation]:
        """One piece of advice per failed rule (not per check), most severe first."""
        return list(self.analyze().recommendations)

    def comparisons(self) -> list[Any]:
        """The typed :class:`~qavf.compare.results.DatasetComparison` of each CMP-001
        comparison the report kept (``metrics["comparison"]`` of its first result).

        A comparison too large for the evidence limits keeps only its findings and is
        not listed; one trimmed to fit lists what it left out in ``omitted``.
        """
        from qavf.compare.results import DatasetComparison
        from qavf.reports.comparison import COMPARISON_METRIC

        return [
            DatasetComparison.model_validate(result.metrics[COMPARISON_METRIC])
            for result in self.results
            if result.rule_id == "CMP-001" and COMPARISON_METRIC in result.metrics
        ]

    def results_for(self, rule_id: str) -> list[RuleResult]:
        """Every result (one per check) produced by ``rule_id``."""
        return [result for result in self.results if result.rule_id == rule_id]

    def get(self, rule_id: str, check: str | None = None) -> RuleResult:
        """Return the result for ``rule_id`` (and ``check``, when given).

        For a rule that reported several checks and no ``check`` is given, the
        most severe failure is returned, or the first result if none failed.

        Raises:
            KeyError: if no matching result exists.
        """
        matches = [r for r in self.results_for(rule_id) if check is None or r.check == check]
        if not matches:
            raise KeyError(f"{rule_id}[{check}]" if check else rule_id)
        failing = [r for r in matches if r.failed]
        if failing:
            return max(failing, key=lambda r: r.severity.rank)
        return matches[0]

    def should_fail(self, threshold: Severity | None = None) -> bool:
        """Whether any failure is at least as severe as ``threshold`` (default: ``fail_on``)."""
        limit = threshold or self.metadata.fail_on
        failed = [r for r in self.results if r.status is Status.FAIL]
        policies = self.metadata.tag_fail_on
        if not policies:
            return meets_threshold(highest_severity(r.severity for r in failed), limit)
        return any(_fails_under(result, limit, policies) for result in failed)

    @property
    def incomplete(self) -> bool:
        """Whether some configured check could not run, or nothing was assessed at all."""
        summary = self.summary
        return summary.errored > 0 or summary.assessed == 0

    def verdict(
        self, threshold: Severity | None = None
    ) -> Literal["passed", "failed", "incomplete"]:
        """``failed`` if a failure meets ``threshold``; else ``incomplete`` if checks went
        unassessed (or nothing was assessed); else ``passed``."""
        return self.analyze(threshold).acceptance.verdict

    def exit_code(self, threshold: Severity | None = None) -> int:
        """Process exit code for CI: ``1`` when the run failed, or was incomplete and
        ``fail_on_incomplete`` is set; else ``0``."""
        return self.analyze(threshold).acceptance.exit_code

    def iter_json_chunks(self) -> Iterator[bytes]:
        """The canonical compact UTF-8 report encoding, shared by budgets and writes."""
        buffer = bytearray()
        for chunk in _iter_snapshot_json(self):
            if buffer and len(buffer) + len(chunk) > 32_768:
                yield bytes(buffer)
                buffer.clear()
            buffer.extend(chunk)
        if buffer:
            yield bytes(buffer)

    def json_size_bytes(self) -> int:
        """Exact persisted JSON size without allocating a report-sized string."""
        return sum(len(chunk) for chunk in self.iter_json_chunks())

    def write_json(self, path: str | Path) -> Path:
        """Stream the canonical compact JSON encoding to an atomic destination."""
        from qavf.io import atomic_stream

        with atomic_stream(path) as stream:
            for chunk in self.iter_json_chunks():
                stream.write(chunk)
        return Path(path)
