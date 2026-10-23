"""Structured immutable audit snapshots produced by the engine and reporters."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    Field,
    computed_field,
    model_validator,
)

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


def _iter_snapshot_json(value: Any) -> Iterator[bytes]:
    """Serialize one validated snapshot without copying its complete object graph."""
    if isinstance(value, SnapshotModel):
        yield b"{"
        names = (*type(value).model_fields, *type(value).model_computed_fields)
        audited = type(value).audit_fields
        for index, name in enumerate(names):
            if index:
                yield b","
            yield from iter_json_string(name)
            yield b":"
            child = getattr(value, name)
            if name in _AUDIT_JSON_FIELDS or name in audited:
                yield from iter_audit_json(child)
            else:
                yield from _iter_snapshot_json(child)
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
    elif isinstance(value, datetime):
        from pydantic_core import to_json

        yield to_json(value)
    else:
        yield json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )


class RuleResult(SnapshotModel):
    """Outcome of one rule execution."""

    audit_fields: ClassVar[tuple[str, ...]] = ("affected_records", "metrics")

    rule_id: str
    name: str
    description: str
    category: str
    check: str | None = None
    """Which of a rule's separately graded checks this is (``None`` for single-check rules)."""
    status: Status
    severity: Severity
    message: str
    affected_records: list[dict[str, Any]] = Field(default_factory=list)
    """Sample of offending records, capped by ``engine.max_affected_records``."""
    affected_count: int = 0
    """Total number of offending items, counted in :attr:`unit` (may exceed
    ``len(affected_records)``, which is a sample)."""
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

    @property
    def label(self) -> str:
        """``rule_id`` plus the check, e.g. ``INP-003 [traffic.aadt: ge 0]``."""
        return f"{self.rule_id} [{self.check}]" if self.check else self.rule_id

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
            highest_severity=highest_severity(r.severity for r in self.results if r.failed),
            by_category=by_category,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def acceptance(self) -> Acceptance:
        """Verdict and exit code under the run's own policy (``fail_on``,
        ``fail_on_incomplete``), recorded so the JSON is self-describing."""
        return Acceptance(
            verdict=self.verdict(),
            exit_code=self.exit_code(),
            fail_on=self.metadata.fail_on,
            fail_on_incomplete=self.metadata.fail_on_incomplete,
        )

    def failures(self, min_severity: Severity = Severity.INFO) -> list[RuleResult]:
        """Failed or errored results at or above ``min_severity``, most severe first."""
        hits = [r for r in self.results if r.failed and r.severity >= min_severity]
        return sorted(hits, key=lambda r: (-r.severity.rank, r.rule_id))

    def recommendations(self) -> list[Recommendation]:
        """One piece of advice per failed rule (not per check), most severe first."""
        advice: dict[str, RuleResult] = {}
        checks: dict[str, list[str]] = {}
        for result in self.failures():
            if not result.recommendation:
                continue
            advice.setdefault(result.rule_id, result)
            if result.check:
                checks.setdefault(result.rule_id, []).append(result.check)
        return [
            Recommendation(
                rule_id=rule_id,
                name=result.name,
                severity=result.severity,
                text=result.recommendation,
                checks=checks.get(rule_id, []),
            )
            for rule_id, result in advice.items()
        ]

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
        return meets_threshold(
            highest_severity(r.severity for r in self.results if r.status is Status.FAIL), limit
        )

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
        if self.should_fail(threshold):
            return "failed"
        if self.incomplete:
            return "incomplete"
        return "passed"

    def exit_code(self, threshold: Severity | None = None) -> int:
        """Process exit code for CI: ``1`` when the run failed, or was incomplete and
        ``fail_on_incomplete`` is set; else ``0``."""
        verdict = self.verdict(threshold)
        if verdict == "failed" or (verdict == "incomplete" and self.metadata.fail_on_incomplete):
            return 1
        return 0

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
