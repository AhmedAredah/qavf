"""Bounded evidence and the report results of the native input checks.

Results take the same shape as the full backend's: the rule's metadata and
wording from :mod:`qavf.input_rules`, one result per failing contract, and one
summary per rule that accounts for every contract that passed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from qavf import input_rules as wording
from qavf.audit_codec import (
    EVIDENCE_OMITTED,
    bounded_diagnostic,
    evidence_json_size,
    normalize_evidence,
)
from qavf.engine.severity import Severity, Status
from qavf.input_rules import INPUT_RULES
from qavf.redaction import redact, redact_text
from qavf.reports.models import RuleResult, check_identity, exact_count


@dataclass(frozen=True, slots=True)
class Summary:
    """How a rule folds the contracts that passed into one result."""

    check_id: str
    message: Any
    """Callable building the message from the number of passing contracts."""
    single_identity: bool = False
    """Whether one passing identity names the summary itself (INP-004)."""


SUMMARIES: dict[str, Summary] = {
    "INP-001": Summary("dataset.available.summary", wording.datasets_accessible),
    "INP-002": Summary("schema.summary", wording.schemas_conform),
    "INP-003": Summary("column.summary", wording.column_checks_passed),
    "INP-004": Summary("key.summary", wording.key_contracts_passed, single_identity=True),
    "INP-005": Summary("key.reference.summary", wording.foreign_keys_resolved),
    "INP-006": Summary("row.expression.summary", wording.expressions_hold),
    "INP-007": Summary("sql.assertion.summary", wording.assertions_hold),
}
"""Rules whose passing contracts are folded into a summary; others report each one."""


@dataclass(order=True)
class _Entry:
    """A passing contract or exemption, with its place in the canonical report order."""

    order: tuple[int, int]
    rule_id: str = field(compare=False)
    kind: str = field(compare=False)
    label: str = field(compare=False, default="")
    identities: tuple[dict[str, Any], ...] = field(compare=False, default=())
    name: str = field(compare=False, default="")
    rows: int = field(compare=False, default=0)


class ResultCollector:
    """Collect native results in the full backend's shape and order."""

    def __init__(self, connection, settings):
        self.connection = connection
        self.settings = settings
        self.results = []
        self.locators = {}
        self._orders = []
        self._entries = []
        self._elapsed = {}
        self._sequence = itertools.count()
        self._positions = {name: index for index, name in enumerate(settings.datasets)}

    def enabled(self, rule_id, dataset=None):
        """Whether ``rule_id`` reports: schema contracts always run, and the
        ``expression``/``sql`` rules run only for configured check items."""
        if rule_id == "INP-006":
            specs = (
                [self.settings.datasets[dataset]]
                if dataset in self.settings.datasets
                else self.settings.datasets.values()
            )
            return any(spec.expressions for spec in specs)
        if rule_id == "INP-007":
            return bool(self.settings.assertions)
        return rule_id != "INP-008"  # freshness checks are not native

    def records(self, query):
        from qavf.core.evidence import SqlEvidenceReader

        return SqlEvidenceReader(
            self.connection,
            max_records=self.settings.engine.max_affected_records,
            max_bytes=self.settings.engine.max_evidence_bytes,
        ).read(query)

    @property
    def evidence_limit(self):
        return self.settings.engine.max_affected_records

    def order(self, owner):
        """A dataset's (or assertion's) place in the report; unowned results first."""
        position = owner if isinstance(owner, int) else self._positions.get(owner, -1)
        return position, next(self._sequence)

    def rule_severity(self, rule_id):
        return INPUT_RULES[rule_id].default_severity

    @staticmethod
    def item_name(check_id, check, datasets, parameters):
        """The ``expression``/``sql`` check item a result reports on, as the full backend."""
        if check_id == "row.expression" and datasets:
            return (parameters or {}).get("name") or f"expression on {datasets[0]}"
        if check_id == "sql.assertion":
            return check
        return None

    def report(
        self,
        rule_id,
        owner,
        status,
        message,
        *,
        check=None,
        check_id=None,
        datasets=(),
        columns=(),
        parameters=None,
        checks=(),
        records=None,
        count=None,
        count_exact=True,
        total=None,
        unit=None,
        affected_rows=None,
        metrics=None,
        severity=None,
    ):
        """Record one result. ``records`` is ``None`` when no evidence was collected."""
        selected = owner if isinstance(owner, str) and owner in self._positions else None
        if not self.enabled(rule_id, selected):
            return
        status = Status(status)
        severity = Severity(severity) if severity else self.rule_severity(rule_id)
        if status == Status.ERROR:
            severity = max(severity, Severity.ERROR)
        affected = records is not None
        sample = list(records or ())
        if count is None:
            count = len(sample)
        row_column = self._row_column(datasets)
        if affected and row_column is not None:
            sample = self.locators[datasets[0]].annotate(sample, row_column)
        evidence = BoundedEvidence.admit(
            self.settings.engine,
            status,
            message,
            sample,
            dict(metrics or {}),
            count,
            affected=affected,
            omitted=bool(getattr(records, "omitted", False)),
        )
        if affected_rows is None and unit == "rows" and affected:
            affected_rows = count
        rule = INPUT_RULES[rule_id]
        self._orders.append(self.order(owner))
        self.results.append(
            RuleResult(
                rule_id=rule_id,
                name=rule.name,
                description=rule.description,
                category=rule.category,
                check=check,
                check_id=check_id or rule_id,
                check_name=rule.check_name,
                item_name=self.item_name(check_id, check, datasets, parameters),
                columns=list(columns),
                parameters=redact(parameters or {}),
                checks=[check_identity(rule_id, **redact(dict(item))) for item in checks],
                status=status,
                severity=severity,
                message=evidence.message,
                affected_records=evidence.records,
                affected_count=count,
                affected_count_exact=exact_count(status, count, count_exact),
                evidence_complete=evidence.complete,
                evidence_truncated=evidence.truncated,
                total_records=total,
                unit=unit,
                affected_rows=affected_rows,
                recommendation=rule.recommendation,
                metrics=evidence.metrics,
                datasets=list(datasets),
            )
        )

    def _row_column(self, datasets):
        """The row-position field of a single multi-file dataset's evidence, if any."""
        if len(datasets) != 1 or datasets[0] not in self.locators:
            return None
        return self.locators[datasets[0]].row_column

    def _ledger(self, rule_id, owner, kind, label="", identities=(), name="", rows=0):
        selected = owner if isinstance(owner, str) and owner in self._positions else None
        if self.enabled(rule_id, selected):
            self._entries.append(
                _Entry(self.order(owner), rule_id, kind, label, tuple(identities), name, rows)
            )

    def passed(self, rule_id, owner, label, *identities):
        """Record a contract that passed, for its rule's summary."""
        self._ledger(rule_id, owner, "passed", label, identities)

    def assessed(self, rule_id, owner, *identities):
        """Record contracts that passed without a summary label of their own (INP-004)."""
        self._ledger(rule_id, owner, "identity", identities=identities)

    def exempt(self, rule_id, owner, name, label, rows):
        """Rows a rule exempted under its policy, reported with its summary."""
        if rows:
            self._ledger(rule_id, owner, "exempt", label, name=name, rows=rows)

    def unattributed(self, rule_ids, elapsed):
        """Time spent on contracts that passed, for their rules' summaries."""
        for rule_id in rule_ids:
            self._elapsed[rule_id] = self._elapsed.get(rule_id, 0.0) + elapsed / len(rule_ids)

    def mark(self):
        """A point to roll back to when a stage's results must be discarded."""
        return len(self.results), next(self._sequence)

    def rollback(self, mark):
        """Discard results and ledger entries recorded after ``mark``."""
        count, sequence = mark
        del self.results[count:]
        del self._orders[count:]
        self._entries = [entry for entry in self._entries if entry.order[1] < sequence]

    def _entries_of(self, rule_id, kind):
        return sorted(e for e in self._entries if e.rule_id == rule_id and e.kind == kind)

    def _exemptions(self, rule_id):
        metrics = {}
        for entry in self._entries_of(rule_id, "exempt"):
            metrics.setdefault(entry.name, {})[entry.label] = entry.rows
        return metrics

    def finish(self, rule_ids):
        """Fold passing contracts into summaries; mark rules that had nothing to check."""
        for rule_id in rule_ids:
            if not self.enabled(rule_id):
                continue
            produced = [i for i, result in enumerate(self.results) if result.rule_id == rule_id]
            passes = self._entries_of(rule_id, "passed")
            metrics = self._exemptions(rule_id)
            summary = SUMMARIES.get(rule_id)
            before = len(self.results)
            if summary is not None and passes:
                self._summarize(rule_id, summary, passes, metrics)
            elif produced and metrics:
                first = min(produced, key=self._orders.__getitem__)
                result = self.results[first]
                self.results[first] = result.model_copy(
                    update={"metrics": {**result.metrics, **metrics}}
                )
            elif not produced:
                self.report(rule_id, None, Status.SKIPPED, INPUT_RULES[rule_id].unconfigured)
            if len(self.results) > before and rule_id in self._elapsed:
                self.results[-1] = self.results[-1].model_copy(
                    update={"execution_time_s": round(self._elapsed[rule_id], 6)}
                )

    def _summarize(self, rule_id, summary, passes, metrics):
        if summary.single_identity:
            entries = self._entries_of(rule_id, "identity")
            identities = [identity for entry in entries for identity in entry.identities]
            message = summary.message(len(identities))
            own = dict(identities[0]) if len(identities) == 1 else {"check_id": summary.check_id}
        else:
            identities = [identity for entry in passes for identity in entry.identities]
            message = summary.message(len(passes))
            own = {"check_id": summary.check_id}
        self.report(
            rule_id,
            None,
            Status.PASS,
            message,
            check_id=own.get("check_id"),
            datasets=tuple(own.get("datasets", ())),
            columns=tuple(own.get("columns", ())),
            parameters=own.get("parameters"),
            checks=identities,
            metrics={**metrics, "passed_checks": [entry.label for entry in passes]},
        )


@dataclass(frozen=True, slots=True)
class BoundedEvidence:
    """A result's optional payload within ``max_affected_records`` and the byte budget.

    Counts and identity are the outcome ledger. Records, metrics and the message
    share the byte budget; the passing-check ledger is never dropped.
    """

    message: str
    records: list[dict[str, Any]]
    metrics: dict[str, Any]
    complete: bool | None
    truncated: bool

    @classmethod
    def admit(cls, engine, status, message, sample, metrics, count, *, affected, omitted):
        budget = engine.max_evidence_bytes
        coverage = metrics.pop("passed_checks", None)
        metrics = normalize_evidence(redact(metrics)) if metrics else {}
        if metrics and evidence_json_size(metrics) > budget:
            metrics_dropped, metrics = True, {}
        else:
            metrics_dropped = False
        records = [normalize_evidence(dict(record)) for record in sample]
        records = records[: engine.max_affected_records]
        kept = list(itertools.takewhile(_Budget(budget).admits, records))
        metrics_size = evidence_json_size(metrics) if metrics else 0
        while kept and evidence_json_size(kept) + metrics_size > budget:
            kept.pop()
        used = (evidence_json_size(kept) if kept else 0) + metrics_size
        omitted = omitted or len(kept) != len(records)
        if omitted or metrics_dropped:
            message += EVIDENCE_OMITTED
        message = bounded_diagnostic(redact_text(message), max(0, budget - used))
        if coverage is not None:
            metrics["passed_checks"] = list(coverage)
        complete = metrics.get("duplicate_positions_complete")
        truncated = bool(metrics.get("duplicate_positions_truncated"))
        if (affected and count > len(kept)) or omitted:
            complete, truncated = False, True
        if status == Status.FAIL and not count:
            complete = False
        if status not in (Status.PASS, Status.FAIL) and complete is not None:
            complete = None
        return cls(message, kept, metrics, complete, truncated or metrics_dropped)


class _Budget:
    """Admit records while their encoded sizes fit the evidence byte budget."""

    def __init__(self, budget):
        self.remaining = budget

    def admits(self, record):
        size = evidence_json_size(record)
        if size > self.remaining:
            return False
        self.remaining -= size
        return True
