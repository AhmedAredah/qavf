"""Bounded evidence and structured outcomes shared by the SQL check executors."""

from __future__ import annotations

from qavf.audit_codec import evidence_json_size
from qavf.engine.severity import Severity, Status
from qavf.redaction import redact, redact_text
from qavf.reports.models import RuleResult


class ResultCollector:
    def __init__(self, connection, settings):
        self.connection = connection
        self.settings = settings
        self.results = []

    def enabled(self, rule_id, dataset=None):
        rule = self.settings.rules.get(rule_id)
        engine = self.settings.engine
        return (
            (rule is None or rule.enabled)
            and (
                dataset is None
                or rule is None
                or rule.params.get("datasets") is None
                or dataset in rule.params["datasets"]
            )
            and (engine.run_unconfigured_rules or rule is not None)
            and "input" not in engine.exclude_categories
            and (engine.include_categories is None or "input" in engine.include_categories)
        )

    def records(self, query):
        from qavf.core.evidence import SqlEvidenceReader

        return SqlEvidenceReader(
            self.connection,
            max_records=self.settings.engine.max_affected_records,
            max_bytes=self.settings.engine.max_evidence_bytes,
        ).read(query)

    def add(
        self,
        rule_id,
        name,
        dataset,
        columns=(),
        *,
        count=0,
        total=None,
        records=None,
        parameters=None,
        metrics=None,
        unit="rows",
        severity=None,
        status=None,
        message=None,
        affected_rows=None,
        check=None,
        check_id=None,
        affected_count_exact=None,
    ):
        if not self.enabled(rule_id, dataset):
            return
        status = Status(status or (Status.FAIL if count else Status.PASS))
        assessed = status in (Status.PASS, Status.FAIL)
        count_known = assessed and count is not None and not (status == Status.FAIL and count == 0)
        if affected_count_exact is not None:
            count_known = count_known and affected_count_exact
        count = count or 0
        parameters = redact(dict(parameters or {}))
        source_truncated = bool(getattr(records, "truncated", False))
        records = list(records or [])
        metrics = redact(dict(metrics or {}))
        original = len(records)
        records = records[: self.settings.engine.max_affected_records]
        budget = self.settings.engine.max_evidence_bytes
        # Counts and identity are the outcome ledger. Records, metrics and
        # diagnostics share one optional payload budget, including list framing.
        duplicate_complete = metrics.get("duplicate_positions_complete")
        truncated = source_truncated or bool(metrics.get("duplicate_positions_truncated"))
        metrics_dropped = bool(metrics) and evidence_json_size(metrics) > budget
        if metrics_dropped:
            metrics = {}
        metric_bytes = evidence_json_size(metrics) if metrics else 0
        admitted = []
        record_bytes = 2  # Nonempty array framing.
        for record in records:
            size = evidence_json_size(record) + bool(admitted)
            if record_bytes + size + metric_bytes > budget:
                break
            admitted.append(record)
            record_bytes += size
        records = admitted
        truncated |= len(records) < original or metrics_dropped or count > len(records)
        remaining = max(0, budget - metric_bytes - (record_bytes if records else 0))
        default_message = (
            "Check could not be assessed"
            if status == Status.ERROR
            else "Check skipped"
            if status == Status.SKIPPED
            else (f"{count} offending {unit}" if count else "Check failed")
            if status == Status.FAIL
            else "Check passed"
        )
        message = redact_text(message or default_message)
        encoded_message = message.encode("utf-8")
        if len(encoded_message) > remaining:
            message = encoded_message[:remaining].decode("utf-8", errors="ignore")
            truncated = True
        rule = self.settings.rules.get(rule_id)
        complete = (
            count_known
            and (count == 0 or len(records) >= count)
            and duplicate_complete is not False
            and not source_truncated
            if assessed
            else None
        )
        self.results.append(
            RuleResult(
                rule_id=rule_id,
                name=name,
                description=name,
                category="input",
                check=check or f"{dataset}.{','.join(columns)}: {name}",
                check_id=check_id or name,
                scope="run"
                if dataset == "*"
                else "column"
                if len(columns) == 1
                else "columns"
                if columns
                else "dataset",
                columns=list(columns),
                parameters=parameters,
                status=status,
                severity=severity
                or (
                    rule.severity
                    if rule and rule.severity
                    else Severity.CRITICAL
                    if rule_id in ("INP-001", "INP-002")
                    else Severity.ERROR
                ),
                message=message,
                affected_records=records,
                affected_count=count,
                affected_count_exact=count_known,
                evidence_complete=complete,
                evidence_truncated=truncated,
                affected_rows=affected_rows
                if affected_rows is not None
                else count
                if unit == "rows" and count_known
                else None,
                total_records=total,
                unit=unit,
                metrics=metrics,
                datasets=[] if dataset == "*" else [dataset],
            )
        )

    def row_check(
        self,
        rule_id,
        name,
        loaded,
        condition,
        *,
        columns=(),
        parameters=None,
        table=None,
        total=None,
        severity=None,
        metrics=None,
    ):
        if not self.enabled(rule_id, loaded.name):
            return
        from qavf.sources.sql import quote_identifier as qi

        source = qi(table or loaded.table)
        count = self.connection.execute(
            f"SELECT count(*) FROM {source} WHERE {condition}"
        ).fetchone()[0]
        records = []
        limit = self.settings.engine.max_affected_records
        if count and limit:
            selected = list(dict.fromkeys([loaded.row_column, *columns]))
            records = self.records(
                f"SELECT {', '.join(qi(c) for c in selected)} FROM "
                f"{source} WHERE {condition} ORDER BY "
                f"{qi(loaded.row_column)} LIMIT {limit}"
            )
        self.add(
            rule_id,
            name,
            loaded.name,
            columns,
            count=count,
            total=loaded.rows if total is None else total,
            records=records,
            parameters=parameters,
            metrics=metrics,
            severity=severity,
        )
