"""Plans and executes rules, producing a :class:`~qavf.reports.models.QAReport`."""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import logging
import os
import platform
import queue
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import polars as pl
from pydantic_core import to_json

from qavf._version import __version__
from qavf.audit_codec import canonical_digest
from qavf.engine.compilation import (
    ConfigurationError as ConfigurationError,
)
from qavf.engine.compilation import (
    build_registry as build_registry,
)
from qavf.engine.compilation import (
    build_source_registry as build_source_registry,
)
from qavf.engine.compilation import (
    check_config_keys,
    compile_rules,
    plugin_status,
)
from qavf.engine.compilation import (
    required_check as _required_check,
)
from qavf.engine.compilation import (
    required_rule as _required_rule,
)
from qavf.engine.context import DataContext, DatasetNotAvailableError
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.resources import ResourceLimitError, retained_size
from qavf.engine.session import Invocation
from qavf.engine.severity import Severity, Status
from qavf.redaction import exception_details, redact, redact_text
from qavf.reports.models import (
    QAReport,
    RuleInfo,
    RuleResult,
    RunMetadata,
    evidence_json_size,
    normalize_evidence,
)
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceRegistry

logger = logging.getLogger(__name__)

_MAX_DEFAULT_WORKERS = 8
# Stuck daemon tasks retain capacity across repeated calls in the same process.
_STRANDED_THREADS: set[threading.Thread] = set()
_STRANDED_LOCK = threading.Lock()

#: Libraries whose versions can change a result; recorded in every report.
_RECORDED_PACKAGES = (
    "polars",
    "duckdb",
    "pandas",
    "pydantic",
    "numpy",
    "pyarrow",
    "deepdiff",
    "scipy",
)


class QARunner:
    """Executes the configured rules against a :class:`DataContext`.

    Rules run on a thread pool when ``engine.parallel`` is set. Threads (rather
    than processes) are used because rule work is dominated by Polars, DuckDB
    and NumPy calls that release the GIL, and because it lets every rule share
    the loaded datasets without copying them.

    Args:
        settings: Validated configuration. Defaults are used when omitted.
        registry: Rule registry; discovered from ``settings.plugins`` when omitted.
        sources: Source registry; discovered from ``settings.plugins`` when omitted.
    """

    def __init__(
        self,
        settings: QAVFSettings | None = None,
        registry: RuleRegistry | None = None,
        sources: SourceRegistry | None = None,
    ) -> None:
        self.settings = settings or QAVFSettings()
        from qavf.execution.policy import check_policy

        check_policy(self.settings.engine)
        check_config_keys(self.settings)
        self._supplied_registries = registry is not None or sources is not None
        if self.settings.engine.execution_mode == "isolated" and (
            registry is not None or sources is not None
        ):
            raise ConfigurationError(
                "isolated execution requires configured plugins, not live rule/source registries"
            )
        self._registry = registry
        self._sources = sources
        if self.settings.engine.execution_mode != "isolated":
            self._registry = self.registry  # Preserve eager in-process discovery.
            self._sources = self.sources
        self.needs_configuration: dict[str, list[str]] = {}
        """Discovered rules skipped because they require parameters, with those parameters."""
        self.excluded: dict[str, str] = {}
        """Discovered rules left out of the plan, with the reason."""

    @property
    def registry(self) -> RuleRegistry:
        if self._registry is None:
            self._registry = build_registry(self.settings)
        return self._registry

    @property
    def sources(self) -> SourceRegistry:
        if self._sources is None:
            self._sources = build_source_registry(self.settings)
        return self._sources

    # -------------------------------------------------------------------- plan
    def plan(self) -> list[Rule]:
        """Compile rules and publish diagnostics only after complete validation.

        A failed compilation preserves the previous complete diagnostics. No
        partially prepared plan can become the runner's visible state.
        """
        compiled = compile_rules(self.settings, self.registry, self.sources)
        self.needs_configuration = {
            key: list(fields) for key, fields in compiled.needs_configuration.items()
        }
        self.excluded = dict(compiled.excluded)
        return list(compiled.rules)

    def plugin_status(self) -> dict[str, str]:
        """Every entry point tried, as ``rules:name`` / ``sources:name`` -> status."""
        return plugin_status(self.registry, self.sources)

    def context(
        self,
        frames: Mapping[str, Any] | None = None,
        threads: int | None = None,
        *,
        workers: int | None = None,
    ) -> DataContext:
        """Build the data context for this configuration.

        Args:
            frames: In-memory datasets.
            threads: DuckDB threads per connection (``None``: DuckDB's default).

        Raises:
            ConfigurationError: for an invalid dataset source.
        """
        from qavf.execution.policy import check_policy

        check_policy(self.settings.engine)
        if self.settings.engine.execution_mode == "isolated":
            raise ConfigurationError("use run() to create a supervised isolated context")
        try:
            return DataContext.from_settings(
                self.settings, frames, self.sources, threads, execution_workers=workers
            )
        except ValueError as exc:
            raise ConfigurationError(exception_details(exc)) from None

    # --------------------------------------------------------------------- run
    def run(
        self,
        ctx: DataContext | None = None,
        frames: Mapping[str, Any] | None = None,
    ) -> QAReport:
        """Execute the plan and return the report.

        Args:
            ctx: Data context. Built from the settings (and ``frames``) when omitted.
            frames: In-memory datasets (Polars, pandas or Arrow) that supply or
                override configured ones. Ignored when ``ctx`` is given.
        """
        from qavf.execution.policy import check_policy

        check_policy(self.settings.engine)
        if self.settings.engine.execution_mode == "isolated":
            if self._supplied_registries:
                raise ConfigurationError("isolated execution requires configured plugins")
            if ctx is not None:
                raise ConfigurationError("isolated execution cannot borrow a live DataContext")
            from qavf.execution.dispatch import validate_isolated

            return validate_isolated(self.settings, frames)
        rules = self.plan()
        workers = self._worker_count(len(rules))
        # Parallel rules each run DuckDB queries with their own thread pool; share the
        # CPUs between the two levels instead of oversubscribing them.
        from qavf.engine.thread_budget import connection_threads

        threads = connection_threads(self.settings.engine.native_threads, workers)
        context = ctx or self.context(frames, threads, workers=workers)
        from qavf.engine.planning import build_plan

        with contextlib.ExitStack() as ownership:
            if ctx is None:
                ownership.callback(context.close)
            execution_plan = build_plan(rules, self.settings, context.declared())
            if ctx is None:
                context._execution_plan = execution_plan
            started = datetime.now(UTC)
            logger.info("Running %d rule(s) with %d worker(s)", len(rules), workers)

            timeout = self.settings.engine.rule_timeout_s
            if timeout is not None:
                batches = self._run_with_deadlines(rules, context, workers, timeout)
            elif workers > 1:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qavf") as pool:
                    batches = list(pool.map(lambda rule: self._execute(rule, context), rules))
                context.finish_inactive_workers()
            else:
                batches = [self._execute(rule, context) for rule in rules]
            for name, reason in context.verify_inputs().items():
                batches.append(
                    [
                        RuleResult(
                            rule_id="QAVF-INPUT-INTEGRITY",
                            name="Input identity verification",
                            description="The assessed source must retain its recorded identity.",
                            category="input",
                            status=Status.ERROR,
                            severity=Severity.ERROR,
                            check=name,
                            datasets=[name],
                            message=_bounded_diagnostic(
                                redact_text(reason),
                                self.settings.engine.max_evidence_bytes,
                            ),
                        )
                    ]
                )
        finished = datetime.now(UTC)

        # Everything below only describes the run: it never loads data.
        config = self.effective_config()
        report = QAReport(
            metadata=RunMetadata(
                run_id=uuid.uuid4().hex[:12],
                project=self.settings.project,
                qavf_version=__version__,
                started_at=started,
                finished_at=finished,
                config_path=str(self.settings.source_path) if self.settings.source_path else None,
                datasets=context.describe(),
                unused_datasets=context.unused(),
                unavailable_datasets=context.unavailable(),
                parallel=workers > 1,
                max_workers=workers,
                extensions=redact(self.settings.extensions),
                excluded_rules=dict(sorted(self.excluded.items())),
                plugins=self.plugin_status(),
                fail_on=self.settings.engine.fail_on,
                fail_on_incomplete=self.settings.engine.fail_on_incomplete,
                config=config,
                config_digest=config_digest(config),
                rules=(infos := [_rule_info(rule) for rule in rules]),
                plan_digest=config_digest(
                    {"rules": [info.model_dump(mode="python") for info in infos]}
                ),
                environment=environment(),
                resource_usage=context.resource_usage(),
                query_plans=context.query_plans() + execution_plan.describe(),
            ),
            results=self._with_requirements(
                [
                    *(result for batch in batches for result in batch),
                    *self._unconfigured_results(),
                ]
            ),
        )
        report = self._fit_report_budget(report)
        summary = report.summary
        logger.info(
            "QA finished in %.2fs: %d passed, %d failed, %d errored, %d skipped (highest: %s)",
            report.metadata.duration_s,
            summary.passed,
            summary.failed,
            summary.errored,
            summary.skipped,
            summary.highest_severity.value if summary.highest_severity else "none",
        )
        return report

    def _fit_report_budget(self, report: QAReport) -> QAReport:
        """Bound persisted report bytes without losing assessment/coverage outcomes."""
        limit = self.settings.engine.max_report_bytes
        encoded_size = report.json_size_bytes()
        if encoded_size <= limit:
            return report
        results = [
            r.model_copy(
                update={
                    "affected_records": [],
                    "metrics": {"passed_checks": r.metrics["passed_checks"]}
                    if r.metrics.get("passed_checks")
                    else {},
                    "message": r.message + " Detailed evidence omitted by engine.max_report_bytes.",
                }
            )
            for r in report.results
        ]
        metadata = report.metadata.model_copy(
            update={
                "config": {},
                "extensions": {},
                "query_plans": [],
                "rules": [r.model_copy(update={"params": {}}) for r in report.metadata.rules],
                "resource_usage": {
                    **report.metadata.resource_usage,
                    "report_details_omitted": [
                        "affected_records",
                        "noncoverage_metrics",
                        "config",
                        "extensions",
                        "rule_params",
                        "query_plans",
                    ],
                    "original_report_bytes": encoded_size,
                    "config_and_plan_digests": (
                        "refer to the original executed configuration "
                        "and plan before detail omission"
                    ),
                },
            }
        )
        trimmed = report.model_copy(update={"metadata": metadata, "results": results})
        essential_size = trimmed.json_size_bytes()
        if essential_size > limit:
            raise ResourceLimitError(
                f"report outcome ledger requires {essential_size:,} "
                "bytes after optional evidence removal; "
                f"engine.max_report_bytes is {limit:,}. Increase the "
                "budget; no success report was emitted."
            )
        return trimmed

    def effective_config(self) -> dict[str, Any]:
        """The configuration in effect, defaults included, with secrets redacted."""
        data = self.settings.model_dump(mode="python", by_alias=True)
        redacted: dict[str, Any] = redact(data)
        return redacted

    # ---------------------------------------------------------------- internal
    def _with_requirements(self, results: list[RuleResult]) -> list[RuleResult]:
        """``results`` plus an ERROR for each ``engine.require`` entry not assessed."""
        added = []
        for entry in self.settings.engine.require:
            rule_id, check = _required_rule(entry), _required_check(entry)
            # A passing check may be folded into its rule's summary (``passed_checks``).
            matching = [
                r
                for r in results
                if r.rule_id == rule_id
                and (
                    check is None
                    or r.check == check
                    or check in (r.metrics.get("passed_checks") or ())
                )
            ]
            assessed = [r for r in matching if r.status in (Status.PASS, Status.FAIL)]
            if assessed and not any(r.status is Status.ERROR for r in matching):
                continue
            if rule_id in self.excluded:
                reason = f"the rule was excluded: {self.excluded[rule_id]}"
            elif not matching:
                reason = "no result was produced for it"
            elif any(r.status is Status.ERROR for r in matching):
                reason = "it could not be assessed"
            else:
                reason = "it was skipped"
            rule_cls = self.registry.get(rule_id)
            added.append(
                RuleResult(
                    rule_id=rule_id,
                    name=rule_cls.name,
                    description=rule_cls.description,
                    category=rule_cls.category,
                    check=f"required: {entry}",
                    status=Status.ERROR,
                    severity=max(rule_cls.default_severity, Severity.ERROR),
                    message=f"Not assessed: required check {entry!r} was not assessed: {reason}.",
                    recommendation=rule_cls.recommendation,
                )
            )
        return [*results, *added]

    def _unconfigured_results(self) -> list[RuleResult]:
        """SKIPPED results for discovered rules that need parameters nobody supplied."""
        results = []
        for rule_id, fields in sorted(self.needs_configuration.items()):
            rule_cls = self.registry.get(rule_id)
            results.append(
                RuleResult(
                    rule_id=rule_id,
                    name=rule_cls.name,
                    description=rule_cls.description,
                    category=rule_cls.category,
                    status=Status.SKIPPED,
                    severity=rule_cls.default_severity,
                    message=(
                        f"Not configured: this rule requires parameter(s) "
                        f"{', '.join(fields)} under rules.{rule_id}.params."
                    ),
                    recommendation=rule_cls.recommendation,
                )
            )
        return results

    def _worker_count(self, n_rules: int) -> int:
        engine = self.settings.engine
        if not engine.parallel or n_rules < 2:
            return 1
        limit = engine.max_workers or min(_MAX_DEFAULT_WORKERS, os.cpu_count() or 1)
        return max(1, min(limit, n_rules))

    def _run_with_deadlines(
        self, rules: list[Rule], ctx: DataContext, workers: int, timeout: float
    ) -> list[list[RuleResult]]:
        """Bounded cooperative deadlines; timed-out threads are never replaced.

        Python cannot forcibly terminate arbitrary plugin/native work. A stranded
        worker consumes its slot until it exits; queued work becomes unassessed
        when every slot is stranded. DuckDB calls are cooperatively interrupted.
        """
        grace_end = time.monotonic() + self.settings.engine.timeout_grace_s
        with _STRANDED_LOCK:
            prior_workers = tuple(_STRANDED_THREADS)
        for prior in prior_workers:
            if prior.is_alive() and time.monotonic() < grace_end:
                prior.join(timeout=max(0.0, grace_end - time.monotonic()))
        with _STRANDED_LOCK:
            _STRANDED_THREADS.intersection_update(t for t in _STRANDED_THREADS if t.is_alive())
            capacity = max(0, workers - len(_STRANDED_THREADS))
        if not capacity:
            return [
                [
                    self._result(
                        rule,
                        Status.ERROR,
                        (
                            "Not assessed: worker capacity is retained by "
                            "timed-out work from an earlier run."
                        ),
                        severity=max(rule.severity, Severity.ERROR),
                    )
                ]
                for rule in rules
            ]
        pending: queue.Queue[int] = queue.Queue()
        for index in range(len(rules)):
            pending.put(index)
        results: list[list[RuleResult] | None] = [None] * len(rules)
        started: dict[int, tuple[float, Invocation]] = {}
        abandoned: set[int] = set()
        lock = threading.Lock()

        worker_threads: dict[Invocation, threading.Thread] = {}

        def worker() -> None:
            try:
                while True:
                    try:
                        index = pending.get_nowait()
                    except queue.Empty:
                        return
                    invocation = Invocation()
                    with lock:
                        started[index] = (time.monotonic(), invocation)
                        worker_threads[invocation] = threading.current_thread()
                    outcome = self._execute(rules[index], ctx, invocation)
                    with lock:
                        started.pop(index, None)
                        if index in abandoned:
                            return
                        results[index] = outcome
            finally:
                ctx.finish_worker()

        for _ in range(min(capacity, len(rules))):
            threading.Thread(target=worker, name="qavf-rule", daemon=True).start()
        while True:
            cancel = []
            with lock:
                now = time.monotonic()
                for index, (stamp, invocation) in list(started.items()):
                    if now - stamp > timeout and index not in abandoned:
                        abandoned.add(index)
                        cancel.append(invocation)
                        with _STRANDED_LOCK:
                            _STRANDED_THREADS.add(worker_threads[invocation])
                        rule = rules[index]
                        results[index] = [
                            self._result(
                                rule,
                                Status.ERROR,
                                f"Not assessed: the rule did not finish within {timeout:g}s "
                                "(engine.rule_timeout_s). Cooperative cancellation was requested; "
                                "arbitrary plugin code may still be running "
                                "within its occupied worker slot.",
                                timeout,
                                severity=max(rule.severity, Severity.ERROR),
                            )
                        ]
                if len(abandoned) >= min(capacity, len(rules)) and rules:
                    while True:
                        try:
                            index = pending.get_nowait()
                        except queue.Empty:
                            break
                        rule = rules[index]
                        results[index] = [
                            self._result(
                                rule,
                                Status.ERROR,
                                (
                                    "Not assessed: all bounded worker slots are "
                                    "occupied by timed-out work."
                                ),
                                severity=max(rule.severity, Severity.ERROR),
                            )
                        ]
                finished = all(result is not None for result in results)
            for invocation in cancel:
                ctx.cancel_rule(invocation)
            if finished:
                grace = time.monotonic() + self.settings.engine.timeout_grace_s
                while ctx._active and time.monotonic() < grace:
                    time.sleep(min(0.005, max(0.0, grace - time.monotonic())))
                return [result for result in results if result is not None]
            time.sleep(0.005)

    def _execute(
        self, rule: Rule, ctx: DataContext, invocation: Invocation | None = None
    ) -> list[RuleResult]:
        """Isolate the complete invocation, including entry and resource release."""
        began = False
        results: list[RuleResult] = []
        try:
            ctx.begin_rule(invocation)
            began = True
            started, loaded = time.perf_counter(), ctx.thread_load_seconds()
            results = self._execute_isolated(rule, ctx)
            elapsed = round(time.perf_counter() - started, 6)
            load = round(ctx.thread_load_seconds() - loaded, 6)
            results = [
                result.model_copy(
                    update={
                        "execution_time_s": elapsed if index == 0 else 0.0,
                        "load_time_s": load if index == 0 else 0.0,
                    }
                )
                for index, result in enumerate(results)
            ]
        except Exception as exc:
            results = [
                self._result(
                    rule,
                    Status.ERROR,
                    f"Rule invocation failed: {exception_details(exc)}",
                    severity=max(rule.severity, Severity.ERROR),
                )
            ]
        finally:
            if began:
                try:
                    if ctx._execution_plan is not None:
                        ctx.release(ctx._execution_plan.complete(rule.rule_id))
                except Exception as exc:
                    results.append(
                        self._result(
                            rule,
                            Status.ERROR,
                            f"Rule cleanup failed: {exception_details(exc)}",
                            severity=max(rule.severity, Severity.ERROR),
                        )
                    )
                finally:
                    ctx.end_rule()
        return results

    def _execute_isolated(self, rule: Rule, ctx: DataContext) -> list[RuleResult]:
        """Run one rule, converting every outcome (including a crash) to results.

        The isolation covers the whole lifecycle (dependency checks, evaluation and
        turning findings into results), so one misbehaving rule can never abort the
        run or prevent the report from being written.
        """
        try:
            return self._execute_unguarded(rule, ctx)
        except Exception as exc:
            logger.error(
                "Rule %s failed outside its evaluation: %s: %s",
                rule.rule_id,
                type(exc).__name__,
                exception_details(exc),
            )
            return [
                self._result(
                    rule,
                    Status.ERROR,
                    f"Rule could not be completed: {type(exc).__name__}: {exc}",
                    severity=max(rule.severity, Severity.ERROR),
                )
            ]

    def _execute_unguarded(self, rule: Rule, ctx: DataContext) -> list[RuleResult]:
        missing = [name for name in rule.requires if not ctx.has(name)]
        if missing:
            reasons = "; ".join(str(ctx.unavailable_reason(name)) for name in missing)
            if rule.rule_id in self.settings.rules:
                # Explicitly configured: the user intended this check, so it is blocked.
                return [
                    self._result(
                        rule,
                        Status.ERROR,
                        f"Not assessed: required dataset(s) unavailable: {reasons}.",
                        severity=max(rule.severity, Severity.ERROR),
                    )
                ]
            return [
                self._result(
                    rule,
                    Status.SKIPPED,
                    f"Skipped: required dataset(s) not available: {', '.join(missing)}.",
                )
            ]

        start = time.perf_counter()
        try:
            outcome = rule.evaluate(ctx)
        except DatasetNotAvailableError as exc:
            # The rule needed data it could not get: not assessed, never a silent skip.
            return [
                self._result(
                    rule,
                    Status.ERROR,
                    f"Not assessed: {exc.args[0]}.",
                    time.perf_counter() - start,
                    severity=max(rule.severity, Severity.ERROR),
                )
            ]
        except Exception as exc:
            logger.error(
                "Rule %s raised: %s: %s", rule.rule_id, type(exc).__name__, exception_details(exc)
            )
            return [
                self._result(
                    rule,
                    Status.ERROR,
                    f"Rule could not be evaluated: {type(exc).__name__}: {exc}",
                    time.perf_counter() - start,
                    # An unassessed check must never read as less serious than a failed one.
                    severity=max(rule.severity, Severity.ERROR),
                )
            ]
        elapsed = time.perf_counter() - start

        findings = [outcome] if isinstance(outcome, Finding) else list(outcome)
        if not findings:
            return [self._result(rule, Status.SKIPPED, "Rule reported nothing.", elapsed)]
        # Time is measured per rule; attribute it to the first result only.
        results = []
        for index, finding in enumerate(findings):
            severity = None
            if not finding.assessed:
                status = Status.ERROR
                severity = max(finding.severity or rule.severity, Severity.ERROR)
            elif finding.skipped:
                status = Status.SKIPPED
            else:
                status = Status.PASS if finding.passed else Status.FAIL
            try:
                result = self._result(
                    rule,
                    status,
                    finding.message,
                    elapsed if index == 0 else 0.0,
                    finding=finding,
                    severity=severity,
                )
                to_json(result)  # evidence that cannot be serialised must not break the report
            except Exception as exc:
                logger.error(
                    "Rule %s returned a finding that could not be recorded: %s: %s",
                    rule.rule_id,
                    type(exc).__name__,
                    exception_details(exc),
                )
                result = self._result(
                    rule,
                    Status.ERROR,
                    f"Rule result could not be recorded: {type(exc).__name__}: {exc}",
                    severity=max(rule.severity, Severity.ERROR),
                )
            results.append(result)
        logger.debug("%s produced %d result(s) in %.3fs", rule.rule_id, len(results), elapsed)
        return results

    def _result(
        self,
        rule: Rule,
        status: Status,
        message: str,
        elapsed: float = 0.0,
        finding: Finding | None = None,
        severity: Severity | None = None,
    ) -> RuleResult:
        records, count, sample_truncated = self._sample(finding.affected if finding else None)
        if finding and finding.affected_count is not None:
            count = finding.affected_count
        unit = finding.unit if finding else None
        affected_rows = finding.affected_rows if finding else None
        if affected_rows is None and unit == "rows" and finding and finding.affected is not None:
            affected_rows = count
        byte_limit = self.settings.engine.max_evidence_bytes
        raw_metrics = finding.metrics if finding else {}
        # Coverage is the outcome ledger, not optional evidence. It is preserved
        # exactly and bounded only by the complete-report budget.
        coverage = raw_metrics.get("passed_checks")
        evidence_metrics = {
            key: value for key, value in raw_metrics.items() if key != "passed_checks"
        }
        metrics = (
            _jsonable(evidence_metrics) if retained_size(evidence_metrics) <= byte_limit else {}
        )
        metrics_dropped = bool(evidence_metrics) and not metrics
        used = 0
        bounded_records = []
        for record in records:
            if retained_size(record) > byte_limit - used:
                break
            size = evidence_json_size(record)
            if used + size > byte_limit:
                break
            bounded_records.append(record)
            used += size
        metrics_size = evidence_json_size(metrics) if metrics else 0
        if used + metrics_size > byte_limit:
            metrics_dropped = bool(metrics)
            metrics = {}
        # The same canonical codec sizes optional payloads and persisted reports.
        # Include the list framing for nonempty evidence, not an estimate of each row.
        while (
            bounded_records
            and evidence_json_size(bounded_records)
            + (evidence_json_size(metrics) if metrics else 0)
            > byte_limit
        ):
            bounded_records.pop()
        used = (evidence_json_size(bounded_records) if bounded_records else 0) + (
            evidence_json_size(metrics) if metrics else 0
        )
        if len(bounded_records) != len(records) or metrics_dropped or sample_truncated:
            message += (
                " Evidence was truncated or omitted: unsupported values or "
                "engine.max_evidence_bytes."
            )
        message = _bounded_diagnostic(redact_text(message), max(0, byte_limit - used))
        if coverage is not None:
            metrics["passed_checks"] = _jsonable(coverage)
        return RuleResult(
            rule_id=rule.rule_id,
            name=rule.name,
            description=rule.description,
            category=rule.category,
            check=finding.check if finding else None,
            status=status,
            severity=severity or (finding.severity if finding else None) or rule.severity,
            # Messages often quote errors, which may quote connection strings or URLs.
            message=redact_text(message),
            affected_records=bounded_records,
            affected_count=count,
            total_records=finding.total_records if finding else None,
            unit=unit,
            affected_rows=affected_rows,
            execution_time_s=round(elapsed, 6),
            recommendation=rule.recommendation,
            metrics=metrics,
            datasets=list(finding.datasets if finding and finding.datasets else rule.requires),
        )

    def _sample(
        self, affected: pl.DataFrame | Sequence[Mapping[str, Any]] | None
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """Bound raw rows before Python/JSON conversion; preserve the exact count."""
        limit = self.settings.engine.max_affected_records
        remaining = self.settings.engine.max_evidence_bytes
        if affected is None:
            return [], 0, False
        records: list[dict[str, Any]] = []
        raw: Mapping[str, Any]
        count = affected.height if isinstance(affected, pl.DataFrame) else len(affected)
        for index in range(min(limit, count)):
            if isinstance(affected, pl.DataFrame):
                row = affected.slice(index, 1)
                if int(row.estimated_size()) > remaining:
                    return records, count, True
                raw = row.to_dicts()[0]
            else:
                raw = affected[index]
            if retained_size(raw) > remaining:
                return records, count, True
            converted = _jsonable(dict(raw))
            size = evidence_json_size(converted)
            if size > remaining:
                return records, count, True
            records.append(converted)
            remaining -= size
        return records, count, False


def config_digest(config: Mapping[str, Any]) -> str:
    """SHA-256 of the collision-safe canonical audit representation."""
    return canonical_digest(config)


@functools.cache
def _packages_distributions() -> Mapping[str, list[str]]:
    """Top-level package -> distributions; scanning every installed one is slow."""
    try:
        return importlib.metadata.packages_distributions()
    except Exception:
        return {}


@functools.cache
def _distribution(module: str) -> str | None:
    """``name==version`` of the installed distribution providing ``module``, if any."""
    top = module.split(".", 1)[0]
    names = _packages_distributions().get(top) or []
    for name in names:
        try:
            return f"{name}=={importlib.metadata.version(name)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _rule_info(rule: Rule) -> RuleInfo:
    module = type(rule).__module__
    return RuleInfo(
        rule_id=rule.rule_id,
        severity=rule.severity,
        params=redact(rule.params.model_dump(mode="python")),
        module=module,
        distribution=_distribution(module),
    )


@functools.cache
def _environment() -> tuple[tuple[str, str], ...]:
    info = {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
        "qavf": __version__,
    }
    for package in _RECORDED_PACKAGES:
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return tuple(info.items())


def environment() -> dict[str, str]:
    """Python, platform and the versions of QAVF and its computing libraries."""
    return dict(_environment())


def _jsonable(value: Any) -> Any:
    """Normalize through the report codec's supported, detached evidence contract."""
    return normalize_evidence(value)


def _bounded_diagnostic(message: str, limit: int) -> str:
    """Bound encoded optional diagnostics; status/check/count remain the outcome ledger."""
    if not message or evidence_json_size(message) <= limit:
        return message
    suffix = " [diagnostic truncated]"
    if evidence_json_size(suffix) > limit:
        return ""
    low, high = 0, len(message)
    while low < high:
        middle = (low + high + 1) // 2
        if evidence_json_size(message[:middle] + suffix) <= limit:
            low = middle
        else:
            high = middle - 1
    return message[:low] + suffix
