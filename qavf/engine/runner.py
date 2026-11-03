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
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic_core import to_json

from qavf._version import __version__
from qavf.audit_codec import EVIDENCE_OMITTED, canonical_digest
from qavf.audit_codec import bounded_diagnostic as _bounded_diagnostic
from qavf.checks.requirements import Requirements, RuleDescription
from qavf.checks.selection import item_tags, row_item_key
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
from qavf.engine.context import DataContext, DatasetNotAvailableError
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.resources import ResourceLimitError, retained_size
from qavf.engine.row_locator import RowLocator
from qavf.engine.session import Invocation
from qavf.engine.severity import Severity, Status
from qavf.redaction import exception_details, redact, redact_text
from qavf.reports.models import (
    QAReport,
    RuleInfo,
    RuleResult,
    RunMetadata,
    check_identity,
    evidence_json_size,
    exact_count,
    normalize_evidence,
)
from qavf.run_control import CANCELLED_MESSAGE, RunControl
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceRegistry

logger = logging.getLogger(__name__)

_MAX_DEFAULT_WORKERS = 8
#: What a rule may raise and still be reported as unassessed. A plugin calling
#: ``sys.exit`` must not end the host process; a caller's KeyboardInterrupt does.
RULE_FAILURES: tuple[type[BaseException], ...] = (Exception, SystemExit)
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
        self._tag_index: dict[tuple[str | None, str], tuple[str, ...]] | None = None
        self._sources = sources
        if self.settings.engine.execution_mode != "isolated":
            self._registry = self.registry  # Preserve eager in-process discovery.
            self._sources = self.sources
        self.excluded: dict[str, str] = {}
        """Check items left out of the plan (disabled), by item name, with the reason."""

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
        self.excluded = dict(compiled.excluded)
        return list(compiled.rules)

    def plugin_status(self) -> dict[str, str]:
        """Every entry point tried, as ``rules:name`` / ``sources:name`` -> status."""
        return plugin_status(
            self.registry,
            self.sources,
            comparison_plugins=self.settings.plugins.entry_points,
        )

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
        *,
        control: RunControl | None = None,
    ) -> QAReport:
        """Execute the plan and return the report.

        Args:
            ctx: Data context. Built from the settings (and ``frames``) when omitted.
            frames: In-memory datasets (Polars, pandas or Arrow) that supply or
                override configured ones. Ignored when ``ctx`` is given.
            control: Progress events and cancellation (see :mod:`qavf.run_control`).
        """
        from qavf.execution.policy import check_policy

        check_policy(self.settings.engine)
        if self.settings.engine.execution_mode == "isolated":
            if self._supplied_registries:
                raise ConfigurationError("isolated execution requires configured plugins")
            if ctx is not None:
                raise ConfigurationError("isolated execution cannot borrow a live DataContext")
            if control is not None:
                raise ConfigurationError(
                    "progress events and cancellation require engine.execution_mode: in_process"
                )
            from qavf.execution.dispatch import validate_isolated

            return validate_isolated(self.settings, frames)
        rules = self.plan()
        workers = self._worker_count(len(rules), frames)
        # Parallel rules each run DuckDB queries with their own thread pool; share the
        # CPUs between the two levels instead of oversubscribing them.
        from qavf.engine.thread_budget import connection_threads
        from qavf.resource_plan import ResourcePlan

        resource_plan = ResourcePlan.build(self.settings, frames, requested_workers=workers)
        workers = resource_plan.workers
        per_connection = (
            resource_plan.memory_bytes // 2 // resource_plan.connection_slots
            if resource_plan.memory_bytes is not None
            else None
        )
        threads = connection_threads(
            self.settings.engine.native_threads, workers, memory_bytes=per_connection
        )
        with resource_plan.bind(self.settings, frames):
            context = ctx or self.context(frames, threads, workers=workers)
        from qavf.engine.planning import build_plan

        with contextlib.ExitStack() as ownership:
            if ctx is None:
                ownership.callback(context.close)
            execution_plan = build_plan(rules, self.settings, context.declared())
            if ctx is None:
                context._execution_plan = execution_plan
            context.reporting_snapshot()
            started = datetime.now(UTC)
            logger.info("Running %d rule(s) with %d worker(s)", len(rules), workers)
            if control is not None:
                control.started(len(rules))
                control.token.add_listener(context.cancel_active_rules)
                ownership.callback(control.token.remove_listener, context.cancel_active_rules)

            timeout = self.settings.engine.rule_timeout_s
            if timeout is not None:
                batches = self._run_with_deadlines(rules, context, workers, timeout, control)
            elif workers > 1:
                execute = functools.partial(self._execute, ctx=context, control=control)
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qavf") as pool:
                    batches = list(pool.map(execute, rules))
                context.finish_inactive_workers()
            else:
                batches = [self._execute(rule, context, control=control) for rule in rules]
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
        snapshot = context.reporting_snapshot()
        report = QAReport(
            metadata=RunMetadata(
                run_id=uuid.uuid4().hex[:12],
                project=self.settings.project,
                qavf_version=__version__,
                started_at=started,
                finished_at=finished,
                config_path=str(self.settings.source_path) if self.settings.source_path else None,
                datasets=snapshot["datasets"],
                unused_datasets=snapshot["unused_datasets"],
                unavailable_datasets=snapshot["unavailable_datasets"],
                parallel=workers > 1,
                max_workers=workers,
                extensions=redact(self.settings.extensions),
                excluded_rules=dict(sorted(self.excluded.items())),
                plugins=self.plugin_status(),
                fail_on=self.settings.run.fail_on,
                fail_on_incomplete=self.settings.run.fail_on_incomplete,
                tag_fail_on=self.settings.run.tag_fail_on(),
                cancelled=control is not None and control.cancelled,
                config=config,
                config_digest=config_digest(config),
                rules=(infos := [_rule_info(rule) for rule in rules]),
                plan_digest=config_digest(
                    {"rules": [info.model_dump(mode="python") for info in infos]}
                ),
                environment=environment(),
                resource_usage=snapshot["resource_usage"],
                query_plans=snapshot["query_plans"] + execution_plan.describe(),
            ),
            results=self._with_requirements([result for batch in batches for result in batch]),
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
        if control is not None:
            control.finished(report)
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
                    "evidence_complete": False if r.affected_count else r.evidence_complete,
                    "evidence_truncated": True,
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
        """``results`` plus an ERROR for each ``run.require`` entry not assessed."""
        if not self.settings.run.require:
            return results
        return Requirements(
            self.settings, self._describe, {rule.check_name: rule.rule_id for rule in self.registry}
        ).apply(results)

    def _describe(self, rule_id: str) -> RuleDescription | None:
        if rule_id not in self.registry:
            return None
        rule = self.registry.get(rule_id)
        return RuleDescription(
            rule.rule_id,
            rule.check_name,
            rule.name,
            rule.description,
            rule.category,
            rule.default_severity,
            rule.recommendation,
        )

    def _worker_count(self, n_rules: int, frames: Mapping[str, Any] | None = None) -> int:
        engine = self.settings.engine
        if not engine.parallel or n_rules < 2:
            return 1
        limit = engine.max_workers or min(_MAX_DEFAULT_WORKERS, os.cpu_count() or 1)
        from qavf.engine.thread_budget import memory_workers

        requested = memory_workers(max(1, min(limit, n_rules)), engine.memory_budget_bytes)
        return requested

    def _run_with_deadlines(
        self,
        rules: list[Rule],
        ctx: DataContext,
        workers: int,
        timeout: float,
        control: RunControl | None = None,
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
        lock = threading.Condition()

        worker_threads: dict[Invocation, threading.Thread] = {}

        def worker() -> None:
            index: int | None = None
            try:
                while True:
                    with lock:
                        if index is not None and index in abandoned:
                            return
                        try:
                            following = pending.get_nowait()
                        except queue.Empty:
                            return
                        if index is not None:
                            started.pop(index, None)
                        index = following
                        invocation = Invocation()
                        started[index] = (time.monotonic(), invocation)
                        worker_threads[invocation] = threading.current_thread()
                        lock.notify_all()
                    if control is not None and not control.cancelled:
                        control.step_started((rules[index].rule_id,))
                    try:
                        outcome = self._execute(
                            rules[index],
                            ctx,
                            invocation,
                            finish_invocation=False,
                            control=control,
                            emit=False,
                        )
                    except BaseException as exc:
                        # A plugin can terminate its worker with SystemExit;
                        # it must not leave queued rules without a live owner.
                        rule = rules[index]
                        outcome = [
                            self._result(
                                rule,
                                Status.ERROR,
                                f"Rule worker failed: {exception_details(exc)}",
                                severity=max(rule.severity, Severity.ERROR),
                            )
                        ]
                    with lock:
                        if index in abandoned:
                            return
                        results[index] = outcome
                        if control is not None:
                            control.step_finished((rules[index].rule_id,), None, outcome)
                        lock.notify_all()
            finally:
                try:
                    # Keep the session active until the native handles are closed.
                    # An owned context must not race this cleanup with a second
                    # close after the deadline monitor returns an incomplete report.
                    for cleanup in (ctx.finish_worker, ctx.end_rule):
                        try:
                            cleanup()
                        except BaseException as exc:
                            with lock:
                                if index is not None and index not in abandoned:
                                    rule = rules[index]
                                    results[index] = [
                                        *(results[index] or []),
                                        self._result(
                                            rule,
                                            Status.ERROR,
                                            f"Rule worker cleanup failed: {exception_details(exc)}",
                                            severity=max(rule.severity, Severity.ERROR),
                                        ),
                                    ]
                finally:
                    # Publishing a rule result does not release its worker's
                    # connection. Keep the final invocation deadline-monitored
                    # until cleanup completes; the next invocation transfers
                    # this ownership atomically when a worker is reused.
                    with lock:
                        if index is not None:
                            started.pop(index, None)
                        lock.notify_all()

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
                        if control is not None:
                            control.step_finished((rule.rule_id,), None, results[index] or [])
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
                        if control is not None:
                            control.step_finished((rule.rule_id,), None, results[index] or [])
                finished = all(result is not None for result in results) and all(
                    index in abandoned for index in started
                )
            for invocation in cancel:
                ctx.cancel_rule(invocation)
            if finished:
                grace = time.monotonic() + self.settings.engine.timeout_grace_s
                while ctx._active and time.monotonic() < grace:
                    time.sleep(min(0.005, max(0.0, grace - time.monotonic())))
                return [result for result in results if result is not None]
            with lock:
                # Recheck under the notification lock: completion can race cancellation.
                if all(result is not None for result in results) and all(
                    index in abandoned for index in started
                ):
                    continue
                deadlines = [
                    stamp + timeout
                    for index, (stamp, _) in started.items()
                    if index not in abandoned
                ]
                delay = max(0.0, min(deadlines) - time.monotonic()) if deadlines else None
                lock.wait(timeout=delay)

    def _execute(
        self,
        rule: Rule,
        ctx: DataContext,
        invocation: Invocation | None = None,
        *,
        finish_invocation: bool = True,
        control: RunControl | None = None,
        emit: bool = True,
    ) -> list[RuleResult]:
        """Run one rule as one progress step, honouring a cancellation request.

        A rule that has not started when the run is cancelled, or that ends with
        an error after it, is reported as not assessed because of the
        cancellation rather than because of the interrupt it caused.
        """
        if control is None:
            return self._invoke(rule, ctx, invocation, finish_invocation=finish_invocation)
        if control.cancelled:
            results = [self._cancelled(rule)]
        else:
            if emit:
                control.step_started((rule.rule_id,))
            results = self._invoke(
                rule, ctx, invocation, finish_invocation=finish_invocation, control=control
            )
            if control.cancelled and any(result.status is Status.ERROR for result in results):
                results = [self._cancelled(rule)]
        if emit:
            control.step_finished((rule.rule_id,), None, results)
        return results

    def _cancelled(self, rule: Rule) -> RuleResult:
        return self._result(
            rule, Status.ERROR, CANCELLED_MESSAGE, severity=max(rule.severity, Severity.ERROR)
        )

    def _invoke(
        self,
        rule: Rule,
        ctx: DataContext,
        invocation: Invocation | None = None,
        *,
        finish_invocation: bool = True,
        control: RunControl | None = None,
    ) -> list[RuleResult]:
        """Isolate the complete invocation, including entry and resource release."""
        began = False
        results: list[RuleResult] = []
        try:
            ctx.begin_rule(invocation)
            began = True
            if control is not None and control.cancelled:
                # Cancelled between the caller's check and this rule's start.
                return [self._cancelled(rule)]
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
                        ctx.release(ctx._execution_plan.complete(rule.key))
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
                    if finish_invocation:
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
            if rule.binding is not None:
                # A configured check item: the user intended this check, so it is blocked.
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
            # A generator rule runs while it is consumed: keep that inside the guard.
            findings = [outcome] if isinstance(outcome, Finding) else list(outcome)
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
        except RULE_FAILURES as exc:
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

        locator = RowLocator(ctx)
        if not findings:
            return [self._result(rule, Status.SKIPPED, "Rule reported nothing.", elapsed)]
        # Time is measured per rule; attribute it to the first result only.
        results = []
        for index, finding in enumerate(findings):
            try:
                if not isinstance(finding, Finding):
                    raise TypeError("each rule result must be a Finding")
                # Subclasses may override construction hooks. Reapply the base
                # verdict/count contract at the engine's admission boundary.
                Finding.__post_init__(finding)
                finding = self._locate_evidence(finding, locator)
                severity = None
                if not finding.assessed:
                    status = Status.ERROR
                    severity = max(finding.severity or rule.severity, Severity.ERROR)
                elif finding.skipped:
                    status = Status.SKIPPED
                else:
                    status = Status.PASS if finding.passed else Status.FAIL
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

    def _locate_evidence(self, finding: Finding, locator: RowLocator) -> Finding:
        """Admit a raw prefix before provenance work, preserving the full count.

        Location metadata is included in the final byte check in ``_result``.
        This preflight avoids expanding arbitrary plugin output merely to drop
        it, and never changes a plugin's frame or count semantics.
        """
        affected = finding.affected
        if affected is None:
            return finding
        count = len(affected)
        limit = min(count, self.settings.engine.max_affected_records)
        remaining = self.settings.engine.max_evidence_bytes
        admitted = 0
        while admitted < limit and remaining:
            raw = (
                affected.slice(admitted, 1)
                if isinstance(affected, pl.DataFrame)
                else affected[admitted]
            )
            size = (
                int(raw.estimated_size()) if isinstance(raw, pl.DataFrame) else retained_size(raw)
            )
            if size > remaining:
                break
            remaining -= size
            admitted += 1
        prefix = (
            affected.head(admitted)
            if isinstance(affected, pl.DataFrame)
            else [affected[index] for index in range(admitted)]
        )
        message = finding.message
        if admitted < limit:
            message += " Evidence was truncated or omitted: engine.max_evidence_bytes."
        bounded = replace(
            finding,
            affected=prefix,
            message=message,
            affected_count=finding.affected_count if finding.affected_count is not None else count,
        )
        return locator.locate(bounded)

    def _tags(self, rule: Rule, finding: Finding | None) -> list[str]:
        """The explicit tags of the check item a result reports on."""
        if rule.binding is not None:
            return list(rule.binding.tags)
        if finding is None or not finding.check_id:
            return []
        key = row_item_key(finding.check_id, rule.item_name(finding), finding.datasets)
        if key is None:
            return []
        if self._tag_index is None:
            self._tag_index = item_tags(self.settings)
        return list(self._tag_index.get(key, ()))

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
        coverage = (
            _coverage_labels(raw_metrics["passed_checks"])
            if "passed_checks" in raw_metrics
            else None
        )
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
            message += EVIDENCE_OMITTED
        message = _bounded_diagnostic(redact_text(message), max(0, byte_limit - used))
        if coverage is not None:
            metrics["passed_checks"] = _jsonable(coverage)
        complete = raw_metrics.get("duplicate_positions_complete")
        evidence_truncated = bool(raw_metrics.get("duplicate_positions_truncated"))
        if finding and finding.affected is not None and count > len(bounded_records):
            complete, evidence_truncated = False, True
        if len(bounded_records) != len(records) or sample_truncated:
            complete, evidence_truncated = False, True
        if metrics_dropped:
            evidence_truncated = True
        if status is Status.FAIL and not count:
            complete = False
        return RuleResult(
            rule_id=rule.rule_id,
            name=rule.name,
            description=(rule.binding.description if rule.binding else None) or rule.description,
            category=rule.category,
            check=finding.check if finding else None,
            check_name=rule.check_name,
            item_name=rule.item_name(finding),
            tags=self._tags(rule, finding),
            check_id=finding.check_id if finding and finding.check_id else rule.rule_id,
            columns=list(finding.columns if finding else ()),
            parameters=redact(
                finding.parameters
                if finding and finding.parameters
                else rule.params.model_dump(mode="python", exclude_defaults=True)
            ),
            checks=[check_identity(rule.rule_id, **redact(dict(check))) for check in finding.checks]
            if finding and finding.checks
            else [],
            status=status,
            severity=severity or (finding.severity if finding else None) or rule.severity,
            # Messages often quote errors, which may quote connection strings or URLs.
            message=redact_text(message),
            affected_records=bounded_records,
            affected_count=count,
            affected_count_exact=exact_count(
                status, count, finding.affected_count_exact if finding else True
            ),
            evidence_complete=complete,
            evidence_truncated=evidence_truncated,
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


@functools.cache
def _builtin_distribution() -> tuple[bool, str | None]:
    """Canonical metadata for built-ins, without scanning unrelated distributions.

    Some editable installations do not declare their top-level packages. Keep
    their provenance unknown, as with ``packages_distributions()``. A checkout
    without canonical metadata retains the generic lookup for other providers.
    """
    try:
        distribution = importlib.metadata.distribution("qavf")
        declared = (distribution.read_text("top_level.txt") or "").split()
        claims_package = (
            "qavf" in declared
            if declared
            else any(
                len(path.parts) > 1 and path.parts[0] == "qavf" for path in distribution.files or ()
            )
        )
        name = distribution.metadata["Name"]
        return True, f"{name}=={distribution.version}" if claims_package and name else None
    except Exception:
        # Missing/unreadable optional metadata keeps the established fallback.
        return False, None


def _is_builtin_rule(rule_type: type[Rule]) -> bool:
    """Recognize an actual exported class in this package's built-in modules."""
    if not rule_type.__module__.startswith("qavf.rules."):
        return False
    module = sys.modules.get(rule_type.__module__)
    if module is None or getattr(module, rule_type.__name__, None) is not rule_type:
        return False
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not isinstance(origin, str):
        return False
    try:
        package = Path(__file__).resolve().parent.parent
        expected = package.joinpath(*rule_type.__module__.split(".")[1:])
        return Path(origin).resolve() in (
            expected.with_suffix(".py").resolve(),
            (expected / "__init__.py").resolve(),
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _rule_distribution(rule_type: type[Rule]) -> str | None:
    if _is_builtin_rule(rule_type):
        available, distribution = _builtin_distribution()
        if available:
            return distribution
    return _distribution(rule_type.__module__)


def _rule_info(rule: Rule) -> RuleInfo:
    module = type(rule).__module__
    return RuleInfo(
        rule_id=rule.rule_id,
        check_name=rule.check_name,
        item_name=rule.binding.name if rule.binding else None,
        severity=rule.severity,
        params=redact(rule.params.model_dump(mode="python")),
        module=module,
        distribution=_rule_distribution(type(rule)),
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


def _coverage_labels(value: Any) -> list[str]:
    """Validate the reserved outcome ledger before exact requirement matching."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("Finding.metrics.passed_checks must be a sequence of nonempty strings")
    if any(not isinstance(label, str) or not label.strip() for label in value):
        raise TypeError("Finding.metrics.passed_checks must contain only nonempty strings")
    return list(value)
