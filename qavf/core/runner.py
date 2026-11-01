"""Small native orchestration; source adaptation and check compilation are separate."""

from __future__ import annotations

import functools
import tempfile
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

import duckdb

from qavf._version import __version__
from qavf.core.checks import DatasetChecks
from qavf.core.configuration import prepare
from qavf.core.results import ResultCollector
from qavf.core.sources import InputReader
from qavf.engine.severity import Status
from qavf.redaction import redact, redact_text
from qavf.reports.models import DatasetInfo, QAReport, RunMetadata
from qavf.sources.base import SourceNotFoundError
from qavf.sources.memory import reservation_is_stricter
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import resolve_sql_paths, validate_query
from qavf.sources.sql import sql_literal as lit


def _dataset_rules(spec):
    """Configured work to mark unassessed when its source cannot be evaluated."""
    if spec.schema_spec is not None:
        yield "INP-002"
        if spec.schema_spec.columns:
            yield "INP-003"
        if (
            spec.schema_spec.primary_key
            or spec.schema_spec.unique_keys
            or spec.schema_spec.unique_rows
            or spec.schema_spec.grouped_values
        ):
            yield "INP-004"
        if spec.schema_spec.na_groups or any(
            c.na_values for c in spec.schema_spec.columns.values()
        ):
            yield "INP-009"
    if spec.foreign_keys:
        yield "INP-005"
    if spec.expressions:
        yield "INP-006"


def _blocked(out, name, rule_ids, exc, *, stage):
    for rule_id in rule_ids:
        out.add(
            rule_id,
            stage,
            name,
            status=Status.ERROR,
            check_id=f"{rule_id}.{stage}",
            parameters={"stage": stage, "error_type": type(exc).__name__},
            message=f"Not assessed: {redact_text(str(exc))}",
        )


def _fit_report(report, limit):
    """Remove optional details before refusing an oversized outcome ledger."""
    original_size = report.json_size_bytes()
    if original_size <= limit:
        return report
    results = [
        result.model_copy(
            update={
                "affected_records": [],
                "metrics": {},
                "message": "",
                "evidence_complete": False if result.affected_count else result.evidence_complete,
                "evidence_truncated": True,
            }
        )
        for result in report.results
    ]
    metadata = report.metadata.model_copy(
        update={
            "config": {},
            "extensions": {},
            "query_plans": [],
            "resource_usage": {
                **report.metadata.resource_usage,
                "report_details_omitted": True,
                "original_report_bytes": original_size,
            },
        }
    )
    report = report.model_copy(update={"results": results, "metadata": metadata})
    if report.json_size_bytes() > limit:
        from qavf.execution.errors import ExecutionLimitError

        raise ExecutionLimitError(
            "report outcome ledger exceeds max_report_bytes after optional evidence removal"
        )
    return report


class NativeRunner:
    """Own one run; delegate source, planning, policy and evidence responsibilities."""

    def __init__(self, settings, inputs):
        self.settings = settings
        self.inputs = inputs
        self.datasets = []
        self.loaded = []
        self.unavailable = {}
        self.timings = {}

    def _stage(self, out, name, rule_ids, execute):
        from qavf.core.execution import Deadline

        if not any(out.enabled(rule, name) for rule in rule_ids):
            return
        start, offset = perf_counter(), len(out.results)
        deadline = Deadline(out.connection, self.settings.engine.rule_timeout_s)
        try:
            with deadline:
                execute()
        except Exception as exc:
            from qavf.optional import OptionalDependencyError

            if isinstance(exc, OptionalDependencyError):
                raise
            _blocked(out, name, rule_ids, exc, stage="execution")
        if deadline.expired:
            del out.results[offset:]
            _blocked(
                out,
                name,
                rule_ids,
                TimeoutError("native rule stage exceeded rule_timeout_s"),
                stage="timeout",
            )
        elapsed = perf_counter() - start
        self.timings[f"{name}:{','.join(rule_ids)}:{getattr(execute, '__name__', 'check')}"] = (
            elapsed
        )
        number = len(out.results) - offset
        for index in range(offset, len(out.results)):
            out.results[index] = out.results[index].model_copy(
                update={
                    "execution_time_s": elapsed / number,
                }
            )

    def _release(self, reader, checks):
        for executor in checks.values():
            for name in (executor.data.name, executor.expression_table, executor.typed):
                reader.connection.execute(f"DROP VIEW IF EXISTS {qi(name)}")
            reader.release(executor.data)

    def _group(self, names, out, reader, query_inputs, parents):
        checks = {}
        for name in names:
            value, spec = self.inputs[name], self.settings.datasets[name]
            active = tuple(
                rule for rule in ("INP-001", *_dataset_rules(spec)) if out.enabled(rule, name)
            )
            if not active and not query_inputs and name not in parents:
                continue
            before = perf_counter()
            try:
                loaded = reader.load(name, value, spec)
                self.loaded.append(loaded)
                metadata = loaded.metadata
                self.datasets.append(
                    DatasetInfo(
                        name=name,
                        rows=loaded.rows,
                        columns=list(loaded.original_columns),
                        dtypes=loaded.original_types,
                        load_time_s=perf_counter() - before,
                        source=metadata.get("source"),
                        loaded_at=metadata.get("loaded_at"),
                        fingerprint=metadata.get("fingerprint"),
                        renamed=metadata.get("renamed", {}),
                        notes={
                            key: value
                            for key, value in metadata.items()
                            if key
                            not in {
                                "source",
                                "loaded_at",
                                "fingerprint",
                                "renamed",
                            }
                        },
                    )
                )
                out.add("INP-001", "load", name, total=loaded.rows)
            except Exception as exc:
                from qavf.optional import OptionalDependencyError

                if isinstance(exc, OptionalDependencyError):
                    raise
                self.unavailable[name] = redact_text(str(exc))
                if not spec.required and isinstance(exc, SourceNotFoundError):
                    out.add(
                        "INP-001",
                        "load",
                        name,
                        status=Status.SKIPPED,
                        parameters={"required": False},
                        message="Optional dataset is absent",
                    )
                else:
                    _blocked(out, name, active, exc, stage="load")
                continue
            if (
                not any(out.enabled(rule, name) for rule in _dataset_rules(spec))
                and not query_inputs
                and name not in parents
            ):
                reader.release(loaded)
                continue
            try:
                checks[name] = DatasetChecks(out, loaded, spec)
            except Exception as exc:
                _blocked(out, name, _dataset_rules(spec), exc, stage="compile")
                reader.release(loaded)
        for name, executor in checks.items():
            if executor.schema is not None:
                for ids, method in (
                    (("INP-002",), executor.shape),
                    (("INP-002", "INP-003"), executor.columns),
                    (("INP-004",), executor.keys),
                    (("INP-009",), executor.applicability),
                ):
                    self._stage(out, name, ids, method)
            self._stage(out, name, ("INP-005",), functools.partial(executor.foreign_keys, checks))
            self._stage(out, name, ("INP-006",), executor.expressions)
        return checks

    def _assertions(self, out):
        if not out.enabled("INP-007"):
            return
        for assertion in self.settings.assertions:
            snapshot = "__qavf_assertion_" + uuid4().hex
            self._stage(
                out, "*", ("INP-007",), functools.partial(self._assertion, out, assertion, snapshot)
            )

    def _assertion(self, out, assertion, snapshot):
        """Run one SQL assertion into a temporary snapshot and record its result."""
        try:
            sql = resolve_sql_paths(
                validate_query(assertion.sql),
                self.settings.base_dir,
                known_tables=self.settings.datasets,
            )
            out.connection.execute(f"CREATE TEMP TABLE {qi(snapshot)} AS {sql}")
            count = out.connection.execute(f"SELECT count(*) FROM {qi(snapshot)}").fetchone()[0]
            records = (
                out.records(
                    f"SELECT * FROM {qi(snapshot)} "
                    f"LIMIT {self.settings.engine.max_affected_records}"
                )
                if count
                else []
            )
            out.add(
                "INP-007",
                "assertion",
                "*",
                count=count,
                records=records,
                parameters={"name": assertion.name, "sql": assertion.sql},
                severity=assertion.severity,
            )
        except Exception as exc:
            out.add(
                "INP-007",
                "assertion",
                "*",
                status=Status.ERROR,
                message=redact_text(str(exc)),
                parameters={"name": assertion.name, "sql": assertion.sql},
            )
        finally:
            out.connection.execute(f"DROP TABLE IF EXISTS {qi(snapshot)}")

    def run(self):
        from qavf.audit_codec import canonical_digest
        from qavf.core.connections import SourceConnections
        from qavf.core.planning import NativePlan
        from qavf.core.provenance import environment, requirements
        from qavf.engine.thread_budget import connection_threads
        from qavf.reports.models import RuleInfo
        from qavf.resource_plan import ResourcePlan
        from qavf.text_policies import register_sql_functions, requires_sql_functions

        settings = self.settings
        started = datetime.now(UTC)
        resources = ResourcePlan.build(settings, frames=self.inputs, requested_workers=1)
        usage = {
            "backend": "duckdb",
            "duckdb_version": duckdb.__version__,
            "resource_plan": resources.diagnostics(),
        }
        parent = (
            settings.resolve(settings.engine.temp_directory)
            if settings.engine.temp_directory
            else None
        )
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(prefix="qavf-", dir=parent) as temporary,
            duckdb.connect(config={"threads": 1}) as con,
        ):
            con.execute(f"SET temp_directory={lit(temporary)}")
            con.execute("SET preserve_insertion_order=true")
            con.execute("SET TimeZone='UTC'")
            if requires_sql_functions(settings):
                register_sql_functions(con)
            threads = connection_threads(
                settings.engine.native_threads, 1, memory_bytes=resources.memory_bytes
            )
            con.execute(f"SET threads={threads}")
            if settings.engine.duckdb_memory_limit:
                con.execute(f"SET memory_limit={lit(settings.engine.duckdb_memory_limit)}")
            if resources.memory_bytes is not None and reservation_is_stricter(
                settings.engine.duckdb_memory_limit, resources.memory_bytes
            ):
                con.execute(f"SET memory_limit={lit(str(resources.memory_bytes) + 'B')}")
            con.execute(f"SET max_temp_directory_size={lit(str(resources.spill_bytes) + 'B')}")
            usage.update(
                native_threads_per_connection=threads,
                native_memory_limit=con.execute(
                    "SELECT current_setting('memory_limit')"
                ).fetchone()[0],
                native_spill_limit_bytes=resources.spill_bytes,
                native_connection_count=1,
            )
            out = ResultCollector(con, settings)
            plan = NativePlan.build(settings, out, self.inputs)
            sources = SourceConnections(con, settings)
            reader = InputReader(
                con, settings, projections=plan.projections, query_resolver=sources.resolve
            )
            query_inputs = bool(settings.assertions and out.enabled("INP-007")) or any(
                spec.expressions and out.enabled("INP-006", name)
                for name, spec in settings.datasets.items()
            )
            parents = {
                fk.ref_dataset
                for name, spec in settings.datasets.items()
                if out.enabled("INP-005", name)
                for fk in spec.foreign_keys
            }
            parents.update(plan.source_parents)
            retained = {}
            for names in plan.groups:
                checks = self._group(names, out, reader, query_inputs, parents)
                if settings.assertions and out.enabled("INP-007"):
                    retained.update(checks)
                else:
                    self._release(reader, checks)
            self._assertions(out)
            for loaded in self.loaded:
                try:
                    reader.verify(loaded)
                except Exception as exc:
                    _blocked(out, loaded.name, ("INP-001",), exc, stage="source_integrity")
            self._release(reader, retained)
        config = redact(settings.model_dump(mode="python", by_alias=True))
        results = requirements(out.results, settings.engine.require)
        rules = [
            RuleInfo(
                rule_id=rule_id,
                severity=next(result.severity for result in results if result.rule_id == rule_id),
                params=settings.rules[rule_id].params if rule_id in settings.rules else {},
                module="qavf.core.checks",
                distribution=f"qavf=={__version__}",
            )
            for rule_id in dict.fromkeys(result.rule_id for result in results)
        ]
        usage.update(
            stage_timings_s=self.timings,
            timing_scope="stage elapsed time divided among its results",
            snapshot_lifetime="independent dependency component",
            dependency_components=len(plan.groups),
        )
        report = QAReport(
            metadata=RunMetadata(
                run_id=str(uuid4()),
                project=settings.project,
                qavf_version=__version__,
                started_at=started,
                finished_at=datetime.now(UTC),
                datasets=self.datasets,
                config_path=str(settings.source_path) if settings.source_path else None,
                fail_on=settings.engine.fail_on,
                fail_on_incomplete=settings.engine.fail_on_incomplete,
                config=config,
                config_digest=canonical_digest(config),
                environment=environment(),
                plan_digest=canonical_digest([result.identity.model_dump() for result in results]),
                rules=rules,
                unavailable_datasets=self.unavailable,
                unused_datasets=[
                    name
                    for name in settings.datasets
                    if name not in {d.name for d in self.datasets} and name not in self.unavailable
                ],
                resource_usage=usage,
                extensions=redact(settings.extensions),
                parallel=False,
                max_workers=1,
            ),
            results=results,
        )
        return _fit_report(report, settings.engine.max_report_bytes)


def validate_core(data=None, *, schema=None, config=None, resources=None, profile=None):
    settings, inputs = prepare(data, schema, config, resources, profile)
    if settings.engine.execution_mode == "isolated":
        from qavf.core.execution import validate_isolated

        return validate_isolated(settings, inputs)
    return NativeRunner(settings, inputs).run()
