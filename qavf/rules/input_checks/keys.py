"""Primary keys, alternate keys and duplicate-row contracts (INP-004)."""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.rules._execution import (
    key_relation,
)
from qavf.rules._key_summary import KeySelection
from qavf.rules._row_summary import row_aliases, summarize_rows
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import (
    fresh_internal,
    outcome,
    plural,
    present,
)
from qavf.settings import (
    SchemaSpec,
)
from qavf.sources.sql import quote_identifier

from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _key_missing,
    _schemas,
    _SelectsDatasets,
)


class DuplicateKeyRule(_SelectsDatasets):
    """Primary keys identify rows uniquely."""

    rule_id: ClassVar[str] = "INP-004"
    name: ClassVar[str] = "Unique primary keys"
    description: ClassVar[str] = "Each dataset's primary-key columns identify its rows uniquely."
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Duplicate keys double-count values when tables are joined. De-duplicate at the "
        "source, or add the missing dimension to the primary key."
    )
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        schemas, blocked = _schemas(ctx, self.params.datasets, "key check")
        failures += blocked
        exempt: dict[str, int] = {}
        for dataset, schema in schemas.items():
            try:
                if not (schema.primary_key or schema.unique_keys or schema.unique_rows):
                    continue
                snapshot = None
                if (
                    schema.unique_rows
                    and (schema.primary_key or schema.unique_keys)
                    and not ctx.is_materialized(dataset)
                    and not ctx.repeatable_scan(dataset)
                ):
                    # Whole-row equality needs every column. Observe volatile inputs
                    # once so all key contracts describe that same complete snapshot.
                    snapshot = ctx.get(dataset)
                clean = self._summarized_keys(
                    ctx, dataset, schema, failures, exempt, snapshot=snapshot
                )
                del snapshot
                if schema.unique_rows:
                    clean &= self._unique_rows(ctx, dataset, failures)
                if clean:
                    passed.append(dataset)
            except Exception as exc:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: keys on {dataset}: {type(exc).__name__}: {exc}.",
                        check=dataset,
                        datasets=(dataset,),
                    )
                )
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'dataset')} with unique keys.",
            "No primary or unique keys are configured.",
            summary_metrics={"rows_exempt_null_unique_keys": exempt} if exempt else {},
        )

    def _summarized_keys(
        self,
        ctx: DataContext,
        dataset: str,
        schema: SchemaSpec,
        failures: list[Finding],
        exempt: dict[str, int],
        *,
        snapshot: pl.DataFrame | None = None,
    ) -> bool:
        """Check key contracts using source-side frequencies, never unrelated columns."""
        clean = True
        actual = pl.DataFrame(schema=ctx.schema(dataset))
        contracts = [(schema.primary_key, True)] if schema.primary_key else []
        contracts.extend((key, False) for key in schema.unique_keys)
        for columns, primary in contracts:
            if self._absent(
                dataset, actual, columns, "Primary key" if primary else "Unique key", failures
            ):
                clean = False
                continue

            frequency = fresh_internal("occurrences", actual.columns)
            selected = snapshot
            if selected is None and ctx.is_materialized(dataset):
                # In particular, keep existing IPC spills scoped and projected;
                # ctx.scan may otherwise prefer a caller's original lazy plan.
                selected = ctx.project(dataset, columns)
            relation = key_relation(ctx, dataset, columns, frequency, _snapshot=selected)
            missing, _ = _key_missing(pl.DataFrame(schema=relation.schema), columns, None)
            missing_sql = " OR ".join(f"{quote_identifier(c)} IS NULL" for c in columns)
            summaries = relation.summarize_many(
                {
                    "all": KeySelection(),
                    "nulls": KeySelection(
                        missing_sql, missing, ctx.evidence_limit if primary else 0
                    ),
                    "valid": KeySelection(f"NOT ({missing_sql})", ~missing),
                    "duplicates": KeySelection(
                        f"NOT ({missing_sql}) AND {quote_identifier(frequency)} > 1",
                        ~missing & (pl.col(frequency) > 1),
                        ctx.evidence_limit,
                    ),
                }
            )
            all_keys, nulls = summaries["all"], summaries["nulls"]
            valid_keys, dupes = summaries["valid"], summaries["duplicates"]
            null_count, rows = nulls.rows, all_keys.rows
            label = f"{dataset} ({', '.join(columns)})"
            if primary and null_count:
                clean = False
                failures.append(
                    Finding.fail(
                        f"{plural(null_count, 'row')} of {rows:,} in {label} "
                        "have a null key value.",
                        check=f"{dataset}: null keys",
                        affected=temporal_evidence(
                            present(nulls.sample, {frequency: "occurrences"})
                        ),
                        affected_count=null_count,
                        total_records=rows,
                        unit="rows",
                        datasets=(dataset,),
                    )
                )
            elif null_count:
                exempt[label] = null_count
            if not dupes.keys:
                continue
            clean = False
            involved = dupes.rows
            failures.append(
                Finding.fail(
                    f"{plural(dupes.keys, 'duplicated key')} of {valid_keys.keys:,} in {label}, "
                    f"shared by {plural(involved, 'row')} of {rows:,}.",
                    check=dataset if primary else f"{dataset}: unique ({', '.join(columns)})",
                    affected=temporal_evidence(present(dupes.sample, {frequency: "occurrences"})),
                    affected_count=dupes.keys,
                    total_records=valid_keys.keys,
                    unit="keys",
                    affected_rows=involved,
                    metrics={
                        "duplicated_keys": dupes.keys,
                        "rows_involved": involved,
                        "surplus_rows": involved - dupes.keys,
                    },
                    datasets=(dataset,),
                )
            )
        return clean

    @staticmethod
    def _absent(
        dataset: str, frame: pl.DataFrame, columns: list[str], what: str, failures: list[Finding]
    ) -> bool:
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            failures.append(
                Finding.fail(
                    f"{what} of {dataset} names column(s) that do not exist: {', '.join(absent)}.",
                    check=dataset
                    if what == "Primary key"
                    else f"{dataset}: unique ({', '.join(columns)})",
                    affected=[{"dataset": dataset, "missing_key_column": c} for c in absent],
                    unit="columns",
                    datasets=(dataset,),
                )
            )
        return bool(absent)

    @staticmethod
    def _unique_rows(ctx: DataContext, dataset: str, failures: list[Finding]) -> bool:
        """Count every repeated raw row and keep only bounded original-row evidence."""
        schema = ctx.schema(dataset)
        row, problem = row_aliases(list(schema))
        selected = [*schema, row, problem]
        if not schema:
            return True
        if ctx.execution_kind(dataset) == "native_sql":
            q = quote_identifier
            columns = ", ".join(q(column) for column in schema)
            summary = summarize_rows(
                ctx,
                dataset,
                problem=problem,
                labels=("duplicate row",),
                sample_columns=selected,
                sample_order=row,
                sql="SELECT *, "
                f"CASE WHEN count(*) OVER (PARTITION BY {columns}) > 1 "
                f"THEN 'duplicate row' END AS {q(problem)} FROM "
                f"(SELECT *, row_number() OVER () - 1 AS {q(row)} FROM {q(dataset)}) source",
            )
        else:
            summary = summarize_rows(
                ctx,
                dataset,
                problem=problem,
                labels=("duplicate row",),
                sample_columns=selected,
                build=lambda source: source.with_row_index(row).with_columns(
                    pl.when(pl.struct(list(schema)).is_duplicated())
                    .then(pl.lit("duplicate row"))
                    .otherwise(None)
                    .alias(problem)
                ),
            )
        count = summary.invalid
        if not count:
            return True
        failures.append(
            Finding.fail(
                f"{plural(count, 'row')} of {summary.rows:,} in {dataset} "
                f"{'is' if count == 1 else 'are'} identical to another row in every column.",
                check=f"{dataset}: duplicate rows",
                affected=temporal_evidence(
                    present(summary.sample, {row: "_row", problem: "violates"})
                ),
                affected_count=count,
                total_records=summary.rows,
                unit="rows",
                datasets=(dataset,),
            )
        )
        return False
