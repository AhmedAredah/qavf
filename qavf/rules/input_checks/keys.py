"""Primary keys, alternate keys and duplicate-row contracts (INP-004)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar, Literal, TypedDict

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.rules._conditional_keys import ConditionalKeySource
from qavf.rules._contracts import contract_error
from qavf.rules._duplicate_rows import DuplicateRows
from qavf.rules._execution import (
    key_relation,
    observed_keys,
)
from qavf.rules._grouped_values import grouped_values
from qavf.rules._key_summary import KeySelection
from qavf.rules._key_values import CanonicalKeyValues
from qavf.rules._row_summary import row_aliases, summarize_rows
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import (
    absent_column_hint,
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


class _KeyIdentity(TypedDict):
    check_id: str
    columns: tuple[str, ...]
    parameters: dict[str, Any]


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
        exempt_na: dict[str, int] = {}
        identities: list[dict[str, Any]] = []
        for dataset, schema in schemas.items():
            first_identity = len(identities)
            first_failure = len(failures)
            try:
                if not (
                    schema.primary_key
                    or schema.unique_keys
                    or schema.unique_rows
                    or schema.grouped_values
                ):
                    continue
                snapshot = None
                conditional = any(key.where for key in schema.alternate_keys)
                if (conditional or schema.grouped_values) and not (
                    ctx.is_materialized(dataset) or ctx.repeatable_scan(dataset)
                ):
                    # Conditions may read non-key columns. Keep volatile inputs
                    # coherent across counts and original-row evidence.
                    snapshot = ctx.get(dataset)
                if (
                    schema.unique_rows
                    and (schema.primary_key or schema.unique_keys)
                    and not ctx.is_materialized(dataset)
                    and not ctx.repeatable_scan(dataset)
                ):
                    # Whole-row equality needs every column. Observe volatile inputs
                    # once so all key contracts describe that same complete snapshot.
                    snapshot = ctx.get(dataset)
                contracts = [
                    *([schema.primary_key] if schema.primary_key else []),
                    *(key.columns for key in schema.alternate_keys),
                ]
                if (
                    not schema.unique_rows
                    and len(contracts) > 1
                    and not ctx.is_materialized(dataset)
                    and not ctx.repeatable_scan(dataset)
                ):
                    columns = list(dict.fromkeys(c for key in contracts for c in key))
                    columns = [c for c in columns if c in ctx.schema(dataset)]
                    if columns:
                        with ctx.observe(dataset, columns) as observed:
                            self._summarized_keys(
                                ctx,
                                dataset,
                                schema,
                                failures,
                                exempt,
                                exempt_na,
                                identities,
                                observation=observed,
                            )
                    else:
                        self._summarized_keys(
                            ctx, dataset, schema, failures, exempt, exempt_na, identities
                        )
                else:
                    self._summarized_keys(
                        ctx,
                        dataset,
                        schema,
                        failures,
                        exempt,
                        exempt_na,
                        identities,
                        snapshot=snapshot,
                    )
                del snapshot
                if schema.unique_rows and self._unique_rows(ctx, dataset, failures):
                    identities.append(
                        {
                            "check_id": "key.unique_rows",
                            "datasets": (dataset,),
                            "columns": tuple(ctx.schema(dataset)),
                            "parameters": {},
                        }
                    )
                failures.extend(
                    grouped_values(ctx, dataset, contract) for contract in schema.grouped_values
                )
                if len(identities) > first_identity:
                    if len(failures) == first_failure:
                        passed.append(dataset)
                    else:
                        # A partial success names the assessed contract, never
                        # the dataset as though every configured key passed.
                        passed.extend(
                            f"{dataset}: {identity['check_id']} ({', '.join(identity['columns'])})"
                            for identity in identities[first_identity:]
                        )
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
            f"{plural(len(identities), 'key contract')} passed.",
            "No primary or unique keys are configured.",
            summary_metrics={
                **({"rows_exempt_null_unique_keys": exempt} if exempt else {}),
                **({"exempt_na_rows": exempt_na} if exempt_na else {}),
            },
            checks=tuple(identities),
            **(identities[0] if len(identities) == 1 else {"check_id": "key.summary"}),
        )

    def _summarized_keys(
        self,
        ctx: DataContext,
        dataset: str,
        schema: SchemaSpec,
        failures: list[Finding],
        exempt: dict[str, int],
        exempt_na: dict[str, int],
        identities: list[dict[str, Any]],
        *,
        snapshot: pl.DataFrame | None = None,
        observation: Observation | None = None,
    ) -> bool:
        """Check key contracts using source-side frequencies, never unrelated columns."""
        clean = True
        actual = pl.DataFrame(schema=ctx.schema(dataset))
        contracts: list[
            tuple[list[str], bool, Literal["forbid", "equal", "exempt"], str | None, str | None]
        ] = [(schema.primary_key, True, "forbid", None, None)] if schema.primary_key else []
        contracts.extend(
            (key.columns, False, key.nulls, key.where, key.name) for key in schema.alternate_keys
        )
        for columns, primary, null_policy, where, name in contracts:
            first_failure = len(failures)
            identity: _KeyIdentity = {
                "check_id": "key.primary" if primary else "key.unique",
                "columns": tuple(columns),
                "parameters": {
                    "nulls": null_policy,
                    "rows_per_key": schema.duplicate_rows_per_key,
                    **({"where": where} if where is not None else {}),
                    **({"name": name} if name is not None else {}),
                },
            }
            if self._absent(
                dataset, actual, columns, "Primary key" if primary else "Unique key", failures
            ):
                failures[-1] = replace(failures[-1], **identity)
                clean = False
                continue

            try:
                frequency = fresh_internal("occurrences", actual.columns)
                selected = snapshot
                if selected is None and ctx.is_materialized(dataset):
                    # In particular, keep existing IPC spills scoped and projected;
                    # ctx.scan may otherwise prefer a caller's original lazy plan.
                    selected = ctx.project(dataset, columns)
                key_values = CanonicalKeyValues(ctx, dataset, columns)
                label = f"{dataset} ({', '.join(columns)})"
                scoped = ConditionalKeySource.build(ctx, dataset, columns, where) if where else None
                na_rows = 0
                if key_values.na_policies:
                    if scoped is not None:
                        na_rows = scoped.exempt_na_rows()
                    elif observation is not None:
                        na_rows = int(
                            observation.collect(
                                observation.scan().select(key_values.na_expr().sum())
                            ).item()
                        )
                    elif selected is not None:
                        na_rows = int(selected.select(key_values.na_expr().sum()).item())
                    elif ctx.execution_kind(dataset) == "native_sql":
                        na_rows = int(
                            ctx.sql(
                                f"SELECT count(*) FILTER (WHERE {key_values.na_sql()}) "
                                f"FROM {quote_identifier(dataset)}",
                                datasets=(dataset,),
                                normalize_missing=False,
                            ).item()
                        )
                    else:

                        def na_reduction(
                            plan: pl.LazyFrame, values: CanonicalKeyValues = key_values
                        ) -> pl.LazyFrame:
                            return plan.select(values.na_expr().sum())

                        na_rows = int(ctx.reduce(dataset, na_reduction).item())
                    if na_rows:
                        exempt_na[label] = na_rows
                relation = (
                    scoped.relation(frequency)
                    if scoped is not None
                    else (
                        observed_keys(
                            ctx,
                            dataset,
                            observation.scan(),
                            columns,
                            frequency,
                            collector=observation.collect,
                            exclude_na=True,
                        )
                        if observation is not None
                        else key_relation(
                            ctx, dataset, columns, frequency, _snapshot=selected, exclude_na=True
                        )
                    )
                )
                missing, _ = _key_missing(pl.DataFrame(schema=relation.schema), columns, None)
                missing_sql = " OR ".join(f"{quote_identifier(c)} IS NULL" for c in columns)
                valid_sql = "true" if null_policy == "equal" else f"NOT ({missing_sql})"
                valid = pl.lit(True) if null_policy == "equal" else ~missing
                summaries = relation.summarize_many(
                    {
                        "all": KeySelection(),
                        "nulls": KeySelection(
                            missing_sql,
                            missing,
                            ctx.evidence_limit if null_policy == "forbid" else 0,
                        ),
                        "valid": KeySelection(valid_sql, valid),
                        "duplicates": KeySelection(
                            f"({valid_sql}) AND {quote_identifier(frequency)} > 1",
                            valid & (pl.col(frequency) > 1),
                            ctx.evidence_limit,
                        ),
                    }
                )
                all_keys, nulls = summaries["all"], summaries["nulls"]
                valid_keys, dupes = summaries["valid"], summaries["duplicates"]
                null_count, rows = nulls.rows, all_keys.rows + na_rows
                if null_policy == "forbid" and null_count:
                    clean = False
                    failures.append(
                        Finding.fail(
                            f"{plural(null_count, 'row')} of {rows:,} in {label} "
                            "have a null key value.",
                            check=f"{dataset}: null keys" if primary else f"{label}: null keys",
                            affected=temporal_evidence(
                                present(nulls.sample, {frequency: "occurrences"})
                            ),
                            affected_count=null_count,
                            total_records=rows,
                            unit="rows",
                            datasets=(dataset,),
                            **{**identity, "check_id": identity["check_id"] + ".nulls"},
                        )
                    )
                elif null_policy == "exempt" and null_count:
                    exempt[label] = null_count
                if not dupes.keys:
                    if len(failures) == first_failure:
                        identities.append({**identity, "datasets": (dataset,)})
                    continue
                clean = False
                involved = dupes.rows
                positions = fresh_internal("rows", [*relation.schema, frequency])
                lookup = (
                    scoped.positions(positions, schema.duplicate_rows_per_key)
                    if scoped is not None
                    else DuplicateRows.of(
                        ctx,
                        dataset,
                        columns,
                        positions,
                        selected=selected,
                        observation=observation,
                        limit=schema.duplicate_rows_per_key,
                    )
                )
                evidence = lookup.attach(dupes.sample, frequency)
                failures.append(
                    Finding.fail(
                        f"{plural(dupes.keys, 'duplicated key')} of {valid_keys.keys:,} "
                        f"in {label}, "
                        f"shared by {plural(involved, 'row')} of {rows:,}.",
                        check=dataset if primary else f"{dataset}: unique ({', '.join(columns)})",
                        affected=temporal_evidence(
                            present(evidence, {frequency: "occurrences", positions: "rows"})
                        ),
                        affected_count=dupes.keys,
                        total_records=valid_keys.keys,
                        unit="keys",
                        affected_rows=involved,
                        metrics={
                            "duplicated_keys": dupes.keys,
                            "rows_involved": involved,
                            "surplus_rows": involved - dupes.keys,
                            **lookup.metrics,
                        },
                        datasets=(dataset,),
                        **identity,
                    )
                )
            except Exception as exc:
                clean = False
                label = dataset if primary else f"{dataset}: unique ({', '.join(columns)})"
                failures.append(replace(contract_error(label, exc, (dataset,)), **identity))

        return clean

    @staticmethod
    def _absent(
        dataset: str, frame: pl.DataFrame, columns: list[str], what: str, failures: list[Finding]
    ) -> bool:
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            failures.append(
                Finding.fail(
                    f"{what} of {dataset} names column(s) that do not exist: {', '.join(absent)}"
                    f"{''.join(absent_column_hint(c, frame.columns) for c in absent)}.",
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
                check_id="key.unique_rows",
                columns=tuple(schema),
                parameters={},
            )
        )
        return False
