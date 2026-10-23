"""Foreign-key match policies, cardinality and orphan contracts (INP-005)."""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.rules._execution import (
    KeyBackend,
    KeyRelation,
    align_key_relations,
    key_relation,
    observed_keys,
    paired_key_backend,
)
from qavf.rules._key_summary import KeySelection, KeySummary
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import (
    fresh_internal,
    outcome,
    plural,
    present,
)
from qavf.settings import (
    ForeignKeySpec,
    QAVFSettings,
)
from qavf.sources.sql import quote_identifier

from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _must_assess,
    _SelectsDatasets,
)


class ReferentialIntegrityRule(_SelectsDatasets):
    """Foreign-key values exist in the dataset they reference."""

    rule_id: ClassVar[str] = "INP-005"
    name: ClassVar[str] = "Referential integrity"
    description: ClassVar[str] = (
        "Every configured foreign-key value exists in the referenced dataset. Keys are "
        "compared exactly; key types that cannot be compared exactly are reported."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Orphaned keys are silently dropped by inner joins. Add the missing reference rows or "
        "correct the keys in the child dataset."
    )
    Params: ClassVar[type[RuleParams]] = DatasetSelection

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        selected = (
            set(self.params.datasets)
            if self.params.datasets is not None
            else set(settings.datasets)
        )
        for name in tuple(selected):
            spec = settings.datasets.get(name)
            if spec is not None:
                selected.update(fk.ref_dataset for fk in spec.foreign_keys)
        return DataRequirements(frozenset(selected))

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        selection = self.params.datasets
        failures: list[Finding] = []
        passed: list[str] = []
        exempt: dict[str, int] = {}
        for dataset, spec in ctx.specs.items():
            if selection is not None and dataset not in selection:
                continue
            for fk in spec.foreign_keys:
                label = f"{dataset}({', '.join(fk.columns)}) -> {fk.ref_dataset}"
                try:
                    if not ctx.has(dataset):
                        if _must_assess(ctx, dataset, selection):
                            failures.append(self._blocked(ctx, label, fk, dataset))
                        continue
                    if not ctx.has(fk.ref_dataset):
                        # The child has rows to check, so a missing parent always blocks the check.
                        failures.append(self._blocked(ctx, label, fk, fk.ref_dataset))
                        continue
                    child_columns, parent_columns = ctx.schema(dataset), ctx.schema(fk.ref_dataset)
                    if any(c not in child_columns for c in fk.columns) or any(
                        c not in parent_columns for c in fk.target_columns
                    ):
                        failures.append(
                            Finding.unassessed(
                                f"Not assessed: foreign key {label} refers to columns that do not "
                                f"exist.",
                                check=label,
                                severity=fk.severity,
                            )
                        )
                        continue
                    backend = paired_key_backend(ctx, dataset, fk.ref_dataset)
                    stable = all(
                        ctx.is_materialized(name) or ctx.repeatable_scan(name)
                        for name in (dataset, fk.ref_dataset)
                    )
                    if backend is not None and stable:
                        clean = self._relation_rules(
                            ctx, dataset, label, fk, failures, exempt, backend
                        )
                    else:
                        projections = {dataset: list(fk.columns)}
                        projections.setdefault(fk.ref_dataset, []).extend(fk.target_columns)
                        with ctx.projected_observations(projections) as observations:
                            frequency = fresh_internal("rows", child_columns)
                            parent_frequency = fresh_internal(
                                "rows", [*child_columns, *parent_columns, frequency]
                            )
                            child = observed_keys(
                                ctx,
                                dataset,
                                observations[dataset].scan(),
                                fk.columns,
                                frequency,
                                collector=observations.collect,
                            )
                            parent = observed_keys(
                                ctx,
                                fk.ref_dataset,
                                observations[fk.ref_dataset].scan(),
                                fk.target_columns,
                                parent_frequency,
                                collector=observations.collect,
                            )
                            clean = self._check_relations(
                                ctx, dataset, label, fk, failures, exempt, child, parent
                            )
                    if clean:
                        passed.append(label)
                except Exception as exc:
                    failures.append(
                        Finding.unassessed(
                            f"Not assessed: {label}: {type(exc).__name__}: {exc}.",
                            check=label,
                            severity=fk.severity,
                            datasets=(dataset, fk.ref_dataset),
                        )
                    )
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'foreign key')} fully resolved.",
            "No foreign keys are configured.",
            # Exempt rows follow the declared match policy, including on failure.
            summary_metrics={"rows_exempt_null_keys": exempt} if exempt else {},
        )

    @staticmethod
    def _relation_rules(
        ctx: DataContext,
        dataset: str,
        label: str,
        fk: ForeignKeySpec,
        failures: list[Finding],
        exempt: dict[str, int],
        backend: KeyBackend,
    ) -> bool:
        """Assess grouped relations without transferring complete key sets to Python."""
        frequency = fresh_internal("rows", ctx.schema(dataset))
        parent_frequency = fresh_internal(
            "rows",
            [*ctx.schema(dataset), *ctx.schema(fk.ref_dataset), frequency],
        )
        child = key_relation(ctx, dataset, fk.columns, frequency, backend=backend)
        parent = key_relation(
            ctx, fk.ref_dataset, fk.target_columns, parent_frequency, backend=backend
        )
        return ReferentialIntegrityRule._check_relations(
            ctx, dataset, label, fk, failures, exempt, child, parent
        )

    @staticmethod
    def _check_relations(
        ctx: DataContext,
        dataset: str,
        label: str,
        fk: ForeignKeySpec,
        failures: list[Finding],
        exempt: dict[str, int],
        child: KeyRelation,
        parent: KeyRelation,
    ) -> bool:
        """Judge one coherent child/parent observation on either execution backend."""
        frequency = child.frequency
        child_missing = pl.any_horizontal(pl.col(name).is_null() for name in fk.columns)
        child_all = pl.all_horizontal(pl.col(name).is_null() for name in fk.columns)
        missing_sql = " OR ".join(f"{quote_identifier(name)} IS NULL" for name in fk.columns)
        all_sql = " AND ".join(f"{quote_identifier(name)} IS NULL" for name in fk.columns)
        selections = {
            "total": KeySelection(),
            "nulls": KeySelection(
                all_sql if fk.match == "full" else missing_sql,
                child_all if fk.match == "full" else child_missing,
            ),
            "valid": KeySelection(f"NOT ({missing_sql})", ~child_missing),
        }
        if fk.match == "full":
            selections["partial"] = KeySelection(
                f"({missing_sql}) AND NOT ({all_sql})",
                child_missing & ~child_all,
                ctx.evidence_limit,
            )
        if fk.cardinality == "one_to_one":
            selections["duplicates"] = KeySelection(
                f"NOT ({missing_sql}) AND {quote_identifier(frequency)} > 1",
                ~child_missing & (pl.col(frequency) > 1),
                ctx.evidence_limit,
            )
        summaries = child.summarize_many(selections)
        total, nulls = summaries["total"], summaries["nulls"]
        if nulls.rows:
            exempt[label] = nulls.rows
        clean = True
        if fk.match == "full":
            partial = summaries["partial"]
            if partial.rows:
                clean = False
                failures.append(
                    _partial_finding(dataset, label, fk, partial, total.rows, frequency)
                )
        valid_child = child.filter(f"NOT ({missing_sql})", ~child_missing)
        parent_missing = pl.any_horizontal(pl.col(name).is_null() for name in fk.target_columns)
        parent_missing_sql = " OR ".join(
            f"{quote_identifier(name)} IS NULL" for name in fk.target_columns
        )
        valid_parent = parent.filter(f"NOT ({parent_missing_sql})", ~parent_missing)
        sides = []
        if fk.cardinality in ("many_to_one", "one_to_one"):
            sides.append(("parent", valid_parent, fk.target_columns, fk.ref_dataset))
        if fk.cardinality == "one_to_one":
            sides.append(("child", valid_child, fk.columns, dataset))
        for side, relation, columns, name in sides:
            occurrences = relation.frequency
            duplicates = (
                summaries["duplicates"]
                if side == "child"
                else relation.filter(
                    f"{quote_identifier(occurrences)} > 1",
                    pl.col(occurrences) > 1,
                ).summarize(ctx.evidence_limit)
            )
            if not duplicates.keys:
                continue
            clean = False
            failures.append(
                _cardinality_finding(
                    dataset, label, fk, side, columns, name, duplicates, occurrences
                )
            )
        valid_parent = valid_parent.rename(dict(zip(fk.target_columns, fk.columns, strict=True)))
        left, right = align_key_relations(valid_child, valid_parent)
        orphans = left.anti(right).summarize(ctx.evidence_limit)
        if not orphans.keys:
            return clean
        valid = summaries["valid"]
        failures.append(_orphan_finding(dataset, label, fk, orphans, valid.keys, frequency))
        return False

    @staticmethod
    def _blocked(ctx: DataContext, label: str, fk: ForeignKeySpec, missing: str) -> Finding:
        return Finding.unassessed(
            f"Not assessed: foreign key {label}: {ctx.unavailable_reason(missing)}.",
            check=label,
            severity=fk.severity,
        )


def _partial_finding(
    dataset: str,
    label: str,
    fk: ForeignKeySpec,
    partial: KeySummary,
    total_rows: int,
    frequency: str,
) -> Finding:
    """Use the same MATCH FULL row count and bounded evidence for every backend."""
    return Finding.fail(
        f"{plural(partial.rows, 'row')} of {label} have a partly null key "
        "(match: full allows only all-null keys).",
        check=f"{label}: partly null keys",
        severity=fk.severity,
        affected=temporal_evidence(present(partial.sample, {frequency: "occurrences"})),
        affected_count=partial.rows,
        total_records=total_rows,
        unit="rows",
        datasets=(dataset,),
    )


def _cardinality_finding(
    dataset: str,
    label: str,
    fk: ForeignKeySpec,
    side: str,
    columns: list[str],
    name: str,
    duplicates: KeySummary,
    frequency: str,
) -> Finding:
    """Report duplicate-key impact consistently for grouped and eager execution."""
    return Finding.fail(
        f"{plural(duplicates.keys, 'key')} of {name}({', '.join(columns)}) occur "
        f"more than once, but {label} is {fk.cardinality.replace('_', '-')}: a "
        "join would multiply rows.",
        check=f"{label}: {side} keys not unique",
        severity=fk.severity,
        affected=temporal_evidence(present(duplicates.sample, {frequency: "occurrences"})),
        affected_count=duplicates.keys,
        affected_rows=duplicates.rows,
        unit="keys",
        datasets=(dataset, fk.ref_dataset),
    )


def _orphan_finding(
    dataset: str,
    label: str,
    fk: ForeignKeySpec,
    orphans: KeySummary,
    total_keys: int,
    frequency: str,
) -> Finding:
    """Keep distinct-key totals separate from affected row totals across backends."""
    return Finding.fail(
        f"{plural(orphans.keys, 'orphaned key')} of {total_keys:,} in "
        f"{label}, used by {plural(orphans.rows, 'row')}.",
        check=label,
        severity=fk.severity,
        affected=temporal_evidence(present(orphans.sample, {frequency: "rows"})),
        affected_count=orphans.keys,
        total_records=total_keys,
        unit="keys",
        affected_rows=orphans.rows,
        datasets=(dataset, fk.ref_dataset),
    )
