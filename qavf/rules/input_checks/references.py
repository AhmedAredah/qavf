"""Foreign-key match policies, cardinality and orphan contracts (INP-005)."""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, RuleParams
from qavf.engine.severity import Severity
from qavf.keys import KeyTypeError, align_keys
from qavf.rules._execution import (
    KeyBackend,
    align_key_relations,
    key_frequencies,
    key_relation,
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
    SchemaSpec,
)
from qavf.sources.sql import quote_identifier

from ._datasets import (
    CATEGORY,
    DatasetSelection,
    _key_missing,
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
                    rows_name = fresh_internal("rows", child_columns)
                    parent_rows = fresh_internal("rows", parent_columns)
                    backend = paired_key_backend(ctx, dataset, fk.ref_dataset)
                    if backend is not None:
                        if self._relation_rules(ctx, dataset, label, fk, failures, exempt, backend):
                            passed.append(label)
                        continue
                    child = key_frequencies(ctx, dataset, fk.columns, rows_name)
                    parent = key_frequencies(ctx, fk.ref_dataset, fk.target_columns, parent_rows)
                    child_schema = spec.schema_spec
                    parent_spec = ctx.specs.get(fk.ref_dataset)
                    parent_schema = parent_spec.schema_spec if parent_spec else None
                    clean = self._key_rules(
                        dataset,
                        label,
                        fk,
                        child,
                        parent,
                        failures,
                        exempt,
                        ctx.evidence_limit,
                        child_schema,
                        parent_schema,
                        rows_name,
                        parent_rows,
                    )
                    child_missing, _ = _key_missing(child, fk.columns, child_schema)
                    parent_missing, _ = _key_missing(parent, fk.target_columns, parent_schema)
                    position = fresh_internal("position", [*fk.columns, rows_name])
                    keys = child.filter(~child_missing).select(*fk.columns, rows_name)
                    reference = (
                        parent.filter(~parent_missing)
                        .select(fk.target_columns)
                        .unique()
                        .rename(dict(zip(fk.target_columns, fk.columns, strict=True)))
                    )
                    try:
                        left, right = align_keys(keys.select(fk.columns), reference, fk.columns)
                    except KeyTypeError as exc:
                        failures.append(
                            Finding.unassessed(
                                f"Not assessed: {label}: {exc}.",
                                check=label,
                                severity=fk.severity,
                                datasets=(dataset, fk.ref_dataset),
                            )
                        )
                        continue
                    matched = left.with_row_index(position).join(
                        right, on=fk.columns, how="anti", nulls_equal=True
                    )
                    # Report orphans with their original (unconverted) key values.
                    orphans = (
                        keys.with_row_index(position)
                        .join(matched.select(position), on=position)
                        .drop(position)
                    )
                    if orphans.is_empty():
                        if clean:
                            passed.append(label)
                        continue
                    # Counted in distinct keys; ``rows`` holds each key's frequency.
                    orphan_rows = int(orphans[rows_name].sum())
                    failures.append(
                        _orphan_finding(
                            dataset,
                            label,
                            fk,
                            KeySummary(
                                orphans.height, orphan_rows, orphans.head(ctx.evidence_limit)
                            ),
                            keys.height,
                            rows_name,
                        )
                    )
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
    def _key_rules(
        dataset: str,
        label: str,
        fk: ForeignKeySpec,
        child: pl.DataFrame,
        parent: pl.DataFrame,
        failures: list[Finding],
        exempt: dict[str, int],
        limit: int,
        child_schema: SchemaSpec | None,
        parent_schema: SchemaSpec | None,
        child_frequency: str,
        parent_frequency: str,
    ) -> bool:
        """Null-key policy (``match``) and ``cardinality``; ``False`` if any is violated."""
        clean = True
        null_any, null_all = _key_missing(child, fk.columns, child_schema)
        exempt_mask = null_all if fk.match == "full" else null_any
        if exempted := int(child.filter(exempt_mask)[child_frequency].sum()):
            exempt[label] = exempted
        if fk.match == "full":
            partial_keys = child.filter(null_any & ~null_all)
            partial_count = int(partial_keys[child_frequency].sum())
            if partial_count:
                clean = False
                failures.append(
                    _partial_finding(
                        dataset,
                        label,
                        fk,
                        KeySummary(partial_keys.height, partial_count, partial_keys.head(limit)),
                        int(child[child_frequency].sum()),
                        child_frequency,
                    )
                )
        sides = []
        if fk.cardinality in ("many_to_one", "one_to_one"):
            sides.append(("parent", parent, fk.target_columns, fk.ref_dataset))
        if fk.cardinality == "one_to_one":
            sides.append(("child", child, fk.columns, dataset))
        for side, frame, columns, name in sides:
            schema = parent_schema if side == "parent" else child_schema
            missing, _ = _key_missing(frame, columns, schema)
            occurrences = parent_frequency if side == "parent" else child_frequency
            dupes = frame.filter(~missing & (pl.col(occurrences) > 1))
            if dupes.is_empty():
                continue
            clean = False
            failures.append(
                _cardinality_finding(
                    dataset,
                    label,
                    fk,
                    side,
                    columns,
                    name,
                    KeySummary(dupes.height, int(dupes[occurrences].sum()), dupes.head(limit)),
                    occurrences,
                )
            )
        return clean

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
