"""Structure: columns, types, units, nullability, row counts and key uniqueness.

Columns are compared after alignment, so a mapped rename is not a removal. A
removed and an added column that look alike (similar names and similar values)
are suggested as a possible rename; suggestions are evidence only and are never
applied.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, ClassVar

import polars as pl
from pydantic import Field

from qavf.aggregation import numeric_expr
from qavf.compare.levels import SIDES, ComparisonLevel, LevelInputs, graded, register_level
from qavf.compare.results import (
    LevelResult,
    NullabilityChange,
    RenameSuggestion,
    StructureResult,
    TypeChange,
    UnitChange,
)
from qavf.engine.registry import RuleParams
from qavf.numeric import numeric_close
from qavf.regression._schema import _kind, _nested, _type_changed
from qavf.regression.models import SchemaPolicy
from qavf.rules._grouped import duplicate_groups
from qavf.rules.helpers import plural
from qavf.rules.regression_checks import ColumnTolerance
from qavf.typed_values import encode_temporal


class StructureParams(RuleParams):
    """Parameters of the ``structure`` level."""

    allow_added: bool = False
    """Columns only in the candidate are acceptable."""
    allow_removed: bool = False
    """Columns only in the baseline are acceptable."""
    schema_policy: SchemaPolicy = Field(default="compatible", alias="schema")
    """``schema`` in configuration: how column types are compared (as in REG-001)."""
    rows: ColumnTolerance = Field(default_factory=lambda: ColumnTolerance(exact=True))
    """Tolerance on the row-count change; exact by default."""
    suggest_renames: bool = True
    max_suggestions: int = Field(default=10, ge=0)
    min_score: float = Field(default=0.6, ge=0, le=1)
    """Lowest similarity (0-1) for a rename suggestion."""
    sample_values: int = Field(default=200, ge=1, le=10_000)
    """Distinct values per column compared for rename suggestions."""


def _samples(
    inputs: LevelInputs, side: str, columns: list[str], limit: int
) -> dict[str, set[Any] | None]:
    """Smallest distinct values per scalar column, for value-similarity scores."""
    schema = inputs.schema(side)  # type: ignore[arg-type]
    scalar = [c for c in columns if not _nested(schema[c]) and schema[c] != pl.Object]
    samples: dict[str, set[Any] | None] = {c: None for c in columns}
    if not scalar:
        return samples
    plan = inputs.scan(side).select(  # type: ignore[arg-type]
        numeric_expr(c, schema[c]).drop_nulls().unique().sort().head(limit).implode().alias(c)
        for c in scalar
    )
    collected = inputs.collect(plan)
    for column in scalar:
        values = collected[column][0]
        encoded = encode_temporal(pl.DataFrame({column: values}))[column].to_list()
        samples[column] = set(encoded)
    return samples


def _suggestions(
    inputs: LevelInputs, removed: list[str], added: list[str], params: StructureParams
) -> list[RenameSuggestion]:
    if not removed or not added or not params.max_suggestions:
        return []
    removed, added = removed[:64], added[:64]  # bounded pairwise scoring
    baseline = _samples(inputs, "baseline", removed, params.sample_values)
    candidate = _samples(inputs, "candidate", added, params.sample_values)
    left, right = inputs.schema("baseline"), inputs.schema("candidate")
    scored = []
    for old in removed:
        for new in added:
            names = SequenceMatcher(None, old.casefold(), new.casefold()).ratio()
            a, b = baseline[old], candidate[new]
            values = None
            if a is not None and b is not None and _kind(left[old]) == _kind(right[new]):
                union = a | b
                values = len(a & b) / len(union) if union else 1.0
            score = names if values is None else (names + values) / 2
            if score >= params.min_score:
                scored.append((-score, old, new, names, values))
    used_old: set[str] = set()
    used_new: set[str] = set()
    suggestions = []
    for negative, old, new, names, values in sorted(scored):
        if old in used_old or new in used_new:
            continue
        used_old.add(old)
        used_new.add(new)
        suggestions.append(
            RenameSuggestion(
                removed=old,
                added=new,
                score=round(-negative, 6),
                name_similarity=round(names, 6),
                value_similarity=None if values is None else round(values, 6),
            )
        )
        if len(suggestions) == params.max_suggestions:
            break
    return suggestions


@register_level
class StructureLevel(ComparisonLevel):
    """Columns added/removed, types, units, nullability, rows and key uniqueness."""

    name: ClassVar[str] = "structure"
    description: ClassVar[str] = (
        "Columns added or removed after mapping (with suggested renames), type, unit and "
        "nullability changes, row counts and key uniqueness on each side."
    )
    Params: ClassVar[type[RuleParams]] = StructureParams
    Result: ClassVar[type[LevelResult]] = StructureResult

    def evaluate(self, inputs: LevelInputs, name: str) -> StructureResult:
        params: StructureParams = self.params  # type: ignore[assignment]
        ignored = set(inputs.spec.ignore_columns)
        left, right = inputs.schema("baseline"), inputs.schema("candidate")
        removed = [c for c in left if c not in right and c not in ignored]
        added = [c for c in right if c not in left and c not in ignored]
        shared = inputs.compared_columns()
        suggestions = _suggestions(inputs, removed, added, params) if params.suggest_renames else []
        checks = []

        evidence = inputs.evidence()
        evidence.extend({"column": c, "change": "removed"} for c in removed)
        evidence.extend({"column": c, "change": "added"} for c in added)
        problems = (0 if params.allow_removed else len(removed)) + (
            0 if params.allow_added else len(added)
        )
        hint = "; ".join(f"{s.removed} -> {s.added}? ({s.score:.2f})" for s in suggestions)
        checks.append(
            graded(
                "columns",
                problems,
                f"Same columns ({len(shared):,} compared).",
                f"{plural(len(removed), 'column')} removed, {plural(len(added), 'column')} "
                f"added" + (f"; possible renames: {hint}" if hint else ""),
                evidence=evidence,
                total=len(set(left) | set(right)),
                unit="columns",
                metrics={"rename_suggestions": [s.model_dump() for s in suggestions]},
            )
        )

        type_changes = [
            TypeChange(column=c, baseline=str(left[c]), candidate=str(right[c]))
            for c in shared
            if _type_changed(left[c], right[c], params.schema_policy)
        ]
        evidence = inputs.evidence()
        evidence.extend(change.model_dump() for change in type_changes)
        checks.append(
            graded(
                "types",
                len(type_changes),
                "Column types agree.",
                f"{plural(len(type_changes), 'column')} changed type: "
                + ", ".join(change.column for change in type_changes),
                evidence=evidence,
                total=len(shared),
                unit="columns",
            )
        )

        unit_changes = []
        for column in shared:
            units = [inputs.units[side].get(column) for side in SIDES]
            if None not in units and units[0] != units[1]:
                unit_changes.append(
                    UnitChange(column=column, baseline=units[0], candidate=units[1])
                )
        evidence = inputs.evidence()
        evidence.extend(change.model_dump() for change in unit_changes)
        checks.append(
            graded(
                "units",
                len(unit_changes),
                "Declared units agree.",
                f"{plural(len(unit_changes), 'column')} declared in different units: "
                + ", ".join(f"{u.column} ({u.baseline} vs {u.candidate})" for u in unit_changes),
                evidence=evidence,
                total=len(shared),
                unit="columns",
            )
        )

        missing = {side: inputs.missing_counts(side, shared) for side in SIDES}
        nullability = []
        for column in shared:
            base, cand = missing["baseline"][column], missing["candidate"][column]
            declared = [inputs.nullable[side].get(column) for side in SIDES]
            observed_change = bool(base) != bool(cand)
            declared_change = None not in declared and declared[0] != declared[1]
            if observed_change or declared_change:
                nullability.append(
                    NullabilityChange(
                        column=column,
                        baseline_missing=base,
                        candidate_missing=cand,
                        baseline_declared=declared[0],
                        candidate_declared=declared[1],
                    )
                )
        evidence = inputs.evidence()
        evidence.extend(change.model_dump() for change in nullability)
        checks.append(
            graded(
                "nullability",
                len(nullability),
                "Columns keep their nullability.",
                f"{plural(len(nullability), 'column')} changed nullability: "
                + ", ".join(change.column for change in nullability),
                evidence=evidence,
                total=len(shared),
                unit="columns",
            )
        )

        rows = [inputs.rows(side) for side in SIDES]
        tolerance = params.rows
        rtol = 0.0 if tolerance.exact else tolerance.rtol or 0.0
        atol = 0.0 if tolerance.exact else tolerance.atol or 0.0
        close = numeric_close(rows[0], rows[1], rtol, atol)
        checks.append(
            graded(
                "rows",
                0 if close else 1,
                f"Row counts agree ({rows[0]:,}).",
                f"Row count changed from {rows[0]:,} to {rows[1]:,} ({rows[1] - rows[0]:+,}).",
                metrics={"baseline_rows": rows[0], "candidate_rows": rows[1]},
            )
        )

        duplicates: dict[str, int] = {}
        keys = list(inputs.spec.keys)
        if keys:
            for side in SIDES:
                relation = inputs.relation(side)
                if absent := [key for key in keys if key not in relation.schema]:
                    raise ValueError(f"{side} lacks key column(s) {', '.join(absent)}")
                duplicates[side] = duplicate_groups(relation, keys)
            repeated = sum(duplicates.values())
            checks.append(
                graded(
                    "keys",
                    repeated,
                    f"Keys ({', '.join(keys)}) are unique on both sides.",
                    "Keys repeat: "
                    + ", ".join(
                        f"{plural(n, 'key')} in the {s}" for s, n in duplicates.items() if n
                    ),
                    unit="keys",
                    metrics={"duplicate_keys": duplicates},
                )
            )
        return StructureResult(
            level=self.name,
            name=name,
            checks=checks,
            baseline_rows=rows[0],
            candidate_rows=rows[1],
            columns_added=added,
            columns_removed=removed,
            type_changes=type_changes,
            unit_changes=unit_changes,
            nullability_changes=nullability,
            rename_suggestions=suggestions,
            duplicate_keys=duplicates,
        )
