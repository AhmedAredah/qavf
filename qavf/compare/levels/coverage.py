"""Coverage: keys present on one side only, overall and by group; null-rate changes.

Key sets are grouped and anti-joined natively with the lossless key alignment
REC-004 uses; only counts and bounded samples reach Python. Null keys match
null keys, as in the keyed record comparison.
"""

from __future__ import annotations

from fractions import Fraction
from typing import ClassVar

import polars as pl
from pydantic import Field

from qavf.compare.levels import SIDES, ComparisonLevel, LevelInputs, graded, register_level
from qavf.compare.results import CoverageResult, LevelResult, NullRateChange
from qavf.compare.spec import Side
from qavf.engine.registry import RuleParams
from qavf.rules._execution import KeyRelation, align_key_relations
from qavf.rules.helpers import fresh_internal, plural


class CoverageParams(RuleParams):
    """Parameters of the ``coverage`` level."""

    allow_added: bool = False
    """Keys only in the candidate are acceptable (e.g. new sites)."""
    allow_removed: bool = False
    """Keys only in the baseline are acceptable."""
    group_by: tuple[str, ...] = ()
    """Break one-sided keys down by these columns (read from the side holding the key)."""
    max_null_rate_change: float = Field(default=0.0, ge=0, le=1)
    """Largest acceptable change of a column's missing fraction (0.01 = one point)."""


def _key_relation(inputs: LevelInputs, side: Side, keys: list[str], frequency: str) -> KeyRelation:
    plan = inputs.scan(side).select(keys)
    schema = dict(plan.collect_schema())
    return KeyRelation(
        inputs.ctx,
        (inputs.datasets[side],),
        tuple(keys),
        frequency,
        {**schema, frequency: pl.UInt64()},
        lazy=plan.group_by(keys, maintain_order=True).agg(
            pl.len().cast(pl.UInt64).alias(frequency)
        ),
        normalize_missing=False,
        collector=inputs.collect,
    )


def _breakdown(
    inputs: LevelInputs,
    side: Side,
    own: KeyRelation,
    other: KeyRelation,
    group_by: list[str],
    count: str,
) -> tuple[int, list[dict]]:
    """One-sided keys per group: every group's count, a bounded largest-first sample."""
    keys = list(own.columns)
    casts = {key: own.schema[key] for key in keys if inputs.schema(side)[key] != own.schema[key]}
    rows = (
        inputs.scan(side)
        .select([*keys, *(c for c in group_by if c not in keys)])
        .with_columns(pl.col(key).cast(dtype) for key, dtype in casts.items())
        .unique()
    )
    assert other.lazy is not None
    grouped = (
        rows.join(other.lazy.select(keys), on=keys, how="anti", nulls_equal=True)
        .group_by(group_by)
        .agg(pl.len().cast(pl.UInt64).alias(count))
    )
    groups = int(inputs.collect(grouped.select(pl.len())).item())
    sample = inputs.collect(
        grouped.sort(
            [count, *group_by], descending=[True, *[False] * len(group_by)], nulls_last=True
        ).head(inputs.ctx.evidence_limit)
    )
    return groups, sample.to_dicts()


def _null_rates(inputs: LevelInputs, columns: list[str], threshold: float):
    counts = {side: inputs.missing_counts(side, columns) for side in SIDES}
    limit = Fraction(str(threshold))
    changes, exceeded = [], []
    for column in columns:
        rates = [
            Fraction(counts[side][column], inputs.rows(side)) if inputs.rows(side) else None
            for side in SIDES
        ]
        change = None if None in rates else rates[1] - rates[0]  # type: ignore[operator]
        record = NullRateChange(
            column=column,
            baseline_rate=None if rates[0] is None else float(rates[0]),
            candidate_rate=None if rates[1] is None else float(rates[1]),
            change=None if change is None else float(change),
        )
        if change:
            changes.append(record)
            if abs(change) > limit:
                exceeded.append(record)
    return changes, exceeded


@register_level
class CoverageLevel(ComparisonLevel):
    """Keys only in the baseline or only in the candidate, and null-rate changes."""

    name: ClassVar[str] = "coverage"
    description: ClassVar[str] = (
        "Keys present only in the baseline or only in the candidate, overall and by an "
        "optional breakdown, and changes of each column's missing-value rate."
    )
    Params: ClassVar[type[RuleParams]] = CoverageParams
    Result: ClassVar[type[LevelResult]] = CoverageResult
    requires_keys: ClassVar[bool] = True

    def evaluate(self, inputs: LevelInputs, name: str) -> CoverageResult:
        params: CoverageParams = self.params  # type: ignore[assignment]
        keys = list(inputs.spec.keys)
        group_by = list(params.group_by)
        for side in SIDES:
            schema = inputs.schema(side)
            if absent := [c for c in [*keys, *group_by] if c not in schema]:
                raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
        frequency = fresh_internal(
            "key_rows", [*inputs.schema("baseline"), *inputs.schema("candidate")]
        )
        base, cand = align_key_relations(
            _key_relation(inputs, "baseline", keys, frequency),
            _key_relation(inputs, "candidate", keys, frequency),
        )
        limit = inputs.ctx.evidence_limit
        removed = base.anti(cand).summarize(limit)
        added = cand.anti(base).summarize(limit)
        totals = {
            side: relation.summarize(0) for side, relation in zip(SIDES, (base, cand), strict=True)
        }
        null_key = pl.any_horizontal(pl.col(key).is_null() for key in keys)
        null_rows = {
            side: int(
                inputs.collect(inputs.scan(side).select(null_key.fill_null(False).sum())).item()
            )
            for side in SIDES
        }
        evidence = inputs.evidence()
        for label, summary in (("only in baseline", removed), ("only in candidate", added)):
            for record in summary.sample.drop(frequency).sort(keys, nulls_last=True).to_dicts():
                evidence.append({**record, "coverage": label})
        breakdown: dict[str, object] = {}
        if group_by:
            count = fresh_internal("keys", group_by)
            for side, own, other, label in (
                ("baseline", base, cand, "only_in_baseline"),
                ("candidate", cand, base, "only_in_candidate"),
            ):
                groups, sample = _breakdown(inputs, side, own, other, group_by, count)
                breakdown[label] = {
                    "groups": groups,
                    "largest": [
                        {**{k: v for k, v in row.items() if k != count}, "keys": row[count]}
                        for row in sample
                    ],
                }
        problems = (0 if params.allow_removed else removed.keys) + (
            0 if params.allow_added else added.keys
        )
        union = totals["baseline"].keys + added.keys
        checks = [
            graded(
                "keys",
                problems,
                f"Both sides cover the same {plural(totals['baseline'].keys, 'key')}.",
                f"{plural(removed.keys, 'key')} only in the baseline, "
                f"{plural(added.keys, 'key')} only in the candidate (of {union:,}).",
                evidence=evidence,
                total=union,
                unit="keys",
                metrics={"breakdown": breakdown, "null_key_rows": null_rows},
            )
        ]
        columns = [c for c in inputs.compared_columns() if c not in keys]
        changes, exceeded = _null_rates(inputs, columns, params.max_null_rate_change)
        evidence = inputs.evidence()
        evidence.extend(change.model_dump() for change in exceeded)
        checks.append(
            graded(
                "null rates",
                len(exceeded),
                f"Missing-value rates agree within {params.max_null_rate_change:g}.",
                f"{plural(len(exceeded), 'column')} changed missing-value rate by more than "
                f"{params.max_null_rate_change:g}: " + ", ".join(c.column for c in exceeded),
                evidence=evidence,
                total=len(columns),
                unit="columns",
            )
        )
        return CoverageResult(
            level=self.name,
            name=name,
            checks=checks,
            keys=keys,
            baseline_keys=totals["baseline"].keys,
            candidate_keys=totals["candidate"].keys,
            shared_keys=totals["baseline"].keys - removed.keys,
            only_in_baseline=removed.keys,
            only_in_candidate=added.keys,
            rows_only_in_baseline=removed.rows,
            rows_only_in_candidate=added.rows,
            null_key_rows=null_rows,
            group_by=group_by,
            null_rates=changes,
        )
