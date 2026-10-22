"""Groups: measures per group on each side, aligned and compared by materiality.

Each side is reduced natively to one G-row frame of measures (see
:mod:`qavf.compare.measures`); only those aggregates reach Python, where
comparators and the exact materiality judgement run group by group with
bounded evidence. Groups on one side only are reported; a chosen measure's
top-k membership changes are reported as rank changes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, ClassVar, Literal, Protocol

import polars as pl
from pydantic import Field, model_validator

from qavf.compare.comparators import COMPARATORS, ComparatorContext, Materiality
from qavf.compare.levels import SIDES, ComparisonLevel, LevelInputs, graded, register_level
from qavf.compare.measures import MEASURES, Measure, MeasureInput
from qavf.compare.results import (
    CheckResult,
    GroupsResult,
    LevelResult,
    MeasureComparison,
    RankChange,
)
from qavf.compare.spec import ComparisonSpec, Side, _Configured
from qavf.engine.registry import RuleParams
from qavf.keys import align_keys
from qavf.numeric import exact_rational, report_number
from qavf.rules._grouped import ProjectedRelation, grouped_reduce
from qavf.rules.helpers import LargestEvidence, fresh_internal, plural


class MeasureSpec(_Configured):
    """A measure and how its change is judged: ``sum(x)``-style or a mapping."""

    measure: str
    column: str | None = None
    name: str | None = None
    """Label in reports; ``measure(column)`` when omitted."""
    comparators: tuple[str, ...] = ("diff", "rel_diff")
    rtol: float | None = Field(default=None, ge=0)
    """Relative tolerance; the column's, then the level's, then the comparison's."""
    atol: float | None = Field(default=None, ge=0)
    exact: bool = False
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _selector(cls) -> str:
        return "measure"

    @model_validator(mode="before")
    @classmethod
    def _call(cls, data: Any) -> Any:
        if isinstance(data, str) and data.endswith(")") and "(" in data:
            measure, _, column = data[:-1].partition("(")
            return {"measure": measure.strip(), "column": column.strip() or None}
        return data

    @model_validator(mode="after")
    def _known(self) -> MeasureSpec:
        measure = MEASURES.create(self.measure, self.params)
        if measure.requires_column and self.column is None:
            raise ValueError(f"measure {self.measure!r} needs a 'column'")
        for comparator in self.comparators:
            COMPARATORS.create(comparator)
        if self.exact and (self.rtol is not None or self.atol is not None):
            raise ValueError("'exact' cannot be combined with 'rtol' or 'atol'")
        return self

    @property
    def label(self) -> str:
        if self.name is not None:
            return self.name
        return self.measure if self.column is None else f"{self.measure}({self.column})"


class RankSpec(RuleParams):
    """Which groups lead by one measure, compared between the sides."""

    measure: str
    """A measure label of this level."""
    top_k: int = Field(default=10, ge=1)
    descending: bool = True
    """Largest values lead (``false``: smallest)."""


class GroupsParams(RuleParams):
    """Parameters of the ``groups`` level."""

    group_by: tuple[str, ...] = ()
    """Group columns (empty: compare whole-dataset measures)."""
    measures: tuple[MeasureSpec, ...] = Field(
        default_factory=lambda: (MeasureSpec(measure="count"),)
    )
    """Measures compared per group (``count``, ``sum(x)``, ``{measure: quantile, column:
    x, p: 0.9}``, ...); rows per group when omitted."""
    missing_groups: Literal["fail", "ignore"] = "fail"
    """A group present on one side only: a difference (``fail``) or left out."""
    rtol: float | None = Field(default=None, ge=0)
    """Default relative tolerance of this level; the comparison's when omitted."""
    atol: float | None = Field(default=None, ge=0)
    rank: RankSpec | None = None

    @model_validator(mode="after")
    def _consistent(self) -> GroupsParams:
        if not self.measures:
            raise ValueError("'measures' must name at least one measure")
        labels = [measure.label for measure in self.measures]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"measures repeat the label(s) {', '.join(repeated)}; set 'name'")
        if self.rank is not None:
            if self.rank.measure not in labels:
                raise ValueError(f"rank measure {self.rank.measure!r} is not one of {labels}")
            if not self.group_by:
                raise ValueError("'rank' needs 'group_by'")
        return self


class _LevelTolerance(Protocol):
    """A level's own default tolerances (``None``: the comparison's)."""

    @property
    def rtol(self) -> float | None: ...

    @property
    def atol(self) -> float | None: ...


def materiality(spec: ComparisonSpec, level: _LevelTolerance, measure: MeasureSpec) -> Materiality:
    """The measure's tolerance: its own, the column's, the level's, then the comparison's."""
    if measure.exact:
        return Materiality(0.0, 0.0)
    rtol, atol = spec.tolerance(measure.column)
    column = spec.tolerances.get(measure.column) if measure.column is not None else None
    if column is None:
        rtol = level.rtol if level.rtol is not None else rtol
        atol = level.atol if level.atol is not None else atol
    return Materiality(
        measure.rtol if measure.rtol is not None else rtol,
        measure.atol if measure.atol is not None else atol,
    )


def _rows(frame: pl.DataFrame, columns: list[str]) -> Iterator[tuple[Any, ...]]:
    for batch in frame.select(columns).iter_slices(16_384):
        yield from batch.iter_rows()


@dataclass(frozen=True)
class AlignedGroups:
    """Both sides reduced per group and aligned by group (a full outer join).

    Only the G-row aggregates are in memory. Levels that compare other keyed
    reductions (time buckets, series) build on this rather than reducing again.
    """

    keys: list[str]
    measures: list[tuple[MeasureSpec, Measure]]
    aliases: list[str]
    """Value column of each measure in ``frames``."""
    presence: str
    """Row-count column in ``frames``."""
    frames: dict[Side, pl.DataFrame]
    """Per side: the keys, the row count and one column per measure alias."""
    columns: dict[Side, dict[str, str]]
    """Each side's column in ``joined`` for the row count and every measure alias."""
    joined: pl.DataFrame
    only_baseline: pl.DataFrame
    only_candidate: pl.DataFrame
    both: pl.DataFrame


def reduce_groups(
    inputs: LevelInputs,
    keys: list[str],
    specs: tuple[MeasureSpec, ...],
    relations: Mapping[Side, ProjectedRelation] | None = None,
) -> AlignedGroups:
    """Reduce each side natively per ``keys`` group and align the groups losslessly.

    ``relations`` replaces the aligned sides, e.g. with a derived bucket column;
    measures read them through the same :class:`MeasureInput` contract.
    """
    measures = [(spec, MEASURES.create(spec.measure, spec.params)) for spec in specs]
    sides = {side: (relations or {}).get(side) or inputs.relation(side) for side in SIDES}
    taken = [*sides["baseline"].schema, *sides["candidate"].schema]
    presence = fresh_internal("group_rows", taken)
    aliases: list[str] = []
    for index in range(len(measures)):
        aliases.append(fresh_internal(f"measure_{index}", [*taken, presence, *aliases]))
    frames: dict[Side, pl.DataFrame] = {}
    for side in SIDES:
        relation = sides[side]
        needed = [*keys, *(c for spec, m in measures for c in m.columns(spec.column))]
        if absent := sorted({c for c in needed if c not in relation.schema}):
            raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
        frame = grouped_reduce(relation, keys, {presence: pl.len().cast(pl.UInt64)})
        for (spec, measure), alias in zip(measures, aliases, strict=True):
            data = MeasureInput(relation, tuple(keys), spec.column, alias, side)
            measure.check(data)
            values = measure.compute(data).select([*keys, alias])
            frame = (
                frame.join(values, on=keys, how="left", nulls_equal=True)
                if keys
                else pl.concat([frame, values], how="horizontal")
            )
        frames[side] = frame
    names = {
        side: {
            column: fresh_internal(f"{side}_{column}", [*taken, *keys])
            for column in [presence, *aliases]
        }
        for side in SIDES
    }
    base = frames["baseline"].rename(names["baseline"])
    cand = frames["candidate"].rename(names["candidate"])
    if keys:
        base, cand = align_keys(base, cand, keys)
        joined = base.join(cand, on=keys, how="full", coalesce=True, nulls_equal=True)
    else:
        joined = pl.concat([base, cand], how="horizontal")
    joined = joined.sort(keys, nulls_last=True) if keys else joined
    b_present = pl.col(names["baseline"][presence]).is_not_null()
    c_present = pl.col(names["candidate"][presence]).is_not_null()
    return AlignedGroups(
        keys=keys,
        measures=measures,
        aliases=aliases,
        presence=presence,
        frames=frames,
        columns=names,
        joined=joined,
        only_baseline=joined.filter(b_present & ~c_present),
        only_candidate=joined.filter(~b_present & c_present),
        both=joined.filter(b_present & c_present),
    )


def one_sided_check(
    inputs: LevelInputs, aligned: AlignedGroups, check: str, noun: str, allowed: bool
) -> CheckResult:
    """Keys present on one side only, with bounded evidence in key order."""
    evidence = inputs.evidence()
    only_b, only_c = aligned.only_baseline, aligned.only_candidate
    for label, frame in (("only in baseline", only_b), ("only in candidate", only_c)):
        for record in frame.select(aligned.keys).iter_rows(named=True):
            evidence.append({**record, check: label})
    return graded(
        check,
        only_b.height + only_c.height,
        f"Both sides have the same {plural(aligned.both.height, noun)}.",
        f"{plural(only_b.height, noun)} only in the baseline, "
        f"{plural(only_c.height, noun)} only in the candidate.",
        evidence=evidence,
        total=aligned.joined.height,
        unit=f"{noun}s",
        allowed=allowed,
    )


def compare_measure(
    inputs: LevelInputs,
    aligned: AlignedGroups,
    index: int,
    judged: Materiality,
    noun: str = "group",
) -> tuple[MeasureComparison, CheckResult]:
    """Compare measure ``index`` key by key: comparators, materiality, largest changes."""
    spec = aligned.measures[index][0]
    alias = aligned.aliases[index]
    left, right = aligned.columns["baseline"][alias], aligned.columns["candidate"][alias]
    both, keys = aligned.both, aligned.keys
    comparators = [(name, COMPARATORS.create(name)) for name in spec.comparators]
    context = ComparatorContext()
    if "share_change" in spec.comparators:
        totals = []
        for column in (left, right):
            values = (exact_rational(value) for (value,) in _rows(both, [column]))
            totals.append(sum((v for v in values if v is not None), Fraction(0)))
        context = ComparatorContext(totals[0], totals[1])
    leaders = LargestEvidence(inputs.ctx.evidence_limit)
    material = 0
    for position, row in enumerate(_rows(both, [*keys, left, right])):
        *key, before, after = row
        if not judged.material(before, after):
            continue
        material += 1
        b, c = exact_rational(before), exact_rational(after)
        record: dict[str, Any] = dict(zip(keys, key, strict=True))
        record["measure"] = spec.label
        record["baseline"] = report_number(b) if b is not None else before
        record["candidate"] = report_number(c) if c is not None else after
        magnitude = 0.0
        if b is not None and c is not None:
            for name, comparator in comparators:
                record[name] = report_number(comparator.compare(b, c, context))
            magnitude = float(report_number(abs(c - b)) or 0)
        leaders.add(magnitude, position, record)
    evidence = inputs.evidence()
    evidence.extend(leaders.records)
    evidence.count = material
    comparison = MeasureComparison(
        label=spec.label,
        measure=spec.measure,
        column=spec.column,
        comparators=list(spec.comparators),
        compared=both.height,
        material=material,
        rtol=judged.rtol,
        atol=judged.atol,
    )
    check = graded(
        spec.label,
        material,
        f"{spec.label} agrees in {plural(both.height, noun)} "
        f"(rtol={judged.rtol:g}, atol={judged.atol:g}).",
        f"{spec.label} changed materially in {plural(material, noun)} of {both.height:,}.",
        evidence=evidence,
        total=both.height,
        unit=f"{noun}s",
    )
    return comparison, check


@register_level
class GroupsLevel(ComparisonLevel):
    """Measures per group compared with comparators, materiality and rank changes."""

    name: ClassVar[str] = "groups"
    description: ClassVar[str] = (
        "Selected measures per group on each side, aligned by group: groups on one side "
        "only, material changes by comparator and tolerance, and top-k rank changes."
    )
    Params: ClassVar[type[RuleParams]] = GroupsParams
    Result: ClassVar[type[LevelResult]] = GroupsResult

    def evaluate(self, inputs: LevelInputs, name: str) -> GroupsResult:
        params: GroupsParams = self.params  # type: ignore[assignment]
        group_by = list(params.group_by)
        aligned = reduce_groups(inputs, group_by, params.measures)
        checks = [
            one_sided_check(inputs, aligned, "groups", "group", params.missing_groups == "ignore")
        ]
        comparisons = []
        for index, (spec, _) in enumerate(aligned.measures):
            judged = materiality(inputs.spec, params, spec)
            comparison, check = compare_measure(inputs, aligned, index, judged)
            comparisons.append(comparison)
            checks.append(check)
        rank = None
        if params.rank is not None:
            labels = [spec.label for spec, _ in aligned.measures]
            alias = aligned.aliases[labels.index(params.rank.measure)]
            rank, check = self._rank(inputs, params, aligned.frames, group_by, alias)
            checks.append(check)
        return GroupsResult(
            level=self.name,
            name=name,
            checks=checks,
            group_by=group_by,
            baseline_groups=aligned.frames["baseline"].height,
            candidate_groups=aligned.frames["candidate"].height,
            only_in_baseline=aligned.only_baseline.height,
            only_in_candidate=aligned.only_candidate.height,
            measures=comparisons,
            rank=rank,
        )

    @staticmethod
    def _rank(
        inputs: LevelInputs,
        params: GroupsParams,
        frames: dict[Side, pl.DataFrame],
        group_by: list[str],
        alias: str,
    ):
        assert params.rank is not None
        rank = params.rank
        order = fresh_internal("rank_order", [*group_by, alias])
        leaders: dict[str, dict[tuple[Any, ...], int]] = {}
        for side in SIDES:
            frame = frames[side]
            ordering = [
                None if (value := exact_rational(v)) is None else float(value)
                for v in frame[alias].to_list()
            ]
            top = (
                frame.select(group_by)
                .with_columns(pl.Series(order, ordering, dtype=pl.Float64))
                .filter(pl.col(order).is_not_null())
                .sort(
                    [order, *group_by],
                    descending=[rank.descending, *[False] * len(group_by)],
                    nulls_last=True,
                )
                .head(rank.top_k)
            )
            leaders[side] = {
                row: position for position, row in enumerate(top.select(group_by).iter_rows(), 1)
            }
        before, after = leaders["baseline"], leaders["candidate"]
        evidence = inputs.evidence()
        entered = [group for group in after if group not in before]
        left = [group for group in before if group not in after]
        moved = [group for group in after if group in before and before[group] != after[group]]
        for change, groups in (("entered", entered), ("left", left), ("moved", moved)):
            for group in groups:
                evidence.append(
                    {
                        **dict(zip(group_by, group, strict=True)),
                        "change": change,
                        "baseline_rank": before.get(group),
                        "candidate_rank": after.get(group),
                    }
                )
        result = RankChange(
            measure=rank.measure,
            top_k=rank.top_k,
            entered=len(entered),
            left=len(left),
            moved=len(moved),
        )
        check = graded(
            f"rank: {rank.measure}",
            len(entered) + len(left),
            f"The top {rank.top_k} groups by {rank.measure} are unchanged"
            + (f" ({len(moved)} moved within it)." if moved else "."),
            f"{plural(len(entered), 'group')} entered and {plural(len(left), 'group')} left the "
            f"top {rank.top_k} by {rank.measure}.",
            evidence=evidence,
            total=rank.top_k,
            unit="groups",
            metrics={"moved": len(moved)},
        )
        return result, check
