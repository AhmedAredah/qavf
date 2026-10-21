"""Groups: measures per group on each side, aligned and compared by materiality.

Each side is reduced natively to one G-row frame of measures (see
:mod:`qavf.compare.measures`); only those aggregates reach Python, where
comparators and the exact materiality judgement run group by group with
bounded evidence. Groups on one side only are reported; a chosen measure's
top-k membership changes are reported as rank changes.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from fractions import Fraction
from typing import Any, ClassVar, Literal

import polars as pl
from pydantic import Field, model_validator

from qavf.compare.comparators import (
    COMPARATORS,
    ComparatorContext,
    Materiality,
    exact_value,
    presented,
)
from qavf.compare.levels import SIDES, ComparisonLevel, LevelInputs, graded, register_level
from qavf.compare.measures import MEASURES, MeasureInput
from qavf.compare.results import GroupsResult, LevelResult, MeasureComparison, RankChange
from qavf.compare.spec import ComparisonSpec, Side, _Configured
from qavf.engine.registry import RuleParams
from qavf.keys import align_keys
from qavf.rules._grouped import grouped_reduce
from qavf.rules.helpers import BoundedEvidence, fresh_internal, plural


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


def _materiality(spec: ComparisonSpec, params: GroupsParams, measure: MeasureSpec) -> Materiality:
    if measure.exact:
        return Materiality(0.0, 0.0)
    rtol, atol = spec.tolerance(measure.column)
    column = spec.tolerances.get(measure.column) if measure.column is not None else None
    if column is None:
        rtol = params.rtol if params.rtol is not None else rtol
        atol = params.atol if params.atol is not None else atol
    return Materiality(
        measure.rtol if measure.rtol is not None else rtol,
        measure.atol if measure.atol is not None else atol,
    )


def _rows(frame: pl.DataFrame, columns: list[str]) -> Iterator[tuple[Any, ...]]:
    for batch in frame.select(columns).iter_slices(16_384):
        yield from batch.iter_rows()


class _Leaders:
    """The ``limit`` records with the largest magnitude, ties in group order."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.heap: list[tuple[float, int, dict[str, Any]]] = []
        self.count = 0

    def add(self, magnitude: float, position: int, record: dict[str, Any]) -> None:
        self.count += 1
        if not self.limit:
            return
        item = (magnitude, -position, record)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
        elif item[:2] > self.heap[0][:2]:
            heapq.heapreplace(self.heap, item)

    def into(self, evidence: BoundedEvidence) -> BoundedEvidence:
        for _, _, record in sorted(self.heap, key=lambda item: (-item[0], -item[1])):
            evidence.append(record)
        evidence.count = self.count
        return evidence


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
        measures = [(spec, MEASURES.create(spec.measure, spec.params)) for spec in params.measures]
        taken = [*inputs.schema("baseline"), *inputs.schema("candidate")]
        presence = fresh_internal("group_rows", taken)
        aliases: list[str] = []
        for index in range(len(measures)):
            aliases.append(fresh_internal(f"measure_{index}", [*taken, presence, *aliases]))
        frames: dict[Side, pl.DataFrame] = {}
        for side in SIDES:
            relation = inputs.relation(side)
            needed = [*group_by, *(c for spec, m in measures for c in m.columns(spec.column))]
            if absent := sorted({c for c in needed if c not in relation.schema}):
                raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
            frame = grouped_reduce(relation, group_by, {presence: pl.len().cast(pl.UInt64)})
            for (spec, measure), alias in zip(measures, aliases, strict=True):
                data = MeasureInput(relation, tuple(group_by), spec.column, alias, side)
                measure.check(data)
                values = measure.compute(data).select([*group_by, alias])
                frame = (
                    frame.join(values, on=group_by, how="left", nulls_equal=True)
                    if group_by
                    else pl.concat([frame, values], how="horizontal")
                )
            frames[side] = frame
        names = {
            side: {
                column: fresh_internal(f"{side}_{column}", [*taken, *group_by])
                for column in [presence, *aliases]
            }
            for side in SIDES
        }
        base = frames["baseline"].rename(names["baseline"])
        cand = frames["candidate"].rename(names["candidate"])
        if group_by:
            base, cand = align_keys(base, cand, group_by)
            joined = base.join(cand, on=group_by, how="full", coalesce=True, nulls_equal=True)
        else:
            joined = pl.concat([base, cand], how="horizontal")
        joined = joined.sort(group_by, nulls_last=True) if group_by else joined
        b_present = pl.col(names["baseline"][presence]).is_not_null()
        c_present = pl.col(names["candidate"][presence]).is_not_null()
        only_b = joined.filter(b_present & ~c_present)
        only_c = joined.filter(~b_present & c_present)
        both = joined.filter(b_present & c_present)
        checks = []
        evidence = inputs.evidence()
        for label, frame in (("only in baseline", only_b), ("only in candidate", only_c)):
            for record in frame.select(group_by).iter_rows(named=True):
                evidence.append({**record, "groups": label})
        one_sided = only_b.height + only_c.height
        checks.append(
            graded(
                "groups",
                one_sided,
                f"Both sides have the same {plural(both.height, 'group')}.",
                f"{plural(only_b.height, 'group')} only in the baseline, "
                f"{plural(only_c.height, 'group')} only in the candidate.",
                evidence=evidence,
                total=joined.height,
                unit="groups",
                allowed=params.missing_groups == "ignore",
            )
        )
        comparisons = []
        for (spec, _), alias in zip(measures, aliases, strict=True):
            comparison, check = self._measure(
                inputs,
                params,
                spec,
                both,
                group_by,
                names["baseline"][alias],
                names["candidate"][alias],
            )
            comparisons.append(comparison)
            checks.append(check)
        rank = None
        if params.rank is not None:
            alias = aliases[[spec.label for spec, _ in measures].index(params.rank.measure)]
            rank, check = self._rank(inputs, params, frames, group_by, alias)
            checks.append(check)
        return GroupsResult(
            level=self.name,
            name=name,
            checks=checks,
            group_by=group_by,
            baseline_groups=frames["baseline"].height,
            candidate_groups=frames["candidate"].height,
            only_in_baseline=only_b.height,
            only_in_candidate=only_c.height,
            measures=comparisons,
            rank=rank,
        )

    @staticmethod
    def _measure(
        inputs: LevelInputs,
        params: GroupsParams,
        spec: MeasureSpec,
        both: pl.DataFrame,
        group_by: list[str],
        left: str,
        right: str,
    ):
        materiality = _materiality(inputs.spec, params, spec)
        comparators = [(name, COMPARATORS.create(name)) for name in spec.comparators]
        context = ComparatorContext()
        if "share_change" in spec.comparators:
            totals = []
            for column in (left, right):
                values = (exact_value(value) for (value,) in _rows(both, [column]))
                totals.append(sum((v for v in values if v is not None), Fraction(0)))
            context = ComparatorContext(totals[0], totals[1])
        leaders = _Leaders(inputs.ctx.evidence_limit)
        material = 0
        for position, row in enumerate(_rows(both, [*group_by, left, right])):
            *keys, before, after = row
            if not materiality.material(before, after):
                continue
            material += 1
            b, c = exact_value(before), exact_value(after)
            record: dict[str, Any] = dict(zip(group_by, keys, strict=True))
            record["measure"] = spec.label
            record["baseline"] = presented(b) if b is not None else before
            record["candidate"] = presented(c) if c is not None else after
            magnitude = 0.0
            if b is not None and c is not None:
                for name, comparator in comparators:
                    record[name] = presented(comparator.compare(b, c, context))
                magnitude = float(presented(abs(c - b)) or 0)
            leaders.add(magnitude, position, record)
        evidence = leaders.into(inputs.evidence())
        comparison = MeasureComparison(
            label=spec.label,
            measure=spec.measure,
            column=spec.column,
            comparators=list(spec.comparators),
            compared=both.height,
            material=material,
            rtol=materiality.rtol,
            atol=materiality.atol,
        )
        check = graded(
            spec.label,
            material,
            f"{spec.label} agrees in {plural(both.height, 'group')} "
            f"(rtol={materiality.rtol:g}, atol={materiality.atol:g}).",
            f"{spec.label} changed materially in {plural(material, 'group')} of {both.height:,}.",
            evidence=evidence,
            total=both.height,
            unit="groups",
        )
        return comparison, check

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
                None if (value := exact_value(v)) is None else float(value)
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
