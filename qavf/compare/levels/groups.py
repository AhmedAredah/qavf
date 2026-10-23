"""Groups: measures per group on each side, aligned and compared by materiality.

Each side is reduced natively to one G-row frame of measures (see
:mod:`qavf.compare.measures`); only those aggregates reach Python, where
comparators and the exact materiality judgement run group by group with
bounded evidence. Groups on one side only are reported; a chosen measure's
top-k membership changes are reported as rank changes.

Unless the decision policy is ``material``, each measure's groups are also
tested (:mod:`qavf.compare.significance`): what the test reads per group is
reduced natively with the measure, the groups form one family for the
multiple-testing correction, and each group is decided by materiality and
significance together.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from heapq import nsmallest
from typing import Any, ClassVar, Literal, Protocol

import polars as pl
from pydantic import Field, model_validator

from qavf.compare.comparators import COMPARATORS, ComparatorContext, Materiality
from qavf.compare.levels import (
    SIDES,
    ComparisonLevel,
    LevelInputs,
    graded,
    not_assessed,
    register_level,
)
from qavf.compare.measures import MEASURES, Measure, MeasureInput
from qavf.compare.results import (
    CheckResult,
    GroupsResult,
    LevelResult,
    MeasureComparison,
    MemberValues,
    RankChange,
)
from qavf.compare.significance import (
    Extra,
    Inference,
    MeasureTest,
    Member,
    compatibility,
    evidence_fields,
    judge,
    member_input,
    plan_measure,
    summary_metrics,
)
from qavf.compare.significance import label as member_label
from qavf.compare.spec import ComparisonSpec, InferenceParams, Side, TestConfig, _Configured
from qavf.engine.registry import RuleParams
from qavf.key_codec import canonical_key
from qavf.keys import align_keys
from qavf.numeric import exact_rational, report_number
from qavf.rules._grouped import ProjectedRelation, grouped_reduce
from qavf.rules.helpers import BoundedEvidence, LargestEvidence, fresh_internal, plural
from qavf.typed_values import encode_temporal, has_temporal


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
    test: TestConfig | None = None
    """Significance test of the change per group; the measure's default when
    omitted (``mean``: ``welch_t``, ``median``: ``mann_whitney``, ``count``:
    ``poisson``, ...), ``none``: judged by materiality alone."""
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
        test = self.test.create() if self.test is not None else None
        if test is not None and (problem := compatibility(test, measure, self.column)):
            raise ValueError(problem)
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


class GroupsParams(InferenceParams):
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


def measure_columns(specs: Sequence[MeasureSpec]) -> set[str]:
    """Declared source dependencies of built-in or custom grouped measures."""
    return {
        column
        for spec in specs
        for column in MEASURES.create(spec.measure, spec.params).columns(spec.column)
    }


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


def precise_frame(frame: pl.DataFrame, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Protect sub-microsecond identities while preserving ordinary Python dates.

    Python's temporal scalars cannot carry nanoseconds. Encode those logical
    types consistently, including zero ticks, so labels remain identical across
    subsets and batches. Other ordinary temporal values keep their public types.
    """
    selected = [
        name
        for name, dtype in frame.schema.items()
        if (columns is None or name in columns)
        and (
            (isinstance(dtype, pl.Datetime | pl.Duration) and dtype.time_unit == "ns")
            or dtype == pl.Time
            or (isinstance(dtype, pl.List | pl.Array | pl.Struct) and has_temporal(dtype))
        )
    ]
    return encode_temporal(frame, selected) if selected else frame


def _rows(frame: pl.DataFrame, columns: list[str]) -> Iterator[tuple[Any, ...]]:
    for batch in frame.select(columns).iter_slices(16_384):
        yield from precise_frame(batch).iter_rows()


def group_partitions(
    frame: pl.DataFrame, keys: list[str]
) -> Iterator[tuple[tuple[Any, ...], pl.DataFrame]]:
    """Keep native groups distinct before converting their labels to Python.

    ``partition_by(as_dict=True)`` converts temporal keys to Python datetimes,
    which can merge different nanoseconds. A list of native partitions avoids
    that conversion; only the bounded key row is encoded for reporting.
    """
    if not keys:
        yield (), frame
        return
    for part in frame.partition_by(keys, maintain_order=True):
        key = precise_frame(part.select(keys).head(1)).row(0)
        yield key, part


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
    """Each side's column in ``joined`` for the row count and every measure and extra alias."""
    joined: pl.DataFrame
    only_baseline: pl.DataFrame
    only_candidate: pl.DataFrame
    both: pl.DataFrame
    extras: list[dict[str, str]] = field(default_factory=list)
    """Per measure: the alias of each extra reduction (what its test reads), by role."""


def reduce_groups(
    inputs: LevelInputs,
    keys: list[str],
    specs: tuple[MeasureSpec, ...],
    relations: Mapping[Side, ProjectedRelation] | None = None,
    extras: Sequence[Mapping[str, Extra]] | None = None,
) -> AlignedGroups:
    """Reduce each side natively per ``keys`` group and align the groups losslessly.

    ``relations`` replaces the aligned sides, e.g. with a derived bucket column;
    measures read them through the same :class:`MeasureInput` contract.
    ``extras`` adds, per measure, further reductions by role (what a test reads:
    values, moments, successes), aligned with the measures.
    """
    measures = [(spec, MEASURES.create(spec.measure, spec.params)) for spec in specs]
    sides = {side: (relations or {}).get(side) or inputs.relation(side) for side in SIDES}
    taken = [*sides["baseline"].schema, *sides["candidate"].schema]
    presence = fresh_internal("group_rows", taken)
    aliases: list[str] = []
    for index in range(len(measures)):
        aliases.append(fresh_internal(f"measure_{index}", [*taken, presence, *aliases]))
    roles: list[dict[str, str]] = []
    for index, extra in enumerate(extras or ()):
        named: dict[str, str] = {}
        for role in extra:
            used = [*taken, presence, *aliases, *(a for r in roles for a in r.values())]
            named[role] = fresh_internal(f"extra_{index}_{role}", [*used, *named.values()])
        roles.append(named)
    frames: dict[Side, pl.DataFrame] = {}
    for side in SIDES:
        relation = sides[side]
        needed = [*keys, *(c for spec, m in measures for c in m.columns(spec.column))]
        if absent := sorted({c for c in needed if c not in relation.schema}):
            raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
        reductions = {presence: pl.len().cast(pl.UInt64)}
        separate: list[tuple[Measure, MeasureInput]] = []
        for (spec, measure), alias in zip(measures, aliases, strict=True):
            data = MeasureInput(relation, tuple(keys), spec.column, alias, side)
            measure.check(data)
            expression = measure.reduction(data)
            if expression is None:
                separate.append((measure, data))
            else:
                reductions[alias] = expression
        for extra, named in zip(extras or (), roles, strict=True):
            reductions.update(
                {named[role]: build(relation, bool(keys)) for role, build in extra.items()}
            )
        frame = grouped_reduce(relation, keys, reductions)
        for measure, data in separate:
            values = measure.compute(data).select([*keys, data.alias])
            frame = (
                frame.join(values, on=keys, how="left", nulls_equal=True)
                if keys
                else pl.concat([frame, values], how="horizontal")
            )
        frames[side] = frame
    extra_aliases = [alias for named in roles for alias in named.values()]
    names = {
        side: {
            column: fresh_internal(f"{side}_{column}", [*taken, *keys])
            for column in [presence, *aliases, *extra_aliases]
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
        extras=roles,
    )


def measure_tests(
    inputs: LevelInputs,
    specs: Sequence[MeasureSpec],
    relation: ProjectedRelation,
    inference: Inference,
) -> list[MeasureTest | None]:
    """Each measure's test plan (``None`` for every measure under ``material``)."""
    rows = sum(inputs.rows(side) for side in SIDES)
    return [
        plan_measure(spec, MEASURES.create(spec.measure, spec.params), relation, inference, rows)
        for spec in specs
    ]


def member_values(inputs: LevelInputs, aligned: AlignedGroups, index: int) -> MemberValues:
    """Measure ``index`` on both sides per key (one-sided keys too), in key order, bounded
    by the comparison's ``chart_points`` and the evidence byte limit: the report's charts."""
    spec = aligned.measures[index][0]
    alias = aligned.aliases[index]
    left, right = (aligned.columns[side][alias] for side in SIDES)
    kept = BoundedEvidence(inputs.spec.chart_points, byte_limit=inputs.ctx.evidence_bytes)
    keys = aligned.keys
    for row in _rows(aligned.joined, [*keys, left, right]):
        if len(kept.records) >= kept.limit:
            kept.count += 1
            continue
        values = [exact_rational(value) for value in row[len(keys) :]]
        kept.append(
            {
                **dict(zip(keys, row[: len(keys)], strict=True)),
                "baseline": report_number(values[0]),
                "candidate": report_number(values[1]),
            }
        )
    return MemberValues(
        measure=spec.label,
        keys=list(keys),
        points=kept.records,
        total=kept.count,
        truncated=kept.truncated,
    )


def one_sided_check(
    inputs: LevelInputs, aligned: AlignedGroups, check: str, noun: str, allowed: bool
) -> CheckResult:
    """Keys present on one side only, with bounded evidence in key order."""
    evidence = inputs.evidence()
    only_b, only_c = aligned.only_baseline, aligned.only_candidate
    for label, frame in (("only in baseline", only_b), ("only in candidate", only_c)):
        for record in precise_frame(frame.select(aligned.keys)).iter_rows(named=True):
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


def _record(
    keys: list[str],
    key: list[Any],
    spec: MeasureSpec,
    before: Any,
    after: Any,
    comparators: list[tuple[str, Any]],
    context: ComparatorContext,
) -> tuple[dict[str, Any], float]:
    """A changed member's evidence record and its magnitude (for largest-first order)."""
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
    return record, magnitude


_VERDICTS = {
    "material_and_significant": (
        "changed materially and significantly",
        "material and significant",
    ),
    "significant": ("changed significantly", "significant"),
    "equivalence": ("is not shown equivalent", "non-equivalent"),
}


def compare_measure(
    inputs: LevelInputs,
    aligned: AlignedGroups,
    index: int,
    judged: Materiality,
    noun: str = "group",
    plan: MeasureTest | None = None,
    inference: Inference | None = None,
    *,
    include_missing: bool = False,
) -> tuple[MeasureComparison, CheckResult]:
    """Compare measure ``index`` key by key: comparators, materiality, largest changes.

    With a test ``plan`` (any policy but ``material``), each key's change is also
    tested and the family is decided by ``inference``'s policy; failing keys
    carry their p-values, effect and interval as evidence.
    """
    spec = aligned.measures[index][0]
    alias = aligned.aliases[index]
    left, right = aligned.columns["baseline"][alias], aligned.columns["candidate"][alias]
    both, keys = aligned.both, aligned.keys
    if include_missing:
        # Additive contributions cover the union: absence means zero, while an
        # undefined aggregate on a present side remains undefined.
        both = aligned.joined.with_columns(
            pl.when(pl.col(aligned.columns[side][aligned.presence]).is_null())
            .then(pl.lit(0).cast(aligned.joined.schema[column]))
            .otherwise(pl.col(column))
            .alias(column)
            for side, column in zip(SIDES, (left, right), strict=True)
        )
    comparators = [(name, COMPARATORS.create(name)) for name in spec.comparators]
    context = ComparatorContext()
    if "share_change" in spec.comparators:
        totals = []
        for column in (left, right):
            values = (exact_rational(value) for (value,) in _rows(both, [column]))
            totals.append(sum((v for v in values if v is not None), Fraction(0)))
        context = ComparatorContext(totals[0], totals[1])
    tested = plan is not None and inference is not None
    roles = aligned.extras[index] if tested and index < len(aligned.extras) else {}
    extra = [aligned.columns[side][alias] for side in SIDES for alias in roles.values()]
    shares: dict[Side, float | None] = {"baseline": None, "candidate": None}
    if tested and plan is not None and plan.share:
        for side in SIDES:
            column = roles["successes"]
            shares[side] = float(sum(v for v in aligned.frames[side][column].to_list() if v))
    leaders = LargestEvidence(inputs.ctx.evidence_limit)
    material = 0
    members: list[Member] = []
    changes: list[tuple[list[Any], Any, Any]] = []
    unmeasured = inputs.evidence()

    def complete(check: CheckResult) -> CheckResult:
        if not unmeasured.count:
            return check
        metrics = {**check.metrics, "not_measured": unmeasured.count}
        if check.assessed and not check.passed:
            return check.model_copy(
                update={
                    "message": f"{check.message} {unmeasured.count:,} {noun}(s) not measured.",
                    "metrics": metrics,
                }
            )
        evidence = inputs.evidence()
        evidence.extend(check.affected_records)
        evidence.extend(unmeasured.records)
        evidence.count = check.affected_count + unmeasured.count
        return check.model_copy(
            update={
                "passed": False,
                "assessed": False,
                "message": f"Not assessed: {spec.label}: {unmeasured.count:,} {noun}(s) have "
                "an undefined measure on a present side."
                + (f" {check.message}" if not check.assessed else ""),
                "affected_count": evidence.count,
                "affected_records": evidence.records,
                "truncated": evidence.truncated,
                "metrics": metrics,
            }
        )

    for position, row in enumerate(_rows(both, [*keys, left, right, *extra])):
        key = list(row[: len(keys)])
        before, after = row[len(keys)], row[len(keys) + 1]
        if before is None or after is None:
            unmeasured.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "measure": spec.label,
                    "reason": "the measure is undefined on a present side",
                }
            )
            continue
        changed = judged.material(before, after)
        material += changed
        if tested:
            assert plan is not None and inference is not None
            name = member_label(keys, key)
            members.append(
                _member(plan, inference, position, row[len(keys) :], roles, shares, judged, name)
                if plan.usable
                else Member(changed, name)
            )
            changes.append((key, before, after))
            continue
        if not changed:
            continue
        record, magnitude = _record(keys, key, spec, before, after, comparators, context)
        leaders.add(magnitude, position, record)
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
    if both.is_empty():
        return comparison, not_assessed(spec.label, f"no {noun} to compare on both sides")
    tolerance = f"rtol={judged.rtol:g}, atol={judged.atol:g}"
    if not tested:
        evidence = inputs.evidence()
        evidence.extend(leaders.records)
        evidence.count = material
        check = graded(
            spec.label,
            material,
            f"{spec.label} agrees in {plural(both.height, noun)} ({tolerance}).",
            f"{spec.label} changed materially in {plural(material, noun)} of {both.height:,}.",
            evidence=evidence,
            total=both.height,
            unit=f"{noun}s",
        )
        return comparison, complete(check)
    assert plan is not None and inference is not None
    judgement = judge(plan, inference, members, inputs.ctx.evidence_limit)
    failed, unassessed = judgement.failed, judgement.unassessed
    shown = failed or unassessed
    for position in shown:
        key, before, after = changes[position]
        record, magnitude = _record(keys, key, spec, before, after, comparators, context)
        record |= evidence_fields(judgement.results[position], judgement.decisions[position])
        leaders.add(magnitude, position, record)
    evidence = inputs.evidence()
    evidence.extend(leaders.records)
    evidence.count = len(shown)
    significance = judgement.significance
    metrics = {"material": material, **summary_metrics(significance)}
    if not plan.applicable:
        why = f"{tolerance}; no sound test, judged by materiality"
        check = graded(
            spec.label,
            len(failed),
            f"{spec.label} agrees in {plural(both.height, noun)} ({why}).",
            f"{spec.label} changed materially in {plural(len(failed), noun)} of "
            f"{both.height:,} ({why}).",
            evidence=evidence,
            total=both.height,
            unit=f"{noun}s",
            metrics=metrics,
        )
    elif failed or not unassessed:
        verb, adjective = _VERDICTS[inference.decision]
        settings = f"{plan.name}; {inference.describe()}; {tolerance}"
        check = graded(
            spec.label,
            len(failed),
            f"{spec.label} has no {adjective} change in {plural(both.height, noun)} "
            f"({material:,} material; {settings}).",
            f"{spec.label} {verb} in {plural(len(failed), noun)} of {both.height:,} "
            f"({settings})" + (f"; {len(unassessed):,} not assessed" if unassessed else "") + ".",
            evidence=evidence,
            total=both.height,
            unit=f"{noun}s",
            metrics=metrics,
        )
    else:
        reason = judgement.decisions[unassessed[0]].reason
        check = CheckResult(
            check=spec.label,
            passed=False,
            assessed=False,
            message=f"Not assessed: {spec.label}: {plural(len(unassessed), noun)} need a "
            f"{plan.name} test that could not run ({reason}).",
            affected_count=len(unassessed),
            total=both.height,
            unit=f"{noun}s",
            affected_records=evidence.records,
            truncated=evidence.truncated,
            metrics=metrics,
        )
    comparison = comparison.model_copy(update={"significance": significance})
    return comparison, complete(check.model_copy(update={"significance": significance}))


def _member(
    plan: MeasureTest,
    inference: Inference,
    position: int,
    values: Sequence[Any],
    roles: Mapping[str, str],
    shares: dict[Side, float | None],
    judged: Materiality,
    name: str | None,
) -> Member:
    """A compared key as a family member: its materiality and its test input."""
    before, after, *rest = values
    count = len(roles)
    extras = {
        side: dict(zip(roles, rest[i * count : (i + 1) * count], strict=True))
        for i, side in enumerate(SIDES)
    }
    margin = judged.margin(before, after) if inference.decision == "equivalence" else None
    data, problem = member_input(plan, inference, position, (before, after), extras, shares, margin)
    return Member(judged.material(before, after), name, data, problem)


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

    def required_columns(self) -> frozenset[str]:
        params: GroupsParams = self.params  # type: ignore[assignment]
        return frozenset(set(params.group_by) | measure_columns(params.measures))

    def evaluate(self, inputs: LevelInputs, name: str) -> GroupsResult:
        params: GroupsParams = self.params  # type: ignore[assignment]
        group_by = list(params.group_by)
        inference = Inference.of(inputs.spec, params)
        plans = measure_tests(inputs, params.measures, inputs.relation("baseline"), inference)
        extras = [plan.extras if plan is not None else {} for plan in plans]
        aligned = reduce_groups(inputs, group_by, params.measures, extras=extras)
        checks = [
            one_sided_check(inputs, aligned, "groups", "group", params.missing_groups == "ignore")
        ]
        comparisons = []
        for index, (spec, _) in enumerate(aligned.measures):
            judged = materiality(inputs.spec, params, spec)
            comparison, check = compare_measure(
                inputs, aligned, index, judged, plan=plans[index], inference=inference
            )
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
        leaders: dict[str, dict[bytes, int]] = {}
        labels: dict[bytes, tuple[Any, ...]] = {}
        for side in SIDES:
            # Sort keys natively first for the documented deterministic tie order.
            # The heap retains only k exact numeric order keys, never Float64 ranks.
            frame = frames[side].sort(group_by, nulls_last=True)

            def candidates(source: pl.DataFrame) -> Iterator[tuple[Fraction, int, tuple[Any, ...]]]:
                for position, row in enumerate(_rows(source, [*group_by, alias])):
                    value = exact_rational(row[-1])
                    if value is None:
                        raise ValueError("a group has an undefined or nonnumeric rank measure")
                    yield (-value if rank.descending else value), position, tuple(row[:-1])

            try:
                top = nsmallest(rank.top_k, candidates(frame))
            except ValueError as exc:
                return None, not_assessed(f"rank: {rank.measure}", f"{side}: {exc}")
            if not top:
                return None, not_assessed(f"rank: {rank.measure}", f"{side} has no groups to rank")
            leaders[side] = {}
            for position, row in enumerate(top, 1):
                key = canonical_key(row[2])
                labels[key] = row[2]
                leaders[side][key] = position
        before, after = leaders["baseline"], leaders["candidate"]
        evidence = inputs.evidence()
        entered = [group for group in after if group not in before]
        left = [group for group in before if group not in after]
        moved = [group for group in after if group in before and before[group] != after[group]]
        for change, groups in (("entered", entered), ("left", left), ("moved", moved)):
            for group in groups:
                evidence.append(
                    {
                        **dict(zip(group_by, labels[group], strict=True)),
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
