"""Temporal: measures per time bucket (and group) on each side, plus expected-grid gaps.

Both aligned sides get a bucket label column from the shared bucketing module
(:mod:`qavf.temporal.buckets`: calendar buckets in a time zone, or cyclic
profiles), computed natively. The ``groups`` machinery then reduces each side
per ``group_by`` and bucket and compares the measures bucket by bucket, so
measures, comparators and materiality mean exactly what they mean for groups.
Significance does too: each bucket's change is tested from both sides' values
in that bucket, and the buckets of a measure form one family for the
correction. Rows whose time is missing or cannot be placed (unparseable, an
ambiguous or skipped local time) make the level not assessed rather than being
dropped.

Calendar alignment (this year against last year) is the ``shift_time``
alignment step, so it applies before any level, keyed records included.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal

import polars as pl
from pydantic import Field, model_validator

from qavf.compare.levels import (
    SIDES,
    ComparisonLevel,
    LevelInputs,
    graded,
    not_assessed,
    register_level,
)
from qavf.compare.levels.groups import (
    AlignedGroups,
    MeasureSpec,
    compare_measure,
    materiality,
    measure_tests,
    member_values,
    one_sided_check,
    reduce_groups,
)
from qavf.compare.results import CheckResult, LevelResult, TemporalResult
from qavf.compare.significance import Inference
from qavf.compare.spec import InferenceParams, Side
from qavf.engine.registry import RuleParams
from qavf.rules._grouped import ProjectedRelation
from qavf.rules.helpers import fresh_internal, plural
from qavf.settings import Bound
from qavf.temporal.buckets import TimeBucketing, bucket_grid, bucket_labels, bucket_scalar


class _TimeColumn(RuleParams):
    time_column: str
    """Date, Datetime, text (with ``time_format``) or numeric period column."""
    group_by: tuple[str, ...] = ()
    """One series per combination of these columns."""


class TimeAxisParams(TimeBucketing, _TimeColumn):
    """Which time column, bucket and groups make the series of a temporal level."""

    max_buckets: int = Field(default=1_000_000, ge=1)
    """Largest expected bucket grid; a longer one is not assessed."""


@dataclass(frozen=True)
class BucketedSides:
    """Both aligned sides with a bucket label column added natively."""

    relations: dict[Side, ProjectedRelation]
    column: str
    """Name of the label column (``bucket`` unless a side already uses it)."""
    dtype: pl.DataType
    """Label type, the same on both sides."""
    raw: dict[Side, pl.DataType]
    """Each side's time column type."""


def bucketed_sides(inputs: LevelInputs, params: TimeAxisParams) -> BucketedSides:
    """Add bucket labels to both sides; refuse rows that cannot be placed.

    Raises:
        ValueError: for a missing column, a time type the bucket cannot apply to,
            rows with a missing or unplaceable time, or sides whose labels differ
            in type (e.g. numeric periods against dates).
    """
    taken = [*inputs.schema("baseline"), *inputs.schema("candidate")]
    column = "bucket" if "bucket" not in {c.casefold() for c in taken} else None
    column = column or fresh_internal("bucket", taken)
    time = params.time_column
    relations: dict[Side, ProjectedRelation] = {}
    labels: dict[Side, pl.DataType] = {}
    raw: dict[Side, pl.DataType] = {}
    problems = []
    for side in SIDES:
        schema = inputs.schema(side)
        if absent := [c for c in (time, *params.group_by) if c not in schema]:
            raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
        try:
            label, labels[side] = bucket_labels(time, schema[time], params)
        except ValueError as exc:
            raise ValueError(f"{side}: {exc}") from None
        raw[side] = schema[time]
        relation = inputs.relation(side)
        assert relation.lazy is not None
        value = pl.col(time)
        missing, unplaced = relation.collect_plan(
            relation.lazy.select(
                value.is_null().sum().alias("missing"),
                (value.is_not_null() & label.is_null()).sum().alias("unplaced"),
            )
        ).row(0)
        if missing:
            problems.append(f"{side}: {plural(missing, 'row')} without a {time!r} value")
        if unplaced:
            problems.append(
                f"{side}: {plural(unplaced, 'row')} whose {time!r} cannot be placed in a "
                f"{params.describe()} bucket (unparseable, or a local time that a DST change "
                "makes ambiguous or skips)"
            )
        relations[side] = replace(
            relation,
            lazy=relation.lazy.with_columns(label.alias(column)),
            schema={**relation.schema, column: labels[side]},
        )
    if problems:
        raise ValueError("; ".join(problems) + "; exclude such rows with a 'where' condition")
    if labels["baseline"] != labels["candidate"]:
        raise ValueError(
            f"the time columns bucket to different types (baseline {labels['baseline']}, "
            f"candidate {labels['candidate']}); cast them alike or set 'time_zone'"
        )
    return BucketedSides(relations, column, labels["baseline"], raw)


class TimeMembers(TimeBucketing):
    """Time buckets as members of a level: ``{time_column: day, bucket: month}``."""

    time_column: str
    """Date, Datetime, text (with ``time_format``) or numeric period column."""
    max_buckets: int = Field(default=1_000_000, ge=1)


@dataclass(frozen=True)
class KeyedSides:
    """The members a level compares: ``group_by`` columns, then a bucket label if any."""

    keys: list[str]
    relations: dict[Side, ProjectedRelation] | None = None
    """Both sides with the bucket label added (``None``: the aligned sides)."""
    bucket: str | None = None
    """How time was bucketed, e.g. ``month in UTC``."""


def keyed_sides(
    inputs: LevelInputs, group_by: tuple[str, ...], time: TimeMembers | None
) -> KeyedSides:
    """Members keyed by ``group_by`` and, with ``time``, a time bucket (:func:`bucketed_sides`).

    Zones are group columns too (an attribute or the ``assign_zones`` step's), so
    ``group_by: [zone]`` with ``time`` keys members by zone and bucket.

    Raises:
        ValueError: as :func:`bucketed_sides`, or for a missing group column.
    """
    for side in SIDES:
        if absent := [c for c in group_by if c not in inputs.schema(side)]:
            raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
    if time is None:
        return KeyedSides(list(group_by))
    axis = TimeAxisParams(
        time_column=time.time_column,
        group_by=group_by,
        max_buckets=time.max_buckets,
        bucket=time.bucket,
        time_zone=time.time_zone,
        time_format=time.time_format,
    )
    sides = bucketed_sides(inputs, axis)
    return KeyedSides([*group_by, sides.column], sides.relations, time.describe())


class TemporalParams(TimeAxisParams, InferenceParams):
    """Parameters of the ``temporal`` level."""

    measures: tuple[MeasureSpec, ...] = Field(
        default_factory=lambda: (MeasureSpec(measure="count"),)
    )
    """Measures compared per bucket (``count``, ``sum(x)``, ``mean(x)``, ...)."""
    missing_buckets: Literal["fail", "ignore"] = "fail"
    """A (group, bucket) present on one side only: a difference or left out."""
    gaps: bool = True
    """Report buckets of the expected calendar grid that a side lacks."""
    allow_gaps: bool = False
    """Report gaps without failing."""
    start: Bound | None = None
    """First bucket searched for gaps (a date or time; the sides' common span by default)."""
    end: Bound | None = None
    rtol: float | None = Field(default=None, ge=0)
    """Default relative tolerance of this level; the comparison's when omitted."""
    atol: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> TemporalParams:
        if not self.measures:
            raise ValueError("'measures' must name at least one measure")
        labels = [measure.label for measure in self.measures]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"measures repeat the label(s) {', '.join(repeated)}; set 'name'")
        return self


@register_level
class TemporalLevel(ComparisonLevel):
    """Measures per time bucket compared, with one-sided buckets and grid gaps."""

    name: ClassVar[str] = "temporal"
    description: ClassVar[str] = (
        "Both sides bucketed by time (calendar buckets in a time zone, or cyclic profiles) "
        "within optional groups: buckets on one side only, expected-grid gaps on each side, "
        "and material changes of each measure per bucket, largest first."
    )
    Params: ClassVar[type[RuleParams]] = TemporalParams
    Result: ClassVar[type[LevelResult]] = TemporalResult

    def evaluate(self, inputs: LevelInputs, name: str) -> TemporalResult:
        params: TemporalParams = self.params  # type: ignore[assignment]
        sides = bucketed_sides(inputs, params)
        keys = [*params.group_by, sides.column]
        inference = Inference.of(inputs.spec, params)
        plans = measure_tests(inputs, params.measures, sides.relations["baseline"], inference)
        extras = [plan.extras if plan is not None else {} for plan in plans]
        aligned = reduce_groups(inputs, keys, params.measures, sides.relations, extras)
        allowed = params.missing_buckets == "ignore"
        checks = [one_sided_check(inputs, aligned, "buckets", "bucket", allowed)]
        comparisons = []
        for index, (spec, _) in enumerate(aligned.measures):
            judged = materiality(inputs.spec, params, spec)
            comparison, check = compare_measure(
                inputs, aligned, index, judged, "bucket", plans[index], inference
            )
            comparisons.append(comparison)
            checks.append(check)
        window: dict[str, Any] = {}
        if params.gaps:
            try:
                check, window = self._gaps(inputs, params, sides, aligned)
            except ValueError as exc:
                check = not_assessed("gaps", str(exc).rstrip("."))
            checks.append(check)
        return TemporalResult(
            level=self.name,
            name=name,
            checks=checks,
            time_column=params.time_column,
            bucket=params.describe(),
            group_by=list(params.group_by),
            baseline_buckets=aligned.frames["baseline"].height,
            candidate_buckets=aligned.frames["candidate"].height,
            only_in_baseline=aligned.only_baseline.height,
            only_in_candidate=aligned.only_candidate.height,
            measures=comparisons,
            values=[member_values(inputs, aligned, i) for i in range(len(aligned.measures))],
            **window,
        )

    @staticmethod
    def _gaps(
        inputs: LevelInputs, params: TemporalParams, sides: BucketedSides, aligned: AlignedGroups
    ) -> tuple[CheckResult, dict[str, Any]]:
        """Each side's (group, bucket) keys missing from the expected grid of the window."""
        column, group_by = sides.column, list(params.group_by)
        frames = {side: aligned.frames[side].select([*group_by, column]) for side in SIDES}
        if any(frame.is_empty() for frame in frames.values()):
            raise ValueError("a side has no buckets")
        if params.bucket is not None and params.bucket.cyclic:
            cycle = pl.Series(dtype=sides.dtype)
            grid = bucket_grid(cycle, cycle, params.bucket, limit=params.max_buckets)
            first, last = grid.head(1), grid.tail(1)
        else:
            # The common span: the later of the first buckets to the earlier of the last.
            bounds = []
            ends = ((params.start, pl.min, True), (params.end, pl.max, False))
            for configured, extreme, later in ends:
                if configured is not None:
                    raw = sides.raw["baseline"]
                    bound = bucket_scalar(configured, params.time_column, raw, params)
                else:
                    edges = pl.concat([f.select(extreme(column))[column] for f in frames.values()])
                    bound = edges.sort().tail(1) if later else edges.sort().head(1)
                bounds.append(bound.cast(sides.dtype))
            first, last = bounds
            if bool((first > last).any()):
                raise ValueError("the sides share no time window to search for gaps")
            grid = bucket_grid(first, last, params.bucket, limit=params.max_buckets)
        expected = grid.alias(column).to_frame()
        evidence = inputs.evidence()
        gaps: dict[str, int] = {}
        series = 0
        count = fresh_internal("missing", [*group_by, column])
        for side in SIDES:
            frame = frames[side]
            inside = frame.join(expected, on=column, how="semi")
            if group_by:
                groups = frame.select(group_by).unique()
                present = groups.join(
                    inside.group_by(group_by).agg(pl.len().alias(count)),
                    on=group_by,
                    how="left",
                    nulls_equal=True,
                ).with_columns(pl.col(count).fill_null(0))
            else:
                present = pl.DataFrame({count: [inside.height]})
            series += present.height
            missing = present.with_columns((len(grid) - pl.col(count)).alias(count))
            gaps[side] = int(missing[count].sum())
            short = missing.filter(pl.col(count) > 0)
            short = short.sort(group_by, nulls_last=True) if group_by else short
            rows = short.select(group_by).iter_rows(named=True) if group_by else [{}] * short.height
            # Evidence: the first groups' missing buckets, in order; counts stay complete.
            for row in rows:
                if len(evidence.records) >= evidence.limit:
                    break
                own = inside
                for key, value in row.items():
                    own = own.filter(
                        pl.col(key).is_null() if value is None else pl.col(key) == value
                    )
                absent = expected.join(own.select(column), on=column, how="anti").sort(column)
                for (label,) in absent.iter_rows():
                    evidence.append({**row, column: label, "side": side})
        evidence.count = gaps["baseline"] + gaps["candidate"]
        per_side = f"{gaps['baseline']:,} in the baseline, {gaps['candidate']:,} in the candidate"
        check = graded(
            "gaps",
            evidence.count,
            f"Every series has all {plural(len(grid), 'expected bucket')} of the window.",
            f"{plural(evidence.count, 'expected bucket')} missing ({per_side}).",
            evidence=evidence,
            total=len(grid) * series,
            unit="buckets",
            allowed=params.allow_gaps,
            metrics={"expected_per_series": len(grid), "missing": gaps},
        )
        window = {
            "window_start": first.to_list()[0],
            "window_end": last.to_list()[0],
            "expected_buckets": len(grid),
            "gaps": gaps,
        }
        return check, window
