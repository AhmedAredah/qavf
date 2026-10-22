"""Temporal QA: step changes, trend direction, trend outliers and gaps.

Each rule works on *series*: a value column summed per time period within
optional group columns, e.g. NOx by county. The period is a numeric column
(such as a year) or, with ``bucket``, a calendar bucket of a Date, Datetime or
text column (``day``, ``15 minutes``, ``month`` in ``time_zone``; see
:mod:`qavf.temporal.buckets`). Bucketed periods have two exact positions:

* the **sequence** position, for spacing in time: TMP-001 rates (per bucket)
  and the TMP-002/003 trends. Consecutive buckets are one apart, so a DST
  fall-back hour is the bucket after the first 01:00 and nothing is skipped
  or doubled;
* the **calendar** position, for cycles: TMP-007 seasons are calendar
  position mod ``season_length``, so with hourly buckets and 24 a season is
  the local hour, before and after every DST change (both fall-back 01:00
  hours are observations of season 1; the skipped spring hour has none).

TMP-004 expects the exact calendar grid of buckets; TMP-005, 006 and 008 use
only the order of the buckets. No series are examined until they are
configured, so these rules skip by default.

``TMP-002`` and ``TMP-003`` need the ``stats`` extra (``pip install qavf[stats]``);
without it they report SKIPPED with an explanation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, localcontext
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl
from pydantic import Field

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.inference._support import scipy_stats
from qavf.inference.multiple import adjust
from qavf.inference.pvalues import t_p_value
from qavf.inference.results import Correction
from qavf.inference.trend import kendall_tau
from qavf.numeric import exact_number, numeric_close
from qavf.redaction import exception_details
from qavf.rules._contracts import isolate_contract, unassessed_contract
from qavf.rules._exact_grouped import exact_grouped, supports_exact
from qavf.rules._execution import projected_frame
from qavf.rules._grouped import (
    ProjectedRelation,
    duplicate_groups,
    grouped_aggregate,
    normalized_relation,
    scoped_relation,
    supports_grouped,
)
from qavf.rules._ordered_grouped import ordered_grouped
from qavf.rules.helpers import (
    AggregatedMetric,
    BoundedEvidence,
    above,
    aggregate,
    align_keys,
    annotate,
    duplicated_groups,
    fresh_internal,
    normalize_missing,
    plural,
    present,
    shared,
    usable,
)
from qavf.settings import Bound, NumericBound, QAVFSettings
from qavf.sources.sql import quote_identifier
from qavf.statistics.series import best_split as _best_split
from qavf.statistics.series import normal_equations
from qavf.stats import OutlierMethod, checked_float_series, detect_outliers
from qavf.temporal.buckets import (
    TimeBucketing,
    bucket_grid,
    bucket_labels,
    bucket_scalar,
    calendar_position,
    sequence_positions,
)

logger = logging.getLogger(__name__)

CATEGORY = "temporal"


def _name(spec: SeriesSpec, role: str) -> str:
    """Allocate intermediate roles independently of user key/time names."""
    return fresh_internal(role, (*spec.group_by, spec.time_column))


_MISSING_STATS = "Not assessed: this rule needs the stats extra (pip install 'qavf[stats]')."


class SeriesSpec(TimeBucketing, AggregatedMetric):
    """A series: ``value`` combined (``aggregate``, a sum by default) per
    ``time_column`` value (or ``bucket`` of it) within ``group_by``."""

    time_column: str = "year"
    """Numeric time column (e.g. a year), or a Date, Datetime or text column with
    ``bucket``. Rates of change are per unit of it (per bucket of the exact bucket
    sequence when bucketed); seasons use the local calendar slot."""
    expected_trend: Literal["increasing", "decreasing"] | None = None
    """Direction the series should move in; used by the trend-direction rule."""
    max_change_rate: float | None = Field(default=None, gt=0)
    """Per-series override of the step-change threshold."""


class SeriesParams(RuleParams):
    """Base parameters: which series a temporal rule examines."""

    series: tuple[SeriesSpec, ...] = ()
    """Series to examine; the rule skips until at least one is configured."""


class _TemporalRule(Rule):
    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        names = {spec.dataset for spec in self.params.series}
        if reference := getattr(self.params, "groups_from", None):
            names.add(reference)
        return DataRequirements(frozenset(names))


def _aggregate(
    ctx: DataContext, spec: SeriesSpec, blocked: list[Finding], require_finite: bool = True
) -> pl.DataFrame | None:
    """One row per group and period, sorted.

    Returns ``None`` when the data cannot serve ``spec``, after recording an
    unassessed finding in ``blocked``: a configured series must never vanish.
    With ``require_finite``, groups containing a NaN or infinite period value
    are reported in ``blocked`` and left out, because no trend statistic can be
    trusted on them; the rest of the series is still assessed.
    """
    # Every temporal rule needs the same aggregate for the same series: computed once.
    key = (
        "temporal-series",
        spec.dataset,
        spec.group_by,
        spec.time_column,
        spec.value,
        spec.aggregate,
        spec.weight,
        spec.order_by,
        spec.bucketing.model_dump_json(),
    )
    series = shared(ctx, key, blocked, lambda found: _aggregate_raw(ctx, spec, found))
    if series is None or not require_finite:
        return series
    # Null here means "no usable value" for the period (all inputs missing or NaN).
    bad = series.filter(
        pl.col(_name(spec, "value")).is_null()
        | ~pl.col(_name(spec, "value")).cast(pl.Float64).is_finite()
    )
    if bad.is_empty():
        return series
    blocked.append(
        Finding.unassessed(
            f"Not assessed: {plural(bad.height, 'period value')} of series {spec.label} "
            f"{'is' if bad.height == 1 else 'are'} missing or non-finite (no usable value, NaN "
            f"or infinite); the affected group(s) could not be tested.",
            check=f"{spec.label}: missing or non-finite values",
            affected=present(bad.head(ctx.evidence_limit), {_name(spec, "value"): "value"}),
            affected_count=bad.height,
            unit="period values",
            datasets=(spec.dataset,),
        )
    )
    keys = list(spec.group_by)
    remaining = (
        series.join(bad.select(keys).unique(), on=keys, how="anti", nulls_equal=True)
        if keys
        else series.head(0)
    )
    return remaining if not remaining.is_empty() else None


def _aggregate_raw(
    ctx: DataContext, spec: SeriesSpec, blocked: list[Finding]
) -> pl.DataFrame | None:
    """Aggregate the series per group and period; ``None`` (recorded in ``blocked``) if
    unusable."""
    columns = (*spec.group_by, spec.time_column, spec.value, *spec.aggregation_columns)
    if reason := usable(ctx, spec.dataset, columns):
        blocked.append(
            Finding.unassessed(
                f"Not assessed: series {spec.label}: {reason}.",
                check=spec.label,
                datasets=(spec.dataset,),
            )
        )
        return None
    keys = [*spec.group_by, spec.time_column]
    exact = spec.aggregate in {"first", "last"} or supports_exact(
        ctx.schema(spec.dataset)[spec.value], spec.aggregate
    )
    # Buckets are computed on the scoped lazy relation, so every source kind
    # (including native SQL ones) is bucketed by the same native expressions.
    if exact or spec.bucket is not None:
        with scoped_relation(ctx, spec.dataset, columns) as scoped:
            valid = _valid_times(scoped, spec, blocked, ctx.evidence_limit)
            if valid is None:
                return None
            assert isinstance(valid, ProjectedRelation) and valid.lazy is not None
            assert valid.observation is not None
            if spec.aggregate in {"first", "last"}:
                return ordered_grouped(
                    valid.observation,
                    valid.lazy,
                    keys,
                    spec.value,
                    _name(spec, "value"),
                    spec.aggregate,
                    spec.order_by,
                ).sort(keys)
            if exact:
                return exact_grouped(
                    ctx,
                    valid.observation,
                    valid.lazy,
                    keys,
                    spec.value,
                    _name(spec, "value"),
                    spec.aggregate,
                    weight=spec.weight,
                ).sort(keys)
            return _grouped_series(valid, spec, keys, blocked)
    relation = normalized_relation(ctx, spec.dataset, columns)
    source = (
        relation
        if relation is not None and supports_grouped(relation, spec.value, spec.aggregate)
        else normalize_missing(
            ctx, spec.dataset, projected_frame(ctx, spec.dataset, columns), columns
        )
    )
    frame = _valid_times(source, spec, blocked, ctx.evidence_limit)
    if frame is None:
        return None
    return _grouped_series(frame, spec, keys, blocked)


def _grouped_series(
    frame: pl.DataFrame | ProjectedRelation,
    spec: SeriesSpec,
    keys: list[str],
    blocked: list[Finding],
) -> pl.DataFrame | None:
    """Reduce the non-exact aggregates (integer sums, counts, extrema, ``none``)."""
    if spec.aggregate == "none" and (
        repeated := duplicate_groups(frame, keys)
        if isinstance(frame, ProjectedRelation)
        else duplicated_groups(frame, keys)
    ):
        blocked.append(
            Finding.unassessed(
                f"Not assessed: series {spec.label} uses aggregate: none, which needs one "
                f"row per group and period, but {plural(repeated, 'group/period')} "
                f"{'has' if repeated == 1 else 'have'} several; choose an aggregate.",
                check=spec.label,
                datasets=(spec.dataset,),
            )
        )
        return None
    if isinstance(frame, ProjectedRelation):
        result = grouped_aggregate(frame, keys, spec.value, _name(spec, "value"), spec.aggregate)
        assert result is not None
        return result.sort(keys)
    return aggregate(
        frame, keys, spec.value, _name(spec, "value"), spec.aggregate, spec.weight, spec.order_by
    ).sort(keys)


def _valid_times(
    frame: pl.DataFrame | ProjectedRelation,
    spec: SeriesSpec,
    blocked: list[Finding],
    evidence_limit: int = 50,
) -> pl.DataFrame | ProjectedRelation | None:
    """``frame`` without groups that have rows with a missing or non-finite time.

    Time must be numeric (a year, a period number), or bucketed (``bucket``).
    A row without a usable time cannot be placed in any period, so its group's
    totals are incomplete: the group is reported as not assessed, with the
    offending rows, and left out. The rest of the series is still assessed.
    """
    if spec.bucket is not None:
        assert isinstance(frame, ProjectedRelation)
        return _bucketed_times(frame, spec, blocked, evidence_limit)
    dtype = frame.schema[spec.time_column]
    if not dtype.is_numeric():
        blocked.append(
            Finding.unassessed(
                f"Not assessed: series {spec.label}: time column {spec.time_column!r} must be "
                f"numeric (e.g. a year), found {dtype}; set 'bucket' to use calendar periods "
                "of a Date, Datetime or text column.",
                check=spec.label,
                datasets=(spec.dataset,),
            )
        )
        return None
    time = pl.col(spec.time_column)
    invalid = time.is_null() | (~time.is_finite() if dtype.is_float() else pl.lit(False))
    if isinstance(frame, ProjectedRelation):
        quoted = quote_identifier(spec.time_column)
        invalid_sql = f"{quoted} IS NULL" + (
            f" OR NOT isfinite({quoted})" if dtype.is_float() else ""
        )
        count, bad = frame.filter(invalid, invalid_sql).count_sample(evidence_limit)
    else:
        count, bad = _count_sample(frame, invalid, evidence_limit)
    if not count:
        return frame
    keys = list(spec.group_by)
    blocked.append(
        Finding.unassessed(
            f"Not assessed: {plural(count, 'row')} of series {spec.label} "
            f"{'has' if count == 1 else 'have'} a missing or non-finite time value "
            f"({spec.time_column}); the affected group(s) could not be tested.",
            check=f"{spec.label}: invalid time values",
            affected=bad.head(evidence_limit),
            affected_count=count,
            unit="rows",
            datasets=(spec.dataset,),
        )
    )
    if isinstance(frame, ProjectedRelation):
        if not keys:
            return None
        remaining_relation = frame.without_groups(invalid, invalid_sql, keys)
        return remaining_relation if remaining_relation.count() else None
    remaining = (
        frame.join(
            frame.filter(invalid).select(keys).unique(), on=keys, how="anti", nulls_equal=True
        )
        if keys
        else frame.head(0)
    )
    return remaining if not remaining.is_empty() else None


def _bucketed_times(
    relation: ProjectedRelation, spec: SeriesSpec, blocked: list[Finding], evidence_limit: int
) -> ProjectedRelation | None:
    """``relation`` with its time column replaced by bucket labels, natively.

    Rows whose time is missing or cannot be placed in a bucket (unparseable text,
    a naive local time that is ambiguous or skipped in ``time_zone``) exclude
    their group, reported with the offending rows exactly as invalid numeric times.
    """
    assert relation.lazy is not None
    try:
        label, dtype = bucket_labels(spec.time_column, relation.schema[spec.time_column], spec)
    except ValueError as exc:
        blocked.append(
            Finding.unassessed(
                f"Not assessed: series {spec.label}: {exc}.",
                check=spec.label,
                datasets=(spec.dataset,),
            )
        )
        return None
    count, bad = relation.filter(label.is_null(), "").count_sample(evidence_limit)
    labelled = replace(
        relation,
        lazy=relation.lazy.with_columns(label.alias(spec.time_column)),
        schema={**relation.schema, spec.time_column: dtype},
    )
    if not count:
        return labelled
    blocked.append(
        Finding.unassessed(
            f"Not assessed: {plural(count, 'row')} of series {spec.label} "
            f"{'has' if count == 1 else 'have'} a time value ({spec.time_column}) that is "
            f"missing or cannot be placed in a {spec.describe()} bucket (unparseable, or a "
            "local time that is ambiguous or does not exist in the time zone); the affected "
            "group(s) could not be tested.",
            check=f"{spec.label}: invalid time values",
            affected=bad.head(evidence_limit),
            affected_count=count,
            unit="rows",
            datasets=(spec.dataset,),
        )
    )
    if not spec.group_by:
        return None
    remaining = labelled.without_groups(pl.col(spec.time_column).is_null(), "", spec.group_by)
    return remaining if remaining.count() else None


def _sequence(spec: SeriesSpec, series: pl.DataFrame) -> pl.Series:
    """Spacing of each period in time: the period itself, or its exact bucket sequence
    position (consecutive buckets one apart, across DST changes too). For rates,
    trends and anything measured over elapsed periods."""
    if spec.bucket is None:
        return series[spec.time_column]
    return sequence_positions(series[spec.time_column], spec.bucket)


def _calendar(spec: SeriesSpec, series: pl.DataFrame) -> pl.Expr:
    """Calendar slot of each period: the period itself, or its wall-clock bucket position
    (an hourly bucket's local hour mod 24). For seasons and other cycles."""
    time = pl.col(spec.time_column)
    if spec.bucket is None:
        return time
    return calendar_position(time, series.schema[spec.time_column], spec.bucket)


def _groups(
    series: pl.DataFrame, spec: SeriesSpec
) -> Iterator[tuple[dict[str, Any], pl.DataFrame]]:
    """Iterate grouped frames without retaining a Python object for every group."""
    if not spec.group_by:
        yield {}, series
        return
    for key, frame in series.group_by(list(spec.group_by), maintain_order=True):
        yield dict(zip(spec.group_by, key, strict=True)), frame


def _count_sample(frame: pl.DataFrame, predicate: pl.Expr, limit: int) -> tuple[int, pl.DataFrame]:
    """Count on a boolean vector; gather only retained evidence rows."""
    mask = frame.select(predicate.fill_null(False)).to_series()
    count = int(mask.sum())
    if limit == 0 or count == 0:
        return count, frame.head(0)
    return count, frame[mask.arg_true().head(limit)]


def _outcome(
    records: BoundedEvidence,
    used: list[SeriesSpec],
    examined: int,
    fail_message: str,
    ok_message: str,
    blocked: list[Finding],
    untestable: str = "no group has enough periods to test",
    too_short: int = 0,
    *,
    unit: str,
) -> list[Finding]:
    """Shared tail for the temporal rules.

    Series that could not be read are reported by ``blocked``; usable series
    that offered nothing testable are reported as not assessed rather than
    passing vacuously. Groups too short to test are disclosed (``too_short``).
    """
    datasets = tuple(dict.fromkeys(spec.dataset for spec in used))
    metrics = {
        "series": [spec.label for spec in used],
        "examined": examined,
        "too_short": too_short,
    }
    if too_short and examined:
        blocked.append(
            Finding.unassessed(
                f"Not assessed: {too_short} groups had too few periods to test.",
                check="incomplete series assessment",
                affected_count=too_short,
                unit="groups",
                datasets=datasets,
            )
        )
    if too_short:
        note = f" {plural(too_short, 'group')} had too few periods to test."
        fail_message, ok_message = fail_message + note, ok_message + note
    if not used:
        return blocked or [Finding.skip("No series are configured.")]
    if records.count:
        main = Finding.fail(
            fail_message,
            affected=records.records,
            affected_count=records.count,
            total_records=examined,
            unit=unit,
            metrics=metrics,
            datasets=datasets,
        )
    elif not examined:
        main = Finding.unassessed(
            f"Not assessed: {untestable} in {', '.join(s.label for s in used)}.",
            total_records=0,
            unit=unit,
            metrics=metrics,
            datasets=datasets,
        )
    else:
        main = Finding.ok(
            ok_message, total_records=examined, unit=unit, metrics=metrics, datasets=datasets
        )
    return [*blocked, main]


# ----------------------------------------------------------------- TMP-001
class StepChangeParams(SeriesParams):
    """Parameters for :class:`StepChangeRule`."""

    max_change_rate: float = Field(default=0.25, gt=0)
    """Largest accepted compound change per time unit, as a fraction (0.25 = 25 % per
    year when the time column holds years)."""
    min_value: float = Field(default=0.0, ge=0)
    """Ignore steps where both values are at or below this magnitude."""


class StepChangeRule(_TemporalRule):
    """Consecutive periods do not jump implausibly."""

    rule_id: ClassVar[str] = "TMP-001"
    name: ClassVar[str] = "Period-over-period change within bounds"
    description: ClassVar[str] = (
        "The compound change per time unit between consecutive periods stays within the "
        "configured threshold for each series."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "Review the flagged steps against known drivers. A jump in one group only usually "
        "means an input for that period was swapped, duplicated or left at a default."
    )
    Params: ClassVar[type[RuleParams]] = StepChangeParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        steps = too_short = 0
        for spec in self.params.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                series = _aggregate(ctx, spec, blocked)
                if series is None:
                    continue
                used.append(spec)
                too_short += (
                    series.group_by(list(spec.group_by)).len().filter(pl.col("len") < 2).height
                    if spec.group_by
                    else int(series.height < 2)
                )
                limit = spec.max_change_rate or self.params.max_change_rate
                partition = list(spec.group_by) or None
                time, value = pl.col(spec.time_column), pl.col(_name(spec, "value"))

                def lagged(expression: pl.Expr, over: list[str] | None = partition) -> pl.Expr:
                    shifted = expression.shift(1)
                    return shifted.over(over) if over else shifted

                # Bucketed periods are labels; rates are per bucket of their exact
                # sequence (a DST hour is one bucket after the previous one).
                positions = []
                if spec.bucket is not None:
                    position = _name(spec, "position")
                    series = series.with_columns(_sequence(spec, series).alias(position))
                    positions.append(
                        lagged(pl.col(position)).alias(_name(spec, "previous_position"))
                    )
                stepped = series.with_columns(
                    lagged(value).alias(_name(spec, "previous_value")),
                    lagged(time).alias(_name(spec, "previous_period")),
                    *positions,
                ).filter(pl.col(_name(spec, "previous_value")).is_not_null())
                # Steps below min_value were examined and judged immaterial: they count.
                steps += stepped.height
                # Compare signs directly: abs(Int64.min) cannot be represented.
                stepped = stepped.filter(
                    (pl.col(_name(spec, "value")) > self.params.min_value)
                    | (pl.col(_name(spec, "value")) < -self.params.min_value)
                    | (pl.col(_name(spec, "previous_value")) > self.params.min_value)
                    | (pl.col(_name(spec, "previous_value")) < -self.params.min_value)
                )
                periods = (
                    (_name(spec, "previous_position"), _name(spec, "position"))
                    if spec.bucket is not None
                    else (_name(spec, "previous_period"), spec.time_column)
                )
                stepped = stepped.with_columns(_compound_rates(stepped, spec, periods))
                # A step from zero (or a negative base) has no finite rate but is still a jump.
                undefined = pl.col(_name(spec, "change_rate")).is_null() & (
                    pl.col(_name(spec, "value")) != pl.col(_name(spec, "previous_value"))
                )
                count, flagged = _count_sample(
                    stepped,
                    undefined | above(pl.col(_name(spec, "change_rate")).abs(), limit),
                    max(0, ctx.evidence_limit - len(records.records)),
                )
                flagged = present(
                    flagged.drop(
                        [_name(spec, "position"), _name(spec, "previous_position")],
                        strict=False,
                    ),
                    {
                        _name(spec, "value"): "value",
                        _name(spec, "previous_value"): "previous_value",
                        _name(spec, "previous_period"): "previous_period",
                        _name(spec, "change_rate"): "change_rate",
                    },
                )
                records.count += count
                for row in flagged.to_dicts():
                    records.records.append(annotate(row, series=spec.label, threshold=limit))
        return _outcome(
            records,
            used,
            steps,
            f"{plural(records.count, 'step')} exceed the allowed rate of change.",
            f"All {steps:,} period-over-period steps are within bounds.",
            blocked,
            too_short=too_short,
            unit="steps",
        )


def _compound_change(previous: Any, current: Any, before: Any, now: Any) -> float | None:
    """Stable compound rate; exact differences precede approximate transcendental work."""
    if previous <= 0 or current < 0:
        return None
    if current == 0:
        return -1.0
    # Decimal spans the complete Float64 domain including subnormals. This slow
    # path is for Decimal/Int128 values and platforms without extended precision.
    with localcontext() as context:
        context.prec = 2200
        p, c = exact_number(previous), exact_number(current)
        gap = exact_number(now) - exact_number(before)
        if gap <= 0:
            raise ValueError("consecutive periods need a positive time difference")
        delta = (c - p) / p
        log_ratio = (
            math.log1p(float(delta))
            if abs(delta) <= Decimal("0.5")
            else math.log(float(c)) - math.log(float(p))
        )
        exponent = float(exact_number(log_ratio) / gap)
    try:
        return math.expm1(exponent)
    except OverflowError:
        return math.inf


def _compound_rates(
    frame: pl.DataFrame, spec: SeriesSpec, periods: Sequence[str] | None = None
) -> pl.Series:
    """O(n) vector kernel, with an exact scalar fallback for wider numeric domains.

    ``periods`` names the previous and current period positions (the time column
    and its predecessor by default; bucket ordinals for bucketed series).
    """
    earlier, later = periods or (_name(spec, "previous_period"), spec.time_column)
    columns = [_name(spec, "previous_value"), _name(spec, "value"), earlier, later]
    narrow = frame.select(columns)
    wider = any(dtype.is_decimal() or dtype == pl.Int128 for dtype in narrow.dtypes)
    if wider or np.finfo(np.longdouble).nmant < 63:
        return pl.Series(
            _name(spec, "change_rate"),
            [_compound_change(*row) for row in narrow.iter_rows()],
            dtype=pl.Float64,
        )
    previous, current, before, now = (
        np.asarray(narrow[name].to_numpy(), dtype=np.longdouble) for name in columns
    )
    gap = now - before
    if np.any(gap <= 0) or not np.isfinite(gap).all():
        return pl.Series(
            _name(spec, "change_rate"),
            [_compound_change(*row) for row in narrow.iter_rows()],
            dtype=pl.Float64,
        )
    valid = (previous > 0) & (current >= 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        delta = (current - previous) / previous
        log_ratio = np.where(
            np.abs(delta) <= 0.5, np.log1p(delta), np.log(current) - np.log(previous)
        )
        rates = np.asarray(np.expm1(log_ratio / gap), dtype=np.float64)
    if np.isnan(rates[valid]).any():
        raise ValueError("compound rate could not be evaluated numerically")
    return pl.Series(_name(spec, "change_rate"), rates).set(pl.Series(~valid), None)


# ----------------------------------------------------------------- TMP-002
class TrendDirectionParams(SeriesParams):
    """Parameters for :class:`TrendDirectionRule`."""

    min_points: int = Field(default=3, ge=3)
    min_tau: float = Field(default=0.5, gt=0, le=1)
    """Kendall's tau magnitude, in the wrong direction, at which a series is flagged."""
    alpha: float | None = Field(default=None, gt=0, lt=1)
    """When set, additionally require the wrong-way trend to be significant at this level."""
    correction: Correction = "none"
    """Multiple-testing correction of the Kendall p-values across every series examined
    (``bh``, ``by``, ``holm``, ``bonferroni``); applies with ``alpha``."""


class TrendDirectionRule(_TemporalRule):
    """Series with an expected direction actually move that way."""

    rule_id: ClassVar[str] = "TMP-002"
    name: ClassVar[str] = "Trend direction as expected"
    description: ClassVar[str] = (
        "Series declared as increasing or decreasing do not trend the opposite way "
        "(Kendall rank correlation of value against time)."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "Confirm the growth assumptions behind the series. A series moving against its "
        "expected direction usually means periods were loaded in the wrong order or a "
        "growth factor was applied inversely."
    )
    Params: ClassVar[type[RuleParams]] = TrendDirectionParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        if not any(spec.expected_trend for spec in self.params.series):
            return [Finding.skip("No series declares an expected trend.")]
        if scipy_stats() is None:
            return [Finding.unassessed(_MISSING_STATS)]
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        examined = 0
        too_short = 0
        family: list[float] = []
        """Kendall p-values of every examined series with a defined tau."""
        against: list[tuple[int, float, dict[str, Any]]] = []
        """Wrong-way series: their place in ``family``, tau and evidence."""
        for spec in self.params.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                if spec.expected_trend is None:
                    continue
                series = _aggregate(ctx, spec, blocked)
                if series is None:
                    continue
                used.append(spec)
                sign = 1.0 if spec.expected_trend == "increasing" else -1.0
                for key, frame in _groups(series, spec):
                    with isolate_contract(spec.label, (spec.dataset,), blocked, affected=[key]):
                        if frame.height < self.params.min_points:
                            too_short += 1
                            continue
                        result = kendall_tau(
                            _sequence(spec, frame).to_numpy(),
                            frame[_name(spec, "value")].to_numpy(),
                        )
                        examined += 1
                        if not result.assessed:  # an undefined tau (constant values)
                            continue
                        assert result.statistic is not None and result.p_value is not None
                        family.append(result.p_value)
                        tau = result.statistic
                        if sign * tau > -self.params.min_tau:
                            continue
                        record = annotate(
                            key,
                            series=spec.label,
                            expected=spec.expected_trend,
                            kendall_tau=round(tau, 4),
                            periods=frame.height,
                            first_value=frame[_name(spec, "value")][0],
                            last_value=frame[_name(spec, "value")][-1],
                        )
                        against.append((len(family) - 1, tau, record))
        adjusted = adjust(family, self.params.correction)
        for index, tau, record in against:
            if self.params.alpha is not None and adjusted[index] > self.params.alpha:
                continue
            records.append(
                {
                    **record,
                    "p_value": round(family[index], 6),
                    "adjusted_p": round(float(adjusted[index]), 6),
                    "correction": self.params.correction,
                    "effect": round(tau, 4),
                }
            )
        return _outcome(
            records,
            used,
            examined,
            f"{records.count:,} series trend against expectation.",
            f"All {examined:,} series follow their expected direction.",
            blocked,
            too_short=too_short,
            unit="series",
        )


# ----------------------------------------------------------------- TMP-003
class TrendOutlierParams(SeriesParams):
    """Parameters for :class:`TrendOutlierRule`."""

    min_points: int = Field(default=5, ge=4)
    """Fewest periods needed to fit and test a trend."""
    alpha: float = Field(default=0.05, gt=0, lt=1)
    """Significance level of the Bonferroni-adjusted studentized-residual outlier test."""
    min_relative_deviation: float = Field(default=0.05, ge=0)
    """A period is only flagged if it also departs from the fitted trend by at least this
    fraction, so statistically detectable but immaterial wobbles on a near-perfect
    trend are not reported."""
    log_transform: bool = True
    """Fit the trend on log(value) when every value is positive (multiplicative change)."""
    correction: Correction = "bonferroni"
    """Multiple-testing correction of the residual p-values within each series
    (``holm``, ``bh``, ``by``, ``none``)."""


class TrendOutlierRule(_TemporalRule):
    """Individual periods do not fall off their series' own trend line."""

    rule_id: ClassVar[str] = "TMP-003"
    name: ClassVar[str] = "Trend consistency"
    description: ClassVar[str] = (
        "No period is both a statistically significant outlier from its series' fitted "
        "trend (OLS of value on time, Bonferroni-adjusted studentized residuals) and "
        "materially off that trend."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "A single off-trend period suggests its inputs differ in kind from its neighbours. "
        "Compare that period's inputs and processing with the adjacent periods."
    )
    Params: ClassVar[type[RuleParams]] = TrendOutlierParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        if not self.params.series:
            return [Finding.skip("No series are configured.")]
        if scipy_stats() is None:
            return [Finding.unassessed(_MISSING_STATS)]
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        points = 0
        too_short = 0
        for spec in self.params.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                series = _aggregate(ctx, spec, blocked)
                if series is None:
                    continue
                used.append(spec)
                for key, frame in _groups(series, spec):
                    with isolate_contract(spec.label, (spec.dataset,), blocked, affected=[key]):
                        if frame.height < self.params.min_points:
                            too_short += 1
                            continue
                        try:
                            positions = _sequence(spec, frame)
                            times = checked_float_series(positions).to_numpy()
                            values = checked_float_series(frame[_name(spec, "value")]).to_numpy()
                        except ValueError as exc:
                            blocked.append(
                                Finding.unassessed(
                                    f"Not assessed: {spec.label}: {exc}.",
                                    check=f"{spec.label}: precision",
                                    datasets=(spec.dataset,),
                                )
                            )
                            continue
                        test = _trend_outlier_test(
                            times,
                            values,
                            self.params.log_transform,
                            self.params.min_relative_deviation,
                            self.params.correction,
                        )
                        if test is None:
                            blocked.append(
                                Finding.unassessed(
                                    f"Not assessed: {spec.label} could not be fitted numerically.",
                                    check=f"{spec.label}: numerical fit",
                                    affected=[key],
                                    datasets=(spec.dataset,),
                                )
                            )
                            continue
                        points += len(values)
                        flagged = (test.adjusted_p < self.params.alpha) & (
                            np.abs(test.relative_deviation) >= self.params.min_relative_deviation
                        )
                        for index in np.flatnonzero(flagged):
                            time_value = times[index]
                            period = (
                                frame[spec.time_column][int(index)]
                                if spec.bucket is not None
                                else int(time_value)
                                if float(time_value).is_integer()
                                else float(time_value)
                            )
                            records.append(
                                annotate(
                                    {**key, spec.time_column: period},
                                    series=spec.label,
                                    value=float(values[index]),
                                    trend_value=float(test.fitted[index]),
                                    relative_deviation=round(
                                        float(test.relative_deviation[index]), 4
                                    ),
                                    studentized_residual=round(float(test.residuals[index]), 3),
                                    adjusted_p=float(test.adjusted_p[index]),
                                    p_value=float(test.p_value[index]),
                                    correction=self.params.correction,
                                    effect=round(float(test.relative_deviation[index]), 4),
                                )
                            )
        return _outcome(
            records,
            used,
            points,
            f"{plural(records.count, 'period')} fall materially off their series' trend.",
            f"All {points:,} values are consistent with their trends.",
            blocked,
            too_short=too_short,
            unit="values",
        )


@dataclass(frozen=True, slots=True)
class _TrendTest:
    """Per-period outcome of the trend outlier test, in the original units."""

    fitted: np.ndarray
    relative_deviation: np.ndarray
    residuals: np.ndarray
    adjusted_p: np.ndarray
    p_value: np.ndarray
    """Two-sided p-value of each studentized residual, before the correction."""


def _trend_outlier_test(
    times: np.ndarray,
    values: np.ndarray,
    log_transform: bool,
    min_relative_deviation: float = 0.05,
    correction: Correction = "bonferroni",
) -> _TrendTest | None:
    """O(n) externally studentized OLS residuals, with a numerical uncertainty guard.

    Centre before scaling in extended precision: a large offset must not erase
    small but materially important departures. Deleted SSE uses the exact linear
    regression identity, avoiding n separate fits.
    """
    if len(values) < 4 or not (np.isfinite(times).all() and np.isfinite(values).all()):
        return None
    use_log = log_transform and bool((values > 0).all())
    response = np.log(values) if use_log else values
    # Where long double is only double (Windows, Apple Silicon), a near-perfect
    # trend with a large offset loses its small residuals; fit exactly instead.
    exact = np.finfo(np.longdouble).nmant < 63
    core = (_exact_trend_core if exact else _extended_trend_core)(
        times, values, response, use_log
    )
    if core is None or isinstance(core, _TrendTest):
        return core
    x, sxx, residual, sse, fitted, deviation, difference = core
    if not np.isfinite(fitted).all():
        return None
    # A backward-error-size residual may be roundoff. Never erase a materially
    # large original-unit discrepancy by declaring a near-perfect global fit.
    # The exact fit keeps the extended-precision threshold for consistent results.
    rounding = 64 * (2.0**-63 if exact else np.finfo(np.longdouble).eps)
    zeros = np.zeros(len(values))
    if sse <= len(values) * rounding * rounding:
        material = np.abs(deviation) >= min_relative_deviation
        distinguishable = difference > (
            8 * np.abs(np.spacing(np.asarray(values, dtype=float)))
        )
        if np.any(material & distinguishable):
            return None
        return _TrendTest(fitted, deviation, zeros, np.ones(len(values)), np.ones(len(values)))
    leverage = x.dtype.type(1) / len(values) + x * x / sxx
    remaining = 1 - leverage
    if np.any(remaining <= 0):
        return None
    deleted_sse = np.maximum(0, sse - residual * residual / remaining)
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = np.asarray(
            residual / np.sqrt(deleted_sse / (len(values) - 3) * remaining), dtype=float
        )
    # Two-sided Student t p-values (n - 3 degrees of freedom), corrected over the
    # series' residuals: Bonferroni by default, as in the outlier test of Weisberg.
    p_value = np.asarray(t_p_value(studentized, len(values) - 3), dtype=float)
    if np.isnan(studentized).any() or np.isnan(p_value).any():
        return None
    return _TrendTest(fitted, deviation, studentized, adjust(p_value, correction), p_value)


def _relative_deviation(original: Any, fitted: Any) -> np.ndarray:
    return np.asarray(
        np.where(
            fitted != 0,
            (original - fitted) / np.abs(fitted),
            np.where(original == 0, 0, np.inf),
        ),
        dtype=float,
    )


def _constant_trend(values: np.ndarray) -> _TrendTest | None:
    if not bool((values == values[0]).all()):
        return None
    zeros = np.zeros(len(values))
    ones = np.ones(len(values))
    return _TrendTest(values.astype(float), zeros, zeros, ones, ones)


def _extended_trend_core(
    times: np.ndarray, values: np.ndarray, response: np.ndarray, use_log: bool
) -> Any:
    """Vectorized long-double fit: centred x, sxx, scaled residuals, SSE and fit."""
    extended = np.longdouble
    x = np.asarray(times, dtype=extended)
    y = np.asarray(response, dtype=extended)
    x = x - x[0]
    y_origin = y[0]
    y = y - y_origin
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    x_scale = np.max(np.abs(x))
    y_scale = np.max(np.abs(y))
    if x_scale == 0:
        return None
    if y_scale == 0:
        return _constant_trend(values)
    x = x / x_scale
    y = y / y_scale
    x = x - x.mean()
    y_centre = y.mean()
    z = y - y_centre
    sxx = np.dot(x, x)
    if not np.isfinite(sxx) or sxx == 0:
        return None
    fitted_z = np.dot(x, z) / sxx * x
    residual = z - fitted_z
    sse = np.dot(residual, residual)
    fitted_response = (fitted_z + y_centre) * y_scale + y_origin
    original = np.asarray(values, dtype=extended)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fitted = np.exp(fitted_response) if use_log else fitted_response
        deviation = _relative_deviation(original, fitted)
    return (
        x, sxx, residual, sse, np.asarray(fitted, dtype=float), deviation,
        np.abs(original - fitted),
    )


def _scaled_integers(values: np.ndarray) -> tuple[list[int], int]:
    """Float64 values as integers over one common power-of-two denominator."""
    ratios = [value.as_integer_ratio() for value in values.tolist()]
    denominator = max(den for _, den in ratios)
    return [num * (denominator // den) for num, den in ratios], denominator


def _exact_trend_core(
    times: np.ndarray, values: np.ndarray, response: np.ndarray, use_log: bool
) -> Any:
    """The same fit in exact integer arithmetic, rounded once to Float64 at the end.

    This slow path is for platforms without extended precision. Float64 inputs are
    integers over a power-of-two denominator, so the normal equations are exact
    integer expressions and integer true division rounds each result correctly.
    """
    x, _ = _scaled_integers(times)
    y, unit = _scaled_integers(response)
    count = len(y)
    if all(value == x[0] for value in x):
        return None
    y_range = max(abs(value - y[0]) for value in y)
    if y_range == 0:
        return _constant_trend(values)
    # count * (x - mean x), count**2 * Sxx and count * Sxy, exactly.
    terms = normal_equations(x, y)
    sum_y, centred, spread, cross = terms.sum_y, terms.centred, terms.spread, terms.cross
    # With D = count * spread, fitted_i = numerator_i / D and residual_i = r_i / D
    # in the common integer y units; D is positive because x is not constant.
    denominator = count * spread
    numerator = [sum_y * spread + cross * count * c for c in centred]
    residual = [denominator * value - n for value, n in zip(y, numerator, strict=True)]
    widest = max(abs(c) for c in centred)
    scale = denominator * y_range
    shape = (
        np.array([c / widest for c in centred]),
        spread / (widest * widest),
        np.array([r / scale for r in residual]),
        sum(r * r for r in residual) / (scale * scale),
    )
    fitted_response = np.array([n / (denominator * unit) for n in numerator])
    if use_log:
        # exp() of the fitted log is approximate anyway; compare in Float64.
        with np.errstate(over="ignore", invalid="ignore"):
            fitted = np.exp(fitted_response)
        return (
            *shape, fitted, _relative_deviation(values, fitted), np.abs(values - fitted)
        )
    # Response and original values coincide, so the difference is the residual.
    deviation = np.array([
        r / abs(n) if n != 0 else (0.0 if value == 0 else math.inf)
        for r, n, value in zip(residual, numerator, y, strict=True)
    ])
    difference = np.array([abs(r) / (denominator * unit) for r in residual])
    return (*shape, fitted_response, deviation, difference)


# ----------------------------------------------------------------- TMP-004
class MissingPeriodParams(SeriesParams):
    """Parameters for :class:`MissingPeriodRule`."""

    expected_periods: tuple[Bound, ...] | None = None
    """Periods every group must have; the union of observed periods when omitted.
    For a bucketed series these are dates or datetimes (each names its bucket), and
    when omitted every bucket between the first and last observed one is expected."""
    max_expected_periods: int = Field(default=1_000_000, ge=1)
    """Largest calendar grid of a bucketed series; a longer one is not assessed."""
    expected_groups: tuple[dict[str, Any], ...] | None = None
    """Group combinations that must exist (``{county: 48001, pollutant: 2}``), so a
    group missing altogether is reported, not only gaps in groups that exist."""
    groups_from: str | None = None
    """A dataset whose distinct combinations of each series' ``group_by`` columns are
    expected (e.g. an inventory of county x pollutant)."""


class MissingPeriodRule(_TemporalRule):
    """Every group has every period."""

    rule_id: ClassVar[str] = "TMP-004"
    name: ClassVar[str] = "No missing periods"
    description: ClassVar[str] = (
        "Each series group (for example each county and pollutant) has a value for every period."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Re-run or re-export the missing periods for the affected groups. Set "
        "'expected_periods' so a period missing from every group is detected as well."
    )
    Params: ClassVar[type[RuleParams]] = MissingPeriodParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        groups = 0
        for spec in self.params.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                series = _aggregate(ctx, spec, blocked, require_finite=False)
                if series is None:
                    continue
                try:
                    expected = self._expected(ctx, spec, series)
                except ValueError as exc:
                    blocked.append(
                        unassessed_contract(
                            spec.label,
                            exception_details(exc),
                            (spec.dataset,),
                            check=f"{spec.label}: expected periods",
                        )
                    )
                    continue
                try:
                    absent = self._absent_groups(ctx, spec, series, blocked)
                except Exception as exc:
                    blocked.append(
                        unassessed_contract(
                            spec.label,
                            exception_details(exc),
                            (spec.dataset,),
                            check=f"{spec.label}: expected groups",
                        )
                    )
                    continue
                if absent is None:
                    continue
                used.append(spec)
                shown_all = (
                    expected.to_list()
                    if isinstance(expected, pl.Series)
                    else sorted(int(m) if m == int(m) else m for m in expected)
                )
                shown_sample = shown_all[: ctx.evidence_limit]
                for key in absent:
                    groups += 1
                    records.append(
                        annotate(
                            key,
                            series=spec.label,
                            missing_periods=shown_sample,
                            missing_group=True,
                            **(
                                {"missing_period_count": len(shown_all)}
                                if len(shown_all) > ctx.evidence_limit
                                else {}
                            ),
                        )
                    )
                for key, frame in _groups(series, spec):
                    with isolate_contract(spec.label, (spec.dataset,), blocked, affected=[key]):
                        groups += 1
                        # A period whose values are all missing is as absent as one with no rows.
                        with_value = frame.filter(pl.col(_name(spec, "value")).is_not_null())
                        if isinstance(expected, pl.Series):
                            # Labels compare natively: two fall-back hours stay distinct.
                            present_periods = with_value.select(spec.time_column)
                            shown = (
                                expected.to_frame(spec.time_column)
                                .join(present_periods, on=spec.time_column, how="anti")
                                .sort(spec.time_column)[spec.time_column]
                                .to_list()
                            )
                        else:
                            missing = sorted(
                                expected - set(with_value[spec.time_column].to_list())
                            )
                            shown = [int(m) if m == int(m) else m for m in missing]
                        if shown:
                            records.append(
                                annotate(
                                    key,
                                    series=spec.label,
                                    missing_periods=shown[: ctx.evidence_limit],
                                    **(
                                        {"missing_period_count": len(shown)}
                                        if len(shown) > ctx.evidence_limit
                                        else {}
                                    ),
                                )
                            )
        return _outcome(
            records,
            used,
            groups,
            f"{plural(records.count, 'group')} lack one or more periods.",
            f"All {groups:,} groups cover every period.",
            blocked,
            unit="groups",
        )

    def _expected(
        self, ctx: DataContext, spec: SeriesSpec, series: pl.DataFrame
    ) -> set[Any] | pl.Series:
        """Expected periods: numbers, or (bucketed) the ordered bucket labels.

        A bucketed series expects its configured periods' buckets, or else every
        bucket of the calendar grid from its first to its last observed bucket, so
        a bucket missing from every group is found too.

        Raises:
            ValueError: if the expected periods do not suit the time column.
        """
        configured = self.params.expected_periods
        temporal = [p for p in configured or () if isinstance(p, date | datetime)]
        if spec.bucket is None:
            if temporal:
                raise ValueError(
                    "expected_periods are dates or times; set the series' 'bucket' to use them"
                )
            return set(configured or series[spec.time_column].unique().to_list())
        time = spec.time_column
        if configured:
            raw = ctx.schema(spec.dataset)[time]
            labels = [bucket_scalar(p, time, raw, spec) for p in configured]
            return pl.concat(labels).unique().sort()
        first, last = (series.select(bound(time))[time] for bound in (pl.min, pl.max))
        return bucket_grid(first, last, spec.bucket, limit=self.params.max_expected_periods)

    def _absent_groups(
        self, ctx: DataContext, spec: SeriesSpec, series: pl.DataFrame, blocked: list[Finding]
    ) -> Iterator[dict[str, Any]] | list[dict[str, Any]] | None:
        """Expected group combinations with no rows at all; ``None`` when the
        expectation cannot be applied (recorded in ``blocked``)."""
        p = self.params
        if p.expected_groups is None and p.groups_from is None:
            return []
        keys = list(spec.group_by)
        if not keys:
            blocked.append(
                Finding.unassessed(
                    f"Not assessed: series {spec.label} has no group_by, so expected groups "
                    f"cannot apply.",
                    check=f"{spec.label}: expected groups",
                    datasets=(spec.dataset,),
                )
            )
            return None
        actual = series.select(keys).unique()
        parts: list[pl.DataFrame] = []
        if p.expected_groups is not None:
            if any(set(group) != set(keys) for group in p.expected_groups):
                raise ValueError(f"expected_groups entries must name exactly {keys}")
            columns: dict[str, pl.Series] = {}
            for key in keys:
                values = [group[key] for group in p.expected_groups]
                kinds = {type(value) for value in values if value is not None}
                if len(kinds) > 1:
                    raise ValueError(
                        f"expected group {key!r} has heterogeneous types; declare one exact type"
                    )
                columns[key] = (
                    pl.Series(key, values, strict=True) if values else actual[key].head(0)
                )
            parts.append(pl.DataFrame(columns))
        if p.groups_from is not None:
            if reason := usable(ctx, p.groups_from, keys):
                raise ValueError(f"groups_from {reason}")
            relation = normalized_relation(ctx, p.groups_from, keys, require_repeatable=False)
            if relation is not None:
                parts.append(relation.distinct(keys).collect())
            else:
                reference = normalize_missing(
                    ctx, p.groups_from, projected_frame(ctx, p.groups_from, keys), keys
                )
                parts.append(reference.select(keys).unique())
        expected = parts[0]
        for part in parts[1:]:
            expected, part = align_keys(expected, part, keys)
            expected = pl.concat([expected, part])
        expected, actual = align_keys(expected, actual, keys)
        absent = expected.unique(maintain_order=True).join(
            actual, on=keys, how="anti", nulls_equal=True
        )
        return absent.iter_rows(named=True)


# ----------------------------------------------------------------- TMP-005
class StaleValueParams(SeriesParams):
    """Parameters for :class:`StaleValueRule`."""

    max_repeats: int = Field(default=3, ge=2)
    """A run of this many (or more) consecutive periods with the same value is flagged:
    a stuck sensor, a value carried forward, an input that stopped updating."""
    atol: float = Field(default=0.0, ge=0)
    """Values within this absolute difference count as the same."""
    ignore_values: tuple[NumericBound, ...] = ()
    """Values that may legitimately stay constant (e.g. ``0`` for a closed site)."""
    copied_periods: bool = True
    """Also flag a period in which every group (at least two) has exactly the previous
    period's values: an output copied forward instead of recomputed."""


class StaleValueRule(_TemporalRule):
    """Series do not repeat the same value period after period."""

    rule_id: ClassVar[str] = "TMP-005"
    name: ClassVar[str] = "Stale or repeated values"
    description: ClassVar[str] = (
        "No group of a series keeps the same value for max_repeats consecutive periods, "
        "and no period repeats the previous period's values in every group."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "Identical values over several periods usually mean an input stopped updating or "
        "a step reused last period's output. Check the producing step for the flagged "
        "periods; list legitimately constant values in ignore_values."
    )
    Params: ClassVar[type[RuleParams]] = StaleValueParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        p = self.params
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        examined = too_short = 0
        for spec in p.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                series = _aggregate(ctx, spec, blocked)
                if series is None:
                    continue
                used.append(spec)
                group_count = 0
                for key, frame in _groups(series, spec):
                    with isolate_contract(spec.label, (spec.dataset,), blocked, affected=[key]):
                        group_count += 1
                        if frame.height < p.max_repeats:
                            too_short += 1
                            continue
                        examined += 1
                        records.extend(self._runs(key, frame, spec))
                if p.copied_periods and group_count >= 2:
                    records.extend(self._copies(series, spec))
        datasets = tuple(dict.fromkeys(spec.dataset for spec in used))
        metrics = {
            "series": [spec.label for spec in used],
            "examined": examined,
            "too_short": too_short,
        }
        if not used:
            return blocked or [Finding.skip("No series are configured.")]
        if too_short and examined:
            blocked.append(
                Finding.unassessed(
                    f"Not assessed: {too_short} groups have too few periods for stale-run testing.",
                    check="incomplete stale assessment",
                    affected_count=too_short,
                    unit="groups",
                    datasets=datasets,
                )
            )
        if records.count:
            main = Finding.fail(
                f"{plural(records.count, 'stale run or copied period')} in "
                f"{plural(examined, 'group')}.",
                affected=records.records,
                affected_count=records.count,
                unit="runs",
                metrics=metrics,
                datasets=datasets,
            )
        elif not examined:
            main = Finding.unassessed(
                f"Not assessed: no group has at least {p.max_repeats} periods to test.",
                metrics=metrics,
                datasets=datasets,
            )
        else:
            main = Finding.ok(
                f"No stale values in {plural(examined, 'group')}.",
                total_records=examined,
                unit="groups",
                metrics=metrics,
                datasets=datasets,
            )
        return [*blocked, main]

    def _same(self, a: Any, b: Any) -> bool:
        return a is not None and b is not None and numeric_close(a, b, atol=self.params.atol)

    def _runs(
        self, key: dict[str, Any], frame: pl.DataFrame, spec: SeriesSpec
    ) -> Iterator[dict[str, Any]]:
        """Runs of at least ``max_repeats`` equal consecutive values in one group."""
        p = self.params
        periods = frame[spec.time_column].to_list()
        values = frame[_name(spec, "value")].to_list()
        start = 0
        for end in range(1, len(values) + 1):
            if end < len(values) and self._same(values[end], values[start]):
                continue
            length = end - start
            value = values[start]
            ignored = value is not None and any(
                numeric_close(value, v, atol=p.atol) for v in p.ignore_values
            )
            if length >= p.max_repeats and value is not None and not ignored:
                yield annotate(
                    key,
                    series=spec.label,
                    kind="repeated value",
                    value=value,
                    first_period=periods[start],
                    last_period=periods[end - 1],
                    periods=length,
                )
            start = end

    def _copies(self, series: pl.DataFrame, spec: SeriesSpec) -> Iterator[dict[str, Any]]:
        """Compare current groups to the immediately preceding global period natively."""
        time, keys = spec.time_column, list(spec.group_by)
        global_previous = fresh_internal("global_previous", series.columns)
        group_previous = fresh_internal("group_previous", series.columns)
        before = fresh_internal("copy_before", series.columns)
        periods = (
            series.select(time)
            .unique()
            .sort(time)
            .with_columns(pl.col(time).shift(1).alias(global_previous))
        )
        work = series.with_columns(
            pl.col(time).shift(1).over(keys).alias(group_previous),
            pl.col(_name(spec, "value")).shift(1).over(keys).alias(before),
        ).join(periods, on=time, how="left")
        matched = (pl.col(group_previous) == pl.col(global_previous)).fill_null(False)
        same_name = fresh_internal("same", work.columns)
        ignored_name = fresh_internal("ignored", (*work.columns, same_name))
        same_values = []
        ignored_values = []
        for current, previous in work.select(_name(spec, "value"), before).iter_rows():
            same_values.append(numeric_close(current, previous, atol=self.params.atol))
            ignored_values.append(
                any(
                    numeric_close(current, value, atol=self.params.atol)
                    and numeric_close(previous, value, atol=self.params.atol)
                    for value in self.params.ignore_values
                )
            )
        work = work.with_columns(
            pl.Series(same_name, same_values, dtype=pl.Boolean),
            pl.Series(ignored_name, ignored_values, dtype=pl.Boolean),
        )
        same, ignored = pl.col(same_name), pl.col(ignored_name)
        copied = fresh_internal("copied", work.columns)
        informative = fresh_internal("informative", (*work.columns, copied))
        group_count = fresh_internal("groups", (*work.columns, copied, informative))
        summary = (
            work.group_by(time, maintain_order=True)
            .agg(
                pl.col(global_previous).first().alias(global_previous),
                (matched & same).all().alias(copied),
                (matched & ~ignored).sum().alias(informative),
                pl.len().alias(group_count),
            )
            .filter(pl.col(copied) & (pl.col(informative) >= 2))
        )
        for row in summary.sort(time).iter_rows(named=True):
            yield {
                "series": spec.label,
                "kind": "copied period",
                "period": row[time],
                "copied_from": row[global_previous],
                "groups": row[group_count],
            }


# ----------------------------------------------------------------- TMP-006
class CumulativeParams(SeriesParams):
    """Parameters for :class:`CumulativeRule`."""

    allow_reset_to_zero: bool = False
    """Accept a drop to exactly zero (a counter that legitimately rolls over or is reset
    at the start of a reporting cycle); any other decrease is still flagged."""


class CumulativeRule(_TemporalRule):
    """Cumulative series (running totals, odometers, counters) never decrease."""

    rule_id: ClassVar[str] = "TMP-006"
    name: ClassVar[str] = "Cumulative series never decrease"
    description: ClassVar[str] = (
        "Each group of a cumulative series is non-decreasing from period to period; a "
        "decrease means a reset, a lost increment or a mixed-up period."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "A cumulative value that drops was reset or re-based. Find the producing step's "
        "change at that period; set allow_reset_to_zero if resets to zero are expected."
    )
    Params: ClassVar[type[RuleParams]] = CumulativeParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        steps = too_short = 0
        for spec in self.params.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                series = _aggregate(ctx, spec, blocked)
                if series is None:
                    continue
                used.append(spec)
                too_short += (
                    series.group_by(list(spec.group_by)).len().filter(pl.col("len") < 2).height
                    if spec.group_by
                    else int(series.height < 2)
                )
                partition = list(spec.group_by)
                previous = pl.col(_name(spec, "value")).shift(1)
                previous_time = pl.col(spec.time_column).shift(1)
                if partition:
                    previous, previous_time = (
                        previous.over(partition),
                        previous_time.over(partition),
                    )
                work = series.with_columns(
                    previous.alias(_name(spec, "previous_value")),
                    previous_time.alias(_name(spec, "previous_period")),
                )
                steps += work.filter(pl.col(_name(spec, "previous_value")).is_not_null()).height
                violation = pl.col(_name(spec, "value")) < pl.col(_name(spec, "previous_value"))
                if self.params.allow_reset_to_zero:
                    violation &= pl.col(_name(spec, "value")) != 0
                count, sample = _count_sample(
                    work,
                    violation,
                    max(0, ctx.evidence_limit - len(records.records)),
                )
                records.count += count
                for row in sample.iter_rows(named=True):
                    records.records.append(
                        annotate(
                            {k: row[k] for k in spec.group_by},
                            series=spec.label,
                            period=row[spec.time_column],
                            previous_period=row[_name(spec, "previous_period")],
                            previous_value=row[_name(spec, "previous_value")],
                            value=row[_name(spec, "value")],
                            decrease=_difference(
                                row[_name(spec, "previous_value")], row[_name(spec, "value")]
                            ),
                        )
                    )
        return _outcome(
            records,
            used,
            steps,
            f"{plural(records.count, 'decrease')} in cumulative series.",
            f"All {steps:,} steps are non-decreasing.",
            blocked,
            untestable="no group has two periods to compare",
            too_short=too_short,
            unit="steps",
        )


def _difference(a: Any, b: Any) -> int | Decimal:
    """Evidence subtraction cannot overflow or depend on Decimal's ambient context."""
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    with localcontext() as context:
        context.prec = 2200
        return exact_number(a) - exact_number(b)


# ----------------------------------------------------------------- TMP-007
class SeasonalParams(SeriesParams):
    """Parameters for :class:`SeasonalAnomalyRule`."""

    season_length: int = Field(default=12, ge=2)
    """Periods per cycle: 12 for monthly data indexed 1..12 or as a running month
    number, 4 for quarters, 7 for days of the week. The season of a period is
    ``period mod season_length``; for a bucketed series the period is its local
    calendar position, so monthly buckets with 12 give calendar months and hourly
    buckets with 24 the local hour, unchanged by DST."""
    threshold: float = Field(default=3.5, gt=0)
    """Modified z-score (median/MAD) above which a value is anomalous for its season."""
    min_cycles: int = Field(default=4, ge=3)
    """Fewest values of a season (one per cycle) needed to judge it."""
    min_relative_deviation: float = Field(default=0.1, ge=0)
    """Also require this relative difference from the season median."""


class SeasonalAnomalyRule(_TemporalRule):
    """No value is anomalous compared with the same season in other cycles."""

    rule_id: ClassVar[str] = "TMP-007"
    name: ClassVar[str] = "Seasonal anomalies"
    description: ClassVar[str] = (
        "Each value is compared with the values of the same season (month, quarter, "
        "weekday) in the other cycles of its group; robust outliers are flagged."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "A value unlike the same season in other years points to a missing or duplicated "
        "input for that period, or an event worth documenting."
    )
    Params: ClassVar[type[RuleParams]] = SeasonalParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        p = self.params
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        examined = too_short = 0
        season = ""
        for spec in p.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                series = _aggregate(ctx, spec, blocked)
                if series is None:
                    continue
                used.append(spec)
                season = fresh_internal("season", series.columns)
                seasonal = series.with_columns(
                    (_calendar(spec, series) % p.season_length).alias(season)
                )
                for key, frame in _groups(seasonal, spec):
                    with isolate_contract(spec.label, (spec.dataset,), blocked, affected=[key]):
                        for (season_value,), part in frame.group_by(season, maintain_order=True):
                            with isolate_contract(
                                spec.label,
                                (spec.dataset,),
                                blocked,
                                affected=[annotate(key, season=season_value)],
                            ):
                                try:
                                    values = checked_float_series(
                                        part[_name(spec, "value")]
                                    ).to_numpy()
                                except ValueError as exc:
                                    blocked.append(
                                        Finding.unassessed(
                                            f"Not assessed: {spec.label}: {exc}.",
                                            check=f"{spec.label}: precision",
                                            datasets=(spec.dataset,),
                                        )
                                    )
                                    continue
                                if len(values) < p.min_cycles:
                                    too_short += 1
                                    continue
                                result = detect_outliers(
                                    values, OutlierMethod.MODIFIED_ZSCORE, p.threshold
                                )
                                examined += len(values)
                                unit = float(np.abs(values).max())
                                median = float(np.median(values / unit)) * unit if unit else 0.0
                                for index in np.flatnonzero(result.mask):
                                    relative = (
                                        abs(values[index] / median - 1.0) if median else math.inf
                                    )
                                    if relative < p.min_relative_deviation:
                                        continue
                                    records.append(
                                        annotate(
                                            key,
                                            series=spec.label,
                                            period=part[spec.time_column][int(index)],
                                            season=season_value,
                                            value=float(values[index]),
                                            season_median=median,
                                            score=round(float(result.scores[index]), 3),
                                        )
                                    )
        return _outcome(
            records,
            used,
            examined,
            f"{plural(records.count, 'value')} anomalous for their season.",
            f"All {examined:,} values are consistent with their seasons.",
            blocked,
            untestable=f"no season has at least {p.min_cycles} cycles",
            too_short=too_short,
            unit="values",
        )


# ----------------------------------------------------------------- TMP-008
class LevelShiftParams(SeriesParams):
    """Parameters for :class:`LevelShiftRule`."""

    max_work: int = Field(default=20_000_000, ge=1)
    """Maximum observations processed across exact candidate splits for the whole rule.
    A group exceeding the remaining budget is explicitly unassessed; no approximate
    answer is substituted. Work across all configured groups and series is charged."""
    min_segment: int = Field(default=4, ge=2)
    """Fewest periods on each side of a candidate shift."""
    threshold: float = Field(default=5.0, gt=0)
    """Shift in medians, in units of the pooled robust spread (1.4826 x MAD of the
    values around their segment medians), above which a series is flagged."""
    min_relative_shift: float = Field(default=0.2, ge=0)
    """Also require the medians to differ by this fraction of the earlier median."""


class LevelShiftRule(_TemporalRule):
    """No series jumps to a sustained new level (a regime change)."""

    rule_id: ClassVar[str] = "TMP-008"
    name: ClassVar[str] = "Level shifts"
    description: ClassVar[str] = (
        "No group of a series changes to a sustained new level: the best split into two "
        "segments does not separate their medians by more than the threshold (a robust "
        "screen, not a formal change-point test)."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "A sustained step usually follows a method, unit or source change at that period. "
        "Confirm whether the change is real before trusting trends across it."
    )
    Params: ClassVar[type[RuleParams]] = LevelShiftParams

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        p = self.params
        records = BoundedEvidence(ctx.evidence_limit)
        blocked: list[Finding] = []
        used: list[SeriesSpec] = []
        examined = too_short = 0
        remaining_work = p.max_work
        for spec in p.series:
            with isolate_contract(spec.label, (spec.dataset,), blocked):
                series = _aggregate(ctx, spec, blocked)
                if series is None:
                    continue
                used.append(spec)
                for key, frame in _groups(series, spec):
                    with isolate_contract(spec.label, (spec.dataset,), blocked, affected=[key]):
                        try:
                            values = checked_float_series(frame[_name(spec, "value")]).to_numpy()
                        except ValueError as exc:
                            blocked.append(
                                Finding.unassessed(
                                    f"Not assessed: {spec.label}: {exc}.",
                                    check=f"{spec.label}: precision",
                                    datasets=(spec.dataset,),
                                )
                            )
                            continue
                        if len(values) < 2 * p.min_segment:
                            too_short += 1
                            continue
                        work = len(values) * (len(values) - 2 * p.min_segment + 1)
                        if work > remaining_work:
                            blocked.append(
                                Finding.unassessed(
                                    f"Not assessed: {spec.label} exact level-shift scan "
                                    f"needs {work:,} "
                                    f"work units, exceeding remaining work budget "
                                    f"{remaining_work:,} (max_work={p.max_work:,}).",
                                    check=f"{spec.label}: level-shift resource limit",
                                    affected=[key],
                                    metrics={
                                        "work_units": work,
                                        "max_work": p.max_work,
                                        "remaining_work": remaining_work,
                                    },
                                    datasets=(spec.dataset,),
                                )
                            )
                            continue
                        remaining_work -= work
                        best = _best_split(values, p.min_segment, p.min_relative_shift)
                        examined += 1
                        if best is None:
                            continue
                        split, before, after, score = best
                        # _best_split already applies materiality before restoring
                        # rounded evidence medians to their original offset.
                        if score > p.threshold:
                            records.append(
                                annotate(
                                    key,
                                    series=spec.label,
                                    shift_period=frame[spec.time_column][split],
                                    median_before=before,
                                    median_after=after,
                                    score=round(score, 3),
                                )
                            )
        return _outcome(
            records,
            used,
            examined,
            f"{plural(records.count, 'series')} shift to a new level.",
            f"No level shift in {examined:,} series.",
            blocked,
            untestable=f"no group has at least {2 * p.min_segment} periods",
            too_short=too_short,
            unit="series",
        )
