"""Spatial (cross-sectional) QA: entity outliers, entity-versus-group comparisons, coverage.

An *entity* is whatever the data is broken down by spatially: counties,
monitoring sites, road links, zones. Entities differ greatly in size, so raw
totals are poor outlier signals; metrics can be *normalised* by another column
(emissions per VMT, crashes per million vehicle-miles) before entities are
compared. Nothing is examined until metrics are configured.
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction
from heapq import nsmallest
from typing import Any, ClassVar

import polars as pl
from pydantic import Field, model_validator

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.rules._contracts import unassessed_contract
from qavf.rules._spatial_aggregation import spatial_aggregate
from qavf.rules.helpers import (
    Aggregate,
    AggregatedMetric,
    OrderBy,
    above,
    align_keys,
    annotate,
    below,
    check_aggregation,
    fresh_internal,
    ordering_columns,
    plural,
    present,
    shared,
    usable,
)
from qavf.settings import QAVFSettings
from qavf.stats import (
    DEFAULT_THRESHOLDS,
    OutlierMethod,
    checked_float_series,
    flag_outliers,
    minimum_samples,
)

logger = logging.getLogger(__name__)

CATEGORY = "spatial"


def _metric_names(ctx: DataContext, spec: SpatialMetric) -> tuple[str, ...]:
    taken = set(ctx.schema(spec.dataset))
    if spec.normalize_by is not None:
        taken.update(ctx.schema(spec.normalize_by.dataset))
    names = []
    for name in (
        "value",
        "denominator",
        "metric",
        "group",
        "group_median",
        "group_entities",
        "ratio_to_group",
        "relative_deviation",
        "problem",
    ):
        allocated = fresh_internal(name, taken)
        taken.add(allocated)
        names.append(allocated)
    return tuple(names)


class Normalizer(RuleParams):
    """Denominator for a metric, combined (``aggregate``, a sum by default) to the keys
    it shares with the metric."""

    dataset: str
    value: str
    on: tuple[str, ...] | None = None
    """Columns to join the denominator on; the metric's keys the denominator also has
    when omitted. Set it so a renamed or extra column cannot change the join grain."""
    aggregate: Aggregate = "sum"
    weight: str | None = None
    order_by: OrderBy = None

    @model_validator(mode="after")
    def _aggregation(self) -> Normalizer:
        check_aggregation(self.aggregate, self.weight, self.order_by)
        return self


class SpatialMetric(AggregatedMetric):
    """A per-entity metric, compared across entities within each ``group_by`` group."""

    entity_column: str
    """Column identifying the entity (county, site, link, …)."""
    normalize_by: Normalizer | None = None

    @model_validator(mode="after")
    def _join_keys(self) -> SpatialMetric:
        on = self.normalize_by.on if self.normalize_by else None
        if on is not None:
            keys = {*self.group_by, self.entity_column}
            if not on or not set(on) <= keys:
                raise ValueError(
                    f"normalize_by.on must be a non-empty subset of the metric's keys "
                    f"{sorted(keys)}, got {list(on)}"
                )
        return self

    @property
    def metric_label(self) -> str:
        """Label including the normaliser, e.g. ``inventory.nox per activity.vmt``."""
        if self.normalize_by is None:
            return self.label
        return f"{self.label} per {self.normalize_by.dataset}.{self.normalize_by.value}"


def _entity_metric(
    ctx: DataContext, spec: SpatialMetric, blocked: list[Finding]
) -> pl.DataFrame | None:
    """The entity metric for ``spec``, built once per run and shared between rules."""
    key = (
        "spatial-metric",
        spec.dataset,
        spec.value,
        spec.entity_column,
        spec.group_by,
        spec.aggregate,
        spec.weight,
        spec.order_by,
        spec.normalize_by,
    )

    def compute(found: list[Finding]) -> pl.DataFrame | None:
        try:
            return _build_entity_metric(ctx, spec, found)
        except Exception as exc:
            found.append(_unassessed_metric(spec, f"{type(exc).__name__}: {exc}"))
            return None

    return shared(ctx, key, blocked, compute)


def _build_entity_metric(
    ctx: DataContext, spec: SpatialMetric, blocked: list[Finding]
) -> pl.DataFrame | None:
    """One row per group and entity with a usable ``metric`` value.

    Returns ``None`` when the data cannot serve ``spec`` at all. Entities whose
    value cannot be computed (no usable numerator value, or a missing, zero,
    negative or non-finite denominator) are reported in ``blocked`` with the
    reason, never silently left out: they are often exactly the ones that need
    review.
    """
    (
        VALUE,
        DENOMINATOR,
        METRIC,
        _GROUP,
        _GROUP_MEDIAN,
        _GROUP_ENTITIES,
        _RATIO,
        _RELATIVE_DEVIATION,
        PROBLEM,
    ) = _metric_names(ctx, spec)
    keys = [*spec.group_by, spec.entity_column]
    if reason := usable(ctx, spec.dataset, (*keys, spec.value, *spec.aggregation_columns)):
        blocked.append(_unassessed_metric(spec, reason))
        return None
    prepared = spatial_aggregate(
        ctx,
        spec.dataset,
        keys,
        spec.value,
        VALUE,
        spec.aggregate,
        spec.weight,
        spec.order_by,
        exclude_missing_keys=True,
    )
    missing_count = prepared.missing_keys
    if missing_count:
        blocked.append(
            Finding.unassessed(
                f"Not assessed: {missing_count} rows of {spec.metric_label} "
                "have missing entity/group keys.",
                check=f"{spec.metric_label}: missing keys",
                affected_count=missing_count,
                affected=prepared.missing_sample,
                unit="rows",
                datasets=(spec.dataset,),
            )
        )
    if repeated := prepared.repeated_groups:
        blocked.append(
            _unassessed_metric(
                spec,
                f"aggregate: none needs one row per entity, but "
                f"{plural(repeated, 'entity')} {'has' if repeated == 1 else 'have'} several",
            )
        )
        return None
    values = prepared.values
    if prepared.error is not None:
        raise prepared.error
    assert values is not None
    numerator = pl.col(VALUE)
    problem = (
        pl.when(numerator.is_null())
        .then(pl.lit("no value"))
        .when(~numerator.cast(pl.Float64).is_finite())
        .then(pl.lit("non-finite value"))
    )
    datasets: tuple[str, ...] = (spec.dataset,)
    norm = spec.normalize_by
    if norm is None:
        joined = values.with_columns(numerator.alias(METRIC))
    else:
        extra = (
            *([norm.weight] if norm.weight is not None else []),
            *ordering_columns(norm.order_by),
        )
        if reason := usable(ctx, norm.dataset, (norm.value, *extra)):
            blocked.append(_unassessed_metric(spec, reason))
            return None
        shared = (
            list(norm.on)
            if norm.on is not None
            else [key for key in keys if key in ctx.schema(norm.dataset)]
        )
        if not shared:
            raise ValueError("normalization needs shared keys; configure normalize_by.on")
        if reason := usable(ctx, norm.dataset, shared):
            blocked.append(_unassessed_metric(spec, reason))
            return None
        prepared_denominator = spatial_aggregate(
            ctx,
            norm.dataset,
            shared,
            norm.value,
            DENOMINATOR,
            norm.aggregate,
            norm.weight,
            norm.order_by,
        )
        if repeated := prepared_denominator.repeated_groups:
            blocked.append(
                _unassessed_metric(
                    spec,
                    f"denominator aggregate: none needs one row per join key; "
                    f"{repeated} denominator groups have several rows",
                )
            )
            return None
        denominator = prepared_denominator.values
        if prepared_denominator.error is not None:
            raise prepared_denominator.error
        assert denominator is not None
        divisor = pl.col(DENOMINATOR)
        problem = (
            problem.when(divisor.is_null())
            .then(pl.lit("missing denominator"))
            .when(~divisor.cast(pl.Float64).is_finite())
            .then(pl.lit("non-finite denominator"))
            .when(divisor == 0)
            .then(pl.lit("zero denominator"))
            .when(divisor < 0)
            .then(pl.lit("negative denominator"))
        )
        # Normalization is itself a conversion boundary. Checking only its
        # floating output cannot recover distinctions lost during division.
        for aggregate_frame, column in ((values, VALUE), (denominator, DENOMINATOR)):
            if aggregate_frame.schema[column].is_numeric():
                checked_float_series(aggregate_frame[column])
        joined = values.join(denominator, on=shared, how="left").with_columns(
            (numerator / divisor).alias(METRIC)
        )
        problem = problem.when((numerator != 0) & (pl.col(METRIC) == 0)).then(
            pl.lit("nonzero normalized metric underflows Float64")
        )
        magnitude = pl.col(METRIC).abs()
        precision_sensitive = joined.filter(
            (magnitude >= 2**52) | ((magnitude > 0) & (magnitude < math.ldexp(1.0, -1022)))
        )
        for numerator_value, denominator_value, result in precision_sensitive.select(
            VALUE, DENOMINATOR, METRIC
        ).iter_rows():
            if (
                denominator_value > 0
                and math.isfinite(result)
                and Fraction(result) != Fraction(numerator_value) / Fraction(denominator_value)
            ):
                domain = "large-value" if abs(result) >= 2**52 else "subnormal-value"
                raise ValueError(f"normalization cannot preserve exact {domain} precision")
        datasets = (spec.dataset, norm.dataset)
    problem = problem.when(~pl.col(METRIC).cast(pl.Float64).is_finite()).then(
        pl.lit("non-finite computed metric")
    )
    labelled = joined.with_columns(problem.otherwise(None).alias(PROBLEM))
    invalid = labelled.filter(pl.col(PROBLEM).is_not_null())
    if not invalid.is_empty():
        evidence = present(
            invalid.drop(METRIC),
            {VALUE: "value", DENOMINATOR: "denominator", PROBLEM: "problem"},
        )
        blocked.append(
            Finding.unassessed(
                f"Not assessed: {plural(invalid.height, 'entity value')} of metric "
                f"{spec.metric_label} could not be computed (no usable value, or a missing, "
                f"zero, negative or non-finite denominator, or an unrepresentable metric).",
                check=f"{spec.metric_label}: unusable values",
                affected=evidence.head(ctx.evidence_limit),
                affected_count=invalid.height,
                total_records=joined.height,
                unit="entity values",
                datasets=datasets,
            )
        )
    return labelled.filter(pl.col(PROBLEM).is_null()).drop(PROBLEM).sort(keys)


def _unassessed_metric(spec: SpatialMetric, reason: str) -> Finding:
    return unassessed_contract(
        f"metric {spec.metric_label}", reason, (spec.dataset,), check=spec.metric_label
    )


# ----------------------------------------------------------------- SPA-001
class EntityOutlierParams(RuleParams):
    """Parameters for :class:`EntityOutlierRule`."""

    metrics: tuple[SpatialMetric, ...] = ()
    """Metrics to compare across entities; the rule skips until one is configured."""
    method: OutlierMethod = OutlierMethod.MODIFIED_ZSCORE
    threshold: float | None = Field(default=None, gt=0)
    """Method threshold; the conventional default for the method when omitted."""
    min_entities: int = Field(default=5, ge=3)
    """Fewest entities in a group for outlier detection to be attempted."""
    min_relative_deviation: float = Field(default=0.10, ge=0)
    """An entity is only flagged if it also differs from the group median by at least
    this fraction. Stops tightly clustered groups, where any tiny spread is
    'statistically' extreme, from producing immaterial findings."""


class EntityOutlierRule(Rule):
    """No entity is a statistical outlier among its peers."""

    rule_id: ClassVar[str] = "SPA-001"
    name: ClassVar[str] = "Entity outlier detection"
    description: ClassVar[str] = (
        "Within each group, no entity's (normalised) metric is a statistical outlier "
        "relative to the other entities (z-score, modified z-score or IQR)."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "Check the outlying entity's inputs first. An entity that is an outlier on every "
        "metric usually has a denominator (activity) problem; one that is an outlier on a "
        "single metric has a problem specific to that measure."
    )
    Params: ClassVar[type[RuleParams]] = EntityOutlierParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset(
                name
                for metric in self.params.metrics
                for name in (
                    metric.dataset,
                    *([metric.normalize_by.dataset] if metric.normalize_by else []),
                )
            )
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        p = self.params
        # A z-score cannot flag anything in small groups: they are untested, not clean.
        limit = DEFAULT_THRESHOLDS[p.method] if p.threshold is None else p.threshold
        minimum = max(p.min_entities, minimum_samples(p.method, limit))
        largest_group = 0
        records: list[dict[str, Any]] = []
        used: list[str] = []
        datasets: list[str] = []
        blocked: list[Finding] = []
        examined = 0
        flagged_count = 0
        ledger: list[dict[str, Any]] = []
        for spec in p.metrics:
            target = {"target": spec.metric_label, "outcome": "unassessed", "examined": 0}
            ledger.append(target)
            blocked_before = len(blocked)
            try:
                frame = _entity_metric(ctx, spec, blocked)
                if frame is None:
                    continue
                (
                    VALUE,
                    DENOMINATOR,
                    METRIC,
                    _GROUP,
                    _GROUP_MEDIAN,
                    _GROUP_ENTITIES,
                    _RATIO,
                    RELATIVE_DEVIATION,
                    _PROBLEM,
                ) = _metric_names(ctx, spec)
                used.append(spec.metric_label)
                datasets.append(spec.dataset)
                if spec.normalize_by is not None:
                    datasets.append(spec.normalize_by.dataset)
                try:
                    scan = flag_outliers(
                        frame, METRIC, spec.group_by, p.method, p.threshold, minimum
                    )
                except ValueError as exc:
                    blocked.append(_unassessed_metric(spec, str(exc)))
                    continue
                sizes = frame.filter(
                    pl.col(METRIC).is_not_null() & pl.col(METRIC).cast(pl.Float64).is_finite()
                )
                if spec.group_by:
                    sizes = sizes.group_by(list(spec.group_by)).len()
                    largest_group = max([largest_group, *sizes["len"].to_list()])
                else:
                    largest_group = max(largest_group, sizes.height)
                median = pl.col(scan.columns.group_median)
                deviation = (
                    pl.when(median != 0)
                    .then(pl.col(METRIC) / median.abs() - median.sign())
                    .otherwise(float("inf"))
                    .alias(RELATIVE_DEVIATION)
                )
                examined += scan.tested
                untested = scan.frame.filter(pl.col(scan.columns.score).is_null())
                if not untested.is_empty():
                    blocked.append(
                        Finding.unassessed(
                            f"Not assessed: {untested.height} entity values of {spec.metric_label} "
                            "belong to groups too small or numerically untestable.",
                            check=f"{spec.metric_label}: incomplete assessment",
                            affected=untested.head(ctx.evidence_limit),
                            affected_count=untested.height,
                            unit="entity values",
                            datasets=(spec.dataset,),
                        )
                    )
                flagged = (
                    scan.outliers.with_columns(deviation)
                    .filter(pl.col(RELATIVE_DEVIATION).abs() >= p.min_relative_deviation)
                    .drop(VALUE, DENOMINATOR, strict=False)
                )
                flagged = present(
                    flagged, {METRIC: "metric", RELATIVE_DEVIATION: "relative_deviation"}
                )
                target.update(
                    outcome=(
                        "unassessed"
                        if not untested.is_empty() or not scan.tested
                        else "fail"
                        if flagged.height
                        else "pass"
                    ),
                    examined=scan.tested,
                    flagged=flagged.height,
                    unassessed=untested.height,
                )
                if scan.frame.is_empty():
                    blocked.append(_unassessed_metric(spec, "no usable entity values"))
                flagged_count += flagged.height  # counted in full; only a sample is kept
                for row in flagged.head(max(0, ctx.evidence_limit - len(records))).to_dicts():
                    records.append(
                        annotate(row, metric_name=spec.metric_label, method=p.method.value)
                    )
            except Exception as exc:
                blocked.append(_unassessed_metric(spec, f"{type(exc).__name__}: {exc}"))
            finally:
                # Metric preparation can omit invalid entities before the outlier
                # scan. Its incomplete assessment belongs to this same target.
                target_blocked = blocked[blocked_before:]
                if target_blocked:
                    counts: dict[str, int] = {}
                    for finding in target_blocked:
                        if finding.affected_count is not None:
                            unit = finding.unit or "unspecified"
                            counts[unit] = counts.get(unit, 0) + finding.affected_count
                    target.update(
                        outcome="unassessed",
                        unassessed=(
                            counts.get("entity values", 0)
                            if all(
                                f.unit == "entity values" and f.affected_count is not None
                                for f in target_blocked
                            )
                            else None
                        ),
                        unassessed_by_unit=counts,
                    )
        if not p.metrics:
            return [Finding.skip("No spatial metrics are configured.")]
        metrics = {
            "metrics": used,
            "method": p.method.value,
            "examined": examined,
            "target_outcomes": ledger,
        }
        if not examined:
            largest = f" (the largest has {largest_group})" if largest_group else ""
            return [
                *blocked,
                Finding.unassessed(
                    f"Not assessed: no group has at least {minimum} entities with a usable "
                    f"metric value{largest}.",
                    metrics=metrics,
                ),
            ]
        unique = tuple(dict.fromkeys(datasets))
        if flagged_count:
            return [
                *blocked,
                Finding.fail(
                    f"{plural(flagged_count, 'outlying entity value')} among {examined:,} "
                    f"({p.method.value}).",
                    affected=records,
                    affected_count=flagged_count,
                    total_records=examined,
                    unit="entity values",
                    metrics=metrics,
                    datasets=unique,
                ),
            ]
        return [
            *blocked,
            Finding.ok(
                f"No outliers among {examined:,} entity values ({p.method.value}).",
                total_records=examined,
                unit="entity values",
                metrics=metrics,
                datasets=unique,
            ),
        ]


# ----------------------------------------------------------------- SPA-002
class GroupComparisonParams(RuleParams):
    """Parameters for :class:`GroupComparisonRule`."""

    metrics: tuple[SpatialMetric, ...] = ()
    """Metrics to compare against group medians."""
    groups: dict[str, tuple[Any, ...]] = Field(default_factory=dict)
    """``group name -> entity ids`` (e.g. regions of counties). All entities form one
    group when empty; entities in no group are not compared."""
    max_ratio: float = Field(default=2.0, gt=1)
    """An entity is flagged when its metric is more than this factor above or below
    its group's median."""
    min_entities: int = Field(default=3, ge=2)
    require_all_groups_per_partition: bool = False
    """Require every configured group within every observed group_by partition.
    By default, only observed group/partition combinations are compared; a group
    missing everywhere is still unassessed. Coverage is a separate contract."""

    @model_validator(mode="after")
    def _unambiguous_groups(self) -> GroupComparisonParams:
        empty = sorted(name for name, ids in self.groups.items() if not ids)
        if empty:
            raise ValueError(
                f"group(s) {', '.join(map(repr, empty))} are empty; list their entities"
            )
        owner: dict[Any, str] = {}
        for name, ids in self.groups.items():
            for entity in ids:
                other = owner.setdefault(entity, name)
                if other != name:
                    raise ValueError(
                        f"entity {entity!r} is in both group {other!r} and group {name!r}; "
                        f"each entity must belong to one group"
                    )
        return self


class GroupComparisonRule(Rule):
    """Entities resemble the group they belong to."""

    rule_id: ClassVar[str] = "SPA-002"
    name: ClassVar[str] = "Entity versus group comparison"
    description: ClassVar[str] = (
        "Each entity's (normalised) metric stays within a configured factor of the median "
        "of its group."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.WARNING
    recommendation: ClassVar[str] = (
        "Entities in one group should behave alike. For a flagged entity, compare its inputs "
        "with a neighbour's, or regroup the entities if it is legitimately different."
    )
    Params: ClassVar[type[RuleParams]] = GroupComparisonParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset(
                name
                for metric in self.params.metrics
                for name in (
                    metric.dataset,
                    *([metric.normalize_by.dataset] if metric.normalize_by else []),
                )
            )
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        p = self.params
        membership = {entity: group for group, ids in p.groups.items() for entity in ids}
        records: list[dict[str, Any]] = []
        used: list[str] = []
        datasets: list[str] = []
        blocked: list[Finding] = []
        examined = 0
        flagged_count = 0
        for spec in p.metrics:
            try:
                frame = _entity_metric(ctx, spec, blocked)
                if frame is None:
                    continue
                (
                    VALUE,
                    DENOMINATOR,
                    METRIC,
                    GROUP,
                    GROUP_MEDIAN,
                    GROUP_ENTITIES,
                    RATIO,
                    _RELATIVE_DEVIATION,
                    _PROBLEM,
                ) = _metric_names(ctx, spec)
                used.append(spec.metric_label)
                datasets.append(spec.dataset)
                if spec.normalize_by is not None:
                    datasets.append(spec.normalize_by.dataset)
                try:
                    checked_float_series(frame[METRIC])
                except ValueError as exc:
                    blocked.append(_unassessed_metric(spec, str(exc)))
                    continue
                entity = pl.col(spec.entity_column)
                if p.groups:
                    configured = pl.Series(
                        spec.entity_column, list(membership), strict=True
                    ).to_frame()
                    aligned, configured = align_keys(
                        frame.select(spec.entity_column), configured, [spec.entity_column]
                    )
                    frame = frame.with_columns(aligned[spec.entity_column])
                    membership = dict(
                        zip(
                            configured[spec.entity_column].to_list(),
                            membership.values(),
                            strict=True,
                        )
                    )
                # Grouping is decided by whether groups are configured, never by what they
                # happen to select.
                group = (
                    entity.replace_strict(membership, default=None, return_dtype=pl.String)
                    if p.groups
                    else pl.lit("ALL")
                )
                partition = [*spec.group_by, GROUP]
                grouped = (
                    frame.filter(pl.col(METRIC).is_not_null())
                    .with_columns(group.alias(GROUP))
                    .filter(pl.col(GROUP).is_not_null())
                    .with_columns(
                        pl.col(METRIC).median().over(partition).alias(GROUP_MEDIAN),
                        pl.len().over(partition).alias(GROUP_ENTITIES),
                    )
                )
                # Every configured partition must remain visible, including implicit
                # ALL groups split by group_by columns.
                sizes = grouped.group_by(partition, maintain_order=True).agg(
                    pl.len().alias(GROUP_ENTITIES)
                )
                absent = sorted(set(p.groups) - set(sizes[GROUP].to_list())) if p.groups else []
                if p.require_all_groups_per_partition and p.groups and spec.group_by:
                    # Assign native partition identities once; Python never compares
                    # temporal/nested keys or scans the whole size table per partition.
                    partition_id = fresh_internal("partition", sizes.columns)
                    partitions = frame.select(spec.group_by).unique(maintain_order=True)
                    partitions = partitions.with_row_index(partition_id)
                    indexed = sizes.join(
                        partitions, on=list(spec.group_by), how="left", nulls_equal=True
                    )
                    observed: dict[int, set[str]] = {}
                    for index, group_name in indexed.select(partition_id, GROUP).iter_rows():
                        observed.setdefault(index, set()).add(group_name)
                    configured_groups = set(p.groups)
                    absent_count = partitions.height * len(configured_groups) - sizes.height
                    absent = []
                    for index in range(partitions.height):
                        remaining = ctx.evidence_limit - len(absent)
                        if not remaining:
                            break
                        seen = observed.get(index, set())
                        if len(seen) < len(configured_groups):
                            absent.extend(nsmallest(remaining, configured_groups - seen))
                else:
                    absent_count = len(absent)
                small = sizes.filter(pl.col(GROUP_ENTITIES) < p.min_entities)
                issue_count = small.height + absent_count
                if issue_count:
                    issues = pl.concat(
                        [
                            small.head(ctx.evidence_limit),
                            pl.DataFrame(
                                {
                                    GROUP: absent[: ctx.evidence_limit],
                                    GROUP_ENTITIES: [0] * min(len(absent), ctx.evidence_limit),
                                },
                                schema={GROUP: pl.String, GROUP_ENTITIES: pl.UInt32},
                            ),
                        ],
                        how="diagonal_relaxed",
                    ).head(ctx.evidence_limit)
                    blocked.append(
                        Finding.unassessed(
                            f"Not assessed: {issue_count} groups of {spec.metric_label} have fewer "
                            f"than {p.min_entities} entities with usable values: "
                            f"{', '.join(str(v) for v in issues[GROUP].to_list())}.",
                            check=f"{spec.metric_label}: groups too small",
                            affected=present(issues, {GROUP: "group", GROUP_ENTITIES: "entities"}),
                            affected_count=issue_count,
                            unit="groups",
                            datasets=(spec.dataset,),
                        )
                    )
                compared = grouped.filter(pl.col(GROUP_ENTITIES) >= p.min_entities).with_columns(
                    # A zero median makes every non-zero value infinitely far from it;
                    # a sign opposite to the median gives a negative ratio, which the
                    # lower bound flags. Nothing is left undefined and silently passed.
                    pl.when(pl.col(GROUP_MEDIAN) != 0)
                    .then(pl.col(METRIC) / pl.col(GROUP_MEDIAN))
                    .when(pl.col(METRIC) == 0)
                    .then(1.0)
                    .otherwise(float("inf"))
                    .alias(RATIO)
                )
                examined += compared.height
                flagged = compared.filter(
                    above(pl.col(RATIO), p.max_ratio) | below(pl.col(RATIO), 1.0 / p.max_ratio)
                ).drop(VALUE, DENOMINATOR, strict=False)
                flagged = present(
                    flagged,
                    {
                        METRIC: "metric",
                        GROUP: "group",
                        GROUP_MEDIAN: "group_median",
                        GROUP_ENTITIES: "group_entities",
                        RATIO: "ratio_to_group",
                    },
                )
                flagged_count += flagged.height  # counted in full; only a sample is kept
                for row in flagged.head(max(0, ctx.evidence_limit - len(records))).to_dicts():
                    records.append(annotate(row, metric_name=spec.metric_label))
            except Exception as exc:
                blocked.append(_unassessed_metric(spec, f"{type(exc).__name__}: {exc}"))
        if not used:
            return blocked or [Finding.skip("No spatial metrics are configured.")]
        metrics = {
            "metrics": used,
            "groups": sorted(p.groups) or ["ALL"],
            "examined": examined,
            "partition_scope": "all groups per observed partition"
            if p.require_all_groups_per_partition
            else "observed combinations",
        }
        if not examined:
            return [
                *blocked,
                Finding.unassessed(
                    f"Not assessed: no group has at least {p.min_entities} entities with a "
                    f"usable metric value.",
                    metrics=metrics,
                ),
            ]
        unique = tuple(dict.fromkeys(datasets))
        if flagged_count:
            return [
                *blocked,
                Finding.fail(
                    f"{plural(flagged_count, 'entity value')} of {examined:,} differ from their "
                    f"group median by more than a factor of {p.max_ratio:g}.",
                    affected=records,
                    affected_count=flagged_count,
                    total_records=examined,
                    unit="entity values",
                    metrics=metrics,
                    datasets=unique,
                ),
            ]
        return [
            *blocked,
            Finding.ok(
                f"All {examined:,} entity values are within a factor of {p.max_ratio:g} of their "
                f"group median.",
                total_records=examined,
                unit="entity values",
                metrics=metrics,
                datasets=unique,
            ),
        ]


# ----------------------------------------------------------------- SPA-003
class CoverageParams(RuleParams):
    """Parameters for :class:`CoverageRule`.

    Temporal identifiers must be representable at microsecond precision; use
    explicit physical integer keys for finer resolution. Unsupported precision
    is unassessed rather than merged during Python-domain conversion.
    """

    entity_column: str | None = None
    """Column identifying entities; the rule skips until this is set."""
    time_column: str | None = None
    """When set (and present), coverage is checked separately for every period."""
    expected_periods: tuple[Any, ...] | None = None
    """Periods every time-indexed dataset must cover; the union of periods found
    across the checked datasets when omitted."""
    expected_entities: tuple[Any, ...] | None = None
    """Entities every dataset must cover. When omitted: the entities of
    ``reference_dataset`` if set, otherwise the union across datasets."""
    reference_dataset: str | None = None
    """Authoritative entity list (e.g. a site inventory), read from its entity column."""
    datasets: tuple[str, ...] | None = None
    require_time_column: bool = False
    """With ``time_column``: a checked dataset without it is not assessed, instead of
    being checked for entities only (so losing the column cannot weaken the check)."""
    unexpected_entities: bool = False
    """Also report entities that are not expected (with ``expected_entities`` or
    ``reference_dataset``): coverage then means the same set, not only a superset."""


class CoverageRule(Rule):
    """Every entity-level dataset covers the same entities (in every period)."""

    rule_id: ClassVar[str] = "SPA-003"
    name: ClassVar[str] = "Entity coverage"
    description: ClassVar[str] = (
        "Each dataset with the entity column contains every expected entity, in every "
        "period when a time column is configured."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "An entity missing from one dataset but present in others is dropped by joins and "
        "understates totals. Add its rows, or list 'expected_entities' explicitly if the "
        "study area is narrower than the data."
    )
    Params: ClassVar[type[RuleParams]] = CoverageParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        if self.params.datasets is None:
            return None  # Auto-discovery includes caller-supplied frames.
        return DataRequirements(
            frozenset(
                (
                    *self.params.datasets,
                    *([self.params.reference_dataset] if self.params.reference_dataset else []),
                )
            )
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        from qavf.rules._coverage import evaluate_coverage

        return evaluate_coverage(ctx, self.params)
