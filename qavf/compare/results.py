"""Immutable, serializable results of a dataset comparison.

Each level reports typed details plus graded checks; a check becomes one CMP-001
finding. Evidence is bounded by the run's record and byte limits while counts
stay complete. Values of arbitrary type (keys, categories) live in
``affected_records``/``metrics`` or in fields with audit encoding, so a result
survives JSON transport from an isolated worker with its types intact.

``significance`` slots hold the statistical judgement of a family of tests
(:class:`~qavf.inference.results.Significance`); they are empty under the
``material`` decision policy, which runs no test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, SerializeAsAny, ValidationInfo, field_serializer, field_validator

from qavf.inference.results import Significance, TestResult
from qavf.reports.models import SnapshotModel, encode_non_finite


class CheckResult(SnapshotModel):
    """One graded check of a level; ``assessed=False`` is never a pass."""

    audit_fields: ClassVar[tuple[str, ...]] = ("affected_records", "metrics")

    check: str
    passed: bool
    assessed: bool = True
    message: str
    affected_count: int = 0
    """Complete count of problems, in ``unit``; evidence may hold fewer."""
    total: int | None = None
    """Items examined, in ``unit``."""
    unit: str | None = None
    affected_records: list[dict[str, Any]] = Field(default_factory=list)
    """Bounded, deterministically ordered evidence."""
    truncated: bool = False
    """Whether counted problems were left out of ``affected_records``."""
    metrics: dict[str, Any] = Field(default_factory=dict)
    significance: Significance | None = None
    """The family of tests this check's decision used, when it used any."""


class LevelResult(SnapshotModel):
    """A level's checks; subclasses add the level's typed details."""

    level: str
    """Registered level name, which selects the result type when reading JSON."""
    name: str
    """Report label (several levels of one kind need distinct names)."""
    checks: list[CheckResult] = Field(default_factory=list)
    significance: Significance | None = None

    detail_fields: ClassVar[tuple[str, ...]] = ()
    """Per-member detail (series, zones, chart values) a byte budget may leave out."""

    @property
    def assessed(self) -> bool:
        return all(check.assessed for check in self.checks)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def without_details(self) -> LevelResult:
        """This result with its :attr:`detail_fields` reset (checks and counts kept)."""
        fields = type(self).model_fields
        return self.model_copy(
            update={name: fields[name].get_default(call_default_factory=True)
                    for name in self.detail_fields}
        )  # fmt: skip


class _Values(SnapshotModel):
    """Encode non-finite and rich values in ``baseline``/``candidate`` fields."""

    audit_fields: ClassVar[tuple[str, ...]] = ("baseline", "candidate")

    @field_serializer("baseline", "candidate", when_used="json", check_fields=False)
    def _encode_values(self, value: Any) -> Any:
        return encode_non_finite(value)


# ---------------------------------------------------------------- structure
class TypeChange(SnapshotModel):
    column: str
    baseline: str
    candidate: str


class UnitChange(SnapshotModel):
    column: str
    baseline: str | None
    candidate: str | None


class NullabilityChange(SnapshotModel):
    column: str
    baseline_missing: int
    candidate_missing: int
    baseline_declared: bool | None = None
    """Declared ``nullable`` in the baseline schema, when declared."""
    candidate_declared: bool | None = None


class RenameSuggestion(SnapshotModel):
    """A removed and an added column that may be one renamed column. Never applied."""

    removed: str
    added: str
    score: float
    name_similarity: float
    value_similarity: float | None = None
    """Jaccard similarity of bounded sorted samples of distinct values."""


class StructureResult(LevelResult):
    baseline_rows: int = 0
    candidate_rows: int = 0
    columns_added: list[str] = Field(default_factory=list)
    columns_removed: list[str] = Field(default_factory=list)
    type_changes: list[TypeChange] = Field(default_factory=list)
    unit_changes: list[UnitChange] = Field(default_factory=list)
    nullability_changes: list[NullabilityChange] = Field(default_factory=list)
    rename_suggestions: list[RenameSuggestion] = Field(default_factory=list)
    duplicate_keys: dict[str, int] = Field(default_factory=dict)
    """Key values repeated on each side (groups of rows sharing one key)."""


# ----------------------------------------------------------------- coverage
class NullRateChange(SnapshotModel):
    column: str
    baseline_rate: float | None
    candidate_rate: float | None
    change: float | None


class CoverageResult(LevelResult):
    keys: list[str] = Field(default_factory=list)
    baseline_keys: int = 0
    candidate_keys: int = 0
    shared_keys: int = 0
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    rows_only_in_baseline: int = 0
    rows_only_in_candidate: int = 0
    null_key_rows: dict[str, int] = Field(default_factory=dict)
    group_by: list[str] = Field(default_factory=list)
    null_rates: list[NullRateChange] = Field(default_factory=list)


# ------------------------------------------------------------------ records
class Transition(_Values):
    """How often one baseline value became one candidate value."""

    baseline: Any = None
    candidate: Any = None
    count: int


class ColumnChanges(SnapshotModel):
    column: str
    changed: int = 0
    """Values that differ beyond tolerance (the regression engine's decision)."""
    rate: float | None = None
    """``changed`` over matched rows."""
    max_abs_diff: float | None = None
    """Largest absolute difference among differing finite pairs, within tolerance or not."""
    max_rel_diff: float | None = None
    null_to_value: int = 0
    value_to_null: int = 0
    nan_to_value: int = 0
    value_to_nan: int = 0
    transitions: list[Transition] = Field(default_factory=list)
    """Most frequent ``baseline -> candidate`` changes of a categorical column."""
    distinct_transitions: int | None = None
    """All distinct changes of a categorical column, including those not listed."""
    significance: Significance | None = None
    """Paired tests of the column's shift (per ``shift_by`` group), when run."""

    @field_serializer("max_abs_diff", "max_rel_diff", when_used="json")
    def _encode(self, value: Any) -> Any:
        return encode_non_finite(value)


class RecordsResult(LevelResult):
    keys: list[str] = Field(default_factory=list)
    matched: int = 0
    rows_added: int = 0
    rows_removed: int = 0
    rows_changed: int = 0
    rows_unchanged: int = 0
    values_changed: int = 0
    columns: list[ColumnChanges] = Field(default_factory=list)
    execution: str | None = None
    """How the keyed comparison ran (see :class:`qavf.regression.ComparisonResult`)."""
    shifted: list[str] = Field(default_factory=list)
    """Columns whose systematic shift fails the decision policy."""


# ------------------------------------------------------------------- groups
class MeasureComparison(SnapshotModel):
    label: str
    measure: str
    column: str | None = None
    comparators: list[str] = Field(default_factory=list)
    compared: int = 0
    """Groups present on both sides."""
    material: int = 0
    """Groups whose values differ beyond the tolerance."""
    rtol: float = 0.0
    atol: float = 0.0
    significance: Significance | None = None


class RankChange(SnapshotModel):
    measure: str
    top_k: int
    entered: int = 0
    """Groups in the candidate's top ``k`` but not the baseline's."""
    left: int = 0
    moved: int = 0
    """Groups in both top lists at a different rank."""


class GroupsResult(LevelResult):
    group_by: list[str] = Field(default_factory=list)
    baseline_groups: int = 0
    candidate_groups: int = 0
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    measures: list[MeasureComparison] = Field(default_factory=list)
    rank: RankChange | None = None


class _Audited(SnapshotModel):
    """Encode rich values (dates, zoned instants, group keys) in the audit fields."""

    @field_serializer(
        "window_start", "window_end", "group", "peaks", "level_shifts", "clusters", "points",
        "appeared", "disappeared",
        when_used="json", check_fields=False,
    )  # fmt: skip
    def _encode_audited(self, value: Any) -> Any:
        return encode_non_finite(value)


class MemberValues(_Audited):
    """One measure's value on each side per member (group, bucket, zone), for charts.

    Members are in key order and bounded by the comparison's ``chart_points`` and the
    evidence byte limit; ``total`` counts them all.
    """

    audit_fields: ClassVar[tuple[str, ...]] = ("points",)

    measure: str
    keys: list[str] = Field(default_factory=list)
    points: list[dict[str, Any]] = Field(default_factory=list)
    """``{key: ..., "baseline": b, "candidate": c}`` per member (``None``: absent)."""
    total: int = 0
    truncated: bool = False


# ----------------------------------------------------------------- temporal
class TemporalResult(LevelResult, _Audited):
    """Measures per time bucket (within ``group_by``) on each side."""

    audit_fields: ClassVar[tuple[str, ...]] = ("window_start", "window_end")
    detail_fields: ClassVar[tuple[str, ...]] = ("values",)

    time_column: str = ""
    bucket: str = ""
    """How time was bucketed, e.g. ``day in America/Chicago``."""
    group_by: list[str] = Field(default_factory=list)
    baseline_buckets: int = 0
    """Distinct (group, bucket) keys of the baseline."""
    candidate_buckets: int = 0
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    window_start: Any = None
    """First bucket of the window searched for gaps (the sides' common span unless set)."""
    window_end: Any = None
    expected_buckets: int | None = None
    """Buckets of the expected grid in the window, for each group."""
    gaps: dict[str, int] = Field(default_factory=dict)
    """Expected (group, bucket) keys missing on each side."""
    measures: list[MeasureComparison] = Field(default_factory=list)
    """Per measure; its ``significance`` holds the per-bucket tests."""
    values: list[MemberValues] = Field(default_factory=list)
    """Each measure's value per (group, bucket) on both sides, for the report's charts."""


# ------------------------------------------------------------------ shapes
class ShapeAssessment(SnapshotModel):
    """How many series one shape check assessed and found different."""

    check: str
    assessed: int = 0
    different: int = 0
    significance: Significance | None = None


class SeriesShape(_Audited):
    """Shape metrics of one series (one ``group_by`` group) on both sides."""

    audit_fields: ClassVar[tuple[str, ...]] = ("group", "peaks", "level_shifts")

    group: dict[str, Any] = Field(default_factory=dict)
    buckets: int = 0
    """Buckets present on both sides (the paired metrics use these)."""
    pearson: float | None = None
    spearman: float | None = None
    slope_baseline: float | None = None
    """Least-squares change per bucket."""
    slope_candidate: float | None = None
    profile_distance: float | None = None
    """Half the summed absolute differences of the normalized cyclic profiles (0 to 1)."""
    peaks: dict[str, Any] = Field(default_factory=dict)
    """Bucket of each side's peak."""
    peak_shift: int | None = None
    """Candidate peak position minus baseline peak position, in buckets."""
    lag: int | None = None
    """Best-correlated lag: the candidate follows the baseline by this many buckets."""
    lag_correlation: float | None = None
    level_shifts: dict[str, Any] = Field(default_factory=dict)
    """First bucket after each side's level shift (``None``: no shift)."""
    significance: list[TestResult] = Field(default_factory=list)
    """This series' tests (correlation, trend, lag), each labelled with its check."""


class ShapeResult(LevelResult):
    """Shapes of each series (``group_by`` group) of bucketed values compared."""

    detail_fields: ClassVar[tuple[str, ...]] = ("series",)

    time_column: str = ""
    bucket: str = ""
    group_by: list[str] = Field(default_factory=list)
    measure: str = ""
    compared: int = 0
    """Series on both sides."""
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    too_short: int = 0
    """Series on both sides with too few common buckets (not assessed)."""
    assessments: list[ShapeAssessment] = Field(default_factory=list)
    series: list[SeriesShape] = Field(default_factory=list)
    """Per-series metrics in group order, bounded by the evidence limit."""


# ------------------------------------------------------------------ spatial
class ZoneMap(SnapshotModel):
    """Simplified zone outlines for the report's map, in a ``width x height`` view box.

    ``paths`` are SVG path data per zone (its label as text), north up; ``note``
    says why there is no map (no geometry, no spatial extension, too large).
    """

    width: float = 0.0
    height: float = 0.0
    paths: dict[str, str] = Field(default_factory=dict)
    centres: dict[str, list[float]] = Field(default_factory=dict)
    """A label position inside each zone's bounding box."""
    vertices: int = 0
    note: str | None = None


class SpatialResult(LevelResult, _Audited):
    """Measures per zone on each side, expected zones, and spatial patterns of change."""

    audit_fields: ClassVar[tuple[str, ...]] = ("clusters",)
    detail_fields: ClassVar[tuple[str, ...]] = ("values", "clusters", "outlines")

    zone_column: str = ""
    zones: str = ""
    """Where the zones come from: a zone layer, a grid, or the zone column alone."""
    baseline_zones: int = 0
    candidate_zones: int = 0
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    expected_zones: int | None = None
    """Zones of the layer (or the grid's extent) each side should have."""
    missing: dict[str, int] = Field(default_factory=dict)
    """Expected zones each side lacks."""
    measures: list[MeasureComparison] = Field(default_factory=list)
    """Per measure; its ``significance`` holds the per-zone tests."""
    pattern_measure: str | None = None
    """The measure whose per-zone values the pattern checks read."""
    spearman: float | None = None
    """Spearman correlation of the zones' values on the two sides."""
    morans_i: float | None = None
    """Global Moran's I of the per-zone differences."""
    weights: str | None = None
    """The neighbour weights of the pattern checks (method, pairs, islands)."""
    islands: int = 0
    """Compared zones without a neighbour."""
    clusters: list[dict[str, Any]] = Field(default_factory=list)
    """Zones in significant clusters of change (LISA), bounded by the evidence limit."""
    values: list[MemberValues] = Field(default_factory=list)
    """Each measure's value per zone on both sides, for the report's charts and map."""
    outlines: ZoneMap | None = None
    """The zones' simplified outlines for the report's map (``map: false``: none)."""


# ---------------------------------------------------------------- locations
class LocationsResult(LevelResult):
    """Where features matched by key moved, and how polygon features changed."""

    keys: list[str] = Field(default_factory=list)
    kind: str = ""
    """``points`` (coordinate columns) or ``geometries`` (a geometry column)."""
    crs: str = ""
    unit: str = ""
    """Unit of distances: metres (haversine) for a geographic CRS, else CRS units."""
    matched: int = 0
    """Features on both sides."""
    unlocated: int = 0
    """Matched features without a usable location on one or both sides."""
    moved: int = 0
    """Features that moved more than ``max_distance``."""
    max_moved: float | None = None
    """Largest distance a matched feature moved."""
    area_changed: int = 0
    shape_changed: int = 0


# ------------------------------------------------------------- distribution
class QuantileDelta(SnapshotModel):
    """One quantile on each side and whether its change is material."""

    p: float
    baseline: float
    candidate: float
    delta: float
    material: bool = False


class BinShare(SnapshotModel):
    """A shared bin (or category) and its share of each side's values."""

    label: str
    baseline: float
    candidate: float


class DistributionMember(_Audited):
    """One column's distributions in one member (group, zone or bucket) on both sides."""

    audit_fields: ClassVar[tuple[str, ...]] = ("group", "appeared", "disappeared")

    column: str
    kind: str
    """``numeric`` or ``categorical``."""
    group: dict[str, Any] = Field(default_factory=dict)
    baseline_n: int = 0
    candidate_n: int = 0
    psi: float | None = None
    """Population stability index over the baseline's quantile bins (or the categories)."""
    ks: float | None = None
    """Kolmogorov-Smirnov distance (numeric)."""
    anderson_darling: float | None = None
    """Anderson-Darling k-sample statistic, when tests run (numeric)."""
    wasserstein: float | None = None
    """Earth mover's distance in data units (numeric)."""
    jensen_shannon: float | None = None
    """Jensen-Shannon divergence (base 2) on the shared bins or categories."""
    quantiles: list[QuantileDelta] = Field(default_factory=list)
    bins: list[BinShare] = Field(default_factory=list)
    """Each side's share per shared bin or category, for the report's overlays."""
    appeared: list[Any] = Field(default_factory=list)
    """Categories only in the candidate (bounded by the evidence limit)."""
    disappeared: list[Any] = Field(default_factory=list)
    material: list[str] = Field(default_factory=list)
    """The metrics beyond their thresholds."""
    test: TestResult | None = None


class ColumnDistribution(SnapshotModel):
    """How one column's distribution changed across the members compared."""

    column: str
    kind: str
    compared: int = 0
    """Members present on both sides."""
    material: int = 0
    """Members with a metric beyond its threshold."""
    appeared: int = 0
    """Categories (per member) only in the candidate."""
    disappeared: int = 0
    significance: Significance | None = None


class DistributionResult(LevelResult):
    """Per column (and member), distribution distances, quantile deltas and categories."""

    detail_fields: ClassVar[tuple[str, ...]] = ("members",)

    group_by: list[str] = Field(default_factory=list)
    bucket: str | None = None
    """How time was bucketed into members, when it was."""
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    columns: list[ColumnDistribution] = Field(default_factory=list)
    members: list[DistributionMember] = Field(default_factory=list)
    """Per column and member, in order, bounded by the evidence limit."""


# ------------------------------------------------------------ relationships
class CorrelationChange(_Audited):
    """One pair of columns' correlation on each side within one member."""

    audit_fields: ClassVar[tuple[str, ...]] = ("group",)

    group: dict[str, Any] = Field(default_factory=dict)
    method: str
    columns: list[str]
    baseline: float | None = None
    candidate: float | None = None
    pairs: dict[str, int] = Field(default_factory=dict)
    change: float | None = None
    material: bool = False
    test: TestResult | None = None


class RatioChange(_Audited):
    """A declared ratio ``sum(numerator) / sum(denominator)`` per member on each side."""

    audit_fields: ClassVar[tuple[str, ...]] = ("group",)

    name: str
    group: dict[str, Any] = Field(default_factory=dict)
    baseline: float | None = None
    candidate: float | None = None
    change: float | None = None
    """``candidate - baseline``."""
    material: bool = False
    test: TestResult | None = None


class Agreement(_Audited):
    """How a column's matched values agree, candidate against baseline, in one member."""

    audit_fields: ClassVar[tuple[str, ...]] = ("group",)

    column: str
    group: dict[str, Any] = Field(default_factory=dict)
    pairs: int = 0
    bias: float | None = None
    """Mean of ``candidate - baseline`` (Bland-Altman)."""
    sd: float | None = None
    lower: float | None = None
    """Lower 95 % limit of agreement."""
    upper: float | None = None
    ccc: float | None = None
    """Lin's concordance correlation."""
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    smape: float | None = None
    r_squared: float | None = None
    slope: float | None = None
    """``candidate = intercept + slope * baseline`` (``fit``: OLS or Deming)."""
    intercept: float | None = None
    fit: str = "ols"
    material: list[str] = Field(default_factory=list)
    """The metrics beyond their thresholds."""
    test: TestResult | None = None


class RelationshipsResult(LevelResult):
    """Correlation changes, ratio invariants and agreement of matched values."""

    detail_fields: ClassVar[tuple[str, ...]] = ("correlations", "ratios", "agreement")

    group_by: list[str] = Field(default_factory=list)
    bucket: str | None = None
    correlations: list[CorrelationChange] = Field(default_factory=list)
    """Per method, pair and member, in order, bounded by the evidence limit."""
    ratios: list[RatioChange] = Field(default_factory=list)
    agreement: list[Agreement] = Field(default_factory=list)
    matched: int | None = None
    """Keyed rows on both sides (the agreement metrics use these)."""


# ------------------------------------------------------------- contribution
class MemberContribution(_Audited):
    """One member's change of the total and its share of the total change."""

    audit_fields: ClassVar[tuple[str, ...]] = ("group",)

    group: dict[str, Any] = Field(default_factory=dict)
    baseline: float | int = 0
    candidate: float | int = 0
    change: float | int = 0
    exact_change: str = "0"
    """The change as an exact rational, ``"7/2"``."""
    share: float | None = None
    """``change / total change`` (``None`` when the total did not change)."""
    cumulative: float | None = None
    """Share of the total change explained by this and every larger contribution."""
    in_minimal_set: bool = False


class DimensionContribution(SnapshotModel):
    """What explains the total change along one dimension (columns, buckets or both)."""

    dimension: str
    keys: list[str] = Field(default_factory=list)
    members: int = 0
    total_change: float | int = 0
    exact_total_change: str = "0"
    minimal_set: int | None = None
    """Fewest members whose changes explain ``share`` of the total change."""
    explained: float | None = None
    """Share of the total change the minimal set explains."""
    contributions: list[MemberContribution] = Field(default_factory=list)
    """Largest contributions first, bounded by the comparison's ``chart_points``."""
    truncated: bool = False
    significance: Significance | None = None


class RateEffect(_Audited):
    """One stratum's weights and rates on each side and its parts of the rate change."""

    audit_fields: ClassVar[tuple[str, ...]] = ("group",)

    group: dict[str, Any] = Field(default_factory=dict)
    baseline_weight: float = 0.0
    candidate_weight: float = 0.0
    baseline_rate: float | None = None
    candidate_rate: float | None = None
    mix: float = 0.0
    rate: float = 0.0
    interaction: float = 0.0


class RateDecompositionResult(SnapshotModel):
    """An overall rate's change split exactly into mix, rate and interaction effects."""

    name: str
    numerator: str
    denominator: str
    dimension: str
    method: str
    baseline_rate: float = 0.0
    candidate_rate: float = 0.0
    change: float = 0.0
    mix: float = 0.0
    rate: float = 0.0
    interaction: float = 0.0
    exact: dict[str, str] = Field(default_factory=dict)
    """``change``, ``mix``, ``rate`` and ``interaction`` as exact rationals; ``mix + rate +
    interaction == change`` holds exactly."""
    strata: list[RateEffect] = Field(default_factory=list)
    """Largest effects first, bounded by the evidence limit."""
    truncated: bool = False


class ContributionResult(LevelResult):
    """Each member's contribution to the total change, and mix versus rate effects."""

    measure: str = ""
    share: float = 0.8
    """The share of the total change the minimal sets explain."""
    baseline_total: float | int | None = None
    candidate_total: float | int | None = None
    total_change: float | int | None = None
    dimensions: list[DimensionContribution] = Field(default_factory=list)
    rates: list[RateDecompositionResult] = Field(default_factory=list)

    def without_details(self) -> LevelResult:
        return self.model_copy(
            update={
                "dimensions": [d.model_copy(update={"contributions": []}) for d in self.dimensions],
                "rates": [r.model_copy(update={"strata": []}) for r in self.rates],
            }
        )


# --------------------------------------------------------------- comparison
class DatasetComparison(SnapshotModel):
    """Outcome of comparing a candidate dataset with its baseline."""

    name: str
    baseline: str
    candidate: str
    keys: list[str] = Field(default_factory=list)
    column_map: dict[str, str] = Field(default_factory=dict)
    alignment: list[str] = Field(default_factory=list)
    """The alignment steps applied, in order, then their data-dependent notes."""
    baseline_rows: int | None = None
    """Rows after alignment (``None`` when alignment did not complete)."""
    candidate_rows: int | None = None
    error: str | None = None
    """Why nothing could be compared (unavailable data, refused alignment)."""
    levels: list[SerializeAsAny[LevelResult]] = Field(default_factory=list)
    execution_limits: dict[str, Any] = Field(default_factory=dict)
    """Effective isolated worker controls; empty for in-process comparisons."""
    omitted: list[str] = Field(default_factory=list)
    """Detail left out to fit a byte budget (see :meth:`bounded`); empty when complete."""

    @field_validator("levels", mode="before")
    @classmethod
    def _typed_levels(cls, value: Any, info: ValidationInfo) -> Any:
        """Read each level with its registered result type (plugins included).

        JSON input is re-read as JSON so audit-encoded evidence decodes exactly.
        """
        if not isinstance(value, list | tuple):
            return value
        from qavf.compare.levels import LEVELS

        typed = []
        for item in value:
            if isinstance(item, dict) and item.get("level") in LEVELS:
                result = LEVELS.get(item["level"]).Result
                item = (
                    result.model_validate_json(json.dumps(item))
                    if info.mode == "json"
                    else result.model_validate(item)
                )
            typed.append(item)
        return typed

    @property
    def assessed(self) -> bool:
        """Every configured check was assessed."""
        return self.error is None and all(level.assessed for level in self.levels)

    @property
    def passed(self) -> bool:
        """Every check was assessed and passed."""
        return self.assessed and all(level.passed for level in self.levels)

    def level(self, name: str) -> LevelResult:
        """The level reported as ``name``.

        Raises:
            KeyError: if no level has that name.
        """
        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(f"no level named {name!r}")

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.error is not None:
            return f"{self.name}: not assessed: {self.error}"
        failed = [
            f"{level.name}: {check.check}"
            for level in self.levels
            for check in level.checks
            if not check.passed
        ]
        if not failed:
            return f"{self.name}: datasets agree ({len(self.levels)} level(s) compared)"
        return f"{self.name}: differs: " + "; ".join(failed)

    @property
    def verdict(self) -> Literal["passed", "failed", "incomplete"]:
        """``failed`` when an assessed check failed, else ``incomplete`` when something
        was not assessed, else ``passed``."""
        failed = any(
            check.assessed and not check.passed for level in self.levels for check in level.checks
        )
        if failed:
            return "failed"
        return "passed" if self.passed else "incomplete"

    @property
    def exit_code(self) -> int:
        """``0`` when every check was assessed and passed, else ``1`` (as ``qavf run``)."""
        return 0 if self.passed else 1

    # ------------------------------------------------------------ reporting
    def bounded(self, max_bytes: int) -> DatasetComparison:
        """This comparison within ``max_bytes`` of JSON, leaving out detail in order.

        First the levels' per-member detail (series, zones, chart values), then the
        leading tests of each family, then the checks' evidence records; each step
        taken is disclosed in :attr:`omitted`. Verdicts, counts and messages always
        stay. The result may still exceed ``max_bytes`` when they alone do.
        """
        if self.json_size_bytes() <= max_bytes:
            return self
        steps: list[tuple[str, Any]] = [
            ("per-member level details", lambda level: level.without_details()),
            ("tests listed in each significance summary", _without_tests),
            ("check evidence records", _without_evidence),
        ]
        current = self
        omitted = list(self.omitted)
        for disclosure, strip in steps:
            omitted.append(disclosure)
            current = current.model_copy(
                update={"levels": [strip(level) for level in current.levels], "omitted": omitted}
            )
            if current.json_size_bytes() <= max_bytes:
                break
        return current

    def iter_json_chunks(self) -> Iterator[bytes]:
        """The compact UTF-8 JSON encoding (as :meth:`QAReport.iter_json_chunks`)."""
        from qavf.reports.models import _iter_snapshot_json

        buffer = bytearray()
        for chunk in _iter_snapshot_json(self):
            if buffer and len(buffer) + len(chunk) > 32_768:
                yield bytes(buffer)
                buffer.clear()
            buffer.extend(chunk)
        if buffer:
            yield bytes(buffer)

    def json_size_bytes(self) -> int:
        return sum(len(chunk) for chunk in self.iter_json_chunks())

    def write_json(self, path: str | Path, *, max_bytes: int = 16_777_216) -> Path:
        """Write the comparison as JSON, within ``max_bytes`` (detail left out is disclosed
        in ``omitted``); read it back with ``DatasetComparison.model_validate_json``.

        Raises:
            ResourceLimitError: when verdicts, counts and messages alone exceed it.
        """
        from qavf.engine.resources import ResourceLimitError
        from qavf.io import atomic_stream

        fitted = self.bounded(max_bytes)
        size = fitted.json_size_bytes()
        if size > max_bytes:
            raise ResourceLimitError(
                f"the comparison needs {size:,} bytes of JSON without its details; "
                f"max_bytes is {max_bytes:,}"
            )
        with atomic_stream(path) as stream:
            for chunk in fitted.iter_json_chunks():
                stream.write(chunk)
        return Path(path)

    def render_html(self, title: str | None = None, *, limits: Any = None) -> str:
        """This comparison as a self-contained HTML page (the report's comparison section)."""
        from qavf.reports.comparison import render_comparison

        return render_comparison(self, title, limits=limits)

    def write_html(self, path: str | Path, title: str | None = None, *, limits: Any = None) -> Path:
        """Write :meth:`render_html` to ``path``, creating parent directories."""
        from qavf.reports.comparison import write_comparison

        return write_comparison(self, path, title, limits=limits)


def _without_tests(level: LevelResult) -> LevelResult:
    def lean(significance: Significance | None) -> Significance | None:
        return None if significance is None else significance.model_copy(update={"tests": []})

    checks = [c.model_copy(update={"significance": lean(c.significance)}) for c in level.checks]
    return level.model_copy(update={"checks": checks, "significance": lean(level.significance)})


def _without_evidence(level: LevelResult) -> LevelResult:
    checks = [
        check.model_copy(
            update={"affected_records": [], "truncated": bool(check.affected_records)}
        )
        for check in level.checks
    ]
    return level.model_copy(update={"checks": checks})
