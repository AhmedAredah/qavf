"""Spatial: measures per zone on each side, expected zones, and where changes cluster.

The zone column (an attribute such as ``county``, or one added by the
``assign_zones`` step) is a group key: each side is reduced per zone by the
``groups`` machinery, so measures, comparators, materiality, significance and
the decision policies mean exactly what they mean for groups. On top of that:

* ``zones``: zones on one side only;
* ``expected``: zones of the zone layer (or the grid's ``extent``) a side
  lacks; the zones come from ``layer``/``grid`` or, when omitted, from the
  ``assign_zones`` step that produced the zone column;
* pattern checks on one measure's per-zone values (``pattern_measure``, the
  first by default):

  - ``rank``: Spearman's correlation of the zones' values, against
    ``min_spearman`` (tested with Fisher's z and the Fieller-Hartley-Pearson
    variance: is it below the minimum?);
  - ``shares``: which zones gained or lost share of their side's total beyond
    ``max_change`` (a two-proportion test per zone when the values are counts,
    corrected across zones);
  - ``moran``: global Moran's I of the per-zone ``difference`` (are the changes
    spatially clustered?), material from ``min_i``, tested by permutation;
  - ``lisa``: local Moran's I of the differences per zone, permutation p-values
    corrected across zones, naming the zones in significant clusters of change
    (``hot spot``: a larger change among larger changes, ``cold spot``: the
    opposite) whose own change is material. It needs a test, so it runs under
    every policy but ``material``.

Neighbours (``weights``) come from the zone layer's polygons (queen or rook
contiguity, or nearest neighbours and distance bands of centroids), from grid
cell positions, or from a table of pairs. Pattern checks that need geometry
are not assessed without DuckDB's spatial extension; everything else still
runs. For ``moran`` and ``lisa`` the ``equivalence`` policy means
``material_and_significant``: a pattern is a difference, not an equivalence.
Evidence is bounded and deterministic (largest changes first, zones in order).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, ClassVar, Literal

import numpy as np
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
from qavf.compare.levels.groups import (
    AlignedGroups,
    MeasureSpec,
    compare_measure,
    materiality,
    measure_tests,
    one_sided_check,
    reduce_groups,
)
from qavf.compare.results import CheckResult, LevelResult, SpatialResult
from qavf.compare.significance import (
    Inference,
    MeasureTest,
    Member,
    evidence_fields,
    judge,
    summary_metrics,
)
from qavf.compare.spec import ComparisonSpec, InferenceParams, Side
from qavf.engine.registry import RuleParams
from qavf.inference.registry import TESTS, TestInput
from qavf.inference.results import Alternative, TestResult
from qavf.inference.spatial import (
    DEFAULT_PERMUTATIONS,
    local_morans_i,
    morans_i,
    morans_statistic,
)
from qavf.inference.trend import rank_correlation_test
from qavf.keys import align_keys
from qavf.numeric import exact_rational, report_number
from qavf.rules.helpers import LargestEvidence, plural
from qavf.spatial.crs import DEFAULT_CRS, is_geographic
from qavf.spatial.extension import fetch, spatial_connection
from qavf.spatial.grid import Grid, cell_centres, grid_cells, label_positions
from qavf.spatial.hierarchy import pairs_frame
from qavf.spatial.layers import ZoneLayer, load_zones, zone_ids
from qavf.spatial.weights import (
    SpatialWeights,
    WeightsSpec,
    contiguity,
    distance_band,
    from_table,
    grid_contiguity,
    nearest,
)
from qavf.statistics.series import spearman


class RankCheck(RuleParams):
    """The zones rank alike: Spearman's correlation of their values reaches a minimum."""

    min_spearman: float = Field(default=0.9, gt=-1, lt=1)
    """Lowest Spearman correlation of the zones' values on the two sides."""


class ShareCheck(RuleParams):
    """No zone's share of its side's total changes by more than ``max_change``."""

    max_change: float = Field(default=0.01, ge=0, le=1)
    """Largest change of a zone's share (0.01: one percentage point)."""


class MoranCheck(RuleParams):
    """The per-zone differences are not spatially clustered (global Moran's I)."""

    min_i: float = Field(default=0.1, ge=-1, le=1)
    """Moran's I at or above which clustering of the differences is material."""


class LisaCheck(RuleParams):
    """No zone sits in a significant cluster of material changes (local Moran's I)."""


@dataclass(frozen=True)
class ZoneSource:
    """Where a level's zones come from: a layer, a grid, or the column alone."""

    layer: ZoneLayer | None = None
    grid: Grid | None = None
    crs: str = DEFAULT_CRS
    """CRS of grid cells (the grid's, else the assigned points')."""

    def describe(self) -> str:
        if self.layer is not None:
            return f"zone layer {self.layer.label()!r}"
        if self.grid is not None:
            return self.grid.describe()
        return "zone column"


def zone_source(spec: ComparisonSpec, column: str, layer: Any, grid: Any) -> ZoneSource:
    """The level's own ``layer``/``grid``, else the ``assign_zones`` step producing ``column``.

    With per-side layers and a crosswalk the zones are those of the side the
    crosswalk maps to; without a crosswalk, the candidate's. A crosswalk of both
    sides (a hierarchy) leaves zones without geometry.
    """
    if layer is not None or grid is not None:
        crs = grid.crs if grid is not None and grid.crs else DEFAULT_CRS
        return ZoneSource(layer, grid, crs)
    from qavf.compare.align import ALIGNMENT_STEPS
    from qavf.compare.zones import AssignZonesParams

    for config in spec.steps:
        params = ALIGNMENT_STEPS.create(config.step, config.params).params
        if not isinstance(params, AssignZonesParams):
            continue
        if params.column != column or params.zones is not None:
            continue
        if params.crosswalk is not None and params.crosswalk_side == "both":
            return ZoneSource()  # the zones are the crosswalk's parents, without geometry
        target: Side = "candidate"
        if params.crosswalk is not None and params.crosswalk_side == "candidate":
            target = "baseline"
        grid = params.grid
        crs = grid.crs if grid is not None and grid.crs else params.side_crs(target)
        return ZoneSource(params.side_layer(target), grid, crs)
    return ZoneSource()


class SpatialParams(InferenceParams):
    """Parameters of the ``spatial`` level."""

    zone_column: str = "zone"
    """The zone id column (an attribute, or the ``assign_zones`` step's ``column``)."""
    measures: tuple[MeasureSpec, ...] = Field(
        default_factory=lambda: (MeasureSpec(measure="count"),)
    )
    """Measures compared per zone (``count``, ``sum(x)``, ...)."""
    missing_zones: Literal["fail", "ignore"] = "fail"
    """A zone present on one side only: a difference or left out."""
    expected: bool | None = None
    """Report expected zones (the layer's, or the grid extent's cells) a side lacks:
    whenever zones are known by default, ``true`` to require it, ``false`` to skip."""
    allow_missing: bool = False
    """Report missing expected zones without failing."""
    layer: ZoneLayer | None = None
    """The zones, when not from an ``assign_zones`` step."""
    grid: Grid | None = None
    """Grid cells as zones, when not from an ``assign_zones`` step."""
    weights: WeightsSpec = Field(default_factory=WeightsSpec)
    """Neighbours of the ``moran`` and ``lisa`` checks."""
    pattern_measure: str | None = None
    """The measure label the pattern checks read; the first measure by default."""
    difference: str = "diff"
    """Comparator turning a zone's two values into the difference ``moran`` and ``lisa``
    read (``diff``, ``rel_diff``, ``log_ratio``, ...)."""
    permutations: int = Field(default=DEFAULT_PERMUTATIONS, ge=19)
    """Permutations of the Moran and LISA tests."""
    rank: RankCheck | None = None
    """Spearman correlation of the zones, when their ordering matters (``rank: {}``).

    Opt-in: where zones have similar values, immaterial noise reorders them, so a
    rank check would fail although nothing material changed. Other checks are on
    by default; ``null`` leaves one out."""
    shares: ShareCheck | None = Field(default_factory=ShareCheck)
    """Zones that gained or lost share of their side's total."""
    moran: MoranCheck | None = Field(default_factory=MoranCheck)
    """Global Moran's I of the per-zone differences."""
    lisa: LisaCheck | None = Field(default_factory=LisaCheck)
    """Local Moran's I: the zones in significant clusters of material change."""
    rtol: float | None = Field(default=None, ge=0)
    """Default relative tolerance of this level; the comparison's when omitted."""
    atol: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> SpatialParams:
        if not self.measures:
            raise ValueError("'measures' must name at least one measure")
        labels = [measure.label for measure in self.measures]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"measures repeat the label(s) {', '.join(repeated)}; set 'name'")
        if self.pattern_measure is not None and self.pattern_measure not in labels:
            raise ValueError(f"pattern_measure {self.pattern_measure!r} is not one of {labels}")
        if self.layer is not None and self.grid is not None:
            raise ValueError("zones come from a 'layer' or a 'grid', not both")
        COMPARATORS.create(self.difference)
        return self


@dataclass(frozen=True)
class _Zones:
    """The pattern measure's values on the zones both sides have, in zone order."""

    ids: pl.Series
    baseline: list[Fraction]
    candidate: list[Fraction]


def _floats(values: list[Fraction]) -> np.ndarray:
    return np.array([float(v) for v in values])


def _given(result: TestResult) -> Callable[[], TestResult]:
    """A member's test that already ran (LISA tests the whole family at once)."""
    return lambda: result


@register_level
class SpatialLevel(ComparisonLevel):
    """Measures per zone, zones on one side or missing, rank, share and cluster changes."""

    name: ClassVar[str] = "spatial"
    description: ClassVar[str] = (
        "Measures per zone on each side (attribute zones or zones from assign_zones): zones "
        "on one side only, expected zones a side lacks, material changes per measure, rank "
        "correlation, share redistribution, global Moran's I of the changes and the local "
        "(LISA) clusters of change, corrected for multiple testing."
    )
    Params: ClassVar[type[RuleParams]] = SpatialParams
    Result: ClassVar[type[LevelResult]] = SpatialResult

    def datasets(self) -> frozenset[str]:
        params: SpatialParams = self.params  # type: ignore[assignment]
        found = params.layer.datasets() if params.layer is not None else frozenset()
        table = params.weights.table
        return found | (table.datasets() if table is not None else frozenset())

    def evaluate(self, inputs: LevelInputs, name: str) -> SpatialResult:
        params: SpatialParams = self.params  # type: ignore[assignment]
        column = params.zone_column
        for side in SIDES:
            if column not in inputs.schema(side):
                raise ValueError(f"{side} lacks the zone column {column!r}")
        source = zone_source(inputs.spec, column, params.layer, params.grid)
        inference = Inference.of(inputs.spec, params)
        plans = measure_tests(inputs, params.measures, inputs.relation("baseline"), inference)
        extras = [plan.extras if plan is not None else {} for plan in plans]
        aligned = reduce_groups(inputs, [column], params.measures, extras=extras)
        allowed = params.missing_zones == "ignore"
        checks = [one_sided_check(inputs, aligned, "zones", "zone", allowed)]
        comparisons = []
        for index, (spec, _) in enumerate(aligned.measures):
            judged = materiality(inputs.spec, params, spec)
            comparison, check = compare_measure(
                inputs, aligned, index, judged, "zone", plans[index], inference
            )
            comparisons.append(comparison)
            checks.append(check)
        details: dict[str, Any] = {}
        if params.expected is not False:
            expected = self._expected(inputs, params, source, aligned, details)
            if expected is not None:
                checks.append(expected)
        labels = [spec.label for spec, _ in aligned.measures]
        index = labels.index(params.pattern_measure) if params.pattern_measure else 0
        spec = aligned.measures[index][0]
        judged = materiality(inputs.spec, params, spec)
        zones = self._zones(aligned, index)
        if params.rank is not None:
            checks.append(self._rank(params.rank, zones, inference, details))
        if params.shares is not None:
            checks.append(self._shares(inputs, params.shares, aligned, index, inference))
        if params.moran is not None or (params.lisa is not None and inference.tests):
            checks.extend(self._clusters(inputs, params, source, zones, judged, inference, details))
        return SpatialResult(
            level=self.name,
            name=name,
            checks=checks,
            zone_column=column,
            zones=source.describe(),
            baseline_zones=aligned.frames["baseline"].height,
            candidate_zones=aligned.frames["candidate"].height,
            only_in_baseline=aligned.only_baseline.height,
            only_in_candidate=aligned.only_candidate.height,
            measures=comparisons,
            pattern_measure=spec.label,
            **details,
        )

    # ----------------------------------------------------------- expected zones
    @staticmethod
    def _expected(
        inputs: LevelInputs,
        params: SpatialParams,
        source: ZoneSource,
        aligned: AlignedGroups,
        details: dict[str, Any],
    ) -> CheckResult | None:
        """Expected zones each side lacks; ``None`` when no zones are known (by default)."""
        column = params.zone_column
        try:
            if source.layer is not None:
                expected = zone_ids(inputs.ctx, source.layer)
            elif source.grid is not None and source.grid.extent is not None:
                expected = grid_cells(source.grid)
            elif params.expected:
                raise ValueError("no zone layer or grid extent says which zones to expect")
            else:
                return None
        except Exception as exc:
            return not_assessed("expected", str(exc).rstrip("."))
        grid = expected.alias(column).to_frame()
        evidence = inputs.evidence()
        missing: dict[str, int] = {}
        for side in SIDES:
            present = aligned.frames[side].select(column).drop_nulls()
            want, have = align_keys(grid, present, [column])
            absent = want.join(have, on=column, how="anti").sort(column)
            missing[side] = absent.height
            for (zone,) in absent.iter_rows():
                evidence.append({column: zone, "side": side})
        evidence.count = sum(missing.values())
        details |= {"expected_zones": grid.height, "missing": missing}
        per_side = (
            f"{missing['baseline']:,} in the baseline, {missing['candidate']:,} in the candidate"
        )
        return graded(
            "expected",
            evidence.count,
            f"Both sides have all {plural(grid.height, 'expected zone')} ({source.describe()}).",
            f"{plural(evidence.count, 'expected zone')} missing ({per_side}).",
            evidence=evidence,
            total=2 * grid.height,
            unit="zones",
            allowed=params.allow_missing,
            metrics={"expected": grid.height, "missing": missing},
        )

    # ----------------------------------------------------------------- patterns
    @staticmethod
    def _zones(aligned: AlignedGroups, index: int) -> _Zones:
        """Shared, non-null zones with a numeric value on both sides, in zone order."""
        alias = aligned.aliases[index]
        left, right = (aligned.columns[side][alias] for side in SIDES)
        column = aligned.keys[0]
        ids, before, after = [], [], []
        for zone, b, c in aligned.both.select(column, left, right).iter_rows():
            exact = exact_rational(b), exact_rational(c)
            if zone is None or exact[0] is None or exact[1] is None:
                continue
            ids.append(zone)
            before.append(exact[0])
            after.append(exact[1])
        series = pl.Series(column, ids, dtype=aligned.both.schema[column])
        return _Zones(series, before, after)

    @staticmethod
    def _decided(
        check: str,
        plan: MeasureTest,
        inference: Inference,
        members: list[Member],
        records: list[dict[str, Any]],
        limit: int,
        ok: str,
        failed: str,
        noun: str,
        magnitudes: list[float] | None = None,
    ) -> CheckResult:
        """Judge one family and grade it; members failing carry their test as evidence.

        Failures decide the check (with the not-assessed members counted in its
        message); without failures, members whose test could not run leave it not
        assessed. Evidence keeps the ``limit`` largest ``magnitudes``.
        """
        judgement = judge(plan, inference, members, limit)
        leaders = LargestEvidence(limit)
        shown = judgement.failed or judgement.unassessed
        for position in shown:
            record = records[position] | evidence_fields(
                judgement.results[position], judgement.decisions[position]
            )
            leaders.add((magnitudes or [0.0] * len(records))[position], position, record)
        significance = judgement.significance
        if judgement.unassessed and not judgement.failed:
            reason = judgement.decisions[judgement.unassessed[0]].reason
            return CheckResult(
                check=check,
                passed=False,
                assessed=False,
                message=f"Not assessed: {check}: {plural(len(judgement.unassessed), noun)} need a "
                f"{plan.name} test that could not run ({reason}).",
                affected_count=len(judgement.unassessed),
                total=len(members),
                unit=f"{noun}s",
                affected_records=leaders.records,
                metrics=summary_metrics(significance),
                significance=significance,
            )
        skipped = len(judgement.unassessed)
        result = graded(
            check,
            len(judgement.failed),
            ok,
            failed.format(count=plural(len(judgement.failed), noun))
            + (f" ({skipped:,} not assessed)" if skipped else ""),
            total=len(members),
            unit=f"{noun}s",
            metrics=summary_metrics(significance),
        )
        return result.model_copy(
            update={
                "affected_records": leaders.records,
                "truncated": len(judgement.failed) > len(leaders.records),
                "significance": significance,
            }
        )

    def _rank(
        self, check: RankCheck, zones: _Zones, inference: Inference, details: dict[str, Any]
    ) -> CheckResult:
        if len(zones.ids) < 3:
            return not_assessed("rank", "fewer than 3 zones have a value on both sides")
        rho = spearman(zones.baseline, zones.candidate)
        flat = (len(set(zones.baseline)) == 1, len(set(zones.candidate)) == 1)
        low = flat[0] != flat[1] or (rho is not None and not rho.at_least(check.min_spearman))
        value = None if rho is None else rho.value
        details["spearman"] = value
        record = {"spearman": value, "zones": len(zones.ids), "min_spearman": check.min_spearman}
        how = f"min_spearman={check.min_spearman:g}"
        ok = f"The zones rank alike (Spearman {value if value is None else f'{value:.4g}'}, {how})."
        bad = f"The zones rank differently (Spearman {value}, {how})."
        if not inference.tests:
            return graded("rank", int(low), ok, bad, total=1, unit="measures", metrics=record)
        side: Alternative = "greater" if inference.decision == "equivalence" else "less"
        xb, xc = _floats(zones.baseline), _floats(zones.candidate)
        testable = rho is not None and not any(flat)

        def test() -> TestResult:
            return rank_correlation_test(xb, xc, rho=check.min_spearman, alternative=side)

        member = Member(low, "zones", applicable=testable, run=test if testable else None)
        plan = MeasureTest("spearman", None, testable, None if testable else "a side is flat")
        result = self._decided("rank", plan, inference, [member], [record], 1, ok, bad, "measure")
        return result.model_copy(update={"metrics": {**result.metrics, **record}})

    def _shares(
        self,
        inputs: LevelInputs,
        check: ShareCheck,
        aligned: AlignedGroups,
        index: int,
        inference: Inference,
    ) -> CheckResult:
        alias = aligned.aliases[index]
        column = aligned.keys[0]
        left, right = (aligned.columns[side][alias] for side in SIDES)
        rows = list(aligned.joined.select(column, left, right).iter_rows())
        values = [(zone, exact_rational(b), exact_rational(c)) for zone, b, c in rows]
        totals = [
            sum((v for v in column_values if v is not None), Fraction(0))
            for column_values in ([row[1] for row in values], [row[2] for row in values])
        ]
        if not totals[0] or not totals[1]:
            return not_assessed("shares", "a side's total of the measure is zero")
        limit = Fraction(str(check.max_change))
        whole = all(
            v is not None and v >= 0 and v.denominator == 1 for row in values for v in row[1:]
        )
        test = TESTS.create("two_proportion") if whole else None
        plan = (
            MeasureTest("two_proportion", test)
            if whole
            else MeasureTest("none", None, False, "the values are not counts; judged by size")
        )
        members, records, magnitudes = [], [], []
        changed = 0
        for position, (zone, b, c) in enumerate(values):
            before = (b or Fraction(0)) / totals[0]
            after = (c or Fraction(0)) / totals[1]
            change = after - before
            material = abs(change) > limit
            changed += material
            data = None
            if whole:
                data = TestInput(
                    counts=(float(b or 0), float(c or 0)),
                    totals=(float(totals[0]), float(totals[1])),
                    seed=inference.seed + position,
                    max_work=inference.max_work,
                )
            members.append(Member(material, f"{column}={zone}", data, applicable=whole))
            records.append(
                {
                    column: zone,
                    "baseline_share": report_number(before),
                    "candidate_share": report_number(after),
                    "change": report_number(change),
                    "direction": "gained" if change > 0 else "lost",
                }
            )
            magnitudes.append(float(abs(change)))
        ok = f"No zone's share changed by more than {check.max_change:g} ({len(values):,} zones)."
        bad = "{count} changed share by more than " + f"{check.max_change:g}."
        if not inference.tests:
            leaders = LargestEvidence(inputs.ctx.evidence_limit)
            for position, record in enumerate(records):
                if members[position].material:
                    leaders.add(magnitudes[position], position, record)
            evidence = inputs.evidence()
            evidence.extend(leaders.records)
            evidence.count = changed
            return graded(
                "shares", changed, ok, bad.format(count=plural(changed, "zone")),
                evidence=evidence, total=len(values), unit="zones",
            )  # fmt: skip
        return self._decided(
            "shares",
            plan,
            inference,
            members,
            records,
            inputs.ctx.evidence_limit,
            ok,
            bad,
            "zone",
            magnitudes=magnitudes,
        )

    # ----------------------------------------------------------------- clusters
    def _weights(
        self, inputs: LevelInputs, params: SpatialParams, source: ZoneSource, ids: pl.Series
    ) -> SpatialWeights:
        """Neighbour weights among ``ids`` (the compared zones, in order).

        Raises:
            ValueError: when no neighbour source applies or zones are unknown to it.
            SpatialUnavailable: when polygons are needed without the extension.
        """
        spec = params.weights
        if spec.method == "table":
            assert spec.table is not None
            table = pairs_frame(inputs.ctx, spec.table)
            ids_frame = ids.to_frame()
            inside = table.join(ids_frame.rename({ids.name: "from"}), on="from", how="semi").join(
                ids_frame.rename({ids.name: "to"}), on="to", how="semi"
            )
            weights = from_table(ids, inside)
        elif source.grid is not None:
            grid = source.grid
            if spec.method in ("queen", "rook"):
                positions = label_positions(ids)
                weights = grid_contiguity(ids, positions["ix"], positions["iy"], spec.method)
            else:
                x, y = cell_centres(ids, grid)
                weights = self._by_distance(ids, x, y, spec, is_geographic(source.crs))
        elif source.layer is not None:
            with spatial_connection(inputs.ctx) as con:
                loaded = load_zones(con, inputs.ctx, source.layer)
                index = pl.DataFrame({"id": loaded.ids}).with_row_index("order")
                wanted, known = align_keys(ids.alias("id").to_frame(), index, ["id"])
                found = wanted.join(known, on="id", how="left")
                if unknown := found["order"].null_count():
                    raise ValueError(f"{plural(unknown, 'compared zone')} are not in the layer")
                keep = [int(i) for i in found["order"]]
                if spec.method in ("queen", "rook"):
                    full = contiguity(con, loaded.table, loaded.ids, spec.method)
                    weights = full.subset(keep)
                else:
                    centres = fetch(
                        con,
                        "SELECT zone_order, ST_X(ST_Centroid(geom)) AS x, "
                        f'ST_Y(ST_Centroid(geom)) AS y FROM "{loaded.table}" ORDER BY zone_order',
                    )
                    x, y = centres["x"].to_numpy()[keep], centres["y"].to_numpy()[keep]
                    weights = self._by_distance(ids, x, y, spec, is_geographic(loaded.crs))
        else:
            raise ValueError(
                "no neighbours for the zones: set the level's 'layer' or 'grid', zones from an "
                "assign_zones step, or weights: {method: table}"
            )
        return weights.standardized() if spec.row_standardize else weights

    @staticmethod
    def _by_distance(
        ids: pl.Series, x: Any, y: Any, spec: WeightsSpec, geographic: bool
    ) -> SpatialWeights:
        if spec.method == "knn":
            return nearest(ids, x, y, spec.k, geographic=geographic)
        if spec.method == "distance_band":
            assert spec.distance is not None
            return distance_band(ids, x, y, spec.distance, geographic=geographic)
        raise ValueError(f"weights method {spec.method!r} needs zone polygons or grid cells")

    def _clusters(
        self,
        inputs: LevelInputs,
        params: SpatialParams,
        source: ZoneSource,
        zones: _Zones,
        judged: Materiality,
        inference: Inference,
        details: dict[str, Any],
    ) -> list[CheckResult]:
        """The ``moran`` and ``lisa`` checks of the per-zone differences."""
        names = [n for n, on in (("moran", params.moran), ("lisa", params.lisa)) if on]
        if not inference.tests:
            names = [n for n in names if n != "lisa"]
        comparator = COMPARATORS.create(params.difference)
        totals = ComparatorContext(
            sum(zones.baseline, Fraction(0)), sum(zones.candidate, Fraction(0))
        )
        kept, differences = [], []
        for position, (b, c) in enumerate(zip(zones.baseline, zones.candidate, strict=True)):
            value = comparator.compare(b, c, totals)
            if value is not None:
                kept.append(position)
                differences.append(float(value))
        if len(kept) < 3:
            reason = f"fewer than 3 zones have a defined {params.difference} on both sides"
            return [not_assessed(name, reason) for name in names]
        ids = zones.ids.gather(kept)
        try:
            weights = self._weights(inputs, params, source, ids)
        except Exception as exc:
            return [not_assessed(name, f"neighbour weights: {exc}".rstrip(".")) for name in names]
        details |= {"weights": weights.describe(), "islands": len(weights.islands)}
        if not weights.pairs:
            return [not_assessed(name, "no compared zone has a neighbour") for name in names]
        d = np.array(differences)
        pattern = replace(inference, decision="material_and_significant") if (
            inference.decision == "equivalence"
        ) else inference  # fmt: skip
        results: list[CheckResult] = []
        uniform = bool(np.all(d == d[0]))
        if params.moran is not None:
            results.append(self._moran(params, weights, d, uniform, pattern, details))
        if "lisa" in names:
            results.append(
                self._lisa(
                    inputs, params, zones, kept, weights, d, uniform, judged, pattern, details
                )
            )
        return results

    @staticmethod
    def _moran(
        params: SpatialParams,
        weights: SpatialWeights,
        d: np.ndarray,
        uniform: bool,
        inference: Inference,
        details: dict[str, Any],
    ) -> CheckResult:
        assert params.moran is not None
        threshold = params.moran.min_i
        context = f"{params.difference} per zone, {weights.describe()}"
        if uniform:
            return graded(
                "moran", 0, f"The zones' differences are uniform: no spatial pattern ({context}).",
                "", total=1, unit="measures", metrics={"morans_i": None},
            )  # fmt: skip
        statistic = morans_statistic(d, weights.neighbours)
        details["morans_i"] = statistic
        material = statistic >= threshold
        record = {"morans_i": statistic, "min_i": threshold, "zones": len(d)}
        found = f"Moran's I {statistic:.4g}"
        ok = f"The changes are not clustered ({found}, min_i={threshold:g}; {context})."
        bad = f"The changes cluster in space ({found} >= {threshold:g}; {context})."
        if not inference.tests:
            return graded("moran", int(material), ok, bad, total=1, unit="measures", metrics=record)

        def test() -> TestResult:
            return morans_i(
                d, weights.neighbours, alternative="greater", permutations=params.permutations,
                seed=inference.seed, max_work=inference.max_work,
            )  # fmt: skip

        member = Member(material, "differences", run=test)
        result = SpatialLevel._decided(
            "moran", MeasureTest("morans_i", None), inference, [member], [record], 1, ok, bad,
            "measure",
        )  # fmt: skip
        return result.model_copy(update={"metrics": {**result.metrics, **record}})

    @staticmethod
    def _lisa(
        inputs: LevelInputs,
        params: SpatialParams,
        zones: _Zones,
        kept: list[int],
        weights: SpatialWeights,
        d: np.ndarray,
        uniform: bool,
        judged: Materiality,
        inference: Inference,
        details: dict[str, Any],
    ) -> CheckResult:
        column = zones.ids.name
        limit = inputs.ctx.evidence_limit
        if uniform:
            return graded(
                "lisa", 0, "The zones' differences are uniform: no cluster of change.", "",
                total=len(kept), unit="zones",
            )  # fmt: skip
        local = local_morans_i(
            d, weights.neighbours, row_standardize=False, permutations=params.permutations,
            seed=inference.seed, max_work=inference.max_work,
        )  # fmt: skip
        counts = weights.counts()
        members, records, magnitudes = [], [], []
        for position, original in enumerate(kept):
            zone = zones.ids[position]
            b, c = zones.baseline[original], zones.candidate[original]
            quadrant = local.quadrants[position]
            cluster = quadrant in ("HH", "LL")
            result = local.results[position]
            members.append(
                Member(
                    cluster and judged.material(b, c),
                    f"{column}={zone}",
                    applicable=counts[position] > 0,
                    run=_given(result),
                )
            )
            records.append(
                {
                    column: zone,
                    "cluster": {"HH": "hot spot", "LL": "cold spot"}.get(quadrant or "", "none"),
                    "quadrant": quadrant,
                    "local_i": result.statistic,
                    "baseline": report_number(b),
                    "candidate": report_number(c),
                    params.difference: float(d[position]),
                }
            )
            magnitudes.append(abs(float(d[position])))
        ok = f"No zone sits in a significant cluster of material change ({weights.describe()})."
        bad = "{count} sit in significant clusters of material change (" + weights.describe() + ")."
        check = SpatialLevel._decided(
            "lisa", MeasureTest("local_morans_i", None), inference, members, records, limit, ok,
            bad, "zone", magnitudes=magnitudes,
        )  # fmt: skip
        details["clusters"] = [
            {key: record[key] for key in (column, "cluster", "quadrant", "local_i", "p_value",
                                          "adjusted_p", params.difference)}
            for record in check.affected_records
            if "p_value" in record
        ]  # fmt: skip
        # A cluster can only be named when the smallest p-value survives the correction.
        smallest = {"smallest_p": 1 / (params.permutations + 1), "islands": len(weights.islands)}
        return check.model_copy(update={"metrics": {**check.metrics, **smallest}})
