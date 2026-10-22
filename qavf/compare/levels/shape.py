"""Series shape: do the two versions of each time series move alike?

Each side is bucketed and reduced exactly as by the ``temporal`` level (one
measure per bucket within ``group_by``); every group is then one series per
side, positioned by exact bucket ordinals. Shape checks compare the series on
the buckets both sides have:

* ``correlation``: Pearson and Spearman correlation;
* ``trend``: least-squares slope per bucket of each side and their difference;
* ``profile``: the normalized cyclic profile (e.g. by hour of day) of each side;
* ``peak``: when each side peaks, and how high;
* ``lag``: the cross-correlation lag that matches best (a series shifted by
  one bucket is found as lag 1);
* ``level shift``: the TMP-008 two-segment screen, on each side.

Correlations, slopes and profiles are exact rationals until reported, and
thresholds are decided exactly. A series too short to judge is not assessed,
never passed.

Unless the decision policy is ``material``, a threshold breach must also be
significant, judged across the series of each check (one family per check):

* ``correlation``: Fisher's z test that the Pearson correlation is below
  ``min_pearson``, with Bartlett's effective sample size for autocorrelated
  buckets (a low Spearman correlation alone, or one flat series, has no test
  and is judged by its threshold);
* ``trend``: the slope difference with a null-imposed circular block bootstrap
  (or Newey-West HAC errors);
* ``lag``: a block bootstrap of the correlation gain of the best lag over lag 0;
* ``profile``, ``peak``, ``level shift``: no sound test; judged by threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from qavf.compare.comparators import Materiality
from qavf.compare.levels import (
    SIDES,
    ComparisonLevel,
    LevelInputs,
    graded,
    not_assessed,
    register_level,
)
from qavf.compare.levels.groups import MeasureSpec, reduce_groups
from qavf.compare.levels.temporal import TimeAxisParams, bucketed_sides
from qavf.compare.results import (
    CheckResult,
    LevelResult,
    SeriesShape,
    ShapeAssessment,
    ShapeResult,
)
from qavf.compare.significance import Inference, MeasureTest, Member, evidence_fields, judge
from qavf.compare.spec import InferenceParams, Side
from qavf.engine.registry import RuleParams
from qavf.inference.results import Alternative, Significance, TestResult
from qavf.inference.trend import correlation_test, lag_test, slope_difference
from qavf.numeric import exact_rational, report_number
from qavf.rules.helpers import BoundedEvidence, fresh_internal, plural
from qavf.statistics.series import (
    Correlation,
    best_lag,
    best_split,
    normal_equations,
    pearson,
    spearman,
)
from qavf.temporal.buckets import (
    Bucket,
    TimeBucketing,
    bucket_labels,
    calendar_position,
    sequence_positions,
)


class CorrelationCheck(RuleParams):
    """Both correlations must reach their minimum (``null`` skips one)."""

    min_pearson: float | None = Field(default=0.9, ge=-1, le=1)
    """Lowest Pearson correlation of the paired bucket values."""
    min_spearman: float | None = Field(default=0.9, ge=-1, le=1)
    """Lowest Spearman (rank) correlation."""
    dependence: Literal["effective_n", "none"] = "effective_n"
    """Sample size of the Fisher z test: Bartlett's effective size for autocorrelated
    buckets, or the bucket count."""


class TrendCheck(RuleParams):
    """The slopes differ materially beyond ``max(atol, rtol * max(|b|, |c|))``."""

    rtol: float = Field(default=0.1, ge=0)
    """Relative tolerance of the least-squares slopes (change per bucket)."""
    atol: float = Field(default=0.0, ge=0)
    dependence: Literal["block", "hac", "none"] = "block"
    """Uncertainty of the slope difference: a null-imposed circular block bootstrap,
    Newey-West HAC errors, or independent residuals."""


class ProfileCheck(RuleParams):
    """Cyclic profiles (mean per position, normalized) may differ by ``max_distance``."""

    profile: Bucket
    """A cyclic profile of the buckets: ``hour_of_day``, ``day_of_week``, ``month_of_year``..."""
    max_distance: float = Field(default=0.05, ge=0, le=1)
    """Largest half-sum of absolute share differences (0: identical, 1: disjoint)."""

    @model_validator(mode="after")
    def _cyclic(self) -> ProfileCheck:
        if not self.profile.cyclic:
            raise ValueError(f"'profile' must be a cyclic profile, not {self.profile.unit!r}")
        return self


class PeakCheck(RuleParams):
    """The peaks fall within ``max_shift`` buckets and agree in magnitude."""

    max_shift: int = Field(default=0, ge=0)
    """Buckets the candidate's peak may move (around the cycle for profiles)."""
    rtol: float = Field(default=0.1, ge=0)
    """Relative tolerance of the peak values."""
    atol: float = Field(default=0.0, ge=0)


class LagCheck(RuleParams):
    """The best-correlated lag within ``max_lag`` buckets is at most ``allowed``."""

    max_lag: int = Field(default=3, ge=1)
    """Lags searched in each direction (cross-correlation of the two series)."""
    allowed: int = Field(default=0, ge=0)
    """Largest acceptable best lag."""


class LevelShiftCheck(RuleParams):
    """Level shifts (TMP-008's screen) appear on both sides, at most ``max_offset`` apart."""

    min_segment: int = Field(default=4, ge=2)
    """Fewest buckets on each side of a candidate shift."""
    threshold: float = Field(default=5.0, gt=0)
    """Gap in medians, in pooled robust spreads, that makes a shift."""
    min_relative_shift: float = Field(default=0.2, ge=0)
    """Also require the medians to differ by this fraction of the earlier one."""
    max_offset: int = Field(default=0, ge=0)
    """Buckets between the two sides' shifts still counted as the same shift."""


class ShapeParams(TimeAxisParams, InferenceParams):
    """Parameters of the ``series_shape`` level."""

    measure: MeasureSpec = Field(default_factory=lambda: MeasureSpec(measure="count"))
    """The series value per bucket (``sum(x)``, ``mean(x)``, ...; rows by default)."""
    min_buckets: int = Field(default=4, ge=3)
    """Fewest buckets both sides must have for a series to be assessed."""
    missing_series: Literal["fail", "ignore"] = "fail"
    """A series (group) on one side only: a difference or left out."""
    correlation: CorrelationCheck | None = Field(default_factory=CorrelationCheck)
    """Pearson and Spearman correlation; ``null`` leaves the check out (so for each)."""
    trend: TrendCheck | None = Field(default_factory=TrendCheck)
    """Least-squares slope per bucket on each side."""
    profile: ProfileCheck | None = None
    """Cyclic profile of the buckets (off unless configured)."""
    peak: PeakCheck | None = Field(default_factory=PeakCheck)
    """Peak timing and magnitude."""
    lag: LagCheck | None = Field(default_factory=LagCheck)
    """Best cross-correlation lag."""
    level_shift: LevelShiftCheck | None = Field(default_factory=LevelShiftCheck)
    """Level shifts present in only one version."""

    @model_validator(mode="after")
    def _consistent(self) -> ShapeParams:
        if self.profile is not None and (self.bucket is None or self.bucket.cyclic):
            raise ValueError("'profile' needs a calendar 'bucket' to take the profile of")
        return self

    @property
    def needed(self) -> int:
        """Fewest common buckets: ``min_buckets``, or two level-shift segments."""
        segments = 2 * self.level_shift.min_segment if self.level_shift is not None else 0
        return max(self.min_buckets, segments)


@dataclass(frozen=True, slots=True)
class _Point:
    value: Fraction
    label: Any
    profile: int | None


type _Series = dict[int, _Point]


def _series(
    frame: pl.DataFrame, group_by: list[str], columns: tuple[str, str, str, str | None]
) -> dict[tuple[Any, ...], _Series]:
    """Each group's buckets by ordinal; buckets without a measure value are absent."""
    label, value, position, profile = columns
    result: dict[tuple[Any, ...], _Series] = {}
    selected = [*group_by, label, value, position, *([profile] if profile else [])]
    for row in frame.select(selected).iter_rows():
        key, rest = tuple(row[: len(group_by)]), row[len(group_by) :]
        exact = exact_rational(rest[1])
        if exact is None:
            continue
        points = result.setdefault(key, {})
        if rest[2] in points:
            raise ValueError("two buckets of a series share one position")
        points[rest[2]] = _Point(exact, rest[0], rest[3] if profile else None)
    return result


def _constant(values: Sequence[Fraction]) -> bool:
    return len(set(values)) == 1


def _circular(shift: int, period: int | None) -> int:
    if period is None:
        return shift
    return (shift + period // 2) % period - period // 2


def _value(correlation: Correlation | None) -> float | None:
    return None if correlation is None else correlation.value


_TESTED = {"correlation": "fisher_z", "trend": "slope_difference", "lag": "lag"}
"""The test of each shape check that has one."""


@dataclass
class _Check:
    """One shape check's running tally over series.

    Without inference a breach of the threshold is a difference. With it, each
    series is a family member (``test`` runs it when the policy needs it) and
    :meth:`decide` judges the family once every series is recorded.
    """

    name: str
    evidence: BoundedEvidence
    inference: Inference | None = None
    assessed: int = 0
    different: int = 0
    not_assessed: int = 0
    members: list[Member] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    series: list[int] = field(default_factory=list)
    significance: Significance | None = None
    results: list[TestResult | None] = field(default_factory=list)

    def record(
        self,
        different: bool,
        record: dict[str, Any],
        series: int = -1,
        test: Callable[[], TestResult] | None = None,
    ) -> None:
        """A series' threshold judgement; ``test`` (``None``: no sound test) runs lazily."""
        if self.inference is None:
            self.assessed += 1
            if different:
                self.different += 1
                self.evidence.append(record)
            return
        self.members.append(Member(different, None, applicable=test is not None, run=test))
        self.records.append(record)
        self.series.append(series)

    def decide(self, limit: int) -> None:
        """Test, correct and decide the family of recorded series."""
        if self.inference is None:
            return
        method = _TESTED.get(self.name, "none")
        plan = MeasureTest(method, None)
        if method == "none":
            plan = MeasureTest(method, None, False, f"no sound test for {self.name}")
        labelled = [
            Member(m.material, _label(r), m.data, m.problem, m.applicable, m.run)
            for m, r in zip(self.members, self.records, strict=True)
        ]
        judgement = judge(plan, self.inference, labelled, limit)
        self.results = judgement.results
        self.significance = judgement.significance
        for position, decision in enumerate(judgement.decisions):
            if not decision.assessed:
                self.not_assessed += 1
                continue
            self.assessed += 1
            if decision.failed:
                self.different += 1
                record = self.records[position]
                result = judgement.results[position]
                self.evidence.append({**record, **evidence_fields(result, decision)})


def _label(record: dict[str, Any]) -> str | None:
    keys = [key for key in record if not isinstance(record[key], dict | list)][:3]
    return ", ".join(f"{key}={record[key]}" for key in keys) or None


@register_level
class SeriesShapeLevel(ComparisonLevel):
    """Correlation, trend, profile, peak, lag and level-shift differences per series."""

    name: ClassVar[str] = "series_shape"
    description: ClassVar[str] = (
        "Per series (group) of one bucketed measure: Pearson and Spearman correlation, "
        "trend slopes, cyclic profile, peak timing and size, best cross-correlation lag "
        "and level shifts present in one version only, each with its own threshold."
    )
    Params: ClassVar[type[RuleParams]] = ShapeParams
    Result: ClassVar[type[LevelResult]] = ShapeResult

    def evaluate(self, inputs: LevelInputs, name: str) -> ShapeResult:
        params: ShapeParams = self.params  # type: ignore[assignment]
        group_by = list(params.group_by)
        inference = Inference.of(inputs.spec, params)
        tests = inference if inference.tests else None
        sides = bucketed_sides(inputs, params)
        keys = [*group_by, sides.column]
        aligned = reduce_groups(inputs, keys, (params.measure,), sides.relations)
        value = aligned.aliases[0]
        taken = [*aligned.frames["baseline"].columns]
        position = fresh_internal("position", taken)
        profile = fresh_internal("profile", [*taken, position]) if params.profile else None
        # Series are keyed by sequence position (one apart per bucket, across DST
        # changes), counted from the same first bucket on both sides; a cyclic
        # profile slot is a calendar position (a fall-back hour is its local hour).
        labels = pl.concat([aligned.frames[side][sides.column] for side in SIDES])
        span = labels.drop_nulls().sort()
        frames = {}
        for side in SIDES:
            frame = aligned.frames[side]
            sequence = sequence_positions(
                frame[sides.column],
                params.bucket,
                first=span.head(1),
                last=span.tail(1),
                limit=params.max_buckets,
            )
            frame = frame.with_columns(sequence.alias(position))
            if params.profile is not None and profile is not None:
                axis = TimeBucketing(bucket=params.profile.profile, time_zone=params.time_zone)
                label, dtype = bucket_labels(sides.column, sides.dtype, axis)
                slot = calendar_position(label, dtype, params.profile.profile)
                frame = frame.with_columns(slot.alias(profile))
            frames[side] = frame
        series = {
            side: _series(frames[side], group_by, (sides.column, value, position, profile))
            for side in SIDES
        }
        period = params.bucket.cycle_length if params.bucket is not None else None
        groups = (
            list(aligned.joined.select(group_by).unique(maintain_order=True).iter_rows())
            if group_by
            else [()]
        )
        present = {
            side: set(aligned.frames[side].select(group_by).iter_rows())
            if group_by
            else ({()} if aligned.frames[side].height else set())
            for side in SIDES
        }
        checks = {
            check: _Check(check, inputs.evidence(), tests)
            for check, enabled in (
                ("correlation", params.correlation),
                ("trend", params.trend),
                ("profile", params.profile),
                ("peak", params.peak),
                ("lag", params.lag),
                ("level shift", params.level_shift),
            )
            if enabled is not None
        }
        one_sided, short = inputs.evidence(), inputs.evidence()
        only: dict[Side, int] = {"baseline": 0, "candidate": 0}
        shapes: list[SeriesShape] = []
        compared = too_short = 0
        for key in groups:
            group = dict(zip(group_by, key, strict=True))
            on = {side: key in present[side] for side in SIDES}
            if on["baseline"] != on["candidate"]:
                lone: Side = "baseline" if on["baseline"] else "candidate"
                only[lone] += 1
                one_sided.append({**group, "series": f"only in {lone}"})
                continue
            # A series whose measure has no value in some buckets (all inputs missing)
            # just has fewer buckets; with too few it is not assessed.
            b, c = series["baseline"].get(key, {}), series["candidate"].get(key, {})
            compared += 1
            common = sorted(set(b) & set(c))
            if len(common) < params.needed:
                too_short += 1
                short.append({**group, "common_buckets": len(common)})
                continue
            kept = len(shapes) if len(shapes) < inputs.ctx.evidence_limit else -1
            shape = self._shape(
                params, group, b, c, common, period, checks, tests, compared - 1, kept
            )
            if kept >= 0:
                shapes.append(shape)
        for check in checks.values():
            check.decide(inputs.ctx.evidence_limit)
            for index, test in zip(check.series, check.results, strict=True):
                if index >= 0 and test is not None:
                    tagged = test.model_copy(update={"label": check.name})
                    found = [*shapes[index].significance, tagged]
                    shapes[index] = shapes[index].model_copy(update={"significance": found})
        results: list[CheckResult] = [
            graded(
                "series",
                sum(only.values()),
                f"Both sides have the same {plural(compared, 'series')}.",
                f"{plural(only['baseline'], 'series')} only in the baseline, "
                f"{plural(only['candidate'], 'series')} only in the candidate.",
                evidence=one_sided,
                total=compared + sum(only.values()),
                unit="series",
                allowed=params.missing_series == "ignore",
            )
        ]
        if too_short:
            results.append(
                CheckResult(
                    check="too short",
                    passed=False,
                    assessed=False,
                    message=f"Not assessed: {plural(too_short, 'series')} of {compared:,} "
                    f"have fewer than {params.needed} buckets on both sides.",
                    affected_count=too_short,
                    total=compared,
                    unit="series",
                    affected_records=short.records,
                    truncated=short.truncated,
                )
            )
        for check in checks.values():
            if not check.assessed and not check.not_assessed:
                results.append(
                    not_assessed(check.name, f"no series has {params.needed} buckets on both sides")
                )
                continue
            if check.not_assessed and not check.different:
                reason = next(
                    (r.reason for r in check.results if r is not None and not r.assessed), ""
                )
                outcome = CheckResult(
                    check=check.name,
                    passed=False,
                    assessed=False,
                    message=f"Not assessed: {check.name}: {plural(check.not_assessed, 'series')} "
                    f"need a {_TESTED.get(check.name)} test that could not run ({reason}).",
                    affected_count=check.not_assessed,
                    total=check.assessed + check.not_assessed,
                    unit="series",
                )
            else:
                how = "" if tests is None else f" ({tests.describe()})"
                outcome = graded(
                    check.name,
                    check.different,
                    f"{check.name.capitalize()} agrees in {plural(check.assessed, 'series')}{how}.",
                    f"{check.name.capitalize()} differs in {plural(check.different, 'series')} "
                    f"of {check.assessed:,}{how}"
                    + (f"; {check.not_assessed:,} not assessed" if check.not_assessed else "")
                    + ".",
                    evidence=check.evidence,
                    total=check.assessed,
                    unit="series",
                )
            results.append(outcome.model_copy(update={"significance": check.significance}))
        return ShapeResult(
            level=self.name,
            name=name,
            checks=results,
            time_column=params.time_column,
            bucket=params.describe(),
            group_by=group_by,
            measure=params.measure.label,
            compared=compared,
            only_in_baseline=only["baseline"],
            only_in_candidate=only["candidate"],
            too_short=too_short,
            assessments=[
                ShapeAssessment(
                    check=c.name,
                    assessed=c.assessed,
                    different=c.different,
                    significance=c.significance,
                )
                for c in checks.values()
            ],
            series=shapes,
        )

    @staticmethod
    def _shape(
        params: ShapeParams,
        group: dict[str, Any],
        b: _Series,
        c: _Series,
        common: list[int],
        period: int | None,
        checks: dict[str, _Check],
        inference: Inference | None = None,
        ordinal: int = 0,
        position: int = -1,
    ) -> SeriesShape:
        """Every enabled shape metric of one series, tallied into ``checks``.

        With ``inference`` each check that has a test records it, to run when the
        family is decided; ``ordinal`` (the series' place) offsets the seed and
        ``position`` is the series' place in the reported metrics (-1: not kept).
        """
        xb = [b[p].value for p in common]
        xc = [c[p].value for p in common]
        flat = (_constant(xb), _constant(xc))
        seed = 0 if inference is None else inference.seed + ordinal
        work = 0 if inference is None else inference.max_work
        floats = (
            np.array([float(v) for v in xb]),
            np.array([float(v) for v in xc]),
            np.array(common, dtype=float),
        )
        shape: dict[str, Any] = {"group": group, "buckets": len(common)}
        if (correlation := params.correlation) is not None:
            r = pearson(xb, xc) if not any(flat) else None
            rho = spearman(xb, xc) if not any(flat) else None
            # Two flat series share their shape; one flat series does not.
            low = flat[0] != flat[1] or any(
                coefficient is not None and limit is not None and not coefficient.at_least(limit)
                for coefficient, limit in (
                    (r, correlation.min_pearson),
                    (rho, correlation.min_spearman),
                )
            )
            found = {"pearson": _value(r), "spearman": _value(rho)}
            shape |= found
            limit = correlation.min_pearson
            pearson_low = r is not None and limit is not None and not r.at_least(limit)
            # Fisher's z tests the Pearson correlation against its minimum; a flat
            # series, or a low Spearman correlation alone, is judged by its threshold.
            testable = (
                not any(flat) and limit is not None and -1 < limit < 1 and (pearson_low or not low)
            )
            test = None
            if testable and inference is not None:
                assert limit is not None
                side: Alternative = "greater" if inference.decision == "equivalence" else "less"
                dependence = correlation.dependence

                def test() -> TestResult:
                    return correlation_test(
                        floats[0], floats[1], rho=limit, alternative=side, dependence=dependence
                    )

            record = {**group, **found, "buckets": len(common)}
            checks["correlation"].record(low, record, position, test)
        if (trend := params.trend) is not None:
            positions = [Fraction(p) for p in common]
            before, after = (normal_equations(positions, values).slope for values in (xb, xc))
            assert before is not None and after is not None  # distinct positions
            changed = Materiality(trend.rtol, trend.atol).material(before, after)
            found = {"slope_baseline": report_number(before)}
            found["slope_candidate"] = report_number(after)
            shape |= found
            change = report_number(after - before)

            def slopes() -> TestResult:
                assert trend is not None
                return slope_difference(
                    floats[2], floats[0], floats[1], dependence=trend.dependence,
                    seed=seed, max_work=work,
                )  # fmt: skip

            checks["trend"].record(changed, {**group, **found, "change": change}, position, slopes)
        if (profile := params.profile) is not None:
            distance = _profile_distance(b, c, common)
            shape["profile_distance"] = float(distance)
            different = distance > Fraction(str(profile.max_distance))
            checks["profile"].record(different, {**group, "distance": float(distance)}, position)
        if (peak := params.peak) is not None:
            top_b = max(common, key=lambda p: (b[p].value, -p))
            top_c = max(common, key=lambda p: (c[p].value, -p))
            shift = _circular(top_c - top_b, period)
            higher = Materiality(peak.rtol, peak.atol).material(b[top_b].value, c[top_c].value)
            shape |= {"peaks": {"baseline": b[top_b].label, "candidate": c[top_c].label}}
            shape["peak_shift"] = shift
            found = {
                "baseline_peak": b[top_b].label,
                "candidate_peak": c[top_c].label,
                "shift": shift,
                "baseline_value": report_number(b[top_b].value),
                "candidate_value": report_number(c[top_c].value),
            }
            checks["peak"].record(
                abs(shift) > peak.max_shift or higher, {**group, **found}, position
            )
        if (lag := params.lag) is not None:
            values_b = {p: point.value for p, point in b.items()}
            values_c = {p: point.value for p, point in c.items()}
            best = best_lag(values_b, values_c, lag.max_lag, params.min_buckets, period)
            # No defined lag: two flat series agree, otherwise the shapes differ.
            agrees = abs(best.lag) <= lag.allowed if best is not None else all(flat)
            found = {
                "lag": None if best is None else best.lag,
                "correlation": None if best is None else best.correlation.value,
                "correlation_at_lag_0": _value(None if best is None else best.at_zero),
            }
            shape |= {"lag": found["lag"], "lag_correlation": found["correlation"]}
            gain = None
            if best is not None and best.lag != 0:
                shift = best.lag
                pairs = (
                    {p: float(v) for p, v in values_b.items()},
                    {p: float(v) for p, v in values_c.items()},
                )

                def gain() -> TestResult:
                    return lag_test(
                        pairs[0], pairs[1], shift, period=period, seed=seed, max_work=work,
                        minimum=params.min_buckets,
                    )  # fmt: skip

            checks["lag"].record(not agrees, {**group, **found}, position, gain)
        if (screen := params.level_shift) is not None:
            # TMP-008's screen in binary64 (values rounded once), on the common buckets.
            at: list[int | None] = []
            for values in (xb, xc):
                observations = np.array([float(v) for v in values])
                split = best_split(observations, screen.min_segment, screen.min_relative_shift)
                shifted = split is not None and split[3] > screen.threshold
                at.append(common[split[0]] if split is not None and shifted else None)
            first, second = at
            differs = (first is None) != (second is None) or (
                first is not None and second is not None and abs(first - second) > screen.max_offset
            )
            labels = {
                "baseline": None if first is None else b[first].label,
                "candidate": None if second is None else c[second].label,
            }
            shape["level_shifts"] = labels
            found = {"baseline_shift": labels["baseline"], "candidate_shift": labels["candidate"]}
            checks["level shift"].record(differs, {**group, **found}, position)
        return SeriesShape(**shape)


def _profile_distance(b: _Series, c: _Series, common: list[int]) -> Fraction:
    """Half the summed absolute differences of the two profiles, exactly.

    A profile is the mean per cycle position over the common buckets, divided by
    the sum of the means' magnitudes, so it compares shape, not level: 0 for
    proportional profiles, 1 for profiles with no position in common. An all-zero
    profile has all-zero shares.
    """
    shares = []
    for series in (b, c):
        values: dict[int, list[Fraction]] = {}
        for p in common:
            point = series[p]
            assert point.profile is not None
            values.setdefault(point.profile, []).append(point.value)
        means = {k: sum(v, Fraction(0)) / len(v) for k, v in values.items()}
        total = sum((abs(m) for m in means.values()), Fraction(0))
        shares.append({k: (m / total if total else Fraction(0)) for k, m in means.items()})
    positions = set(shares[0]) | set(shares[1])
    zero = Fraction(0)
    differences = (abs(shares[0].get(k, zero) - shares[1].get(k, zero)) for k in positions)
    return sum(differences, zero) / 2
