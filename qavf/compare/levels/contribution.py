"""Contribution: what explains the total difference between the versions.

The measure is additive (``count``, ``sum(x)`` or ``null_count``), so the
total's change is exactly the sum of its members' changes. Along each
dimension (``group_by``-style columns, zones, a time bucket, or combinations
such as zone x month) the members are reduced by the ``groups`` machinery
(:func:`~qavf.compare.levels.groups.reduce_groups`), and:

* every member's change (a member on one side only counts 0 on the other) is
  its contribution, absolute and as a share of the total change, sorted
  largest first with the cumulative share;
* the smallest set of members explaining ``share`` (0.8) of the total change
  is named (:func:`qavf.statistics.decomposition.minimal_set`);
* each member's change is judged by materiality and significance exactly as a
  ``groups`` measure (:func:`~qavf.compare.levels.groups.compare_measure`),
  one family per dimension.

``rates`` split the change of an overall rate ``sum(numerator) /
sum(denominator)`` (``fatal / vmt``) over a dimension's strata into a **mix**
effect (the denominator's composition shifted), a **rate** effect (the strata's
rates changed) and their **interaction**, exactly: the parts are rationals that
sum to the change with no residual (:mod:`qavf.statistics.decomposition`).
The ``total`` check judges the measure's overall change; a ``rate`` check the
overall rate's (a Poisson test of the totals when the numerators are counts).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from qavf.compare.levels import SIDES, ComparisonLevel, LevelInputs, not_assessed, register_level
from qavf.compare.levels.groups import (
    AlignedGroups,
    MeasureSpec,
    compare_measure,
    materiality,
    measure_tests,
    reduce_groups,
)
from qavf.compare.levels.temporal import TimeMembers, keyed_sides
from qavf.compare.results import (
    CheckResult,
    ContributionResult,
    DimensionContribution,
    LevelResult,
    MemberContribution,
    RateDecompositionResult,
    RateEffect,
)
from qavf.compare.significance import Inference, MeasureTest, Member, family_check
from qavf.compare.spec import InferenceParams
from qavf.engine.registry import RuleParams
from qavf.inference.registry import TESTS, TestInput
from qavf.inference.results import TestResult
from qavf.numeric import exact_rational, report_number
from qavf.statistics.decomposition import (
    DecompositionMethod,
    contributions,
    minimal_set,
    rate_decomposition,
)

ADDITIVE = ("count", "sum", "null_count")
"""Measures whose members' values add up to the side's total."""


class Dimension(RuleParams):
    """What members the total splits into: ``zone``, ``[zone, fc]`` or with ``time``."""

    columns: tuple[str, ...] = ()
    time: TimeMembers | None = None
    """Also split by time bucket: ``{time_column: day, bucket: month}`` (zone x month)."""
    name: str | None = None
    """Label in reports; the columns (and bucket) joined by `` x `` when omitted."""

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"columns": [data]}
        if isinstance(data, list | tuple):
            return {"columns": list(data)}
        return data

    @model_validator(mode="after")
    def _something(self) -> Dimension:
        if not self.columns and self.time is None:
            raise ValueError("a dimension needs 'columns', 'time' or both")
        return self

    @property
    def label(self) -> str:
        if self.name is not None:
            return self.name
        parts = [*self.columns]
        if self.time is not None:
            parts.append(self.time.bucket.describe() if self.time.bucket else self.time.time_column)
        return " x ".join(parts)


class RateCheck(RuleParams):
    """An overall rate ``sum(numerator) / sum(denominator)`` decomposed over a dimension."""

    numerator: str
    denominator: str
    name: str | None = None
    """Label in reports; ``numerator / denominator`` when omitted."""
    dimension: str | None = None
    """The dimension label whose members are the strata; the first dimension by default."""
    method: DecompositionMethod = "three_term"
    """``three_term``: mix, rate and interaction (shift-share); ``symmetric``: Kitagawa's
    two terms, each difference weighted by the mean of the other factor."""
    rtol: float | None = Field(default=None, ge=0)
    """Relative tolerance of the overall rate; the level's, then the comparison's."""
    atol: float | None = Field(default=None, ge=0)
    test: Literal["poisson", "none"] | None = None
    """Poisson test of the overall rate (whole-number numerators, the default for them),
    or ``none``: judged by materiality."""

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            numerator, slash, denominator = data.partition("/")
            if not slash or not numerator.strip() or not denominator.strip():
                raise ValueError(f"a rate is 'numerator / denominator', got {data!r}")
            return {"numerator": numerator.strip(), "denominator": denominator.strip()}
        return data

    @property
    def label(self) -> str:
        return self.name or f"{self.numerator} / {self.denominator}"


class ContributionParams(InferenceParams):
    """Parameters of the ``contribution`` level."""

    measure: MeasureSpec = Field(default_factory=lambda: MeasureSpec(measure="count"))
    """An additive measure: ``count``, ``sum(x)`` or ``null_count(x)``."""
    dimensions: tuple[Dimension, ...] = ()
    """Dimensions to explain the total change along (``zone``, ``[zone, fc]``, ...)."""
    share: float = Field(default=0.8, gt=0, le=1)
    """Share of the total change the minimal set of members must explain."""
    rates: tuple[RateCheck, ...] = ()
    """Rates decomposed into mix and rate effects, ``"fatal / vmt"`` or a mapping."""
    rtol: float | None = Field(default=None, ge=0)
    """Default relative tolerance of this level; the comparison's when omitted."""
    atol: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> ContributionParams:
        if self.measure.measure not in ADDITIVE:
            raise ValueError(
                f"measure {self.measure.measure!r} is not additive; use {', '.join(ADDITIVE)}"
            )
        if not self.dimensions and not self.rates:
            raise ValueError("configure 'dimensions', 'rates' or both")
        labels = [dimension.label for dimension in self.dimensions]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"dimensions repeat the label(s) {', '.join(repeated)}; set 'name'")
        for rate in self.rates:
            if not self.dimensions:
                raise ValueError(f"rate {rate.label!r} needs a dimension to decompose over")
            if rate.dimension is not None and rate.dimension not in labels:
                raise ValueError(f"rate dimension {rate.dimension!r} is not one of {labels}")
        return self


def _exact(value: Any) -> Fraction:
    exact = exact_rational(value)
    return exact if exact is not None else Fraction(0)


@register_level
class ContributionLevel(ComparisonLevel):
    """Members' contributions to the total change, minimal sets and mix versus rate."""

    name: ClassVar[str] = "contribution"
    description: ClassVar[str] = (
        "What explains the total change of an additive measure: each member's contribution "
        "per dimension (columns, zones, time buckets or combinations), absolute and as a "
        "share, the smallest set of members explaining a share of it, and an exact mix "
        "versus rate (shift-share) decomposition of declared rates."
    )
    Params: ClassVar[type[RuleParams]] = ContributionParams
    Result: ClassVar[type[LevelResult]] = ContributionResult

    def evaluate(self, inputs: LevelInputs, name: str) -> ContributionResult:
        params: ContributionParams = self.params  # type: ignore[assignment]
        inference = Inference.of(inputs.spec, params)
        spec = params.measure
        judged = materiality(inputs.spec, params, spec)
        plans = measure_tests(inputs, (spec,), inputs.relation("baseline"), inference)
        extras = [plans[0].extras if plans[0] is not None else {}]
        whole = reduce_groups(inputs, [], (spec,), extras=extras)
        _, total = compare_measure(inputs, whole, 0, judged, "total", plans[0], inference)
        checks: list[CheckResult] = [total.model_copy(update={"check": "total"})]
        alias = whole.aliases[0]
        totals = [_exact(whole.frames[side][alias][0]) for side in SIDES]
        dimensions: list[DimensionContribution] = []
        strata: dict[str, AlignedGroups] = {}
        for dimension in params.dimensions:
            try:
                sides = keyed_sides(inputs, dimension.columns, dimension.time)
                relations = sides.relations
                baseline = relations["baseline"] if relations else inputs.relation("baseline")
                plans = measure_tests(inputs, (spec,), baseline, inference)
                extras = [plans[0].extras if plans[0] is not None else {}]
                aligned = reduce_groups(inputs, sides.keys, (spec,), relations, extras)
            except ValueError as exc:
                checks.append(not_assessed(f"contribution: {dimension.label}", str(exc)))
                continue
            comparison, check = compare_measure(
                inputs, aligned, 0, judged, "member", plans[0], inference
            )
            checks.append(check.model_copy(update={"check": f"contribution: {dimension.label}"}))
            dimensions.append(
                self._dimension(inputs, params, dimension.label, aligned, comparison.significance)
            )
            if params.rates:
                strata[dimension.label] = self._strata(inputs, params, sides.keys, relations)
        rates: list[RateDecompositionResult] = []
        for rate in params.rates:
            label = rate.dimension or params.dimensions[0].label
            try:
                if label not in strata:
                    raise ValueError(f"the dimension {label!r} could not be reduced")
                check, decomposed = self._rate(inputs, params, rate, label, strata[label])
            except ValueError as exc:
                checks.append(not_assessed(f"rate: {rate.label}", str(exc).rstrip(".")))
                continue
            checks.append(check)
            rates.append(decomposed)
        return ContributionResult(
            level=self.name,
            name=name,
            checks=checks,
            measure=spec.label,
            share=params.share,
            baseline_total=report_number(totals[0]),
            candidate_total=report_number(totals[1]),
            total_change=report_number(totals[1] - totals[0]),
            dimensions=dimensions,
            rates=rates,
        )

    @staticmethod
    def _dimension(
        inputs: LevelInputs,
        params: ContributionParams,
        label: str,
        aligned: AlignedGroups,
        significance: Any,
    ) -> DimensionContribution:
        """Every member's contribution, sorted, and the minimal set explaining ``share``."""
        alias = aligned.aliases[0]
        left, right = (aligned.columns[side][alias] for side in SIDES)
        presence = [aligned.columns[side][aligned.presence] for side in SIDES]
        keys = aligned.keys
        members = []
        for row in aligned.joined.select([*keys, left, right, *presence]).iter_rows():
            key = tuple(row[: len(keys)])
            b, c, on_b, on_c = row[len(keys) :]
            members.append(
                (
                    key,
                    _exact(b) if on_b is not None else None,
                    _exact(c) if on_c is not None else None,
                )
            )
        ranked = contributions(members)
        changes = [item.change for item in ranked]
        total = sum(changes, Fraction(0))
        chosen = minimal_set(changes, Fraction(str(params.share)))
        picked = set(chosen or ())
        limit = inputs.spec.chart_points
        kept = []
        running = Fraction(0)
        for position, item in enumerate(ranked):
            running += item.change
            if position >= limit:
                continue
            kept.append(
                MemberContribution(
                    group=dict(zip(keys, item.key, strict=True)),
                    baseline=report_number(item.baseline) or 0,
                    candidate=report_number(item.candidate) or 0,
                    change=report_number(item.change) or 0,
                    exact_change=str(item.change),
                    share=float(item.change / total) if total else None,
                    cumulative=float(running / total) if total else None,
                    in_minimal_set=position in picked,
                )
            )
        explained = sum((changes[i] for i in picked), Fraction(0))
        return DimensionContribution(
            dimension=label,
            keys=list(keys),
            members=len(ranked),
            total_change=report_number(total) or 0,
            exact_total_change=str(total),
            minimal_set=None if chosen is None else len(chosen),
            explained=float(explained / total) if total and chosen is not None else None,
            contributions=kept,
            truncated=len(ranked) > len(kept),
            significance=significance,
        )

    # ------------------------------------------------------------------ rates
    @staticmethod
    def _strata(
        inputs: LevelInputs, params: ContributionParams, keys: list[str], relations: Any
    ) -> AlignedGroups:
        specs = tuple(
            MeasureSpec(measure="sum", column=column, name=f"{index}:{column}")
            for rate in params.rates
            for index, column in enumerate((rate.numerator, rate.denominator))
        )
        unique = {spec.label: spec for spec in specs}
        return reduce_groups(inputs, keys, tuple(unique.values()), relations)

    def _rate(
        self,
        inputs: LevelInputs,
        params: ContributionParams,
        rate: RateCheck,
        dimension: str,
        aligned: AlignedGroups,
    ) -> tuple[CheckResult, RateDecompositionResult]:
        labels = [spec.label for spec, _ in aligned.measures]
        aliases = [
            aligned.aliases[labels.index(f"{index}:{column}")]
            for index, column in enumerate((rate.numerator, rate.denominator))
        ]
        keys = aligned.keys
        columns = [aligned.columns[side][alias] for alias in aliases for side in SIDES]
        strata: list[tuple[Any, Fraction, Fraction, Fraction, Fraction]] = []
        for row in aligned.joined.select([*keys, *columns]).iter_rows():
            nb, nc, db, dc = (_exact(value) for value in row[len(keys) :])
            strata.append((tuple(row[: len(keys)]), nb, db, nc, dc))
        decomposed = rate_decomposition(strata, rate.method)
        inference = Inference.of(inputs.spec, params)
        judged = materiality(
            inputs.spec, params, MeasureSpec(measure="count", rtol=rate.rtol, atol=rate.atol)
        )
        before, after = decomposed.baseline_rate, decomposed.candidate_rate
        material = judged.material(before, after)
        sums = [sum((stratum[i] for stratum in strata), Fraction(0)) for i in (1, 2, 3, 4)]
        whole = all(value >= 0 and value.denominator == 1 for value in (sums[0], sums[2]))
        wanted = rate.test or ("poisson" if whole else "none")
        if wanted == "poisson" and not whole:
            raise ValueError(
                f"the Poisson test needs whole non-negative sums of {rate.numerator!r}"
            )
        test = TESTS.create("poisson") if wanted == "poisson" else None
        run = None
        if test is not None:
            data = TestInput(
                counts=(float(sums[0]), float(sums[2])),
                totals=(float(sums[1]), float(sums[3])),
                seed=inference.seed,
                max_work=inference.max_work,
            )
            runner = test.equivalent if inference.decision == "equivalence" else test.run

            def run() -> TestResult:
                return runner(data)

        effects = sorted(decomposed.strata, key=lambda e: -abs(e.total))
        limit = inputs.ctx.evidence_limit
        shown = [
            RateEffect(
                group=dict(zip(keys, effect.key, strict=True)),
                baseline_weight=float(effect.baseline_weight),
                candidate_weight=float(effect.candidate_weight),
                baseline_rate=None if effect.baseline_rate is None else float(effect.baseline_rate),
                candidate_rate=(
                    None if effect.candidate_rate is None else float(effect.candidate_rate)
                ),
                mix=float(effect.mix),
                rate=float(effect.rate),
                interaction=float(effect.interaction),
            )
            for effect in effects[:limit]
        ]
        parts = {
            "change": decomposed.change,
            "mix": decomposed.mix,
            "rate": decomposed.rate,
            "interaction": decomposed.interaction,
        }
        result = RateDecompositionResult(
            name=rate.label,
            numerator=rate.numerator,
            denominator=rate.denominator,
            dimension=dimension,
            method=rate.method,
            baseline_rate=float(before),
            candidate_rate=float(after),
            exact={name: str(value) for name, value in parts.items()},
            strata=shown,
            truncated=len(effects) > len(shown),
            **{name: float(value) for name, value in parts.items()},
        )
        record = {
            "rate": rate.label,
            "baseline": float(before),
            "candidate": float(after),
            **{name: float(value) for name, value in parts.items()},
        }
        plan = (
            MeasureTest("poisson", test)
            if test is not None
            else MeasureTest("none", None, False, "no test declared; judged by materiality")
        )
        member = Member(material, rate.label, applicable=run is not None, run=run)
        how = "" if not inference.tests else f"; {plan.name}, {inference.describe()}"
        split = (
            f"mix {float(decomposed.mix):+.6g}, rate {float(decomposed.rate):+.6g}, "
            f"interaction {float(decomposed.interaction):+.6g}"
        )
        # A passing material change passed because it was not significant: say so.
        passed = (
            f"{rate.label} changed materially but not significantly"
            if material
            else f"{rate.label} did not change materially"
        )
        check = family_check(
            f"rate: {rate.label}",
            plan,
            inference,
            [member],
            [record],
            limit,
            f"{passed} ({float(before):.6g} to {float(after):.6g}; {split}{how}).",
            f"{rate.label} changed from {float(before):.6g} to {float(after):.6g} ({split}{how})",
            "rate",
        )
        metrics = {**check.metrics, **{name: str(value) for name, value in parts.items()}}
        metrics["strata"] = len(strata)
        return check.model_copy(update={"metrics": metrics}), result


__all__ = ["ContributionLevel", "ContributionParams"]
