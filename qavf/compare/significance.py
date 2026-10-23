"""Statistical significance in comparison levels: settings, tests, inputs, decisions.

A level resolves its :class:`Inference` settings (its own ``decision``,
``alpha``, ``correction``, ``seed`` and ``max_work``, else the comparison's).
Each measure then gets a :class:`MeasureTest`: its configured ``test``, else
the measure's default for its column (``mean``: Welch's t, ``sum`` of
integers: Poisson, ...), or none when no sound test exists (``min``, ``max``,
``distinct_count``), in which case materiality decides alone. What the test
reads (raw values, moments, or successes and trials per group) is reduced
natively next to the measure, so only per-group inputs reach Python.

:func:`judge` decides a family (the groups of a measure, the buckets of a
temporal measure): tests run where the policy needs them, the family is
corrected together (members left untested count as ``p = 1``, which keeps
every correction valid), and each member is decided by
:func:`qavf.inference.decision.decide`. The ``material`` policy runs no test
and leaves every ``significance`` slot empty, exactly as before inference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import polars as pl

from qavf.aggregation import numeric_expr
from qavf.compare.measures import Measure, MeasureInput
from qavf.compare.results import CheckResult
from qavf.compare.spec import ComparisonSpec, Side
from qavf.inference.decision import Decision, decide, needs_test
from qavf.inference.independent import Moments
from qavf.inference.multiple import correct
from qavf.inference.registry import TESTS, InferenceTest, Needs, TestInput
from qavf.inference.results import (
    Correction,
    DecisionPolicy,
    Significance,
    TestResult,
    not_assessed,
)
from qavf.rules._grouped import ProjectedRelation
from qavf.rules.helpers import LargestEvidence, plural
from qavf.stats import checked_float_series

type Extra = Callable[[ProjectedRelation, bool], pl.Expr]
"""A per-group reduction a test reads, built from one side's relation; the flag
says whether it is reduced per group (``False``: over the whole side)."""


@dataclass(frozen=True, slots=True)
class Inference:
    """Resolved significance settings of one level."""

    decision: DecisionPolicy
    alpha: float
    correction: Correction
    seed: int
    max_work: int

    @classmethod
    def of(cls, spec: ComparisonSpec, level: object) -> Inference:
        """The level's own settings where set, else the comparison's."""

        def pick(name: str) -> Any:
            own = getattr(level, name, None)
            return own if own is not None else getattr(spec, name)

        return cls(
            *(pick(name) for name in ("decision", "alpha", "correction", "seed", "max_work"))
        )

    @property
    def tests(self) -> bool:
        """Whether any test runs (every policy but ``material``)."""
        return self.decision != "material"

    def describe(self) -> str:
        return f"{self.decision.replace('_', ' ')}, {self.correction}, alpha={self.alpha:g}"


def compatibility(test: InferenceTest, measure: Measure, column: str | None) -> str | None:
    """Why ``test`` cannot read ``measure`` per group, or ``None`` when it can."""
    needs: Needs = test.needs
    if needs == "paired":
        return f"{test.name} is a paired test; it applies to the records level"
    if needs in ("samples", "moments") and column is None:
        return f"{test.name} reads the group's values; the measure has no column"
    if needs == "counts" and not measure.counts:
        return f"{test.name} needs event counts; {measure.name!r} values are not counts"
    if needs == "proportions" and type(measure).proportions is Measure.proportions:
        return f"{test.name} needs successes and trials; {measure.name!r} is not a share"
    if getattr(test, "uses_statistic", False) and measure.statistic is None:
        return f"{test.name} compares a sample statistic; {measure.name!r} has none"
    return None


def float_values(series: pl.Series) -> np.ndarray:
    """Non-missing values as Float64; integers must convert exactly (decimals round once).

    Raises:
        ValueError: for integers beyond Float64's exact range.
    """
    values = series.drop_nulls()
    if values.dtype.is_integer():
        return checked_float_series(values).to_numpy()
    return values.cast(pl.Float64).to_numpy()


def centered_values(values: Sequence[Fraction]) -> np.ndarray:
    """Translate exact values for location-invariant tests, without losing variation."""
    if not values:
        return np.empty(0, dtype=float)
    try:
        direct = [float(value) for value in values]
    except OverflowError:
        direct = []
    if direct and all(
        math.isfinite(number) and Fraction(number) == value
        for number, value in zip(direct, values, strict=True)
    ):
        return np.asarray(direct, dtype=float)
    origin = values[0]
    converted = []
    for value in values:
        shifted = value - origin
        try:
            number = float(shifted)
        except OverflowError:
            raise ValueError("centred values cannot be represented exactly as Float64") from None
        if not math.isfinite(number) or Fraction(number) != shifted:
            raise ValueError("centred values cannot be represented exactly as Float64")
        converted.append(number)
    return np.asarray(converted, dtype=float)


def _values(relation: ProjectedRelation, column: str) -> pl.Expr:
    return numeric_expr(column, relation.schema[column]).cast(pl.Float64)


def _lossy(relation: ProjectedRelation, column: str) -> pl.Expr:
    """Integers that Float64 cannot hold exactly (decimals round once to binary64)."""
    dtype = relation.schema[column]
    if not dtype.is_integer():
        return pl.lit(0, dtype=pl.UInt64)
    value = pl.col(column)
    changed = value.cast(pl.Float64).cast(dtype, strict=False).ne_missing(value)
    return (value.is_not_null() & changed).sum().cast(pl.UInt64)


def sample_extras(column: str) -> dict[str, Extra]:
    """The reductions that collect a column's values per group: ``values`` (non-missing,
    as Float64) and ``lossy`` (integers that Float64 cannot hold exactly)."""

    def values(side: ProjectedRelation, grouped: bool) -> pl.Expr:
        kept = _values(side, column).drop_nulls()
        return kept if grouped else kept.implode()

    return {"lossy": lambda side, grouped: _lossy(side, column), "values": values}


@dataclass(frozen=True)
class MeasureTest:
    """How one measure's members (groups, buckets) are tested."""

    name: str
    """The test, or ``none``."""
    test: InferenceTest | None
    applicable: bool = True
    """``False``: no sound test exists; materiality decides."""
    reason: str | None = None
    """Why no test applies, or why the test cannot read this measure."""
    statistic: str | None = None
    probability: float | None = None
    extras: dict[str, Extra] = field(default_factory=dict)
    """Per-group reductions the test reads, by role."""
    share: bool = False
    """Proportions whose trials are the side's total of successes (a share)."""

    @property
    def usable(self) -> bool:
        return self.applicable and self.test is not None and self.reason is None


def plan_measure(
    spec: Any, measure: Measure, relation: ProjectedRelation, inference: Inference, rows: int
) -> MeasureTest | None:
    """The measure's test and the reductions it reads; ``None`` under ``material``.

    ``relation`` is one aligned side (its schema decides the default test);
    ``rows`` counts both sides' rows: a test reading raw values collects them
    only within ``max_work``, and is otherwise not assessed.
    """
    if not inference.tests:
        return None
    column = spec.column
    dtype = relation.schema.get(column) if column is not None else None
    if spec.test is not None:
        test = spec.test.create()
        if test is None:
            return MeasureTest("none", None, False, "the measure declares no test (test: none)")
    else:
        default = measure.default_test(dtype)
        if default is None:
            return MeasureTest(
                "none", None, False, f"no sound test for {measure.name!r}; judged by materiality"
            )
        test = TESTS.create(default)
    problem = compatibility(test, measure, column)
    if problem is None and test.needs == "samples" and rows > inference.max_work:
        problem = f"collecting {rows:,} values exceeds max_work {inference.max_work:,}"
    if problem is not None:
        return MeasureTest(test.name, test, True, problem)
    extras: dict[str, Extra] = {}
    share = False
    if test.needs in ("samples", "moments"):
        assert column is not None
        extras["lossy"] = lambda side, grouped: _lossy(side, column)
    if test.needs == "samples":
        assert column is not None
        extras = sample_extras(column)
    elif test.needs == "moments":
        assert column is not None

        def moment(name: str) -> Extra:
            def build(side: ProjectedRelation, grouped: bool) -> pl.Expr:
                # Moments of values shifted by the group's first value lose no
                # small variation around a large offset.
                x = _values(side, column)
                origin = x.drop_nulls().first()
                if name == "n":
                    return x.count()
                if name == "mean":
                    return origin + (x - origin).mean()
                return (x - origin).var(ddof=1)

            return build

        for name in ("n", "mean", "var"):
            extras[name] = moment(name)
    elif test.needs == "proportions":
        parts = measure.proportions(MeasureInput(relation, (), column, "", "baseline"))
        assert parts is not None
        share = parts[1] is None

        def part(index: int) -> Extra:
            def build(side: ProjectedRelation, grouped: bool) -> pl.Expr:
                found = measure.proportions(MeasureInput(side, (), column, "", "baseline"))
                assert found is not None
                chosen = found[index]
                assert chosen is not None
                return chosen

            return build

        extras["successes"] = part(0)
        if not share:
            extras["trials"] = part(1)
    return MeasureTest(
        test.name,
        test,
        statistic=measure.statistic,
        probability=measure.probability(),
        extras=extras,
        share=share,
    )


@dataclass(frozen=True, slots=True)
class Member:
    """One compared member of a family: its materiality and its test input.

    A member is tested by the family's component from ``data``, or by its own
    ``run`` (levels whose tests are not components, such as shape checks).
    ``applicable=False`` marks a member without a sound test (judged by
    materiality, outside the correction).
    """

    material: bool
    label: str | None
    data: TestInput | None = None
    problem: str | None = None
    """Why this member's test cannot run (no values, non-integer counts, ...)."""
    applicable: bool = True
    run: Callable[[], TestResult] | None = None


def _whole(value: Any) -> float | None:
    exact = None if value is None else Fraction(value)
    return float(exact) if exact is not None and exact.denominator == 1 and exact >= 0 else None


def member_input(
    plan: MeasureTest,
    inference: Inference,
    position: int,
    measured: tuple[Any, Any],
    extras: dict[Side, dict[str, Any]],
    totals: dict[Side, float | None],
    margin: float | None,
) -> tuple[TestInput | None, str | None]:
    """The test input of one member, or why it cannot be built."""
    assert plan.test is not None
    needs = plan.test.needs
    common: dict[str, Any] = {
        "statistic": plan.statistic or "mean",
        "probability": plan.probability,
        "seed": inference.seed + position,
        "max_work": inference.max_work,
        "margin": None if margin is None else (-margin, margin),
    }
    if needs in ("samples", "moments") and any(extras[s].get("lossy") for s in extras):
        return None, "values cannot be represented exactly as Float64"
    if needs == "samples":
        arrays = [np.asarray(extras[side]["values"] or [], dtype=float) for side in extras]
        return TestInput(baseline=arrays[0], candidate=arrays[1], **common), None
    if needs == "moments":
        moments = []
        for side in extras:
            values = extras[side]
            n = int(values["n"] or 0)
            mean = math.nan if values["mean"] is None else float(values["mean"])
            var = math.nan if values["var"] is None else float(values["var"])
            moments.append(Moments(n, mean, var))
        return TestInput(moments=(moments[0], moments[1]), **common), None
    if needs == "counts":
        counts = [_whole(value) for value in measured]
        if any(count is None for count in counts):
            return None, "the measure's values are not whole non-negative counts on both sides"
        return TestInput(counts=(counts[0], counts[1]), **common), None  # type: ignore[arg-type]
    if needs == "proportions":
        successes = [_whole(extras[side]["successes"]) for side in extras]
        trials = [totals[side] if plan.share else _whole(extras[side]["trials"]) for side in extras]
        if any(value is None for value in (*successes, *trials)):
            return None, "successes and trials must be whole non-negative counts"
        return TestInput(
            counts=(successes[0], successes[1]),  # type: ignore[arg-type]
            totals=(trials[0], trials[1]),  # type: ignore[arg-type]
            **common,
        ), None
    return None, f"{plan.name} needs {needs}, which this level cannot provide"


@dataclass(frozen=True)
class Judgement:
    """Each member's decision and test, and the family's summary."""

    decisions: list[Decision]
    results: list[TestResult | None]
    significance: Significance

    @property
    def failed(self) -> list[int]:
        return [i for i, decision in enumerate(self.decisions) if decision.failed]

    @property
    def unassessed(self) -> list[int]:
        return [i for i, decision in enumerate(self.decisions) if not decision.assessed]


def judge(
    plan: MeasureTest,
    inference: Inference,
    members: Sequence[Member],
    limit: int,
) -> Judgement:
    """Test, correct and decide one family; ``limit`` bounds the tests kept as evidence.

    Members the policy needs a test for are tested; the applicable members form
    the family the correction counts (those left untested count as ``p = 1``).
    """
    policy = inference.decision
    results: list[TestResult | None] = []
    applicable = [plan.applicable and member.applicable for member in members]
    for member, testable in zip(members, applicable, strict=True):
        if not testable or not needs_test(policy, member.material):
            results.append(None)
            continue
        reason = plan.reason or member.problem
        if reason is not None:
            results.append(not_assessed(plan.name, reason, label=member.label))
        elif member.run is not None:
            results.append(member.run().model_copy(update={"label": member.label}))
        elif plan.test is None or member.data is None:
            results.append(not_assessed(plan.name, "no test input", label=member.label))
        else:
            test = plan.test.equivalent if policy == "equivalence" else plan.test.run
            results.append(test(member.data).model_copy(update={"label": member.label}))
    run = [i for i, result in enumerate(results) if result is not None]
    family = sum(applicable)
    corrected = correct(
        [results[i] for i in run],  # type: ignore[misc]
        inference.correction,
        inference.alpha,
        family_size=family,
    )
    for i, result in zip(run, corrected, strict=True):
        results[i] = result
    decisions = [
        decide(policy, member.material, result, alpha=inference.alpha, applicable=testable)
        for member, result, testable in zip(members, results, applicable, strict=True)
    ]
    tested = [result for result in results if result is not None]
    summary = _summary(plan, inference, decisions, tested, family, limit)
    return Judgement(decisions, results, summary)


def _summary(
    plan: MeasureTest,
    inference: Inference,
    decisions: list[Decision],
    tested: list[TestResult],
    size: int,
    limit: int,
) -> Significance:
    return Significance(
        method=plan.name,
        decision=inference.decision,
        alpha=inference.alpha,
        correction=inference.correction,
        applicable=plan.applicable,
        reason=plan.reason,
        family_size=size,
        significant=sum(bool(result.significant) for result in tested),
        failed=sum(decision.failed for decision in decisions),
        not_assessed=sum(not decision.assessed for decision in decisions),
        tests=tested[:limit],
    )


def evidence_fields(result: TestResult | None, decision: Decision) -> dict[str, Any]:
    """What a failing member's evidence record adds: the test and the decision."""
    if result is None:
        return {"decision": decision.reason}
    return {
        "test": result.method,
        "p_value": result.p_value,
        "adjusted_p": result.adjusted_p,
        "effect": result.effect,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "decision": decision.reason,
    }


def summary_metrics(significance: Significance) -> dict[str, Any]:
    """The family's counts for a check's metrics (and its finding)."""
    return {
        "decision": significance.decision,
        "test": significance.method,
        "alpha": significance.alpha,
        "correction": significance.correction,
        "family_size": significance.family_size,
        "significant": significance.significant,
        "not_assessed": significance.not_assessed,
    }


def family_check(
    check: str,
    plan: MeasureTest,
    inference: Inference,
    members: Sequence[Member],
    records: Sequence[dict[str, Any]],
    limit: int,
    ok: str,
    failed: str,
    noun: str,
    magnitudes: Sequence[float] | None = None,
    unmeasured: Sequence[dict[str, Any]] = (),
) -> CheckResult:
    """Judge one family and grade it; failing members carry their test as evidence.

    Failures decide the check (the members not assessed are counted in its
    message); without failures, members whose test could not run, or that could
    not be measured at all (``unmeasured`` records, each with a ``reason``),
    leave it not assessed: never a pass. Evidence keeps the ``limit`` largest
    ``magnitudes``. Under ``material`` no test runs and material members fail.
    ``failed`` may name the failing members as ``{count}``.
    """
    from qavf.compare.levels import graded, not_assessed

    if not members and not unmeasured:
        return not_assessed(check, f"no {noun} to compare on both sides")
    sizes = list(magnitudes) if magnitudes is not None else [0.0] * len(records)
    leaders = LargestEvidence(limit)
    significance: Significance | None = None
    metrics: dict[str, Any] = {}
    reason = None
    if inference.tests:
        judgement = judge(plan, inference, members, limit)
        failing, unassessed = judgement.failed, judgement.unassessed
        significance = judgement.significance
        metrics = summary_metrics(significance)
        for position in failing or unassessed:
            record = records[position] | evidence_fields(
                judgement.results[position], judgement.decisions[position]
            )
            leaders.add(sizes[position], position, record)
        if unassessed:
            reason = judgement.decisions[unassessed[0]].reason
    else:
        failing = [i for i, member in enumerate(members) if member.material]
        unassessed = []
        for position in failing:
            leaders.add(sizes[position], position, records[position])
    total = len(members) + len(unmeasured)
    if (unassessed or unmeasured) and not failing:
        evidence = [*leaders.records, *unmeasured][:limit]
        if unassessed:
            message = (
                f"Not assessed: {check}: {plural(len(unassessed), noun)} need a "
                f"{plan.name} test that could not run ({reason})."
            )
            if unmeasured:
                message = message[:-1] + f"; {plural(len(unmeasured), noun)} not measured."
        else:
            message = (
                f"Not assessed: {check}: {plural(len(unmeasured), noun)} could not be "
                f"measured ({unmeasured[0].get('reason')})."
            )
        return CheckResult(
            check=check,
            passed=False,
            assessed=False,
            message=message,
            affected_count=len(unassessed) + len(unmeasured),
            total=total,
            unit=f"{noun}s",
            affected_records=evidence,
            truncated=len(unassessed) + len(unmeasured) > len(evidence),
            metrics=metrics,
            significance=significance,
        )
    skipped = len(unassessed) + len(unmeasured)
    result = graded(
        check,
        len(failing),
        ok,
        failed.format(count=plural(len(failing), noun))
        + (f" ({skipped:,} not assessed)" if skipped else ""),
        total=total,
        unit=f"{noun}s",
        metrics=metrics,
    )
    return result.model_copy(
        update={
            "affected_records": leaders.records,
            "truncated": len(failing) > len(leaders.records),
            "significance": significance,
        }
    )


def label(keys: Sequence[str], values: Sequence[Any]) -> str | None:
    """A member's label: ``key=value`` pairs (``None`` without keys)."""
    if not keys:
        return None
    return ", ".join(f"{key}={value}" for key, value in zip(keys, values, strict=True))
