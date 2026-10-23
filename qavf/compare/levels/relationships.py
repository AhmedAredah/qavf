"""Relationships: how columns relate on each side, and how the two versions agree.

Members are ``group_by`` groups and, with ``time``, time buckets within them
(:func:`~qavf.compare.levels.temporal.keyed_sides`), as for every keyed level.

* ``correlation``: each pair of the selected columns is correlated on each
  side (Pearson, and Spearman on average ranks, reduced natively per member);
  a change beyond ``max_change`` is material and Fisher's z test of two
  independent correlations decides significance (Spearman with the
  Fieller-Hartley-Pearson variance), corrected across the pairs and members of
  each method;
* ``ratios``: declared invariants such as ``fatal / crashes`` or ``nox / vmt``,
  ``sum(numerator) / sum(denominator)`` per member, exact; material beyond
  their tolerance, tested as two Poisson rates when the numerators are counts
  (otherwise judged by materiality alone);
* ``agreement``: on records matched by the comparison's ``keys`` (members by
  the candidate's values), each column's Bland-Altman bias and limits of
  agreement, Lin's concordance correlation, MAE, RMSE, bias, MAPE, sMAPE, R²
  and the candidate-on-baseline line (OLS or Deming); thresholds per metric
  decide materiality and the joint test of ``slope = 1, intercept = 0``
  significance.

None of these tests has an equivalence form: under ``decision: equivalence``
the checks are not assessed. A member that cannot be measured (too few pairs,
a constant column, a zero denominator) is not assessed, never passed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from qavf.compare.levels import SIDES, ComparisonLevel, LevelInputs, not_assessed, register_level
from qavf.compare.levels.groups import (
    MeasureSpec,
    group_partitions,
    materiality,
    precise_frame,
    reduce_groups,
)
from qavf.compare.levels.temporal import KeyedSides, TimeMembers, keyed_sides
from qavf.compare.results import (
    Agreement,
    CheckResult,
    CorrelationChange,
    LevelResult,
    RatioChange,
    RelationshipsResult,
)
from qavf.compare.significance import (
    Inference,
    MeasureTest,
    Member,
    family_check,
)
from qavf.compare.significance import label as member_label
from qavf.compare.spec import ComparisonSpec, InferenceParams, Side
from qavf.engine.registry import RuleParams
from qavf.inference import effects
from qavf.inference._support import Unassessable
from qavf.inference.agreement import FitMethod, fit_line, identity_test
from qavf.inference.registry import TESTS, TestInput
from qavf.inference.results import TestResult
from qavf.inference.results import not_assessed as untested
from qavf.inference.trend import FIELLER_VARIANCE, compare_correlations
from qavf.keys import align_keys
from qavf.numeric import exact_rational, report_number
from qavf.rules._grouped import ProjectedRelation
from qavf.rules.helpers import fresh_internal, plural
from qavf.statistics.series import pearson, spearman


class CorrelationCheck(RuleParams):
    """Correlations of every pair of ``columns`` should not change by ``max_change``."""

    columns: tuple[str, ...]
    """Numeric columns; every pair is compared."""
    methods: tuple[Literal["pearson", "spearman"], ...] = ("pearson", "spearman")
    max_change: float = Field(default=0.1, ge=0, le=2)
    """Largest acceptable change of a correlation coefficient."""
    min_pairs: int = Field(default=4, ge=4)
    """Fewest complete rows on each side for a correlation to be measured."""

    @model_validator(mode="after")
    def _pairs(self) -> CorrelationCheck:
        if len(set(self.columns)) < 2 or len(set(self.columns)) != len(self.columns):
            raise ValueError("'columns' must name at least two distinct columns")
        if not self.methods:
            raise ValueError("'methods' must name pearson, spearman or both")
        return self


class RatioCheck(RuleParams):
    """A ratio invariant ``sum(numerator) / sum(denominator)``: ``fatal / crashes``."""

    numerator: str
    denominator: str
    name: str | None = None
    """Label in reports; ``numerator / denominator`` when omitted."""
    rtol: float | None = Field(default=None, ge=0)
    """Relative tolerance of the ratio; the level's, then the comparison's."""
    atol: float | None = Field(default=None, ge=0)
    test: Literal["poisson", "none"] | None = None
    """``poisson``: two event rates per exposure (whole-number numerators); ``none``:
    judged by materiality alone. By default Poisson for integer numerators."""

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            numerator, slash, denominator = data.partition("/")
            if not slash or not numerator.strip() or not denominator.strip():
                raise ValueError(f"a ratio is 'numerator / denominator', got {data!r}")
            return {"numerator": numerator.strip(), "denominator": denominator.strip()}
        return data

    @property
    def label(self) -> str:
        return self.name or f"{self.numerator} / {self.denominator}"


class AgreementCheck(RuleParams):
    """Matched values agree: thresholds per metric (``null`` leaves one out)."""

    columns: tuple[str, ...]
    """Numeric columns compared record by record."""
    fit: FitMethod = "ols"
    """The candidate-on-baseline line: ``ols`` (the baseline as exact) or ``deming``
    (both sides measured with error, in ``error_ratio``)."""
    error_ratio: float = Field(default=1.0, gt=0)
    """Deming: the candidate's error variance over the baseline's."""
    min_pairs: int = Field(default=3, ge=3)
    """Fewest matched pairs for a member to be measured."""
    max_abs_bias: float | None = Field(default=None, ge=0)
    """Largest ``|mean(candidate - baseline)|``, in data units."""
    max_loa: float | None = Field(default=None, ge=0)
    """Largest ``|limit of agreement|`` (bias -/+ 1.96 sd), in data units."""
    min_ccc: float | None = Field(default=0.99, ge=-1, le=1)
    """Lowest Lin's concordance correlation."""
    max_mae: float | None = Field(default=None, ge=0)
    max_rmse: float | None = Field(default=None, ge=0)
    max_mape: float | None = Field(default=None, ge=0)
    """Largest mean absolute percentage difference (percent)."""
    max_smape: float | None = Field(default=None, ge=0)
    min_r_squared: float | None = Field(default=None, le=1)
    max_slope_deviation: float | None = Field(default=0.05, ge=0)
    """Largest ``|slope - 1|`` of the fitted line."""
    max_abs_intercept: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _columns(self) -> AgreementCheck:
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("'columns' must name distinct columns")
        return self


class RelationshipsParams(InferenceParams):
    """Parameters of the ``relationships`` level."""

    group_by: tuple[str, ...] = ()
    """Compare within each group (a zone column too); the whole side when empty."""
    time: TimeMembers | None = None
    """Also compare per time bucket: ``{time_column: day, bucket: month}``."""
    correlation: CorrelationCheck | None = None
    ratios: tuple[RatioCheck, ...] = ()
    """Ratio invariants, ``"fatal / crashes"`` or ``{numerator, denominator, ...}``."""
    agreement: AgreementCheck | None = None
    """Agreement of matched records (needs the comparison's ``keys``)."""
    rtol: float | None = Field(default=None, ge=0)
    """Default relative tolerance of the ratios; the comparison's when omitted."""
    atol: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _something(self) -> RelationshipsParams:
        if self.correlation is None and not self.ratios and self.agreement is None:
            raise ValueError("configure 'correlation', 'ratios' or 'agreement'")
        labels = [ratio.label for ratio in self.ratios]
        if repeated := sorted({label for label in labels if labels.count(label) > 1}):
            raise ValueError(f"ratios repeat the label(s) {', '.join(repeated)}; set 'name'")
        return self


@dataclass
class _Family:
    """One check's members: their materiality, evidence and tests."""

    members: list[Member] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    magnitudes: list[float] = field(default_factory=list)
    unmeasured: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        material: bool,
        name: str | None,
        record: dict[str, Any],
        magnitude: float,
        run: Callable[[], TestResult] | None,
    ) -> None:
        self.members.append(Member(material, name, applicable=run is not None, run=run))
        self.records.append(record)
        self.magnitudes.append(magnitude)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _equivalence(method: str) -> TestResult:
    return untested(method, f"{method} has no equivalence form")


@register_level
class RelationshipsLevel(ComparisonLevel):
    """Correlation changes, ratio invariants and agreement of matched records."""

    name: ClassVar[str] = "relationships"
    description: ClassVar[str] = (
        "How columns relate on each side and how the versions agree, per member (group, "
        "zone or time bucket): Pearson and Spearman correlation changes (Fisher z), ratio "
        "invariants such as fatal / crashes (Poisson rates), and for keyed records "
        "Bland-Altman limits, concordance, error metrics and the candidate-on-baseline "
        "line tested against the identity."
    )
    Params: ClassVar[type[RuleParams]] = RelationshipsParams
    Result: ClassVar[type[LevelResult]] = RelationshipsResult

    def configuration_errors(self, spec: ComparisonSpec) -> list[str]:
        params: RelationshipsParams = self.params  # type: ignore[assignment]
        if params.agreement is not None and not spec.keys:
            return ["'agreement' needs the comparison's 'keys' to match records"]
        return []

    def required_columns(self) -> frozenset[str]:
        params: RelationshipsParams = self.params  # type: ignore[assignment]
        columns = set(params.group_by)
        if params.time is not None:
            columns.add(params.time.time_column)
        if params.correlation is not None:
            columns.update(params.correlation.columns)
        if params.agreement is not None:
            columns.update(params.agreement.columns)
        for ratio in params.ratios:
            columns.update((ratio.numerator, ratio.denominator))
        return frozenset(columns)

    def evaluate(self, inputs: LevelInputs, name: str) -> RelationshipsResult:
        params: RelationshipsParams = self.params  # type: ignore[assignment]
        inference = Inference.of(inputs.spec, params)
        sides = keyed_sides(inputs, params.group_by, params.time)
        relations = sides.relations or {side: inputs.relation(side) for side in SIDES}
        checks: list[CheckResult] = []
        details: dict[str, Any] = {}
        limit = inputs.ctx.evidence_limit
        if params.correlation is not None:
            try:
                found, changes = self._correlations(
                    inputs, params.correlation, sides, relations, inference
                )
            except ValueError as exc:
                found = [not_assessed("correlation", str(exc).rstrip("."))]
                changes = []
            checks.extend(found)
            details["correlations"] = changes[:limit]
        ratios: list[RatioChange] = []
        for ratio in params.ratios:
            try:
                check, ratio_changes = self._ratio(
                    inputs, params, ratio, sides, relations, inference
                )
            except ValueError as exc:
                check = not_assessed(f"ratio: {ratio.label}", str(exc).rstrip("."))
                ratio_changes = []
            checks.append(check)
            ratios.extend(ratio_changes)
        details["ratios"] = ratios[:limit]
        if params.agreement is not None:
            try:
                found, agreement, matched = self._agreement(
                    inputs, params.agreement, sides, relations, inference
                )
                details |= {"agreement": agreement[:limit], "matched": matched}
            except ValueError as exc:
                found = [not_assessed("agreement", str(exc).rstrip("."))]
            checks.extend(found)
        return RelationshipsResult(
            level=self.name,
            name=name,
            checks=checks,
            group_by=list(params.group_by),
            bucket=sides.bucket,
            **details,
        )

    @staticmethod
    def _decide(
        inputs: LevelInputs,
        check: str,
        plan: MeasureTest,
        inference: Inference,
        family: _Family,
        what: str,
        noun: str,
    ) -> CheckResult:
        how = "" if not inference.tests else f"; {plan.name}, {inference.describe()}"
        size = len(family.members)
        return family_check(
            check,
            plan,
            inference,
            family.members,
            family.records,
            inputs.ctx.evidence_limit,
            f"{what} holds in {plural(size, noun)}{how}.",
            f"{what} changed in {{count}} of {size:,}{how}",
            noun,
            magnitudes=family.magnitudes,
            unmeasured=family.unmeasured,
        )

    # ---------------------------------------------------------- correlations
    def _correlations(
        self,
        inputs: LevelInputs,
        check: CorrelationCheck,
        sides: KeyedSides,
        relations: dict[Side, ProjectedRelation],
        inference: Inference,
    ) -> tuple[list[CheckResult], list[CorrelationChange]]:
        for side in SIDES:
            for column in check.columns:
                dtype = relations[side].schema.get(column)
                if dtype is None or not dtype.is_numeric():
                    raise ValueError(f"{side} column {column!r} is missing or not numeric")
        pair_count = len(check.columns) * (len(check.columns) - 1) // 2
        work = (
            max(1, sum(inputs.rows(side) for side in SIDES)) * 2 * pair_count * len(check.methods)
        )
        if work > inference.max_work:
            raise ValueError(
                f"correlation pair/method work {work:,} exceeds max_work {inference.max_work:,}"
            )
        pairs = list(combinations(check.columns, 2))
        # One pair at a time: raw samples no longer expand quadratically in width.
        # Results retain scalar coefficients only. Selected methods alone execute.
        measured: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
        keys = sides.keys
        for index, (a, b) in enumerate(pairs):
            roles = {"x": _complete(a, b, a), "y": _complete(a, b, b)}
            aligned = reduce_groups(
                inputs, keys, (MeasureSpec(measure="count"),), relations, [roles]
            )
            named = aligned.extras[0]
            columns = [aligned.columns[side][named[axis]] for side in SIDES for axis in ("x", "y")]
            for row in precise_frame(aligned.both.select([*keys, *columns]), keys).iter_rows():
                group = dict(zip(keys, row[: len(keys)], strict=True))
                values = row[len(keys) :]
                found_r: dict[str, Any] = {"unusable": False}
                for side, (x, y) in zip(SIDES, (values[:2], values[2:]), strict=True):
                    found_r[side] = _exact_correlations(x or [], y or [], check.methods)
                measured.append((group, index, found_r))
        results: list[CheckResult] = []
        changes: list[CorrelationChange] = []
        for method in check.methods:
            family = _Family()
            found: list[CorrelationChange] = []
            for group, index, correlations in measured:
                a, b = pairs[index]
                nb, rb_ = correlations["baseline"][0], correlations["baseline"][1][method]
                nc, rc_ = correlations["candidate"][0], correlations["candidate"][1][method]
                rb, rc = _float(rb_), _float(rc_)
                sizes = {"baseline": nb, "candidate": nc}
                if correlations["unusable"]:
                    rb = rc = None
                record = {**group, "columns": f"{a}, {b}", "baseline": rb, "candidate": rc,
                          **{f"{side}_pairs": n for side, n in sizes.items()}}  # fmt: skip
                if rb is None or rc is None or min(sizes.values()) < check.min_pairs:
                    reason = (
                        f"fewer than {check.min_pairs} complete rows on a side"
                        if min(sizes.values()) < check.min_pairs
                        else "values cannot be represented exactly as Float64"
                        if correlations["unusable"]
                        else "a column is constant; the correlation is undefined"
                    )
                    family.unmeasured.append({**record, "reason": reason})
                    continue
                change = rc - rb
                material = abs(change) > check.max_change
                variance = FIELLER_VARIANCE if method == "spearman" else 1.0

                def run(
                    rb: float = rb,
                    rc: float = rc,
                    sizes: dict[str, int] = sizes,
                    variance: float = variance,
                ) -> TestResult:
                    if inference.decision == "equivalence":
                        return _equivalence("fisher_z")
                    return compare_correlations(
                        (rb, sizes["baseline"]), (rc, sizes["candidate"]), variance=variance
                    )  # fmt: skip

                label = ", ".join(
                    filter(None, [member_label(keys, list(group.values())), f"{a}~{b}"])
                )
                family.add(material, label, {**record, "change": change}, abs(change), run)
                found.append(
                    CorrelationChange(
                        group=group, method=method, columns=[a, b], baseline=rb,
                        candidate=rc, pairs=sizes, change=change, material=material,
                    )
                )  # fmt: skip
            plan = MeasureTest("fisher_z", None)
            what = f"{method.capitalize()} correlation of {', '.join(check.columns)}"
            decided = self._decide(
                inputs, f"correlation: {method}", plan, inference, family, what, "pair"
            )
            results.append(decided)
            changes.extend(_with_tests(found, family, decided))
        return results, changes

    # ---------------------------------------------------------------- ratios
    def _ratio(
        self,
        inputs: LevelInputs,
        params: RelationshipsParams,
        ratio: RatioCheck,
        sides: KeyedSides,
        relations: dict[Side, ProjectedRelation],
        inference: Inference,
    ) -> tuple[CheckResult, list[RatioChange]]:
        specs = (
            MeasureSpec(measure="sum", column=ratio.numerator, name="numerator"),
            MeasureSpec(measure="sum", column=ratio.denominator, name="denominator"),
        )
        aligned = reduce_groups(inputs, sides.keys, specs, relations)
        judged = materiality(
            inputs.spec, params, MeasureSpec(measure="count", rtol=ratio.rtol, atol=ratio.atol)
        )
        keys = aligned.keys
        columns = [aligned.columns[side][alias] for alias in aligned.aliases for side in SIDES]
        rows = list(precise_frame(aligned.both.select([*keys, *columns]), keys).iter_rows())
        parts = [[exact_rational(v) for v in row[len(keys) :]] for row in rows]
        whole = all(
            value is not None and value >= 0 and value.denominator == 1
            for numbers in parts
            for value in numbers[:2]
        )
        wanted = ratio.test or ("poisson" if whole else "none")
        if wanted == "poisson" and not whole:
            raise ValueError(
                f"the Poisson test needs whole non-negative sums of {ratio.numerator!r}"
            )
        test = TESTS.create("poisson") if wanted == "poisson" else None
        plan = (
            MeasureTest("poisson", test)
            if test is not None
            else MeasureTest("none", None, False, "no test declared; judged by materiality")
        )
        family = _Family()
        found: list[RatioChange] = []
        for position, (row, numbers) in enumerate(zip(rows, parts, strict=True)):
            group = dict(zip(keys, row[: len(keys)], strict=True))
            nb, nc, db, dc = numbers
            record = {**group, "ratio": ratio.label}
            if nb is None or nc is None or not db or not dc or db < 0 or dc < 0:
                family.unmeasured.append(
                    {**record, "reason": "a side has no positive denominator or no numerator"}
                )
                continue
            before, after = nb / db, nc / dc
            material = judged.material(before, after)
            record |= {"baseline": report_number(before), "candidate": report_number(after),
                       "change": report_number(after - before)}  # fmt: skip
            run = None
            if test is not None:
                data = TestInput(
                    counts=(float(nb), float(nc)), totals=(float(db), float(dc)),
                    seed=inference.seed + position, max_work=inference.max_work,
                )  # fmt: skip
                runner = test.equivalent if inference.decision == "equivalence" else test.run

                def run(data: TestInput = data, runner: Any = runner) -> TestResult:
                    return runner(data)

            name = member_label(keys, list(group.values()))
            family.add(material, name, record, float(abs(after - before)), run)
            found.append(
                RatioChange(
                    name=ratio.label, group=group, baseline=float(before),
                    candidate=float(after), change=float(after - before), material=material,
                )
            )  # fmt: skip
        noun = "bucket" if params.time is not None else "group"
        check = self._decide(
            inputs, f"ratio: {ratio.label}", plan, inference, family, ratio.label, noun
        )
        return check, _with_tests(found, family, check)

    # ------------------------------------------------------------- agreement
    def _agreement(
        self,
        inputs: LevelInputs,
        check: AgreementCheck,
        sides: KeyedSides,
        relations: dict[Side, ProjectedRelation],
        inference: Inference,
    ) -> tuple[list[CheckResult], list[Agreement], int]:
        keys = list(inputs.spec.keys)
        # A member key may also identify the record (site, year, ...). It is
        # already projected in that case, but still belongs in the partition.
        members = list(sides.keys)
        values = sum(inputs.rows(side) for side in SIDES) * len(check.columns)
        if values > inference.max_work:
            raise ValueError(
                f"collecting {values:,} agreement values exceeds max_work {inference.max_work:,}"
            )
        frames: dict[Side, pl.DataFrame] = {}
        for side in SIDES:
            relation = relations[side]
            wanted = [*keys, *(members if side == "candidate" else []), *check.columns]
            if absent := [c for c in wanted if c not in relation.schema]:
                raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
            for column in check.columns:
                if not relation.schema[column].is_numeric():
                    raise ValueError(f"{side} column {column!r} is not numeric")
            assert relation.lazy is not None
            frame = relation.collect_plan(relation.lazy.select(list(dict.fromkeys(wanted))))
            if frame.select(keys).is_duplicated().any():
                raise ValueError(f"the {side} repeats key values; records cannot be matched")
            frames[side] = frame
        taken = [*frames["baseline"].columns, *frames["candidate"].columns]
        renamed = {c: fresh_internal(f"baseline_{c}", taken) for c in check.columns}
        base, cand = align_keys(
            frames["baseline"].select(*keys, *check.columns).rename(renamed),
            frames["candidate"],
            keys,
        )
        # Key order makes the floating-point metrics independent of row order.
        matched = base.join(cand, on=keys, how="inner", nulls_equal=True).sort(keys)
        results: list[CheckResult] = []
        agreement: list[Agreement] = []
        ordered = sorted(group_partitions(matched, members), key=lambda item: _order(item[0]))
        noun = "bucket" if sides.bucket is not None else "group"
        for column in check.columns:
            family = _Family()
            found: list[Agreement] = []
            for key, frame in ordered:
                group = dict(zip(members, key, strict=True))
                pair = frame.select(renamed[column], column).drop_nulls()
                floating = [name for name, dtype in pair.schema.items() if dtype.is_float()]
                if floating:
                    pair = pair.filter(pl.all_horizontal(pl.col(floating).is_finite()))
                record = {**group, "column": column, "pairs": pair.height}
                if pair.height < check.min_pairs:
                    reason = f"fewer than min_pairs {check.min_pairs} matched pairs"
                    family.unmeasured.append({**record, "reason": reason})
                    continue
                exact = any(dtype.is_integer() or dtype.is_decimal() for dtype in pair.dtypes)
                if exact:
                    originals = [
                        pair.to_series(i).to_list()
                        if pair.dtypes[i].is_integer()
                        else list(map(Fraction, pair.to_series(i)))
                        for i in (0, 1)
                    ]
                    b = c = np.empty(0)
                    if check.fit != "ols" or inference.tests:
                        # A shared translation preserves the identity null. Admission
                        # prevents differences being rounded away by the inference kernel.
                        origin = originals[0][0]
                        centred = [[v - origin for v in values] for values in originals]
                        lossless = all(
                            math.isfinite(float(v)) and Fraction(float(v)) == v
                            for values in centred
                            for v in values
                        )
                        if not lossless:
                            family.unmeasured.append(
                                {
                                    **record,
                                    "reason": "centred exact values cannot be represented "
                                    "as Float64 for inference",
                                }
                            )
                            continue
                        b, c = (np.asarray(values, dtype=float) for values in centred)
                    measured = _exact_agreement(check, column, group, *originals)
                else:
                    b, c = (pair.to_series(i).cast(pl.Float64).to_numpy() for i in (0, 1))
                    measured = _agreement(check, column, group, b, c)
                record |= {
                    name: getattr(measured, name)
                    for name in ("bias", "lower", "upper", "ccc", "mae", "rmse", "slope",
                                 "intercept")
                }  # fmt: skip
                record["material"] = ", ".join(measured.material) or "none"

                def run(b: np.ndarray = b, c: np.ndarray = c) -> TestResult:
                    if inference.decision == "equivalence":
                        return _equivalence("identity_fit")
                    return identity_test(b, c, method=check.fit, error_ratio=check.error_ratio)

                name = member_label(members, list(key)) if members else column
                magnitude = abs(measured.bias or 0.0)
                family.add(bool(measured.material), name, record, magnitude, run)
                found.append(measured)
            plan = MeasureTest("identity_fit", None)
            what = f"Agreement of {column}"
            decided = self._decide(
                inputs, f"agreement: {column}", plan, inference, family, what, noun
            )
            results.append(decided)
            agreement.extend(_with_tests(found, family, decided))
        return results, agreement, matched.height


def _numbers(name: str, side: ProjectedRelation) -> pl.Expr:
    from qavf.aggregation import numeric_expr

    return numeric_expr(name, side.schema[name]).cast(pl.Float64)


def _both(a: str, b: str, side: ProjectedRelation) -> pl.Expr:
    return _numbers(a, side).is_not_null() & _numbers(b, side).is_not_null()


def _complete(a: str, b: str, column: str) -> Callable[[ProjectedRelation, bool], pl.Expr]:
    """A member's values of ``column`` over its rows where both ``a`` and ``b`` exist."""

    def build(side: ProjectedRelation, grouped: bool) -> pl.Expr:
        from qavf.aggregation import numeric_expr

        kept = numeric_expr(column, side.schema[column]).filter(_both(a, b, side))
        return kept if grouped else kept.implode()

    return build


def _exact_correlations(
    x: list[Any], y: list[Any], methods: tuple[str, ...] = ("pearson", "spearman")
) -> tuple[int, dict[str, float | None]]:
    """Pearson and Spearman correlations of paired values, exact until rounded once."""
    if len(x) < 2:
        return len(x), dict.fromkeys(methods)
    xs, ys = [Fraction(v) for v in x], [Fraction(v) for v in y]
    kernels = {"pearson": pearson, "spearman": spearman}
    found = {method: kernels[method](xs, ys) for method in methods}
    return len(x), {name: None if r is None else r.value for name, r in found.items()}


def _order(key: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((value is None, str(type(value)), value) for value in key)


def _agreement(
    check: AgreementCheck, column: str, group: dict[str, Any], b: np.ndarray, c: np.ndarray
) -> Agreement:
    """Every agreement metric of one member's matched pairs and which pass a threshold."""
    limits = effects.bland_altman(b, c)
    slope: float | None
    intercept: float | None
    try:
        line = fit_line(b, c, method=check.fit, error_ratio=check.error_ratio)
        slope, intercept = line.slope, line.intercept
    except Unassessable:
        slope = intercept = None
    found = {
        "bias": limits.bias,
        "sd": limits.sd,
        "lower": limits.lower,
        "upper": limits.upper,
        "ccc": effects.concordance(b, c),
        "mae": effects.mae(b, c),
        "rmse": effects.rmse(b, c),
        "mape": effects.mape(b, c),
        "smape": effects.smape(b, c),
        "r_squared": effects.r_squared(b, c),
        "slope": slope,
        "intercept": intercept,
    }
    found = {name: _float(value) for name, value in found.items()}
    return _agreement_result(check, column, group, len(b), found)


def _agreement_result(
    check: AgreementCheck,
    column: str,
    group: dict[str, Any],
    pairs: int,
    found: dict[str, Any],
) -> Agreement:
    """Judge exact metrics before converting the report's display values."""
    slope, intercept = found["slope"], found["intercept"]
    material = []
    above = (
        ("bias", abs(found["bias"] or 0.0), check.max_abs_bias),
        ("limits", max(abs(found["lower"] or 0.0), abs(found["upper"] or 0.0)), check.max_loa),
        ("mae", found["mae"], check.max_mae),
        ("rmse", found["rmse"], check.max_rmse),
        ("mape", found["mape"], check.max_mape),
        ("smape", found["smape"], check.max_smape),
        ("intercept", None if intercept is None else abs(intercept), check.max_abs_intercept),
        ("slope", None if slope is None else abs(slope - 1), check.max_slope_deviation),
    )
    for name, value, limit in above:
        if limit is not None and (
            value is None
            or value > (Fraction(str(limit)) if isinstance(value, Fraction) else limit)
        ):
            material.append(name)
    below = (("ccc", found["ccc"], check.min_ccc), ("r_squared", found["r_squared"],
                                                       check.min_r_squared))  # fmt: skip
    for name, value, limit in below:
        if limit is not None and (
            value is None
            or value < (Fraction(str(limit)) if isinstance(value, Fraction) else limit)
        ):
            material.append(name)
    return Agreement(
        column=column,
        group=group,
        pairs=pairs,
        fit=check.fit,
        material=material,
        **{name: _float(value) for name, value in found.items()},
    )


def _exact_agreement(
    check: AgreementCheck,
    column: str,
    group: dict[str, Any],
    b: list[Fraction],
    c: list[Fraction],
) -> Agreement:
    """Exact differences and centred moments for integral/decimal measurements."""
    n = len(b)
    sb, sc = sum(b), sum(c)
    mb, mc = Fraction(sb) / n, Fraction(sc) / n
    # Exact sufficient statistics need no centred sample copies; Python integer
    # accumulation stays cheap for the common integral input path.
    sbb = Fraction(sum(v * v for v in b)) - Fraction(sb * sb) / n
    scc = Fraction(sum(v * v for v in c)) - Fraction(sc * sc) / n
    sbc = Fraction(sum(a * v for a, v in zip(b, c, strict=True))) - Fraction(sb * sc) / n
    d = [after - before for before, after in zip(b, c, strict=True)]
    bias = Fraction(sum(d)) / n
    squares = Fraction(sum(v * v for v in d))
    sd = math.sqrt(float((squares - n * bias * bias) / (n - 1)))
    slope: Fraction | float | None = sbc / sbb if sbb else None
    if check.fit == "deming":
        ratio = Fraction(str(check.error_ratio))
        gap = scc - ratio * sbb
        discriminant = gap * gap + 4 * ratio * sbc * sbc
        numerator = math.isqrt(discriminant.numerator)
        denominator_root = math.isqrt(discriminant.denominator)
        root = (
            Fraction(numerator, denominator_root)
            if numerator * numerator == discriminant.numerator
            and denominator_root * denominator_root == discriminant.denominator
            else Fraction(math.sqrt(float(discriminant)))
        )
        # Rationalize the negative-gap branch to avoid subtractive cancellation.
        slope = (
            ((gap + root) / (2 * sbc) if gap >= 0 else 2 * ratio * sbc / (root - gap))
            if sbc
            else None
        )
    intercept = mc - Fraction(slope) * mb if slope is not None else None
    denominator = sbb + scc + n * (mb - mc) ** 2
    found: dict[str, Any] = {
        "bias": bias,
        "sd": sd,
        "lower": bias if sd == 0 else float(bias) - 1.96 * sd,
        "upper": bias if sd == 0 else float(bias) + 1.96 * sd,
        "ccc": 2 * sbc / denominator if denominator else None,
        "mae": sum(map(abs, d), Fraction()) / n,
        "rmse": math.sqrt(float(squares / n)),
        "mape": None
        if any(v == 0 for v in b)
        else sum(abs(Fraction(delta) / before) for delta, before in zip(d, b, strict=True))
        / n
        * 100,
        "smape": sum(
            2 * abs(Fraction(delta)) / (abs(before) + abs(after)) if before or after else Fraction()
            for delta, before, after in zip(d, b, c, strict=True)
        )
        / n
        * 100,
        "r_squared": 1 - squares / sbb if sbb else None,
        "slope": slope,
        "intercept": intercept,
    }
    result = _agreement_result(check, column, group, n, found)
    if check.max_rmse is not None:
        # Compare squared quantities without rounding at the threshold.
        material = [name for name in result.material if name != "rmse"]
        if squares / n > Fraction(str(check.max_rmse)) ** 2:
            material.append("rmse")
        result = result.model_copy(update={"material": material})
    return result


def _with_tests[T: (CorrelationChange, RatioChange, Agreement)](
    found: list[T], family: _Family, check: CheckResult
) -> list[T]:
    """Attach each member's test (from the check's family summary) to its detail."""
    tests = check.significance.tests if check.significance is not None else []
    by_label = {test.label: test for test in tests}
    labelled = [m.label for m in family.members]
    return [
        item.model_copy(update={"test": by_label[name]}) if name in by_label else item
        for item, name in zip(found, labelled, strict=True)
    ]
