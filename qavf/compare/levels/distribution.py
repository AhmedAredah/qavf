"""Distribution: how each column's distribution changed, per member or overall.

Members are ``group_by`` groups (zones are group columns too) and, with
``time``, time buckets within them, keyed exactly as by the ``groups`` and
``temporal`` levels (:func:`~qavf.compare.levels.temporal.keyed_sides`). Each
side's values are collected per member natively (the reductions significance
tests read, so the work budget ``max_work`` applies), then compared:

* numeric columns: the population stability index over the baseline's
  quantile bins with support tails (STA-002's binning), the Kolmogorov-Smirnov
  and Wasserstein distances, the Jensen-Shannon divergence on those shared
  bins and quantile-by-quantile deltas; tested by KS (or Anderson-Darling);
* categorical columns: PSI and Jensen-Shannon over the categories, tested by
  the chi-square (or G) test of homogeneity; categories that appeared or
  disappeared are reported in their own check.

A member is material when a metric passes its threshold (``thresholds``,
``quantiles``); each column's members form one family for the correction, and
the decision policy combines materiality with the test. The distances have no
equivalence form, so ``decision: equivalence`` leaves the checks not assessed.
A member with too few values on a side is not assessed, never passed.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from qavf.compare.levels.groups import (
    MeasureSpec,
    materiality,
    one_sided_check,
    precise_frame,
    reduce_groups,
)
from qavf.compare.levels.temporal import KeyedSides, TimeMembers, keyed_sides
from qavf.compare.results import (
    BinShare,
    CheckResult,
    ColumnDistribution,
    DistributionMember,
    DistributionResult,
    LevelResult,
    QuantileDelta,
)
from qavf.compare.significance import (
    Inference,
    MeasureTest,
    Member,
    family_check,
    sample_extras,
)
from qavf.compare.significance import label as member_label
from qavf.compare.spec import InferenceParams, Side
from qavf.engine.registry import RuleParams
from qavf.inference.categorical import chi_square, g_test
from qavf.inference.effects import (
    jensen_shannon,
    ks_distance,
    population_stability,
    quantile_deltas,
    wasserstein,
)
from qavf.inference.registry import TESTS, TestInput
from qavf.inference.results import TestResult
from qavf.inference.results import not_assessed as untested
from qavf.key_codec import canonical_key
from qavf.rules._grouped import ProjectedRelation
from qavf.rules.helpers import plural
from qavf.statistics.ordered import bin_index, power_of_two, quantile_cutpoints


class DistributionColumn(RuleParams):
    """A compared column: ``speed`` or ``{column: fc, kind: categorical}``."""

    column: str
    kind: Literal["auto", "numeric", "categorical"] = "auto"
    """``auto``: numeric columns are numeric, every other type categorical; integer
    codes can be compared as ``categorical``."""

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        return {"column": data} if isinstance(data, str) else data


class Thresholds(RuleParams):
    """When a distribution change is material (``null`` leaves a metric out)."""

    max_psi: float | None = Field(default=0.2, ge=0)
    """Population stability index (0.1 small, 0.2 notable, 0.25 large)."""
    max_ks: float | None = Field(default=0.1, ge=0, le=1)
    """Kolmogorov-Smirnov distance, the largest gap of the CDFs (numeric)."""
    max_wasserstein: float | None = Field(default=None, ge=0)
    """Earth mover's distance in the column's units (numeric; off by default)."""
    max_js: float | None = Field(default=0.05, ge=0, le=1)
    """Jensen-Shannon divergence in bits on the shared bins or categories."""


class QuantileCheck(RuleParams):
    """Quantile-by-quantile deltas, material beyond ``max(atol, rtol * max(|b|, |c|))``."""

    probabilities: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    rtol: float | None = Field(default=None, ge=0)
    """Relative tolerance; the column's, then the level's, then the comparison's."""
    atol: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _probabilities(self) -> QuantileCheck:
        if not self.probabilities or any(not 0 <= p <= 1 for p in self.probabilities):
            raise ValueError("'probabilities' must be values in [0, 1]")
        return self


class DistributionParams(InferenceParams):
    """Parameters of the ``distribution`` level."""

    columns: tuple[DistributionColumn, ...]
    """Columns whose distributions are compared."""
    group_by: tuple[str, ...] = ()
    """Compare within each group (a zone column too); the whole side when empty."""
    time: TimeMembers | None = None
    """Also compare per time bucket: ``{time_column: day, bucket: month}``."""
    bins: int = Field(default=10, ge=2, le=100)
    """Quantile bins of the baseline for PSI and Jensen-Shannon (numeric); a baseline
    with no more distinct values than this uses one bin per value."""
    thresholds: Thresholds = Field(default_factory=Thresholds)
    quantiles: QuantileCheck | None = Field(default_factory=QuantileCheck)
    """Quantile deltas; ``null`` leaves them out."""
    test: Literal["ks", "anderson_darling"] = "ks"
    """Significance test of a numeric distribution change."""
    categorical_test: Literal["chi_square", "g_test"] = "chi_square"
    """Significance test of a categorical distribution change (homogeneity)."""
    min_values: int = Field(default=2, ge=1)
    """Fewest values on each side for a member to be measured."""
    missing_groups: Literal["fail", "ignore"] = "fail"
    """A member (group, bucket) present on one side only: a difference or left out."""
    new_categories: Literal["fail", "ignore"] = "fail"
    """Categories that appeared or disappeared: a difference, or reported only."""
    rtol: float | None = Field(default=None, ge=0)
    """Default relative tolerance of the quantile deltas; the comparison's when omitted."""
    atol: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> DistributionParams:
        names = [c.column for c in self.columns]
        if not names:
            raise ValueError("'columns' must name at least one column")
        if repeated := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"columns repeat {', '.join(repeated)}")
        if clash := sorted(set(names) & set(self.group_by)):
            raise ValueError(f"column(s) {', '.join(clash)} are also group columns")
        return self


@dataclass(frozen=True)
class _Measured:
    """One member's distribution metrics, its evidence record and its test."""

    member: DistributionMember
    record: dict[str, Any]
    material: bool
    magnitude: float
    run: Callable[[], TestResult]

    @property
    def group(self) -> dict[str, Any]:
        return dict(self.member.group)


def _float(value: Any) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _bin_labels(cutpoints: np.ndarray, unit: float) -> list[str]:
    edges = [f"{edge * unit:.4g}" for edge in cutpoints]
    labels = [f"< {edges[0]}"]
    labels += [f"{low} to {high}" for low, high in itertools.pairwise(edges)]
    return [*labels, f">= {edges[-1]}"]


def numeric_bins(
    baseline: np.ndarray, candidate: np.ndarray, bins: int, *, display_limit: int | None = None
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Shared bins of two samples and each side's counts, as STA-002 bins them.

    The baseline's quantile edges with support tails (:func:`quantile_cutpoints`
    on values divided by one power of two), or, when the baseline has no more
    distinct values than ``bins``, one bin per distinct value of either side.
    """
    distinct, counts = np.unique(baseline, return_counts=True)
    if len(distinct) <= bins:
        values = np.unique(np.concatenate([baseline, candidate]))
        b = np.searchsorted(values, baseline)
        c = np.searchsorted(values, candidate)
        size = len(values)
        return (
            [f"{v:.6g}" for v in values[:display_limit]],
            np.bincount(b, minlength=size),
            np.bincount(c, minlength=size),
        )
    unit = power_of_two(float(max(abs(distinct[0]), abs(distinct[-1]))))
    cutpoints = quantile_cutpoints(zip(distinct / unit, counts, strict=True), len(baseline), bins)
    size = len(cutpoints) + 1
    return (
        _bin_labels(cutpoints, unit),
        np.bincount(bin_index(cutpoints, baseline / unit), minlength=size),
        np.bincount(bin_index(cutpoints, candidate / unit), minlength=size),
    )


@register_level
class DistributionLevel(ComparisonLevel):
    """Distribution distances, quantile deltas and category changes per column and member."""

    name: ClassVar[str] = "distribution"
    description: ClassVar[str] = (
        "Each column's distribution per member (group, zone or time bucket) on both sides: "
        "PSI, Kolmogorov-Smirnov, Anderson-Darling, Wasserstein, Jensen-Shannon and "
        "quantile deltas for numbers; chi-square, PSI, Jensen-Shannon and categories that "
        "appeared or disappeared for categories; thresholds per metric, tests corrected "
        "across members."
    )
    Params: ClassVar[type[RuleParams]] = DistributionParams
    Result: ClassVar[type[LevelResult]] = DistributionResult

    def required_columns(self) -> frozenset[str]:
        params: DistributionParams = self.params  # type: ignore[assignment]
        columns = {*params.group_by, *(c.column for c in params.columns)}
        if params.time is not None:
            columns.add(params.time.time_column)
        return frozenset(columns)

    def evaluate(self, inputs: LevelInputs, name: str) -> DistributionResult:
        params: DistributionParams = self.params  # type: ignore[assignment]
        inference = Inference.of(inputs.spec, params)
        sides = keyed_sides(inputs, params.group_by, params.time)
        noun = "bucket" if params.time is not None else "group"
        relations = sides.relations or {side: inputs.relation(side) for side in SIDES}
        present = reduce_groups(inputs, sides.keys, (MeasureSpec(measure="count"),), relations)
        allowed = params.missing_groups == "ignore"
        checks = [one_sided_check(inputs, present, f"{noun}s", noun, allowed)]
        columns: list[ColumnDistribution] = []
        kept: list[DistributionMember] = []
        limit = inputs.ctx.evidence_limit
        for column in params.columns:
            try:
                numeric = self._kind(inputs, column) == "numeric"
                compare = self._numeric if numeric else self._categorical
                summary, found, measured = compare(
                    inputs, params, sides, relations, column.column, inference
                )
            except ValueError as exc:
                checks.append(not_assessed(column.column, str(exc).rstrip(".")))
                continue
            columns.append(summary)
            checks.extend(found)
            kept.extend(measured[: max(0, limit - len(kept))])
        return DistributionResult(
            level=self.name,
            name=name,
            checks=checks,
            group_by=list(params.group_by),
            bucket=sides.bucket,
            only_in_baseline=present.only_baseline.height,
            only_in_candidate=present.only_candidate.height,
            columns=columns,
            members=kept,
        )

    @staticmethod
    def _kind(inputs: LevelInputs, column: DistributionColumn) -> str:
        types = []
        for side in SIDES:
            schema = inputs.schema(side)
            if column.column not in schema:
                raise ValueError(f"{side} lacks column {column.column!r}")
            types.append(schema[column.column])
        numeric = all(dtype.is_numeric() for dtype in types)
        if column.kind == "numeric" and not numeric:
            raise ValueError(f"{column.column!r} is not numeric on both sides ({types})")
        if column.kind == "auto":
            return "numeric" if numeric else "categorical"
        return column.kind

    # -------------------------------------------------------------- numeric
    def _numeric(
        self,
        inputs: LevelInputs,
        params: DistributionParams,
        sides: KeyedSides,
        relations: dict[Side, ProjectedRelation],
        column: str,
        inference: Inference,
    ) -> tuple[ColumnDistribution, list[CheckResult], list[DistributionMember]]:
        rows = sum(inputs.rows(side) for side in SIDES)
        if rows > inference.max_work:
            raise ValueError(f"collecting {rows:,} values exceeds max_work {inference.max_work:,}")
        spec = MeasureSpec(measure="count", column=column, name=column)
        aligned = reduce_groups(inputs, sides.keys, (spec,), relations, [sample_extras(column)])
        roles = aligned.extras[0]
        judged = None
        if params.quantiles is not None:
            quantile = params.quantiles
            judged = materiality(
                inputs.spec,
                params,
                MeasureSpec(
                    measure="median", column=column, rtol=quantile.rtol, atol=quantile.atol
                ),
            )
        measured: list[_Measured] = []
        unmeasured: list[dict[str, Any]] = []
        keys = aligned.keys
        lossy = [aligned.columns[side][roles["lossy"]] for side in SIDES]
        values = [aligned.columns[side][roles["values"]] for side in SIDES]
        selected = precise_frame(aligned.both.select([*keys, *lossy, *values]), keys)
        for position, row in enumerate(selected.iter_rows()):
            group = dict(zip(keys, row[: len(keys)], strict=True))
            if any(row[len(keys) : len(keys) + 2]):
                reason = "values cannot be represented exactly as Float64"
                unmeasured.append({**group, "column": column, "reason": reason})
                continue
            # Sorted: every metric and test is then independent of the rows' order.
            b, c = (np.sort(np.asarray(v or [], dtype=float)) for v in row[len(keys) + 2 :])
            if min(len(b), len(c)) < params.min_values:
                reason = f"fewer than min_values {params.min_values} values on a side"
                unmeasured.append({**group, "column": column, "reason": reason})
                continue
            measured.append(
                self._numeric_member(params, column, group, b, c, judged, inference, position)
            )
        plan = MeasureTest(params.test, None)
        return self._column(inputs, params, column, "numeric", measured, unmeasured, plan)

    def _numeric_member(
        self,
        params: DistributionParams,
        column: str,
        group: dict[str, Any],
        b: np.ndarray,
        c: np.ndarray,
        judged: Materiality | None,
        inference: Inference,
        position: int,
    ) -> _Measured:
        thresholds = params.thresholds
        labels, b_counts, c_counts = numeric_bins(b, c, params.bins, display_limit=params.bins * 2)
        psi = population_stability(zip(b_counts, c_counts, strict=True), (len(b), len(c)))
        js = jensen_shannon(b_counts, c_counts)
        ks = ks_distance(b, c)
        distance = wasserstein(b, c)
        quantiles = []
        if params.quantiles is not None and judged is not None:
            for p, qb, qc in quantile_deltas(b, c, params.quantiles.probabilities):
                quantiles.append(
                    QuantileDelta(
                        p=p, baseline=qb, candidate=qc, delta=qc - qb,
                        material=judged.material(qb, qc),
                    )
                )  # fmt: skip
        material = [
            name
            for name, value, limit in (
                ("psi", psi, thresholds.max_psi),
                ("ks", ks, thresholds.max_ks),
                ("wasserstein", distance, thresholds.max_wasserstein),
                ("jensen_shannon", js, thresholds.max_js),
            )
            if limit is not None and value > limit
        ]
        if any(q.material for q in quantiles):
            material.append("quantiles")
        member = DistributionMember(
            column=column,
            kind="numeric",
            group=group,
            baseline_n=len(b),
            candidate_n=len(c),
            psi=_float(psi),
            ks=_float(ks),
            wasserstein=_float(distance),
            jensen_shannon=_float(js),
            quantiles=quantiles,
            bins=_shares(labels, b_counts, c_counts),
            material=material,
        )
        test = TESTS.create(params.test)
        data = TestInput(
            baseline=b, candidate=c, seed=inference.seed + position, max_work=inference.max_work
        )

        def run() -> TestResult:
            if inference.decision == "equivalence":
                return test.equivalent(data)
            return test.run(data)

        record = {**group, "column": column, "psi": member.psi, "ks": member.ks,
                  "wasserstein": member.wasserstein, "jensen_shannon": member.jensen_shannon,
                  "material": ", ".join(material) or "none"}  # fmt: skip
        return _Measured(member, record, bool(material), max(psi, 0.0), run)

    # ---------------------------------------------------------- categorical
    def _categorical(
        self,
        inputs: LevelInputs,
        params: DistributionParams,
        sides: KeyedSides,
        relations: dict[Side, ProjectedRelation],
        column: str,
        inference: Inference,
    ) -> tuple[ColumnDistribution, list[CheckResult], list[DistributionMember]]:
        kept = {
            side: replace(
                relations[side],
                lazy=relations[side].lazy.filter(pl.col(column).is_not_null()),  # type: ignore[union-attr]
            )
            for side in SIDES
        }
        keys = sides.keys
        counted = reduce_groups(inputs, [*keys, column], (MeasureSpec(measure="count"),), kept)
        # Membership is established before removing missing category values.
        members = reduce_groups(inputs, keys, (MeasureSpec(measure="count"),), relations)
        both = members.both.select(keys) if keys else None
        left, right = (counted.columns[side][counted.presence] for side in SIDES)
        rows = counted.joined
        if both is not None:
            rows = rows.join(both, on=keys, how="semi", nulls_equal=True)
        elif not members.both.height:
            rows = rows.clear()
        tables: dict[bytes, tuple[tuple[Any, ...], list[tuple[Any, int, int]]]] = {}
        member_keys = precise_frame(members.both.select(keys)).iter_rows() if keys else [()]
        for key in member_keys:
            tables[canonical_key(key)] = (key, [])
        for row in precise_frame(rows.select([*keys, column, left, right])).iter_rows():
            key = tuple(row[: len(keys)])
            tables.setdefault(canonical_key(key), (key, []))[1].append(
                (row[len(keys)], int(row[-2] or 0), int(row[-1] or 0))
            )
        measured: list[_Measured] = []
        unmeasured: list[dict[str, Any]] = []
        evidence = inputs.evidence()
        changed = {"appeared": 0, "disappeared": 0}
        limit = inputs.ctx.evidence_limit
        for position, (key, table) in enumerate(tables.values()):
            group = dict(zip(keys, key, strict=True))
            b_counts = np.array([t[1] for t in table])
            c_counts = np.array([t[2] for t in table])
            appeared = [t[0] for t in table if not t[1] and t[2]]
            disappeared = [t[0] for t in table if t[1] and not t[2]]
            for change, categories in (("appeared", appeared), ("disappeared", disappeared)):
                changed[change] += len(categories)
                for category in categories:
                    evidence.append({**group, column: category, "change": change})
            if min(b_counts.sum(), c_counts.sum()) < params.min_values:
                reason = f"fewer than min_values {params.min_values} values on a side"
                unmeasured.append({**group, "column": column, "reason": reason})
                continue
            measured.append(
                self._categorical_member(
                    params, column, group, table, appeared[:limit], disappeared[:limit],
                    inference, position,
                )
            )  # fmt: skip
        plan = MeasureTest(params.categorical_test, None)
        summary, results, found = self._column(
            inputs, params, column, "categorical", measured, unmeasured, plan
        )
        evidence.count = sum(changed.values())
        category_check = graded(
            f"{column}: categories",
            evidence.count,
            f"{column} has the same categories on both sides.",
            f"{column}: {_categories(changed['appeared'])} appeared and "
            f"{_categories(changed['disappeared'])} disappeared.",
            evidence=evidence,
            unit="categories",
            allowed=params.new_categories == "ignore",
            metrics=changed,
        )
        summary = summary.model_copy(update=changed)
        return summary, [*results, category_check], found

    def _categorical_member(
        self,
        params: DistributionParams,
        column: str,
        group: dict[str, Any],
        table: list[tuple[Any, int, int]],
        appeared: list[Any],
        disappeared: list[Any],
        inference: Inference,
        position: int,
    ) -> _Measured:
        thresholds = params.thresholds
        b_counts = np.array([t[1] for t in table], dtype=float)
        c_counts = np.array([t[2] for t in table], dtype=float)
        totals = (float(b_counts.sum()), float(c_counts.sum()))
        psi = population_stability(zip(b_counts, c_counts, strict=True), totals)
        js = jensen_shannon(b_counts, c_counts)
        material = [
            name
            for name, value, limit in (
                ("psi", psi, thresholds.max_psi),
                ("jensen_shannon", js, thresholds.max_js),
            )
            if limit is not None and value > limit
        ]
        order = np.argsort(-(b_counts / totals[0] + c_counts / totals[1]), kind="stable")
        labels = [str(table[i][0]) for i in order[: params.bins * 2]]
        member = DistributionMember(
            column=column,
            kind="categorical",
            group=group,
            baseline_n=int(totals[0]),
            candidate_n=int(totals[1]),
            psi=_float(psi),
            jensen_shannon=_float(js),
            bins=_shares(labels, b_counts[order], c_counts[order]),
            appeared=appeared,
            disappeared=disappeared,
            material=material,
        )
        counts = np.vstack([b_counts, c_counts])
        homogeneity = chi_square if params.categorical_test == "chi_square" else g_test
        method = params.categorical_test

        def run() -> TestResult:
            if inference.decision == "equivalence":
                return untested(method, f"{method} has no equivalence form")
            return homogeneity(counts)

        record = {**group, "column": column, "psi": member.psi,
                  "jensen_shannon": member.jensen_shannon,
                  "material": ", ".join(material) or "none"}  # fmt: skip
        return _Measured(member, record, bool(material), max(psi, 0.0), run)

    # -------------------------------------------------------------- family
    @staticmethod
    def _column(
        inputs: LevelInputs,
        params: DistributionParams,
        column: str,
        kind: str,
        measured: list[_Measured],
        unmeasured: list[dict[str, Any]],
        plan: MeasureTest,
    ) -> tuple[ColumnDistribution, list[CheckResult], list[DistributionMember]]:
        """Decide one column's family of members (one test each) and grade it."""
        inference = Inference.of(inputs.spec, params)
        noun = "bucket" if params.time is not None else "group"
        members = [
            Member(item.material, member_label(list(item.group), list(item.group.values())),
                   run=item.run)
            for item in measured
        ]  # fmt: skip
        how = "" if not inference.tests else f"; {plan.name}, {inference.describe()}"
        check = family_check(
            column,
            plan,
            inference,
            members,
            [item.record for item in measured],
            inputs.ctx.evidence_limit,
            f"{column}: no material distribution change in {plural(len(measured), noun)}{how}.",
            f"{column}: the distribution changed in {{count}} of {len(measured):,}{how}",
            noun,
            magnitudes=[item.magnitude for item in measured],
            unmeasured=unmeasured,
        )
        tests = check.significance.tests if check.significance is not None else []
        by_label = {test.label: test for test in tests}
        found = [
            item.member.model_copy(update={"test": by_label[member.label]})
            if member.label in by_label
            else item.member
            for item, member in zip(measured, members, strict=True)
        ]
        summary = ColumnDistribution(
            column=column,
            kind=kind,
            compared=len(measured) + len(unmeasured),
            material=sum(item.material for item in measured),
            significance=check.significance,
        )
        return summary, [check], found


def _shares(labels: list[str], b: np.ndarray, c: np.ndarray) -> list[BinShare]:
    totals = (float(np.sum(b)) or 1.0, float(np.sum(c)) or 1.0)
    shown = min(len(labels), len(b), len(c))
    result = [
        BinShare(label=label, baseline=float(x) / totals[0], candidate=float(y) / totals[1])
        for label, x, y in zip(labels[:shown], b[:shown], c[:shown], strict=True)
    ]
    if shown < len(b):
        result.append(
            BinShare(
                label=f"Other ({len(b) - shown:,} bins)",
                baseline=float(np.sum(b[shown:])) / totals[0],
                candidate=float(np.sum(c[shown:])) / totals[1],
            )
        )
    return result


def _categories(count: int) -> str:
    return f"{count:,} categor{'y' if count == 1 else 'ies'}"
