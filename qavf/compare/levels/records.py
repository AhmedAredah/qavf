"""Records: keyed match of rows, change counts, magnitudes and transitions.

Row classification and the changed-value decision are the regression engine's
exact keyed comparison (``qavf.regression``): the same key policy, tolerance
arithmetic and per-column counts as REG-001. Additional descriptions (null/NaN
transitions, difference magnitudes, categorical transitions and the largest
changes) are native reductions over the matched pairs; only bounded results
reach Python.

Value changes are facts: every value beyond tolerance is counted, whatever the
decision policy. Unless the policy is ``material``, a ``shift`` check also asks
whether each numeric column moved systematically: a paired test (``test``,
``paired_t`` by default) of ``candidate - baseline`` over the matched rows, per
column and optionally per ``shift_by`` group, with the mean difference judged
for materiality by the column's tolerance and the family of columns (and
groups) corrected together.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, ClassVar

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from qavf._float_accumulator import ExactFloatAccumulator
from qavf.compare.comparators import Materiality
from qavf.compare.levels import ComparisonLevel, LevelInputs, graded, not_assessed, register_level
from qavf.compare.levels.groups import group_partitions, precise_frame
from qavf.compare.results import CheckResult, ColumnChanges, LevelResult, RecordsResult, Transition
from qavf.compare.significance import (
    Inference,
    Judgement,
    MeasureTest,
    Member,
    evidence_fields,
    float_values,
    judge,
    summary_metrics,
)
from qavf.compare.spec import InferenceParams, TestConfig
from qavf.engine.registry import RuleParams
from qavf.inference.registry import TestInput
from qavf.inference.results import Significance
from qavf.keys import key_casts
from qavf.regression._indexed import ProjectedInput, compare_indexed, supported
from qavf.regression._numeric import _difference_size, _settled, numbers_close
from qavf.regression._plan import ComparisonPlan, comparison_plan
from qavf.regression._schema import _nested
from qavf.regression.compare_outputs import _compare_projected
from qavf.regression.models import ComparisonResult, SchemaPolicy
from qavf.rules.helpers import BoundedEvidence, LargestEvidence, fresh_internal, plural
from qavf.typed_values import encode_temporal


class RecordsParams(InferenceParams):
    """Parameters of the ``records`` level."""

    allow_added: bool = False
    """Rows only in the candidate are acceptable."""
    allow_removed: bool = False
    """Rows only in the baseline are acceptable."""
    schema_policy: SchemaPolicy = Field(default="compatible", alias="schema")
    """``schema`` in configuration: columns whose types differ in kind are not
    value-compared (the structure level reports them)."""
    top_n: int = Field(default=10, ge=0)
    """Largest numeric changes kept per column as evidence."""
    max_transitions: int = Field(default=20, ge=0)
    """Most frequent value transitions kept per categorical column."""
    categorical: tuple[str, ...] | None = None
    """Columns whose value transitions are counted; text, categorical and boolean
    columns when omitted."""
    test: TestConfig = Field(default_factory=lambda: TestConfig(test="paired_t"))
    """Paired test of each numeric column's shift (``paired_t``, ``wilcoxon``,
    ``sign``, ``paired_permutation``, ``paired_bootstrap``)."""
    shift_by: tuple[str, ...] = ()
    """Also test the shift within groups of these key or compared columns
    (candidate values)."""

    @model_validator(mode="after")
    def _paired(self) -> RecordsParams:
        test = self.test.create()
        if test is not None and test.needs != "paired":
            raise ValueError(f"the records level needs a paired test, not {test.name!r}")
        return self


def _categorical(dtype: pl.DataType) -> bool:
    return dtype in (pl.String, pl.Boolean) or isinstance(dtype, pl.Categorical | pl.Enum)


def _difference(b: pl.Expr, c: pl.Expr, db: pl.DataType, dc: pl.DataType) -> pl.Expr:
    """Exact ``|c - b|`` for integers and decimals; binary64 when a float is involved."""
    if db == pl.Int128 or dc == pl.Int128:
        # The difference of two signed 128-bit values can need 129 bits. Compute
        # it with Python's unbounded integers before rounding the display size.
        return pl.struct(b.alias("before"), c.alias("after")).map_elements(
            lambda pair: _difference_size(pair["before"], pair["after"])[0],
            return_dtype=pl.Float64,
        )
    if db.is_integer() and dc.is_integer():
        return (c.cast(pl.Int128) - b.cast(pl.Int128)).abs()
    if not (db.is_float() or dc.is_float()):
        scale = max(
            (dtype.scale or 0) if isinstance(dtype, pl.Decimal) else 0 for dtype in (db, dc)
        )
        return (c.cast(pl.Decimal(38, scale)) - b.cast(pl.Decimal(38, scale))).abs()
    return (c.cast(pl.Float64) - b.cast(pl.Float64)).abs()


def _finite(b: pl.Expr, c: pl.Expr, db: pl.DataType, dc: pl.DataType) -> pl.Expr:
    """Both values present and, for floats, finite."""
    finite = b.is_not_null() & c.is_not_null()
    for expr, dtype in ((b, db), (c, dc)):
        if dtype.is_float():
            finite = finite & expr.is_finite()
    return finite


def _keyed(inputs: LevelInputs, plan: ComparisonPlan) -> ComparisonResult:
    baseline, candidate = inputs.observations["baseline"], inputs.observations["candidate"]
    if supported(plan):
        with inputs.ctx.work_store("comparison record alignment") as store:
            return compare_indexed(
                store,
                ProjectedInput(baseline, plan.shared),
                ProjectedInput(candidate, plan.shared),
                plan,
            )
    columns = list(plan.shared)
    return _compare_projected(
        plan,
        inputs.collect(inputs.scan("baseline").select(columns)),
        inputs.collect(inputs.scan("candidate").select(columns)),
        execution="projected_fallback",
        fallback_reason="opaque Object columns need an eager comparison",
    )


class _Matched:
    """Matched pairs as one native join, with collision-free side column names."""

    def __init__(self, inputs: LevelInputs, plan: ComparisonPlan, columns: list[str]) -> None:
        keys = list(plan.keys)
        left, right = plan.baseline_schema, plan.candidate_schema
        taken = [*left, *right]
        self.names: dict[str, tuple[str, str]] = {}
        for column in columns:
            before = fresh_internal(f"baseline_{column}", taken)
            taken.append(before)
            after = fresh_internal(f"candidate_{column}", taken)
            taken.append(after)
            self.names[column] = (before, after)
        casts: tuple[dict[str, pl.DataType], dict[str, pl.DataType]] = ({}, {})
        for key in keys:
            # The keyed comparison has already proven any cross-type alignment exact.
            to_left, to_right = key_casts(key, left[key], right[key], exact_float=lambda: True)
            if to_left is not None:
                casts[0][key] = to_left
            if to_right is not None:
                casts[1][key] = to_right
        sides = []
        for index, side in enumerate(("baseline", "candidate")):
            sides.append(
                inputs.scan(side)  # type: ignore[arg-type]
                .select([*keys, *columns])
                .with_columns(pl.col(key).cast(dtype) for key, dtype in casts[index].items())
                .rename({column: self.names[column][index] for column in columns})
            )
        self.plan = sides[0].join(sides[1], on=keys, how="inner", nulls_equal=True)
        self.keys = keys

    def pair(self, column: str) -> tuple[pl.Expr, pl.Expr]:
        b, c = self.names[column]
        return pl.col(b), pl.col(c)


def _describe(
    inputs: LevelInputs,
    plan: ComparisonPlan,
    params: RecordsParams,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Per-column transition counts, magnitudes, categorical transitions and top changes."""
    left, right = plan.baseline_schema, plan.candidate_schema
    columns = [c for c in plan.values if not (_nested(left[c]) or _nested(right[c]))]
    stats: dict[str, dict[str, Any]] = {column: {} for column in plan.values}
    if not columns:
        return stats, []
    matched = _Matched(inputs, plan, columns)
    top_n = min(params.top_n, inputs.ctx.evidence_limit) if inputs.ctx.evidence_bytes else 0
    consumers = 1 + sum(
        bool(top_n)
        if left[c].is_numeric() and right[c].is_numeric()
        else (
            c in params.categorical
            if params.categorical is not None
            else _categorical(left[c]) and _categorical(right[c])
        )
        for c in columns
    )
    if consumers > 1:
        # One owned join is reused only when downstream descriptions read it repeatedly.
        with inputs.observations["baseline"].derived(lambda _: matched.plan) as observation:
            matched.plan = observation.scan()
            return _describe_matched(inputs, plan, params, columns, stats, matched)
    return _describe_matched(inputs, plan, params, columns, stats, matched)


def _describe_matched(
    inputs: LevelInputs,
    plan: ComparisonPlan,
    params: RecordsParams,
    columns: list[str],
    stats: dict[str, dict[str, Any]],
    matched: _Matched,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    left, right = plan.baseline_schema, plan.candidate_schema
    reductions: list[pl.Expr] = []
    numeric = [c for c in columns if left[c].is_numeric() and right[c].is_numeric()]
    for index, column in enumerate(columns):
        b, c = matched.pair(column)
        reductions += [
            (b.is_null() & c.is_not_null()).sum().alias(f"n2v{index}"),
            (b.is_not_null() & c.is_null()).sum().alias(f"v2n{index}"),
        ]
        if left[column].is_float() or right[column].is_float():
            bn = b.is_nan().fill_null(False) if left[column].is_float() else pl.lit(False)
            cn = c.is_nan().fill_null(False) if right[column].is_float() else pl.lit(False)
            reductions += [
                (bn & ~cn & c.is_not_null()).sum().alias(f"nan2v{index}"),
                (~bn & b.is_not_null() & cn).sum().alias(f"v2nan{index}"),
            ]
        if column in numeric:
            finite = _finite(b, c, left[column], right[column])
            gap = _difference(b, c, left[column], right[column])
            differing = finite & (gap != 0)
            relative = gap.cast(pl.Float64) / b.cast(pl.Float64).abs()
            reductions += [
                gap.filter(differing).max().alias(f"abs{index}"),
                relative.filter(differing & (b != 0)).max().alias(f"rel{index}"),
            ]
    row = inputs.collect(matched.plan.select(reductions)).row(0, named=True)
    for index, column in enumerate(columns):
        stats[column] = {
            "null_to_value": int(row[f"n2v{index}"] or 0),
            "value_to_null": int(row[f"v2n{index}"] or 0),
            "nan_to_value": int(row.get(f"nan2v{index}") or 0),
            "value_to_nan": int(row.get(f"v2nan{index}") or 0),
        }
        if column in numeric:
            largest = row[f"abs{index}"]
            stats[column]["max_abs_diff"] = None if largest is None else float(largest)
            stats[column]["max_rel_diff"] = row[f"rel{index}"]
    selected = params.categorical
    categorical = [
        c
        for c in columns
        if (
            c in selected
            if selected is not None
            else _categorical(left[c]) and _categorical(right[c])
        )
    ]
    if selected is not None and (unknown := sorted(set(selected) - set(columns))):
        raise ValueError(f"categorical names column(s) not compared: {', '.join(unknown)}")
    transition_bytes = inputs.ctx.evidence_bytes
    for column in categorical:
        b, c = matched.pair(column)
        names = matched.names[column]
        count = fresh_internal("transitions", [*names, *matched.keys])
        transitions = (
            matched.plan.filter(b.ne_missing(c))
            .group_by(list(names))
            .agg(pl.len().cast(pl.UInt64).alias(count))
        )
        stats[column]["distinct_transitions"] = int(
            inputs.collect(transitions.select(pl.len())).item()
        )
        transition_limit = min(params.max_transitions, inputs.ctx.evidence_limit)
        if not transition_limit or not transition_bytes:
            stats[column]["transitions"] = []
            continue
        top = inputs.collect(
            transitions.sort(
                [count, *names], descending=[True, False, False], nulls_last=True
            ).head(transition_limit)
        )
        top = encode_temporal(top)
        evidence = BoundedEvidence(transition_limit, byte_limit=transition_bytes)
        evidence.extend({"baseline": x, "candidate": y, "count": n} for x, y, n in top.iter_rows())
        transition_bytes -= evidence.bytes_used
        stats[column]["transitions"] = [Transition(**record) for record in evidence.records]
    changes: list[dict[str, Any]] = []
    top_n = min(params.top_n, inputs.ctx.evidence_limit) if inputs.ctx.evidence_bytes else 0
    if top_n:
        for column in numeric:
            b, c = matched.pair(column)
            rtol, atol = plan.tolerance(column)
            pending = ~_settled(b, c, left[column], right[column], rtol, atol)
            gap = _difference(b, c, left[column], right[column])
            magnitude = fresh_internal("magnitude", [*plan.baseline_schema, *matched.names[column]])
            finite = _finite(b, c, left[column], right[column])
            candidates = inputs.collect(
                matched.plan.filter(pending.fill_null(True) & finite)
                .with_columns(gap.alias(magnitude))
                .sort([magnitude, *matched.keys], descending=[True, *[False] * len(matched.keys)])
                .head(4 * top_n + 16)
                .select([*matched.keys, *matched.names[column]])
            )
            kept = 0
            for record in encode_temporal(candidates, matched.keys).iter_rows(named=True):
                before, after = (
                    record.pop(matched.names[column][0]),
                    record.pop(matched.names[column][1]),
                )
                if numbers_close(before, after, rtol, atol):
                    continue
                absolute_size, relative_size = _difference_size(before, after)
                changes.append(
                    {
                        "column": column,
                        **record,
                        "baseline": before,
                        "candidate": after,
                        "abs_diff": absolute_size,
                        "rel_diff": relative_size,
                    }
                )
                kept += 1
                if kept == top_n:
                    break
    return stats, changes


@dataclass(frozen=True)
class _Shift:
    check: CheckResult
    family: Significance
    columns: dict[str, Significance]
    shifted: list[str]


def _shift(
    inputs: LevelInputs,
    plan: ComparisonPlan,
    params: RecordsParams,
    inference: Inference,
    rows: int,
) -> _Shift | None:
    """Paired tests of each numeric column's shift over matched rows (per ``shift_by``)."""
    left, right = plan.baseline_schema, plan.candidate_schema
    numeric = [
        c
        for c in plan.values
        if left[c].is_numeric() and right[c].is_numeric() and c not in params.shift_by
    ]
    if not numeric or rows == 0:
        return None
    test = params.test.create()
    tested = MeasureTest("none", None, False, "the level declares no test (test: none)")
    if test is not None:
        tested = MeasureTest(test.name, test, statistic="mean")
    groups = list(params.shift_by)
    if unknown := sorted(set(groups) - set(plan.keys) - set(plan.values)):
        raise ValueError(f"shift_by names column(s) not compared: {', '.join(unknown)}")
    matched = _Matched(inputs, plan, [*numeric, *(g for g in groups if g not in plan.keys)])
    if len(numeric) > 1:
        # Reuse the join across column tests under the observation's memory/spill
        # policy. One-column tests keep their narrow direct plan.
        with inputs.observations["baseline"].derived(lambda _: matched.plan) as observation:
            matched.plan = observation.scan()
            return _shift_matched(inputs, plan, inference, numeric, tested, groups, matched, rows)
    return _shift_matched(inputs, plan, inference, numeric, tested, groups, matched, rows)


def _shift_matched(
    inputs: LevelInputs,
    plan: ComparisonPlan,
    inference: Inference,
    numeric: list[str],
    tested: MeasureTest,
    groups: list[str],
    matched: _Matched,
    rows: int,
) -> _Shift:
    members: list[Member] = []
    described: list[dict[str, Any]] = []
    for column in numeric:
        b, c = matched.pair(column)
        keys = [pl.col(g) if g in plan.keys else matched.pair(g)[1] for g in groups]
        names = [fresh_internal(f"group_{i}", matched.names[column]) for i in range(len(groups))]
        rtol, atol = plan.tolerance(column)
        judged = Materiality(rtol, atol)
        for group, pairs, means, problem in _paired_values(
            inputs, matched.plan, keys, names, groups, (b, c), rows, inference
        ):
            usable = means[0] is not None and means[1] is not None
            # An unknown mean is not proof of immateriality. Force the policy to
            # inspect the missing input and report it as unassessed.
            material = judged.material(*means) if usable else True
            if not usable and problem is None:
                problem = "no finite matched pairs to measure the shift"
            margin = judged.margin(*means) if usable else None
            data = None
            if pairs is not None:
                data = TestInput(
                    baseline=pairs[0],
                    candidate=pairs[1],
                    seed=inference.seed + len(members),
                    max_work=inference.max_work,
                    margin=None if margin is None else (-margin, margin),
                )
            label = ", ".join([column, *(f"{k}={v}" for k, v in group.items())])
            members.append(Member(bool(material), label, data, problem))
            count = len(pairs[0]) if pairs is not None else None
            described.append(
                {"column": column, **group, "pairs": count,
                 "baseline_mean": None if means[0] is None else float(means[0]),
                 "candidate_mean": None if means[1] is None else float(means[1])}
            )  # fmt: skip
    judgement = judge(tested, inference, members, inputs.ctx.evidence_limit)
    return _shift_check(inputs, numeric, tested, inference, judgement, described)


type _Paired = tuple[
    dict[str, Any], tuple[np.ndarray, np.ndarray] | None, tuple[Any, Any], str | None
]
"""A group, its matched values (``None``: not collected), their means and any problem."""


def _paired_values(
    inputs: LevelInputs,
    plan: pl.LazyFrame,
    keys: list[pl.Expr],
    names: list[str],
    groups: list[str],
    pair: tuple[pl.Expr, pl.Expr],
    rows: int,
    inference: Inference,
) -> list[_Paired]:
    """Per group: the matched values (when within ``max_work``), their means and any problem.

    Beyond the budget only group identities are collected. A rounded native mean
    cannot establish exact materiality, so those groups remain unassessed.
    """
    b, c = pair
    grouping = [k.alias(n) for k, n in zip(keys, names, strict=True)]
    if rows * 2 > inference.max_work:
        problem = f"{rows:,} matched rows exceed max_work {inference.max_work:,}"
        group_rows = (
            precise_frame(
                inputs.collect(plan.select(grouping).unique().sort(names, nulls_last=True))
            ).iter_rows()
            if names
            else [()]
        )
        return [
            (dict(zip(groups, row, strict=True)), None, (None, None), problem) for row in group_rows
        ]
    frame = inputs.collect(plan.select(*grouping, b.alias("b"), c.alias("c")))
    ordered = frame.sort(names, nulls_last=True) if names else frame
    found: list[_Paired] = []
    for key, part in group_partitions(ordered, names):
        group = dict(zip(groups, key, strict=True))
        part = part.filter(pl.all_horizontal(pl.col("b", "c").cast(pl.Float64).is_finite()))
        average = tuple(_exact_mean(part[column]) for column in ("b", "c"))
        try:
            values = float_values(part["b"]), float_values(part["c"])
        except ValueError as exc:
            found.append((group, None, (average[0], average[1]), str(exc)))
            continue
        found.append((group, values, (average[0], average[1]), None))
    return found


def _exact_mean(values: pl.Series) -> Fraction | None:
    """Materiality of bounded paired samples must not depend on rounded sums."""
    if not len(values):
        return None
    if values.dtype.is_float():
        accumulator = ExactFloatAccumulator()
        accumulator.update(values)
        return accumulator.mean()
    total = sum(values) if values.dtype.is_integer() else sum(map(Fraction, values), Fraction())
    return Fraction(total) / len(values)


def _shift_check(
    inputs: LevelInputs,
    numeric: list[str],
    tested: MeasureTest,
    inference: Inference,
    judgement: Judgement,
    described: list[dict[str, Any]],
) -> _Shift:
    failed, unassessed = judgement.failed, judgement.unassessed
    leaders = LargestEvidence(inputs.ctx.evidence_limit)
    for position in failed or unassessed:
        record = dict(described[position])
        b, c = record["baseline_mean"], record["candidate_mean"]
        record |= evidence_fields(judgement.results[position], judgement.decisions[position])
        magnitude = abs(c - b) if b is not None and c is not None else 0.0
        leaders.add(magnitude, position, record)
    evidence = inputs.evidence()
    evidence.extend(leaders.records)
    evidence.count = len(failed or unassessed)
    family = judgement.significance
    columns: dict[str, Significance] = {}
    for column in numeric:
        mine = [i for i, record in enumerate(described) if record["column"] == column]
        tests = [judgement.results[i] for i in mine]
        kept = [result for result in tests if result is not None]
        columns[column] = family.model_copy(
            update={
                "significant": sum(bool(result.significant) for result in kept),
                "failed": sum(judgement.decisions[i].failed for i in mine),
                "not_assessed": sum(not judgement.decisions[i].assessed for i in mine),
                "tests": kept[: inputs.ctx.evidence_limit],
            }
        )
    shifted = sorted({described[i]["column"] for i in failed}, key=numeric.index)
    members = len(described)
    metrics = summary_metrics(family)
    unit = "columns" if members == len(numeric) else "column groups"
    if failed or not unassessed:
        how = f"{tested.name}; {inference.describe()}"
        check = graded(
            "shift",
            len(failed),
            f"No column shifts systematically ({plural(members, unit[:-1])}; {how}).",
            f"{plural(len(shifted), 'column')} shift systematically: {', '.join(shifted)} ({how})."
            + (f" {len(unassessed):,} not assessed." if unassessed else ""),
            evidence=evidence,
            total=members,
            unit=unit,
            metrics=metrics,
        )
    else:
        reason = judgement.decisions[unassessed[0]].reason
        check = CheckResult(
            check="shift",
            passed=False,
            assessed=False,
            message=f"Not assessed: shift: {plural(len(unassessed), unit[:-1])} need a "
            f"{tested.name} test that could not run ({reason}).",
            affected_count=len(unassessed),
            total=members,
            unit=unit,
            affected_records=evidence.records,
            truncated=evidence.truncated,
            metrics=metrics,
        )
    check = check.model_copy(update={"significance": family})
    return _Shift(check, family, columns, shifted)


@register_level
class RecordsLevel(ComparisonLevel):
    """Rows added, removed, changed and unchanged, with per-column change detail."""

    name: ClassVar[str] = "records"
    description: ClassVar[str] = (
        "Keyed match of rows: added, removed, changed and unchanged rows; per-column "
        "change counts, rates and magnitudes; null/NaN and categorical transitions; the "
        "largest changes as evidence."
    )
    Params: ClassVar[type[RuleParams]] = RecordsParams
    Result: ClassVar[type[LevelResult]] = RecordsResult
    requires_keys: ClassVar[bool] = True

    def evaluate(self, inputs: LevelInputs, name: str) -> RecordsResult:
        params: RecordsParams = self.params  # type: ignore[assignment]
        spec = inputs.spec
        left, right = inputs.schema("baseline"), inputs.schema("candidate")
        present = set(left) | set(right)
        plan = comparison_plan(
            left,
            right,
            spec.keys,
            rtol=spec.rtol,
            atol=spec.atol,
            ignore_columns=[c for c in spec.ignore_columns if c in present],
            max_differences=inputs.ctx.evidence_limit,
            max_evidence_bytes=inputs.ctx.evidence_bytes,
            name=name,
            column_tolerances={c: spec.tolerance(c) for c in spec.tolerances if c in present},
            schema=params.schema_policy,
        )
        result = _keyed(inputs, plan)
        stats, changes = _describe(inputs, plan, params)
        matched = result.baseline_rows - result.rows_removed
        columns = [
            ColumnChanges(
                column=column,
                changed=result.column_changes.get(column, 0),
                rate=result.column_changes.get(column, 0) / matched if matched else None,
                **stats[column],
            )
            for column in plan.values
        ]
        rows = inputs.evidence()
        values = inputs.evidence()
        values.extend(changes)
        numeric = {record["column"] for record in changes}
        for difference in result.differences:
            record = {"kind": difference.kind, **difference.key}
            if difference.kind in ("row_added", "row_removed"):
                rows.append(record)
            elif difference.kind == "value_changed" and difference.column not in numeric:
                values.append(
                    {
                        "column": difference.column,
                        **difference.key,
                        "baseline": difference.baseline,
                        "candidate": difference.candidate,
                    }
                )
        # Counts are complete even where evidence above was capped.
        rows.count = result.rows_added + result.rows_removed
        values.count = result.values_changed
        one_sided = (0 if params.allow_removed else result.rows_removed) + (
            0 if params.allow_added else result.rows_added
        )
        not_compared = [c for c in plan.dtypes_changed if c not in plan.values]
        checks = [
            graded(
                "rows",
                one_sided,
                f"Every row matches by key ({matched:,} rows).",
                f"{plural(result.rows_removed, 'row')} removed, "
                f"{plural(result.rows_added, 'row')} added.",
                evidence=rows,
                total=matched + result.rows_added + result.rows_removed,
                unit="rows",
            ),
            graded(
                "values",
                result.values_changed,
                f"Matched rows agree within tolerance ({len(plan.values)} column(s)).",
                f"{plural(result.values_changed, 'value')} changed in "
                f"{plural(result.rows_changed, 'row')}: "
                + ", ".join(
                    f"{column} ({count:,})" for column, count in result.column_changes.items()
                ),
                evidence=values,
                total=matched * len(plan.values),
                unit="values",
                metrics={
                    "column_changes": dict(result.column_changes),
                    "not_compared": not_compared,
                    "max_abs_diff": result.max_abs_diff,
                    "max_rel_diff": result.max_rel_diff,
                    "nonfinite_changes": result.nonfinite_changes,
                },
            ),
        ]
        if not_compared and not plan.values:
            checks[1] = not_assessed(
                "values",
                "no value column can be compared; incompatible types: " + ", ".join(not_compared),
            ).model_copy(
                update={"metrics": {"not_compared": not_compared}, "total": 0, "unit": "values"}
            )
        inference = Inference.of(spec, params)
        shift = _shift(inputs, plan, params, inference, matched) if inference.tests else None
        if shift is not None:
            checks.append(shift.check)
            columns = [
                column.model_copy(update={"significance": shift.columns.get(column.column)})
                for column in columns
            ]
        return RecordsResult(
            level=self.name,
            name=name,
            checks=checks,
            significance=None if shift is None else shift.family,
            shifted=[] if shift is None else shift.shifted,
            keys=list(plan.keys),
            matched=matched,
            rows_added=result.rows_added,
            rows_removed=result.rows_removed,
            rows_changed=result.rows_changed,
            rows_unchanged=matched - result.rows_changed,
            values_changed=result.values_changed,
            columns=columns,
            execution=result.execution,
        )
