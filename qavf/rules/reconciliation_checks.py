"""Reconciliation: reusable contracts for common data-pipeline invariants.

These encode checks users would otherwise hand-write as SQL: totals that must agree
between tables, shares that must sum to one, a complete expected key grid, key sets
that must match in both directions, and datasets that must come from one batch. Each
states its policy for null keys, a missing side, duplicates and non-finite values,
and none passes merely because nothing was there to violate it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ClassVar, Literal

import polars as pl
from pydantic import Field, model_validator

from qavf.aggregation import exact_sum_expr
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.keys import align_keys
from qavf.numeric import exact_float_sum
from qavf.rules._contracts import contract_error
from qavf.rules._exact_grouped import exact_grouped
from qavf.rules._execution import (
    KeyBackend,
    KeyRelation,
    align_key_relations,
    key_relation,
    normalized_plan,
    observed_keys,
    paired_key_backend,
    projected_frame,
)
from qavf.rules._grouped import (
    ProjectedRelation,
    grouped_aggregate,
    normalized_relation,
    scoped_relation,
    supports_grouped,
)
from qavf.rules._key_summary import KeySelection
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import (
    fresh_internal,
    missing_expr,
    normalize_missing,
    numeric_close,
    outcome,
    plural,
    present,
    usable,
)
from qavf.settings import NumericBound, QAVFSettings
from qavf.sources.sql import quote_identifier

CATEGORY = "reconciliation"


# ======================================================================= REC-001
class Side(RuleParams):
    """One side of a reconciliation: a dataset's value summed (or counted) per key."""

    dataset: str
    value: str | None = None
    """Column to sum; omit to count rows instead."""
    keys: tuple[str, ...] = ()
    """Columns to total by; the reconciliation's ``on`` when omitted. Listed in the
    same order as ``on`` (they may have other names on this side)."""


class Reconciliation(RuleParams):
    """Totals that must agree between two datasets, key by key."""

    name: str | None = None
    left: Side
    right: Side
    on: tuple[str, ...] = ()
    """Key columns totals are compared on (empty: one grand total per side)."""
    rtol: float | None = Field(default=None, ge=0)
    """Relative tolerance. Default: exact for row counts and integer totals, 1e-9 when
    a side sums floats (to allow ordinary measurement and representation differences)."""
    atol: float = Field(default=0.0, ge=0)
    missing_side: Literal["fail", "ignore"] = "fail"
    """A key with a total on one side only: a violation (``fail``, the default) or left
    out (``ignore``, e.g. when the right side covers a subset)."""
    null_keys: Literal["fail", "exempt"] = "fail"
    """Rows with a null key component cannot be attributed to a key: by default they
    make the reconciliation fail; ``exempt`` leaves them out (counted)."""

    @model_validator(mode="after")
    def _keys(self) -> Reconciliation:
        for side in (self.left, self.right):
            if side.keys and len(side.keys) != len(self.on):
                raise ValueError(
                    f"{side.dataset}: 'keys' must list {len(self.on)} column(s), matching 'on'"
                )
        return self

    def label(self) -> str:
        return self.name or (
            f"{self.left.dataset}.{self.left.value or 'rows'} = "
            f"{self.right.dataset}.{self.right.value or 'rows'}"
        )


class ReconciliationParams(RuleParams):
    """Parameters for :class:`TotalsReconcileRule`."""

    reconciliations: tuple[Reconciliation, ...] = ()


def _declared_unit(ctx: DataContext, dataset: str, column: str | None) -> str | None:
    """The ``unit`` declared for ``dataset.column`` in its schema, if any."""
    spec = ctx.specs.get(dataset)
    schema = spec.schema_spec if spec is not None else None
    if column is None or schema is None or column not in schema.columns:
        return None
    return schema.columns[column].unit


def _comparison_sum(series: pl.Series) -> pl.Expr:
    """Retain exact float totals as canonical rational strings until comparison.

    String storage is portable across Arrow/Polars and never introduces Object
    columns. A float total may exceed Float64 range or differ below one ulp;
    rounding it before a zero-tolerance comparison would lose that distinction.
    """
    if not series.dtype.is_float():
        return exact_sum_expr(series)
    if series.is_infinite().any():
        raise ValueError("non-finite observations cannot be aggregated")

    def accumulate(values: pl.Series) -> str | None:
        result = exact_float_sum(values)
        return None if result is None else str(result)

    return pl.col(series.name).map_batches(accumulate, return_dtype=pl.String, returns_scalar=True)


def _grouped_totals(
    frame: pl.DataFrame, keys: list[str], expressions: list[pl.Expr]
) -> pl.DataFrame:
    """Normalize older Polars UDF failures into assessable arithmetic errors."""
    try:
        return (
            frame.group_by(keys, maintain_order=True).agg(expressions)
            if keys
            else frame.select(expressions)
        )
    except pl.exceptions.PanicException as exc:
        raise ValueError(f"exact aggregation could not be assessed: {exc}") from exc


def _side_totals(
    frame: pl.DataFrame,
    side: Side,
    keys: list[str],
    on: list[str],
    alias: str,
    present_name: str,
    schema: Any = None,
) -> tuple[pl.DataFrame, int]:
    """Independent presence/count and exact totals; no float coercion for counts."""
    specs = schema.columns if schema is not None else {}
    null_key = (
        pl.any_horizontal([missing_expr(frame, k, specs.get(k)) for k in keys])
        if keys
        else pl.lit(False)
    )
    null_rows = int(frame.select(null_key.sum()).item()) if keys else 0
    usable_rows = frame.filter(~null_key)
    if side.value is None:
        expr = pl.len().cast(pl.UInt64)
    else:
        dtype = frame.schema[side.value]
        if not dtype.is_numeric() and dtype != pl.Null:
            raise ValueError(f"{side.dataset}.{side.value} requires numeric values, found {dtype}")
        missing = missing_expr(usable_rows, side.value, specs.get(side.value))
        usable_rows = usable_rows.with_columns(
            pl.when(missing).then(None).otherwise(pl.col(side.value))
        )
        expr = _comparison_sum(usable_rows[side.value])
    expressions = [expr.alias(alias), pl.len().cast(pl.UInt64).alias(present_name)]
    totals = _grouped_totals(usable_rows, keys, expressions)
    if keys:
        totals = totals.rename(dict(zip(keys, on, strict=True)))
    return totals, null_rows


def _context_side_totals(
    ctx: DataContext,
    side: Side,
    keys: list[str],
    on: list[str],
    alias: str,
    present_name: str,
    observation: Observation | None = None,
) -> tuple[pl.DataFrame, int, int]:
    """Reduce exact typed totals at the source; retain exact projected float kernels."""
    schema = ctx.schema(side.dataset)
    spec = ctx.specs.get(side.dataset)
    contract = spec.schema_spec if spec is not None else None
    columns = list(dict.fromkeys([*keys, *([side.value] if side.value else [])]))
    dtype = schema[side.value] if side.value is not None else None
    provided = None
    if observation is not None:
        plan = normalized_plan(ctx, side.dataset, observation.scan())
        provided = ProjectedRelation(
            ctx,
            side.dataset,
            dict(plan.collect_schema()),
            lazy=plan,
            observation=observation,
            collector=observation.collect,
        )
    if dtype is not None and not dtype.is_numeric() and dtype != pl.Null:
        raise ValueError(f"{side.dataset}.{side.value} requires numeric values, found {dtype}")
    if dtype is not None and dtype.is_float() and side.value is not None:
        with (
            nullcontext(provided)
            if provided is not None
            else scoped_relation(ctx, side.dataset, columns)
        ) as scoped:
            assert scoped.lazy is not None and scoped.observation is not None
            missing = (
                pl.any_horizontal(pl.col(key).is_null() for key in keys) if keys else pl.lit(False)
            )
            rows, nulls = scoped.collect_plan(
                scoped.lazy.select(pl.len().alias("rows"), missing.sum().alias("nulls"))
            ).row(0)
            exact_totals = exact_grouped(
                ctx,
                scoped.observation,
                scoped.lazy.filter(~missing),
                keys,
                side.value,
                alias,
                "sum",
                rational=True,
                presence_alias=present_name,
            )
            return exact_totals.rename(dict(zip(keys, on, strict=True))), int(nulls), int(rows)
    relation = provided or normalized_relation(ctx, side.dataset, columns)
    how: Literal["count", "sum"] = "count" if side.value is None else "sum"
    if relation is not None and supports_grouped(relation, side.value, how):
        missing = (
            pl.any_horizontal(pl.col(key).is_null() for key in keys) if keys else pl.lit(False)
        )
        absent = " OR ".join(f"{quote_identifier(key)} IS NULL" for key in keys) or "FALSE"
        metadata = [pl.len().alias("rows"), missing.sum().alias("nulls")]
        sql = f"count(*), coalesce(sum(CASE WHEN {absent} THEN 1 ELSE 0 END), 0)"
        if side.value is not None:
            value = pl.col(side.value).filter(~missing)
            metadata.extend([value.min().alias("minimum"), value.max().alias("maximum")])
            selected = f"CASE WHEN NOT ({absent}) THEN {quote_identifier(side.value)} END"
            sql += f", min({selected}), max({selected})"
        summary = relation.summary(metadata, sql).row(0)
        rows, nulls = int(summary[0]), int(summary[1])
        bounds = (summary[2], summary[3], rows - nulls) if side.value is not None else None
        totals = grouped_aggregate(
            relation.filter(~missing, f"NOT ({absent})"),
            keys,
            side.value,
            alias,
            how,
            presence_alias=present_name,
            bounds=bounds,
        )
        assert totals is not None
        return totals.rename(dict(zip(keys, on, strict=True))), nulls, rows
    if not columns:
        count = int(
            ctx.sql(f"SELECT count(*) FROM {quote_identifier(side.dataset)}").item()
            if ctx.execution_kind(side.dataset) == "native_sql"
            else ctx.reduce(side.dataset, lambda frame: frame.select(pl.len())).item()
        )
        return pl.DataFrame({alias: [count], present_name: [count]}), 0, count
    frame = projected_frame(ctx, side.dataset, columns)
    totals, nulls = _side_totals(frame, side, keys, on, alias, present_name, contract)
    return totals, nulls, frame.height


class TotalsReconcileRule(Rule):
    """Totals agree between datasets, key by key (control totals, input-output balance)."""

    rule_id: ClassVar[str] = "REC-001"
    name: ClassVar[str] = "Totals reconcile"
    description: ClassVar[str] = (
        "For each configured reconciliation, the sum (or row count) of one dataset equals "
        "that of another for every key, within tolerance: input-to-output conservation, "
        "published control totals, component sums."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "A total that does not reconcile means rows were lost, duplicated or "
        "re-attributed between the two datasets. Start with the keys missing on one side, "
        "then the largest differences."
    )
    Params: ClassVar[type[RuleParams]] = ReconciliationParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset(
                {
                    side.dataset
                    for rec in self.params.reconciliations
                    for side in (rec.left, rec.right)
                }
            )
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for rec in self.params.reconciliations:
            try:
                finding = self._reconcile(ctx, rec)
            except Exception as exc:
                finding = contract_error(rec.label(), exc, (rec.left.dataset, rec.right.dataset))
            if finding is None:
                passed.append(rec.label())
            else:
                failures.append(finding)
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'reconciliation')} balance.",
            "No reconciliations are configured.",
        )

    def _reconcile(self, ctx: DataContext, rec: Reconciliation) -> Finding | None:
        label, on = rec.label(), list(rec.on)
        sides = []
        for side in (rec.left, rec.right):
            keys = list(side.keys) or on
            columns = (*keys, *([side.value] if side.value else []))
            if reason := usable(ctx, side.dataset, columns):
                return Finding.unassessed(f"Not assessed: {label}: {reason}.", check=label)
            sides.append((side, keys, ctx.schema(side.dataset)))
        units = [_declared_unit(ctx, side.dataset, side.value) for side, _, _ in sides]
        if None not in units and units[0] != units[1]:
            return Finding.unassessed(
                f"Not assessed: {label}: the totals are in different "
                f"units ({units[0]} vs {units[1]}); convert one side first.",
                check=label,
            )
        # Source-side keys can differ from ``on`` and must not collide with
        # aggregation aliases before their canonical rename.
        taken = list(dict.fromkeys([*on, *(column for _, _, schema in sides for column in schema)]))
        names = []
        for name in ("left_total", "right_total", "left_present", "right_present", "problem"):
            fresh = fresh_internal(name, taken)
            names.append(fresh)
            taken.append(fresh)
        left_name, right_name, lp, rp, problem = names
        (ls, lk, _), (rs, rk, _) = sides
        if (
            ls.dataset == rs.dataset
            and not ctx.is_materialized(ls.dataset)
            and not ctx.repeatable_scan(ls.dataset)
        ):
            projection = list(
                dict.fromkeys(
                    [
                        *lk,
                        *rk,
                        *([ls.value] if ls.value else []),
                        *([rs.value] if rs.value else []),
                    ]
                )
            )
            if not projection:
                projection = list(ctx.schema(ls.dataset))[:1]
            with ctx.observe(ls.dataset, projection) as observed:
                left, left_null, left_rows = _context_side_totals(
                    ctx, ls, lk, on, left_name, lp, observed
                )
                right, right_null, right_rows = _context_side_totals(
                    ctx, rs, rk, on, right_name, rp, observed
                )
        else:
            left, left_null, left_rows = _context_side_totals(ctx, ls, lk, on, left_name, lp)
            right, right_null, right_rows = _context_side_totals(ctx, rs, rk, on, right_name, rp)
        datasets = (ls.dataset, rs.dataset)
        null_rows = left_null + right_null
        if null_rows and rec.null_keys == "fail":
            return Finding.fail(
                f"{label}: {plural(null_rows, 'row')} have a null key and "
                "cannot be attributed (null_keys: exempt to leave them out).",
                check=label,
                affected_count=null_rows,
                unit="rows",
                metrics={"null_key_rows": {ls.dataset: left_null, rs.dataset: right_null}},
                datasets=datasets,
            )
        if on:
            left, right = align_keys(left, right, on)
            joined = left.join(right, on=on, how="full", coalesce=True, nulls_equal=True)
        else:
            joined = pl.concat([left, right], how="horizontal")
        if joined.is_empty() or (not left_rows and not right_rows):
            return Finding.unassessed(
                f"Not assessed: {label}: no rows to reconcile.", check=label, datasets=datasets
            )
        a, b = pl.col(left_name), pl.col(right_name)
        left_present, right_present = pl.col(lp).fill_null(0) > 0, pl.col(rp).fill_null(0) > 0
        both = left_present & right_present
        finite_a = a.is_finite() if joined.schema[left_name].is_float() else a.is_not_null()
        finite_b = b.is_finite() if joined.schema[right_name].is_float() else b.is_not_null()
        valid = a.is_not_null() & b.is_not_null() & finite_a & finite_b
        floats = any(
            side.value is not None and schema[side.value].is_float() for side, _, schema in sides
        )
        rtol = rec.rtol if rec.rtol is not None else (1e-9 if floats else 0.0)
        ad, bd = joined.schema[left_name], joined.schema[right_name]
        if rtol == 0 and rec.atol == 0 and ad.is_integer() and bd.is_integer():
            close = a.cast(pl.Int128) == b.cast(pl.Int128)
        elif rtol == 0 and rec.atol == 0 and ad == bd:
            close = a == b
        else:
            # Exact arithmetic is required for mixed types and nonzero tolerances
            # on integer/Decimal totals. Only aggregated groups reach this path.
            close = pl.lit(
                pl.Series(
                    [
                        numeric_close(x, y, rtol, rec.atol)
                        for x, y in zip(joined[left_name], joined[right_name], strict=True)
                    ]
                )
            )
        bad = both & (~valid | ~close.fill_null(False))
        if rec.missing_side == "fail":
            bad = bad | ~both
        count = int(joined.select(bad.sum()).item())
        assessed = int(joined.select((both & valid).sum()).item())
        if not count:
            if not assessed:
                return Finding.unassessed(
                    f"Not assessed: {label}: no comparable totals.", check=label, datasets=datasets
                )
            return None
        reason_expr = (
            pl.when(~left_present)
            .then(pl.lit(f"missing in {ls.dataset}"))
            .when(~right_present)
            .then(pl.lit(f"missing in {rs.dataset}"))
            .when(a.is_null() | b.is_null())
            .then(pl.lit("no usable values"))
            .when(~finite_a | ~finite_b)
            .then(pl.lit("non-finite total"))
            .otherwise(pl.lit("totals differ"))
        )
        sample = (
            joined.lazy()
            .with_columns(reason_expr.alias(problem))
            .filter(bad)
            .drop(lp, rp)
            .limit(ctx.evidence_limit)
            .collect()
        )
        evidence = present(
            sample,
            {
                left_name: f"{ls.dataset}_total",
                right_name: f"{rs.dataset}_total",
                problem: "problem",
            },
        )
        return Finding.fail(
            f"{label}: {plural(count, 'key')} of {joined.height:,} do not reconcile.",
            check=label,
            affected=temporal_evidence(evidence),
            affected_count=count,
            total_records=joined.height,
            unit="keys",
            metrics={"null_key_rows_exempt": null_rows} if null_rows else {},
            datasets=datasets,
        )


# ======================================================================= REC-002
class ShareSpec(RuleParams):
    """Shares (fractions, percentages) that must add up to a total within each group."""

    name: str | None = None
    dataset: str
    value: str
    group_by: tuple[str, ...] = ()
    total: NumericBound = 1.0
    """What each group's shares must sum to (1.0 for fractions, 100 for percent)."""
    atol: float = Field(default=1e-6, ge=0)
    missing: Literal["fail", "ignore"] = "fail"
    """A missing share makes its group's sum unknowable: a violation by default."""

    def label(self) -> str:
        return self.name or f"{self.dataset}.{self.value} sums to {self.total:g}"


class SharesParams(RuleParams):
    """Parameters for :class:`SharesSumRule`."""

    shares: tuple[ShareSpec, ...] = ()


class SharesSumRule(Rule):
    """Within each group, shares sum to their total."""

    rule_id: ClassVar[str] = "REC-002"
    name: ClassVar[str] = "Shares sum to total"
    description: ClassVar[str] = (
        "Within each group, the configured share column sums to its total (1 or 100) "
        "within an absolute tolerance."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Shares that do not sum to their total usually mean a missing category, a "
        "duplicated row, or shares normalised over a different set of groups."
    )
    Params: ClassVar[type[RuleParams]] = SharesParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(frozenset({spec.dataset for spec in self.params.shares}))

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for spec in self.params.shares:
            try:
                finding = self._share(ctx, spec)
            except Exception as exc:
                finding = contract_error(spec.label(), exc, (spec.dataset,))
            if finding is None:
                passed.append(spec.label())
            else:
                failures.append(finding)
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'share set')} sum correctly.",
            "No shares are configured.",
        )

    @staticmethod
    def _share(ctx: DataContext, spec: ShareSpec) -> Finding | None:
        label, keys = spec.label(), list(spec.group_by)
        if reason := usable(ctx, spec.dataset, (*keys, spec.value)):
            raise ValueError(reason)
        dtype = ctx.schema(spec.dataset)[spec.value]
        if not dtype.is_numeric() and dtype != pl.Null:
            raise ValueError(f"shares require a numeric column, found {dtype}")
        total_name = fresh_internal("share_total", keys)
        missing_name = fresh_internal("missing_shares", [*keys, total_name])
        relation = normalized_relation(ctx, spec.dataset, [*keys, spec.value])
        sums: pl.DataFrame | None
        if dtype.is_float():
            with scoped_relation(ctx, spec.dataset, [*keys, spec.value]) as scoped:
                assert scoped.lazy is not None and scoped.observation is not None
                if not scoped.count():
                    return Finding.unassessed(
                        f"Not assessed: {label}: the dataset has no rows.", check=label
                    )
                sums = exact_grouped(
                    ctx,
                    scoped.observation,
                    scoped.lazy,
                    keys,
                    spec.value,
                    total_name,
                    "sum",
                    rational=True,
                    missing_alias=missing_name,
                    nonfinite_missing=True,
                )
        elif relation is not None and supports_grouped(relation, spec.value, "sum"):
            value, quoted = pl.col(spec.value), quote_identifier(spec.value)
            rows, minimum, maximum = relation.summary(
                [pl.len(), value.min().alias("minimum"), value.max().alias("maximum")],
                f"count(*), min({quoted}), max({quoted})",
            ).row(0)
            if not rows:
                return Finding.unassessed(
                    f"Not assessed: {label}: the dataset has no rows.", check=label
                )
            sums = grouped_aggregate(
                relation,
                keys,
                spec.value,
                total_name,
                "sum",
                missing_alias=missing_name,
                bounds=(minimum, maximum, int(rows)),
            )
            assert sums is not None
        else:
            frame = normalize_missing(
                ctx, spec.dataset, projected_frame(ctx, spec.dataset, [*keys, spec.value])
            )
            if frame.is_empty():
                return Finding.unassessed(
                    f"Not assessed: {label}: the dataset has no rows.", check=label
                )
            missing = missing_expr(frame, spec.value)
            if dtype.is_float():
                missing = missing | ~pl.col(spec.value).is_finite()
            marker = fresh_internal("missing", [*frame.columns, total_name, missing_name])
            normalized = frame.with_columns(
                missing.alias(marker), pl.when(missing).then(None).otherwise(pl.col(spec.value))
            )
            agg = [
                _comparison_sum(normalized[spec.value]).alias(total_name),
                pl.col(marker).sum().alias(missing_name),
            ]
            sums = _grouped_totals(normalized, keys, agg)
        off = pl.lit(
            pl.Series(
                [not numeric_close(value, spec.total, atol=spec.atol) for value in sums[total_name]]
            )
        )
        bad = off | (pl.col(missing_name) > 0) if spec.missing == "fail" else off
        count = int(sums.select(bad.sum()).item())
        if not count:
            return None
        sample = sums.lazy().filter(bad).limit(ctx.evidence_limit).collect()
        return Finding.fail(
            f"{label}: {plural(count, 'group')} of {sums.height:,} do not sum "
            f"to {spec.total:g} (or have missing shares).",
            check=label,
            affected=temporal_evidence(
                present(sample, {total_name: "sum", missing_name: "missing_shares"})
            ),
            affected_count=count,
            total_records=sums.height,
            unit="groups",
            datasets=(spec.dataset,),
        )


# ======================================================================= REC-003
class Dimension(RuleParams):
    """The expected values of one key column: listed, or taken from a reference."""

    values: tuple[Any, ...] | None = None
    from_dataset: str | None = None
    """A reference dataset whose distinct values of the same-named column are expected."""

    @model_validator(mode="after")
    def _one(self) -> Dimension:
        if (self.values is None) == (self.from_dataset is None):
            raise ValueError("give exactly one of 'values' or 'from_dataset'")
        if self.values is not None and not self.values:
            raise ValueError("'values' must not be empty")
        return self


class GridSpec(RuleParams):
    """Every combination of the dimensions' expected values must be present."""

    name: str | None = None
    dataset: str
    dimensions: dict[str, Dimension] = Field(min_length=1)
    unexpected: bool = False
    """Also report combinations present but not expected."""
    unique: bool = False
    """Each combination must occur at most once."""

    def label(self) -> str:
        return self.name or f"{self.dataset} grid ({', '.join(self.dimensions)})"


class GridParams(RuleParams):
    """Parameters for :class:`KeyGridRule`."""

    grids: tuple[GridSpec, ...] = ()


class KeyGridRule(Rule):
    """A dataset covers the full expected grid of key combinations."""

    rule_id: ClassVar[str] = "REC-003"
    name: ClassVar[str] = "Complete key grid"
    description: ClassVar[str] = (
        "Each configured dataset contains every combination of its dimensions' expected "
        "values (e.g. county x year x pollutant), optionally no others and no repeats."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "A missing combination is silently treated as zero by most downstream sums. "
        "Re-run the producing step for the missing keys, or narrow the expected values."
    )
    Params: ClassVar[type[RuleParams]] = GridParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset(
                {
                    name
                    for grid in self.params.grids
                    for name in (
                        grid.dataset,
                        *(
                            d.from_dataset
                            for d in grid.dimensions.values()
                            if d.from_dataset is not None
                        ),
                    )
                }
            )
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for grid in self.params.grids:
            try:
                finding = self._grid(ctx, grid)
            except Exception as exc:
                finding = [contract_error(grid.label(), exc, (grid.dataset,))]
            if finding is None:
                passed.append(grid.label())
            else:
                failures.extend(finding)
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'key grid')} complete.",
            "No key grids are configured.",
        )

    def _grid(self, ctx: DataContext, grid: GridSpec) -> list[Finding] | None:
        from qavf.rules._grid import evaluate_grid

        return evaluate_grid(ctx, grid)


# ======================================================================= REC-004
class KeySide(RuleParams):
    dataset: str
    columns: tuple[str, ...] = Field(min_length=1)


class KeySetPair(RuleParams):
    """Two datasets whose key sets must be identical."""

    name: str | None = None
    left: KeySide
    right: KeySide

    @model_validator(mode="after")
    def _arity(self) -> KeySetPair:
        if len(self.left.columns) != len(self.right.columns):
            raise ValueError("left and right must list the same number of key columns")
        return self

    def label(self) -> str:
        return self.name or (
            f"{self.left.dataset}({', '.join(self.left.columns)}) = "
            f"{self.right.dataset}({', '.join(self.right.columns)})"
        )


class KeySetParams(RuleParams):
    """Parameters for :class:`KeySetsMatchRule`."""

    pairs: tuple[KeySetPair, ...] = ()


class KeySetsMatchRule(Rule):
    """Two datasets have exactly the same set of keys (both directions)."""

    rule_id: ClassVar[str] = "REC-004"
    name: ClassVar[str] = "Key sets match"
    description: ClassVar[str] = (
        "Each configured pair of datasets has identical sets of (non-null) keys: every key "
        "of one exists in the other, in both directions."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Keys on one side only mean rows were dropped or invented between the two "
        "datasets; a one-directional foreign key would not notice the other direction."
    )
    Params: ClassVar[type[RuleParams]] = KeySetParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset(
                {side.dataset for pair in self.params.pairs for side in (pair.left, pair.right)}
            )
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for pair in self.params.pairs:
            try:
                finding = self._pair(ctx, pair)
            except Exception as exc:
                finding = contract_error(pair.label(), exc, (pair.left.dataset, pair.right.dataset))
            if finding is None:
                passed.append(pair.label())
            else:
                failures.append(finding)
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'key set pair')} match.",
            "No key set pairs are configured.",
        )

    @staticmethod
    def _pair(ctx: DataContext, pair: KeySetPair) -> Finding | None:
        projections: dict[str, list[str]] = {}
        for side in (pair.left, pair.right):
            if reason := usable(ctx, side.dataset, side.columns):
                raise ValueError(reason)
            projections.setdefault(side.dataset, []).extend(side.columns)
        backend = paired_key_backend(ctx, pair.left.dataset, pair.right.dataset)
        if backend is not None and all(ctx.repeatable_scan(name) for name in projections):
            return KeySetsMatchRule._pair_relations(ctx, pair, backend)
        frequency = fresh_internal(
            "rows", [column for name in projections for column in ctx.schema(name)]
        )
        with ctx.projected_observations(projections) as observations:
            relations = []
            for side in (pair.left, pair.right):
                relation = observed_keys(
                    ctx,
                    side.dataset,
                    observations[side.dataset].scan(),
                    side.columns,
                    frequency,
                    collector=observations.collect,
                )
                missing = pl.any_horizontal(pl.col(name).is_null() for name in side.columns)
                relations.append(relation.filter("TRUE", ~missing))
            return KeySetsMatchRule._compare_relations(ctx, pair, *relations)

    @staticmethod
    def _pair_relations(ctx: DataContext, pair: KeySetPair, backend: KeyBackend) -> Finding | None:
        relations = []
        frequency = fresh_internal(
            "rows",
            [*ctx.schema(pair.left.dataset), *ctx.schema(pair.right.dataset)],
        )
        for side in (pair.left, pair.right):
            if reason := usable(ctx, side.dataset, side.columns):
                raise ValueError(reason)
            relation = key_relation(
                ctx,
                side.dataset,
                side.columns,
                frequency,
                backend=backend,
            )
            missing = pl.any_horizontal(pl.col(name).is_null() for name in side.columns)
            sql = " OR ".join(f"{quote_identifier(name)} IS NULL" for name in side.columns)
            relations.append(relation.filter(f"NOT ({sql})", ~missing))
        return KeySetsMatchRule._compare_relations(ctx, pair, *relations)

    @staticmethod
    def _compare_relations(
        ctx: DataContext, pair: KeySetPair, left: KeyRelation, right: KeyRelation
    ) -> Finding | None:
        right = right.rename(dict(zip(pair.right.columns, pair.left.columns, strict=True)))
        left, right = align_key_relations(left, right)
        if not left.summarize(0).keys and not right.summarize(0).keys:
            return Finding.unassessed(
                f"Not assessed: {pair.label()}: no usable keys.",
                check=pair.label(),
            )
        only_left = left.anti(right).summarize(ctx.evidence_limit)
        only_right = right.anti(left).summarize(ctx.evidence_limit)
        count = only_left.keys + only_right.keys
        if not count:
            return None
        names = list(pair.left.columns)
        side_name = fresh_internal("side", names)
        samples = [
            result.sample.select(names).with_columns(pl.lit(f"only in {dataset}").alias(side_name))
            for result, dataset in (
                (only_left, pair.left.dataset),
                (only_right, pair.right.dataset),
            )
        ]
        sample = pl.concat(samples).head(ctx.evidence_limit)
        return Finding.fail(
            f"{pair.label()}: {plural(count, 'key')} on one side only.",
            check=pair.label(),
            affected=temporal_evidence(present(sample, {side_name: "side"})),
            affected_count=count,
            unit="keys",
            datasets=(pair.left.dataset, pair.right.dataset),
        )


# ======================================================================= REC-005
class BatchSpec(RuleParams):
    """Datasets that must all come from one and the same batch (run id, watermark)."""

    name: str | None = None
    datasets: tuple[str, ...] = Field(min_length=2)
    column: str
    """The batch identifier column each dataset carries (e.g. ``run_id``, ``as_of``)."""

    def label(self) -> str:
        return self.name or f"{', '.join(self.datasets)} share one {self.column}"


class BatchParams(RuleParams):
    """Parameters for :class:`ConsistentBatchRule`."""

    batches: tuple[BatchSpec, ...] = ()


class ConsistentBatchRule(Rule):
    """Related datasets carry one, identical batch identifier."""

    rule_id: ClassVar[str] = "REC-005"
    name: ClassVar[str] = "Consistent batch"
    description: ClassVar[str] = (
        "Each configured group of datasets carries a single, identical value in its batch "
        "column (run id, export id, watermark), so they come from one consistent extract."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Datasets from different batches were read while an upstream job was updating "
        "them. Re-extract them together (or use a connection snapshot) before comparing."
    )
    Params: ClassVar[type[RuleParams]] = BatchParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(
            frozenset({name for batch in self.params.batches for name in batch.datasets})
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for batch in self.params.batches:
            label = batch.label()
            values: list[pl.DataFrame] = []
            evidence: list[dict[str, Any]] = []
            valid = True
            try:
                for name in batch.datasets:
                    if reason := usable(ctx, name, (batch.column,)):
                        raise ValueError(reason)
                    from qavf.rules._execution import key_relation

                    frequency = fresh_internal("rows", ctx.schema(name))
                    relation = key_relation(ctx, name, [batch.column], frequency)
                    missing_expr_key = pl.col(batch.column).is_null()
                    quoted = quote_identifier(batch.column)
                    summaries = relation.summarize_many(
                        {
                            "all": KeySelection(),
                            "missing": KeySelection(f"{quoted} IS NULL", missing_expr_key),
                            "identifiers": KeySelection(
                                f"{quoted} IS NOT NULL",
                                ~missing_expr_key,
                                max(1, ctx.evidence_limit),
                            ),
                        }
                    )
                    all_keys = summaries["all"]
                    missing = summaries["missing"].rows
                    identifiers = summaries["identifiers"]
                    evidence.append(
                        {
                            "dataset": name,
                            "distinct_identifiers": identifiers.keys,
                            "missing_rows": missing,
                            "values": temporal_evidence(
                                identifiers.sample.select(batch.column).head(ctx.evidence_limit)
                            )[batch.column].to_list(),
                        }
                    )
                    valid &= bool(all_keys.rows) and identifiers.keys == 1 and not missing
                    values.append(identifiers.sample.select(batch.column))
                if valid:
                    for candidate in values[1:]:
                        first, other = align_keys(values[0], candidate, [batch.column])
                        if not first.equals(other, null_equal=True):
                            valid = False
                if valid:
                    passed.append(label)
                else:
                    failures.append(
                        Finding.fail(
                            f"{label}: every dataset must contain one valid, "
                            f"identical {batch.column}.",
                            check=label,
                            affected=evidence[: ctx.evidence_limit],
                            affected_count=len(evidence),
                            unit="datasets",
                            datasets=batch.datasets,
                        )
                    )
            except Exception as exc:
                failures.append(contract_error(label, exc, batch.datasets))
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'batch group')} consistent.",
            "No batch groups are configured.",
        )
