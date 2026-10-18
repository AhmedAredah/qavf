"""Exact bounded paired batches shared by eager and streamed regression paths."""

from __future__ import annotations

import logging

import polars as pl

from qavf.regression._cells import _cell_change
from qavf.regression._evidence import _capture_cells, _capture_native, _Changes
from qavf.regression._numeric import _difference_size, _settled
from qavf.regression._plan import ComparisonPlan
from qavf.regression._schema import _encode_temporal, _nested, _structure_changed, _type_changed
from qavf.regression.models import ComparisonExecution, ComparisonResult

logger = logging.getLogger(__name__)
_BATCH_ROWS = 16_384
_TOLERANCE_PREFILTER = True


def _fresh_name(occupied: set[str], stem: str) -> str:
    name = stem
    while name in occupied:
        name += "_"
    occupied.add(name)
    return name


class ComparisonAccumulator:
    """Complete counts and bounded evidence; inputs are never retained between batches."""

    def __init__(self, plan: ComparisonPlan, baseline_rows: int, candidate_rows: int) -> None:
        self.plan = plan
        self.state = _Changes(plan.max_differences, plan.max_evidence_bytes)
        self.baseline_rows, self.candidate_rows = baseline_rows, candidate_rows
        self.rows_added = self.rows_removed = 0
        self.structural_changes = {
            column
            for column in plan.dtypes_changed
            if _structure_changed(plan.baseline_schema[column], plan.candidate_schema[column])
            and not _type_changed(
                plan.baseline_schema[column], plan.candidate_schema[column], "compatible"
            )
        }
        for kind, columns in (
            ("column_removed", plan.columns_removed),
            ("column_added", plan.columns_added),
        ):
            for column in columns:
                self.state.record(kind=kind, column=column)
        for column in plan.dtypes_changed:
            if column not in self.structural_changes:
                self._dtype_change(column)

    def _dtype_change(self, column: str) -> None:
        self.state.record(
            kind="dtype_changed",
            column=column,
            baseline=str(self.plan.baseline_schema[column]),
            candidate=str(self.plan.candidate_schema[column]),
        )

    def unmatched(self, kind: str, frame: pl.DataFrame) -> None:
        if kind == "row_removed":
            self.rows_removed += frame.height
        elif kind == "row_added":
            self.rows_added += frame.height
        else:
            raise ValueError(f"unknown unmatched-row kind {kind!r}")
        if not self.state.remaining:
            return
        keys = list(self.plan.keys)
        encoded = _encode_temporal(frame.select(keys).head(self.state.remaining))
        for position in range(encoded.height):
            key = self.state.key(encoded, position, keys)
            if key is None:
                break
            self.state.record(kind=kind, key=key)

    def matched(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        if left.height != right.height:
            raise ValueError("paired comparison batches have different lengths")
        for a, b in zip(left.iter_slices(_BATCH_ROWS), right.iter_slices(_BATCH_ROWS), strict=True):
            _compare_batch(self.plan, self.state, _encode_temporal(a), _encode_temporal(b))

    def finish(
        self, *, execution: ComparisonExecution = "in_memory", fallback_reason: str | None = None
    ) -> ComparisonResult:
        for column in self.plan.dtypes_changed:
            if column in self.structural_changes:
                self._dtype_change(column)
        plan, state = self.plan, self.state
        total = (
            state.values
            + self.rows_added
            + self.rows_removed
            + len(plan.columns_added)
            + len(plan.columns_removed)
            + len(plan.dtypes_changed)
        )
        result = ComparisonResult(
            name=plan.name,
            keys=list(plan.keys),
            rtol=plan.rtol,
            atol=plan.atol,
            column_tolerances=dict(plan.column_tolerances),
            baseline_rows=self.baseline_rows,
            candidate_rows=self.candidate_rows,
            rows_added=self.rows_added,
            rows_removed=self.rows_removed,
            rows_changed=state.rows,
            values_changed=state.values,
            columns_added=list(plan.columns_added),
            columns_removed=list(plan.columns_removed),
            dtypes_changed=list(plan.dtypes_changed),
            max_abs_diff=state.max_abs,
            max_rel_diff=state.max_rel,
            nonfinite_changes=state.nonfinite,
            differences=state.differences,
            truncated=total > len(state.differences),
            execution=execution,
            fallback_reason=fallback_reason,
        )
        logger.info(result.summary())
        return result


def _compare_batch(
    plan: ComparisonPlan, state: _Changes, left: pl.DataFrame, right: pl.DataFrame
) -> None:
    keys, values = list(plan.keys), plan.values
    original_b_schema, original_c_schema = plan.baseline_schema, plan.candidate_schema
    tolerance = plan.tolerance
    batch = left
    changed_rows = pl.Series("changed", [False] * batch.height)
    for column in values:
        a, b = left[column], right[column]
        da, db = a.dtype, b.dtype
        original_da, original_db = original_b_schema[column], original_c_schema[column]
        scalar = not (_nested(original_da) or _nested(original_db))
        same_kind = not _type_changed(original_da, original_db, "compatible")
        cells = pl.DataFrame({"a": a, "b": b})
        settled = cells.select(
            _settled(pl.col("a"), pl.col("b"), da, db, *tolerance(column)).alias("ok")
        )["ok"]
        if not same_kind:
            settled = a.is_null() & b.is_null()
        if not _TOLERANCE_PREFILTER:
            settled = pl.Series("ok", [False] * batch.height)
        pending = (~settled).fill_null(True)
        if not pending.any():
            continue
        positions = pending.arg_true()
        # Equal-dtype scalar values are settled by the exact/native kernel.
        # Integer and decimal unequal pairs still require exact tolerance arithmetic.
        native = (
            _TOLERANCE_PREFILTER
            and scalar
            and same_kind
            and da == db
            and (da.is_float() or not da.is_numeric())
        )
        if native:
            count = len(positions)
            state.values += count
            changed_rows = changed_rows | pending
            if da.is_float():
                finite = pl.col("a").is_finite() & pl.col("b").is_finite()
                gap = (pl.col("a").cast(pl.Float64) - pl.col("b").cast(pl.Float64)).abs()
                relative_expr = (
                    pl.when(gap.is_infinite())
                    .then((pl.col("a") / pl.col("a").abs() - pl.col("b") / pl.col("a").abs()).abs())
                    .otherwise(gap / pl.col("a").abs())
                )
                metrics = (
                    cells.filter(pending)
                    .select(
                        (
                            (~pl.col("a").is_finite()).fill_null(False)
                            | (~pl.col("b").is_finite()).fill_null(False)
                        )
                        .sum()
                        .alias("nf"),
                        pl.when(finite).then(gap).max().alias("abs"),
                        pl.when(finite & (pl.col("a") != 0)).then(relative_expr).max().alias("rel"),
                    )
                    .row(0, named=True)
                )
                state.nonfinite += metrics["nf"] or 0
                state.maxima(metrics["abs"], metrics["rel"])
            for position in positions.head(state.remaining):
                key = state.key(left, position, keys)
                if key is None:
                    break
                captured = _capture_native(a, b, position)
                absolute, relative = _difference_size(captured["baseline"], captured["candidate"])
                state.record(
                    kind="value_changed",
                    key=key,
                    column=column,
                    **captured,
                    abs_diff=absolute,
                    rel_diff=relative,
                    rtol=tolerance(column)[0],
                    atol=tolerance(column)[1],
                )
            continue
        flags = [False] * batch.height
        for position in positions:
            before, after = a[position], b[position]
            different, path, changed_nonfinite = _cell_change(
                before, after, column, tolerance, original_da, original_db
            )
            if not different:
                continue
            flags[position] = True
            state.values += 1
            state.nonfinite += int(changed_nonfinite)
            absolute, relative = _difference_size(before, after)
            state.maxima(absolute, relative)
            if state.remaining:
                key = state.key(left, position, keys)
                if key is None:
                    continue
                captured = _capture_cells(before, after, path)
                state.record(
                    kind="value_changed",
                    key=key,
                    column=column,
                    path=path,
                    **captured,
                    abs_diff=absolute,
                    rel_diff=relative,
                    rtol=tolerance(column)[0],
                    atol=tolerance(column)[1],
                )
        changed_rows = changed_rows | pl.Series(flags)
    state.rows += int(changed_rows.sum())


def _compare_aligned(
    plan: ComparisonPlan,
    base: pl.DataFrame,
    cand: pl.DataFrame,
    *,
    execution: ComparisonExecution = "in_memory",
    fallback_reason: str | None = None,
) -> ComparisonResult:
    accumulator = ComparisonAccumulator(plan, base.height, cand.height)
    occupied = set(base.columns) | set(cand.columns)
    brow, crow = _fresh_name(occupied, "__qavf_brow"), _fresh_name(occupied, "__qavf_crow")
    keys = list(plan.keys)
    aligned = (
        base.select(keys)
        .with_row_index(brow)
        .join(
            cand.select(keys).with_row_index(crow),
            on=keys,
            how="full",
            nulls_equal=True,
            coalesce=True,
            maintain_order="left_right",
        )
    )
    accumulator.unmatched("row_removed", aligned.filter(pl.col(crow).is_null()))
    accumulator.unmatched("row_added", aligned.filter(pl.col(brow).is_null()))
    matched = aligned.filter(pl.col(brow).is_not_null() & pl.col(crow).is_not_null())
    for batch in matched.iter_slices(_BATCH_ROWS):
        accumulator.matched(base[batch[brow]], cand[batch[crow]])
    return accumulator.finish(execution=execution, fallback_reason=fallback_reason)
