"""Canonical row predicates over bounded native batches when SQL semantics differ."""

from __future__ import annotations

from collections.abc import Callable
from types import NotImplementedType
from typing import Any

import polars as pl

from qavf.engine.registry import Finding
from qavf.engine.sql_read import NativeReadUnavailable, SqlReadSession
from qavf.rules._column_batch import ColumnInputs, independent_scope
from qavf.rules._predicates import compatibility_error, row_tests
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import present
from qavf.settings import ROW_CONSTRAINTS, ColumnCheck
from qavf.sources.sql import quote_identifier


def native_row_check(
    inputs: ColumnInputs,
    column: str,
    dependencies: tuple[str, ...],
    check: ColumnCheck | None,
    label: str,
) -> Finding | NotImplementedType | None:
    """Count every regex violation using the same Polars predicate as eager/lazy.

    Row-only contracts and simple per-row scopes can be evaluated in batches.
    Aggregate-dependent scopes and mixed exact aggregate contracts retain their
    projected reference path; they must not change meaning at batch boundaries.
    Evidence retains original dependency values, position and reasons from this
    same scan. Unrelated payload columns are neither decoded nor replayed.
    """
    if check is None or check.regex is None:
        return NotImplemented
    constraints = check.constraints()
    if set(constraints) - {*ROW_CONSTRAINTS, "max_null_fraction"}:
        return NotImplemented
    if independent_scope(check.where, inputs.actual):
        return NotImplemented
    inputs.validate(dependencies)
    dtype = inputs.actual[column]
    for name, value in constraints.items():
        if name in ROW_CONSTRAINTS and (error := compatibility_error(name, value, dtype)):
            raise ValueError(error)
    scope = pl.sql_expr(check.where).fill_null(False) if check.where else pl.lit(True)
    predicates = [
        (name, scope & pl.col(column).is_not_null() & ~predicate(column).fill_null(True))
        for name, predicate in row_tests(constraints, dtype).values()
    ]
    bad = pl.any_horizontal(predicate for _, predicate in predicates).fill_null(False)
    reasons = pl.concat_str(
        [pl.when(predicate).then(pl.lit(name)).otherwise(None) for name, predicate in predicates],
        separator="; ",
        ignore_nulls=True,
    )
    evidence = inputs.ctx.evidence_limit > 0
    saved = inputs.saved_for(dependencies) if evidence else {}
    selected = [*dependencies, *saved.values(), *([inputs.row] if evidence else [])]
    projection = ", ".join(map(quote_identifier, selected))
    statement = f"SELECT {projection} FROM {inputs.native(dependencies)}"

    def consumer_factory() -> Callable[
        [SqlReadSession], tuple[int, int, int, list[dict[str, Any]]]
    ]:
        def consume(session: SqlReadSession) -> tuple[int, int, int, list[dict[str, Any]]]:
            applicable = missing = count = kept = 0
            samples: list[dict[str, Any]] = []
            for frame in session.batches(statement):
                scoped = frame.filter(scope)
                applicable += scoped.height
                missing += scoped[column].null_count()
                count += int(frame.select(bad.sum()).item())
                remaining = inputs.ctx.evidence_limit - kept
                if remaining > 0:
                    sample = (
                        frame.filter(bad).head(remaining).with_columns(reasons.alias(inputs.reason))
                    )
                    kept += sample.height
                    if sample.height:
                        original = sample.drop(list(saved)).rename({v: k for k, v in saved.items()})
                        # Owning only sampled scalar values avoids retaining the
                        # complete Arrow batch behind a zero-copy frame slice.
                        samples.extend(
                            temporal_evidence(
                                present(
                                    original.select(*dependencies, inputs.row, inputs.reason),
                                    {inputs.row: "_row", inputs.reason: "violates"},
                                )
                            ).to_dicts()
                        )
            return applicable, missing, count, samples

        return consume

    try:
        applicable, missing, count, samples = inputs.ctx.consume_sql(
            consumer_factory, datasets=(inputs.dataset,)
        )
    except NativeReadUnavailable:
        return NotImplemented
    if applicable < check.min_applicable_rows:
        raise ValueError(
            f"only {applicable} row(s) match 'where: {check.where}', "
            f"fewer than min_applicable_rows {check.min_applicable_rows}"
        )
    problems = []
    if check.max_null_fraction is not None:
        fraction = missing / applicable if applicable else 0.0
        if fraction > check.max_null_fraction:
            problems.append(f"missing fraction {fraction:.4g} exceeds {check.max_null_fraction:g}")
    if not count and not problems:
        return None
    dataset = inputs.dataset
    if not count:
        return Finding.fail(
            f"{dataset}.{column} fails '{label}': {'; '.join(problems)}.",
            check=f"{dataset}.{column}: {label}",
            severity=check.severity,
            affected=[{"dataset": dataset, "column": column, "problem": p} for p in problems],
            unit="problems",
            datasets=(dataset,),
        )
    return Finding.fail(
        f"{count:,} value{'s' if count != 1 else ''} of {applicable:,} "
        f"in {dataset}.{column} violate '{label}'."
        + (f" Column-level: {'; '.join(problems)}." if problems else ""),
        check=f"{dataset}.{column}: {label}",
        severity=check.severity,
        affected=samples or None,
        affected_count=count,
        total_records=applicable,
        unit="rows",
        metrics={"column_problems": problems} if problems else {},
        datasets=(dataset,),
    )
