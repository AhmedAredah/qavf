"""Exact scalar column checks over bounded native result batches."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import NotImplementedType
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf._float_accumulator import ExactFloatAccumulator
from qavf.engine.registry import Finding
from qavf.engine.sql_read import NativeReadUnavailable
from qavf.numeric import stable_float_statistic
from qavf.rules._column_batch import ColumnInputs
from qavf.settings import ColumnCheck
from qavf.sources.sql import quote_identifier

if TYPE_CHECKING:
    from qavf.engine.sql_read import SqlReadSession


def native_float_statistics(
    inputs: ColumnInputs,
    column: str,
    dependencies: tuple[str, ...],
    check: ColumnCheck | None,
    label: str,
    within: Callable[[Any, Any], bool],
    show: Callable[[Any], str],
) -> Finding | NotImplementedType | None:
    """Stream sum/mean/deviation contracts without retaining a complete column.

    Unsupported composite/order-statistic contracts keep their exact projected
    path. The source operation owns retries: each attempt creates a fresh
    accumulator, and publishes its result only after complete verification.
    """
    if check is None or not inputs.actual[column].is_float():
        return NotImplemented
    constraints = check.constraints()
    statistics = {key: value for key, value in constraints.items() if key in {"sum", "mean", "std"}}
    if not statistics or set(constraints) - {"sum", "mean", "std", "max_null_fraction"}:
        return NotImplemented
    inputs.validate(dependencies)
    scope = f"coalesce(({check.where}), FALSE)" if check.where else "TRUE"
    statement = (
        f"SELECT {quote_identifier(column)} FROM {inputs.native(dependencies)} WHERE {scope}"
    )

    def consumer_factory() -> Callable[[SqlReadSession], tuple[ExactFloatAccumulator, int, Any]]:
        def consume(session: SqlReadSession) -> tuple[ExactFloatAccumulator, int, Any]:
            accumulator, applicable = ExactFloatAccumulator(), 0

            def values() -> Iterator[pl.Series]:
                nonlocal applicable
                for frame in session.batches(statement):
                    applicable += frame.height
                    series = frame[column]
                    accumulator.update(series)
                    yield series

            deviation = None
            if "std" in statistics:
                deviation = stable_float_statistic(
                    (v for series in values() for v in series), "std"
                )
            else:
                for _ in values():
                    pass
            return accumulator, applicable, deviation

        return consume

    try:
        accumulator, applicable, deviation = inputs.ctx.consume_sql(
            consumer_factory, datasets=(inputs.dataset,)
        )
    except NativeReadUnavailable:
        # A concurrent consumer may have materialized the input since dispatch.
        # Its exact projected reader remains valid; real source failures propagate.
        return NotImplemented
    if applicable < check.min_applicable_rows:
        raise ValueError(
            f"only {applicable} row(s) match 'where: {check.where}', "
            f"fewer than min_applicable_rows {check.min_applicable_rows}"
        )
    problems = []
    for statistic, bounds in statistics.items():
        value = {"sum": accumulator.total, "mean": accumulator.mean}.get(
            statistic, lambda: deviation
        )()
        if not within(value, bounds):
            problems.append(
                f"{statistic} {show(value)} outside [{show(bounds.ge)}, {show(bounds.le)}]"
            )
    if check.max_null_fraction is not None:
        fraction = (applicable - accumulator.count) / applicable if applicable else 0.0
        if fraction > check.max_null_fraction:
            problems.append(f"missing fraction {fraction:.4g} exceeds {check.max_null_fraction:g}")
    if not problems:
        return None
    dataset = inputs.dataset
    return Finding.fail(
        f"{dataset}.{column} fails '{label}': {'; '.join(problems)}.",
        check=f"{dataset}.{column}: {label}",
        severity=check.severity,
        affected=[{"dataset": dataset, "column": column, "problem": p} for p in problems],
        unit="problems",
        datasets=(dataset,),
    )
