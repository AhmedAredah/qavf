"""Run one comparison inside a data context: observe, align, evaluate each level.

Both datasets are observed once (coherent projections under the context's
source policies), aligned lazily, and each aligned side is evaluated once into
managed spill so every level reads the same rows. A side that cannot be
observed or aligned, or has too few rows, leaves every level not assessed. A
level that raises is not assessed; the other levels still run.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import polars as pl

from qavf.compare.align import AlignmentContext, AlignmentError, align, build_pipeline
from qavf.compare.levels import LEVELS, SIDES, ComparisonLevel, LevelInputs, not_assessed
from qavf.compare.results import DatasetComparison, LevelResult
from qavf.compare.spec import ComparisonSpec, Side
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.redaction import redact_text
from qavf.regression._plan import mapped_schema
from qavf.rules._execution import normalized_plan


def _declared(ctx: DataContext, dataset: str, field: str) -> dict[str, Any]:
    spec = ctx.specs.get(dataset)
    schema = spec.schema_spec if spec is not None else None
    columns = schema.columns if schema is not None else {}
    values = {name: getattr(column, field) for name, column in columns.items()}
    return {name: value for name, value in values.items() if value is not None}


def _counter(observation: Observation) -> Callable[[pl.LazyFrame], int]:
    def count(plan: pl.LazyFrame) -> int:
        return int(observation.collect(plan.select(pl.len())).item())

    return count


def _fixed(plan: pl.LazyFrame) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """A derivation that evaluates an already built plan over the observation."""
    return lambda _: plan


def _reason(exc: Exception) -> str:
    return redact_text(
        f"{type(exc).__name__}: {exc}" if not isinstance(exc, AlignmentError) else str(exc)
    )


def _requirements(spec: ComparisonSpec, levels: list[ComparisonLevel]) -> set[str] | None:
    selected = [level.required_columns() for level in levels]
    if any(columns is None for columns in selected):
        return None
    return set(spec.keys).union(*(columns or () for columns in selected))


def _source_columns(spec: ComparisonSpec, needed: set[str] | None, side: Side) -> list[str] | None:
    """Conservative dependency closure; undeclared plugin steps preserve every column."""
    if needed is None or spec.steps:
        return None
    columns = set(needed)
    for condition in (
        spec.where,
        spec.baseline_where if side == "baseline" else spec.candidate_where,
    ):
        if condition:
            columns.update(pl.sql_expr(condition).meta.root_names())
    for name, expression in reversed(list(spec.derive.items())):
        columns.discard(name)
        columns.update(pl.sql_expr(expression).meta.root_names())
    columns.update(spec.cast)
    columns.update(spec.units)
    columns.update(spec.recode)
    if side == "baseline":
        reverse = {new: old for old, new in spec.columns.items()}
        columns = {reverse.get(column, column) for column in columns} | set(spec.columns)
    return sorted(columns) or None  # An empty projection would erase the row count.


def _normalizes(ctx: DataContext, dataset: str, columns: dict[str, pl.DataType]) -> bool:
    spec = ctx.specs.get(dataset)
    contracts = spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
    return any(
        name in contracts and (contracts[name].missing_values or contracts[name].blank_as_missing)
        for name in columns
    )


def compare_in_context(ctx: DataContext, spec: ComparisonSpec) -> DatasetComparison:
    """Compare ``spec.baseline`` with ``spec.candidate``, two datasets of ``ctx``."""
    configured = [
        (config, LEVELS.create(config.level, config.params)) for config in spec.selected_levels()
    ]
    datasets: dict[Side, str] = {"baseline": spec.baseline, "candidate": spec.candidate}
    steps = build_pipeline(spec)
    needed = _requirements(spec, [level for _, level in configured])
    outcome: dict[str, Any] = {
        "name": spec.label,
        "baseline": spec.baseline,
        "candidate": spec.candidate,
        "keys": list(spec.keys),
        "column_map": dict(spec.columns),
        "alignment": [step.describe() for step in steps],
    }

    def unassessed(reason: str, **extra: Any) -> DatasetComparison:
        levels = [
            level.Result(
                level=config.level, name=config.label, checks=[not_assessed("assessment", reason)]
            )
            for config, level in configured
        ]
        return DatasetComparison(**outcome, **extra, error=reason, levels=levels)

    for dataset in dict.fromkeys(datasets.values()):
        if not ctx.has(dataset):
            return unassessed(
                f"dataset {dataset!r} not available ({ctx.unavailable_reason(dataset)})"
            )
    with ExitStack() as stack:
        try:
            requests: dict[str, list[str] | None] = {}
            for side in SIDES:
                dataset = datasets[side]
                columns = _source_columns(spec, needed, side)
                schema = ctx.schema(dataset)
                aligned_schema = (
                    mapped_schema(schema, spec.columns, "columns")[0]
                    if side == "baseline"
                    else schema
                )
                if clash := sorted(set(spec.derive) & set(aligned_schema)):
                    raise AlignmentError(f"derived column(s) {', '.join(clash)} already exist")
                if columns is not None:
                    columns = [column for column in columns if column in schema] or None
                if dataset not in requests:
                    requests[dataset] = columns
                elif columns is None or requests[dataset] is None:
                    requests[dataset] = None
                else:
                    requests[dataset] = sorted(set(requests[dataset] or ()) | set(columns))
            observed = stack.enter_context(ctx.projected_observations(requests))
            units: dict[Side, dict[str, str | None]] = {}
            nullable: dict[Side, dict[str, bool]] = {}
            aligned: dict[Side, pl.LazyFrame] = {}
            identities: dict[Side, bool] = {}
            for side in SIDES:
                observation = observed[datasets[side]]
                context = AlignmentContext(
                    side,
                    datasets[side],
                    count=_counter(observation),
                    units=_declared(ctx, datasets[side], "unit"),
                    nullable=_declared(ctx, datasets[side], "nullable"),
                    collect=observation.collect,
                    data=ctx,
                )
                original = observation.scan()
                plan = (
                    normalized_plan(ctx, datasets[side], original, declared_only=True)
                    if _normalizes(ctx, datasets[side], observation.schema())
                    else original
                )
                aligned[side] = align(plan, steps, context)
                identities[side] = aligned[side] is original
                if needed:
                    selected = [
                        column for column in aligned[side].collect_schema() if column in needed
                    ]
                    if selected:
                        aligned[side] = aligned[side].select(selected)
                units[side], nullable[side] = context.units, context.nullable
                outcome["alignment"].extend(context.notes)
            sides: dict[Side, Observation] = {
                side: observed[datasets[side]]
                if identities[side]
                else stack.enter_context(observed[datasets[side]].derived(_fixed(aligned[side])))
                for side in SIDES
            }
            rows = {
                side: int(sides[side].collect(sides[side].scan().select(pl.len())).item())
                for side in SIDES
            }
        except Exception as exc:
            return unassessed(f"alignment failed: {_reason(exc)}")
        counts = {"baseline_rows": rows["baseline"], "candidate_rows": rows["candidate"]}
        for side in SIDES:
            if rows[side] < spec.min_rows:
                return unassessed(
                    f"the {side} has {rows[side]:,} aligned row(s), fewer than min_rows "
                    f"({spec.min_rows:,})",
                    **counts,
                )
        inputs = LevelInputs(ctx, spec, sides, datasets, units, nullable, rows)
        results: list[LevelResult] = []
        for config, level in configured:
            try:
                result = level.evaluate(inputs, config.label)
                if not result.checks:
                    result = result.model_copy(
                        update={
                            "checks": [not_assessed("assessment", "the level returned no checks")]
                        }
                    )
                results.append(result)
            except Exception as exc:
                results.append(
                    level.Result(
                        level=config.level,
                        name=config.label,
                        checks=[not_assessed("assessment", _reason(exc))],
                    )
                )
        return DatasetComparison(**outcome, **counts, levels=results)
