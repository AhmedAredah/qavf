"""Coherent relational coverage checks without Python identifier sets."""

from __future__ import annotations

import math
from contextlib import ExitStack
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.rules._domains import align_domains, count, count_sample, union_domains
from qavf.rules._execution import normalized_plan
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import fresh_internal, plural, usable

if TYPE_CHECKING:
    from qavf.rules.spatial_checks import CoverageParams


def evaluate_coverage(ctx: DataContext, params: CoverageParams) -> list[Finding]:
    column, time = params.entity_column, params.time_column
    if column is None:
        return [Finding.skip("No entity column is configured.")]
    blocked: list[Finding] = []
    targets: dict[str, bool] = {}

    def unavailable(name: str, reason: object) -> None:
        blocked.append(
            Finding.unassessed(
                f"Not assessed: coverage of {name}: {reason}.", check=name, datasets=(name,)
            )
        )

    for name in params.datasets if params.datasets is not None else ctx.available():
        try:
            if reason := usable(ctx, name, (column,)):
                if params.datasets is not None:
                    unavailable(name, reason)
                continue
            timed = bool(time and time in ctx.schema(name))
            if time and params.require_time_column and not timed:
                unavailable(name, f"it lacks the time column {time!r} (require_time_column: true)")
                continue
            targets[name] = timed
        except Exception as exc:
            unavailable(name, exc)
    if params.reference_dataset is not None:
        targets.pop(params.reference_dataset, None)
    if not targets:
        return blocked or [Finding.unassessed("Not assessed: no target datasets remain.")]

    def explicit(values: tuple[Any, ...], label: str, name: str) -> pl.LazyFrame:
        if any(
            value is None
            or (isinstance(value, float) and not math.isfinite(value))
            or (isinstance(value, Decimal) and not value.is_finite())
            for value in values
        ):
            raise ValueError(f"expected {label} contain missing or nonfinite values")
        return pl.Series(name, values, strict=True).to_frame().lazy().unique()

    with ExitStack() as stack:
        plans: dict[str, pl.LazyFrame] = {}
        observations = {}
        names = [*targets]
        if params.reference_dataset and params.expected_entities is None:
            names.append(params.reference_dataset)
        for name in dict.fromkeys(names):
            try:
                if reason := usable(ctx, name, (column,)):
                    raise ValueError(reason)
                columns = [column, *([time] if time and time in ctx.schema(name) else [])]
                observation = stack.enter_context(ctx.observe(name, columns))
                plan = normalized_plan(ctx, name, observation.scan()).unique()
                # Retain the existing public precision contract; comparisons themselves
                # stay native, so Python datetime equality never merges DST folds.
                for key, dtype in plan.collect_schema().items():
                    fine = dtype == pl.Time or (
                        isinstance(dtype, pl.Datetime | pl.Duration) and dtype.time_unit == "ns"
                    )
                    if (
                        fine
                        and observation.collect(
                            plan.select((pl.col(key).drop_nulls().cast(pl.Int64) % 1000 != 0).any())
                        ).item()
                    ):
                        raise ValueError(
                            "coverage temporal identifiers require microsecond precision; "
                            "use physical integer keys for finer resolution"
                        )
                plans[name], observations[name] = plan, observation
            except Exception as exc:
                unavailable(name, exc)
                targets.pop(name, None)
        if not targets:
            return blocked or [Finding.unassessed("Not assessed: no target datasets remain.")]
        collect = next(iter(observations.values())).collect
        try:
            if params.expected_entities is not None:
                expected = explicit(params.expected_entities, "entities", column)
            elif params.reference_dataset is not None:
                if params.reference_dataset not in plans:
                    raise ValueError("reference dataset is unavailable")
                expected = plans[params.reference_dataset].select(column).drop_nulls().unique()
            else:
                expected = union_domains((plans[name] for name in targets), column, collect)
            if params.expected_periods is not None:
                periods = explicit(params.expected_periods, "periods", str(time))
            else:
                periods = union_domains(
                    (plans[name] for name, timed in targets.items() if timed), str(time), collect
                )
            entities_count = count(expected, collect)
            periods_count = count(periods, collect)
        except Exception as exc:
            return [*blocked, Finding.unassessed(f"Not assessed: coverage expectations: {exc}.")]
        if not entities_count:
            return [
                *blocked,
                Finding.unassessed("Not assessed: there are no expected entities to check."),
            ]
        if any(targets.values()) and not periods_count:
            return [
                *blocked,
                Finding.unassessed("Not assessed: time-indexed datasets contain no periods."),
            ]
        records: list[dict[str, Any]] = []
        failures = checks = 0
        strict = params.unexpected_entities and (
            params.expected_entities is not None or params.reference_dataset is not None
        )
        for name, timed in targets.items():
            try:
                expected, actual = align_domains(expected, plans[name], [column], collect)
                actual = actual.filter(pl.col(column).is_not_null())
                matched = actual.join(expected, on=column, how="semi")
                extras = actual.join(expected, on=column, how="anti")
                missing_count_name = fresh_internal("missing_count", actual.collect_schema())
                extra_count_name = fresh_internal(
                    "extra_count", [*actual.collect_schema(), missing_count_name]
                )
                if timed:
                    assert time is not None
                    periods, actual = align_domains(periods, actual, [time], collect)
                    # Rebuild the joins after period casts, preserving exact join types.
                    matched = actual.join(expected, on=column, how="semi")
                    extras = actual.join(expected, on=column, how="anti")
                    matched_counts = matched.group_by(time).agg(pl.len().alias(missing_count_name))
                    slices = periods.join(matched_counts, on=time, how="left").with_columns(
                        (pl.lit(entities_count) - pl.col(missing_count_name).fill_null(0)).alias(
                            missing_count_name
                        )
                    )
                    if strict:
                        extra_counts = extras.group_by(time).agg(pl.len().alias(extra_count_name))
                        slices = slices.join(extra_counts, on=time, how="left").with_columns(
                            pl.col(extra_count_name).fill_null(0)
                        )
                    else:
                        slices = slices.with_columns(pl.lit(0).alias(extra_count_name))
                    checks += periods_count
                    order = [time]
                else:
                    matched_count = count(matched, collect)
                    extra_count = count(extras, collect) if strict else 0
                    slices = pl.DataFrame(
                        {
                            missing_count_name: [entities_count - matched_count],
                            extra_count_name: [extra_count],
                        }
                    ).lazy()
                    checks += 1
                    order = []
                bad = slices.filter(
                    (pl.col(missing_count_name) > 0) | (pl.col(extra_count_name) > 0)
                )
                failed, sample = count_sample(
                    bad, max(0, ctx.evidence_limit - len(records)), collect, order=order
                )
                failures += failed
                for index, row in enumerate(sample.iter_rows(named=True)):
                    selected = actual
                    if timed:
                        assert time is not None
                        period = sample.select(time).slice(index, 1).lazy()
                        selected = selected.join(period, on=time, how="semi")
                    missing = expected.join(selected.select(column), on=column, how="anti")
                    _, missing_sample = count_sample(
                        missing, ctx.evidence_limit, collect, order=[column]
                    )
                    record = {
                        "dataset": name,
                        **(
                            temporal_evidence(sample.select(time).slice(index, 1)).row(
                                0, named=True
                            )
                            if timed
                            else {}
                        ),
                        "missing_entities": temporal_evidence(missing_sample)[column].to_list(),
                    }
                    if row[missing_count_name] > ctx.evidence_limit:
                        record["missing_entities_count"] = row[missing_count_name]
                    if strict:
                        extra = selected.join(expected, on=column, how="anti").select(column)
                        _, extra_sample = count_sample(
                            extra, ctx.evidence_limit, collect, order=[column]
                        )
                        record["unexpected_entities"] = temporal_evidence(extra_sample)[
                            column
                        ].to_list()
                        if row[extra_count_name] > ctx.evidence_limit:
                            record["unexpected_entities_count"] = row[extra_count_name]
                    records.append(record)
            except Exception as exc:
                unavailable(name, exc)
        metrics = {"expected_entities": entities_count, "slices_checked": checks}
        if failures:
            return [
                *blocked,
                Finding.fail(
                    f"{plural(failures, 'dataset slice')} of {checks:,} "
                    f"{'do not match' if strict else 'lack one or more of'} "
                    f"the {entities_count} expected entities.",
                    affected=records,
                    affected_count=failures,
                    total_records=checks,
                    unit="slices",
                    metrics=metrics,
                    datasets=tuple(targets),
                ),
            ]
        if not checks:
            return blocked or [Finding.unassessed("Not assessed: no dataset slices were checked.")]
        return [
            *blocked,
            Finding.ok(
                f"All {checks:,} dataset slices cover the {entities_count} expected entities.",
                total_records=checks,
                unit="slices",
                metrics=metrics,
                datasets=tuple(targets),
            ),
        ]
