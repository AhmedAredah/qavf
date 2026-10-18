"""Exact key-grid counts and gap evidence over scoped lazy key relations."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import TYPE_CHECKING

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.registry import Finding
from qavf.rules._domains import Collector, align_domains, count, count_sample
from qavf.rules._execution import normalized_plan
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import fresh_internal, missing_expr, plural, present, usable

if TYPE_CHECKING:
    from qavf.rules.reconciliation_checks import GridSpec


def evaluate_grid(ctx: DataContext, grid: GridSpec) -> list[Finding] | None:
    label, columns = grid.label(), list(grid.dimensions)
    projections: dict[str, list[str]] = {grid.dataset: columns.copy()}
    for column, dimension in grid.dimensions.items():
        if dimension.from_dataset is not None:
            projections.setdefault(dimension.from_dataset, []).append(column)
    for name, selected in projections.items():
        if reason := usable(ctx, name, selected):
            return [Finding.unassessed(f"Not assessed: {label}: {reason}.", check=label)]
    with ctx.projected_observations(projections) as observations:
        collect = observations.collect
        source = normalized_plan(ctx, grid.dataset, observations[grid.dataset].scan()).select(
            columns
        )
        occurrence = fresh_internal("occurrences", columns)
        grouped = source.group_by(columns).agg(pl.len().cast(pl.UInt64).alias(occurrence))
        source_rows = int(collect(grouped.select(pl.col(occurrence).sum())).item())
        domains: dict[str, pl.LazyFrame] = {}
        sizes: dict[str, int] = {}
        for column, dimension in grid.dimensions.items():
            if dimension.from_dataset is not None:
                reference = (
                    normalized_plan(
                        ctx, dimension.from_dataset, observations[dimension.from_dataset].scan()
                    )
                    .select(column)
                    .drop_nulls()
                )
            else:
                reference_frame = pl.Series(column, dimension.values, strict=True).to_frame()
                if reference_frame.select(missing_expr(reference_frame, column).any()).item():
                    raise ValueError(f"expected domain {column} contains missing values")
                reference = reference_frame.lazy()
            grouped, reference = align_domains(grouped, reference, [column], collect)
            domain = reference.select(column).unique().sort(column)
            size = count(domain, collect)
            if not size:
                raise ValueError(f"no expected values for {column}")
            domains[column], sizes[column] = domain, size
        expected_count = prod(sizes.values())
        missing_key = pl.any_horizontal(pl.col(column).is_null() for column in columns)
        null_count = int(
            collect(grouped.filter(missing_key).select(pl.col(occurrence).sum())).item()
        )
        actual = grouped.filter(~missing_key).select(columns)
        valid = actual
        for column, domain in domains.items():
            valid = valid.join(domain, on=column, how="semi")
        valid_count = count(valid, collect)
        missing_count = expected_count - valid_count
        findings: list[Finding] = []
        if missing_count:
            missing = missing_grid_sample(
                observations[grid.dataset],
                valid,
                domains,
                sizes,
                min(ctx.evidence_limit, missing_count),
                collect,
            )
            findings.append(
                Finding.fail(
                    f"{label}: {plural(missing_count, 'expected combination')} "
                    f"of {expected_count:,} missing.",
                    check=label,
                    affected=temporal_evidence(missing),
                    affected_count=missing_count,
                    total_records=expected_count,
                    unit="keys",
                    datasets=(grid.dataset,),
                )
            )
        if grid.unexpected:
            extra_count, sample = count_sample(
                actual.join(valid, on=columns, how="anti"),
                ctx.evidence_limit,
                collect,
                order=columns,
            )
            if extra_count:
                findings.append(
                    Finding.fail(
                        f"{label}: {plural(extra_count, 'unexpected combination')} present.",
                        check=f"{label}: unexpected",
                        affected=temporal_evidence(sample),
                        affected_count=extra_count,
                        unit="keys",
                        datasets=(grid.dataset,),
                    )
                )
            if null_count:
                sample = collect(
                    grouped.filter(missing_key).select(columns).head(ctx.evidence_limit)
                )
                findings.append(
                    Finding.fail(
                        f"{label}: {plural(null_count, 'row')} with a missing key.",
                        check=f"{label}: null keys",
                        affected=temporal_evidence(sample),
                        affected_count=null_count,
                        total_records=source_rows,
                        unit="rows",
                        datasets=(grid.dataset,),
                    )
                )
        if grid.unique:
            repeated_count, sample = count_sample(
                grouped.filter(pl.col(occurrence) > 1), ctx.evidence_limit, collect, order=columns
            )
            if repeated_count:
                findings.append(
                    Finding.fail(
                        f"{label}: {plural(repeated_count, 'combination')} occur more than once.",
                        check=f"{label}: unique",
                        affected=temporal_evidence(present(sample, {occurrence: "occurrences"})),
                        affected_count=repeated_count,
                        unit="keys",
                        datasets=(grid.dataset,),
                    )
                )
        return findings or None


def missing_grid_sample(
    observation: Observation,
    actual: pl.LazyFrame,
    domains: Mapping[str, pl.LazyFrame],
    sizes: Mapping[str, int],
    limit: int,
    collect: Collector,
) -> pl.DataFrame:
    """Merge sorted native domain ranks; retain only K missing rank tuples in Python.

    The Cartesian expected domain is never constructed. Domain rank joins and
    sorting remain lazy/native; a capped managed IPC stream carries actual ranks.
    Python integer arithmetic keeps cardinalities beyond 64 bits exact.
    """
    schema = {column: domain.collect_schema()[column] for column, domain in domains.items()}
    if limit <= 0:
        return pl.DataFrame(schema=schema)
    columns = list(domains)
    used = list(columns)
    ranks = {}
    mappings = {}
    ordered = actual
    for column, domain in domains.items():
        rank = fresh_internal("domain_rank", used)
        used.append(rank)
        ranks[column] = rank
        mapping = domain.with_columns(pl.int_range(0, pl.len(), dtype=pl.UInt64).alias(rank))
        mappings[column] = mapping
        ordered = ordered.join(mapping, on=column, how="left")
    ordered = ordered.select(list(ranks.values())).sort(list(ranks.values()))
    widths = list(sizes.values())
    total = prod(widths)
    rows: list[tuple[int, ...]] = []
    cursor = 0

    def decode(index: int) -> tuple[int, ...]:
        output = []
        for width in reversed(widths):
            index, remainder = divmod(index, width)
            output.append(remainder)
        return tuple(reversed(output))

    with observation.derived(lambda _: ordered) as ranked_observation:
        for batch in ranked_observation.batches():
            for record in batch.iter_rows():
                index = 0
                for rank, width in zip(record, widths, strict=True):
                    index = index * width + rank
                while cursor < index and len(rows) < limit:
                    rows.append(decode(cursor))
                    cursor += 1
                if len(rows) >= limit:
                    break
                cursor = index + 1
            if len(rows) >= limit:
                break
    while cursor < total and len(rows) < limit:
        rows.append(decode(cursor))
        cursor += 1
    result = pl.DataFrame(rows, schema={rank: pl.UInt64 for rank in ranks.values()}, orient="row")
    row = fresh_internal("sample_row", [*ranks.values(), *columns])
    plan = result.with_row_index(row).lazy()
    for column, mapping in mappings.items():
        plan = plan.join(mapping, on=ranks[column], how="left")
    return collect(plan.sort(row).select(columns))
