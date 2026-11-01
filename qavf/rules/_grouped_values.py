"""Bounded compatibility execution of shared grouped-domain SQL plans."""

from __future__ import annotations

from typing import Any

import polars as pl

from qavf.core.policy import NativeValueDomain, native_literal
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.grouped_values import GroupedValuePlan
from qavf.rules._execution import _native_key_type
from qavf.rules._key_values import CanonicalKeyValues
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import fresh_internal, plural
from qavf.settings import ColumnCheck, GroupedValueSpec
from qavf.sources.sql import quote_identifier as qi


def _domain_literals(
    values: CanonicalKeyValues,
    types: dict[str, pl.DataType],
    column: str,
    configured: Any,
) -> list[str]:
    """Typed SQL literals for a configured domain, rejecting lossy members."""
    column_spec = values.contracts.get(column)
    if column_spec is not None:
        configured = column_spec._resolved_check(
            ColumnCheck(
                isin=configured,
                convert_options=column_spec.dtype in ("int", "float", "decimal"),
            )
        ).isin
    dtype = _native_key_type(types[column])
    admitted, foreign = NativeValueDomain(dtype).members(configured)
    if foreign or len(admitted) != len(configured):
        raise ValueError(f"Grouped-value domain for {column!r} is not losslessly representable")
    return [f"CAST({native_literal(value)} AS {dtype})" for value in admitted]


def grouped_values(ctx: DataContext, dataset: str, spec: GroupedValueSpec) -> Finding:
    """Assess one independent grouped contract and retain only bounded evidence."""
    columns = [*spec.group_by, spec.column]
    identity = {
        "check_id": "key.grouped_values",
        "datasets": (dataset,),
        "columns": tuple(columns),
        "parameters": spec.model_dump(mode="python"),
    }
    label = f"{dataset}: {spec.name or 'grouped values (' + ', '.join(columns) + ')'}"
    try:
        actual = ctx.schema(dataset)
        absent = [c for c in columns if c not in actual]
        if absent:
            raise ValueError(f"Grouped-value column(s) missing: {', '.join(absent)}")
        group_na = fresh_internal("group_na", actual)
        value_na = fresh_internal("value_na", [*actual, group_na])
        groups = CanonicalKeyValues(ctx, dataset, spec.group_by)
        values = CanonicalKeyValues(ctx, dataset, [spec.column])
        raw = (
            f"SELECT *, ({groups.na_sql()}) AS {qi(group_na)}, "
            f"({values.na_sql()}) AS {qi(value_na)} FROM {qi(dataset)}"
        )
        canonical, types = CanonicalKeyValues(ctx, dataset, actual if spec.where else columns).sql(
            source_sql=raw, preserve=(group_na, value_na)
        )
        domain = _domain_literals(values, types, spec.column, spec.allowed_values)
        expected = [
            [
                "NULL" if value is None else _domain_literals(values, types, column, [value])[0]
                for column, value in zip(spec.group_by, group, strict=True)
            ]
            for group in spec.expected_groups or []
        ]
        plan = GroupedValuePlan.build(
            spec,
            canonical,
            domain,
            expected,
            eligible_sql=f"NOT {qi(group_na)}",
            value_present_sql=f"NOT {qi(value_na)} AND {qi(spec.column)} IS NOT NULL",
        )
        evidence, count = ctx.sql_sample(plan.violations_sql, ctx.evidence_limit, ordered=True)
        total = int(ctx.sql(plan.total_sql, datasets=(dataset,)).item())
        if count:
            return Finding.fail(
                f"{plural(count, 'group value')} of {total:,} violate their occurrence contract.",
                check=label,
                affected=temporal_evidence(evidence),
                affected_count=count,
                total_records=total,
                unit="group values",
                **identity,
            )
        return Finding.ok(
            f"{plural(total, 'group value')} satisfy their occurrence contract.",
            check=label,
            total_records=total,
            unit="group values",
            **identity,
        )
    except Exception as exc:
        return Finding.unassessed(
            f"Not assessed: {label}: {type(exc).__name__}: {exc}.",
            check=label,
            **identity,
        )
