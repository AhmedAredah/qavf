"""Exact regression over native joined batches, without Python-side full tables."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol

import polars as pl

from qavf.engine.sql_read import NativeReadUnavailable
from qavf.regression import _comparison
from qavf.regression._comparison import ComparisonAccumulator
from qavf.regression._plan import ComparisonPlan
from qavf.regression._schema import _nested
from qavf.regression.models import ComparisonResult
from qavf.sources.sql import quote_identifier
from qavf.sources.typed import needs_typed_reader, needs_typed_sql_binding


class ComparisonSession(Protocol):
    def frame(self, query: str) -> pl.DataFrame: ...
    def batches(self, query: str, batch_rows: int = 65536) -> Iterator[pl.DataFrame]: ...


def native_supported(plan: ComparisonPlan) -> None:
    """Conservatively gate native representation, never approximate an exact type."""
    for side, schema in (("baseline", plan.baseline_schema), ("candidate", plan.candidate_schema)):
        if len({name.casefold() for name in schema}) != len(schema):
            raise NativeReadUnavailable(f"{side} has case-distinct column names")
        for column in plan.shared:
            dtype = schema[column]
            if (
                dtype == pl.Object
                or needs_typed_sql_binding(dtype)
                or (_nested(dtype) and needs_typed_reader(dtype))
            ):
                raise NativeReadUnavailable(
                    f"{side} column {column!r} needs exact typed comparison ({dtype})"
                )
    for key in plan.keys:
        if plan.baseline_schema[key] != plan.candidate_schema[key]:
            raise NativeReadUnavailable(f"key {key!r} requires value-dependent lossless alignment")


def _key_expression(reference: str, dtype: pl.DataType) -> str:
    # SQL column collations must not turn distinct text keys into equal values.
    # The empty concatenation preserves bytes/nulls while avoiding DuckDB 1.2
    # treating an encoded Parquet string dictionary as a VARCHAR group vector.
    return f"encode({reference} || '')" if dtype == pl.String else reference


def _paired_values(plan: ComparisonPlan, column: str, ordinal_sql: str) -> tuple[str, str]:
    """Elide equal non-key text/blob payloads before sorting and batch transfer.

    Exact comparison needs originals only for unequal cells. Keep their values,
    every null, every key and both one-sided rows untouched. This reduces the
    downstream sort/transfer width; the native join still reads original cells.
    """
    left, right = (f"{side}.{quote_identifier(column)}" for side in ("b", "c"))
    dtype = plan.baseline_schema[column]
    if (
        column in plan.keys
        or dtype != plan.candidate_schema[column]
        or dtype not in (pl.String, pl.Binary)
    ):
        return left, right
    equality = f"encode({left}) = encode({right})" if dtype == pl.String else f"{left} = {right}"
    equal_pair = (
        f"b.{ordinal_sql} IS NOT NULL AND c.{ordinal_sql} IS NOT NULL "
        f"AND {left} IS NOT NULL AND {right} IS NOT NULL AND {equality}"
    )
    empty = "CAST('' AS VARCHAR)" if dtype == pl.String else "CAST('' AS BLOB)"
    return (
        f"CASE WHEN {equal_pair} THEN {empty} ELSE {left} END",
        f"CASE WHEN {equal_pair} THEN {empty} ELSE {right} END",
    )


def _relation(table: str, plan: ComparisonPlan | None = None) -> str:
    """A FROM item exposing the baseline under candidate names when columns are mapped."""
    if plan is None or not plan.column_map:
        return quote_identifier(table)
    canonical = list(plan.baseline_schema)
    selected = ", ".join(
        f"{quote_identifier(stored)} AS {quote_identifier(name)}"
        for stored, name in zip(plan.baseline_columns(canonical), canonical, strict=True)
    )
    return f"(SELECT {selected} FROM {quote_identifier(table)}) AS mapped"


def _key_summary(
    session: ComparisonSession,
    relation: str,
    keys: tuple[str, ...],
    schema: Mapping[str, pl.DataType],
    side: str,
    name: str,
) -> int:
    grouped = [
        f"{_key_expression(quote_identifier(key), schema[key])} AS {quote_identifier(f'k{i}')}"
        for i, key in enumerate(keys)
    ]
    key_names = [quote_identifier(f"k{i}") for i in range(len(keys))]
    nan = [(i, key) for i, key in enumerate(keys) if schema[key].is_float()]
    nan_fields = [
        f"COALESCE(bool_or(isnan({quote_identifier(f'k{i}')})), false) "
        f"AS {quote_identifier(f'n{i}')}"
        for i, _ in nan
    ]
    fields = [
        "CAST(COALESCE(sum(n), 0) AS UBIGINT) AS rows",
        "CAST(COALESCE(sum(CASE WHEN n > 1 THEN n ELSE 0 END), 0) AS UBIGINT) AS duplicates",
        *nan_fields,
    ]
    query = (
        f"SELECT {', '.join(fields)} FROM ("
        f"SELECT {', '.join(key_names)}, count(*) AS n FROM ("
        f"SELECT {', '.join(grouped)} FROM {relation}"
        f") AS projected GROUP BY {', '.join(key_names)}"
        ") AS grouped"
    )
    summary = session.frame(query).row(0, named=True)
    for index, key in nan:
        if summary[f"n{index}"]:
            raise ValueError(
                f"{name}: {side} key column {key!r} contains NaN, which cannot identify a "
                "row; replace it (or use null) before comparing"
            )
    if summary["duplicates"]:
        raise ValueError(
            f"{name}: {side} has {summary['duplicates']} row(s) with duplicate key "
            f"({', '.join(keys)}); keys must identify rows uniquely"
        )
    return int(summary["rows"])


def compare_native(
    session: ComparisonSession, baseline: str, candidate: str, plan: ComparisonPlan
) -> ComparisonResult:
    """Consume a source-owned join in deterministic source order with bounded state.

    DuckDB owns the full join and ordering, subject to its memory/spill policy.
    Only bounded batches cross into Python; exact existing scalar kernels decide
    tolerance and evidence. Unsupported domains are rejected before this work.
    """
    native_supported(plan)
    batch_rows = _comparison._BATCH_ROWS
    baseline_rows = _key_summary(
        session, _relation(baseline, plan), plan.keys, plan.baseline_schema, "baseline", plan.name
    )
    candidate_rows = _key_summary(
        session, _relation(candidate), plan.keys, plan.candidate_schema, "candidate", plan.name
    )
    accumulator = ComparisonAccumulator(plan, baseline_rows, candidate_rows)
    columns = list(plan.shared)
    ordered = bool(accumulator.state.remaining)
    occupied = set(plan.baseline_schema) | set(plan.candidate_schema)
    ordinal = "__qavf_comparison_ordinal"
    while ordinal.casefold() in {name.casefold() for name in occupied}:
        ordinal += "_"
    ordinal_sql = quote_identifier(ordinal)
    selected = ", ".join(quote_identifier(column) for column in columns)

    def projected(relation: str) -> str:
        marker = "row_number() OVER ()" if ordered else "CAST(1 AS UTINYINT)"
        return f"SELECT {selected}, {marker} AS {ordinal_sql} FROM {relation}"

    left_names = [f"b{i}" for i in range(len(columns))]
    right_names = [f"c{i}" for i in range(len(columns))]
    fields = [f"b.{ordinal_sql} AS br", f"c.{ordinal_sql} AS cr"]
    paired = [_paired_values(plan, column, ordinal_sql) for column in columns]
    fields += [
        f"{expressions[0]} AS {quote_identifier(alias)}"
        for expressions, alias in zip(paired, left_names, strict=True)
    ]
    fields += [
        f"{expressions[1]} AS {quote_identifier(alias)}"
        for expressions, alias in zip(paired, right_names, strict=True)
    ]
    on = " AND ".join(
        f"{_key_expression('b.' + quote_identifier(key), plan.baseline_schema[key])} "
        f"IS NOT DISTINCT FROM "
        f"{_key_expression('c.' + quote_identifier(key), plan.candidate_schema[key])}"
        for key in plan.keys
    )
    query = (
        f"SELECT {', '.join(fields)} FROM ({projected(_relation(baseline, plan))}) AS b "
        f"FULL OUTER JOIN ({projected(_relation(candidate))}) AS c ON {on}"
    )
    if ordered:
        query += (
            f" ORDER BY CASE WHEN c.{ordinal_sql} IS NULL THEN 0 "
            f"WHEN b.{ordinal_sql} IS NULL THEN 1 ELSE 2 END, "
            f"COALESCE(b.{ordinal_sql}, c.{ordinal_sql})"
        )
    pending_left: list[pl.DataFrame] = []
    pending_right: list[pl.DataFrame] = []
    pending_rows = 0
    for batch in session.batches(query, batch_rows=batch_rows):
        removed = batch.filter(pl.col("cr").is_null())
        added = batch.filter(pl.col("br").is_null())
        matched = batch.filter(pl.col("br").is_not_null() & pl.col("cr").is_not_null())
        if removed.height:
            accumulator.unmatched(
                "row_removed",
                removed.select(left_names).rename(dict(zip(left_names, columns, strict=True))),
            )
        if added.height:
            accumulator.unmatched(
                "row_added",
                added.select(right_names).rename(dict(zip(right_names, columns, strict=True))),
            )
        if matched.height:
            left = matched.select(left_names).rename(dict(zip(left_names, columns, strict=True)))
            right = matched.select(right_names).rename(dict(zip(right_names, columns, strict=True)))
            position = 0
            while position < matched.height:
                take = min(batch_rows - pending_rows, matched.height - position)
                pending_left.append(left.slice(position, take))
                pending_right.append(right.slice(position, take))
                pending_rows += take
                position += take
                if pending_rows == batch_rows:
                    accumulator.matched(
                        pl.concat(pending_left, rechunk=False),
                        pl.concat(pending_right, rechunk=False),
                    )
                    pending_left.clear()
                    pending_right.clear()
                    pending_rows = 0
    if pending_rows:
        accumulator.matched(
            pl.concat(pending_left, rechunk=False), pl.concat(pending_right, rechunk=False)
        )
    return accumulator.finish(execution="native_stream")
