"""SQL dependency analysis and exact typed bindings for the validation engine."""

from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any, cast

import duckdb
import polars as pl
import pyarrow as pa

from qavf.sources.base import SourceError
from qavf.sources.sql import normalize_single_query, quote_identifier, sql_literal


@functools.lru_cache(maxsize=256)
def sql_table_names(query: str) -> set[str]:
    """Extract unqualified base relations from DuckDB's syntax tree.

    Parsing does not bind dummy tables, so JOIN USING and natural joins need no
    guessed schema. Namespace fields and CTE scope are preserved explicitly.
    """
    with duckdb.connect(config={"threads": 1, "memory_limit": "16MB"}) as parser:
        parse_query = query
        # Dynamic PIVOT expands to an enum declaration plus a SELECT in DuckDB;
        # its direct relation still has ordinary namespace binding.
        direct_pivot = re.match(
            r'^\s*PIVOT(?:_WIDER)?\s+((?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z_0-9]*)(?:\s*\.\s*(?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z_0-9]*))*)\s+ON\b',
            query,
            re.I,
        )
        if direct_pivot is not None:
            parse_query = "SELECT * FROM " + direct_pivot.group(1)
        row = parser.execute("SELECT json_serialize_sql(?)", [parse_query]).fetchone()
        assert row is not None
        parsed = json.loads(row[0])
    if parsed.get("error"):
        raise duckdb.ParserException(parsed.get("error_message", "invalid SQL"))
    names: set[str] = set()

    def visit(value: Any, scope: frozenset[str] = frozenset()) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, scope)
        elif isinstance(value, dict):
            entries = value.get("cte_map", {}).get("map", [])
            local = scope | frozenset(str(entry["key"]).lower() for entry in entries)
            if value.get("type") == "BASE_TABLE":
                name = value["table_name"]
                schema = value.get("schema_name", "").lower()
                if (
                    schema in ("", "main")
                    and not value.get("catalog_name")
                    # Explicit main qualification bypasses CTE shadowing.
                    and (schema == "main" or name.lower() not in local)
                ):
                    names.add(name)
            for key, child in value.items():
                if key != "cte_map":
                    recursive = value.get("type") == "RECURSIVE_CTE_NODE" and key == "right"
                    visible = (
                        local | frozenset((str(value["cte_name"]).lower(),)) if recursive else local
                    )
                    visit(child, visible)
            # Nonrecursive CTE bodies can reference a base table with their own name.
            preceding = scope
            for entry in entries:
                visit(entry["value"], preceding)
                preceding = preceding | frozenset((str(entry["key"]).lower(),))

    visit(parsed)
    return names


def check_sql_columns(name: str, columns: Iterable[str]) -> None:
    seen: dict[str, str] = {}
    for column in columns:
        prior = seen.setdefault(column.lower(), column)
        if prior != column:
            raise SourceError(
                f"dataset {name!r} has columns {prior!r} and {column!r} that SQL treats "
                "as the same identifier; rename one before SQL validation"
            )


def missing_sql(column: str, dtype: pl.DataType, spec: Any = None) -> str:
    """Compile the exact typed missing-value predicate for a SQL-bound column.

    Raw column validation can reuse this predicate while retaining original
    NaN/sentinel representations for explicit forbidden-value checks/evidence.
    """
    from qavf.value_policy import is_temporal, is_text, members, temporal_ticks

    quoted = quote_identifier(column)
    conditions = [f"{quoted} IS NULL"]
    if dtype.is_float():
        conditions.append(f"isnan({quoted})")
    if spec is not None:
        values, foreign = members(spec.missing_values, dtype)
        if foreign:
            raise ValueError(f"missing markers for {column!r} have incompatible types")
        if is_temporal(dtype):
            function = (
                f"date_diff('day', DATE '1970-01-01', {quoted})"
                if dtype == pl.Date
                else f"epoch_{cast(pl.Datetime, dtype).time_unit}({quoted})"
            )
            ticks = [temporal_ticks(value, dtype) for value in values]
            literals = [
                str(int(value))
                for value in ticks
                if value.denominator == 1 and -(2**63) <= value < 2**63
            ]
            if literals:
                conditions.append(f"{function} IN ({', '.join(literals)})")
        elif values:
            literals = [
                f"CAST({sql_literal(str(value))} AS DOUBLE)"
                if isinstance(value, float) and not math.isfinite(value)
                else sql_literal(value)
                for value in values
            ]
            conditions.append(f"{quoted} IN ({', '.join(literals)})")
        if spec.blank_as_missing and is_text(dtype):
            # Rust/Polars Unicode White_Space, explicitly supplied to DuckDB trim().
            whitespace = (
                "\t\n\v\f\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004"
                "\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
            )
            conditions.append(f"trim(CAST({quoted} AS VARCHAR), {sql_literal(whitespace)}) = ''")
    return "(" + " OR ".join(conditions) + ")"


def missing_projection(schema: Mapping[str, pl.DataType], columns: Mapping[str, Any]) -> str:
    """Normalize only the typed missing markers of the supplied column contracts."""
    return ", ".join(
        f"CASE WHEN {missing_sql(column, dtype, columns.get(column))} "
        f"THEN NULL ELSE {quote_identifier(column)} END AS {quote_identifier(column)}"
        for column, dtype in schema.items()
    )


def sample_statement(query: str, limit: int, *, ordered: bool = False) -> str:
    """Count and sample one evaluation, with DuckDB enforcing memory/spill limits."""
    if limit < 0:
        raise ValueError("sample limit must be non-negative")
    body = normalize_single_query(query)
    order = " ORDER BY ALL" if ordered else ""
    return (
        f"WITH __qavf_rows AS MATERIALIZED ({body}\n) "
        "SELECT (SELECT count(*) FROM __qavf_rows) AS __count, "
        f"(SELECT list(s) FROM (SELECT * FROM __qavf_rows{order} "
        f"LIMIT {int(limit)}) s) AS __sample"
    )


def sample_result(table: pa.Table) -> tuple[pl.DataFrame, int]:
    """Decode a bounded Arrow list-of-struct sample without collecting its source."""
    count = int(table["__count"][0].as_py())
    structs = table["__sample"].chunk(0).values
    sample = cast(
        pl.DataFrame,
        pl.from_arrow(
            pa.Table.from_arrays(structs.flatten(), names=[field.name for field in structs.type])
        ),
    )
    return sample, count
