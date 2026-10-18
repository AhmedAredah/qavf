"""Exact native coordinate classification; source rows remain unchanged."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.sql import missing_sql
from qavf.rules._predicates import decimal_bound, float_bound
from qavf.sources.sql import quote_identifier, sql_literal

if TYPE_CHECKING:
    from qavf.rules.geo_checks import PointSet


PROBLEMS = (
    "missing coordinate",
    "non-finite coordinate",
    "latitude out of range (latitude and longitude swapped?)",
    "latitude out of range",
    "longitude out of range",
    "(0, 0): a missing location?",
    "outside the bounding box",
)


def coordinate_problem_sql(
    ctx: DataContext, points: PointSet, schema: dict[str, pl.DataType]
) -> str:
    """Match the typed classifier's precedence without lossy numeric casts."""
    spec = ctx.specs.get(points.dataset)
    columns = spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
    values = {
        name: f"(CASE WHEN {missing_sql(name, schema[name], columns.get(name))} "
        f"THEN NULL ELSE {quote_identifier(name)} END)"
        for name in (points.latitude, points.longitude)
    }

    def compare(column: str, operator: str, bound: Any) -> str:
        value, dtype = values[column], schema[column]
        if dtype.is_integer():
            bound = math.ceil(bound) if operator == "ge" else math.floor(bound)
        elif dtype.is_float():
            operator, bound = float_bound(operator, bound)
            value = f"CAST({value} AS DOUBLE)"
        elif dtype.is_decimal():
            operator, bound = decimal_bound(operator, bound, dtype)
        if isinstance(bound, bool):
            return "TRUE" if bound else "FALSE"
        return f"({value} {'>=' if operator == 'ge' else '<='} {sql_literal(bound)})"

    def within(column: str, lower: Any, upper: Any) -> str:
        return f"({compare(column, 'ge', lower)} AND {compare(column, 'le', upper)})"

    lat, lon = values[points.latitude], values[points.longitude]
    missing = f"({lat} IS NULL OR {lon} IS NULL)"
    nonfinite = (
        " OR ".join(
            f"NOT isfinite({values[name]})"
            for name in dict.fromkeys((points.latitude, points.longitude))
            if schema[name].is_float()
        )
        or "FALSE"
    )
    lat_range = within(points.latitude, -90, 90)
    lon_range = within(points.longitude, -180, 180)
    cases: list[tuple[str, str | None]] = [
        (missing, None if points.allow_missing else PROBLEMS[0]),
        (nonfinite, PROBLEMS[1]),
        (
            f"NOT {lat_range} AND {within(points.longitude, -90, 90)} "
            f"AND {within(points.latitude, -180, 180)}",
            PROBLEMS[2],
        ),
        (f"NOT {lat_range}", PROBLEMS[3]),
        (f"NOT {lon_range}", PROBLEMS[4]),
    ]
    if points.null_island:
        cases.append((f"{lat} = 0 AND {lon} = 0", PROBLEMS[5]))
    if points.bbox is not None:
        box = points.bbox
        cases.append(
            (
                f"NOT {within(points.latitude, box.min_lat, box.max_lat)} "
                f"OR NOT {within(points.longitude, box.min_lon, box.max_lon)}",
                PROBLEMS[6],
            )
        )
    return (
        "CASE "
        + " ".join(
            f"WHEN ({condition}) THEN {sql_literal(problem)}" for condition, problem in cases
        )
        + " ELSE NULL END"
    )
