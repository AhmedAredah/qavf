"""Point-to-zone assignment: each point gets the id of the zone polygon it falls in.

The one point-in-polygon path of QAVF: the ``assign_zones`` comparison step and
GEO-001's ``within`` both call :func:`assign`. A plan's distinct points (with
their row counts) are collected natively, transformed to the layer's CRS and
joined to the zones in DuckDB with a bounding-box prefilter (a range join on
each zone's extent) before the exact ``ST_Intersects``/``ST_Contains`` test.
The zone of each point is then joined back to the plan lazily: no row reaches
Python.

Policies (:class:`AssignmentPolicy`), each deterministic:

* ``predicate``: ``intersects`` (default) assigns a point on a zone's boundary
  to that zone; ``contains`` assigns interior points only, so boundary points
  are unassigned.
* A point on the shared boundary of several zones (a *tie*, under
  ``intersects``) goes to the zone whose interior holds it, else to the lowest
  zone id; ties are counted.
* A point inside the interior of several zones (*overlapping* zones):
  ``error`` (default) refuses, ``first`` takes the lowest zone id, ``all``
  assigns it to each of them (its row is repeated per zone).
* A point in no zone, or without usable coordinates (missing, NaN, infinite),
  is *unassigned*; unassigned rows are counted, and more than
  ``max_unassigned`` (a fraction of the rows, 0 by default) leaves whatever
  needed the assignment not assessed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import duckdb
import polars as pl
from pydantic import Field

from qavf.engine.registry import RuleParams
from qavf.rules.helpers import fresh_internal, plural
from qavf.spatial.crs import transform_sql
from qavf.spatial.extension import fetch, register
from qavf.spatial.layers import LoadedLayer


class AssignmentPolicy(RuleParams):
    """How points on boundaries, in overlapping zones or in no zone are treated."""

    predicate: Literal["intersects", "contains"] = "intersects"
    """``intersects``: boundary points belong to the zone; ``contains``: interior only."""
    overlap: Literal["error", "first", "all"] = "error"
    """A point inside several zones: refuse, the lowest zone id, or every such zone."""
    max_unassigned: float = Field(default=0.0, ge=0, le=1)
    """Largest fraction of rows without a zone; more leaves the result not assessed."""


@dataclass(frozen=True)
class Assigned:
    """A plan with its zone column and what the assignment found, in rows."""

    plan: pl.LazyFrame
    """The input plan plus the zone column (null where unassigned)."""
    rows: int
    unassigned: int
    """Rows in no zone, or without usable coordinates."""
    ties: int
    """Rows on the boundary of several zones, resolved by the tie-break."""
    overlaps: int
    """Rows inside several zones (assigned by ``overlap``)."""

    def describe(self) -> str:
        parts = [f"{plural(self.unassigned, 'row')} in no zone"]
        if self.ties:
            parts.append(f"{plural(self.ties, 'row')} on a shared boundary (lowest zone id)")
        if self.overlaps:
            parts.append(f"{plural(self.overlaps, 'row')} inside overlapping zones")
        return ", ".join(parts)

    def check(self, policy: AssignmentPolicy, what: str) -> None:
        """Refuse more unassigned rows than ``policy.max_unassigned`` allows.

        Raises:
            ValueError: naming the count and the fraction.
        """
        if self.unassigned and self.unassigned > policy.max_unassigned * self.rows:
            share = self.unassigned / self.rows if self.rows else 1.0
            raise ValueError(
                f"{what}: {plural(self.unassigned, 'row')} of {self.rows:,} ({share:.2%}) fall "
                f"in no zone or lack usable coordinates, more than max_unassigned "
                f"({policy.max_unassigned:g})"
            )


def coordinate_keys(x: str, y: str) -> tuple[pl.Expr, pl.Expr]:
    """Float64 coordinates with NaN as null and -0.0 as 0.0 (exact join keys)."""
    return tuple(  # type: ignore[return-value]
        pl.col(column).cast(pl.Float64).fill_nan(None) + 0.0 for column in (x, y)
    )


def _located(con: Any, distinct: pl.DataFrame, geometry: bool, crs: str, target: str) -> None:
    """Bind the distinct points, transformed to ``target``, as ``qavf_located``."""
    register(con, "qavf_points", distinct)
    try:
        if geometry:
            point = "ST_GeomFromWKB(key)"
            usable = "key IS NOT NULL"
            kinds = fetch(
                con,
                f"SELECT count(*) AS n FROM qavf_points WHERE {usable} AND "
                f"CAST(ST_GeometryType({point}) AS VARCHAR) <> 'POINT'",
            )["n"][0]
            if kinds:
                raise ValueError(f"{int(kinds):,} value(s) of the point column are not points")
        else:
            point = "ST_Point(key_x, key_y)"
            usable = (
                "key_x IS NOT NULL AND key_y IS NOT NULL AND isfinite(key_x) AND isfinite(key_y)"
            )
        con.execute(
            "CREATE OR REPLACE TEMP TABLE qavf_located AS SELECT pid, pt, ST_X(pt) AS px, "
            f"ST_Y(pt) AS py FROM (SELECT pid, {transform_sql(point, crs, target)} AS pt "
            f"FROM qavf_points WHERE {usable})"
        )
    except duckdb.Error as exc:
        raise ValueError(f"points cannot be placed: {str(exc).splitlines()[0]}") from None
    finally:
        con.unregister("qavf_points")


def matches(
    con: Any,
    distinct: pl.DataFrame,
    layer: LoadedLayer,
    policy: AssignmentPolicy,
    *,
    crs: str,
    geometry: bool = False,
) -> pl.DataFrame:
    """Each distinct point's chosen zone(s): ``pid``, ``zone_order``, ``tie``, ``overlap``.

    ``distinct`` has a ``pid`` and either ``key_x``/``key_y`` (coordinates in
    ``crs``) or ``key`` (WKB points). Points in no zone have no row.

    Raises:
        ValueError: for points inside overlapping zones under ``overlap: error``,
            non-point geometries, or coordinates that cannot be transformed.
    """
    _located(con, distinct, geometry, crs, layer.crs)
    zones = layer.table
    test = "ST_Contains" if policy.predicate == "contains" else "ST_Intersects"
    found = fetch(
        con,
        f"SELECT l.pid, z.zone_order, ST_Contains(z.geom, l.pt) AS inside "
        f'FROM qavf_located l JOIN "{zones}" z ON l.px >= z.xmin AND l.px <= z.xmax '
        f"AND l.py >= z.ymin AND l.py <= z.ymax WHERE {test}(z.geom, l.pt)",
    ).with_columns(pl.col("pid").cast(pl.Int64), pl.col("zone_order").cast(pl.Int64))
    tally = found.group_by("pid").agg(
        pl.len().alias("n"), pl.col("inside").cast(pl.Int64).sum().alias("interior")
    )
    found = found.join(tally, on="pid").with_columns(
        ((pl.col("n") > 1) & (pl.col("interior") <= 1)).alias("tie"),
        (pl.col("interior") > 1).alias("overlap"),
    )
    overlapping = found.filter(pl.col("overlap"))["pid"].n_unique()
    if policy.overlap == "error" and overlapping:
        raise ValueError(
            f"{plural(overlapping, 'distinct point')} lie inside several zones (overlapping "
            "zones); set overlap: first (lowest zone id) or all (one row per zone)"
        )
    # Interior zones win over boundary contacts; among equals the lowest zone id.
    chosen = found.filter((pl.col("interior") == 0) | pl.col("inside")).sort(["pid", "zone_order"])
    single = chosen.filter(~pl.col("overlap")) if policy.overlap == "all" else chosen
    picked = single.unique(subset="pid", keep="first", maintain_order=True)
    if policy.overlap == "all":
        picked = pl.concat([picked, chosen.filter(pl.col("overlap"))]).sort(["pid", "zone_order"])
    return picked.select("pid", "zone_order", "tie", "overlap")


def assign(
    plan: pl.LazyFrame,
    con: Any,
    layer: LoadedLayer,
    policy: AssignmentPolicy,
    *,
    column: str,
    crs: str,
    collect: Callable[[pl.LazyFrame], pl.DataFrame],
    x: str | None = None,
    y: str | None = None,
    geometry: str | None = None,
) -> Assigned:
    """Add ``column``, the zone id of each row's point (``x``/``y`` or WKB ``geometry``).

    ``collect`` evaluates a plan over ``plan``'s source (natively). Unassigned
    rows keep a null zone; :meth:`Assigned.check` applies ``max_unassigned``.

    Raises:
        ValueError: as :func:`matches`, or for a ``column`` the plan already has.
    """
    schema = plan.collect_schema()
    if column in schema:
        raise ValueError(f"the zone column {column!r} already exists")
    taken = [*schema, column]
    names = {role: fresh_internal(f"zone_{role}", taken) for role in ("x", "y", "key", "rows")}
    if geometry is not None:
        keys = {names["key"]: pl.col(geometry)}
        roles = {names["key"]: "key"}
    else:
        assert x is not None and y is not None
        kx, ky = coordinate_keys(x, y)
        keys = {names["x"]: kx, names["y"]: ky}
        roles = {names["x"]: "key_x", names["y"]: "key_y"}
    distinct = collect(
        plan.group_by([key.alias(name) for name, key in keys.items()]).agg(
            pl.len().cast(pl.Int64).alias(names["rows"])
        )
    ).with_row_index("pid")
    distinct = distinct.with_columns(pl.col("pid").cast(pl.Int64))
    points = distinct.select("pid", *(pl.col(k).alias(r) for k, r in roles.items()))
    chosen = matches(con, points, layer, policy, crs=crs, geometry=geometry is not None)
    ids = pl.DataFrame(
        {
            "zone_order": pl.int_range(0, layer.ids.len(), eager=True, dtype=pl.Int64),
            column: layer.ids,
        }
    )
    mapping = chosen.join(ids, on="zone_order").join(distinct, on="pid")
    rows = int(distinct[names["rows"]].sum())
    flags = mapping.unique(subset="pid")
    assigned = int(flags[names["rows"]].sum())
    ties = int(flags.filter(pl.col("tie"))[names["rows"]].sum())
    overlaps = int(flags.filter(pl.col("overlap"))[names["rows"]].sum())
    joined = plan.with_columns(**keys).join(
        mapping.select(*keys, column).lazy(), on=list(keys), how="left"
    )
    return Assigned(joined.drop(list(keys)), rows, rows - assigned, ties, overlaps)
