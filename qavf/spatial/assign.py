"""Point-to-zone assignment: each point gets the id of the zone polygon it falls in.

The one point-in-polygon path of QAVF: the ``assign_zones`` comparison step and
GEO-001's ``within`` both call :func:`assign`. Admitted distinct points (with
their row counts) are collected once, then transformed and matched in batches.
DuckDB resolves boundaries and overlaps before selected mappings reach Polars;
raw candidate matches are never exported. The mapping is joined back lazily
and remains usable after the connection closes.

Limits bound distinct-point, candidate-pair and selected-mapping row counts,
not bytes or process RSS. Retained memory still scales with the admitted point
keys and accepted mapping, including variable-width WKB and zone ids. Native
grouping, joins, geometry evaluation and their spill storage follow the native
connection's resource policy; the final expanded dataset is not materialized
here.

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
from typing import Any, Literal, cast
from uuid import uuid4

import duckdb
import polars as pl
from pydantic import Field

from qavf.engine.registry import RuleParams
from qavf.rules.helpers import fresh_internal, plural
from qavf.sources.sql import quote_identifier
from qavf.spatial.crs import transform_sql
from qavf.spatial.extension import fetch, register
from qavf.spatial.layers import LoadedLayer


class AssignmentLimits(RuleParams):
    """Finite row/work admission for point assignment, rather than an RSS ceiling."""

    batch_size: int = Field(default=16_384, ge=1, strict=True)
    """Maximum distinct points transformed together and selected rows per Arrow batch."""
    max_points: int = Field(default=1_000_000, ge=1, strict=True)
    """Maximum distinct point keys retained; excess points refuse the whole assignment."""
    max_matches: int = Field(default=2_000_000, ge=1, strict=True)
    """Maximum selected point-zone mappings retained, including ``overlap: all`` expansion."""
    max_work: int = Field(default=20_000_000, ge=1, strict=True)
    """Maximum bounding-box candidate pairs admitted across batches, before exact tests."""


class AssignmentPolicy(AssignmentLimits):
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


def _located(
    con: Any,
    distinct: pl.DataFrame,
    geometry: bool,
    crs: str,
    target: str,
    *,
    points: str,
    located: str,
) -> None:
    """Bind one point batch in operation-local relations, transformed to ``target``."""
    register(con, points, distinct)
    source = quote_identifier(points)
    try:
        if geometry:
            point = "ST_GeomFromWKB(key)"
            usable = "key IS NOT NULL"
            kinds = fetch(
                con,
                f"SELECT count(*) AS n FROM {source} WHERE {usable} AND "
                f"CAST(ST_GeometryType({point}) AS VARCHAR) <> 'POINT'",
            )["n"][0]
            if kinds:
                raise ValueError(
                    f"at least {int(kinds):,} value(s) of the point column are not points"
                )
        else:
            point = "ST_Point(key_x, key_y)"
            usable = (
                "key_x IS NOT NULL AND key_y IS NOT NULL AND isfinite(key_x) AND isfinite(key_y)"
            )
        con.execute(
            f"CREATE TEMP TABLE {located} AS SELECT pid, pt, ST_X(pt) AS px, "
            f"ST_Y(pt) AS py FROM (SELECT pid, {transform_sql(point, crs, target)} AS pt "
            f"FROM {source} WHERE {usable})"
        )
    except duckdb.Error as exc:
        raise ValueError(f"points cannot be placed: {str(exc).splitlines()[0]}") from None
    finally:
        con.unregister(points)


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
    ``crs``) or ``key`` (WKB points). Point ids must be unique non-null Int64-compatible
    integers. Points in no zone have no row.

    Raises:
        ValueError: for points inside overlapping zones under ``overlap: error``,
            non-point geometries, or coordinates that cannot be transformed.
    """
    if distinct.height > policy.max_points:
        raise ValueError(f"point assignment exceeds max_points ({policy.max_points:,})")
    if (
        "pid" not in distinct.schema
        or not distinct.schema["pid"].is_integer()
        or distinct["pid"].null_count()
        or distinct["pid"].n_unique() != distinct.height
    ):
        raise ValueError("point ids must be unique non-null integers")
    try:
        distinct = distinct.with_columns(pl.col("pid").cast(pl.Int64, strict=True))
    except pl.exceptions.PolarsError:
        raise ValueError("point ids must fit Int64") from None
    namespace = f"qavf_assignment_{uuid4().hex}"
    points = f"{namespace}_points"
    located = quote_identifier(f"{namespace}_located")
    tally = quote_identifier(f"{namespace}_tally")
    zones = quote_identifier(layer.table)
    bbox = (
        f"FROM {located} l JOIN {zones} z "
        "ON l.px >= z.xmin AND l.px <= z.xmax AND l.py >= z.ymin AND l.py <= z.ymax"
    )
    test = "ST_Contains" if policy.predicate == "contains" else "ST_Intersects"
    inside = "TRUE" if policy.predicate == "contains" else "ST_Contains(z.geom, l.pt)"
    exact = f"{bbox} WHERE {test}(z.geom, l.pt)"
    pieces: list[pl.DataFrame] = []
    work = selected = overlapping = 0
    for batch in distinct.iter_slices(policy.batch_size):
        has_located = has_tally = False
        try:
            _located(con, batch, geometry, crs, layer.crs, points=points, located=located)
            has_located = True
            # Stop the range join after one excess pair; do not enumerate a rejected
            # Cartesian expansion merely to report its complete size.
            candidates = int(
                con.execute(
                    f"SELECT count(*) FROM (SELECT 1 {bbox} LIMIT {policy.max_work - work + 1})"
                ).fetchone()[0]
            )
            work += candidates
            if work > policy.max_work:
                raise ValueError(f"point assignment exceeds max_work ({policy.max_work:,})")
            con.execute(
                f"CREATE TEMP TABLE {tally} AS "
                "SELECT pid, count(*)::BIGINT AS n, "
                "sum(CASE WHEN inside THEN 1 ELSE 0 END)::BIGINT AS interior, "
                "min(zone_order)::BIGINT AS first_any, "
                "min(zone_order) FILTER (WHERE inside)::BIGINT AS first_interior "
                f"FROM (SELECT l.pid, z.zone_order, {inside} AS inside {exact}) "
                "GROUP BY pid"
            )
            has_tally = True
            counts = con.execute(
                "SELECT count(*) FILTER (WHERE interior > 1), "
                + (
                    "coalesce(sum(CASE WHEN interior > 1 THEN interior ELSE 1 END), 0)"
                    if policy.overlap == "all"
                    else "count(*)"
                )
                + f" FROM {tally}"
            ).fetchone()
            overlapping += int(counts[0])
            if policy.overlap == "error" and overlapping:
                # Preserve the total distinct-overlap diagnostic without exporting
                # a partial mapping. Later batches can still refuse invalid inputs.
                pieces.clear()
                continue
            selected += int(counts[1])
            if selected > policy.max_matches:
                raise ValueError(f"point assignment exceeds max_matches ({policy.max_matches:,})")
            sql = (
                "SELECT pid::BIGINT AS pid, coalesce(first_interior, first_any) AS zone_order, "
                "(n > 1 AND interior <= 1) AS tie, (interior > 1) AS overlap "
                f"FROM {tally}"
            )
            if policy.overlap == "all":
                sql += (
                    " WHERE interior <= 1 UNION ALL "
                    "SELECT l.pid::BIGINT, z.zone_order::BIGINT, FALSE AS tie, TRUE AS overlap "
                    f"{bbox} JOIN {tally} t ON t.pid = l.pid AND t.interior > 1 "
                    "WHERE ST_Contains(z.geom, l.pt)"
                )
            query = con.execute(sql + " ORDER BY pid, zone_order")
            reader = (
                query.to_arrow_reader(policy.batch_size)
                if hasattr(query, "to_arrow_reader")
                else query.fetch_record_batch(policy.batch_size)
            )
            try:
                pieces.extend(cast(pl.DataFrame, pl.from_arrow(chunk)) for chunk in reader)
            finally:
                reader.close()
        finally:
            if has_tally:
                con.execute(f"DROP TABLE {tally}")
            if has_located:
                con.execute(f"DROP TABLE {located}")
    if policy.overlap == "error" and overlapping:
        raise ValueError(
            f"{plural(overlapping, 'distinct point')} lie inside several zones (overlapping "
            "zones); set overlap: first (lowest zone id) or all (one row per zone)"
        )
    if not pieces:
        return pl.DataFrame(
            schema={
                "pid": pl.Int64,
                "zone_order": pl.Int64,
                "tie": pl.Boolean,
                "overlap": pl.Boolean,
            }
        )
    result = pl.concat(pieces, rechunk=False)
    return result if distinct["pid"].is_sorted() else result.sort(["pid", "zone_order"])


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

    ``collect`` evaluates the distinct-point aggregation once, exporting at most
    ``max_points + 1`` keys. A limit breach refuses the entire assignment, never
    truncates it. Unassigned rows keep a null zone; :meth:`Assigned.check` applies
    ``max_unassigned``. The retained mapping owns its buffers independently of
    the spatial connection.

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
        plan.group_by([key.alias(name) for name, key in keys.items()])
        .agg(pl.len().cast(pl.Int64).alias(names["rows"]))
        .limit(policy.max_points + 1)
    )
    if distinct.height > policy.max_points:
        raise ValueError(f"point assignment exceeds max_points ({policy.max_points:,})")
    distinct = distinct.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int64).alias("pid"))
    points = distinct.select("pid", *(pl.col(k).alias(r) for k, r in roles.items()))
    chosen = matches(con, points, layer, policy, crs=crs, geometry=geometry is not None)
    ids = pl.DataFrame(
        {
            "zone_order": pl.int_range(0, layer.ids.len(), eager=True, dtype=pl.Int64),
            column: layer.ids,
        }
    )
    rows = int(distinct[names["rows"]].sum())
    flags = (
        chosen.unique(subset="pid")
        .select("pid", "tie", "overlap")
        .join(distinct.select("pid", names["rows"]), on="pid")
    )
    assigned = int(flags[names["rows"]].sum())
    ties = int(flags.filter(pl.col("tie"))[names["rows"]].sum())
    overlaps = int(flags.filter(pl.col("overlap"))[names["rows"]].sum())
    mapping = (
        chosen.select("pid", "zone_order")
        .join(ids, on="zone_order")
        .join(distinct.select("pid", *keys), on="pid")
        .select(*keys, column)
    )
    joined = plan.with_columns(**keys).join(mapping.lazy(), on=list(keys), how="left")
    return Assigned(joined.drop(list(keys)), rows, rows - assigned, ties, overlaps)
