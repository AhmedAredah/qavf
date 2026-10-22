"""The ``assign_zones`` alignment step: give each side's rows a zone before any level runs.

Zones come from one of three places, each through :mod:`qavf.spatial`:

* a zone layer (``layer``, or ``baseline_layer``/``candidate_layer`` when the
  zones were re-drawn between versions): each row's point (``x``/``y`` columns
  or a WKB ``geometry`` column, in ``crs``) is assigned by point-in-polygon
  (:func:`qavf.spatial.assign.assign`), which needs DuckDB's spatial extension;
* a ``grid``: each point's cell label, natively (the extension only when the
  grid's CRS differs from the points');
* an attribute zone column (``zones: taz``), read as it is: county, TAZ or link
  ids need no geometry and no extension.

A ``crosswalk`` then maps one side's zone ids to the other's (``crosswalk_side``,
the baseline by default): 2010 zones to 2020 zones; or, with ``crosswalk_side:
both``, both sides' zones to their parents (links to zones, zones to counties).
With an attribute ``zones`` column and another ``column``, a side the crosswalk
does not rewrite keeps its own ``column`` when it has one. A split
zone's rows repeat once per target zone, with the ``apportion`` columns
multiplied by the split weight (and the weight itself in ``weight_column``).

What the step did is recorded with ``context.note``: rows in no zone (dropped,
up to ``max_unassigned``), rows on shared boundaries or in overlapping zones,
repaired geometries, unmapped and split zones. Without the extension, the
step refuses with the reason, so every level is not assessed, never passed.
Once it ran, ``groups``, ``temporal``, ``records`` and ``spatial`` group by
the zone column like any other column: ``assign_zones`` with ``temporal``
``group_by: [zone]`` compares each zone's series bucket by bucket.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl
from pydantic import model_validator

from qavf.compare.align import (
    AlignmentContext,
    AlignmentError,
    AlignmentStep,
    register_alignment_step,
)
from qavf.compare.spec import Side, SideSelection
from qavf.engine.registry import RuleParams
from qavf.rules.helpers import fresh_internal, plural
from qavf.spatial.assign import Assigned, AssignmentPolicy, assign
from qavf.spatial.crs import DEFAULT_CRS, Crs
from qavf.spatial.extension import SpatialUnavailable, spatial_connection
from qavf.spatial.grid import Grid, assign_cells
from qavf.spatial.hierarchy import SOURCE, Crosswalk, apply_crosswalk, crosswalk_frame, split
from qavf.spatial.layers import ZoneLayer, load_zones


class AssignZonesParams(AssignmentPolicy):
    """Parameters of the ``assign_zones`` step."""

    column: str = "zone"
    """The zone column added (or, with ``zones``, the column the crosswalk rewrites)."""
    x: str | None = None
    """Longitude or easting column of each row's point."""
    y: str | None = None
    """Latitude or northing column."""
    geometry: str | None = None
    """A WKB point column instead of ``x``/``y``."""
    crs: Crs = DEFAULT_CRS
    """CRS of the points; transformed (x first) to the layer's or grid's CRS."""
    baseline_crs: Crs | None = None
    """The baseline's points' CRS when it differs from ``crs``."""
    candidate_crs: Crs | None = None
    layer: ZoneLayer | None = None
    """Zone polygons of both sides."""
    baseline_layer: ZoneLayer | None = None
    """The baseline's zones, when the zones were re-drawn (with ``candidate_layer``)."""
    candidate_layer: ZoneLayer | None = None
    grid: Grid | None = None
    """Regular grid cells instead of polygons."""
    zones: str | None = None
    """An attribute zone column already in the data (county, TAZ, link id)."""
    crosswalk: Crosswalk | None = None
    """Maps ``crosswalk_side``'s zone ids to the other side's (or both sides' zones to
    their parents)."""
    crosswalk_side: SideSelection = "baseline"
    """The side whose zones the crosswalk rewrites, or ``both`` (a hierarchy: links to
    zones, zones to counties)."""
    apportion: tuple[str, ...] = ()
    """Columns a split zone's rows divide by the split weight (e.g. trips)."""
    weight_column: str | None = None
    """Add each row's split weight (1 when not split), e.g. for ``sum(weight)``."""
    unmapped: Literal["error", "drop"] = "error"
    """Rows whose zone the crosswalk lacks: refuse, or drop them (recorded)."""

    @model_validator(mode="after")
    def _one_source(self) -> AssignZonesParams:
        per_side = (self.baseline_layer is None, self.candidate_layer is None)
        if per_side[0] != per_side[1]:
            raise ValueError("'baseline_layer' and 'candidate_layer' go together")
        sources = [self.layer is not None, not per_side[0], self.grid is not None]
        sources.append(self.zones is not None)
        if sum(sources) != 1:
            raise ValueError(
                "assign_zones needs exactly one zone source: 'layer', 'baseline_layer' with "
                "'candidate_layer', 'grid' or an attribute column 'zones'"
            )
        if self.zones is not None:
            if self.crosswalk is None:
                raise ValueError("an attribute zone column needs a 'crosswalk' to apply")
            if self.x is not None or self.y is not None or self.geometry is not None:
                raise ValueError("attribute zones take no point columns")
        elif (self.geometry is None) == (self.x is None or self.y is None):
            raise ValueError("points are 'x' and 'y' columns, or one WKB 'geometry' column")
        if self.grid is not None and self.geometry is not None:
            raise ValueError("grid cells are assigned from 'x' and 'y' columns")
        if (self.apportion or self.weight_column) and self.crosswalk is None:
            raise ValueError("'apportion' and 'weight_column' apply a crosswalk's splits")
        return self

    def side_crs(self, side: Side) -> str:
        own = self.baseline_crs if side == "baseline" else self.candidate_crs
        return own or self.crs

    def side_layer(self, side: Side) -> ZoneLayer | None:
        """The zone layer of ``side`` (``None`` for grids and attribute zones)."""
        own = self.baseline_layer if side == "baseline" else self.candidate_layer
        return own or self.layer

    def crosswalked(self, side: Side) -> bool:
        """Whether the crosswalk rewrites ``side``'s zones."""
        return self.crosswalk is not None and self.crosswalk_side in (side, "both")


@register_alignment_step
class AssignZonesStep(AlignmentStep):
    """Each row's zone id from a layer, a grid or an attribute column, plus a crosswalk."""

    name: ClassVar[str] = "assign_zones"
    description: ClassVar[str] = (
        "Each side's points assigned to zone ids (polygons of a layer, per side if zones "
        "were re-drawn, or grid cells) or an attribute zone column read as it is, then an "
        "optional crosswalk; unassigned, boundary and overlap counts are recorded."
    )
    Params: ClassVar[type[RuleParams]] = AssignZonesParams
    params: AssignZonesParams

    def datasets(self) -> frozenset[str]:
        params = self.params
        found: set[str] = set()
        for layer in (params.layer, params.baseline_layer, params.candidate_layer):
            if layer is not None:
                found |= layer.datasets()
        if params.crosswalk is not None:
            found |= params.crosswalk.datasets()
        return frozenset(found)

    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        params = self.params
        if context.collect is None:
            raise context.refuse("assign_zones needs a native collector for the side")
        try:
            if params.zones is not None:
                plan = self._attribute(plan, context)
            else:
                plan = self._assigned(plan, context)
            if params.crosswalked(context.side):
                plan = self._crosswalk(plan, context)
            elif params.weight_column is not None:
                # The other side's rows are whole: weight 1, so sum(weight) is comparable.
                if params.weight_column in plan.collect_schema():
                    raise ValueError(f"the weight column {params.weight_column!r} already exists")
                plan = plan.with_columns(pl.lit(1.0).alias(params.weight_column))
        except AlignmentError:
            raise
        except (SpatialUnavailable, ValueError) as exc:
            raise context.refuse(f"assign_zones: {exc}") from None
        return plan

    def _attribute(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        params = self.params
        assert params.zones is not None
        schema = plan.collect_schema()
        rewritten = params.crosswalked(context.side)
        if params.column != params.zones and params.column in schema:
            # The other side may already hold the target zones under the new name.
            if rewritten:
                raise ValueError(f"the zone column {params.column!r} already exists")
            return plan
        if params.zones not in schema:
            raise ValueError(f"lacks the zone column {params.zones!r}")
        if params.column != params.zones:
            plan = plan.with_columns(pl.col(params.zones).alias(params.column))
        return plan

    def _assigned(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        params = self.params
        assert context.collect is not None
        columns = [c for c in (params.x, params.y, params.geometry) if c is not None]
        if absent := [c for c in columns if c not in plan.collect_schema()]:
            raise ValueError(f"lacks point column(s) {', '.join(absent)}")
        crs = params.side_crs(context.side)
        layer = params.side_layer(context.side)
        if layer is None:
            assert params.grid is not None and params.x is not None and params.y is not None
            assigned = assign_cells(
                plan,
                params.grid,
                x=params.x,
                y=params.y,
                crs=crs,
                column=params.column,
                collect=context.collect,
                connection=lambda: spatial_connection(context.data),
            )
        else:
            with spatial_connection(context.data) as con:
                loaded = load_zones(con, context.data, layer)
                for note in loaded.notes:
                    context.note(f"assign_zones: {note}")
                assigned = assign(
                    plan,
                    con,
                    loaded,
                    params,
                    column=params.column,
                    crs=crs,
                    collect=context.collect,
                    x=params.x,
                    y=params.y,
                    geometry=params.geometry,
                )
        # The plan joins a Polars mapping of the distinct points: no connection is kept.
        return self._kept(assigned, context).plan

    def _kept(self, assigned: Assigned, context: AlignmentContext) -> Assigned:
        """Apply ``max_unassigned``, record the counts and drop unassigned rows."""
        params = self.params
        assigned.check(params, "points")
        if assigned.unassigned or assigned.ties or assigned.overlaps:
            dropped = "; rows in no zone were dropped" if assigned.unassigned else ""
            context.note(f"assign_zones: {assigned.describe()}{dropped}")
        column = pl.col(params.column)
        return Assigned(
            assigned.plan.filter(column.is_not_null()),
            assigned.rows,
            assigned.unassigned,
            assigned.ties,
            assigned.overlaps,
        )

    def _crosswalk(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        params = self.params
        assert params.crosswalk is not None
        frame = crosswalk_frame(context.data, params.crosswalk)
        schema = plan.collect_schema()
        column = params.column
        if absent := [c for c in params.apportion if c not in schema]:
            raise ValueError(f"lacks apportioned column(s) {', '.join(absent)}")
        if params.weight_column is not None and params.weight_column in schema:
            raise ValueError(f"the weight column {params.weight_column!r} already exists")
        splits = split(frame)
        if splits and not (params.apportion or params.weight_column):
            raise ValueError(
                f"crosswalk {params.crosswalk.label()!r} splits zones; set 'apportion' (columns "
                "to divide) or 'weight_column', or every split row would count in full"
            )
        try:
            sources = frame[SOURCE].cast(schema[column], strict=True)
        except Exception:
            raise ValueError(
                f"crosswalk ids cannot all be represented as {schema[column]}"
            ) from None
        known = pl.col(column).is_in(sources.implode())
        unmapped = pl.col(column).is_not_null() & ~known
        if params.unmapped == "error":
            context.require_none(
                plan,
                unmapped,
                f"{{count}} row(s) have a {column!r} zone the crosswalk "
                f"{params.crosswalk.label()!r} lacks; add it, or set unmapped: drop",
            )
        else:
            dropped = context.count(plan.filter(unmapped))
            if dropped:
                context.note(
                    f"assign_zones dropped {plural(dropped, 'row')} whose zone the crosswalk lacks"
                )
            plan = plan.filter(~unmapped)
        taken = fresh_internal("split_weight", [*schema, params.weight_column or ""])
        weighted = (
            plan.with_columns(pl.lit(1.0).alias(params.weight_column))
            if params.weight_column is not None
            else plan
        )
        apportion = [*params.apportion, *([params.weight_column] if params.weight_column else [])]
        mapped = apply_crosswalk(weighted, column, frame, apportion, taken)
        if splits:
            parts = frame.group_by(SOURCE).len().filter(pl.col("len") > 1).height
            context.note(
                f"assign_zones: crosswalk {params.crosswalk.label()!r} split "
                f"{plural(parts, 'zone')} into several; their rows repeat per target zone"
            )
        return mapped
