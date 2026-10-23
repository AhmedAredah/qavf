"""Simplified zone outlines for maps: rings from a zone layer or grid cells, as SVG paths.

A report draws zones as plain SVG paths, so outlines are simplified in DuckDB
(``ST_SimplifyPreserveTopology`` with a tolerance of a fraction of the layer's
extent) and projected once into a view box ``width`` units wide, north up,
rounded to a tenth of a unit. Coordinates are drawn as they are (a geographic
CRS as plate carree), which is what a small overview map needs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from typing import Any

import polars as pl

from qavf.sources.sql import quote_identifier
from qavf.spatial.extension import fetch
from qavf.spatial.grid import Grid, label_positions

type Ring = list[tuple[float, float]]

SIMPLIFY_FRACTION = 1 / 500
"""Simplification tolerance as a fraction of the layer extent's larger side."""


class OutlineLimitExceeded(ValueError):
    """A complete outline exceeds its vertex budget; no partial map is returned."""

    def __init__(self, vertices: int, max_vertices: int) -> None:
        self.vertices = vertices
        self.max_vertices = max_vertices
        super().__init__(
            f"too large: {vertices:,} outline vertices after simplification "
            f"(map_vertices {max_vertices:,})"
        )


def layer_rings(
    con: Any, table: str, ids: pl.Series, *, max_vertices: int | None = None
) -> dict[Any, list[Ring]]:
    """Each zone's simplified polygon rings (outer and holes), keyed by zone id.

    ``table`` is a loaded zone layer (:func:`qavf.spatial.layers.load_zones`).
    When supplied, ``max_vertices`` is checked natively before any GeoJSON or
    Python coordinates are produced. An accepted layer is simplified again for
    transfer; a rejected one needs only the counting pass. The returned rings
    necessarily occupy space proportional to their vertices. Transfer buffers
    contain at most 64 features, in addition to DuckDB's native working memory.

    Raises:
        OutlineLimitExceeded: if the simplified layer exceeds ``max_vertices``.
        ValueError: when the layer has no outline.
    """
    name = quote_identifier(table)
    extent = fetch(
        con,
        f"SELECT min(xmin) AS x0, max(xmax) AS x1, min(ymin) AS y0, max(ymax) AS y1 FROM {name}",
    ).row(0)
    if any(value is None for value in extent):
        raise ValueError("the zones have no outline to draw")
    tolerance = float(max(extent[1] - extent[0], extent[3] - extent[2]) * SIMPLIFY_FRACTION)
    geometry = "ST_SimplifyPreserveTopology(geom, ?)"
    if max_vertices is not None:
        vertices = int(con.execute(
            f"SELECT coalesce(sum(ST_NPoints({geometry})), 0) FROM {name}", [tolerance]
        ).fetchone()[0])
        if vertices > max_vertices:
            raise OutlineLimitExceeded(vertices, max_vertices)
    query = con.sql(
        f"SELECT zone_order, ST_AsGeoJSON({geometry}) AS shape FROM {name} ORDER BY zone_order",
        params=[tolerance],
    )
    reader = (
        query.to_arrow_reader(64)
        if hasattr(query, "to_arrow_reader")
        else query.fetch_arrow_reader(64)
    )
    rings: dict[Any, list[Ring]] = {}
    try:
        for batch in reader:
            for row in range(batch.num_rows):
                order = batch.column(0)[row].as_py()
                shape = batch.column(1)[row].as_py()
                geometry_json = (
                    json.loads(shape) if shape else {"type": "Polygon", "coordinates": []}
                )
                polygons = (
                    [geometry_json["coordinates"]]
                    if geometry_json["type"] == "Polygon"
                    else geometry_json.get("coordinates", [])
                )
                rings[ids[order]] = [
                    [(float(point[0]), float(point[1])) for point in ring]
                    for polygon in polygons
                    for ring in polygon
                ]
    finally:
        reader.close()
    return rings


def grid_rings(
    labels: pl.Series, grid: Grid, *, max_vertices: int | None = None
) -> dict[Any, list[Ring]]:
    """The square of each grid cell label, optionally bounded before construction.

    Raises:
        OutlineLimitExceeded: if the distinct nonnull cells exceed the vertex budget.
        ValueError: for a malformed cell label (also checked for omitted maps).
    """
    positions = label_positions(labels)
    if max_vertices is not None:
        vertices = 4 * (labels.n_unique() - int(labels.null_count() > 0))
        if vertices > max_vertices:
            raise OutlineLimitExceeded(vertices, max_vertices)
    (ox, oy), size = grid.origin, grid.size
    rings: dict[Any, list[Ring]] = {}
    for label, ix, iy in zip(labels, positions["ix"], positions["iy"], strict=True):
        if label is None:
            continue
        x, y = ox + ix * size, oy + iy * size
        rings[label] = [[(x, y), (x + size, y), (x + size, y + size), (x, y + size)]]
    return rings


@dataclass(frozen=True, slots=True)
class ViewPaths:
    """Zones as SVG path data in a ``width x height`` view box."""

    width: float
    height: float
    paths: dict[str, str]
    centres: dict[str, list[float]]
    vertices: int


def view_paths(rings: Mapping[Any, list[Ring]], width: float = 1000.0) -> ViewPaths:
    """Project rings into the view box (north up) as ``M x y L ... Z`` path data.

    Two passes over the caller's rings retain only extrema and the final SVG
    strings. No flattened or projected coordinate copies are constructed.

    Raises:
        ValueError: for unusable coordinates or extent, or nonfinite/nonpositive width.
    """
    if not math.isfinite(width) or width <= 0:
        raise ValueError("the view width must be finite and positive")
    points = (point for zone in rings.values() for ring in zone for point in ring)
    first = next(points, None)
    if first is None:
        raise ValueError("the zones have no outline to draw")
    x0 = x1 = first[0]
    y0 = y1 = first[1]
    if not math.isfinite(x0) or not math.isfinite(y0):
        raise ValueError("the zones have nonfinite outline coordinates")
    vertices = 1
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("the zones have nonfinite outline coordinates")
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
        vertices += 1
    span = max(x1 - x0, y1 - y0)
    if not span > 0:
        raise ValueError("the zones have no extent to draw")
    if not math.isfinite(span):
        raise ValueError("the zones have no finite extent to draw")
    scale = width / span
    if not math.isfinite(scale) or not math.isfinite((x1 - x0) * scale):
        raise ValueError("the zones exceed the projection range")
    height = round((y1 - y0) * scale, 1) or 1.0
    if not math.isfinite(height):
        raise ValueError("the zones exceed the projection range")
    paths: dict[str, str] = {}
    centres: dict[str, list[float]] = {}
    for zone, zone_rings in rings.items():
        left = right = top = bottom = 0.0
        has_points = False
        with StringIO() as path:
            for ring in zone_rings:
                prefix = "M"
                for x, y in ring:
                    px = round((x - x0) * scale, 1)
                    py = round((y1 - y) * scale, 1)
                    path.write(f"{prefix}{px:g} {py:g}")
                    prefix = " L"
                    if not has_points:
                        has_points = True
                        left = right = px
                        top = bottom = py
                    else:
                        if px < left:
                            left = px
                        if px > right:
                            right = px
                        if py < top:
                            top = py
                        if py > bottom:
                            bottom = py
                if ring:
                    path.write("Z")
            if not has_points:
                continue
            paths[str(zone)] = path.getvalue()
        # Preserve ordinary rounding, avoiding overflow for a finite very wide view.
        centre_x = (left + right) / 2
        centre_y = (top + bottom) / 2
        if not math.isfinite(centre_x):
            centre_x = left + (right - left) / 2
        if not math.isfinite(centre_y):
            centre_y = top + (bottom - top) / 2
        centres[str(zone)] = [round(centre_x, 1), round(centre_y, 1)]
    return ViewPaths(width, height, paths, centres, vertices)
