"""Simplified zone outlines for maps: rings from a zone layer or grid cells, as SVG paths.

A report draws zones as plain SVG paths, so outlines are simplified in DuckDB
(``ST_SimplifyPreserveTopology`` with a tolerance of a fraction of the layer's
extent) and projected once into a view box ``width`` units wide, north up,
rounded to a tenth of a unit. Coordinates are drawn as they are (a geographic
CRS as plate carree), which is what a small overview map needs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from qavf.spatial.extension import fetch
from qavf.spatial.grid import Grid, label_positions

type Ring = list[tuple[float, float]]

SIMPLIFY_FRACTION = 1 / 500
"""Simplification tolerance as a fraction of the layer extent's larger side."""


def layer_rings(con: Any, table: str, ids: pl.Series) -> dict[Any, list[Ring]]:
    """Each zone's simplified polygon rings (outer and holes), keyed by zone id.

    ``table`` is a loaded zone layer (:func:`qavf.spatial.layers.load_zones`).
    """
    name = f'"{table}"'
    extent = fetch(
        con,
        f"SELECT min(xmin) AS x0, max(xmax) AS x1, min(ymin) AS y0, max(ymax) AS y1 FROM {name}",
    ).row(0)
    tolerance = max(extent[1] - extent[0], extent[3] - extent[2]) * SIMPLIFY_FRACTION
    shapes = fetch(
        con,
        "SELECT zone_order, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, ?)) AS shape "
        f"FROM {name} ORDER BY zone_order",
        [float(tolerance)],
    )
    rings: dict[Any, list[Ring]] = {}
    for order, shape in shapes.iter_rows():
        geometry = json.loads(shape) if shape else {"type": "Polygon", "coordinates": []}
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry.get("coordinates", [])
        )
        rings[ids[order]] = [
            [(float(point[0]), float(point[1])) for point in ring]
            for polygon in polygons
            for ring in polygon
        ]
    return rings


def grid_rings(labels: pl.Series, grid: Grid) -> dict[Any, list[Ring]]:
    """The square of each grid cell label."""
    positions = label_positions(labels)
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

    Raises:
        ValueError: when the rings have no extent.
    """
    points = [point for zone in rings.values() for ring in zone for point in ring]
    if not points:
        raise ValueError("the zones have no outline to draw")
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y1 = min(xs), max(ys)
    span = max(max(xs) - x0, y1 - min(ys))
    if not span > 0:
        raise ValueError("the zones have no extent to draw")
    scale = width / span
    height = round((y1 - min(ys)) * scale, 1) or 1.0
    paths: dict[str, str] = {}
    centres: dict[str, list[float]] = {}
    for zone, zone_rings in rings.items():
        parts = []
        projected = []
        for ring in zone_rings:
            coordinates = [
                (round((x - x0) * scale, 1), round((y1 - y) * scale, 1)) for x, y in ring
            ]
            projected.extend(coordinates)
            parts.append(
                "M" + " L".join(f"{x:g} {y:g}" for x, y in coordinates) + "Z" if coordinates else ""
            )
        if not projected:
            continue
        paths[str(zone)] = "".join(parts)
        left, right = min(p[0] for p in projected), max(p[0] for p in projected)
        top, bottom = min(p[1] for p in projected), max(p[1] for p in projected)
        centres[str(zone)] = [round((left + right) / 2, 1), round((top + bottom) / 2, 1)]
    return ViewPaths(width, height, paths, centres, len(points))
