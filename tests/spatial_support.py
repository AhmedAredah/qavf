"""Hand-built spatial fixtures for the spatial tests: polygons, WKB and layer files.

Geometry is written here without QAVF's code: WKB by ``struct``, GeoJSON by
``json``, WKT as text, so the tests' expectations never come from the code
under test.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

Ring = Sequence[tuple[float, float]]


def square(x: float, y: float, size: float = 1.0) -> list[tuple[float, float]]:
    """A closed counter-clockwise ring of the square with corner ``(x, y)``."""
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]


def wkt(ring: Ring) -> str:
    return "POLYGON((" + ", ".join(f"{x!r} {y!r}" for x, y in ring) + "))"


def wkb_polygon(ring: Ring) -> bytes:
    """Little-endian WKB of a one-ring polygon."""
    points = b"".join(struct.pack("<dd", x, y) for x, y in ring)
    return struct.pack("<BII", 1, 3, 1) + struct.pack("<I", len(ring)) + points


def wkb_point(x: float, y: float) -> bytes:
    return struct.pack("<BIdd", 1, 1, x, y)


def geojson(path: Path, zones: dict[str, Ring], crs: str | None = None) -> Path:
    features = [
        {
            "type": "Feature",
            "properties": {"zone": zone},
            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]},
        }
        for zone, ring in zones.items()
    ]
    document: dict = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        document["crs"] = {"type": "name", "properties": {"name": crs}}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def geoparquet(path: Path, zones: dict[str, Ring], epsg: int | None = None) -> Path:
    """GeoParquet 1.0: WKB geometry and ``geo`` metadata (CRS84 unless ``epsg``)."""
    table = pa.table(
        {
            "zone": list(zones),
            "geometry": pa.array([wkb_polygon(ring) for ring in zones.values()], pa.binary()),
        }
    )
    column: dict = {"encoding": "WKB", "geometry_types": ["Polygon"]}
    if epsg is not None:
        column["crs"] = {"id": {"authority": "EPSG", "code": epsg}}
    geo = {"version": "1.0.0", "primary_column": "geometry", "columns": {"geometry": column}}
    table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
    pq.write_table(table, path)
    return path


def wkt_frame(zones: dict[str, Ring], id_column: str = "zone") -> pl.DataFrame:
    return pl.DataFrame({id_column: list(zones), "geometry": [wkt(r) for r in zones.values()]})


def grid_zones(n: int, size: float = 1.0) -> dict[str, list[tuple[float, float]]]:
    """``n x n`` unit squares named ``z<i>_<j>`` (column ``i``, row ``j``)."""
    return {f"z{i}_{j}": square(i * size, j * size, size) for i in range(n) for j in range(n)}


A_AND_L = {
    # A: the square [0, 2]^2; L: the L-shaped polygon wrapping it above and to the right.
    "A": [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
    "L": [(2, 0), (4, 0), (4, 4), (0, 4), (0, 2), (2, 2), (2, 0)],
}
OVERLAPPING = {**A_AND_L, "O": [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)]}
"""A, L and O, a square inside A's interior (an overlap)."""
