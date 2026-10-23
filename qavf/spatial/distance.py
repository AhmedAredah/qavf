"""Distance between coordinates: haversine for longitude/latitude, Euclidean otherwise.

Each formula is written once with NumPy ufuncs, so the same function evaluates
Python floats, NumPy arrays (neighbour weights) and Polars expressions (moved
features, computed natively: Polars dispatches NumPy ufuncs to its own
kernels). The great-circle distance uses the haversine formula on a sphere of
radius :data:`EARTH_RADIUS_M`, the IUGG mean Earth radius; it is well defined
for every pair, antipodes included (the haversine is clamped to 1 against
rounding). It is a sphere, not the WGS 84 ellipsoid: geodesic distances differ
by up to about 0.5 %.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EARTH_RADIUS_M = 6_371_008.8
"""Mean Earth radius ``R1`` of the IUGG, in metres."""


def haversine(lon1: Any, lat1: Any, lon2: Any, lat2: Any, radius: float = EARTH_RADIUS_M) -> Any:
    """Great-circle distance in metres (in ``radius`` units) between degree coordinates."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    half_lat = np.sin((phi2 - phi1) / 2)
    half_lon = np.sin(np.radians(lon2 - lon1) / 2)
    h = half_lat * half_lat + np.cos(phi1) * np.cos(phi2) * half_lon * half_lon
    return 2 * radius * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


def euclidean(x1: Any, y1: Any, x2: Any, y2: Any) -> Any:
    """Straight-line distance in the coordinates' own units."""
    dx, dy = x2 - x1, y2 - y1
    return np.hypot(dx, dy)


def distance(x1: Any, y1: Any, x2: Any, y2: Any, *, geographic: bool) -> Any:
    """:func:`haversine` metres for degree coordinates, else :func:`euclidean` units."""
    return haversine(x1, y1, x2, y2) if geographic else euclidean(x1, y1, x2, y2)
