"""Coordinate reference systems: declared per dataset or zone layer, transformed natively.

A CRS is written ``AUTHORITY:CODE`` (``EPSG:4326``, ``EPSG:32614``,
``OGC:CRS84``); longitude/latitude columns default to ``EPSG:4326``.
Coordinates are always read **x first** (longitude, easting): transforms use
PROJ's ``always_xy`` semantics through ``ST_Transform(..., always_xy := true)``,
so ``EPSG:4326`` data in (longitude, latitude) order is never flipped by the
authority's latitude-first axis order.

Distance follows the CRS: great-circle (haversine) metres for geographic CRSs,
Euclidean CRS units for projected ones (:mod:`qavf.spatial.distance`). The
common degree-based geographic and WGS 84 projected CRSs need no additional
dependency. Other identifiers are resolved through the optional PROJ database
(``qavf[spatial]``), or rejected when their type cannot be established. Unknown
identifiers never silently select a distance formula.

A zone layer read from a file that declares its own CRS (GeoParquet ``geo``
metadata, a shapefile's ``.prj``, a GeoJSON ``crs``) must agree with the
configured one; a mismatch is reported, never silently transformed.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import Annotated, Any

from pydantic import BeforeValidator

from qavf.sources.sql import sql_literal

DEFAULT_CRS = "EPSG:4326"
"""The CRS of longitude/latitude columns unless declared."""
GEOGRAPHIC = frozenset(
    {"EPSG:4326", "OGC:CRS84", "EPSG:4269", "EPSG:4258", "EPSG:4283", "EPSG:4617", "EPSG:4167"}
)
"""Geographic (degree) CRSs: WGS 84, CRS84, NAD83, ETRS89, GDA94, NAD83(CSRS), NZGD2000."""
_SAME_AS_WGS84 = frozenset({"EPSG:4326", "OGC:CRS84"})
"""Identical under x-first axis order."""
_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):([A-Za-z0-9_.]+)$")


def normalize_crs(value: Any) -> str:
    """``AUTHORITY:CODE`` in upper case (``4326`` and ``epsg:4326`` mean ``EPSG:4326``).

    Raises:
        ValueError: for anything else.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        value = f"EPSG:{value}"
    if not isinstance(value, str) or not (match := _PATTERN.match(value.strip())):
        raise ValueError(f"a CRS is written AUTHORITY:CODE (e.g. EPSG:4326), not {value!r}")
    authority, code = match.groups()
    return f"{authority.upper()}:{code.upper()}"


Crs = Annotated[str, BeforeValidator(normalize_crs)]
"""A configured CRS, normalized when the configuration loads."""


def is_geographic(crs: str) -> bool:
    """Resolve degree-based geographic or projected coordinates, failing closed.

    Other angular units, geocentric/vertical CRSs, and unresolvable identifiers
    are rejected rather than interpreted as planar coordinates. Resolution is
    cached and does not occur in coordinate loops.
    """
    return _is_geographic(normalize_crs(crs))


@lru_cache(maxsize=256)
def _is_geographic(crs: str) -> bool:
    if crs in GEOGRAPHIC:
        return True
    # Authoritative WGS 84 projected definitions used by the dependency-free
    # grid API: Web/World Mercator and UTM zones 1-60 in both hemispheres.
    authority, code = crs.split(":", 1)
    if authority == "EPSG" and code.isdecimal():
        number = int(code)
        if number in (3857, 3395) or 32601 <= number <= 32660 or 32701 <= number <= 32760:
            return False
    try:
        from pyproj import CRS
        from pyproj.exceptions import CRSError
    except ImportError:
        raise ValueError(
            f"cannot establish the coordinate type of {crs}; install qavf[spatial] "
            "for authoritative CRS resolution"
        ) from None
    try:
        resolved = CRS.from_user_input(crs)
    except CRSError as exc:
        raise ValueError(f"cannot resolve CRS {crs}") from exc
    if resolved.is_projected:
        return False
    if resolved.is_geographic:
        axes = resolved.axis_info[:2]
        if len(axes) == 2 and all(
            math.isclose(axis.unit_conversion_factor, math.pi / 180, rel_tol=1e-12)
            for axis in axes
        ):
            return True
        raise ValueError(f"geographic CRS {crs} must use degrees for haversine distances")
    raise ValueError(f"CRS {crs} is neither a degree-based geographic nor projected CRS")


def same_crs(a: str, b: str) -> bool:
    """Whether two CRSs place x-first coordinates identically."""
    a, b = normalize_crs(a), normalize_crs(b)
    return a == b or {a, b} <= _SAME_AS_WGS84


def distance_unit(crs: str) -> str:
    """``metres`` for geographic CRSs (haversine), else ``CRS units``."""
    return "metres" if is_geographic(crs) else f"{normalize_crs(crs)} units"


def transform_sql(geometry: str, source: str, target: str) -> str:
    """SQL moving ``geometry`` from ``source`` to ``target`` (x first), or as it is."""
    if same_crs(source, target):
        return geometry
    return (
        f"ST_Transform({geometry}, {sql_literal(normalize_crs(source))}, "
        f"{sql_literal(normalize_crs(target))}, always_xy := true)"
    )


def projjson_crs(projjson: Any) -> str | None:
    """``AUTHORITY:CODE`` of a PROJJSON CRS (text or parsed), when it carries an id."""
    if isinstance(projjson, str):
        try:
            projjson = json.loads(projjson)
        except ValueError:
            return None
    if not isinstance(projjson, dict):
        return None
    ident = projjson.get("id")
    if isinstance(ident, dict) and "authority" in ident and "code" in ident:
        return normalize_crs(f"{ident['authority']}:{ident['code']}")
    return None


def check_declared(configured: str, found: str | None, what: str) -> None:
    """Refuse a file whose own CRS differs from the configured one.

    Raises:
        ValueError: naming both CRSs.
    """
    if found is not None and not same_crs(configured, found):
        raise ValueError(
            f"{what} declares the CRS {found}, but {configured} is configured; set 'crs' to "
            "the file's CRS (points are transformed to the layer's CRS)"
        )
