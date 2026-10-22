"""Coordinate reference systems: declared per dataset or zone layer, transformed natively.

A CRS is written ``AUTHORITY:CODE`` (``EPSG:4326``, ``EPSG:32614``,
``OGC:CRS84``); longitude/latitude columns default to ``EPSG:4326``.
Coordinates are always read **x first** (longitude, easting): transforms use
PROJ's ``always_xy`` semantics through ``ST_Transform(..., always_xy := true)``,
so ``EPSG:4326`` data in (longitude, latitude) order is never flipped by the
authority's latitude-first axis order.

Distance follows the CRS: great-circle (haversine) metres for geographic CRSs,
Euclidean CRS units for projected ones (:mod:`qavf.spatial.distance`). The
geographic CRSs are those listed in :data:`GEOGRAPHIC`; any other CRS is
treated as projected.

A zone layer read from a file that declares its own CRS (GeoParquet ``geo``
metadata, a shapefile's ``.prj``, a GeoJSON ``crs``) must agree with the
configured one; a mismatch is reported, never silently transformed.
"""

from __future__ import annotations

import json
import re
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
    """Whether coordinates in ``crs`` are degrees (distances are then great-circle)."""
    return normalize_crs(crs) in GEOGRAPHIC


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
