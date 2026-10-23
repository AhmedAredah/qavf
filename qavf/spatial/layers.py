"""Zone layers: polygons with an id, from a configured dataset or a file.

A layer is written ``{dataset: zones, id: taz}`` (any configured dataset with an
id column and a WKB or WKT geometry column, e.g. a GeoParquet file or a query)
or ``{path: zones.geojson, id: taz}``: GeoParquet (``.parquet``, geometry as
WKB) is read by Polars, every other file (GeoJSON, shapefile, GeoPackage, ...)
by DuckDB's ``ST_Read``. ``crs`` declares the layer's CRS (``EPSG:4326`` by
default); a file that declares a different one is refused.

Loading for geometry (:func:`load_zones`) validates every zone: ids must be
present and unique, geometries present, polygonal and valid (``ST_IsValid``).
An invalid geometry refuses the layer, so whatever needed it is not assessed,
unless ``repair: true`` replaces it with ``ST_MakeValid`` (kept polygonal),
which is recorded. Zone ids alone (:func:`zone_ids`, e.g. the expected zones of
a coverage check) need no extension for datasets and GeoParquet files.

Zones are ordered by id: the *zone order* is the id's rank, which makes every
tie-break deterministic whatever the file order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
from pydantic import model_validator

from qavf.engine.registry import RuleParams
from qavf.rules.helpers import plural
from qavf.sources.sql import quote_identifier, sql_literal
from qavf.spatial.crs import DEFAULT_CRS, Crs, check_declared, normalize_crs, projjson_crs
from qavf.spatial.extension import fetch, register, spatial_connection

if TYPE_CHECKING:
    from qavf.engine.context import DataContext

ZONE = "zone"
"""Name of the zone id column of a loaded layer frame."""
GEOMETRY = "geometry"
"""Name of its geometry column (WKB binary or WKT text)."""
_PARQUET = (".parquet", ".geoparquet", ".pq")


class ZoneLayer(RuleParams):
    """Zone polygons: ``{dataset: name}`` or ``{path: file}``, with an ``id`` column."""

    dataset: str | None = None
    """A configured dataset holding the zones (an id and a WKB or WKT geometry column)."""
    path: str | None = None
    """A local GeoParquet, GeoJSON, shapefile or other ``ST_Read`` file (relative to the
    configuration)."""
    id: str = "zone"
    """Zone id column."""
    geometry: str | None = None
    """Geometry column: ``geometry`` for datasets and GeoParquet, ``geom`` for ``ST_Read``."""
    crs: Crs = DEFAULT_CRS
    """CRS of the polygons; a file declaring another one is refused."""
    layer: str | None = None
    """Layer of a multi-layer file (``ST_Read``'s ``layer``)."""
    repair: bool = False
    """Replace invalid geometries with ``ST_MakeValid`` (recorded) instead of refusing them."""

    @model_validator(mode="after")
    def _one_source(self) -> ZoneLayer:
        if (self.dataset is None) == (self.path is None):
            raise ValueError("a zone layer needs exactly one of 'dataset' or 'path'")
        if self.layer is not None and (self.path is None or self._parquet):
            raise ValueError("'layer' selects a layer of an ST_Read file")
        return self

    @property
    def _parquet(self) -> bool:
        return self.path is not None and Path(self.path).suffix.lower() in _PARQUET

    @property
    def native(self) -> bool:
        """Whether the zones are read by ``ST_Read`` (which needs the extension)."""
        return self.path is not None and not self._parquet

    @property
    def geometry_column(self) -> str:
        return self.geometry or ("geom" if self.native else "geometry")

    def label(self) -> str:
        return self.dataset if self.dataset is not None else str(self.path)

    def datasets(self) -> frozenset[str]:
        return frozenset({self.dataset}) if self.dataset is not None else frozenset()


@dataclass(frozen=True)
class LoadedLayer:
    """A validated layer bound to a DuckDB connection, valid while it stays open."""

    table: str
    """Relation with ``zone_order``, ``geom`` and its bounding box (``xmin`` ... ``ymax``)."""
    ids: pl.Series
    """Zone ids in zone order (index ``k`` is zone order ``k``)."""
    crs: str
    notes: tuple[str, ...] = ()
    """What loading did that the configuration does not say (repaired geometries)."""


def _path(ctx: DataContext | None, path: str) -> Path:
    return ctx.resolve_path(path) if ctx is not None else Path(path)


def _st_read(path: Path, layer: ZoneLayer) -> str:
    options = f", layer := {sql_literal(layer.layer)}" if layer.layer is not None else ""
    return f"ST_Read({sql_literal(str(path))}{options})"


def _parquet_crs(path: Path, column: str) -> str | None:
    """The CRS a GeoParquet file declares for ``column`` (CRS84 when its metadata omits it)."""
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).metadata or {}
    if b"geo" not in metadata:
        return None
    try:
        columns = json.loads(metadata[b"geo"]).get("columns", {})
    except ValueError:
        return None
    entry = columns.get(column)
    if not isinstance(entry, dict):
        return None
    if "crs" not in entry:
        return "OGC:CRS84"
    return projjson_crs(entry["crs"])


def _native_crs(con: Any, path: Path, layer: ZoneLayer) -> str | None:
    """The CRS ``ST_Read_Meta`` reports for the layer's geometry, if any."""
    rows = con.execute(f"SELECT layers FROM ST_Read_Meta({sql_literal(str(path))})").fetchall()
    for (layers,) in rows:
        for entry in layers or []:
            if layer.layer is not None and entry.get("name") != layer.layer:
                continue
            for field in entry.get("geometry_fields") or []:
                crs = field.get("crs")
                if crs and crs.get("auth_name") and crs.get("auth_code"):
                    return normalize_crs(f"{crs['auth_name']}:{crs['auth_code']}")
            return None
    return None


def _checked_ids(frame: pl.DataFrame, layer: ZoneLayer) -> pl.DataFrame:
    if absent := [c for c in (layer.id,) if c not in frame.columns]:
        raise ValueError(f"zone layer {layer.label()!r} lacks column(s) {', '.join(absent)}")
    ids = frame[layer.id]
    if missing := ids.null_count():
        raise ValueError(f"zone layer {layer.label()!r}: {plural(missing, 'zone')} without an id")
    if repeated := ids.len() - ids.n_unique():
        raise ValueError(
            f"zone layer {layer.label()!r}: zone ids must be unique ({plural(repeated, 'id')} "
            "repeated); dissolve multi-part zones into one geometry per id"
        )
    return frame


def zone_ids(ctx: DataContext | None, layer: ZoneLayer) -> pl.Series:
    """The layer's zone ids in zone order; only ``ST_Read`` files need the extension.

    Raises:
        ValueError: for an unreadable layer, missing or repeated ids.
        SpatialUnavailable: for an ``ST_Read`` file without the extension.
    """
    if layer.dataset is not None:
        if ctx is None:
            raise ValueError(f"zone layer dataset {layer.dataset!r} needs a data context")
        frame = ctx.project(layer.dataset, [layer.id])
    elif not layer.native:
        assert layer.path is not None
        frame = pl.read_parquet(_path(ctx, layer.path), columns=[layer.id])
    else:
        assert layer.path is not None
        path = _path(ctx, layer.path)
        with spatial_connection(ctx) as con:
            frame = fetch(con, f"SELECT {quote_identifier(layer.id)} FROM {_st_read(path, layer)}")
    return _checked_ids(frame, layer)[layer.id].sort()


def layer_frame(ctx: DataContext | None, layer: ZoneLayer, con: Any) -> pl.DataFrame:
    """The layer's ids and geometries (``zone``, ``geometry``), sorted by id.

    ``geometry`` is WKB (binary) or WKT (text). A file's declared CRS is checked
    against the configured one.

    Raises:
        ValueError: for missing columns, ids or a CRS mismatch.
    """
    column = layer.geometry_column
    if layer.dataset is not None:
        if ctx is None:
            raise ValueError(f"zone layer dataset {layer.dataset!r} needs a data context")
        frame = ctx.project(layer.dataset, [layer.id, column])
    else:
        assert layer.path is not None
        path = _path(ctx, layer.path)
        if layer.native:
            check_declared(layer.crs, _native_crs(con, path, layer), f"zone layer {path.name!r}")
            frame = fetch(
                con,
                f"SELECT {quote_identifier(layer.id)}, "
                f"ST_AsWKB({quote_identifier(column)})::BLOB AS {quote_identifier(column)} "
                f"FROM {_st_read(path, layer)}",
            )
        else:
            check_declared(layer.crs, _parquet_crs(path, column), f"zone layer {path.name!r}")
            frame = pl.read_parquet(path, columns=[layer.id, column])
    if column not in frame.columns:
        raise ValueError(f"zone layer {layer.label()!r} lacks the geometry column {column!r}")
    dtype = frame.schema[column]
    if dtype not in (pl.Binary, pl.String):
        raise ValueError(
            f"zone layer {layer.label()!r}: geometry column {column!r} must hold WKB (binary) "
            f"or WKT (text), found {dtype}"
        )
    frame = _checked_ids(frame, layer).sort(layer.id)
    return frame.select(pl.col(layer.id).alias(ZONE), pl.col(column).alias(GEOMETRY))


def load_zones(
    con: Any, ctx: DataContext | None, layer: ZoneLayer, table: str = "qavf_zones"
) -> LoadedLayer:
    """Read, validate and bind the layer to ``con`` (a spatial connection) as ``table``.

    Raises:
        ValueError: for unusable ids or geometries (missing, malformed, not polygonal,
            empty, or invalid without ``repair``).
    """
    key = ("qavf.spatial.layer", layer.model_dump_json(), str(_path(ctx, layer.path or "")))
    frame = (
        ctx.memo(key, lambda: layer_frame(ctx, layer, con))
        if ctx is not None and not layer.native
        else layer_frame(ctx, layer, con)
    )
    label = layer.label()
    source = f"{table}_source"
    register(
        con,
        source,
        frame.select(
            pl.int_range(0, pl.len(), dtype=pl.Int64).alias("zone_order"), pl.col(GEOMETRY)
        ),
    )
    reader = "ST_GeomFromWKB" if frame.schema[GEOMETRY] == pl.Binary else "ST_GeomFromText"
    try:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {quote_identifier(table)} AS SELECT zone_order, "
            f"CASE WHEN geometry IS NULL THEN NULL ELSE {reader}(geometry) END AS geom "
            f"FROM {quote_identifier(source)}"
        )
    except duckdb.Error as exc:
        raise ValueError(f"zone layer {label!r} has malformed geometry: {exc}") from None
    finally:
        con.unregister(source)
    name = quote_identifier(table)
    notes: list[str] = []

    def offenders(condition: str) -> tuple[int, list[Any]]:
        count = int(con.execute(f"SELECT count(*) FROM {name} WHERE {condition}").fetchone()[0])
        rows = fetch(
            con, f"SELECT zone_order FROM {name} WHERE {condition} ORDER BY zone_order LIMIT 5"
        ) if count else None
        orders = rows["zone_order"].to_list() if rows is not None else []
        return count, frame[ZONE].gather(orders).to_list()

    for condition, problem in (
        ("geom IS NULL OR ST_IsEmpty(geom)", "without a geometry"),
        (
            "CAST(ST_GeometryType(geom) AS VARCHAR) NOT IN ('POLYGON', 'MULTIPOLYGON')",
            "that are not polygons",
        ),
    ):
        count, sample = offenders(condition)
        if count:
            raise ValueError(f"zone layer {label!r}: {plural(count, 'zone')} {problem} {sample}")
    count, sample = offenders("NOT ST_IsValid(geom)")
    if count and not layer.repair:
        raise ValueError(
            f"zone layer {label!r}: {plural(count, 'zone')} with invalid geometry {sample} "
            "(self-intersections, unclosed rings, ...); fix them or set repair: true"
        )
    if count:
        con.execute(
            f"UPDATE {name} SET geom = ST_CollectionExtract(ST_MakeValid(geom), 3) "
            "WHERE NOT ST_IsValid(geom)"
        )
        notes.append(
            f"zone layer {label!r}: repaired the invalid geometry of {plural(count, 'zone')}"
        )
        count, sample = offenders(
            "geom IS NULL OR ST_IsEmpty(geom) OR NOT ST_IsValid(geom) OR "
            "CAST(ST_GeometryType(geom) AS VARCHAR) NOT IN ('POLYGON', 'MULTIPOLYGON')"
        )
        if count:
            raise ValueError(
                f"zone layer {label!r}: repair left {plural(count, 'zone')} without a "
                f"nonempty valid polygon {sample}"
            )
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT zone_order, geom, ST_XMin(geom) AS xmin, "
        f"ST_XMax(geom) AS xmax, ST_YMin(geom) AS ymin, ST_YMax(geom) AS ymax FROM {name}"
    )
    return LoadedLayer(table, frame[ZONE], normalize_crs(layer.crs), tuple(notes))
