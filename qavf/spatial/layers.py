"""Zone layers: polygons with an id, from a configured dataset or a file.

A layer is written ``{dataset: zones, id: taz}`` (any configured dataset with an
id column and a WKB or WKT geometry column, e.g. a GeoParquet file or a query)
or ``{path: zones.geojson, id: taz}``: GeoParquet (``.parquet``, geometry as
WKB) is read like any input file (memory target, source policy and identity),
every other file (GeoJSON, shapefile, GeoPackage, ...) by DuckDB's ``ST_Read``
with its size admitted before the fetch. ``crs`` declares the layer's CRS (``EPSG:4326`` by
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

from qavf.config_paths import FilePath
from qavf.duckdb_results import settle
from qavf.engine.registry import RuleParams
from qavf.missingness import missing_count
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
    path: FilePath | None = None
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


def _project_parquet(ctx: DataContext | None, path: Path, columns: list[str]) -> pl.DataFrame:
    """A GeoParquet projection read like any other input: memory admission, the
    run's source policy and source identity, as a crosswalk ``path`` is."""
    from qavf.inputs.auxiliary import project_input
    from qavf.inputs.descriptors import File

    data = File(str(path), format="parquet", columns=columns)
    if ctx is not None:
        return _storage(project_input(ctx, data, columns))
    from qavf.engine.context import DataContext

    with DataContext() as owner:
        return _storage(project_input(owner, data, columns))


def _is_extension(dtype: pl.DataType) -> bool:
    """Whether ``dtype`` is an extension type. Polars 1.32 (the supported floor)
    has no extension types or ``is_extension``; there GeoParquet geometry is
    already plain WKB binary."""
    probe = getattr(dtype, "is_extension", None)
    return bool(probe()) if callable(probe) else False


def _storage(frame: pl.DataFrame) -> pl.DataFrame:
    """Extension columns (GeoParquet's ``geoarrow.wkb``) as their storage (WKB binary)."""
    extensions = [name for name, dtype in frame.schema.items() if _is_extension(dtype)]
    return frame.with_columns(pl.col(c).to_physical() for c in extensions) if extensions else frame


class _NativeLayerRead:
    """One ``ST_Read`` fetch of a layer file under the run's resource policy.

    The fetch's size is estimated in DuckDB and admitted against the memory
    target before any row reaches Polars, the result is admitted again, and the
    file (with its shapefile companions) must be unchanged after the read.
    """

    def __init__(self, ctx: DataContext | None, con: Any, path: Path, layer: ZoneLayer) -> None:
        self.ctx, self.con, self.path, self.layer = ctx, con, path, layer

    def files(self) -> tuple[Path, ...]:
        """The layer file and any sibling with its stem (``.dbf``, ``.shx``, ``.prj``...)."""
        stem = self.path.stem
        siblings = sorted(p for p in self.path.parent.iterdir() if p.stem == stem and p.is_file())
        return tuple(dict.fromkeys([self.path, *siblings]))

    def fetch(self, select: str, sizes: str) -> pl.DataFrame:
        """``SELECT {select}`` from the layer; ``sizes`` is its byte-size expression."""
        from qavf.sources.local_files import LocalManifest

        label = f"zone layer {self.layer.label()!r}"
        source = _st_read(self.path, self.layer)
        manifest = LocalManifest.capture(self.files(), _file_stat)
        if self.ctx is not None and self.ctx.memory_budget_bytes is not None:
            estimate = fetch(self.con, f"SELECT coalesce(sum({sizes}), 0) AS size FROM {source}")
            self.ctx.check_retained_bytes(label, int(estimate["size"][0] or 0))
        frame = fetch(self.con, f"SELECT {select} FROM {source}")
        if self.ctx is not None:
            self.ctx.check_materialization(label, frame)
        manifest.verify(_file_stat)
        return frame


def _file_stat(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


def _id_size(column: str) -> str:
    """DuckDB estimate of one id's bytes (text by length, others 8 bytes)."""
    quoted = quote_identifier(column)
    return f"coalesce(strlen(CAST({quoted} AS VARCHAR)), 0) + 8"


def _checked_ids(frame: pl.DataFrame, layer: ZoneLayer) -> pl.DataFrame:
    if absent := [c for c in (layer.id,) if c not in frame.columns]:
        raise ValueError(f"zone layer {layer.label()!r} lacks column(s) {', '.join(absent)}")
    ids = frame[layer.id]
    if missing := missing_count(ids):
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
        frame = _project_parquet(ctx, _path(ctx, layer.path), [layer.id])
    else:
        assert layer.path is not None
        path = _path(ctx, layer.path)
        with spatial_connection(ctx) as con:
            frame = _NativeLayerRead(ctx, con, path, layer).fetch(
                quote_identifier(layer.id), _id_size(layer.id)
            )
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
            wkb = f"ST_AsWKB({quote_identifier(column)})::BLOB"
            frame = _NativeLayerRead(ctx, con, path, layer).fetch(
                f"{quote_identifier(layer.id)}, {wkb} AS {quote_identifier(column)}",
                f"{_id_size(layer.id)} + coalesce(octet_length({wkb}), 0)",
            )
        else:
            check_declared(layer.crs, _parquet_crs(path, column), f"zone layer {path.name!r}")
            frame = _project_parquet(ctx, path, [layer.id, column])
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
        settle(con)
        con.unregister(source)
    name = quote_identifier(table)
    notes: list[str] = []

    def offenders(condition: str) -> tuple[int, list[Any]]:
        count = int(con.execute(f"SELECT count(*) FROM {name} WHERE {condition}").fetchone()[0])
        rows = (
            fetch(
                con, f"SELECT zone_order FROM {name} WHERE {condition} ORDER BY zone_order LIMIT 5"
            )
            if count
            else None
        )
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
