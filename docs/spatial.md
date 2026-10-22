# Spatial data

QAVF checks and compares data that lives in space: points with coordinates, features
with geometries, and values per zone (a county, a traffic analysis zone, a road link,
a grid cell). The building blocks are one shared module, `qavf.spatial`, used by the
GEO-001 and SPA-003 rules and by dataset comparisons (the `assign_zones` step and
the `spatial` and `locations` levels), so a zone, a cell, a distance and a neighbour
mean the same thing everywhere.

```python
from qavf import compare_datasets

result = compare_datasets(
    "v1/trips.parquet",
    "v2/trips.parquet",
    keys=["trip_id"],
    steps=[{"step": "assign_zones", "x": "origin_lon", "y": "origin_lat",
            "layer": {"path": "zones/taz.geojson", "id": "taz"}}],
    levels=[
        {"level": "spatial", "measures": ["count", "sum(passengers)"]},
        {"level": "locations", "x": "origin_lon", "y": "origin_lat", "max_distance": 50},
    ],
)
spatial = result.level("spatial")
print(spatial.morans_i, [cluster["zone"] for cluster in spatial.clusters])
```

## The spatial extension

Geometry work runs in DuckDB with its `spatial` extension: reading GeoJSON and
shapefiles (`ST_Read`), point-in-polygon, polygon contiguity, CRS transforms
(`ST_Transform`), areas and shape differences. QAVF loads it with `LOAD spatial`
and, if it is not installed, tries DuckDB's own `INSTALL spatial` once.

When the extension cannot be loaded, every check that needs geometry is **not
assessed**, with the reason, never passed: the `assign_zones` step refuses (so the
comparison's levels are not assessed), the `moran` and `lisa` checks of a layer's
zones, `locations` on geometries and GEO-001's `within` are not assessed. Everything
else keeps working without it: attribute zones (`county`, `taz`, link ids), grid
cells in the data's own CRS, crosswalks, neighbours from grid cells or a table,
expected zones from a configured dataset or a GeoParquet file, and SPA-003's zone
layers read from those.

| Environment | How the extension is available offline |
|---|---|
| Docker image | pre-installed with the other DuckDB extensions |
| CI and release checks | `python scripts/verify_release.py extensions` installs and verifies it |
| Your machine | `python -c "import duckdb; duckdb.connect().execute('INSTALL spatial')"` once |

## Zone layers

A zone layer is polygons with an id, from a configured dataset or a local file:

```text
{dataset: zones, id: taz}                       # any dataset: an id and a WKB or WKT column
{dataset: zones, id: taz, geometry: wkt}        # the geometry column's name
{path: zones/taz.parquet, id: taz}              # GeoParquet (WKB), read by Polars
{path: zones/taz.geojson, id: taz}              # GeoJSON, shapefile, GeoPackage: ST_Read
{path: zones/net.gpkg, layer: zones, id: taz}   # one layer of a multi-layer file
{path: zones/taz.shp, id: TAZ, crs: EPSG:2277}  # the layer's CRS (EPSG:4326 by default)
```

Every zone is validated before use: ids present and unique, a geometry present,
polygonal and valid (`ST_IsValid`). A layer with an invalid geometry (a
self-intersection, an unclosed ring) is refused, which leaves whatever needed it not
assessed; `repair: true` replaces each invalid geometry with `ST_MakeValid` (kept
polygonal) and records it. Zones are ordered by id, and that *zone order* breaks
every tie deterministically, whatever the file order.

## Coordinate reference systems

A CRS is written `AUTHORITY:CODE` (`EPSG:4326`, `EPSG:32614`, `OGC:CRS84`).
Longitude/latitude columns default to `EPSG:4326`; zone layers declare theirs with
`crs`. Coordinates are always read **x first** (longitude or easting): transforms use
PROJ's `always_xy` semantics, so data in (longitude, latitude) order is never flipped
by the authority's latitude-first axis order.

- Points are transformed to the layer's (or grid's) CRS before they are assigned.
- A file that declares its own CRS (GeoParquet `geo` metadata, a shapefile's `.prj`,
  a GeoJSON `crs`) must agree with the configured one; a mismatch is refused, never
  silently transformed. `EPSG:4326` and `OGC:CRS84` are the same under x-first order.
- Distances follow the CRS: the haversine great-circle distance in metres for a
  geographic CRS (WGS 84, CRS84, NAD83, ETRS89, GDA94, NAD83(CSRS), NZGD2000), on a
  sphere of the IUGG mean radius 6 371 008.8 m (within about 0.5 % of the
  ellipsoidal geodesic); the Euclidean distance in CRS units for any other CRS.
  `qavf.spatial.haversine` is one formula for floats, NumPy arrays and Polars
  expressions.

## Assigning points to zones

Points (`x`/`y` columns, or a WKB `geometry` column of points) get the id of the zone
polygon they fall in. The distinct points are joined to the zones in DuckDB, with a
bounding-box prefilter (a range join on each zone's extent) before the exact
predicate, and the zone ids are joined back natively: no row reaches Python.

| Situation | Policy | Outcome |
|---|---|---|
| A point on a zone's boundary | `predicate: intersects` (default) | the zone's |
| | `predicate: contains` | unassigned |
| A point on the boundary shared by several zones (a *tie*) | | the zone whose interior holds it, else the lowest zone id; counted |
| A point inside several overlapping zones | `overlap: error` (default) | refused |
| | `overlap: first` | the lowest zone id |
| | `overlap: all` | each of them (the row repeats per zone); counted |
| A point in no zone, or without usable coordinates (missing, NaN, infinite) | `max_unassigned` (0 by default) | unassigned and counted; more than that fraction of the rows is not assessed |

## Grid cells

A grid is a spatial bucketing, the counterpart of a time bucket:

```text
grid: {size: 1000, crs: EPSG:32614}                     # 1 km squares in UTM 14N
grid: {size: 0.01}                                       # 0.01-degree cells of the points' CRS
grid: {size: 500, crs: EPSG:32614, extent: [600000, 3300000, 650000, 3350000]}
```

A point `(x, y)` lies in the cell at position `(floor((x - x0) / size),
floor((y - y0) / size))` from `origin` `(x0, y0)` (0, 0 by default), labelled
`"<ix>_<iy>"` (`"-3_12"`). Cells are half-open: a point on a cell's west or south edge
belongs to it. With an `extent` (`xmin, ymin, xmax, ymax`), the *expected grid* is
every cell the extent overlaps (at most `max_cells`), in position order. Labels are
computed natively in binary64; positions and labels are exact integers. A grid in the
points' own CRS needs no extension. Hexagonal (H3) cells would need a community
extension and are not offered.

## Crosswalks and hierarchies

A crosswalk maps each `from` zone to one or more `to` zones: links to zones, zones to
counties, the 2010 zones to the 2020 zones that replaced them.

```text
crosswalk: {dataset: taz_to_county, from_column: taz, to_column: county}
crosswalk: {path: crosswalks/taz2010_2020.csv, from_column: old, to_column: new,
            weight_column: share}
crosswalk: {pairs: [[A, A1, 0.6], [A, A2, 0.4], [B, B, 1.0]]}
```

Many-to-one is the common case (weight 1). A split zone's weights must sum to 1 within
`tolerance` (1e-9), checked exactly; ids must be present, pairs unique and weights in
(0, 1]. A crosswalk that fails any of this is refused. Hierarchies are crosswalks
chained with `qavf.spatial.compose` (weights multiply exactly), and
`qavf.spatial.aggregate_up` aggregates per-zone values to their parents exactly: sums
of unsplit zones with the exact native accumulators, split zones as exact rationals
of the values and weights, rounded once.

## Neighbour weights

The `moran` and `lisa` checks, and `qavf.inference.morans_i` and `local_morans_i`,
read sparse neighbour weights `{i: {j: w}}` (`qavf.spatial.SpatialWeights`), which
report islands (zones without a neighbour) explicitly:

| `weights.method` | Neighbours |
|---|---|
| `queen` (default) | polygons sharing a vertex or an edge; grid cells differing by at most 1 in each direction |
| `rook` | polygons sharing an edge (a boundary of positive length); grid cells differing by 1 in one direction |
| `knn` (`k`, 4) | the `k` nearest zone centroids or cell centres (ties by zone order) |
| `distance_band` (`distance`) | every pair of centroids at most `distance` apart (metres for a geographic CRS) |
| `table` (`table`) | explicit `from`/`to` (`weight`) pairs, written like a crosswalk |

Overlapping polygons are neighbours under both contiguity rules. Weights are
row-standardized (`row_standardize: true`) unless set otherwise; islands keep no
weight and are never part of a cluster.

## Comparing datasets in space

### `assign_zones`

The alignment step gives each side's rows a zone before any level runs, so `groups`,
`temporal`, `records` and `spatial` can all group by it:

| Option | Meaning |
|---|---|
| `column` | the zone column added (`zone`) |
| `x`, `y` or `geometry` | the points |
| `crs`, `baseline_crs`, `candidate_crs` | the points' CRS (per side when it differs) |
| `layer`, or `baseline_layer` with `candidate_layer` | the zones (per side when they were re-drawn) |
| `grid` | grid cells instead of polygons |
| `zones` | an attribute zone column instead of points (no geometry) |
| `predicate`, `overlap`, `max_unassigned` | the [assignment policies](#assigning-points-to-zones) |
| `crosswalk`, `crosswalk_side` | map one side's zones to the other's (`baseline` by default), or both sides' zones to parents (`both`) |
| `apportion`, `weight_column` | columns a split zone's rows divide by its weight; a column holding the weight |
| `unmapped` | zones the crosswalk lacks: `error` (default) or `drop` (recorded) |

What the step found is recorded in `DatasetComparison.alignment`: rows in no zone
(dropped), ties on shared boundaries, rows in overlapping zones, repaired geometries,
unmapped and split zones. A crosswalk that splits zones needs `apportion` or
`weight_column`, so a split row never counts in full twice.

### `spatial`

Measures per zone on each side, through the same machinery as `groups`: measures,
comparators, materiality, significance and the decision policies mean exactly what
they mean there.

| Check | Compares | Test |
|---|---|---|
| `zones` | zones on one side only (`missing_zones: ignore` to accept) | |
| one per measure | material changes per zone, the largest first | the measure's test per zone, corrected across zones |
| `expected` | zones of the layer, or of the grid's `extent`, a side lacks (`allow_missing`; `expected: false` to skip) | |
| `rank` (opt-in, `rank: {}`) | Spearman's correlation of the zones' values (`min_spearman`, 0.9), for when the ordering of zones matters (for example the top emitters); off by default because immaterial noise reorders zones of similar size | Fisher z with the Fieller-Hartley-Pearson variance: below the minimum? |
| `shares` | zones whose share of their side's total changes beyond `max_change` (0.01), gained or lost | two-proportion per zone when the values are counts, corrected; otherwise by size |
| `moran` | global Moran's I of the per-zone `difference` (`diff`, or any comparator): are the changes clustered? material from `min_i` (0.1) | permutation, one-sided |
| `lisa` | each zone's local Moran's I of the differences: hot spots (a larger change among larger changes) and cold spots of **material** changes | conditional permutation per zone, corrected across zones |

The zones come from the level's `layer` or `grid`, or from the `assign_zones` step that
produced `zone_column`. The pattern checks read one measure (`pattern_measure`, the
first by default) on the zones both sides have; set a check to `null` to leave it
out (`rank` is off unless configured). Under `decision: material` no test runs: `rank`, `shares` and `moran` are judged
by their thresholds and `lisa` (which is a test) is left out. Under `equivalence`,
`moran` and `lisa` use `material_and_significant`: a cluster is a difference.
`SpatialResult` reports the zone counts, `expected_zones`, `missing`, `spearman`,
`morans_i`, the `weights` used, `islands` and the named `clusters` (bounded).
For the report's choropleth it also keeps each zone's values (`values`, bounded by
`chart_points`) and the zones' `outlines`: the layer's polygons simplified in DuckDB
(`ST_SimplifyPreserveTopology`, a tolerance of 1/500 of the extent) or the grid's
cells, as SVG paths; more than `map_vertices` (20,000) vertices, or no spatial
extension, leaves the map out with the reason (`map: false` skips it).

Permutation p-values are at least `1 / (permutations + 1)` (`smallest_p` in the
`lisa` check's metrics). With `m` compared zones, the Benjamini-Hochberg correction
names a single cluster only when that is at most `alpha / m`: the default 999
permutations suffice for about 50 zones at `alpha: 0.05`; raise `permutations` for
more zones.

### `locations`

Features matched by the comparison's `keys` (repeated keys are not assessed):

| Check | Meaning |
|---|---|
| `located` | features with a location on one side only |
| `moved` | points (`x`/`y`), or geometry centroids, moved more than `max_distance` (metres for a geographic CRS, else CRS units), largest first |
| `area` | geometries whose area changed beyond `rtol`/`atol` (square metres on the WGS 84 spheroid for a geographic CRS) |
| `shape` | geometries that are not equal and whose symmetric difference, `area(a) + area(b) - 2 area(a ∩ b)`, exceeds `max_difference` |

Distances are computed natively; geometries need the extension.

### Zones over time

`assign_zones` with the `temporal` level gives per-zone, per-bucket comparisons, and
`series_shape` per-zone series:

```yaml
datasets:
  trips_2024: {path: trips/2024.parquet}
  trips_2025: {path: trips/2025.parquet}
  zones: {path: zones/taz.parquet}            # GeoParquet with columns taz and geometry
rules:
  CMP-001:
    params:
      comparisons:
        - name: trips by zone
          baseline: trips_2024
          candidate: trips_2025
          keys: [trip_id]
          steps:
            - {step: shift_time, column: day, offset: {years: 1}, unaligned: drop}
            - {step: assign_zones, x: lon, y: lat, layer: {dataset: zones, id: taz},
               max_unassigned: 0.001}
          levels:
            - {level: spatial, measures: [count, sum(passengers)], weights: {method: rook}}
            - {level: temporal, time_column: day, bucket: week, group_by: [zone],
               measures: [count]}
            - {level: locations, x: lon, y: lat, max_distance: 25}
```

## Rules

- **GEO-001** takes an optional `within` zone layer or study area per point set
  (`boundary: intersects` or `contains`): valid points outside it fail a separate
  `<point set> within <layer>` check, through the same point-in-polygon path. Without
  the extension that check alone is not assessed; existing configurations are
  unchanged.
- **SPA-003** takes `zone_layer`: the expected entities are the layer's ids
  (`unexpected_entities` then reports entities outside the layer). A dataset or
  GeoParquet layer needs no extension.

## Limitations

- Zones must have unique ids; multi-part zones are one (multi)polygon per id.
- Geodesic distances use a sphere; areas of geographic data use the WGS 84 spheroid.
- A zone's change is tested as `groups` does (rows within a zone independent); the
  spatial dependence between zones is what `moran` and `lisa` look at.
- LISA p-values share one set of permutation draws across zones (as PySAL does): each
  is exact for its zone, the family is correlated; the Benjamini-Hochberg correction
  is valid under positive dependence, Benjamini-Yekutieli (`correction: by`) under any.
