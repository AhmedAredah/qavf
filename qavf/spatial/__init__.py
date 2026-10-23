"""Spatial building blocks shared by rules and dataset comparisons.

The spatial counterpart of :mod:`qavf.temporal`: what a zone, a cell, a
distance or a neighbour means is defined once here, so GEO-001, SPA-003 and
the comparison step and levels (``assign_zones``, ``spatial``, ``locations``)
agree. Single-purpose modules:

* :mod:`~qavf.spatial.extension`: DuckDB's ``spatial`` extension, loaded once
  per connection; without it geometry work is not assessed, never passed;
* :mod:`~qavf.spatial.crs`: declared CRSs and x-first (``always_xy``) transforms;
* :mod:`~qavf.spatial.distance`: haversine and Euclidean distance, one formula
  for floats, arrays and Polars expressions;
* :mod:`~qavf.spatial.layers`: zone polygons from a dataset, GeoParquet,
  GeoJSON or a shapefile, validated;
* :mod:`~qavf.spatial.assign`: point-to-zone assignment (the one
  point-in-polygon path) with explicit boundary, overlap and unassigned policies;
* :mod:`~qavf.spatial.grid`: regular grid cells, labels and the expected grid;
* :mod:`~qavf.spatial.hierarchy`: crosswalks and hierarchies with split weights,
  and exact aggregation up a hierarchy;
* :mod:`~qavf.spatial.weights`: neighbour weights (contiguity, nearest
  neighbours, distance bands, tables) for Moran's I and LISA.

Labels mirror the temporal contract: a zone or cell *label* per row, an
integer *position* (zone order, cell ``(ix, iy)``), and an *expected grid*
(the layer's zones, or the cells over an extent).
"""

from qavf.spatial.assign import Assigned, AssignmentLimits, AssignmentPolicy, assign
from qavf.spatial.crs import DEFAULT_CRS, Crs, is_geographic, normalize_crs, same_crs
from qavf.spatial.distance import EARTH_RADIUS_M, distance, euclidean, haversine
from qavf.spatial.extension import SpatialUnavailable, spatial_connection
from qavf.spatial.grid import Grid, assign_cells, cell_labels, grid_cells, label_positions
from qavf.spatial.hierarchy import (
    Crosswalk,
    aggregate_up,
    compose,
    crosswalk_frame,
    validate_crosswalk,
)
from qavf.spatial.layers import LoadedLayer, ZoneLayer, load_zones, zone_ids
from qavf.spatial.weights import (
    SpatialWeights,
    WeightsSpec,
    contiguity,
    distance_band,
    from_table,
    grid_contiguity,
    nearest,
)

__all__ = [
    "DEFAULT_CRS",
    "EARTH_RADIUS_M",
    "Assigned",
    "AssignmentLimits",
    "AssignmentPolicy",
    "Crosswalk",
    "Crs",
    "Grid",
    "LoadedLayer",
    "SpatialUnavailable",
    "SpatialWeights",
    "WeightsSpec",
    "ZoneLayer",
    "aggregate_up",
    "assign",
    "assign_cells",
    "cell_labels",
    "compose",
    "contiguity",
    "crosswalk_frame",
    "distance",
    "distance_band",
    "euclidean",
    "from_table",
    "grid_cells",
    "grid_contiguity",
    "haversine",
    "is_geographic",
    "label_positions",
    "load_zones",
    "nearest",
    "normalize_crs",
    "same_crs",
    "spatial_connection",
    "validate_crosswalk",
    "zone_ids",
]
