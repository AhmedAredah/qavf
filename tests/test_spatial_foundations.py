"""The shared spatial modules against hand-built oracles: assignment, CRS, distance,
grids, crosswalks, layers and neighbour weights."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qavf.spatial import (
    AssignmentPolicy,
    Crosswalk,
    Grid,
    ZoneLayer,
    aggregate_up,
    assign,
    assign_cells,
    compose,
    contiguity,
    crosswalk_frame,
    distance_band,
    from_table,
    grid_cells,
    grid_contiguity,
    haversine,
    label_positions,
    load_zones,
    nearest,
    normalize_crs,
    spatial_connection,
    validate_crosswalk,
    zone_ids,
)
from qavf.spatial.crs import check_declared, is_geographic, same_crs, transform_sql
from qavf.spatial.distance import euclidean
from qavf.spatial.extension import register
from qavf.spatial.layers import GEOMETRY, ZONE, LoadedLayer


def _support():
    name = "qavf_spatial_support"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("spatial_support.py")
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


S = _support()
COLLECT = pl.LazyFrame.collect


def _assign(zones, points, policy=None, *, tmp_path, crs="EPSG:4326", layer_crs="EPSG:4326"):
    path = S.geojson(tmp_path / "zones.geojson", zones)
    frame = pl.DataFrame(points, schema=["x", "y"], orient="row")
    with spatial_connection() as con:
        loaded = load_zones(con, None, ZoneLayer(path=str(path), crs=layer_crs))
        assigned = assign(
            frame.with_row_index("row").lazy(),
            con,
            loaded,
            policy or AssignmentPolicy(),
            column="zone",
            crs=crs,
            collect=COLLECT,
            x="x",
            y="y",
        )
    rows = assigned.plan.collect().sort("row", "zone")
    found: dict[int, list] = {}
    for row, zone in rows.select("row", "zone").iter_rows():
        found.setdefault(row, [])
        if zone is not None:
            found[row].append(zone)
    return assigned, [found[i] for i in range(len(points))]


# ----------------------------------------------------------------- assignment
POINTS = [
    (1.0, 1.0),  # interior of A
    (3.0, 1.0),  # interior of L
    (1.0, 3.0),  # interior of L (its upper arm)
    (2.0, 1.0),  # the edge A and L share
    (2.0, 2.0),  # the vertex A and L share (L's reflex corner)
    (0.0, 0.0),  # an outer vertex of A only
    (4.0, 4.0),  # an outer vertex of L only
    (5.0, 5.0),  # outside every zone
    (None, 1.0),  # no coordinate
]


def test_intersects_assigns_boundaries_and_breaks_ties_by_the_lowest_zone(tmp_path):
    assigned, zones = _assign(S.A_AND_L, POINTS, tmp_path=tmp_path)
    assert zones == [["A"], ["L"], ["L"], ["A"], ["A"], ["A"], ["L"], [], []]
    assert (assigned.rows, assigned.unassigned, assigned.ties, assigned.overlaps) == (9, 2, 2, 0)


def test_contains_leaves_boundary_points_unassigned(tmp_path):
    policy = AssignmentPolicy(predicate="contains")
    assigned, zones = _assign(S.A_AND_L, POINTS, policy, tmp_path=tmp_path)
    assert zones == [["A"], ["L"], ["L"], [], [], [], [], [], []]
    assert (assigned.unassigned, assigned.ties) == (6, 0)


def test_overlapping_zones_refuse_or_follow_the_policy(tmp_path):
    points = [(1.0, 1.0), (0.25, 0.25), (0.5, 1.0)]  # in O and A; in A only; O's edge in A
    with pytest.raises(ValueError, match="inside several zones"):
        _assign(S.OVERLAPPING, points, tmp_path=tmp_path)
    first, zones = _assign(
        S.OVERLAPPING, points, AssignmentPolicy(overlap="first"), tmp_path=tmp_path
    )
    # On O's edge the point is interior to A only, so A wins without an overlap.
    assert zones == [["A"], ["A"], ["A"]]
    assert (first.overlaps, first.ties) == (1, 1)
    every, zones = _assign(
        S.OVERLAPPING, points, AssignmentPolicy(overlap="all"), tmp_path=tmp_path
    )
    assert zones == [["A", "O"], ["A"], ["A"]]
    assert every.plan.collect().height == 4  # the overlapping row repeats once per zone


def test_unassigned_rows_beyond_the_allowed_fraction_refuse(tmp_path):
    assigned, _ = _assign(S.A_AND_L, POINTS, tmp_path=tmp_path)
    with pytest.raises(ValueError, match=r"2 rows of 9 \(22.22%\)"):
        assigned.check(AssignmentPolicy(), "points")
    assigned.check(AssignmentPolicy(max_unassigned=0.25), "points")


def test_points_in_another_crs_are_transformed_to_the_layer(tmp_path):
    # Web Mercator metres of (lon 1, lat 1) and (lon 3, lat 3), by the spherical formulas.
    radius = 6378137.0

    def mercator(lon, lat):
        return (
            radius * math.radians(lon),
            radius * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)),
        )

    points = [mercator(1, 1), mercator(3, 3)]
    _, zones = _assign(S.A_AND_L, points, crs="EPSG:3857", tmp_path=tmp_path)
    assert zones == [["A"], ["L"]]


def test_wkb_point_columns_are_assigned_like_coordinates(tmp_path):
    path = S.geojson(tmp_path / "zones.geojson", S.A_AND_L)
    frame = pl.DataFrame({"g": [S.wkb_point(1, 1), S.wkb_point(3, 3), None]})
    with spatial_connection() as con:
        loaded = load_zones(con, None, ZoneLayer(path=str(path)))
        assigned = assign(
            frame.lazy(), con, loaded, AssignmentPolicy(), column="zone", crs="EPSG:4326",
            collect=COLLECT, geometry="g",
        )  # fmt: skip
        polygon = pl.DataFrame({"g": [S.wkb_polygon(S.square(0, 0))]})
        with pytest.raises(ValueError, match="not points"):
            assign(
                polygon.lazy(), con, loaded, AssignmentPolicy(), column="zone",
                crs="EPSG:4326", collect=COLLECT, geometry="g",
            )  # fmt: skip
    assert sorted(assigned.plan.collect()["zone"].to_list(), key=str) == ["A", "L", None]
    assert assigned.unassigned == 1


# --------------------------------------------------------------------- layers
def test_layers_load_from_geoparquet_geojson_shapefile_and_wkt_datasets(tmp_path):
    zones = S.grid_zones(2)
    parquet = S.geoparquet(tmp_path / "z.parquet", zones)
    gj = S.geojson(tmp_path / "z.geojson", zones)
    with spatial_connection() as con:
        con.execute(
            f"COPY (SELECT zone, geom FROM ST_Read('{gj.as_posix()}')) TO "
            f"'{(tmp_path / 'z.shp').as_posix()}' WITH (FORMAT GDAL, DRIVER 'ESRI Shapefile')"
        )
        for path in (parquet, gj, tmp_path / "z.shp"):
            loaded = load_zones(con, None, ZoneLayer(path=str(path)))
            assert loaded.ids.to_list() == sorted(zones)
            area = con.execute(f'SELECT sum(ST_Area(geom)) FROM "{loaded.table}"').fetchone()
            assert area == (4.0,)
    # Ids alone need no extension for GeoParquet.
    assert zone_ids(None, ZoneLayer(path=str(parquet))).to_list() == sorted(zones)


def test_invalid_geometry_is_refused_unless_repaired(tmp_path):
    bowtie = [(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]
    path = S.geoparquet(tmp_path / "z.parquet", {"bow": bowtie, "ok": S.square(2, 0)})
    with spatial_connection() as con:
        with pytest.raises(ValueError, match=r"1 zone with invalid geometry \['bow'\]"):
            load_zones(con, None, ZoneLayer(path=str(path)))
        repaired = load_zones(con, None, ZoneLayer(path=str(path), repair=True))
        valid = con.execute(f'SELECT bool_and(ST_IsValid(geom)) FROM "{repaired.table}"')
        assert valid.fetchone() == (True,)
    assert repaired.notes and "repaired" in repaired.notes[0]


def test_missing_repeated_and_non_polygon_zones_are_refused(tmp_path):
    with spatial_connection() as con:
        repeated = S.geoparquet(tmp_path / "r.parquet", {"a": S.square(0, 0)})
        frame = pl.read_parquet(repeated)
        pl.concat([frame, frame]).write_parquet(tmp_path / "rr.parquet")
        with pytest.raises(ValueError, match="must be unique"):
            load_zones(con, None, ZoneLayer(path=str(tmp_path / "rr.parquet")))
        points = pl.DataFrame({"zone": ["p"], "geometry": [S.wkb_point(0, 0)]})
        points.write_parquet(tmp_path / "p.parquet")
        with pytest.raises(ValueError, match="not polygons"):
            load_zones(con, None, ZoneLayer(path=str(tmp_path / "p.parquet")))


def test_a_declared_crs_mismatch_is_reported(tmp_path):
    path = S.geoparquet(tmp_path / "z.parquet", {"a": S.square(0, 0)}, epsg=32614)
    with spatial_connection() as con:
        with pytest.raises(ValueError, match="declares the CRS EPSG:32614, but EPSG:4326"):
            load_zones(con, None, ZoneLayer(path=str(path)))
        loaded = load_zones(con, None, ZoneLayer(path=str(path), crs="EPSG:32614"))
    assert loaded.crs == "EPSG:32614"
    check_declared("EPSG:4326", "OGC:CRS84", "layer")  # the same under x-first order


# ------------------------------------------------------------------------ CRS
def test_crs_names_normalize_and_classify():
    assert normalize_crs(4326) == normalize_crs("epsg:4326") == "EPSG:4326"
    assert is_geographic("EPSG:4326") and not is_geographic("EPSG:32614")
    assert same_crs("OGC:CRS84", "EPSG:4326") and not same_crs("EPSG:4326", "EPSG:3857")
    with pytest.raises(ValueError, match="AUTHORITY:CODE"):
        normalize_crs("WGS84")


def test_transforms_round_trip_and_hit_a_known_projected_coordinate():
    with spatial_connection() as con:
        # The central meridian of UTM zone 14N (99 W) on the equator is (500000, 0).
        point = transform_sql("ST_Point(-99, 0)", "EPSG:4326", "EPSG:32614")
        x, y = con.execute(f"SELECT ST_X(p), ST_Y(p) FROM (SELECT {point} AS p)").fetchone()
        assert x == pytest.approx(500000.0, abs=1e-6) and y == pytest.approx(0.0, abs=1e-6)
        there = transform_sql("ST_Point(-97.123, 30.456)", "EPSG:4326", "EPSG:32614")
        back = transform_sql(f"({there})", "EPSG:32614", "EPSG:4326")
        lon, lat = con.execute(f"SELECT ST_X(p), ST_Y(p) FROM (SELECT {back} AS p)").fetchone()
    assert (lon, lat) == (pytest.approx(-97.123, abs=1e-9), pytest.approx(30.456, abs=1e-9))
    assert transform_sql("g", "OGC:CRS84", "EPSG:4326") == "g"


# ------------------------------------------------------------------- distance
def _great_circle(lon1, lat1, lon2, lat2, radius):
    """Vincenty's formula on a sphere (atan2 form), written independently."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    numerator = math.hypot(
        math.cos(p2) * math.sin(dl),
        math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl),
    )
    denominator = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return radius * math.atan2(numerator, denominator)


def test_haversine_matches_a_published_airport_distance():
    # Nashville (BNA) to Los Angeles (LAX) on a 6372.8 km sphere: 2887.2599506071106 km
    # (the Rosetta Code haversine reference value).
    meters = haversine(-86.67, 36.12, -118.40, 33.94, radius=6372.8e3)
    assert meters == pytest.approx(2_887_259.9506071106, rel=1e-12)
    # One degree of the equator is 1/360 of the circumference.
    assert haversine(0.0, 0.0, 1.0, 0.0) == pytest.approx(2 * math.pi * 6_371_008.8 / 360)


def test_haversine_agrees_with_an_independent_formula_on_floats_arrays_and_polars():
    rng = np.random.default_rng(3)
    lon1, lon2 = rng.uniform(-180, 180, (2, 200))
    lat1, lat2 = rng.uniform(-89, 89, (2, 200))
    expected = [
        _great_circle(a, b, c, d, 6_371_008.8)
        for a, b, c, d in zip(lon1, lat1, lon2, lat2, strict=True)
    ]
    assert haversine(lon1, lat1, lon2, lat2) == pytest.approx(expected, rel=1e-9, abs=1e-6)
    frame = pl.DataFrame({"a": lon1, "b": lat1, "c": lon2, "d": lat2})
    native = frame.select(haversine(pl.col("a"), pl.col("b"), pl.col("c"), pl.col("d")))
    assert native.to_series().to_list() == pytest.approx(expected, rel=1e-9, abs=1e-6)
    assert haversine(0.0, 0.0, 180.0, 0.0) == pytest.approx(math.pi * 6_371_008.8)  # antipodes
    assert euclidean(0.0, 0.0, 3.0, 4.0) == 5.0


coordinates = st.tuples(st.floats(-180, 180, allow_nan=False), st.floats(-90, 90, allow_nan=False))


@settings(max_examples=200, deadline=None)
@given(coordinates, coordinates, coordinates)
def test_haversine_is_symmetric_and_satisfies_the_triangle_inequality(a, b, c):
    ab = haversine(*a, *b)
    assert ab == pytest.approx(haversine(*b, *a), abs=1e-6)
    assert ab <= haversine(*a, *c) + haversine(*c, *b) + 1e-6
    assert 0 <= ab <= math.pi * 6_371_008.8 + 1e-6


# ---------------------------------------------------------------------- grids
def test_grid_labels_are_floor_positions_and_cells_are_half_open():
    grid = Grid(size=10.0, origin=(5.0, 0.0))
    frame = pl.DataFrame({"x": [5.0, 14.999, 15.0, -5.0, -5.000001, None], "y": [0.0] * 6})
    assigned = assign_cells(
        frame.lazy(), grid, x="x", y="y", crs="EPSG:32614", column="cell", collect=COLLECT
    )
    labels = assigned.plan.collect()["cell"].to_list()
    assert labels == ["0_0", "0_0", "1_0", "-1_0", "-2_0", None]
    assert assigned.unassigned == 1
    assert label_positions(pl.Series(["-2_7"])).row(0) == (-2, 7)


def test_the_expected_grid_covers_every_cell_an_extent_overlaps():
    grid = Grid(size=1.0, extent=(0.5, 0.0, 3.0, 2.0))
    cells = grid_cells(grid)
    # x from 0.5 to 3 overlaps columns 0, 1, 2; y from 0 to 2 rows 0 and 1.
    assert cells.len() == 6
    assert cells.to_list() == ["0_0", "1_0", "2_0", "0_1", "1_1", "2_1"]
    with pytest.raises(ValueError, match="more than max_cells"):
        grid_cells(Grid(size=0.001, extent=(0, 0, 10, 10), max_cells=1000))


def test_grid_cells_in_another_crs_transform_the_points_first():
    grid = Grid(size=100_000.0, crs="EPSG:32614")
    frame = pl.DataFrame({"lon": [-99.0], "lat": [0.0]})
    assigned = assign_cells(
        frame.lazy(), grid, x="lon", y="lat", crs="EPSG:4326", column="cell", collect=COLLECT,
        connection=spatial_connection,
    )  # fmt: skip
    assert assigned.plan.collect()["cell"].to_list() == ["5_0"]  # x = 500 000 m


# ---------------------------------------------------------------- crosswalks
def test_split_weights_aggregate_up_to_hand_computed_totals():
    walk = crosswalk_frame(
        None,
        Crosswalk(pairs=[["t1", "c1", 1.0], ["t2", "c1", 0.25], ["t2", "c2", 0.75],
                         ["t3", "c2", 1.0]]),
    )  # fmt: skip
    zones = pl.DataFrame({"taz": ["t1", "t2", "t3"], "trips": [10, 8, 5]})
    parents = aggregate_up(zones, "taz", ["trips"], walk)
    # c1 = 10 + 0.25 * 8 = 12, c2 = 0.75 * 8 + 5 = 11.
    assert parents.rows() == [("c1", 12.0), ("c2", 11.0)]
    unsplit = crosswalk_frame(None, Crosswalk(pairs=[["t1", "c1"], ["t2", "c1"], ["t3", "c2"]]))
    exact = aggregate_up(zones, "taz", ["trips"], unsplit)
    assert exact.rows() == [("c1", 18), ("c2", 5)] and exact.schema["trips"].is_integer()


def test_weights_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError, match=r"do not sum to 1 within 1e-09 \('t2': 0.9"):
        crosswalk_frame(None, Crosswalk(pairs=[["t2", "c1", 0.5], ["t2", "c2", 0.4]]))
    with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
        crosswalk_frame(None, Crosswalk(pairs=[["t2", "c1", 1.5]]))
    with pytest.raises(ValueError, match="repeated"):
        crosswalk_frame(None, Crosswalk(pairs=[["t2", "c1"], ["t2", "c1"]]))
    tolerant = pl.DataFrame({"from": ["a", "a"], "to": ["x", "y"], "weight": [0.5, 0.4999]})
    validate_crosswalk(tolerant, 1e-3, "tolerant")


def test_hierarchies_compose_link_zone_county():
    links = crosswalk_frame(None, Crosswalk(pairs=[["l1", "z1"], ["l2", "z2"]]))
    zones = crosswalk_frame(
        None, Crosswalk(pairs=[["z1", "c1", 1.0], ["z2", "c1", 0.5], ["z2", "c2", 0.5]])
    )
    chained = validate_crosswalk(compose(links, zones), 1e-12, "chained")
    assert sorted(chained.rows()) == [("l1", "c1", 1.0), ("l2", "c1", 0.5), ("l2", "c2", 0.5)]


# -------------------------------------------------------------------- weights
def test_queen_and_rook_contiguity_on_a_three_by_three_grid(tmp_path):
    zones = S.grid_zones(3)
    path = S.geojson(tmp_path / "grid.geojson", zones)
    with spatial_connection() as con:
        loaded = load_zones(con, None, ZoneLayer(path=str(path)))
        queen = contiguity(con, loaded.table, loaded.ids, "queen")
        rook = contiguity(con, loaded.table, loaded.ids, "rook")
    # Corners, edges and the centre of a 3 x 3 grid: 3/5/8 queen and 2/3/4 rook neighbours.
    counts = zip(queen.counts(), rook.counts(), strict=True)
    by_zone = dict(zip(loaded.ids.to_list(), counts, strict=True))
    for zone, counts in by_zone.items():
        i, j = (int(v) for v in zone[1:].split("_"))
        edges = (i in (0, 2)) + (j in (0, 2))
        assert counts == {2: (3, 2), 1: (5, 3), 0: (8, 4)}[edges], zone
    assert (queen.pairs, rook.pairs) == (40, 24) and not queen.islands
    positions = label_positions(pl.Series([f"{i}_{j}" for i in range(3) for j in range(3)]))
    cells = grid_contiguity(pl.Series(range(9)), positions["ix"], positions["iy"], "rook")
    assert sorted(cells.counts()) == sorted(rook.counts())


def test_nearest_neighbours_bands_tables_and_row_standardization():
    ids = pl.Series(["a", "b", "c", "d"])
    x, y = np.array([0.0, 1.0, 3.0, 10.0]), np.zeros(4)
    knn = nearest(ids, x, y, 2, geographic=False)
    assert {i: sorted(row) for i, row in knn.neighbours.items()} == {
        0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [1, 2]
    }  # fmt: skip
    band = distance_band(ids, x, y, 2.0, geographic=False)
    assert band.counts() == [1, 2, 1, 0] and band.islands == [3]
    standard = band.standardized()
    assert standard.neighbours[1] == {0: 0.5, 2: 0.5} and standard.row_standardized
    table = from_table(ids, pl.DataFrame({"from": ["a"], "to": ["d"], "weight": [2.0]}))
    assert table.neighbours[0] == {3: 2.0}
    with pytest.raises(ValueError, match="not zones"):
        from_table(ids, pl.DataFrame({"from": ["a"], "to": ["zz"], "weight": [1.0]}))
    # Geographic bands use haversine metres: one degree of the equator is ~111 km.
    lon = np.array([0.0, 1.0, 2.5])
    assert distance_band(ids[:3], lon, np.zeros(3), 112_000, geographic=True).counts() == [
        1, 1, 0
    ]  # fmt: skip


# -------------------------------------------------------------------- property
interior = st.lists(
    st.tuples(
        st.integers(0, 2), st.integers(0, 2), st.floats(0.01, 0.99), st.floats(0.01, 0.99),
        st.integers(0, 1000),
    ),
    min_size=1,
    max_size=40,
)  # fmt: skip


@settings(max_examples=25, deadline=None)
@given(interior)
def test_assigning_then_aggregating_equals_aggregating_per_zone(rows):
    # Points drawn inside known unit squares: the assigned zone sums are the known ones.
    frame = pl.DataFrame(
        [(f"z{i}_{j}", i + u, j + v, value) for i, j, u, v, value in rows],
        schema=["known", "x", "y", "value"],
        orient="row",
    )
    zones = pl.DataFrame(
        {
            "zone": list(S.grid_zones(3)),
            "geometry": [S.wkb_polygon(r) for r in S.grid_zones(3).values()],
        }
    )
    with spatial_connection() as con:
        register(con, "qavf_zones_source", zones.select(ZONE, GEOMETRY))
        con.execute(
            "CREATE OR REPLACE TEMP TABLE grid AS SELECT row_number() OVER (ORDER BY zone) - 1 "
            "AS zone_order, ST_GeomFromWKB(geometry) AS geom, 0.0 AS xmin FROM qavf_zones_source"
        )
        con.execute(
            "CREATE OR REPLACE TEMP TABLE grid AS SELECT zone_order, geom, ST_XMin(geom) AS xmin, "
            "ST_XMax(geom) AS xmax, ST_YMin(geom) AS ymin, ST_YMax(geom) AS ymax FROM grid"
        )

        layer = LoadedLayer("grid", zones["zone"].sort(), "EPSG:4326")
        assigned = assign(
            frame.lazy(), con, layer, AssignmentPolicy(), column="zone", crs="EPSG:4326",
            collect=COLLECT, x="x", y="y",
        )  # fmt: skip
    result = assigned.plan.collect()
    by_assignment = dict(result.group_by("zone").agg(pl.col("value").sum()).iter_rows())
    by_known = dict(frame.group_by("known").agg(pl.col("value").sum()).iter_rows())
    assert by_assignment == by_known and assigned.unassigned == 0
