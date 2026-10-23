"""Outline projection and native pre-admission retain complete deterministic maps."""

from __future__ import annotations

import json

import polars as pl
import pyarrow as pa
import pytest

from qavf import compare_datasets
from qavf.spatial import outlines
from qavf.spatial.extension import SpatialUnavailable, spatial_connection
from qavf.spatial.grid import Grid
from qavf.spatial.outlines import (
    OutlineLimitExceeded,
    grid_rings,
    layer_rings,
    view_paths,
)


def test_projection_preserves_holes_parts_empty_rings_order_and_rounded_centres():
    rings = {
        "west": [[], [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
                 [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]],
        42: [[(8.0, 2.0), (12.0, 2.0), (12.0, 4.0)],
             [(8.0, 0.0), (9.0, 0.0), (9.0, 1.0)]],
        "empty": [[]],
    }
    drawn = view_paths(rings, width=120.0)
    assert drawn.paths == {
        "west": "M0 40 L0 0 L40 0 L40 40ZM10 30 L20 30 L10 20Z",
        "42": "M80 20 L120 20 L120 0ZM80 40 L90 40 L90 30Z",
    }
    assert list(drawn.paths) == ["west", "42"]
    assert drawn.centres == {"west": [20.0, 20.0], "42": [100.0, 20.0]}
    assert (drawn.width, drawn.height, drawn.vertices) == (120.0, 40.0, 13)


def test_thin_extent_rounding_and_stringified_id_collision_keep_existing_semantics():
    drawn = view_paths({1: [[(0.0, 0.0), (3.0, 0.0)]], "1": [[(1.0, 0.0)]]}, 10.0)
    assert drawn.paths == {"1": "M3.3 0Z"}
    assert drawn.centres == {"1": [3.3, 0.0]}
    assert drawn.vertices == 3 and drawn.height == 1.0


@pytest.mark.parametrize("rings", [{}, {"empty": [[]]}, {"same": [[(2.0, 2.0)] * 3]}])
def test_empty_or_zero_extent_refuses_projection(rings):
    with pytest.raises(ValueError, match=r"no (outline|extent)"):
        view_paths(rings)


@pytest.mark.parametrize("width", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_nonpositive_or_nonfinite_width_refuses_projection(width):
    with pytest.raises(ValueError, match="width must be finite and positive"):
        view_paths({"a": [[(0.0, 0.0), (1.0, 1.0)]]}, width)


def test_grid_budget_counts_distinct_nonnull_cells_at_exact_boundary():
    labels = pl.Series(["-1_2", None, "3_4", "-1_2"])
    grid = Grid(size=2, origin=(10, -4))
    expected = {
        "-1_2": [[(8, 0), (10, 0), (10, 2), (8, 2)]],
        "3_4": [[(16, 4), (18, 4), (18, 6), (16, 6)]],
    }
    assert grid_rings(labels, grid, max_vertices=8) == expected
    with pytest.raises(OutlineLimitExceeded) as caught:
        grid_rings(labels, grid, max_vertices=7)
    assert caught.value.vertices == 8
    assert "map_vertices 7" in str(caught.value)
    assert grid_rings(pl.Series([None], dtype=pl.String), grid, max_vertices=0) == {}


def test_grid_budget_does_not_hide_malformed_labels():
    with pytest.raises(ValueError, match="not grid cell labels"):
        grid_rings(pl.Series(["0_0", "bad"]), Grid(size=1), max_vertices=1)


def test_public_map_budget_preserves_checks_and_complete_count(monkeypatch):
    frame = pl.DataFrame({"zone": ["0_0", "1_0", "1_1"], "value": [1.0, 2.0, 3.0]})
    level = {"level": "spatial", "grid": {"size": 1}, "measures": ["sum(value)"],
             "moran": None, "lisa": None, "shares": None, "rank": None, "expected": False}
    full = compare_datasets(frame, frame, levels=[{**level, "map_vertices": 12}])

    def forbid_projection(*args, **kwargs):
        raise AssertionError("an over-budget map must not be projected")

    monkeypatch.setattr("qavf.compare.levels.spatial.view_paths", forbid_projection)
    small = compare_datasets(frame, frame, levels=[{**level, "map_vertices": 11}])
    assert full.level("spatial").checks == small.level("spatial").checks
    assert len(full.level("spatial").outlines.paths) == 3
    omitted = small.level("spatial").outlines
    assert omitted.vertices == 12 and omitted.paths == {} and omitted.centres == {}
    assert omitted.note == "too large: 12 outline vertices after simplification (map_vertices 11)"


class _Reader:
    def __init__(self, shapes):
        self.closed = False
        self.shapes = shapes

    def __iter__(self):
        yield pa.record_batch({"zone_order": range(len(self.shapes)), "shape": self.shapes})

    def close(self):
        self.closed = True


class _Query:
    def __init__(self, reader):
        self.reader = reader

    def to_arrow_reader(self, rows):
        assert rows == 64
        return self.reader


class _Connection:
    def __init__(self, reader, count=0):
        self.reader = reader
        self.count = count
        self.transferred = False

    def execute(self, sql, params):
        assert "sum(ST_NPoints(ST_SimplifyPreserveTopology" in sql
        assert "ST_AsGeoJSON" not in sql
        return self

    def fetchone(self):
        return (self.count,)

    def sql(self, sql, params):
        assert "ORDER BY zone_order" in sql
        self.transferred = True
        return _Query(self.reader)


def _extent(monkeypatch):
    monkeypatch.setattr(outlines, "fetch", lambda *args: pl.DataFrame(
        {"x0": [0.0], "x1": [10.0], "y0": [0.0], "y1": [5.0]}
    ))


def test_native_vertex_budget_precedes_geojson_reader(monkeypatch):
    _extent(monkeypatch)
    con = _Connection(_Reader([]), count=1_000_000)
    with pytest.raises(OutlineLimitExceeded) as caught:
        layer_rings(con, "zones", pl.Series(["a"]), max_vertices=20_000)
    assert caught.value.vertices == 1_000_000
    assert not con.transferred


@pytest.mark.parametrize("malformed", [False, True])
def test_streaming_reader_closes_on_success_and_decode_error(monkeypatch, malformed):
    _extent(monkeypatch)
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]}
    reader = _Reader(["bad json" if malformed else json.dumps(polygon)])
    con = _Connection(reader, count=3)
    if malformed:
        with pytest.raises(ValueError):
            layer_rings(con, "zones", pl.Series(["a"]), max_vertices=3)
    else:
        assert layer_rings(con, "zones", pl.Series(["a"]), max_vertices=3) == {
            "a": [[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]]
        }
    assert reader.closed


def test_native_polygon_holes_multipolygon_and_exact_budget():
    try:
        with spatial_connection() as con:
            con.execute("""CREATE TABLE zones AS SELECT zone_order, geom, ST_XMin(geom) AS xmin,
                ST_XMax(geom) AS xmax, ST_YMin(geom) AS ymin, ST_YMax(geom) AS ymax FROM
                (VALUES (1, ST_GeomFromText('MULTIPOLYGON (((6 0,8 0,8 2,6 2,6 0)),
                    ((6 4,7 4,7 5,6 5,6 4)))')),
                (0, ST_GeomFromText('POLYGON ((0 0,4 0,4 4,0 4,0 0),
                    (1 1,1 2,2 2,2 1,1 1))'))) AS t(zone_order, geom)""")
            expected = {
                "hole": [[(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
                         [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]],
                "multi": [[(6, 0), (8, 0), (8, 2), (6, 2), (6, 0)],
                          [(6, 4), (7, 4), (7, 5), (6, 5), (6, 4)]],
            }
            ids = pl.Series(["hole", "multi"])
            assert layer_rings(con, "zones", ids, max_vertices=20) == expected
            with pytest.raises(OutlineLimitExceeded) as caught:
                layer_rings(con, "zones", ids, max_vertices=19)
            assert caught.value.vertices == 20
            con.execute("DELETE FROM zones")
            with pytest.raises(ValueError, match="no outline"):
                layer_rings(con, "zones", ids, max_vertices=20)
    except SpatialUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("point", [(float("nan"), 0.0), (0.0, float("inf")),
                                   (float("-inf"), 0.0)])
def test_nonfinite_interior_coordinate_refuses_the_complete_map(point):
    with pytest.raises(ValueError, match="nonfinite outline coordinates"):
        view_paths({"a": [[(0.0, 0.0), (1.0, 1.0), point]]})


@pytest.mark.parametrize("points", [[(-1e308, 0.0), (1e308, 1.0)],
                                    [(0.0, 0.0), (5e-324, 0.0)]])
def test_unrepresentable_projection_refuses_instead_of_emitting_nan(points):
    with pytest.raises(ValueError, match=r"finite extent|projection range"):
        view_paths({"a": [points]})


def test_small_positive_width_is_still_accepted():
    drawn = view_paths({"a": [[(0.0, 0.0), (1.0, 1.0)]]}, width=5e-324)
    assert drawn.paths == {"a": "M0 0 L0 0Z"}
    assert drawn.height == 1.0 and drawn.vertices == 2


def test_huge_finite_view_keeps_finite_centres():
    drawn = view_paths({"all": [[(0.0, 0.0), (1.0, 1.0)]],
                        "corner": [[(0.9, 0.0), (1.0, 1.0)]]}, width=1e308)
    assert drawn.centres == {"all": [5e307, 5e307], "corner": [9.5e307, 5e307]}
    assert drawn.width == drawn.height == 1e308
