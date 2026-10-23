"""Independent assignment oracles, admission boundaries and connection lifetime."""

from __future__ import annotations

import importlib
import math
import struct

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.spatial.assign import AssignmentLimits, AssignmentPolicy, assign, matches
from qavf.spatial.extension import spatial_connection
from qavf.spatial.layers import LoadedLayer

assignment = importlib.import_module("qavf.spatial.assign")


def layer(con, rectangles):
    """Build known rectangles without the production layer loader or matching code."""
    ordered = sorted(rectangles.items())
    con.execute("CREATE TEMP TABLE oracle_zones(zone_order BIGINT, geom GEOMETRY)")
    for order, (_, (xmin, ymin, xmax, ymax)) in enumerate(ordered):
        wkt = f"POLYGON(({xmin} {ymin},{xmax} {ymin},{xmax} {ymax},{xmin} {ymax},{xmin} {ymin}))"
        con.execute("INSERT INTO oracle_zones VALUES (?, ST_GeomFromText(?))", [order, wkt])
    con.execute(
        "CREATE OR REPLACE TEMP TABLE oracle_zones AS SELECT *, ST_XMin(geom) xmin, "
        "ST_XMax(geom) xmax, ST_YMin(geom) ymin, ST_YMax(geom) ymax FROM oracle_zones"
    )
    return LoadedLayer(
        "oracle_zones", pl.Series("zone", [name for name, _ in ordered]), "EPSG:4326"
    )


def rectangle_oracle(points, rectangles, predicate, overlap):
    output, unassigned, ties, overlaps = [], 0, 0, 0
    for row, (x, y) in enumerate(points):
        contacts, interiors = [], []
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            for name, (xmin, ymin, xmax, ymax) in sorted(rectangles.items()):
                inside = xmin < x < xmax and ymin < y < ymax
                touches = xmin <= x <= xmax and ymin <= y <= ymax
                if inside:
                    interiors.append(name)
                if inside or (predicate == "intersects" and touches):
                    contacts.append(name)
        if not contacts:
            unassigned += 1
            output.append((row, None))
            continue
        ties += len(contacts) > 1 and len(interiors) <= 1
        overlaps += len(interiors) > 1
        picked = interiors or contacts
        if overlap != "all" or len(interiors) <= 1:
            picked = picked[:1]
        output.extend((row, zone) for zone in picked)
    return output, (len(points), unassigned, ties, overlaps)


@pytest.mark.parametrize("batch_size", [1, 3, 64])
@pytest.mark.parametrize("predicate", ["contains", "intersects"])
@pytest.mark.parametrize("overlap", ["first", "all"])
def test_assignment_matches_rectangle_oracle_across_batches_and_after_close(
    batch_size,
    predicate,
    overlap,
):
    rectangles = {"z": (1, 0, 2, 1), "a": (0, 0, 1, 1), "m": (0.5, 0.5, 1.5, 1.5)}
    points = [
        (0.75, 0.75),
        (1.0, 0.25),
        (1.0, 0.75),
        (0.1, 0.1),
        (2.0, 2.0),
        (None, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (-0.0, 0.5),
        (+0.0, 0.5),
        (0.75, 0.75),
        (1.0, 0.25),
    ]
    expected, counts = rectangle_oracle(points, rectangles, predicate, overlap)
    frame = pl.DataFrame(points, schema=["x", "y"], orient="row").with_row_index("row")
    with spatial_connection() as con:
        assigned = assign(
            frame.lazy(),
            con,
            layer(con, rectangles),
            AssignmentPolicy(batch_size=batch_size, predicate=predicate, overlap=overlap),
            column="zone",
            crs="EPSG:4326",
            collect=pl.LazyFrame.collect,
            x="x",
            y="y",
        )
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]
    result = assigned.plan.collect().sort("row", "zone")
    assert result.select("row", "zone").rows() == expected
    assert (assigned.rows, assigned.unassigned, assigned.ties, assigned.overlaps) == counts
    assert assigned.plan.collect().sort("row", "zone").equals(result)


@pytest.mark.parametrize("name", ["batch_size", "max_points", "max_matches", "max_work"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", None])
def test_assignment_limits_are_strict_positive_integers(name, value):
    with pytest.raises(ValidationError):
        AssignmentLimits(**{name: value})


def test_distinct_limit_uses_one_bounded_source_collection():
    frame = pl.DataFrame({"x": [float(i) for i in range(20)], "y": [0.0] * 20})
    exported = []

    def collect(plan):
        result = plan.collect()
        exported.append(result.height)
        return result

    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 30, 1)})
        with pytest.raises(ValueError, match="max_points \\(4\\)"):
            assign(
                frame.lazy(),
                con,
                zones,
                AssignmentPolicy(max_points=4),
                column="zone",
                crs="EPSG:4326",
                collect=collect,
                x="x",
                y="y",
            )
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]
    assert exported == [5]


class TraceConnection:
    def __init__(self, connection):
        self.connection = connection
        self.sql = []

    def execute(self, sql, *args):
        self.sql.append(sql)
        self.connection.execute(sql, *args)
        return self

    def __getattr__(self, name):
        return getattr(self.connection, name)


def points(count):
    return pl.DataFrame({"pid": list(range(count)), "key_x": [0.5] * count, "key_y": [0.5] * count})


def test_candidate_work_rejection_stops_count_and_precedes_exact_tests():
    with spatial_connection() as con:
        zones = layer(con, {str(i): (0, 0, 1, 1) for i in range(10)})
        trace = TraceConnection(con)
        with pytest.raises(ValueError, match="max_work \\(7\\)"):
            matches(
                trace,
                points(4),
                zones,
                AssignmentPolicy(max_work=7, overlap="first"),
                crs="EPSG:4326",
            )
        assert any(
            "SELECT count(*) FROM (SELECT 1" in sql and "LIMIT 8" in sql for sql in trace.sql
        )
        assert not any("ST_Contains" in sql or "ST_Intersects" in sql for sql in trace.sql)
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]


def test_selected_cap_is_checked_before_export_and_arrow_batches_are_bounded(monkeypatch):
    sizes = []
    original = assignment.pl.from_arrow

    def observed(chunk, *args, **kwargs):
        sizes.append(chunk.num_rows)
        assert chunk.schema.names == ["pid", "zone_order", "tie", "overlap"]
        return original(chunk, *args, **kwargs)

    monkeypatch.setattr(assignment.pl, "from_arrow", observed)
    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 1, 1), "b": (0, 0, 1, 1)})
        policy = AssignmentPolicy(batch_size=3, max_matches=7, max_work=20, overlap="all")
        with pytest.raises(ValueError, match="max_matches \\(7\\)"):
            matches(con, points(5), zones, policy, crs="EPSG:4326")
        assert sizes == [3, 3]  # second point batch is refused before its four mappings transfer
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]
        sizes.clear()
        chosen = matches(
            con, points(5), zones, policy.model_copy(update={"max_matches": 10}), crs="EPSG:4326"
        )
    assert chosen.height == 10 and max(sizes) <= 3 and sum(sizes) == 10


def test_limits_are_inclusive_and_candidate_budget_accumulates_between_batches():
    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 1, 1), "b": (0, 0, 1, 1)})
        policy = AssignmentPolicy(
            batch_size=1, max_points=3, max_matches=3, max_work=6, overlap="first"
        )
        assert matches(con, points(3), zones, policy, crs="EPSG:4326").height == 3
        with pytest.raises(ValueError, match="max_work"):
            matches(
                con, points(3), zones, policy.model_copy(update={"max_work": 5}), crs="EPSG:4326"
            )
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]


def test_late_overlap_error_reports_all_distinct_points_and_leaves_connection_usable():
    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 2, 2), "b": (0, 0, 1, 1)})
        frame = pl.DataFrame({"pid": [0, 1, 2], "key_x": [1.5, 0.5, 0.6], "key_y": [1.5, 0.5, 0.6]})
        with pytest.raises(ValueError, match="2 distinct points lie inside several zones"):
            matches(con, frame, zones, AssignmentPolicy(batch_size=1), crs="EPSG:4326")
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]


@pytest.mark.parametrize("ids", [[1, 1], [1, None], [1.5, 2.5]])
def test_invalid_point_ids_are_rejected_before_batching(ids):
    frame = pl.DataFrame({"pid": ids, "key_x": [0.5, 0.5], "key_y": [0.5, 0.5]})
    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 1, 1)})
        with pytest.raises(ValueError, match="unique non-null integers"):
            matches(con, frame, zones, AssignmentPolicy(batch_size=1), crs="EPSG:4326")
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]


def test_reader_failure_closes_reader_and_preserves_caller_relations(monkeypatch):
    def fail(chunk):
        raise RuntimeError("conversion failed after native reader started")

    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 1, 1)})
        for name in ("qavf_points", "qavf_located", "qavf_point_tally"):
            con.execute(f"CREATE TEMP TABLE {name} AS SELECT 42 AS kept")
        before = con.execute("SHOW TABLES").fetchall()
        with monkeypatch.context() as scoped:
            scoped.setattr(assignment.pl, "from_arrow", fail)
            with pytest.raises(RuntimeError, match="conversion failed"):
                matches(con, points(3), zones, AssignmentPolicy(batch_size=1), crs="EPSG:4326")
        assert con.execute("SHOW TABLES").fetchall() == before
        for name in ("qavf_points", "qavf_located", "qavf_point_tally"):
            assert con.execute(f"SELECT kept FROM {name}").fetchall() == [(42,)]
        assert (
            matches(con, points(2), zones, AssignmentPolicy(batch_size=1), crs="EPSG:4326").height
            == 2
        )
        assert con.execute("SHOW TABLES").fetchall() == before


@pytest.mark.parametrize("malformed", [False, True])
def test_late_bad_wkb_refuses_whole_assignment_and_cleans_temporaries(malformed):
    point = struct.pack("<BIdd", 1, 1, 0.5, 0.5)
    line = struct.pack("<BII4d", 1, 2, 2, 0.0, 0.0, 1.0, 1.0)
    frame = pl.DataFrame({"pid": [0, 1], "key": [point, b"bad" if malformed else line]})
    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 1, 1)})
        with pytest.raises(ValueError, match=r"points cannot be placed|not points"):
            matches(
                con, frame, zones, AssignmentPolicy(batch_size=1), crs="EPSG:4326", geometry=True
            )
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]


def test_empty_points_and_failed_transform_have_no_temporary_lifetime():
    frame = pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64})
    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 1, 1)})
        assigned = assign(
            frame.lazy(),
            con,
            zones,
            AssignmentPolicy(batch_size=1),
            column="zone",
            crs="EPSG:4326",
            collect=pl.LazyFrame.collect,
            x="x",
            y="y",
        )
        with pytest.raises(ValueError, match="points cannot be placed"):
            matches(con, points(1), zones, AssignmentPolicy(), crs="EPSG:0")
        assert con.execute("SHOW TABLES").fetchall() == [("oracle_zones",)]
    assert assigned.plan.collect().is_empty()
    assert (assigned.rows, assigned.unassigned, assigned.ties, assigned.overlaps) == (0, 0, 0, 0)


def test_wkb_exact_keys_and_crs_transform_survive_multiple_batches_after_close():
    # Distinct endian encodings of one geographic point are separate exact WKB keys.
    wkb = [struct.pack("<BIdd", 1, 1, 0.5, 0.5), struct.pack(">BIdd", 0, 1, 0.5, 0.5), None]
    frame = pl.DataFrame({"g": wkb}).with_row_index("row")
    with spatial_connection() as con:
        zones = layer(con, {"a": (0, 0, 1, 1)})
        assigned = assign(
            frame.lazy(),
            con,
            zones,
            AssignmentPolicy(batch_size=1),
            column="zone",
            crs="EPSG:4326",
            collect=pl.LazyFrame.collect,
            geometry="g",
        )
        radius = 6378137.0
        projected = pl.DataFrame(
            {
                "x": [radius * math.radians(0.5)],
                "y": [radius * math.log(math.tan(math.pi / 4 + math.radians(0.5) / 2))],
            }
        )
        transformed = assign(
            projected.lazy(),
            con,
            zones,
            AssignmentPolicy(batch_size=1),
            column="zone",
            crs="EPSG:3857",
            collect=pl.LazyFrame.collect,
            x="x",
            y="y",
        )
    assert assigned.plan.collect().sort("row")["zone"].to_list() == ["a", "a", None]
    assert transformed.plan.collect()["zone"].to_list() == ["a"]
