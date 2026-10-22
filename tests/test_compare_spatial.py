"""Spatial comparison: the assign_zones step and the spatial and locations levels, against
hand-computed oracles, failing closed without DuckDB's spatial extension."""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qavf import File, ResourceLimits, Status, compare_datasets, validate
from qavf.compare import DatasetComparison
from qavf.execution.os_limits import SUPPORTED_PLATFORMS
from qavf.spatial import extension
from qavf.spatial.extension import SpatialUnavailable


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
N = 6
"""Zones form an N x N grid of unit squares z<i>_<j>."""
BLOCK = {f"z{i}_{j}" for i in range(2) for j in range(2)}
"""The planted region: the four zones in the south-west corner."""


def _check(result, level, check):
    return next(c for c in result.level(level).checks if c.check == check)


def _layer(tmp_path) -> str:
    return str(S.geoparquet(tmp_path / "zones.parquet", S.grid_zones(N)))


def _points(seed: int, boost: float = 1.0, skip: str | None = None) -> pl.DataFrame:
    """Poisson points per zone, uniform inside it (never on an edge); zone intensities
    differ, so the zones' ranking is stable between draws."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        for j in range(N):
            zone = f"z{i}_{j}"
            if zone == skip:
                continue
            rate = 40 + 8 * (i * N + j)
            rate *= boost if zone in BLOCK else 1.0
            for _ in range(rng.poisson(rate)):
                rows.append((i + rng.uniform(0.05, 0.95), j + rng.uniform(0.05, 0.95), zone))
    return pl.DataFrame(rows, schema=["lon", "lat", "truth"], orient="row")


def _spatial(baseline, candidate, layer, **options):
    steps = [{"step": "assign_zones", "x": "lon", "y": "lat", "layer": {"path": layer}}]
    level = {"level": "spatial", "measures": ["count"], **options}
    return compare_datasets(baseline, candidate, steps=steps, levels=[level])


# ---------------------------------------------------------------- assign_zones
def test_assign_zones_gives_each_point_the_zone_it_was_drawn_in(tmp_path):
    points = _points(1)
    steps = [{"step": "assign_zones", "x": "lon", "y": "lat", "layer": {"path": _layer(tmp_path)}}]
    # Filters apply after the steps: no row's assigned zone differs from its known zone.
    wrong = compare_datasets(
        points, points, steps=steps, where="zone <> truth", min_rows=0, levels=["structure"]
    )
    assert (wrong.baseline_rows, wrong.candidate_rows) == (0, 0)
    # One (zone, known zone) group per known zone: assignment neither splits nor merges.
    counts = Counter(points["truth"].to_list())
    exact = compare_datasets(
        points, points, steps=steps, decision="material",
        levels=[{"level": "groups", "group_by": ["zone", "truth"], "measures": ["count"]}],
    )  # fmt: skip
    assert exact.level("groups").baseline_groups == len(counts) and exact.passed


def test_unassigned_points_are_dropped_counted_and_bounded(tmp_path):
    points = _points(2).vstack(
        pl.DataFrame({"lon": [10.0, None], "lat": [10.0, 1.0], "truth": ["out", "none"]})
    )
    layer = _layer(tmp_path)
    steps = [{"step": "assign_zones", "x": "lon", "y": "lat", "layer": {"path": layer}}]
    refused = compare_datasets(points, points, steps=steps, levels=["structure"])
    assert refused.error is not None and "2 rows of" in refused.error
    assert "max_unassigned" in refused.error and not refused.assessed
    allowed = [{**steps[0], "max_unassigned": 0.01}]
    result = compare_datasets(points, points, steps=allowed, levels=["structure"])
    assert result.passed and result.baseline_rows == points.height - 2
    notes = [n for n in result.alignment if n.startswith("baseline: assign_zones")]
    assert notes == ["baseline: assign_zones: 2 rows in no zone; rows in no zone were dropped"]


def test_per_side_layers_and_a_crosswalk_reconcile_redrawn_zones(tmp_path):
    # Version 1 has two halves; version 2 splits the west half in two. A point set
    # assigned to each version's own zones is reconciled by a crosswalk to version 2.
    east = S.square(2, 0, 2)
    old = {"W": S.square(0, 0, 2), "E": east}
    new = {
        "W1": [(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)],
        "W2": [(0, 1), (2, 1), (2, 2), (0, 2), (0, 1)],
        "E": east,
    }
    v1 = S.geojson(tmp_path / "v1.geojson", old)
    v2 = S.geojson(tmp_path / "v2.geojson", new)
    points = pl.DataFrame(
        {"lon": [0.5, 1.5, 3.0, 3.5], "lat": [0.5, 1.5, 1.0, 1.5], "trips": [10.0, 20.0, 5.0, 7.0]}
    )
    step = {
        "step": "assign_zones", "x": "lon", "y": "lat",
        "baseline_layer": {"path": str(v1)}, "candidate_layer": {"path": str(v2)},
        "crosswalk": {"pairs": [["W", "W1", 0.5], ["W", "W2", 0.5], ["E", "E", 1.0]]},
        "apportion": ["trips"], "weight_column": "share",
    }  # fmt: skip
    level = {"level": "groups", "group_by": ["zone"], "measures": ["sum(trips)", "sum(share)"]}
    result = compare_datasets(points, points, steps=[step], levels=[level], decision="material")
    assert result.error is None, result.error
    sums = _check(result, "groups", "sum(trips)")
    # Baseline W (30 trips) splits 15/15; the candidate has W1 = 10, W2 = 20.
    records = {r["zone"]: (r["baseline"], r["candidate"]) for r in sums.affected_records}
    assert records == {"W1": (15, 10), "W2": (15, 20)}
    # Each zone holds one row's worth on both sides: two halves of W's two rows.
    assert _check(result, "groups", "sum(share)").passed
    assert any("split 1 zone" in note for note in result.alignment)


def test_attribute_zones_with_a_crosswalk_need_no_extension(monkeypatch):
    _without_extension(monkeypatch)
    frame = pl.DataFrame({"taz": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    step = {"step": "assign_zones", "zones": "taz", "column": "county",
            "crosswalk": {"pairs": [["a", "north"], ["b", "north"], ["c", "south"]]}}  # fmt: skip
    level = {"level": "groups", "group_by": ["county"], "measures": ["sum(x)"]}
    other = frame.with_columns(county=pl.Series(["north", "north", "south"]))
    result = compare_datasets(frame, other, steps=[step], levels=[level], decision="material")
    assert result.passed, result.summary()
    unmapped = {**step, "crosswalk": {"pairs": [["a", "north"]]}}
    refused = compare_datasets(frame, other, steps=[unmapped], levels=[level])
    assert "2 row(s) have a 'county' zone the crosswalk" in str(refused.error)


# ---------------------------------------------------------------- spatial level
def test_a_planted_regional_change_is_detected_and_named_by_lisa(tmp_path):
    layer = _layer(tmp_path)
    result = _spatial(_points(3), _points(4, boost=1.6), layer)
    spatial = result.level("spatial")
    counts = _check(result, "spatial", "count")
    assert not counts.passed and {r["zone"] for r in counts.affected_records} <= BLOCK
    assert not _check(result, "spatial", "moran").passed and spatial.morans_i > 0.1
    lisa = _check(result, "spatial", "lisa")
    named = {cluster["zone"] for cluster in spatial.clusters}
    assert not lisa.passed and named and named <= BLOCK
    assert all(cluster["cluster"] == "hot spot" for cluster in spatial.clusters)
    assert lisa.significance is not None and lisa.significance.correction == "bh"
    assert spatial.weights.startswith("queen weights") and spatial.islands == 0


def test_uniform_noise_is_not_reported_as_a_spatial_change(tmp_path):
    result = _spatial(_points(5), _points(6), _layer(tmp_path))
    for check in ("zones", "count", "expected", "rank", "shares", "moran", "lisa"):
        found = _check(result, "spatial", check)
        assert found.assessed and found.passed, (check, found.message)
    assert result.level("spatial").clusters == []


def test_zones_on_one_side_only_and_expected_zones_missing(tmp_path):
    result = _spatial(_points(7), _points(7, skip="z5_5"), _layer(tmp_path))
    spatial = result.level("spatial")
    zones = _check(result, "spatial", "zones")
    assert not zones.passed and zones.affected_records == [
        {"zone": "z5_5", "zones": "only in baseline"}
    ]
    expected = _check(result, "spatial", "expected")
    assert not expected.passed and expected.affected_records == [
        {"zone": "z5_5", "side": "candidate"}
    ]
    assert (spatial.expected_zones, spatial.missing) == (36, {"baseline": 0, "candidate": 1})


def test_material_policy_judges_patterns_by_threshold_without_tests(tmp_path):
    result = _spatial(_points(3), _points(4, boost=1.6), _layer(tmp_path), decision="material")
    checks = [c.check for c in result.level("spatial").checks]
    assert "lisa" not in checks  # LISA needs a test
    moran = _check(result, "spatial", "moran")
    assert not moran.passed and moran.significance is None


def test_grid_zones_and_attribute_zones_with_table_weights():
    points = _points(8)
    candidate = _points(9, boost=1.6)
    steps = [{"step": "assign_zones", "x": "lon", "y": "lat", "column": "cell",
              "grid": {"size": 1.0, "extent": [0, 0, N, N]}}]  # fmt: skip
    level = {"level": "spatial", "zone_column": "cell", "weights": {"method": "rook"}}
    result = compare_datasets(points, candidate, steps=steps, levels=[level])
    spatial = result.level("spatial")
    assert spatial.expected_zones == N * N and spatial.weights.startswith("grid rook")
    assert {c["zone" if "zone" in c else "cell"] for c in spatial.clusters} <= {
        f"{i}_{j}" for i in range(2) for j in range(2)
    }
    # Attribute zones: the known zone column, with neighbours from a table.
    pairs = [[f"z{i}_{j}", f"z{i + 1}_{j}"] for i in range(N - 1) for j in range(N)]
    pairs += [[b, a] for a, b in pairs]
    attribute = {"level": "spatial", "zone_column": "truth",
                 "weights": {"method": "table", "table": {"pairs": pairs}}}  # fmt: skip
    plain = compare_datasets(points, candidate, levels=[attribute])
    assert _check(plain, "spatial", "moran").assessed
    assert all(c.check != "expected" for c in plain.level("spatial").checks)


def test_pattern_checks_need_neighbours_and_say_so():
    frame = pl.DataFrame({"county": ["a", "b", "c", "d"], "x": [1.0, 2.0, 3.0, 4.0]})
    level = {
        "level": "spatial",
        "zone_column": "county",
        "measures": ["sum(x)"],
        "decision": "material",
    }
    result = compare_datasets(frame, frame.with_columns(pl.col("x") * 2), levels=[level])
    moran = _check(result, "spatial", "moran")
    assert not moran.assessed and "no neighbours for the zones" in moran.message
    assert _check(result, "spatial", "sum(x)").assessed


# -------------------------------------------------------------- locations level
def test_known_moves_in_metres_and_in_projected_units():
    before = pl.DataFrame({"id": [1, 2, 3, 4], "lon": [-97.0, -97.0, 10.0, None],
                           "lat": [30.0, 30.0, 0.0, 1.0]})  # fmt: skip
    # Feature 1 moves 0.01 degrees north (1111.95 m), 2 moves 0.001 (111.2 m), 3 stays,
    # 4 gains a location.
    after = before.with_columns(
        pl.Series("lat", [30.01, 30.001, 0.0, 1.0]), pl.Series("lon", [-97.0, -97.0, 10.0, 5.0])
    )
    level = {"level": "locations", "x": "lon", "y": "lat", "max_distance": 500}
    result = compare_datasets(before, after, keys=["id"], levels=[level])
    moved = _check(result, "locations", "moved")
    metres = 6_371_008.8 * math.radians(0.01)
    assert moved.affected_count == 1 and moved.affected_records[0]["id"] == 1
    assert moved.affected_records[0]["distance"] == pytest.approx(metres, rel=1e-9)
    located = _check(result, "locations", "located")
    assert located.affected_records == [{"id": 4, "located": "only in candidate"}]
    projected = {**level, "crs": "EPSG:32614", "max_distance": 3.0}
    grid = pl.DataFrame({"id": [1, 2], "lon": [0.0, 10.0], "lat": [0.0, 10.0]})
    shifted = grid.with_columns(pl.Series("lon", [3.0, 14.0]), pl.Series("lat", [0.0, 13.0]))
    result = compare_datasets(grid, shifted, keys=["id"], levels=[projected])
    moved = _check(result, "locations", "moved")
    assert [r["distance"] for r in moved.affected_records] == [5.0]  # a 3-4-5 move
    assert result.level("locations").unit == "EPSG:32614 units"


def test_polygon_area_centroid_and_shape_changes():
    squares = pl.DataFrame(
        {"id": [1, 2, 3], "g": [S.wkb_polygon(S.square(0, 0)), S.wkb_polygon(S.square(5, 5)),
                                 S.wkb_polygon(S.square(9, 9))]}
    )  # fmt: skip
    # 1 doubles to the east (area 1 -> 2, centroid moves 0.5, symmetric difference 1);
    # 2 shifts by 0.25 (same area, difference 0.5); 3 is unchanged.
    after = squares.with_columns(
        pl.Series("g", [S.wkb_polygon([(0, 0), (2, 0), (2, 1), (0, 1), (0, 0)]),
                        S.wkb_polygon(S.square(5.25, 5)), S.wkb_polygon(S.square(9, 9))])
    )  # fmt: skip
    level = {"level": "locations", "geometry": "g", "crs": "EPSG:32614", "max_distance": 0.3}
    result = compare_datasets(squares, after, keys=["id"], levels=[level])
    area = _check(result, "locations", "area")
    assert [(r["id"], r["area_change"]) for r in area.affected_records] == [(1, 1.0)]
    shape = _check(result, "locations", "shape")
    assert [(r["id"], r["difference"]) for r in shape.affected_records] == [(1, 1.0), (2, 0.5)]
    moved = _check(result, "locations", "moved")
    assert [(r["id"], r["distance"]) for r in moved.affected_records] == [(1, 0.5)]
    geographic = compare_datasets(
        squares, after, keys=["id"],
        levels=[{"level": "locations", "geometry": "g", "area": {"rtol": 0.6}, "shape": None}],
    )  # fmt: skip
    # One square degree at the equator is about 1.23e10 m^2 on the WGS 84 spheroid.
    assert _check(geographic, "locations", "area").passed
    assert geographic.level("locations").unit == "metres"


# -------------------------------------------------------------------- zone x time
def test_assign_zones_with_temporal_compares_each_zone_bucket_by_bucket(tmp_path):
    days = [date(2025, 1, 1) + timedelta(days=k) for k in range(5)]
    rows = [(i + 0.5, j + 0.5, day, 10.0) for i in range(2) for j in range(2) for day in days]
    before = pl.DataFrame(rows, schema=["lon", "lat", "day", "trips"], orient="row")
    after = before.with_columns(
        pl.when((pl.col("lon") < 1) & (pl.col("lat") < 1) & (pl.col("day") == days[2]))
        .then(pl.col("trips") + 5)
        .otherwise(pl.col("trips"))
        .alias("trips")
    )
    steps = [{"step": "assign_zones", "x": "lon", "y": "lat", "layer": {"path": _layer(tmp_path)}}]
    level = {"level": "temporal", "time_column": "day", "bucket": "day", "group_by": ["zone"],
             "measures": ["sum(trips)"], "decision": "material"}  # fmt: skip
    result = compare_datasets(before, after, steps=steps, levels=[level])
    sums = _check(result, "temporal", "sum(trips)")
    assert sums.affected_count == 1
    record = sums.affected_records[0]
    assert (record["zone"], record["bucket"], record["diff"]) == ("z0_0", days[2], 5)
    assert result.level("temporal").baseline_buckets == 4 * 5


# ------------------------------------------------------------------ fail closed
def _without_extension(monkeypatch):
    def unavailable(con):
        raise SpatialUnavailable("DuckDB's spatial extension could not be loaded (simulated)")

    monkeypatch.setattr(extension, "load_spatial", unavailable)


def test_a_missing_extension_leaves_geometry_checks_not_assessed(tmp_path, monkeypatch):
    layer = _layer(tmp_path)
    _without_extension(monkeypatch)
    result = _spatial(_points(1), _points(1), layer)
    assert not result.assessed and not result.passed
    assert "spatial extension could not be loaded (simulated)" in str(result.error)
    assert all(not check.assessed for level in result.levels for check in level.checks)
    # Attribute zones still compare; only the checks that need polygons are not assessed.
    level = {"level": "spatial", "zone_column": "truth", "layer": {"path": layer}}
    plain = compare_datasets(_points(1), _points(2), levels=[level])
    statuses = {c.check: c.assessed for c in plain.level("spatial").checks}
    assert statuses["zones"] and statuses["count"] and statuses["expected"]
    assert not statuses["moran"] and not statuses["lisa"]
    moran = _check(plain, "spatial", "moran")
    assert "simulated" in moran.message
    shape = compare_datasets(
        pl.DataFrame({"k": [1], "g": [S.wkb_point(0, 0)]}),
        pl.DataFrame({"k": [1], "g": [S.wkb_point(0, 0)]}),
        keys=["k"], levels=[{"level": "locations", "geometry": "g"}],
    )  # fmt: skip
    assert not shape.level("locations").assessed


# ----------------------------------------------------------------- integration
def test_cmp001_runs_spatial_levels_through_validate(tmp_path):
    zones = S.wkt_frame(S.grid_zones(N))
    config = {
        "datasets": {"v1": {}, "v2": {}, "zones": {}},
        "rules": {
            "CMP-001": {
                "params": {
                    "comparisons": [
                        {
                            "name": "trips",
                            "baseline": "v1",
                            "candidate": "v2",
                            "steps": [
                                {
                                    "step": "assign_zones",
                                    "x": "lon",
                                    "y": "lat",
                                    "layer": {"dataset": "zones"},
                                }
                            ],
                            "levels": [{"level": "spatial", "severity": "WARNING"}],
                        }
                    ]
                }
            }
        },
        "engine": {"run_unconfigured_rules": False},
        "plugins": {"entry_points": False},
    }
    report = validate({"v1": _points(3), "v2": _points(4, 1.6), "zones": zones}, config=config)
    results = {r.check: r for r in report.results_for("CMP-001")}
    assert results["trips: spatial: lisa"].status is Status.FAIL
    assert results["trips: spatial: lisa"].severity.name == "WARNING"
    assert results["trips: spatial: expected"].status is Status.PASS
    names = report.results_for("CMP-001")[0].affected_records or []
    assert isinstance(names, list)
    clean = validate({"v1": _points(5), "v2": _points(6), "zones": zones}, config=config)
    assert all(r.status is Status.PASS for r in clean.results_for("CMP-001"))


def test_unknown_layer_datasets_are_configuration_errors():
    from qavf.engine.compilation import ConfigurationError

    config = {
        "datasets": {"v1": {}, "v2": {}},
        "rules": {"CMP-001": {"params": {"comparisons": [{
            "baseline": "v1", "candidate": "v2",
            "steps": [{"step": "assign_zones", "x": "lon", "y": "lat",
                       "layer": {"dataset": "nowhere"}}],
        }]}}},
    }  # fmt: skip
    with pytest.raises(ConfigurationError, match="unknown dataset 'nowhere'"):
        validate({"v1": _points(1), "v2": _points(1)}, config=config)


def test_documented_spatial_configuration_validates():
    import re

    import yaml

    from qavf.rules.comparison_checks import DatasetComparisonRule
    from qavf.settings import QAVFSettings

    text = (Path(__file__).resolve().parents[1] / "docs" / "spatial.md").read_text("utf-8")
    (configuration,) = re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
    settings = QAVFSettings.model_validate(yaml.safe_load(configuration))
    rule = DatasetComparisonRule(settings.rules["CMP-001"].params)
    assert rule.configuration_errors(settings) == []
    assert rule.data_requirements(settings).datasets == {"trips_2024", "trips_2025", "zones"}


def test_results_round_trip_and_an_isolated_run_agrees(tmp_path):
    layer = _layer(tmp_path)
    _points(3).write_parquet(tmp_path / "v1.parquet")
    _points(4, 1.6).write_parquet(tmp_path / "v2.parquet")
    steps = [{"step": "assign_zones", "x": "lon", "y": "lat", "layer": {"path": layer}}]
    levels = ["spatial", {"level": "locations", "x": "lon", "y": "lat"}]
    local = compare_datasets(
        tmp_path / "v1.parquet", tmp_path / "v2.parquet", steps=steps, levels=levels[:1]
    )
    again = DatasetComparison.model_validate_json(local.model_dump_json())
    assert again == local and type(again.level("spatial")).__name__ == "SpatialResult"
    if sys.platform not in SUPPORTED_PLATFORMS:
        return
    isolated = compare_datasets(
        File(tmp_path / "v1.parquet"),
        File(tmp_path / "v2.parquet"),
        steps=steps,
        levels=levels[:1],
        resources=ResourceLimits(isolate=True, wall_time=180),
    )
    assert isolated.model_dump(exclude={"execution_limits"}) == local.model_dump(
        exclude={"execution_limits"}
    )


def test_spatial_components_are_built_in_whatever_the_plugin_choice(tmp_path):
    steps = [{"step": "assign_zones", "x": "lon", "y": "lat", "layer": {"path": _layer(tmp_path)}}]
    for enabled in (False, True):
        result = compare_datasets(
            _points(1), _points(1), steps=steps, levels=["spatial"], entry_points=enabled
        )
        assert result.passed, result.summary()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"step": "assign_zones", "x": "lon"}, "exactly one zone source"),
        ({"step": "assign_zones", "x": "lon", "layer": {"path": "z.geojson"}}, "'x' and 'y'"),
        ({"step": "assign_zones", "zones": "taz"}, "needs a 'crosswalk'"),
        (
            {
                "step": "assign_zones",
                "x": "a",
                "y": "b",
                "layer": {"path": "z.parquet"},
                "grid": {"size": 1},
            },
            "exactly one zone source",
        ),
        ({"step": "assign_zones", "x": "a", "y": "b", "grid": {"size": 0}}, "greater than 0"),
        ({"step": "assign_zones", "x": "a", "y": "b", "grid": {"size": 1}, "crs": "x"}, "CRS"),
    ],
)
def test_step_configuration_errors_fail_before_reading_data(options, message):
    frame = pl.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match=message):
        compare_datasets(frame, frame, steps=[options])


@pytest.mark.parametrize(
    ("level", "message"),
    [
        ({"level": "spatial", "pattern_measure": "sum(x)"}, "pattern_measure"),
        ({"level": "spatial", "weights": {"method": "distance_band"}}, "need a 'distance'"),
        ({"level": "spatial", "difference": "nope"}, "unknown comparator"),
        ({"level": "locations"}, "'x' and 'y'"),
        ({"level": "locations", "x": "a", "y": "b"}, "keys"),
    ],
)
def test_level_configuration_errors_fail_before_reading_data(level, message):
    frame = pl.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match=message):
        compare_datasets(frame, frame, levels=[level])
