"""Exact grouping identities and numeric comparison boundaries."""

import json
import math
import struct
from datetime import date
from fractions import Fraction

import polars as pl
import pytest

from qavf import compare_datasets
from qavf.compare.comparators import ComparatorContext, LogRatio
from qavf.compare.levels.spatial import SpatialLevel


def compare(baseline, candidate, **options):
    return compare_datasets(baseline, candidate, entry_points=False, **options)


def check(result, name):
    return next(c for c in result.levels[0].checks if c.check == name)


def key_series(kind, values):
    if kind == "nan":
        return pl.Series("g", [float("nan") if x == 0 else 1.0 for x in values])
    dtype = {"datetime": pl.Datetime("ns"), "duration": pl.Duration("ns"), "time": pl.Time}[kind]
    return pl.Series("g", values, dtype=pl.Int64).cast(dtype)


@pytest.mark.parametrize("kind", ["datetime", "duration", "time"])
def test_rank_preserves_submicrosecond_group_identity(kind):
    b = pl.DataFrame({"g": key_series(kind, [0, 1]), "x": [2, 1]})
    c = b.with_columns(pl.Series("x", [1, 2]))
    r = compare(
        b,
        c,
        decision="material",
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": ["sum(x)"],
                "rank": {"measure": "sum(x)", "top_k": 1},
            }
        ],
    )
    assert not check(r, "rank: sum(x)").passed
    assert (r.levels[0].rank.entered, r.levels[0].rank.left) == (1, 1)
    evidence = check(r, "rank: sum(x)").affected_records
    assert len({row["g"] for row in evidence}) == 2


def test_identical_nan_rank_group_does_not_change_membership():
    frame = pl.DataFrame({"g": [float("nan"), 1.0], "x": [2, 1]})
    r = compare(
        frame,
        frame,
        decision="material",
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": ["sum(x)"],
                "rank": {"measure": "sum(x)", "top_k": 1},
            }
        ],
    )
    assert r.passed


@pytest.mark.parametrize(
    "values,measure", [([None, None], "sum(x)"), ([date(2020, 1, 1), date(2020, 1, 2)], "min(x)")]
)
def test_undefined_or_nonnumeric_top_k_is_not_a_rank_pass(values, measure):
    frame = pl.DataFrame({"g": ["a", "b"], "x": values})
    r = compare(
        frame,
        frame,
        decision="material",
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": [measure],
                "rank": {"measure": measure, "top_k": 1},
            }
        ],
    )
    assert r.verdict == "incomplete"
    assert not check(r, f"rank: {measure}").assessed
    assert r.levels[0].rank is None


def test_empty_wkb_is_unlocated_not_an_assessed_centroid_match():
    empty_polygon = struct.pack("<BII", 1, 3, 0)
    frame = pl.DataFrame({"id": [1], "geometry": [empty_polygon]})
    r = compare(
        frame,
        frame,
        keys=["id"],
        decision="material",
        levels=[
            {
                "level": "locations",
                "geometry": "geometry",
                "crs": "EPSG:32614",
            }
        ],
    )
    assert r.verdict == "incomplete"
    assert check(r, "located").metrics["unlocated_on_both"] == 1
    assert not check(r, "moved").assessed


@pytest.mark.parametrize("geographic", [False, True])
def test_point_distances_keep_a_concrete_schema_for_owned_streaming(geographic):
    # NumPy ufunc expressions have an Unknown lazy dtype on Polars 1.32;
    # the owned observation must still stream a typed distance column to IPC.
    baseline = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "x": [0.0, 0.0, None, 0.0],
            "y": [0.0, 0.0, 0.0, None],
        }
    )
    candidate = baseline.with_columns(
        pl.Series("x", [1.0 if geographic else 3.0, 0.0, None, 0.0]),
        pl.Series("y", [0.0 if geographic else 4.0, 0.0, 0.0, None]),
    )
    result = compare(
        baseline.lazy(),
        candidate.lazy(),
        keys=["id"],
        decision="material",
        levels=[
            {
                "level": "locations",
                "x": "x",
                "y": "y",
                "crs": "EPSG:4326" if geographic else "EPSG:3857",
            }
        ],
    )
    moved = check(result, "moved")
    expected = 6_371_008.8 * math.radians(1) if geographic else 5.0
    assert result.verdict == "failed"
    assert moved.assessed and moved.affected_count == 1 and moved.total == 2
    assert moved.affected_records[0]["id"] == 0
    assert moved.affected_records[0]["distance"] == pytest.approx(expected, rel=1e-12)
    assert result.levels[0].unlocated == 2


def test_layer_outline_reader_supports_the_actual_duckdb_relation_api(monkeypatch):
    import duckdb

    from qavf.spatial import outlines

    # Substitute only the spatial SQL; keep the actual version's relation and
    # Arrow reader so this runs even when the floor's spatial extension is absent.
    ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    shape = json.dumps({"type": "Polygon", "coordinates": [ring]})
    ids = pl.Series([f"zone-{i}" for i in range(130)])
    monkeypatch.setattr(
        outlines, "fetch", lambda *_: pl.DataFrame({"x0": [0], "x1": [1], "y0": [0], "y1": [1]})
    )
    with duckdb.connect(config={"threads": 2}) as con:
        class SpatialSQL:
            def sql(self, query, *, params):
                assert "ST_AsGeoJSON" in query and "ORDER BY zone_order" in query
                assert params == [outlines.SIMPLIFY_FRACTION]
                return con.sql(
                    "SELECT i AS zone_order, ? AS shape FROM range(130) t(i) ORDER BY i",
                    params=[shape],
                )

        actual = outlines.layer_rings(SpatialSQL(), "zones", ids)

    assert list(actual) == ids.to_list()
    assert actual == {zone: [ring] for zone in ids}


@pytest.mark.parametrize("kind", ["datetime", "duration", "time", "nan"])
def test_categorical_minimum_applies_to_each_native_group(kind):
    frame = pl.DataFrame({"g": key_series(kind, [0, 1]), "x": ["a", "a"]})
    r = compare(
        frame,
        frame,
        decision="material",
        levels=[
            {
                "level": "distribution",
                "group_by": ["g"],
                "columns": ["x"],
                "min_values": 2,
            }
        ],
    )
    assert r.verdict == "incomplete"
    assert check(r, "x").affected_count == 2
    assert r.levels[0].columns[0].compared == 2


def test_nan_categorical_group_is_measured_once():
    frame = pl.DataFrame({"g": key_series("nan", [0, 0, 1, 1]), "x": ["a"] * 4})
    r = compare(
        frame,
        frame,
        decision="material",
        levels=[
            {
                "level": "distribution",
                "group_by": ["g"],
                "columns": ["x"],
                "min_values": 2,
            }
        ],
    )
    assert r.passed
    assert r.levels[0].columns[0].compared == 2


def grouped_pair():
    b = pl.DataFrame(
        {"id": range(6), "g": key_series("datetime", [0] * 3 + [1] * 3), "x": [1, 2, 3, 4, 5, 6]}
    )
    return b, b.with_columns(pl.Series("x", [11, 12, 13, 4, 5, 6]))


def test_agreement_cannot_discard_failing_nanosecond_partition():
    b, c = grouped_pair()
    r = compare(
        b,
        c,
        keys=["id"],
        decision="material",
        levels=[
            {
                "level": "relationships",
                "group_by": ["g"],
                "agreement": {
                    "columns": ["x"],
                    "min_ccc": None,
                    "max_slope_deviation": None,
                    "max_abs_bias": 0,
                },
            }
        ],
    )
    assert r.verdict == "failed"
    assert (check(r, "agreement: x").total, check(r, "agreement: x").affected_count) == (2, 1)
    assert len(r.levels[0].agreement) == 2


def test_shift_cannot_discard_failing_nanosecond_partition():
    pytest.importorskip("scipy")
    b, c = grouped_pair()
    r = compare(b, c, keys=["id"], atol=0, rtol=0, levels=[{"level": "records", "shift_by": ["g"]}])
    assert not check(r, "shift").passed
    assert check(r, "shift").total == 2
    assert r.levels[0].shifted == ["x"]


@pytest.mark.parametrize("kind", ["datetime", "nan"])
@pytest.mark.parametrize("minimum", [3, 4])
def test_series_group_identity_and_minimum_are_preserved(kind, minimum):
    frame = pl.DataFrame({"g": key_series(kind, [0] * 3 + [1] * 3), "t": range(6), "x": range(6)})
    r = compare(
        frame,
        frame,
        decision="material",
        levels=[
            {
                "level": "series_shape",
                "time_column": "t",
                "group_by": ["g"],
                "measure": "sum(x)",
                "min_buckets": minimum,
                "correlation": None,
                "trend": {},
                "peak": None,
                "lag": None,
                "level_shift": None,
            }
        ],
    )
    assert r.levels[0].compared == 2
    assert r.levels[0].too_short == (2 if minimum == 4 else 0)
    assert r.passed == (minimum == 3)


def test_lossy_paired_inference_cannot_be_certified_immaterial():
    b = pl.DataFrame({"id": [1, 2, 3], "x": [2**53 + 1] * 3})
    c = b.with_columns(pl.Series("x", [2**53 + 3] * 3))
    r = compare(b, c, keys=["id"], levels=["records"], rtol=0, atol=0)
    assert not check(r, "shift").assessed
    assert check(r, "shift").significance.not_assessed == 1


def test_finite_large_shift_mean_does_not_overflow():
    pytest.importorskip("scipy")
    b = pl.DataFrame({"id": [1, 2, 3], "x": [1e308] * 3})
    c = b.with_columns(pl.Series("x", [1.1e308] * 3))
    r = compare(b, c, keys=["id"], levels=["records"], rtol=0, atol=0)
    assert check(r, "shift").assessed and not check(r, "shift").passed
    assert r.levels[0].shifted == ["x"]


@pytest.mark.parametrize("values", [[None] * 3, [float("nan")] * 3])
def test_no_finite_pairs_leave_shift_unassessed(values):
    frame = pl.DataFrame({"id": [1, 2, 3], "x": pl.Series(values, dtype=pl.Float64)})
    r = compare(frame, frame, keys=["id"], levels=["records"])
    assert r.verdict == "incomplete"
    assert not check(r, "shift").assessed


def test_over_budget_shift_cannot_use_a_lossy_mean_to_pass():
    b = pl.DataFrame({"id": [1, 2, 3], "x": [2**60] * 3})
    c = b.with_columns(pl.Series("x", [2**60 + 1] * 3))
    r = compare(b, c, keys=["id"], levels=["records"], max_work=1, rtol=0, atol=0)
    assert not check(r, "shift").assessed
    assert "max_work" in check(r, "shift").message


@pytest.mark.parametrize(
    "before,after", [(-(2**127), 2**127 - 1), (2**127 - 2, 2**127 - 1), (None, 1)]
)
def test_int128_magnitude_never_wraps(before, after):
    b = pl.DataFrame({"id": [1], "x": pl.Series([before], dtype=pl.Int128)})
    c = pl.DataFrame({"id": [1], "x": pl.Series([after], dtype=pl.Int128)})
    r = compare(b, c, keys=["id"], levels=["records"], decision="material", rtol=0, atol=0)
    assert check(r, "values").assessed and not check(r, "values").passed
    assert r.levels[0].columns[0].max_abs_diff == (
        None if before is None else float(abs(after - before))
    )


@pytest.mark.parametrize(
    "before,after", [(10**16, 10**16 + 2), (10**300, 10**300 + 10**285), (10**16, 10**16 - 2)]
)
def test_log_ratio_retains_small_relative_changes(before, after):
    b, c = Fraction(before), Fraction(after)
    assert LogRatio().compare(b, c, ComparatorContext()) == math.log1p(float((c - b) / b))


def test_spatial_unknown_measure_is_not_a_zero_share():
    frame = pl.DataFrame({"z": ["a", "b"], "x": pl.Series([None, 100.0], dtype=pl.Float64)})
    r = compare(
        frame,
        frame,
        decision="material",
        levels=[
            {
                "level": "spatial",
                "zone_column": "z",
                "measures": ["sum(x)"],
                "expected": False,
                "moran": None,
                "lisa": None,
                "map": False,
            }
        ],
    )
    assert not check(r, "shares").assessed


def test_moran_preserves_small_exact_variation_at_large_offset():
    b = pl.DataFrame({"z": [f"{i}_0" for i in range(4)], "x": [0] * 4})
    config = {
        "level": "spatial",
        "zone_column": "z",
        "measures": ["sum(x)"],
        "expected": False,
        "shares": None,
        "lisa": None,
        "map": False,
        "grid": {"size": 1},
        "moran": {"min_i": 0.1},
    }
    small = compare(
        b, b.with_columns(pl.Series("x", range(4))), decision="material", levels=[config]
    )
    large = compare(
        b,
        b.with_columns(pl.Series("x", [2**60 + i for i in range(4)])),
        decision="material",
        levels=[config],
    )
    assert not check(large, "moran").passed
    assert large.levels[0].morans_i == small.levels[0].morans_i


def test_spatial_zone_ids_remain_native_nanoseconds(monkeypatch):
    seen = []
    original = SpatialLevel._rank

    def rank(self, check, zones, inference, details):
        seen.extend(zones.ids.cast(pl.Int64).to_list())
        return original(self, check, zones, inference, details)

    monkeypatch.setattr(SpatialLevel, "_rank", rank)
    frame = pl.DataFrame({"z": key_series("datetime", [0, 1, 2]), "x": [1, 2, 3]})
    r = compare(
        frame,
        frame,
        decision="material",
        levels=[
            {
                "level": "spatial",
                "zone_column": "z",
                "measures": ["sum(x)"],
                "expected": False,
                "rank": {},
                "shares": None,
                "moran": None,
                "lisa": None,
                "map": False,
            }
        ],
    )
    assert r.passed
    assert seen == [0, 1, 2]


def test_series_trend_inference_is_invariant_to_exact_large_translation():
    pytest.importorskip("scipy")
    config = {
        "level": "series_shape",
        "time_column": "t",
        "measure": "sum(x)",
        "min_buckets": 4,
        "correlation": None,
        "trend": {"rtol": 0, "atol": 0, "dependence": "none"},
        "peak": None,
        "lag": None,
        "level_shift": None,
    }
    outcomes = []
    for offset in [0, 2**60]:
        b = pl.DataFrame({"t": range(8), "x": [offset + i for i in range(8)]})
        c = b.with_columns(pl.Series("x", [offset + 2 * i for i in range(8)]))
        outcomes.append(compare(b, c, levels=[config]))
    assert all(check(r, "trend").assessed and not check(r, "trend").passed for r in outcomes)
    assert (
        outcomes[0].levels[0].checks[-1].significance.failed
        == outcomes[1].levels[0].checks[-1].significance.failed
    )
