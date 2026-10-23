"""Spatial and CLI boundary and allocation contracts."""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from qavf._adjacency import CompactAdjacency
from qavf.cli import _comparison_spec, main
from qavf.configuration import read_config_mapping
from qavf.keys import KeyTypeError, cast_key
from qavf.spatial import crs
from qavf.spatial._neighbours import band_graph, nearest_graph
from qavf.spatial.distance import distance, euclidean
from qavf.spatial.hierarchy import apply_crosswalk
from qavf.spatial.weights import distance_band, from_table, nearest
from qavf.temporal.buckets import Bucket, sequence_positions


@pytest.mark.parametrize(
    "suffix, text",
    [
        (".yaml", "levels: [records]\nlevels: [structure]\n"),
        (".json", '{"levels": ["records"], "levels": ["structure"]}'),
    ],
)
def test_comparison_configuration_rejects_duplicate_checks(tmp_path, suffix, text):
    path = tmp_path / f"comparison{suffix}"
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate key"):
        _comparison_spec(path)


def test_malformed_comparison_configuration_has_usage_exit(tmp_path, capsys):
    path = tmp_path / "invalid.yaml"
    path.write_text("levels: [records\n")
    assert main(["compare-datasets", "left.parquet", "right.parquet", "-c", str(path)]) == 2
    assert "invalid YAML" in capsys.readouterr().err


def test_shared_config_preserves_yaml_merge_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("QAVF_REVIEW_TEST_KEY", "id")
    path = tmp_path / "comparison.yaml"
    path.write_text(
        "base: &base {atol: 1}\noptions: {<<: *base, atol: 0}\nkeys: ['${QAVF_REVIEW_TEST_KEY}']\n"
    )
    parsed, _ = read_config_mapping(path)
    assert parsed["options"] == {"atol": 0}
    assert parsed["keys"] == ["id"]


def test_spatial_fractional_ids_cannot_join_integer_zones():
    frame = pl.DataFrame({"zone": [1], "value": [10]})
    crosswalk = pl.DataFrame({"from": [1.9], "to": ["B"], "weight": [1.0]})
    with pytest.raises(ValueError, match="without changing identity"):
        apply_crosswalk(frame.lazy(), "zone", crosswalk, [], "_weight")
    with pytest.raises(ValueError, match="without changing identity"):
        from_table(pl.Series([1, 2]), pl.DataFrame({"from": [1.9], "to": [2.9], "weight": [1.0]}))


def test_fixed_key_cast_preserves_exact_identities():
    assert cast_key(pl.Series([1.0, None, 2.0]), pl.Int64).to_list() == [1, None, 2]
    with pytest.raises(KeyTypeError):
        cast_key(pl.Series([1.1]), pl.Int64)
    with pytest.raises(KeyTypeError):
        cast_key(pl.Series([16_777_217], dtype=pl.Int64), pl.Float32)
    with pytest.raises(KeyTypeError):
        cast_key(pl.Series(["01"]), pl.Int64)
    with pytest.raises(KeyTypeError):
        cast_key(pl.Series([2**64 - 1], dtype=pl.UInt64), pl.Int64)


def test_authoritative_geographic_classification_and_units():
    pytest.importorskip("pyproj")
    crs._is_geographic.cache_clear()
    assert crs.is_geographic("EPSG:4267")
    assert not crs.is_geographic("EPSG:2263")  # Projected US survey feet stay CRS units.
    with pytest.raises(ValueError, match="must use degrees"):
        crs.is_geographic("EPSG:4807")  # NTF (Paris) uses grads.
    with pytest.raises(ValueError, match="cannot resolve"):
        crs.is_geographic("UNKNOWN:123")


def test_unknown_crs_fails_closed_without_optional_resolver():
    crs._is_geographic.cache_clear()
    with patch.dict("sys.modules", {"pyproj": None}):
        assert crs.is_geographic("EPSG:4326")
        assert not crs.is_geographic("EPSG:32614")
        with pytest.raises(ValueError, match=r"install qavf\[spatial\]"):
            crs.is_geographic("EPSG:4267")
    crs._is_geographic.cache_clear()


def _oracle(x, y, k, geographic):
    values = distance(x[:, None], y[:, None], x[None, :], y[None, :], geographic=geographic)
    np.fill_diagonal(values, np.inf)
    return [sorted(np.argsort(row, kind="stable")[:k].tolist()) for row in values]


@pytest.mark.parametrize("geographic", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
def test_neighbours_match_full_distance_oracle_including_ties(geographic, indexed):
    rng = np.random.default_rng(351)
    x = rng.uniform(-180, 180, 120) if geographic else rng.normal(size=120)
    y = rng.uniform(-89, 89, 120) if geographic else rng.normal(size=120)
    x[:5], y[:5] = 0, 0  # Duplicate coordinates and k-th-distance ties.
    context = (
        patch("qavf.spatial._neighbours._tree", return_value=None)
        if not indexed
        else patch.dict("os.environ", {})
    )
    with context:
        graph = nearest(pl.Series(range(len(x))), x, y, 4, geographic=geographic)
    assert [list(row) for row in graph.neighbours.values()] == _oracle(x, y, 4, geographic)
    assert isinstance(graph.neighbours, CompactAdjacency)
    assert graph.neighbours.nbytes <= (len(x) + 1) * 8 + len(x) * 4 * 4
    assert graph.standardized().neighbours[0] == {i: 0.25 for i in graph.neighbours[0]}


def test_dateline_poles_and_inclusive_band_match_distance_oracle():
    x = np.array([-180.0, 180.0, 179.9, -179.9, 0.0, 0.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 90.0, -90.0])
    assert [
        list(r) for r in nearest(pl.Series(range(6)), x, y, 2, geographic=True).neighbours.values()
    ] == _oracle(x, y, 2, True)
    threshold = float(distance(x[0], y[0], x[2], y[2], geographic=True))
    graph = distance_band(pl.Series(range(6)), x, y, threshold, geographic=True)
    values = distance(x[:, None], y[:, None], x[None, :], y[None, :], geographic=True)
    expected = [
        np.flatnonzero((row <= threshold) & (np.arange(6) != i)).tolist()
        for i, row in enumerate(values)
    ]
    assert [list(r) for r in graph.neighbours.values()] == expected


def test_graph_budgets_prevent_unbounded_work_and_retention():
    x = np.arange(100.0)
    with (
        patch("qavf.spatial._neighbours._tree", side_effect=AssertionError("admission too late")),
        pytest.raises(ValueError, match="max_edges"),
    ):
        nearest_graph(x, x, 100, 4, geographic=False, max_edges=399, max_work=1_000)
    with pytest.raises(ValueError, match="max_edges"):
        band_graph(x, x, 100, 1000, geographic=False, max_edges=100, max_work=20_000)
    with (
        patch("qavf.spatial._neighbours._tree", return_value=None),
        pytest.raises(ValueError, match="max_work"),
    ):
        nearest_graph(x, x, 100, 1, geographic=False, max_edges=100, max_work=500)


def test_indexed_sparse_graph_avoids_all_pairs():
    pytest.importorskip("scipy")
    x = np.arange(1000.0)
    graph = nearest_graph(
        x, np.zeros(1000), 1000, 2, geographic=False, max_edges=2000, max_work=5000
    )
    assert graph.pair_count == 2000


def test_compact_adjacency_owns_arrays_and_rejects_invalid_edges():
    offsets, indices = np.array([0, 1, 2]), np.array([1, 0])
    graph = CompactAdjacency(offsets, indices)
    indices[:] = 0
    assert graph[0] == {1: 1.0}
    with pytest.raises(ValueError, match="out-of-range"):
        CompactAdjacency(np.array([0, 1]), np.array([2]))
    with pytest.raises(ValueError, match="sorted unique"):
        CompactAdjacency(np.array([0, 2, 2]), np.array([1, 1]))


@pytest.mark.parametrize("zone", [None, "UTC", "Etc/GMT-5"])
def test_sparse_fixed_offset_positions_do_not_expand_calendar(zone):
    labels = pl.Series([datetime(2020, 1, 1), None, datetime(2021, 1, 1)]).dt.replace_time_zone(
        zone
    )
    with patch("qavf.temporal.buckets.bucket_grid", side_effect=AssertionError("dense grid")):
        positions = sequence_positions(labels, Bucket(unit="second"), limit=2)
    assert positions.to_list() == [0, None, 31_622_400]


def test_sparse_tick_positions_preserve_alignment_and_bounds():
    labels = pl.Series([datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 0, 1)])
    with pytest.raises(ValueError, match="outside"):
        sequence_positions(labels, Bucket(unit="minute"))
    with pytest.raises(ValueError, match="outside"):
        sequence_positions(labels, Bucket(unit="second"), last=labels.head(1))


def test_euclidean_distance_avoids_intermediate_square_overflow():
    assert float(euclidean(0.0, 0.0, 1e200, 0.0)) == 1e200


@pytest.mark.parametrize("scale", [1e-200, 1e160, 1e200])
@pytest.mark.parametrize("indexed", [False, True])
def test_extreme_planar_coordinates_preserve_finite_distances(scale, indexed):
    if indexed:
        pytest.importorskip("scipy")
    x = np.array([-2.0, -0.5, 0.0, 1.0, 2.0]) * scale
    y = np.zeros(5)
    context = (
        patch("qavf.spatial._neighbours._tree", return_value=None)
        if not indexed
        else patch.dict("os.environ", {})
    )
    with context:
        graph = nearest(pl.Series(range(5)), x, y, 2, geographic=False)
        band = distance_band(pl.Series(range(5)), x, y, scale, geographic=False)
    assert [list(row) for row in graph.neighbours.values()] == _oracle(x, y, 2, False)
    values = distance(x[:, None], y[:, None], x[None, :], y[None, :], geographic=False)
    expected = [
        np.flatnonzero((row <= scale) & (np.arange(5) != i)).tolist()
        for i, row in enumerate(values)
    ]
    assert [list(row) for row in band.neighbours.values()] == expected


@pytest.mark.parametrize("indexed", [False, True])
def test_geographic_graph_rejects_noncanonical_longitudes(indexed):
    context = (
        patch("qavf.spatial._neighbours._tree", return_value=None)
        if not indexed
        else patch.dict("os.environ", {})
    )
    with context:
        with pytest.raises(ValueError, match="longitudes"):
            nearest(pl.Series([0, 1]), np.array([0, 1e16]), np.array([0, 0]), 1, geographic=True)
        with pytest.raises(ValueError, match="longitudes"):
            distance_band(
                pl.Series([0, 1]), np.array([0, 1e16]), np.array([0, 0]), 1, geographic=True
            )
