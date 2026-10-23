"""Independent parser, graph representation and early-admission regressions."""

import numpy as np
import polars as pl
import pytest

from qavf._adjacency import CompactAdjacency
from qavf.configuration import read_config_mapping
from qavf.spatial._neighbours import band_graph, nearest_graph
from qavf.spatial.weights import SpatialWeights, distance_band, nearest


def test_yaml_merge_and_literal_merge_name_remain_distinct(tmp_path):
    path = tmp_path / "merged.yaml"
    path.write_text('base: &base {atol: 1}\noptions: {<<: *base, "<<": literal}\n')
    parsed, _ = read_config_mapping(path)
    assert parsed["options"] == {"atol": 1, "<<": "literal"}


@pytest.mark.parametrize(
    "mapping",
    ['{<<: *base, <<: *base}', '{"<<": first, "<<": second}'],
)
def test_yaml_duplicate_merge_or_literal_keys_still_rejected(tmp_path, mapping):
    path = tmp_path / "duplicate.yaml"
    path.write_text(f"base: &base {{atol: 1}}\noptions: {mapping}\n")
    with pytest.raises(ValueError, match="duplicate key '<<'"):
        read_config_mapping(path)


@pytest.mark.parametrize("offsets", [[0.0, 1.0, 2.0], [0.0, 0.5, 2.0], [False, True, True]])
def test_compact_adjacency_requires_integer_offsets(offsets):
    with pytest.raises(ValueError, match="offsets must be integers"):
        CompactAdjacency(np.array(offsets), np.array([1, 0]))


def test_compact_adjacency_rejects_wrapped_offsets():
    with pytest.raises(ValueError, match="offsets or indices"):
        CompactAdjacency(
            np.array([0, 2**63 - 1, 2**64 - 2, 0], dtype=np.uint64),
            np.array([], dtype=np.int64),
        )


@pytest.mark.parametrize("indices", [[1, 0], [1, 1]])
def test_compact_adjacency_rejects_unsigned_unsorted_or_duplicate_indices(indices):
    with pytest.raises(ValueError, match="sorted unique"):
        CompactAdjacency(np.array([0, 2, 2]), np.array(indices, dtype=np.uint64))


def test_compact_zero_rows_standardize_like_mapping_weights():
    ids = pl.Series(["a", "b", "c"])
    mapping = {0: {1: 0.0}, 1: {0: 2.0, 2: 0.0}, 2: {}}
    compact = CompactAdjacency(
        np.array([0, 1, 3, 3]), np.array([1, 0, 2]), np.array([0.0, 2.0, 0.0])
    )
    expected = SpatialWeights(ids, mapping, "test").standardized()
    result = SpatialWeights(ids, compact, "test").standardized()
    assert result.neighbours == expected.neighbours
    assert result.pairs == expected.pairs == 2
    assert result.islands == expected.islands == [0, 2]
    assert result.counts() == expected.counts() == [0, 2, 0]
    assert result.standardized().neighbours == result.neighbours


def test_numpy_integer_counts_and_budgets_are_supported(monkeypatch):
    monkeypatch.setattr("qavf.spatial._neighbours._tree", lambda points: None)
    ids = pl.Series(range(3))
    coordinates = np.arange(3.0)
    graph = nearest(
        ids, coordinates, np.zeros(3), np.int64(1), geographic=False,
        max_edges=np.uint64(3), max_work=np.int64(100),
    )
    assert [list(row) for row in graph.neighbours.values()] == [[1], [0], [1]]
    band = distance_band(
        ids, coordinates, np.zeros(3), 1, geographic=False,
        max_edges=np.int64(4), max_work=np.uint64(100),
    )
    assert band.pairs == 4


class _NoCoordinates:
    def __array__(self, *args, **kwargs):
        pytest.fail("coordinate conversion occurred before admission")


@pytest.mark.parametrize("budget", ["edges", "work"])
def test_nearest_rejects_known_excess_before_converting_coordinates(budget):
    with pytest.raises(ValueError, match=f"max_{budget}"):
        nearest_graph(
            _NoCoordinates(), _NoCoordinates(), 100, 4, geographic=False,
            max_edges=399 if budget == "edges" else 400,
            max_work=99 if budget == "work" else 1000,
        )


def test_band_rejects_known_excess_before_converting_coordinates():
    with pytest.raises(ValueError, match="max_work"):
        band_graph(
            _NoCoordinates(), _NoCoordinates(), 100, 1, geographic=False,
            max_edges=100, max_work=99,
        )


@pytest.mark.parametrize("k", [True, np.bool_(True), 1.0])
def test_noninteger_neighbour_counts_are_rejected_before_conversion(k):
    with pytest.raises(ValueError, match="k must lie"):
        nearest_graph(
            _NoCoordinates(), _NoCoordinates(), 3, k, geographic=False,
            max_edges=10, max_work=100,
        )
