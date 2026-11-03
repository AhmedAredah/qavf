"""Zone-layer files are read under the run's memory target, like any other input."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.resource_errors import ResourceLimitError
from qavf.sources.base import SourceError, SourceIntegrityError
from qavf.spatial import layers
from qavf.spatial.extension import spatial_connection
from qavf.spatial.layers import ZoneLayer, layer_frame, zone_ids


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

TINY = {"memory_budget_bytes": 64, "cache_max_bytes": 0, "spill_max_bytes": 0}
REFUSED = (ResourceLimitError, SourceError)


@pytest.fixture
def parquet_layer(tmp_path):
    return ZoneLayer(path=str(S.geoparquet(tmp_path / "zones.parquet", S.grid_zones(30))))


@pytest.fixture
def native_layer(tmp_path):
    return ZoneLayer(path=str(S.geojson(tmp_path / "zones.geojson", S.grid_zones(30))))


def test_geoparquet_zone_ids_are_refused_over_the_memory_target(parquet_layer):
    with DataContext(**TINY) as ctx, pytest.raises(REFUSED):
        zone_ids(ctx, parquet_layer)


def test_geoparquet_geometries_are_refused_over_the_memory_target(parquet_layer):
    with DataContext(**TINY) as ctx, pytest.raises(REFUSED):
        layer_frame(ctx, parquet_layer, None)


def test_geoparquet_layers_read_within_the_target(parquet_layer):
    with DataContext() as ctx:
        ids = zone_ids(ctx, parquet_layer)
        frame = layer_frame(ctx, parquet_layer, None)
    assert ids.len() == frame.height == 900
    assert zone_ids(None, parquet_layer).to_list() == ids.to_list()


def test_a_missing_geoparquet_id_column_still_names_the_column(tmp_path):
    layer = ZoneLayer(path=str(S.geoparquet(tmp_path / "z.parquet", S.grid_zones(2))), id="taz")
    with DataContext() as ctx, pytest.raises((ValueError, SourceError), match="taz"):
        zone_ids(ctx, layer)


class _RecordingFetch:
    """``fetch`` that records each statement."""

    def __init__(self, original):
        self.original, self.statements = original, []

    def __call__(self, con, sql, parameters=None):
        self.statements.append(sql)
        return self.original(con, sql, parameters)


class _ChangingFetch:
    """``fetch`` that appends to the layer file after reading it."""

    def __init__(self, original, path):
        self.original, self.path = original, path

    def __call__(self, con, sql, parameters=None):
        frame = self.original(con, sql, parameters)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(" ")
        return frame


def _unbudgeted_connection(ctx=None):
    """A spatial connection outside the run's native memory limit, so the tests
    reach QAVF's own admission of the fetched result."""
    return spatial_connection(None)


def test_native_zone_ids_are_refused_before_the_fetch(native_layer, monkeypatch):
    recording = _RecordingFetch(layers.fetch)
    monkeypatch.setattr(layers, "fetch", recording)
    monkeypatch.setattr(layers, "spatial_connection", _unbudgeted_connection)
    with DataContext(**TINY) as ctx, pytest.raises(ResourceLimitError, match="zone layer"):
        zone_ids(ctx, native_layer)
    assert len(recording.statements) == 1 and "sum(" in recording.statements[0]


def test_native_geometries_are_refused_before_the_fetch(native_layer, monkeypatch):
    recording = _RecordingFetch(layers.fetch)
    monkeypatch.setattr(layers, "fetch", recording)
    with (
        DataContext(**TINY) as ctx,
        spatial_connection(None) as con,
        pytest.raises(ResourceLimitError, match="zone layer"),
    ):
        layer_frame(ctx, native_layer, con)
    assert [sql for sql in recording.statements if "ST_Read(" in sql][-1].count("sum(") == 1


def test_native_layers_read_within_the_target(native_layer):
    with DataContext() as ctx, spatial_connection(ctx) as con:
        frame = layer_frame(ctx, native_layer, con)
        assert zone_ids(ctx, native_layer).len() == frame.height == 900
    assert frame.schema["geometry"] == pl.Binary


def test_a_layer_file_changed_during_the_read_is_refused(native_layer, monkeypatch):
    monkeypatch.setattr(layers, "fetch", _ChangingFetch(layers.fetch, native_layer.path))
    with DataContext() as ctx, pytest.raises(SourceIntegrityError, match="changed"):
        zone_ids(ctx, native_layer)
