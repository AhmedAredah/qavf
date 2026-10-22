"""GEO-001 ``within`` and SPA-003 expected entities from a zone layer, with the existing
configurations unchanged."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.rules.spatial_checks import CoverageParams
from qavf.settings import QAVFSettings
from qavf.spatial import extension
from qavf.spatial.extension import SpatialUnavailable
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


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
# A study area: the square from (-98, 30) to (-96, 32), as a WKT dataset.
AREA = S.wkt_frame({"study": S.square(-98.0, 30.0, 2.0)})
POINTS = pl.DataFrame(
    {
        "latitude": [31.0, 31.5, 33.0, 30.0, 91.0, None],
        "longitude": [-97.0, -96.5, -97.0, -97.0, -97.0, -97.0],
    }
)
"""Inside, inside, outside (north), on the south boundary, invalid, missing."""


def _run(frames, rules):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {name: {} for name in frames},
            "rules": rules,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


def _geo(**points):
    report = _run(
        {"t": POINTS, "area": AREA},
        {"GEO-001": {"params": {"points": [{"dataset": "t", **points}]}}},
    )
    return {result.check: result for result in report.results_for("GEO-001")}


def _unavailable(monkeypatch):
    def fail(con):
        raise SpatialUnavailable("DuckDB's spatial extension could not be loaded (simulated)")

    monkeypatch.setattr(extension, "load_spatial", fail)


# ------------------------------------------------------------------ GEO-001
def test_geo001_without_within_is_unchanged():
    results = _geo(allow_missing=True)
    (result,) = results.values()
    assert result.status is Status.FAIL and result.affected_count == 1
    assert [r["problem"] for r in result.affected_records] == ["latitude out of range"]
    assert result.check == "t(latitude, longitude)"


def test_points_outside_the_study_area_fail_a_separate_within_check():
    results = _geo(allow_missing=True, within={"dataset": "area", "id": "zone"})
    coordinates = results["t(latitude, longitude)"]
    assert coordinates.affected_count == 1  # the invalid latitude, as before
    within = results["t(latitude, longitude) within area"]
    assert within.status is Status.FAIL and within.affected_count == 1
    assert within.total_records == 4  # the valid coordinates only
    assert within.affected_records == [{"latitude": 33.0, "longitude": -97.0}]


def test_boundary_points_follow_the_within_policy():
    within = {"dataset": "area", "id": "zone"}
    strict = _geo(allow_missing=True, within=within, boundary="contains")
    outside = strict["t(latitude, longitude) within area"]
    assert outside.affected_count == 2  # (30, -97) lies on the boundary
    assert {r["latitude"] for r in outside.affected_records} == {30.0, 33.0}


def test_within_is_not_assessed_without_the_extension(monkeypatch):
    _unavailable(monkeypatch)
    results = _geo(allow_missing=True, within={"dataset": "area", "id": "zone"})
    within = results["t(latitude, longitude) within area"]
    assert within.status is Status.ERROR and "simulated" in within.message
    assert results["t(latitude, longitude)"].status is Status.FAIL  # still assessed


# ------------------------------------------------------------------ SPA-003
SITES = pl.DataFrame({"zone": ["a", "b"], "value": [1.0, 2.0]})
LAYER = S.wkt_frame({"a": S.square(0, 0), "b": S.square(1, 0), "c": S.square(2, 0)})


def _coverage(params, frames=None):
    frames = frames or {"sites": SITES, "zones": LAYER}
    report = _run(frames, {"SPA-003": {"params": params}})
    return report.results_for("SPA-003")


def test_expected_entities_come_from_a_zone_layer_without_the_extension(monkeypatch):
    _unavailable(monkeypatch)
    params = {
        "entity_column": "zone",
        "datasets": ["sites"],
        "zone_layer": {"dataset": "zones", "id": "zone"},
    }
    (result,) = _coverage(params)
    assert result.status is Status.FAIL and result.affected_count == 1
    assert result.affected_records == [{"dataset": "sites", "missing_entities": ["c"]}]


def test_zone_layer_files_and_strict_coverage(tmp_path):
    path = S.geojson(tmp_path / "zones.geojson", {"a": S.square(0, 0), "b": S.square(1, 0)})
    extra = pl.DataFrame({"zone": ["a", "b", "x"], "value": [1.0, 2.0, 3.0]})
    params = {
        "entity_column": "zone",
        "datasets": ["sites"],
        "zone_layer": {"path": str(path)},
        "unexpected_entities": True,
    }
    (result,) = _coverage(params, {"sites": extra})
    assert result.status is Status.FAIL
    assert result.affected_records == [
        {"dataset": "sites", "missing_entities": [], "unexpected_entities": ["x"]}
    ]
    params["unexpected_entities"] = False
    (result,) = _coverage(params, {"sites": extra})
    assert result.status is Status.PASS


def test_auto_discovery_does_not_check_the_layer_itself():
    params = {"entity_column": "zone", "zone_layer": {"dataset": "zones", "id": "zone"}}
    (result,) = _coverage(params)
    assert result.status is Status.FAIL
    assert [record["dataset"] for record in result.affected_records] == ["sites"]


def test_spa003_configurations_without_a_layer_are_unchanged():
    params = {"entity_column": "zone", "expected_entities": ["a", "b"], "datasets": ["sites"]}
    (result,) = _coverage(params)
    assert result.status is Status.PASS
    both = {**params, "reference_dataset": "sites"}
    CoverageParams.model_validate(both)  # still accepted together, as before
    with pytest.raises(ValidationError, match="zone_layer"):
        CoverageParams.model_validate({**params, "zone_layer": {"dataset": "zones"}})
