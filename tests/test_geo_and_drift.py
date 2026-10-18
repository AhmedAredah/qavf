"""GEO-001 coordinate validity and STA-002 distribution drift."""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.rules.geo_checks import BoundingBox
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


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


# ------------------------------------------------------------------ GEO-001
POINTS = pl.DataFrame(
    {
        "latitude": [30.6, 97.0, 0.0, None, 91.0, 30.0],
        "longitude": [-96.3, 30.6, 0.0, 1.0, 200.0, -150.0],
    }
)
TEXAS = {"min_lat": 25, "max_lat": 37, "min_lon": -107, "max_lon": -93}


def _geo(frame, **points):
    return _run(
        {"t": frame}, {"GEO-001": {"params": {"points": [{"dataset": "t", **points}]}}}
    ).get("GEO-001")


def test_each_kind_of_invalid_coordinate_is_named() -> None:
    result = _geo(POINTS, bbox=TEXAS)
    assert result.status is Status.FAIL and result.affected_count == 5
    problems = {r["_row"]: r["problem"] for r in result.affected_records}
    assert "swapped" in problems[1] and "(0, 0)" in problems[2]
    assert problems[3] == "missing coordinate" and problems[4] == "latitude out of range"
    assert problems[5] == "outside the bounding box"


def test_options_relax_the_checks() -> None:
    frame = pl.DataFrame({"latitude": [30.6, None, 0.0], "longitude": [-96.3, None, 0.0]})
    assert _geo(frame).affected_count == 2
    assert _geo(frame, allow_missing=True, null_island=False).status is Status.PASS


def test_text_coordinates_are_unassessed() -> None:
    frame = pl.DataFrame({"latitude": ["30.6"], "longitude": ["-96.3"]})
    [result] = _run(
        {"t": frame}, {"GEO-001": {"params": {"points": [{"dataset": "t"}]}}}
    ).results_for("GEO-001")
    assert result.status is Status.ERROR and "not numeric" in result.message


def test_bounding_boxes_are_validated() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(min_lat=40, max_lat=30, min_lon=0, max_lon=1)
    with pytest.raises(ValidationError):
        BoundingBox(min_lat=0, max_lat=95, min_lon=0, max_lon=1)


# ------------------------------------------------------------------ STA-002
RNG = np.random.default_rng(0)
REFERENCE = pl.DataFrame(
    {"x": RNG.normal(100, 10, 2000), "c": RNG.choice(["a", "b", "c"], 2000, p=[0.5, 0.3, 0.2])}
)


def _drift(current, **comparison):
    spec = {"dataset": "cur", "reference": "ref", "column": "x", **comparison}
    return _run(
        {"ref": REFERENCE, "cur": current}, {"STA-002": {"params": {"comparisons": [spec]}}}
    )


def test_psi_passes_a_stable_and_fails_a_shifted_distribution() -> None:
    same = pl.DataFrame({"x": RNG.normal(100, 10, 2000)})
    assert _drift(same).get("STA-002").status is Status.PASS
    shifted = pl.DataFrame({"x": RNG.normal(110, 10, 2000)})
    result = _drift(shifted).get("STA-002")
    assert result.status is Status.FAIL and result.metrics["psi"] > 0.2


def test_psi_works_on_categories() -> None:
    current = pl.DataFrame({"c": RNG.choice(["a", "b", "c", "d"], 2000, p=[0.2, 0.2, 0.2, 0.4])})
    spec = {"dataset": "cur", "reference": "ref", "column": "c"}
    report = _run(
        {"ref": REFERENCE, "cur": current}, {"STA-002": {"params": {"comparisons": [spec]}}}
    )
    assert report.get("STA-002").status is Status.FAIL


def test_small_samples_are_unassessed() -> None:
    [result] = _drift(pl.DataFrame({"x": [1.0, 2.0]})).results_for("STA-002")
    assert result.status is Status.ERROR and "min_samples" in result.message


@pytest.mark.skipif(not HAS_SCIPY, reason="the KS test needs the stats extra")
def test_ks_needs_both_significance_and_effect_size() -> None:
    tiny_shift = pl.DataFrame({"x": RNG.normal(100.5, 10, 20000)})
    big = pl.concat([REFERENCE.select("x")] * 10)
    spec = {"dataset": "cur", "reference": "ref", "column": "x", "method": "ks"}
    report = _run({"ref": big, "cur": tiny_shift}, {"STA-002": {"params": {"comparisons": [spec]}}})
    assert report.get("STA-002").status is Status.PASS  # significant, but immaterial
    shifted = pl.DataFrame({"x": RNG.normal(108, 10, 2000)})
    result = _drift(shifted, method="ks").get("STA-002")
    assert result.status is Status.FAIL and result.metrics["alpha_bonferroni"] == 0.01


@pytest.mark.skipif(HAS_SCIPY, reason="checks the behaviour without the stats extra")
def test_ks_without_scipy_is_unassessed() -> None:
    [result] = _drift(pl.DataFrame({"x": RNG.normal(100, 10, 200)}), method="ks").results_for(
        "STA-002"
    )
    assert result.status is Status.ERROR and "stats extra" in result.message
