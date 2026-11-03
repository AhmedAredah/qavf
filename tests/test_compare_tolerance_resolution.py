"""``rtol`` and ``atol`` resolve independently: measure, column, level, comparison."""

from datetime import date
from itertools import product

import polars as pl
import pytest

from qavf import compare_datasets
from qavf.compare.levels.contribution import ContributionParams
from qavf.compare.levels.distribution import DistributionParams
from qavf.compare.levels.groups import GroupsParams, MeasureSpec, materiality
from qavf.compare.levels.relationships import RelationshipsParams
from qavf.compare.levels.spatial import SpatialParams
from qavf.compare.levels.temporal import TemporalParams
from qavf.compare.spec import ComparisonSpec

# Every level whose materiality goes through ``materiality()``.
LEVEL_PARAMS = [
    GroupsParams,
    ContributionParams,
    DistributionParams,
    RelationshipsParams,
    SpatialParams,
    TemporalParams,
]

# Distinct values per scope, so the scope each field came from is visible.
COMPARISON = (0.2, 2.0)
LEVEL_SETTINGS = {
    "none": {},
    "rtol": {"rtol": 0.3},
    "atol": {"atol": 3.0},
    "both": {"rtol": 0.3, "atol": 3.0},
}
COLUMN_SETTINGS = {
    "none": None,
    "rtol": {"rtol": 0.4},
    "atol": {"atol": 4.0},
    "both": {"rtol": 0.4, "atol": 4.0},
    "exact": {"exact": True},
}
MEASURE_SETTINGS = {
    "none": {},
    "rtol": {"rtol": 0.5},
    "atol": {"atol": 5.0},
    "both": {"rtol": 0.5, "atol": 5.0},
    "exact": {"exact": True},
}


def _expected(level: dict, column: dict | None, measure: dict) -> tuple[float, float]:
    if measure.get("exact"):
        return 0.0, 0.0
    column = {"rtol": 0.0, "atol": 0.0} if column and column.get("exact") else (column or {})
    rtol = measure.get("rtol", column.get("rtol", level.get("rtol", COMPARISON[0])))
    atol = measure.get("atol", column.get("atol", level.get("atol", COMPARISON[1])))
    return rtol, atol


def _spec(column: dict | None) -> ComparisonSpec:
    tolerances = {"x": column} if column is not None else {}
    return ComparisonSpec(
        baseline="b",
        candidate="c",
        rtol=COMPARISON[0],
        atol=COMPARISON[1],
        tolerances=tolerances,
        levels=[{"level": "structure"}],
    )


@pytest.mark.parametrize("params", LEVEL_PARAMS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    ("level", "column", "measure"),
    list(product(LEVEL_SETTINGS, COLUMN_SETTINGS, MEASURE_SETTINGS)),
)
def test_each_tolerance_field_comes_from_its_nearest_scope(params, level, column, measure):
    spec = _spec(COLUMN_SETTINGS[column])
    configured = params.model_construct(**{"rtol": None, "atol": None, **LEVEL_SETTINGS[level]})
    judged = materiality(
        spec, configured, MeasureSpec(measure="mean", column="x", **MEASURE_SETTINGS[measure])
    )
    assert (judged.rtol, judged.atol) == _expected(
        LEVEL_SETTINGS[level], COLUMN_SETTINGS[column], MEASURE_SETTINGS[measure]
    )


@pytest.mark.parametrize("params", LEVEL_PARAMS, ids=lambda cls: cls.__name__)
def test_a_columnless_measure_uses_the_level_then_the_comparison(params):
    spec = _spec({"atol": 4.0})
    configured = params.model_construct(rtol=0.3, atol=None)
    judged = materiality(spec, configured, MeasureSpec(measure="count"))
    assert (judged.rtol, judged.atol) == (0.3, COMPARISON[1])


def test_record_values_keep_the_comparison_field_a_column_leaves_unset():
    assert _spec({"atol": 4.0}).tolerance("x") == (COMPARISON[0], 4.0)
    assert _spec({"rtol": 0.4}).tolerance("x") == (0.4, COMPARISON[1])
    assert _spec({"exact": True}).tolerance("x") == (0.0, 0.0)
    assert _spec(None).tolerance("x") == COMPARISON
    assert _spec(None).column_tolerance("x") == (None, None)


GROUPS = [{"level": "groups", "group_by": ["g"], "measures": ["mean(x)"], "decision": "material"}]
BASELINE = pl.DataFrame({"g": ["a", "a"], "x": [100.0, 100.0]})
CANDIDATE = pl.DataFrame({"g": ["a", "a"], "x": [101.0, 101.0]})


def _material(result, level="groups"):
    return {measure.label: measure.material for measure in result.level(level).measures}


def test_a_partial_column_override_keeps_a_looser_level_rtol():
    level = [{**GROUPS[0], "rtol": 0.1}]
    plain = compare_datasets(BASELINE, CANDIDATE, levels=level)
    partial = compare_datasets(BASELINE, CANDIDATE, levels=level, tolerances={"x": {"atol": 0.01}})
    assert _material(plain) == _material(partial) == {"mean(x)": 0}
    assert plain.passed and partial.passed


def test_a_partial_column_override_keeps_a_stricter_level_rtol():
    level = [{**GROUPS[0], "rtol": 0.001}]
    result = compare_datasets(
        BASELINE, CANDIDATE, levels=level, rtol=0.5, tolerances={"x": {"atol": 0.01}}
    )
    assert _material(result) == {"mean(x)": 1}
    assert not result.passed


def test_column_exact_still_overrides_the_level():
    level = [{**GROUPS[0], "rtol": 0.1}]
    result = compare_datasets(BASELINE, CANDIDATE, levels=level, tolerances={"x": {"exact": True}})
    assert _material(result) == {"mean(x)": 1}


def test_temporal_level_keeps_its_rtol_beside_a_partial_column_override():
    days = [date(2025, 1, 1), date(2025, 1, 2)]
    baseline = pl.DataFrame({"t": days, "x": [100.0, 100.0]})
    candidate = pl.DataFrame({"t": days, "x": [101.0, 101.0]})
    level = {
        "level": "temporal",
        "time_column": "t",
        "bucket": "day",
        "measures": ["sum(x)"],
        "rtol": 0.1,
        "decision": "material",
    }
    result = compare_datasets(baseline, candidate, levels=[level], tolerances={"x": {"atol": 0.01}})
    assert result.passed
