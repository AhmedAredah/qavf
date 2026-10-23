"""Public comparison verdicts and resource-work bounds."""

from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf import Dataset, compare_datasets
from qavf.compare.align import AlignmentContext
from qavf.compare.comparators import Materiality
from qavf.compare.levels import LevelInputs
from qavf.compare.levels.spatial import SpatialLevel, SpatialParams, _Zones
from qavf.compare.significance import Inference
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.inference.results import TestResult
from qavf.spatial.weights import SpatialWeights


def compare(base, candidate, **options):
    return compare_datasets(base, candidate, entry_points=False, decision="material", **options)


@pytest.mark.parametrize("decimal", [False, True])
def test_exact_agreement_preserves_small_change_at_large_offset(decimal):
    if decimal:
        values = pl.Series(
            "x",
            [Decimal("100000000000000000000.00") + Decimal(i) for i in (0, 4, 8, 12)],
            dtype=pl.Decimal(30, 2),
        )
        changed = values + Decimal("0.01")
        expected = 0.01
    else:
        values = pl.Series("x", [2**53 + i for i in (0, 4, 8, 12)])
        changed = values + 1
        expected = 1
    base = pl.DataFrame({"id": range(4), "x": values})
    candidate = base.with_columns(changed.alias("x"))
    result = compare(
        base,
        candidate,
        keys="id",
        levels=[
            {
                "level": "relationships",
                "agreement": {
                    "columns": ["x"],
                    "max_mae": 0,
                    "max_abs_bias": 0,
                },
            }
        ],
    )
    assert result.assessed and not result.passed, result.summary()
    detail = result.level("relationships").agreement[0]
    assert detail.mae == expected and detail.bias == expected
    assert detail.slope == 1 and detail.intercept == expected


def test_column_swap_keeps_units_and_nullable_attached_to_values():
    context = AlignmentContext(
        "baseline",
        "b",
        count=lambda _: 0,
        units={"a": "kg", "b": "m"},
        nullable={"a": True, "b": False},
    )
    context.rename({"a": "b", "b": "a"})
    assert context.units == {"a": "m", "b": "kg"}
    assert context.nullable == {"a": False, "b": True}
    base = Dataset(
        pl.DataFrame({"a": [1], "b": [2]}),
        schema={
            "columns": {
                "a": {"unit": "kg"},
                "b": {"unit": "m"},
            }
        },
    )
    candidate = Dataset(
        pl.DataFrame({"a": [2], "b": [1]}),
        schema={
            "columns": {
                "a": {"unit": "kg"},
                "b": {"unit": "kg"},
            }
        },
    )
    result = compare(base, candidate, columns={"a": "b", "b": "a"}, levels=["structure"])
    assert not result.passed
    assert [unit.column for unit in result.level("structure").unit_changes] == ["a"]


@pytest.mark.parametrize("grouped", [False, True])
def test_all_missing_categorical_members_remain_unassessed(grouped):
    base = pl.DataFrame({"g": ["a", "a", "b", "b"], "x": ["yes", "no", None, None]})
    candidate = base.with_columns(
        x=pl.when(pl.col("g") == "b").then(pl.lit("new")).otherwise(pl.col("x"))
    )
    if not grouped:
        base, candidate = base.tail(2), candidate.tail(2)
    result = compare(
        base,
        candidate,
        levels=[
            {
                "level": "distribution",
                "columns": ["x"],
                "group_by": ["g"] if grouped else [],
            }
        ],
    )
    assert not result.assessed and not result.passed
    summary = result.level("distribution").columns[0]
    assert summary.compared == (2 if grouped else 1)


def test_exact_rank_changes_even_when_magnitude_tolerance_allows_change():
    n = 2**53
    base = pl.DataFrame({"g": ["a", "b"], "x": [n, n + 1]})
    candidate = base.with_columns(pl.Series("x", [n + 1, n]))
    result = compare(
        base,
        candidate,
        atol=1,
        rtol=0,
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": ["sum(x)"],
                "rank": {"measure": "sum(x)", "top_k": 1},
            }
        ],
    )
    rank = result.level("groups").rank
    assert rank.entered == rank.left == 1
    assert not result.passed


def test_lisa_labels_follow_original_indices(monkeypatch):
    zones = _Zones(
        pl.Series("zone", ["excluded", "A", "B", "C"]),
        list(map(Fraction, [0, 1, 2, 3])),
        list(map(Fraction, [0, 11, 22, 33])),
    )
    kept = [1, 2, 3]
    weights = SpatialWeights(
        zones.ids.gather(kept), {0: {1: 1.0}, 1: {0: 1.0, 2: 1.0}, 2: {1: 1.0}}, "test"
    )
    known = SimpleNamespace(
        quadrants=["HH"] * 3,
        results=[TestResult(method="local_morans_i", statistic=1, p_value=0) for _ in kept],
    )
    monkeypatch.setattr("qavf.compare.levels.spatial.local_morans_i", lambda *a, **kw: known)
    result = SpatialLevel._lisa(
        SimpleNamespace(ctx=SimpleNamespace(evidence_limit=10)),
        SpatialParams(),
        zones,
        kept,
        weights,
        np.array([10.0, 20.0, 30.0]),
        False,
        Materiality(0, 0),
        Inference("material_and_significant", 0.05, "none", 1, 1000),
        {},
    )
    assert {(r["zone"], r["baseline"], r["candidate"]) for r in result.affected_records} == {
        ("A", 1, 11),
        ("B", 2, 22),
        ("C", 3, 33),
    }


def test_correlation_admits_pair_work_and_bounds_live_samples(monkeypatch):
    import qavf.compare.levels.relationships as relationships

    frame = pl.DataFrame({f"c{i}": [j + i for j in range(20)] for i in range(6)})
    level = {
        "level": "relationships",
        "correlation": {
            "columns": frame.columns,
            "methods": ["pearson"],
        },
    }
    refused = compare(frame, frame, levels=[level], max_work=240)
    assert not refused.assessed and "max_work" in refused.levels[0].checks[0].message
    peaks = []
    original = relationships.reduce_groups

    def reduced(*args, **kwargs):
        result = original(*args, **kwargs)
        peaks.append(
            sum(
                int(f[c].list.len().sum())
                for f in result.frames.values()
                for c, dtype in f.schema.items()
                if isinstance(dtype, pl.List)
            )
        )
        return result

    monkeypatch.setattr(relationships, "reduce_groups", reduced)
    monkeypatch.setattr(relationships, "spearman", lambda *args: pytest.fail("unselected method"))
    assert compare(frame, frame, levels=[level], max_work=1200).passed
    assert max(peaks) == 80  # two columns, two sides, twenty rows; not all 15 pairs


def test_empty_level_selection_and_unusable_locations_never_pass():
    frame = pl.DataFrame({"id": [1], "x": [0.0], "y": [0.0]})
    with pytest.raises(ValidationError, match="at least one"):
        compare(frame, frame, levels=[])
    level = {"level": "locations", "x": "x", "y": "y", "crs": "EPSG:3857"}
    for candidate in [frame.with_columns(id=pl.lit(2)), frame.with_columns(x=pl.lit(None))]:
        result = compare(frame, candidate, keys="id", levels=[level])
        assert not result.assessed and not result.passed


def test_native_measures_share_one_reduction_per_side(monkeypatch):
    import qavf.compare.levels.groups as groups

    frame = pl.DataFrame({f"c{i}": list(range(10)) for i in range(6)})
    original, calls = groups.grouped_reduce, []

    def reduce(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(groups, "grouped_reduce", reduce)
    assert compare(
        frame,
        frame,
        levels=[
            {
                "level": "groups",
                "measures": [f"max({c})" for c in frame.columns],
            }
        ],
    ).passed
    assert len(calls) == 2


def test_projection_drops_unused_payload_and_noop_reuses_observation(monkeypatch):
    frame = pl.DataFrame({"g": ["a", "b"], "x": [1, 2], "payload": ["x" * 10000] * 2})
    original, requests = DataContext.projected_observations, []
    derived, derivations = Observation.derived, []

    def observe(self, selections):
        requests.append(selections)
        return original(self, selections)

    def derive(self, build):
        derivations.append(1)
        return derived(self, build)

    monkeypatch.setattr(DataContext, "projected_observations", observe)
    monkeypatch.setattr(Observation, "derived", derive)
    assert compare(
        frame,
        frame,
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": ["max(x)"],
            }
        ],
    ).passed
    assert all(columns == ["g", "x"] for columns in requests[0].values())
    assert not derivations


def test_distribution_display_cap_keeps_probability_mass_and_exact_metrics():
    base = pl.DataFrame({"x": [0.0] * 100})
    candidate = pl.DataFrame({"x": list(map(float, range(100)))})
    result = compare(
        base,
        candidate,
        levels=[
            {
                "level": "distribution",
                "columns": ["x"],
                "bins": 5,
            }
        ],
    )
    member = result.level("distribution").members[0]
    assert len(member.bins) == 11 and member.bins[-1].label == "Other (90 bins)"
    assert sum(bin.candidate for bin in member.bins) == pytest.approx(1)
    assert member.ks == 0.99


def test_point_location_summaries_only_collect_bounded_results(monkeypatch):
    frame = pl.DataFrame({"id": range(5000), "x": [0.0] * 5000, "y": [0.0] * 5000})
    original, sizes = LevelInputs.collect, []

    def collect(self, plan):
        result = original(self, plan)
        sizes.append(result.height)
        return result

    monkeypatch.setattr(LevelInputs, "collect", collect)
    result = compare(
        frame,
        frame.with_columns(x=pl.lit(1.0)),
        keys="id",
        levels=[
            {
                "level": "locations",
                "x": "x",
                "y": "y",
                "crs": "EPSG:3857",
            }
        ],
    )
    assert result.assessed and result.level("locations").moved == 5000
    assert max(sizes) < 5000


def test_projection_keeps_full_schema_alignment_validation():
    frame = pl.DataFrame({"a": [1], "b": [2], "unused": [3]})
    level = {"level": "groups", "measures": ["max(b)"]}
    rename = compare(frame, frame, levels=[level], columns={"a": "b"})
    assert not rename.assessed and "several baseline columns" in rename.error
    derive = compare(frame, frame, levels=[level], derive={"unused": "a + 1"})
    assert not derive.assessed and "already exist" in derive.error


def test_typed_configuration_rebinds_inputs_and_rejects_overrides():
    from qavf.compare import ComparisonSpec

    frame = pl.DataFrame({"id": [1], "x": [2]})
    spec = ComparisonSpec(baseline="old", candidate="new", keys=("id",), decision="material")
    result = compare_datasets(frame, frame, configuration=spec, entry_points=False)
    assert result.passed and result.baseline == "baseline" and result.candidate == "candidate"
    with pytest.raises(ValueError, match="cannot be combined"):
        compare_datasets(frame, frame, configuration=spec, rtol=0, entry_points=False)


def test_disjoint_groups_are_unassessed_even_when_missing_groups_are_ignored():
    result = compare(
        pl.DataFrame({"g": ["a"]}),
        pl.DataFrame({"g": ["b"]}),
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "missing_groups": "ignore",
            }
        ],
    )
    assert not result.assessed and not result.passed


def test_exact_agreement_threshold_uses_declared_decimal_not_binary_float():
    base = pl.DataFrame(
        {
            "id": range(3),
            "x": pl.Series(
                [
                    Decimal("0"),
                    Decimal("4"),
                    Decimal("8"),
                ],
                dtype=pl.Decimal(30, 18),
            ),
        }
    )
    candidate = base.with_columns(x=pl.col("x") + Decimal("1.100000000000000001"))
    result = compare(
        base,
        candidate,
        keys="id",
        levels=[
            {
                "level": "relationships",
                "agreement": {
                    "columns": ["x"],
                    "max_mae": 1.1,
                    "max_abs_bias": 1.1,
                    "max_rmse": 1.1,
                    "min_ccc": None,
                    "max_slope_deviation": None,
                },
            }
        ],
    )
    assert result.assessed and not result.passed
    assert set(result.level("relationships").agreement[0].material) == {"bias", "mae", "rmse"}


def test_deming_intercept_preserves_unit_shift_at_large_offset():
    base = pl.DataFrame({"id": range(4), "x": [10**18 + i for i in range(4)]})
    candidate = base.with_columns(x=pl.col("x") + 1)
    result = compare(
        base,
        candidate,
        keys="id",
        levels=[
            {
                "level": "relationships",
                "agreement": {
                    "columns": ["x"],
                    "fit": "deming",
                    "min_ccc": None,
                    "max_slope_deviation": None,
                    "max_abs_intercept": 0,
                },
            }
        ],
    )
    assert result.assessed and not result.passed
    assert result.level("relationships").agreement[0].intercept == 1
