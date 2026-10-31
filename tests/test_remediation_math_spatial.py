"""Independent regressions for typed arithmetic, ownership and spatial construction."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from qavf import Dataset, File, Loader, ResourceLimits, compare_datasets
from qavf._adjacency import CompactAdjacency
from qavf.compare.spec import ComparisonSpec
from qavf.engine.context import DataContext
from qavf.inference.effects import PairedValues
from qavf.inference.trend import seasonal_kendall
from qavf.sources.base import SourceError
from qavf.spatial import _neighbours
from qavf.spatial.grid import Grid, assign_cells
from qavf.spatial.hierarchy import Crosswalk, crosswalk_frame
from qavf.temporal.buckets import Bucket, TimeBucketing, bucket_labels
from qavf.typed_values import exact_scale


@pytest.mark.parametrize("factor", [1.00000001, 1e40, -1.00000001])
def test_float32_scaling_uses_binary64_arithmetic(factor):
    frame = pl.DataFrame({"x": pl.Series([1.0, None], dtype=pl.Float32)})
    result = frame.select(exact_scale(pl.col("x"), pl.Float32, factor))["x"]
    assert result.dtype == pl.Float64
    assert result.to_list() == [float(factor), None]


@pytest.mark.parametrize("factor", [Decimal("1e1000"), Decimal("1e-1000")])
def test_scale_factor_must_fit_nonzero_finite_float64(factor):
    with pytest.raises(ValueError, match="finite and nonzero in Float64"):
        exact_scale(pl.col("x"), pl.Float32, factor)


@pytest.mark.parametrize("value,factor", [(1e308, 10), (1e-308, 1e-100)])
def test_float_scaling_refuses_lost_finite_products(value, factor):
    frame = pl.DataFrame({"x": [value]})
    with pytest.raises(ValueError, match="overflow or underflow"):
        frame.select(exact_scale(pl.col("x"), pl.Float64, factor))


def test_float_scaling_keeps_explicit_nonfinite_input_policy():
    frame = pl.DataFrame({"x": [0.0, math.nan, math.inf, None]})
    result = frame.select(exact_scale(pl.col("x"), pl.Float64, 2))["x"]
    assert result[0] == 0 and math.isnan(result[1]) and result[2] == math.inf
    assert result[3] is None


def test_public_float32_units_cannot_round_a_real_difference_to_a_pass():
    baseline = pl.DataFrame({"id": [1], "x": pl.Series([1.0], dtype=pl.Float32)})
    candidate = baseline.with_columns(pl.col("x").cast(pl.Float64))
    options = {
        "keys": ["id"],
        "levels": ["records"],
        "units": {"x": {"factor": 1.00000001}},
        "rtol": 0,
        "atol": 0,
        "decision": "material",
        "entry_points": False,
    }
    assert compare_datasets(baseline, candidate, **options).verdict == "failed"
    assert compare_datasets(candidate, candidate, **options).verdict == "failed"


@pytest.mark.parametrize("every", [1, 3, 6])
def test_nanosecond_bucket_boundaries_match_unbounded_integer_oracle(every):
    width_us = every * 3_600_000_000
    values = [-(2**63) + 1, -(2**63) + 1000, -3600000000001, -1, 0, 1, 2**63 - 1, None]
    dtype = pl.Datetime("ns")
    frame = pl.DataFrame({"t": pl.Series(values, dtype=pl.Int64).cast(dtype)})
    expression, output_dtype = bucket_labels(
        "t", dtype, TimeBucketing(bucket=Bucket(unit="hour", every=every))
    )
    result = frame.select(expression.alias("t"))["t"]
    expected = [None if v is None else (v // (width_us * 1000)) * width_us for v in values]
    assert output_dtype == pl.Datetime("us")
    assert result.cast(pl.Int64).to_list() == expected


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_subday_floor_preserves_negative_ticks(unit):
    dtype = pl.Datetime(unit)
    frame = pl.DataFrame({"t": pl.Series([-1, 0, 1], dtype=pl.Int64).cast(dtype)})
    expression, _ = bucket_labels("t", dtype, TimeBucketing(bucket=Bucket(unit="second")))
    assert frame.select(expression.cast(pl.Int64))["t"].to_list() == [-1_000_000, 0, 0]


def test_unrepresentable_microsecond_bucket_is_refused():
    dtype = pl.Datetime("us")
    frame = pl.DataFrame({"t": pl.Series([-(2**63)], dtype=pl.Int64).cast(dtype)})
    expression, _ = bucket_labels("t", dtype, TimeBucketing(bucket=Bucket(unit="hour")))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        frame.select(expression)


def test_empty_transformed_grid_needs_no_connection_and_preserves_schema():
    frame = pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64, "payload": pl.String})
    assigned = assign_cells(
        frame.lazy(),
        Grid(size=10, crs="EPSG:3857"),
        x="x",
        y="y",
        crs="EPSG:4326",
        column="zone",
        collect=lambda plan: plan.collect(),
    )
    assert assigned.plan.collect().schema == frame.schema | {"zone": pl.String}
    assert (assigned.rows, assigned.unassigned, assigned.ties, assigned.overlaps) == (0, 0, 0, 0)


def test_empty_grid_does_not_bypass_coordinate_schema_or_duplicate_column_errors():
    frame = pl.DataFrame(schema={"x": pl.Float64, "zone": pl.String})
    arguments = dict(x="x", y="y", crs="EPSG:4326", collect=lambda plan: plan.collect())
    with pytest.raises(ValueError, match="already exists"):
        assign_cells(frame.lazy(), Grid(size=10, crs="EPSG:3857"), column="zone", **arguments)
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        assign_cells(frame.lazy(), Grid(size=10, crs="EPSG:3857"), column="cell", **arguments)


def test_reusable_paired_metrics_own_one_consistent_snapshot():
    baseline = np.array([1.0, 2.0, 4.0])
    candidate = np.array([2.0, 3.0, 5.0])
    values = PairedValues(baseline, candidate)
    before = (values.bias(), values.mape(), values.smape(), values.r_squared())
    baseline[:] = 100
    candidate[:] = -100
    assert (values.bias(), values.mape(), values.smape(), values.r_squared()) == before
    assert values.bias() == 1
    assert values.mape() == pytest.approx(100 * (1 + 0.5 + 0.25) / 3)
    for array in (values.b, values.c, values.sb, values.sc, values.difference):
        assert not array.flags.writeable


@pytest.mark.parametrize("tree", [False, True])
def test_knn_preallocation_matches_brute_force_tie_oracle(monkeypatch, tree):
    if tree:
        pytest.importorskip("scipy.spatial")
    else:
        monkeypatch.setattr(_neighbours, "_tree", lambda points: None)
    monkeypatch.setattr(_neighbours, "_BLOCK_PAIRS", 8)
    points = np.array([[0, 0], [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    graph = _neighbours.nearest_graph(
        points[:, 0],
        points[:, 1],
        len(points),
        3,
        geographic=False,
        max_edges=18,
        max_work=1000,
    )
    for i in range(len(points)):
        expected = sorted(
            sorted(
                (j for j in range(len(points)) if j != i),
                key=lambda j: (float(np.linalg.norm(points[i] - points[j])), j),
            )[:3]
        )
        assert list(graph[i]) == expected
    assert graph.nbytes == 8 * 7 + 4 * 18
    assert graph._indices.flags.owndata and graph._offsets.flags.owndata
    assert not graph._indices.flags.writeable


def test_public_adjacency_still_detaches_caller_arrays():
    offsets, indices = np.array([0, 1, 2]), np.array([1, 0])
    graph = CompactAdjacency(offsets, indices)
    offsets[:] = 0
    indices[:] = 0
    assert dict(graph[0]) == {1: 1.0} and dict(graph[1]) == {0: 1.0}


@pytest.mark.parametrize(
    "values,seasons",
    [
        ([1, 3, 2, 8, 5, 9], [0, 1, 0, 1, 0, 1]),
        ([3, 3, 2, 3, 6], [0, 0, 0, 0, 1]),
        ([1, 4, 9, 16], [0, 0, 0, 0]),
    ],
)
def test_preallocated_seasonal_slope_matches_pairwise_oracle(values, seasons):
    slopes = sorted(
        (values[j] - values[i]) / (j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if seasons[i] == seasons[j]
    )
    n = len(slopes)
    expected = (slopes[(n - 1) // 2] + slopes[n // 2]) / 2
    result = seasonal_kendall(values, seasons)
    assert result.assessed and result.effect == pytest.approx(expected, rel=1e-15)


def test_seasonal_slope_byte_admission_is_independent_of_work():
    result = seasonal_kendall([1, 2, 3, 4], [0] * 4, max_work=1_000_000, max_slope_bytes=1)
    assert not result.assessed and "max_slope_bytes" in result.reason
    assert result.n == {"values": 4, "seasons": 1}
    with pytest.raises(ValueError, match="nonnegative integer"):
        seasonal_kendall([1, 2, 3], [0] * 3, max_slope_bytes=True)


@pytest.mark.parametrize("descriptor", [False, True])
def test_crosswalk_csv_ids_keep_leading_zeros_with_projected_reader(tmp_path, descriptor):
    path = tmp_path / "zones.csv"
    path.write_text("old,new,unused\n001,010,discard-me\n002,010,discard-me-too\n")
    if descriptor:
        mapping = Crosswalk(
            data=File(path, text_columns=["old", "new"]),
            from_column="old",
            to_column="new",
        )
    else:
        mapping = Crosswalk(
            path=str(path),
            options={"types": {"old": "VARCHAR", "new": "VARCHAR"}},
            from_column="old",
            to_column="new",
        )
    result = crosswalk_frame(None, mapping)
    assert result.rows() == [("001", "010", 1.0), ("002", "010", 1.0)]
    assert result.columns == ["from", "to", "weight"]


@pytest.mark.parametrize("kind", ["frame", "lazy", "loader"])
def test_crosswalk_primary_input_descriptors_share_rename_and_projection(kind):
    frame = pl.DataFrame({"child": ["001", "002"], "parent": ["010"] * 2, "junk": [1, 2]})
    raw = frame if kind == "frame" else frame.lazy() if kind == "lazy" else Loader(lambda: frame)
    mapping = Crosswalk(data=Dataset(raw, rename={"child": "from", "parent": "to"}))
    with DataContext() as ctx:
        directory = ctx._resources.directory
        result = crosswalk_frame(ctx, mapping)
        assert directory.exists()
        assert result.rows() == [("001", "010", 1.0), ("002", "010", 1.0)]
    assert not directory.exists()


def test_crosswalk_path_row_budget_refuses_the_complete_source(tmp_path):
    path = tmp_path / "zones.csv"
    path.write_text("from,to\n1,2\n2,2\n")
    with pytest.raises(SourceError, match="row_budget"):
        crosswalk_frame(None, Crosswalk(path=str(path), row_budget=1))


def test_crosswalk_input_forms_and_options_are_unambiguous():
    with pytest.raises(ValueError, match="exactly one"):
        Crosswalk(path="zones.csv", data=pl.DataFrame({"from": [1], "to": [2]}))
    with pytest.raises(ValueError, match="reader options require 'path'"):
        Crosswalk(pairs=[(1, 2)], options={"header": True})


def test_one_shot_effects_do_not_change_caller_writeability():
    from qavf.inference.effects import bias, mape

    b, c = np.array([1.0, 2.0]), np.array([2.0, 3.0])
    assert bias(b, c) == 1.0
    assert mape(b, c) == 75.0
    assert b.flags.writeable and c.flags.writeable


@pytest.mark.parametrize("fallback", [False, True])
def test_nearest_graph_observes_cancellation_during_construction(monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(_neighbours, "_tree", lambda points: None)
    monkeypatch.setattr(_neighbours, "_BLOCK_PAIRS", 8)
    calls = 0

    def cancellation():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("cancelled fixture")

    with pytest.raises(RuntimeError, match="cancelled fixture"):
        _neighbours.nearest_graph(
            np.arange(8),
            np.zeros(8),
            8,
            2,
            geographic=False,
            max_edges=100,
            max_work=1000,
            check_cancelled=cancellation,
        )
    assert calls == 3


@pytest.mark.parametrize("kind", ["frame", "dataset", "loader", "file"])
@pytest.mark.parametrize("typed_configuration", [False, True])
def test_crosswalk_descriptors_work_through_public_comparison(tmp_path, kind, typed_configuration):
    frame = pl.DataFrame({"taz": ["a", "b"], "x": [1.0, 2.0]})
    other = frame.with_columns(county=pl.Series(["north", "south"]))
    mapping = pl.DataFrame(
        {"from": ["a", "b"], "to": ["north", "south"], "unused": ["private payload"] * 2}
    )
    if kind == "file":
        path = tmp_path / "zones.csv"
        mapping.write_csv(path)
        raw = File(path)
    elif kind == "loader":
        raw = Loader(lambda: mapping)
    elif kind == "dataset":
        raw = Dataset(mapping)
    else:
        raw = mapping
    options = {
        "steps": [
            {
                "step": "assign_zones",
                "zones": "taz",
                "column": "county",
                "crosswalk": Crosswalk(data=raw),
            }
        ],
        "levels": [{"level": "groups", "group_by": ["county"], "measures": ["sum(x)"]}],
        "decision": "material",
    }
    if typed_configuration:
        configured = ComparisonSpec(baseline="old", candidate="new", **options)
        result = compare_datasets(frame, other, configuration=configured, entry_points=False)
    else:
        result = compare_datasets(frame, other, **options, entry_points=False)
    assert result.passed, result.summary()
    assert "private payload" not in str(result.alignment)
    assert any("input>" in step for step in result.alignment)


@pytest.mark.parametrize("typed_crosswalk", [False, True])
def test_isolated_crosswalk_live_inputs_are_refused_before_transport(
    tmp_path, monkeypatch, typed_crosswalk
):
    from qavf.execution import dispatch

    def unexpected_worker(*args, **kwargs):
        pytest.fail("live auxiliary input reached the isolated worker")

    monkeypatch.setattr(dispatch, "compare_datasets_isolated", unexpected_worker)
    path = tmp_path / "primary.parquet"
    pl.DataFrame({"taz": ["a"], "x": [1.0]}).write_parquet(path)
    live = {"data": pl.DataFrame({"from": ["a"], "to": ["north"]})}
    crosswalk = Crosswalk(**live) if typed_crosswalk else live
    with pytest.raises(ValueError, match="isolated comparisons require serializable auxiliary"):
        compare_datasets(
            path,
            path,
            steps=[{"step": "assign_zones", "zones": "taz", "crosswalk": crosswalk}],
            resources=ResourceLimits(isolate=True),
            entry_points=False,
        )
