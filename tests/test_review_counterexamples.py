"""Counterexamples from the 0.5.0 code-quality review, kept as regression tests.

Each test is named after the review finding it reproduces. A false pass, an
overwritten input or a silent omission must fail here before any fix is trusted.
"""

from __future__ import annotations

import functools
from datetime import date
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from pydantic import field_serializer

from qavf import Dataset, File, ResourceLimits, validate
from qavf.compare.results import LevelResult
from qavf.engine.severity import Status
from qavf.plugins.registry import Component
from qavf.rules._declared_types import convert
from qavf.settings import ColumnSpec
from qavf.sources.file import FileSource


def _statuses(report, *rules: str) -> list[tuple[str, Status]]:
    return [(r.rule_id, r.status) for r in report.results if r.rule_id in rules]


# =========================================================================
# F01: decimal text is converted exactly, never rounded
# =========================================================================
def test_f01_a_decimal_beyond_its_scale_is_invalid_not_rounded() -> None:
    original = "0." + "0" * 37 + "11"  # 1.1e-38 needs 39 fractional digits
    report = validate(
        {"d": Dataset(pl.DataFrame({"x": [original]}), schema={
            "columns": {"x": {"dtype": "decimal", "le": Decimal("1e-38")}}
        })}
    )
    [schema] = [r for r in report.results_for("INP-002") if r.status is Status.FAIL]
    assert schema.affected_records[0]["x"] == original
    assert report.verdict() == "failed"


@pytest.mark.parametrize(
    ("texts", "values", "invalid"),
    [
        (["1.2300", "12", "-3.5", ".5"], ["1.23", "12", "-3.5", "0.5"], [False] * 4),
        (["0." + "0" * 37 + "1"], ["1E-38"], [False]),
        (["0." + "0" * 37 + "11"], [None], [True]),
        # Both fit alone, not together in one 38-digit column: neither is rounded.
        (["9" * 38, "0.5"], ["9" * 38, None], [False, True]),
        (["0.25", "0.250000000000000000000000000000000000000000"], ["0.25", "0.25"], [False] * 2),
    ],
)
def test_f01_decimal_conversion_is_exact(texts, values, invalid) -> None:
    converted = convert(pl.Series("v", texts), ColumnSpec(dtype="decimal"))
    expected = [None if v is None else Decimal(v) for v in values]
    assert converted.values.to_list() == expected
    assert converted.invalid.to_list() == invalid


# =========================================================================
# F08: legitimate header names are never reported as broken
# =========================================================================
@pytest.mark.parametrize(
    ("header", "columns"),
    [("column0,column1", ["column0", "column1"]), ('"rate,count"', ["rate,count"]),
     ('"rate;count"', ["rate;count"])],
)
def test_f08_names_that_look_generated_are_accepted_when_written(tmp_path, header, columns):
    path = tmp_path / "t.csv"
    path.write_text(f"{header}\n{','.join('1' for _ in columns)}\n")
    report = validate(
        File(path, options={"header": True, "delim": ","}),
        schema={"columns": {c: {"ge": 0} for c in columns}},
    )
    assert report.verdict() == "passed", [r.affected_records for r in report.results_for("INP-002")]


# =========================================================================
# F18: declared-text checks stay within a modest memory target
# =========================================================================
def test_f18_text_conversion_needs_no_loaded_column() -> None:
    source = pl.DataFrame({"v": ["000000000000000001"] * 200_000}).lazy()
    report = validate(
        {"d": Dataset(source, schema={"columns": {"v": {"dtype": "int", "ge": 0}}})},
        resources=ResourceLimits(
            target_memory="1MiB", cache_memory=0, max_spill="32MiB", max_workers=1,
            native_threads=1, evidence_rows=0,
        ),
    )
    assert _statuses(report, "INP-002", "INP-003") == [
        ("INP-002", Status.PASS), ("INP-003", Status.PASS)
    ]


# =========================================================================
# Review I-6: Parquet file positions come from footers, not a scan
# =========================================================================
def test_parquet_file_rows_come_from_footers(tmp_path, monkeypatch) -> None:
    for name, rows in (("a", 3), ("b", 2)):
        pl.DataFrame({"x": [-1.0] * rows}).write_parquet(tmp_path / f"{name}.parquet")
    monkeypatch.setattr(FileSource, "_count_file_rows", _never_scanned)
    report = validate(
        str(Path(tmp_path) / "*.parquet"), schema={"columns": {"x": {"ge": 0}}}
    )
    [bound] = [r for r in report.results_for("INP-003") if r.status is Status.FAIL]
    assert [(Path(r["_file"]).name, r["_file_row"]) for r in bound.affected_records][-1] == (
        "b.parquet", 1
    )


def _never_scanned(*_args: object) -> list[tuple[str, int]]:
    raise AssertionError("Parquet row counts must come from footers")


# =========================================================================
# F02: a metric normalized by its own changing dataset reads one snapshot
# =========================================================================
class _Changing:
    """A lazy plan returning ``first`` on its first evaluation, ``later`` after."""

    def __init__(self, first: pl.DataFrame, later: pl.DataFrame) -> None:
        self.first, self.later, self.reads = first, later, 0

    def __call__(self, _: pl.DataFrame) -> pl.DataFrame:
        self.reads += 1
        return self.first if self.reads == 1 else self.later

    def plan(self) -> pl.LazyFrame:
        return self.first.lazy().map_batches(
            self, schema=self.first.schema, projection_pushdown=False,
            predicate_pushdown=False, slice_pushdown=False,
        )


def test_f02_numerator_and_denominator_describe_one_snapshot() -> None:
    from qavf.engine.context import DataContext
    from qavf.rules.spatial_checks import GroupComparisonRule

    a = pl.DataFrame({"entity": [1, 2, 3], "x": [1.0, 1.0, 10.0], "den": [1.0, 1.0, 1.0]})
    b = pl.DataFrame({"entity": [1, 2, 3], "x": [1.0, 1.0, 100.0], "den": [1.0, 1.0, 10.0]})
    params = {
        "metrics": [{"dataset": "d", "value": "x", "entity_column": "entity",
                     "normalize_by": {"dataset": "d", "value": "den", "on": ["entity"]}}],
        "max_ratio": 2, "min_entities": 3,
    }
    source = _Changing(a, b)
    with DataContext(frames={"d": source.plan()}) as ctx:
        [finding] = GroupComparisonRule(params).evaluate(ctx)
    assert not finding.passed and source.reads == 1


# =========================================================================
# F03: an exceeded error limit fails, whatever the identity-line test says
# =========================================================================
def test_f03_an_error_limit_is_a_hard_limit() -> None:
    import numpy as np

    from qavf import compare_datasets

    base = np.arange(1.0, 101.0)
    left = pl.DataFrame({"id": np.arange(100), "x": base})
    # Orthogonal error: the fitted line stays on the identity, the RMSE is 100.
    errors = np.tile([100.0, -100.0, -100.0, 100.0], 25)
    right = left.with_columns(pl.Series("x", base + errors))
    agreement = {"columns": ["x"], "max_rmse": 1.0, "min_ccc": None, "max_slope_deviation": None}
    result = compare_datasets(
        left, right, keys="id", entry_points=False,
        levels=[{"level": "relationships", "agreement": agreement}],
    )
    assert result.verdict == "failed"


# =========================================================================
# F04: a requested column that cannot be compared is never silently skipped
# =========================================================================
def test_f04_partial_column_coverage_is_incomplete() -> None:
    from qavf import compare_datasets

    base = pl.DataFrame({"id": [1, 2], "ok": [1, 2], "changed": [10, 20]})
    cand = pl.DataFrame({"id": [1, 2], "ok": [1, 2], "changed": ["wrong", "bad"]})
    partial = compare_datasets(base, cand, keys="id", entry_points=False, levels=["records"])
    assert partial.verdict == "incomplete"
    ignored = compare_datasets(
        base, cand, keys="id", entry_points=False, levels=["records"], ignore_columns=["changed"]
    )
    assert ignored.verdict == "passed"


# =========================================================================
# F05: a report never replaces an input of its run
# =========================================================================
@pytest.mark.parametrize("target", ["data", "config", "link"])
def test_f05_a_report_path_naming_an_input_is_refused(tmp_path, target, capsys) -> None:
    import json
    import os

    from qavf.cli import main

    data = tmp_path / "data.csv"
    data.write_text("x\n1\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "datasets": {"data": {"path": "data.csv", "schema": {"columns": {"x": {"ge": 0}}}}},
        "report": {"html": False, "json": False},
    }))
    destination = {"data": data, "config": config, "link": tmp_path / "link.csv"}[target]
    if target == "link":
        try:
            os.symlink(data, destination)
        except OSError:
            pytest.skip("symbolic links are not available")
    before = {path: path.read_bytes() for path in (data, config)}
    assert main(["run", "-c", str(config), "--json", str(destination)]) != 0
    assert "would replace the input" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in (data, config)} == before


# =========================================================================
# F06: shared configuration parts are never expanded into a tree
# =========================================================================
def _shared_graph(depth: int) -> dict:
    from qavf.configuration import expand_env

    shared: list = ["leaf"]
    for _ in range(depth):
        shared = [shared] * 8
    return expand_env({"extensions": {"probe": {"graph": shared}}})


def test_f06_shared_configuration_parts_stay_shared_and_bounded() -> None:
    from qavf.inputs.normalize import prepare

    frame = pl.DataFrame({"x": [1]})
    schema = {"columns": {"x": {}}}
    settings, _ = prepare(frame, schema=schema, config=_shared_graph(6), resources=None)
    copied = settings.extensions["probe"]["graph"]
    assert copied[0] is copied[1]
    with pytest.raises(ValueError, match="expands to"):
        prepare(frame, schema=schema, config=_shared_graph(8), resources=None)
    cyclic: dict = {"extensions": {"probe": {}}}
    cyclic["extensions"]["probe"]["self"] = cyclic["extensions"]["probe"]
    with pytest.raises(ValueError, match="cyclic"):
        prepare(frame, schema=schema, config=cyclic, resources=None)


# =========================================================================
# F07: streamed JSON equals Pydantic's, for plugin result models too
# =========================================================================
class _PluginResult(LevelResult):
    """A plugin's level result with extra rich fields and a custom serializer."""

    day: date
    amount: Decimal
    code: int

    @field_serializer("code")
    def _code(self, value: int) -> str:
        return f"code-{value}"


def test_f07_streamed_json_honours_plugin_fields_and_serializers() -> None:
    import json

    from qavf.compare.results import DatasetComparison

    comparison = DatasetComparison(
        name="p", baseline="a", candidate="b",
        levels=[_PluginResult(level="custom", name="p", day=date(2026, 10, 30),
                              amount=Decimal("1.23"), code=12)],
    )
    streamed = json.loads(b"".join(comparison.iter_json_chunks()))
    assert streamed == json.loads(comparison.model_dump_json())
    assert streamed["levels"][0]["code"] == "code-12"


# =========================================================================
# F11: a removal that fails is retried, its bytes stay charged until then
# =========================================================================
class _LockedOnce(type(Path())):  # type: ignore[misc]
    """A path whose first removal fails, as a Windows sharing lock does."""

    failed = False

    def unlink(self, missing_ok: bool = False) -> None:
        if not type(self).failed:
            type(self).failed = True
            raise PermissionError("the file is in use")
        super().unlink(missing_ok=missing_ok)


def test_f11_a_spill_removal_that_fails_is_retried(tmp_path) -> None:
    from qavf.engine.spill import SpillBudget, SpillFile

    budget = SpillBudget(1024)
    spill = SpillFile(_LockedOnce(tmp_path / "spill.bin"), budget)
    with spill.writer() as handle:
        handle.write(b"x" * 100)
    lease = spill.borrow()
    spill.close()
    with pytest.raises(PermissionError):
        lease.close()
    assert spill.pending_removal and spill.path.exists() and budget.used == 100
    lease.close()
    assert not spill.pending_removal and not spill.path.exists() and budget.used == 0


def test_f11_run_cleanup_reports_and_retries_a_directory_it_cannot_remove(
    tmp_path, monkeypatch
) -> None:
    import shutil

    from qavf.engine.resources import RunResources

    resources = RunResources(1024, 10_000, None, tmp_path)
    rmtree = shutil.rmtree
    monkeypatch.setattr(shutil, "rmtree", _refuse_removal)
    with pytest.raises(PermissionError):
        resources.close()
    assert resources.directory.exists()
    monkeypatch.setattr(shutil, "rmtree", rmtree)
    resources.close()
    assert not resources.directory.exists()


def _refuse_removal(*_args: object, **_kwargs: object) -> None:
    raise PermissionError("the directory is in use")


# =========================================================================
# F09: a NaN identifier is a missing identifier on every spatial entry point
# =========================================================================
def test_f09_nan_ids_are_missing_in_layers_crosswalks_and_weights(tmp_path) -> None:
    import math

    from qavf.spatial.hierarchy import Crosswalk, crosswalk_frame
    from qavf.spatial.layers import ZoneLayer, zone_ids
    from qavf.spatial.weights import from_table

    with pytest.raises(ValueError, match="missing value"):
        crosswalk_frame(None, Crosswalk(pairs=((math.nan, "c"),)))
    with pytest.raises(ValueError, match="present and unique"):
        from_table(
            pl.Series([math.nan, 1.0]),
            pl.DataFrame({"from": [math.nan], "to": [1.0], "weight": [1.0]}),
        )
    path = tmp_path / "zones.parquet"
    pl.DataFrame({"zone": [math.nan, 1.0]}).write_parquet(path)
    with pytest.raises(ValueError, match="without an id"):
        zone_ids(None, ZoneLayer(path=str(path)))
    pl.DataFrame({"zone": [2.0, 1.0]}).write_parquet(path)
    assert zone_ids(None, ZoneLayer(path=str(path))).to_list() == [1.0, 2.0]


# =========================================================================
# F10: charts index members by typed identity, never by their display label
# =========================================================================
def test_f10_distinct_groups_with_equal_labels_stay_distinct_in_charts() -> None:
    from types import SimpleNamespace

    from qavf.compare.results import MemberValues
    from qavf.reports.comparison import ChartBudget, MemberLabels, _series_charts
    from qavf.reports.presentation import HtmlLimits

    points = [
        {"g": "a, b", "h": "c", "t": 1, "baseline": 10.0, "candidate": 20.0},
        {"g": "a", "h": "b, c", "t": 1, "baseline": 100.0, "candidate": 110.0},
    ]
    level = SimpleNamespace(values=[MemberValues(measure="m", keys=["g", "h", "t"], points=points)])
    heat, *lines = _series_charts(level, ChartBudget(HtmlLimits()))
    assert len(lines) == 2 and len({view.title for view in lines}) == 2
    assert "+100%" in str(heat.svg) and "+10%" in str(heat.svg)

    records = [{"x": 1.0000001}, {"x": 1.0000002}, {"x": None}, {"x": "\u2014"}, {"x": 2.5}]
    labels = [MemberLabels(["x"], records).label(r) for r in records]
    assert len(set(labels)) == 5 and labels[4] == "2.5"


# =========================================================================
# F12: a refresh rebuilds the installed components from the current entry points
# =========================================================================
class _InstalledProbe(Component):
    name = "probe"


class _CodedProbe(Component):
    name = "coded"


class _Installed:
    """Entry points returned by a probe registry; ``current`` is what is installed now."""

    def __init__(self, *current: object) -> None:
        self.current = list(current)

    def __call__(self) -> list[object]:
        return self.current


class _Entry:
    def __init__(self, name: str, target: object) -> None:
        self.name, self.target = name, target

    def load(self) -> object:
        if isinstance(self.target, Exception):
            raise self.target
        return self.target


def test_f12_refresh_drops_removed_and_failed_components(monkeypatch) -> None:
    from qavf.plugins.registry import ComponentRegistry
    from qavf.plugins.transactions import visibility

    registry: ComponentRegistry = ComponentRegistry("probe", "review.probe", Component, ())
    installed = _Installed(_Entry("example", _InstalledProbe))
    monkeypatch.setattr(registry, "entries", installed)
    registry.register(_CodedProbe)
    registry.refresh()
    with visibility(True):
        assert registry.names() == ["coded", "probe"]
    assert registry.plugins == {"example": "loaded"}

    installed.current = [_Entry("example", ImportError("replacement is broken"))]
    registry.refresh()
    with visibility(True):
        assert registry.names() == ["coded"]
    assert registry.plugins["example"].startswith("error: ImportError")

    installed.current = []
    registry.refresh()
    with visibility(True):
        assert registry.names() == ["coded"]
    assert registry.plugins == {}


# =========================================================================
# F13: whole counts reach tests exactly; an unsupported magnitude is unassessed
# =========================================================================
def test_f13_counts_are_never_rounded_and_extremes_are_unassessed() -> None:
    from qavf.compare.significance import _whole
    from qavf.inference.counts import poisson_rate
    from qavf.numeric import count_input

    assert _whole(2**53) != _whole(2**53 + 1) == 2**53 + 1
    assert count_input(Decimal(2**53 + 1)) == 2**53 + 1 and count_input(2.5) == 2.5
    extreme = poisson_rate((1, 10**16), (1.0, 1e16))
    assert not extreme.assessed and "2**53" in (extreme.reason or "")
    assert poisson_rate((2**53 - 3, 2), (1.0, 1.0)).assessed
    assert poisson_rate((1, 5), (1.0, 1e300)).assessed


# =========================================================================
# F22: an induced subgraph reads only the selected rows
# =========================================================================
def test_f22_a_small_subgraph_costs_its_own_edges_not_the_graphs() -> None:
    import tracemalloc

    import numpy as np

    from qavf._adjacency import CompactAdjacency

    n = 500_000
    ring = np.arange(n)
    graph = CompactAdjacency.from_edges(
        n, np.repeat(ring, 2), np.stack([(ring + 1) % n, (ring - 1) % n], 1).ravel()
    )
    tracemalloc.start()
    try:
        subset = graph.subset([1, 0, 7])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 100_000
    assert [dict(subset[i]) for i in range(3)] == [{1: 1.0}, {0: 1.0}, {}]
    weighted = CompactAdjacency.from_edges(3, np.array([0, 0, 2]), np.array([1, 2, 0]),
                                           np.array([0.5, 2.0, 3.0]))  # fmt: skip
    assert [dict(weighted.subset([2, 0])[i]) for i in range(2)] == [{1: 3.0}, {0: 2.0}]


# =========================================================================
# F16: a chart's construction memory is bounded by its marks, not its labels
# =========================================================================
def test_f16_a_huge_label_is_cut_before_the_chart_is_built() -> None:
    import tracemalloc

    from qavf.reports import charts

    label = "x" * 10_000_000
    tracemalloc.start()
    try:
        drawn = [
            charts.horizontal_bars([charts.BarRow(label, [charts.Segment("s", 1.0, "c")])], label),
            charts.waterfall([charts.WaterfallStep(label, 1.0, "total")], label),
            charts.heat_map([label], [label], {(label, label): 0.5}, label),
            charts.series_lines([label], [1.0], [2.0], label),
            charts.paired_bars([label], [0.5], [0.5], label),
            charts.deltas([label], [0.1], label),
            charts.choropleth(
                {label: "M0 0 L1 1"}, {label: (0.5, 0.5)}, {label: 0.1}, (10, 10), label
            ),
        ]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    assert all(len(svg) < 10 * charts.MAX_TEXT_CHARS + 5_000 for svg in drawn)
    assert "+50%" in drawn[2]  # the cell is still found under its full label


# =========================================================================
# F17: a frame known to exceed its row budget is refused before conversion
# =========================================================================
def _never_converted(*_args: object, **_kwargs: object) -> pl.DataFrame:
    raise AssertionError("an over-budget frame must not be converted")


def test_f17_an_over_budget_pandas_frame_is_not_converted(monkeypatch) -> None:
    pd = pytest.importorskip("pandas")
    schema = {"columns": {"x": {}}}
    within = validate({"d": pd.DataFrame({"x": ["a", "b"]})}, schema=schema,
                      config={"datasets": {"d": {"load_policy": {"row_budget": 2}}}})  # fmt: skip
    assert within.verdict() == "passed"
    monkeypatch.setattr(pl, "from_pandas", _never_converted)
    report = validate(
        {"d": pd.DataFrame({"x": ["y" * 1024] * 20_000})}, schema=schema,
        config={"datasets": {"d": {"load_policy": {"row_budget": 1}}}},
    )  # fmt: skip
    [availability] = report.results_for("INP-001")
    assert availability.status is Status.FAIL and "row_budget" in availability.message


# =========================================================================
# F15: series-shape screens are admitted against max_work before they run
# =========================================================================
def _no_screen(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("a screen beyond the work budget must not run")


def test_f15_shape_screens_beyond_max_work_are_refused_not_run(monkeypatch) -> None:
    from datetime import date, timedelta

    from qavf import compare_datasets
    from qavf.compare.levels import shape
    from qavf.statistics.series import lag_work, split_work

    days = [date(2026, 1, 1) + timedelta(days=d) for d in range(12)]
    frame = pl.DataFrame({"t": days, "x": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0] * 2})
    level = {"level": "series_shape", "time_column": "t", "bucket": "day",
             "measure": "sum(x)", "max_work": 1}  # fmt: skip
    monkeypatch.setattr(shape, "best_split", _no_screen)
    monkeypatch.setattr(shape, "best_lag", _no_screen)
    result = compare_datasets(frame, frame, levels=[level], entry_points=False)
    checks = {c.check: c for c in result.level("series_shape").checks}
    for name in ("lag", "level shift"):
        refused = checks[f"{name}: resource limit"]
        assert not refused.assessed and refused.affected_count == 1
    assert result.verdict == "incomplete"
    assert split_work(12, 4) == 12 * 5 and lag_work(12, 12, 3) == 7 * 24


# =========================================================================
# F24: integer closeness under a tolerance is decided natively and exactly
# =========================================================================
def _no_scalar_cells(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("64-bit integers under a tolerance must not take the per-cell path")


@pytest.mark.parametrize(("rtol", "atol"), [(1e-6, 0), (0, 1), (0.1, 2), (1e-12, 3)])
def test_f24_integer_tolerances_match_the_exact_rule_without_per_cell_work(
    rtol, atol, monkeypatch
) -> None:
    from qavf.regression import _comparison
    from qavf.regression._numeric import _settled, numbers_close
    from qavf.regression.compare_outputs import compare_frames

    edges = [0, 1, -1, 999_999, 1_000_000, 2**53, 2**53 + 1, 2**63 - 1, -(2**63), 10, 13]
    after = [0, 2, 1, 1_000_000, 1_000_001, 2**53 + 1, 2**53 + 3, 2**63 - 2, -(2**63) + 1, 12, 9]
    cells = pl.DataFrame({"a": edges, "b": after}, schema={"a": pl.Int64, "b": pl.Int64})
    settled = cells.select(_settled(pl.col("a"), pl.col("b"), pl.Int64, pl.Int64, rtol, atol))
    assert settled.to_series().to_list() == [
        numbers_close(a, b, rtol, atol) for a, b in zip(edges, after, strict=True)
    ]
    monkeypatch.setattr(_comparison, "_cell_change", _no_scalar_cells)
    base = cells.select(pl.int_range(pl.len()).alias("k"), pl.col("a").alias("v"))
    cand = cells.select(pl.int_range(pl.len()).alias("k"), pl.col("b").alias("v"))
    result = compare_frames(base, cand, "k", rtol=rtol, atol=atol)
    assert result.equal == all(settled.to_series())


# =========================================================================
# F25: a nested integer cell stops at its first difference
# =========================================================================
class _CountedClose:
    """``numbers_close`` that counts the leaves it compares."""

    def __init__(self) -> None:
        from qavf.regression._numeric import numbers_close

        self.close, self.calls = numbers_close, 0

    def __call__(self, *args: object) -> bool:
        self.calls += 1
        return self.close(*args)


def test_f25_integer_structs_exit_early_and_float_structs_keep_nonfinite_counts(
    monkeypatch,
) -> None:
    from qavf.regression import _cells
    from qavf.regression.compare_outputs import compare_frames

    counted = _CountedClose()
    monkeypatch.setattr(_cells, "numbers_close", counted)
    values = list(range(10_000))
    before = pl.DataFrame({"k": [1], "s": [{"v": values}]})
    after = pl.DataFrame({"k": [1], "s": [{"v": [-1, *values[1:]]}]})
    assert not compare_frames(before, after, "k").equal
    assert counted.calls == 1
    floats = pl.DataFrame({"k": [1], "s": [{"v": [1.0, 2.0, 3.0]}]})
    later_nan = pl.DataFrame({"k": [1], "s": [{"v": [0.0, 2.0, float("nan")]}]})
    assert compare_frames(floats, later_nan, "k").nonfinite_changes == 1


# =========================================================================
# F26: one member's numeric preparation is shared by its metrics
# =========================================================================
class _CountedScaling:
    """``_scaled_pair`` that counts how often a pair is prepared."""

    def __init__(self) -> None:
        from qavf.inference import effects

        self.scale, self.calls = effects._scaled_pair, 0

    def __call__(self, *args: object) -> object:
        self.calls += 1
        return self.scale(*args)


@pytest.mark.parametrize(
    ("level", "prepared"),
    [
        ({"level": "relationships", "agreement": {"columns": ["x"]}}, 1),
        ({"level": "distribution", "columns": ["x"], "decision": "material"}, 1),
    ],
)
def test_f26_a_member_is_scaled_once_for_all_its_metrics(level, prepared, monkeypatch) -> None:
    import numpy as np

    from qavf import compare_datasets
    from qavf.inference import effects

    counted = _CountedScaling()
    monkeypatch.setattr(effects, "_scaled_pair", counted)
    x = np.arange(1.0, 51.0)
    base = pl.DataFrame({"id": np.arange(50), "x": x})
    cand = base.with_columns(pl.Series("x", x * 1.01))
    compare_datasets(base, cand, keys="id", entry_points=False, levels=[level])
    assert counted.calls == prepared


# =========================================================================
# F23: failing-row evidence shows the checked columns, not the whole row
# =========================================================================
_EVIDENCE_FRAME = pl.DataFrame({
    "id": [1, 2, 3],
    "x": [-1.0, 5.0, -3.0],
    "y": ["a", "bb", "zz"],
    "scope": [1, 1, 0],
    "unit": ["m", "km", "m"],
    "payload": ["p" * 4096] * 3,
})
_EVIDENCE_SCHEMA = {
    "primary_key": ["id"],
    "columns": {
        "id": {},
        "x": {"ge": 0},
        "y": {"checks": [{"name": "short", "where": "scope = 1", "regex": "^[a-z]$"}]},
        "scope": {"nullable": True},
        "unit": {"nullable": True},
    },
}


@pytest.mark.parametrize("engine", ["native", "lazy", "eager"])
@pytest.mark.parametrize("whole", [False, True])
def test_f23_evidence_projects_checked_columns(engine, whole, tmp_path) -> None:
    schema = {**_EVIDENCE_SCHEMA, "evidence_columns": "all" if whole else "checked"}
    if engine == "native":
        path = tmp_path / "wide.parquet"
        _EVIDENCE_FRAME.write_parquet(path)
        data: object = File(path)
    else:
        data = _EVIDENCE_FRAME.lazy() if engine == "lazy" else _EVIDENCE_FRAME
    report = validate({"d": Dataset(data, schema=schema)})
    failed = {r.check: r for r in report.results_for("INP-003") if r.status is Status.FAIL}
    shown = {check: set(r.affected_records[0]) for check, r in failed.items()}
    everything = set(_EVIDENCE_FRAME.columns)
    assert shown["d.x: ge 0"] >= ({"id", "x", "_row"} | (everything if whole else set()))
    assert ("payload" in shown["d.x: ge 0"]) is whole
    [scoped] = [check for check in shown if check.startswith("d.y:")]
    assert shown[scoped] >= {"id", "y", "scope"}
    assert ("payload" in shown[scoped]) is whole


# =========================================================================
# F20: an unordered evidence sample reads its query once, staging nothing
# =========================================================================
class _CountedRead:
    """``BoundedSample.read`` wrapped to count the samples it streams."""

    def __init__(self) -> None:
        from qavf.engine.sql import BoundedSample

        self.read, self.calls = BoundedSample.read, 0

    def __get__(self, sampler: object, _owner: object) -> object:
        return functools.partial(self.__call__, sampler)

    def __call__(self, sampler: object, *args: object) -> object:
        self.calls += 1
        return self.read(sampler, *args)


def _no_staging(*_args: object, **_kwargs: object) -> str:
    raise AssertionError("an unordered sample must not stage the query in a CTE")


def test_f20_unordered_samples_stream_and_count_exactly(monkeypatch) -> None:
    import duckdb

    from qavf.engine import context
    from qavf.engine.sql import BoundedSample

    connection = duckdb.connect()
    sample, count = BoundedSample(20).read(
        connection, "SELECT i, repeat('x', 1024) AS payload FROM range(100000) t(i)", _ignore
    )
    assert count == 100_000 and sample["i"].to_list() == list(range(20))
    empty, none = BoundedSample(0).read(connection, "SELECT 1 AS a WHERE false", _ignore)
    assert (none, empty.columns) == (0, ["a"])
    connection.execute("CREATE TABLE kept AS SELECT 1 AS a")
    with pytest.raises(duckdb.Error):
        BoundedSample(1).read(connection, "DELETE FROM kept", _ignore)
    assert connection.execute("SELECT count(*) FROM kept").fetchone() == (1,)

    counted = _CountedRead()
    monkeypatch.setattr(context, "sample_statement", _no_staging)
    monkeypatch.setattr(BoundedSample, "read", counted)
    frame = pl.DataFrame({"x": [-1, 2, -3, 4], "payload": ["p"] * 4})
    report = validate(
        {"d": Dataset(frame, expressions=[{"name": "positive", "expr": "x > 0"}])}
    )
    [failed] = [r for r in report.results_for("INP-006") if r.status is Status.FAIL]
    assert failed.affected_count == 2 and counted.calls >= 1


def _ignore() -> None:
    """A cancellation check that never cancels."""


# =========================================================================
# F21: a spilled frame is read natively, outside the cache lock
# =========================================================================
class _ReadWithLockFree:
    """``read_owned_ipc`` that records whether another thread could take the lock."""

    def __init__(self, store: object) -> None:
        from qavf.engine import resources

        self.store, self.read, self.free = store, resources.read_owned_ipc, []

    def __call__(self, path: Path) -> pl.DataFrame:
        import threading

        attempt = threading.Thread(target=self._try_lock)
        attempt.start()
        attempt.join()
        return self.read(path)

    def _try_lock(self) -> None:
        lock = self.store.lock  # type: ignore[attr-defined]
        acquired = lock.acquire(blocking=False)
        self.free.append(acquired)
        if acquired:
            lock.release()


def test_f21_spill_reads_are_native_and_do_not_hold_the_cache_lock(tmp_path, monkeypatch):
    import tracemalloc

    from qavf.engine import resources
    from qavf.engine.resources import FrameStore

    frame = pl.DataFrame({"x": range(1_000_000)})
    store = FrameStore(0, 64_000_000, tmp_path)
    store["d"] = frame  # larger than the memory allocation: spilled
    reader = _ReadWithLockFree(store)
    monkeypatch.setattr(resources, "read_owned_ipc", reader)
    tracemalloc.start()
    try:
        read = store["d"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert read.equals(frame) and reader.free == [True]
    assert peak < 1_000_000  # no Python copy of the 8 MB file
    roomy = FrameStore(10_000_000, 64_000_000, tmp_path / "roomy")
    roomy["d"] = frame
    roomy["other"] = frame  # evicts "d" to its spill
    assert "d" not in roomy.data
    del roomy["other"]
    assert roomy["d"].equals(frame) and "d" in roomy.data  # promoted into free space


# =========================================================================
# F19: cross-CRS grid assignment admits its distinct points, in batches
# =========================================================================
def _grid_cells(limits: object) -> dict[float | None, set[str | None]]:
    """Each longitude's cells from a cross-CRS grid assignment under ``limits``."""
    from qavf.spatial.extension import spatial_connection
    from qavf.spatial.grid import Grid, assign_cells

    frame = pl.DataFrame({"lon": [-99.0, -99.0, -98.0, -100.0, None], "lat": [0.0] * 5})
    assigned = assign_cells(
        frame.lazy(), Grid(size=100_000.0, crs="EPSG:32614"), x="lon", y="lat",
        crs="EPSG:4326", column="cell", collect=pl.LazyFrame.collect,
        connection=spatial_connection, limits=limits,  # type: ignore[arg-type]
    ).plan.collect()  # fmt: skip
    found: dict[float | None, set[str | None]] = {}
    for lon, cell in assigned.select("lon", "cell").iter_rows():
        found.setdefault(lon, set()).add(cell)
    return found


def test_f19_grid_assignment_is_admitted_and_batched() -> None:
    from qavf.spatial.assign import AssignmentLimits

    try:
        whole = _grid_cells(AssignmentLimits())
    except Exception as exc:  # the spatial extension cannot be installed here
        pytest.skip(f"spatial extension unavailable: {exc}")
    assert _grid_cells(AssignmentLimits(batch_size=1)) == whole
    assert whole == {-99.0: {"5_0"}, -98.0: {"6_0"}, -100.0: {"3_0"}, None: {None}}
    with pytest.raises(ValueError, match="max_points"):
        _grid_cells(AssignmentLimits(max_points=3))
