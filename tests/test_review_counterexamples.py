"""Counterexamples from the 0.5.0 code-quality review, kept as regression tests.

Each test is named after the review finding it reproduces. A false pass, an
overwritten input or a silent omission must fail here before any fix is trusted.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from qavf import Dataset, File, ResourceLimits, validate
from qavf.engine.severity import Status
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
def test_f07_streamed_json_honours_plugin_fields_and_serializers() -> None:
    import json
    from datetime import date

    from pydantic import field_serializer

    from qavf.compare.results import DatasetComparison, LevelResult

    class PluginResult(LevelResult):
        day: date
        amount: Decimal
        code: int

        @field_serializer("code")
        def _code(self, value: int) -> str:
            return f"code-{value}"

    comparison = DatasetComparison(
        name="p", baseline="a", candidate="b",
        levels=[PluginResult(level="custom", name="p", day=date(2026, 10, 30),
                             amount=Decimal("1.23"), code=12)],
    )
    streamed = json.loads(b"".join(comparison.iter_json_chunks()))
    assert streamed == json.loads(comparison.model_dump_json())
    assert streamed["levels"][0]["code"] == "code-12"
