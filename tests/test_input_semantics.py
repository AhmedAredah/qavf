"""Representation-independent QA decisions and bounded draft profiling."""

from __future__ import annotations

import math
from decimal import Decimal

import polars as pl
import pytest

from qavf import Dataset, File, ResourceLimits, profile_input, validate
from qavf.cli import EXIT_OK, EXIT_USAGE, main
from qavf.engine.context import DataContext
from qavf.profiling import profile_frame
from qavf.profiling.accumulator import ProfileAccumulator
from qavf.rules._regex import sql_compatible

CONFIG = {
    "engine": {"parallel": False, "run_unconfigured_rules": False},
    "rules": {"INP-003": {}},
    "plugins": {"entry_points": False},
}


def _input(frame, backend, tmp_path):
    if backend == "eager":
        return frame
    if backend == "lazy":
        return frame.lazy()
    path = tmp_path / "input.parquet"
    frame.write_parquet(path)
    return path


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
@pytest.mark.parametrize(
    "value,pattern,failed",
    [
        ("٣", r"\d+", False),
        ("٣", r"\D+", True),
        ("é", r"\w+", False),
        ("é", r"\W+", True),
        ("A", r"\u{41}", False),
        ("A", "[A-Z]+", False),
        ("a", "[A-Z]+", True),
        ("ab", "[a-z&&[^c]]+", False),
    ],
)
def test_regex_decisions_agree_on_all_representations(backend, value, pattern, failed, tmp_path):
    report = validate(
        _input(pl.DataFrame({"x": [value]}), backend, tmp_path),
        schema={"columns": {"x": {"regex": pattern}}},
        config=CONFIG,
    )
    assert report.verdict() == ("failed" if failed else "passed")
    assert all(r.status.value != "ERROR" for r in report.results)


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
@pytest.mark.parametrize("unit,failed", [("NA", True), ("kg", False), ("lb", True)])
def test_units_apply_dependency_missing_policy(backend, unit, failed, tmp_path):
    schema = {
        "columns": {
            "mass": {"unit_column": "unit", "allowed_units": ["kg", "NA"]},
            "unit": {"nullable": True, "missing_values": ["NA"]},
        }
    }
    report = validate(
        _input(pl.DataFrame({"mass": [1.0], "unit": [unit]}), backend, tmp_path),
        schema=schema,
        config=CONFIG,
    )
    assert report.verdict() == ("failed" if failed else "passed")


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
def test_invalid_marker_does_not_suppress_independent_checks(backend, tmp_path):
    report = validate(
        _input(pl.DataFrame({"x": [-1], "y": [1]}), backend, tmp_path),
        schema={"columns": {"x": {"ge": 0}, "y": {"missing_values": ["NA"]}}},
        config=CONFIG,
    )
    assert any(r.check == "data.x: ge 0" and r.affected_count == 1 for r in report.results)
    assert any(r.status.value == "ERROR" and r.check.startswith("data.y") for r in report.results)


@pytest.mark.parametrize("values", [[1.0, math.inf], [-math.inf, 1.0], [-math.inf, 1.0, math.inf]])
@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
def test_profiled_infinite_sides_accept_original_data(values, backend, tmp_path):
    frame = pl.DataFrame({"x": values})
    contract = profile_frame(frame)
    assert (
        validate(_input(frame, backend, tmp_path), schema=contract, config=CONFIG).verdict()
        == "passed"
    )


def test_native_unicode_regex_streams_without_projected_materialization(tmp_path, monkeypatch):
    frame = pl.DataFrame({"x": ["٣"] * 140_000, "original": range(140_000)})
    path = _input(frame, "native", tmp_path)
    monkeypatch.setattr(DataContext, "project", lambda *a, **k: pytest.fail("full-column fallback"))
    report = validate(
        path,
        schema={"columns": {"x": {"regex": r"\D+"}}},
        config=CONFIG,
        resources=ResourceLimits(evidence_rows=3),
    )
    failed = next(r for r in report.results if r.status.value == "FAIL")
    assert failed.affected_count == 140_000
    assert len(failed.affected_records) == 3
    assert [row["original"] for row in failed.affected_records] == [0, 1, 2]


def test_category_state_is_bounded_after_cardinality_threshold():
    accumulator = ProfileAccumulator({"x": pl.String}, max_categories=3, bounds=True)
    for start in range(0, 100, 10):
        accumulator.update(pl.DataFrame({"x": [str(i) for i in range(start, start + 10)]}))
        assert accumulator.columns["x"].categories is None
    assert accumulator.finish(0).schema["columns"]["x"] == {
        "dtype": "str",
        "min_length": 1,
        "max_length": 2,
    }


def test_native_profile_matches_frame_without_get(tmp_path, monkeypatch):
    frame = pl.DataFrame(
        {
            "x": [2**53 + 1, 2**53 + 2] * 40_000,
            "kind": ["a", "b"] * 40_000,
            "d": pl.Series([Decimal("1.01")] * 80_000, dtype=pl.Decimal(18, 2)),
        }
    )
    path = _input(frame, "native", tmp_path)
    monkeypatch.setattr(DataContext, "get", lambda *a, **k: pytest.fail("full source load"))
    result = profile_input(File(path), config={"plugins": {"entry_points": False}})
    assert result.schema == profile_frame(frame)
    assert (result.rows, result.columns) == (80_000, 3)


def test_profile_validates_options_before_loading():
    with pytest.raises(ValueError, match="max_categories"):
        profile_input("does-not-exist.parquet", max_categories=-1)
    with pytest.raises(ValueError, match="isolated profiling"):
        profile_input("does-not-exist.parquet", resources=ResourceLimits(isolate=True))


def test_profile_preserves_row_budget(tmp_path):
    frame = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(Exception, match=r"row budget|row_budget"):
        profile_input(Dataset(_input(frame, "native", tmp_path), load_policy={"row_budget": 2}))


def test_cli_profile_and_init_do_not_call_eager_reader(tmp_path, monkeypatch, capsys):
    frame = pl.DataFrame({"x": [1, 2]})
    path = _input(frame, "native", tmp_path)
    monkeypatch.setattr("qavf.io.read_table", lambda *a, **k: pytest.fail("eager CLI reader"))
    assert main(["profile", str(path)]) == EXIT_OK
    assert "schema:" in capsys.readouterr().out
    assert (
        main(["init", "-d", str(path), "-o", str(tmp_path / "draft.yaml"), "--no-schema"])
        == EXIT_OK
    )
    assert main(["profile", "missing.parquet", "--max-categories", "-1"]) == EXIT_USAGE


@pytest.mark.parametrize("pattern", [r"\d", r"\D", r"\w", r"\u{41}", "(?i)a", "[a-z&&[^c]]"])
def test_regex_engine_specific_features_use_canonical_evaluator(pattern):
    assert not sql_compatible(pattern)


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
def test_profile_input_matches_frame_for_every_backend(backend, tmp_path):
    frame = pl.DataFrame({"n": [1, 2, 3, None], "kind": ["x", "x", "x", None]})
    result = profile_input(_input(frame, backend, tmp_path))
    assert result.schema == profile_frame(frame)
    assert (result.rows, result.columns) == (4, 2)


def test_profile_nanosecond_input_uses_typed_observation(tmp_path):
    frame = pl.DataFrame(
        {"when": pl.Series([-1, 1, 1_001], dtype=pl.Int64).cast(pl.Datetime("ns"))}
    )
    result = profile_input(_input(frame, "native", tmp_path))
    assert result.schema == profile_frame(frame)
    assert result.rows == 3


def test_profile_empty_input_retains_schema(tmp_path):
    frame = pl.DataFrame(schema={"n": pl.Int64, "text": pl.String})
    result = profile_input(_input(frame, "native", tmp_path))
    assert result.rows == 0
    assert result.schema == profile_frame(frame)
