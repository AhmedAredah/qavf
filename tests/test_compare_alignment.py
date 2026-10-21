"""The comparison alignment pipeline: each step alone, their order, and refusals.

Expected values are written out by hand; a refused alignment must make every
level not assessed, never a pass.
"""

from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import Dataset, compare_datasets
from qavf.compare import ALIGNMENT_STEPS, AlignmentContext, AlignmentError, ComparisonSpec
from qavf.compare.align import align, build_pipeline


def _context(side="baseline", units=None):
    return AlignmentContext(
        side, "data", count=lambda plan: plan.collect().height, units=dict(units or {})
    )


def _apply(name, params, frame, side="baseline", units=None):
    step = ALIGNMENT_STEPS.create(name, params)
    context = _context(side, units)
    return step.apply(frame.lazy(), context).collect(), context


def test_column_map_renames_only_the_baseline():
    frame = pl.DataFrame({"a": [1], "b": [2]})
    renamed, context = _apply("columns", {"columns": {"a": "x"}}, frame, units={"a": "kg"})
    assert renamed.columns == ["x", "b"] and context.units == {"x": "kg"}
    assert ALIGNMENT_STEPS.get("columns").sides == ("baseline",)
    with pytest.raises(AlignmentError, match="lacks mapped column"):
        _apply("columns", {"columns": {"zzz": "x"}}, frame)
    with pytest.raises(AlignmentError, match="several baseline columns"):
        _apply("columns", {"columns": {"a": "b"}}, frame)


def test_lossless_casts_convert_and_lossy_casts_refuse():
    frame = pl.DataFrame({"v": [1.0, 2.0, None], "s": ["1", "2", "3"]})
    cast, _ = _apply("cast", {"columns": {"v": "int", "s": {"dtype": "int"}}}, frame)
    assert cast.schema == {"v": pl.Int64, "s": pl.Int64} and cast["v"].to_list() == [1, 2, None]
    lossy = pl.DataFrame({"v": [1.0, 2.5, 3.5]})
    with pytest.raises(AlignmentError, match="would change 2 value"):
        _apply("cast", {"columns": {"v": "int"}}, lossy)
    decimal, _ = _apply("cast", {"columns": {"v": {"dtype": "decimal", "scale": 1}}}, lossy)
    assert decimal["v"].to_list() == [Decimal("1.0"), Decimal("2.5"), Decimal("3.5")]
    with pytest.raises(AlignmentError, match="lacks column"):
        _apply("cast", {"columns": {"missing": "int"}}, frame)
    # Side selection: a candidate-only cast leaves the baseline untouched.
    same, _ = _apply("cast", {"columns": {"v": {"dtype": "int", "side": "candidate"}}}, lossy)
    assert same.schema["v"] == pl.Float64


def test_unit_conversion_is_exact_and_checks_declared_units():
    frame = pl.DataFrame({"kt": [2**60, 3], "d": pl.Series([Decimal("1.25"), None])})
    params = {"columns": {"kt": {"factor": 1000, "from_unit": "kt", "to_unit": "t"}}}
    scaled, context = _apply("units", params, frame, units={"kt": "kt"})
    assert scaled["kt"].to_list() == [2**60 * 1000, 3000] and context.units == {"kt": "t"}
    fraction, _ = _apply("units", {"columns": {"d": {"factor": 0.1}}}, frame)
    assert fraction["d"].to_list() == [Decimal("0.125"), None]
    with pytest.raises(AlignmentError, match="declared in 'g'"):
        _apply("units", params, frame, units={"kt": "g"})
    untouched, _ = _apply("units", params, frame, side="candidate")
    assert untouched["kt"].to_list() == [2**60, 3]
    with pytest.raises(ValidationError):
        ComparisonSpec(baseline="a", candidate="b", units={"x": {"factor": 0}})


def test_recoding_maps_keeps_nulls_and_refuses_unmapped_values_on_request():
    frame = pl.DataFrame({"code": [1, 2, 3, None]})
    params = {"columns": {"code": {"values": {"1": "one", "2": "two"}}}}
    kept, _ = _apply("recode", params, frame)
    assert kept["code"].to_list() == ["one", "two", "3", None]
    nulled, _ = _apply(
        "recode", {"columns": {"code": {"values": [[1, "one"]], "unmapped": "null"}}}, frame
    )
    assert nulled["code"].to_list() == ["one", None, None, None]
    with pytest.raises(AlignmentError, match="2 value"):
        _apply("recode", {"columns": {"code": {"values": {"1": "x"}, "unmapped": "error"}}}, frame)
    with pytest.raises(AlignmentError, match="cannot all be represented"):
        _apply("recode", {"columns": {"code": {"values": {"one": "x"}}}}, frame)
    floats = pl.DataFrame({"code": [1.5, 9.0]})
    with pytest.raises(AlignmentError, match="cannot be kept"):
        _apply("recode", {"columns": {"code": {"values": [[9.0, 9]]}}}, floats)


def test_derived_columns_and_filters():
    frame = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    derived, _ = _apply("derive", {"columns": {"total": "a + b"}}, frame)
    assert derived["total"].to_list() == [11, 22, 33]
    with pytest.raises(AlignmentError, match="already exist"):
        _apply("derive", {"columns": {"a": "b * 2"}}, frame)
    params = {"baseline_where": "a > 1", "candidate_where": "a < 3", "where": "b <> 20"}
    assert _apply("filter", params, frame)[0]["a"].to_list() == [3]
    assert _apply("filter", params, frame, side="candidate")[0]["a"].to_list() == [1]


def test_pipeline_order_lets_later_steps_use_candidate_names():
    spec = ComparisonSpec(
        baseline="b",
        candidate="c",
        columns={"old": "value"},
        cast={"value": "float"},
        units={"value": {"factor": 2}},
        derive={"double": "value * 2"},
        where="double > 4",
    )
    steps = build_pipeline(spec)
    assert [step.name for step in steps] == ["columns", "cast", "units", "derive", "filter"]
    frame = pl.DataFrame({"old": [1, 2, 3]})
    result = align(frame.lazy(), steps, _context()).collect()
    assert result.to_dict(as_series=False) == {"value": [4.0, 6.0], "double": [8.0, 12.0]}


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"columns": {"a": "x", "b": "x"}}, "several baseline columns"),
        ({"keys": ["k", "k"]}, "repeat"),
        ({"keys": ["k"], "ignore_columns": ["k"]}, "cannot also be ignored"),
        ({"where": "a >>> 1"}, "invalid where"),
        ({"derive": {"x": "a +* 1"}}, "invalid derived column"),
        ({"cast": {"x": {"dtype": "decimal"}}}, "scale"),
        ({"levels": ["nope"]}, "unknown comparison level"),
        ({"levels": ["records"]}, "needs the comparison's 'keys'"),
        ({"levels": [{"level": "groups", "nonsense": 1}]}, "invalid comparison level"),
        ({"levels": ["structure", "structure"]}, "repeat the name"),
        ({"steps": [{"step": "missing"}]}, "unknown alignment step"),
        ({"recode": {"c": {"values": {}}}}, "at least one"),
    ],
)
def test_configuration_errors_fail_at_load(spec, message):
    with pytest.raises(ValidationError, match=message):
        ComparisonSpec.model_validate({"baseline": "b", "candidate": "c", **spec})


def test_a_refused_alignment_leaves_every_level_not_assessed():
    baseline = pl.DataFrame({"k": [1, 2], "v": [1.5, 2.0]})
    candidate = pl.DataFrame({"k": [1, 2], "v": [1, 2]})
    result = compare_datasets(baseline, candidate, keys="k", cast={"v": "int"})
    assert result.error is not None and "would change 1 value" in result.error
    assert not result.assessed and not result.passed
    assert [level.name for level in result.levels] == ["structure", "coverage", "records"]
    assert all(not check.assessed for level in result.levels for check in level.checks)


def test_missing_value_markers_of_a_dataset_apply_before_alignment():
    baseline = Dataset(
        pl.DataFrame({"k": [1, 2], "v": [-999, 5]}),
        schema={"columns": {"v": {"missing_values": [-999], "nullable": True}}},
    )
    candidate = pl.DataFrame({"k": [1, 2], "v": [None, 5]})
    result = compare_datasets(baseline, candidate, keys="k", levels=["records"])
    assert result.passed, result.summary()
