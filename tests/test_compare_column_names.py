"""Comparison options naming columns: checked against the schemas, derive in order."""

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import compare_datasets, validate

BASELINE = pl.DataFrame({"k": [1, 2], "x": [1.0, 2.0]})
CANDIDATE = pl.DataFrame({"k": [1, 2], "x": [1.0000001, 2.0]})


def _reasons(result) -> list[str]:
    return [check.message for level in result.levels for check in level.checks]


@pytest.mark.parametrize("levels", [["records"], None])
@pytest.mark.parametrize(
    ("options", "named"),
    [
        ({"tolerances": {"typo": {"exact": True}}}, "tolerances names column 'typo'"),
        ({"ignore_columns": ["typo2"]}, "ignore_columns names column 'typo2'"),
        ({"tolerances": {"xx": {"atol": 1.0}}}, "did you mean 'x'"),
    ],
)
def test_an_unknown_option_column_is_a_configuration_error(options, named, levels):
    result = compare_datasets(BASELINE, CANDIDATE, keys="k", levels=levels, **options)
    assert not result.passed
    assert result.error is not None and result.error.startswith("configuration error")
    assert all(named in reason and "Not assessed" in reason for reason in _reasons(result))


def test_a_column_on_one_side_only_may_be_ignored_or_tolerated():
    candidate = CANDIDATE.with_columns(pl.lit("new").alias("note"))
    result = compare_datasets(
        BASELINE,
        candidate,
        keys="k",
        levels=["records"],
        ignore_columns=["note"],
        tolerances={"x": {"rtol": 1e-3}},
    )
    assert result.error is None and result.passed


def test_option_names_use_the_aligned_candidate_names():
    baseline = BASELINE.rename({"x": "old_x"})
    renamed = compare_datasets(
        baseline,
        CANDIDATE,
        keys="k",
        levels=["records"],
        columns={"old_x": "x"},
        tolerances={"x": {"exact": True}},
    )
    assert renamed.error is None and not renamed.passed  # x is compared exactly
    stale = compare_datasets(
        baseline,
        CANDIDATE,
        keys="k",
        levels=["records"],
        columns={"old_x": "x"},
        tolerances={"old_x": {"exact": True}},
    )
    assert "tolerances names column 'old_x'" in (stale.error or "")


def test_a_derived_column_is_a_known_option_name():
    result = compare_datasets(
        BASELINE, CANDIDATE, keys="k", derive={"x2": "x*2"}, tolerances={"x2": {"rtol": 1e-3}}
    )
    assert result.error is None


def test_the_compare_check_reports_the_same_configuration_error():
    config = {
        "datasets": {"b": {}, "c": {}},
        "checks": [
            {
                "check": "compare",
                "datasets": ["b", "c"],
                "keys": ["k"],
                "levels": ["records"],
                "tolerances": {"typo": {"exact": True}},
            }
        ],
    }
    report = validate({"b": BASELINE, "c": CANDIDATE}, config=config)
    assert report.verdict() != "passed"
    messages = [result.message for result in report.results if result.rule_id == "CMP-001"]
    assert messages and all("tolerances names column 'typo'" in m for m in messages)


def test_derived_columns_apply_in_mapping_order():
    result = compare_datasets(
        BASELINE, BASELINE, keys="k", derive={"x2": "x*2", "x3": "x2+1"}, levels=["records"]
    )
    assert result.error is None and result.passed
    changed = compare_datasets(
        BASELINE,
        BASELINE.with_columns(pl.col("x") + 1),
        keys="k",
        derive={"x2": "x*2", "x3": "x2+1"},
        levels=["records"],
        decision="material",
    )
    assert changed.level("records").values_changed == 6  # x, x2 and x3 on two rows


@pytest.mark.parametrize(
    ("derive", "named"),
    [
        ({"x3": "x2+1", "x2": "x*2"}, "derived column 'x3' uses 'x2'"),
        ({"y": "z+1"}, "derived column 'y' uses 'z'"),
        ({"y": "xx+1"}, "did you mean 'x'"),
    ],
)
def test_a_derive_reference_to_a_missing_column_is_a_configuration_error(derive, named):
    result = compare_datasets(BASELINE, CANDIDATE, keys="k", derive=derive)
    assert result.error is not None and result.error.startswith("configuration error")
    assert named in result.error


@pytest.mark.parametrize(
    "level",
    [
        {"level": "groups", "group_by": ["g"], "measures": ["sum(x)"]},
        {"level": "temporal", "time_column": "t", "bucket": "day", "measures": ["mean(x)"]},
        {"level": "contribution", "dimensions": ["g"], "measure": "sum(x)"},
    ],
)
def test_a_measure_of_an_ignored_column_is_refused(level):
    with pytest.raises(ValidationError, match="uses ignored column"):
        compare_datasets(BASELINE, CANDIDATE, levels=[level], ignore_columns=["x"])
