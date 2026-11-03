"""Optional record descriptions never replace the counts and verdict they describe."""

from decimal import Decimal

import polars as pl
import pytest

from qavf import compare_datasets
from qavf.compare.levels import records
from qavf.regression import compare_outputs

LARGEST = 10**38 - 1


def _decimals(values: list[int]) -> pl.DataFrame:
    column = pl.Series(values, dtype=pl.Int128).cast(pl.Decimal(38, 0))
    return pl.DataFrame({"k": list(range(len(values))), "x": column})


@pytest.mark.parametrize("decision", ["material", "material_and_significant"])
@pytest.mark.parametrize(
    ("before", "after"),
    [([LARGEST], [-LARGEST]), ([-LARGEST], [LARGEST]), ([LARGEST, 5], [-LARGEST, -5])],
)
def test_extreme_decimal_changes_keep_their_counts_and_failure(before, after, decision):
    baseline, candidate = _decimals(before), _decimals(after)
    direct = compare_outputs(baseline, candidate, keys=["k"])
    result = compare_datasets(baseline, candidate, keys="k", levels=["records"], decision=decision)
    level = result.level("records")
    values = next(check for check in level.checks if check.check == "values")
    assert values.assessed and not values.passed
    assert level.rows_changed == direct.rows_changed == len(before)
    assert values.affected_count == direct.values_changed
    assert level.columns[0].max_abs_diff == pytest.approx(2.0 * LARGEST)
    assert "description_unavailable" not in values.metrics
    assert not result.passed


def test_a_scaled_extreme_decimal_change_is_described():
    text = "9" * 28 + "." + "9" * 10
    largest = Decimal(text)
    baseline = pl.DataFrame({"k": [1], "x": pl.Series([text]).cast(pl.Decimal(38, 10))})
    candidate = pl.DataFrame({"k": [1], "x": pl.Series(["-" + text]).cast(pl.Decimal(38, 10))})
    level = compare_datasets(
        baseline, candidate, keys="k", levels=["records"], decision="material"
    ).level("records")
    assert level.rows_changed == 1
    assert level.columns[0].max_abs_diff == pytest.approx(float(2 * largest))


def _failing_description(*_args, **_kwargs):
    raise pl.exceptions.ComputeError("overflow in a description")


def test_a_failed_description_is_reported_beside_the_counts(monkeypatch):
    monkeypatch.setattr(records, "_describe", _failing_description)
    baseline = pl.DataFrame({"k": [1, 2], "x": [1.0, 2.0]})
    candidate = pl.DataFrame({"k": [1, 2], "x": [1.0, 3.0]})
    result = compare_datasets(
        baseline, candidate, keys="k", levels=["records"], decision="material"
    )
    level = result.level("records")
    values = next(check for check in level.checks if check.check == "values")
    assert values.assessed and not values.passed and values.affected_count == 1
    assert level.rows_changed == 1
    assert "overflow in a description" in values.metrics["description_unavailable"]["x"]
    assert not result.passed
