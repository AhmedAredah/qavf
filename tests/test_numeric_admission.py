"""Original scalar identity and deterministic ordered QA contracts."""

from decimal import Decimal
from itertools import permutations

import numpy as np
import polars as pl
import pytest

from qavf._observations import float_observations
from qavf.engine.context import DataContext
from qavf.rules.helpers import AggregatedMetric, aggregate
from qavf.rules.temporal_checks import CumulativeRule
from qavf.stats import OutlierMethod, detect_outliers


@pytest.mark.parametrize("method", list(OutlierMethod))
@pytest.mark.parametrize("container", [list, tuple, lambda values: np.array(values, dtype=object)])
def test_mixed_scalars_cannot_erase_integer_variation(method, container):
    values = container([2**53 + 1] * 4 + [float(2**53)])
    with pytest.raises(ValueError, match="exactly as Float64"):
        detect_outliers(values, method=method)


@pytest.mark.parametrize("boolean", [True, False, np.bool_(True)])
def test_mixed_boolean_is_rejected_before_numeric_dtype_inference(boolean):
    with pytest.raises(ValueError, match="booleans"):
        detect_outliers([boolean, 1, 1, 1, 9], method="iqr")


@pytest.mark.parametrize("container", [list, lambda values: np.array(values, dtype=object)])
def test_extended_float_scalar_is_checked_in_its_original_precision(container):
    original = np.nextafter(np.longdouble(1), np.longdouble(2))
    values = container([original] * 4 + [np.longdouble(1)])
    if original == np.longdouble(float(original)):
        # Platforms where longdouble is Float64 have no extra precision to lose.
        assert np.array_equal(float_observations(values), np.asarray(values, dtype=np.float64))
    else:
        with pytest.raises(ValueError, match="exactly as Float64"):
            detect_outliers(values, method="iqr")


@pytest.mark.parametrize("values", [
    np.array([2**64 - 1], dtype=np.uint64),
    np.array([2**63 - 1], dtype=np.int64),
    [10**400], [Decimal("0.1")], ["1", 2], [1 + 0j, 2],
])
def test_unsupported_or_lossy_observations_fail_explicitly(values):
    with pytest.raises(ValueError):
        float_observations(values)


def test_safe_mixed_values_match_homogeneous_array_without_losing_missing_positions():
    values = [1, 2.0, np.float32(3), np.int64(4), Decimal("5.5"), None, np.nan, np.inf]
    actual = float_observations(values)
    expected = np.array([1, 2, 3, 4, 5.5, np.nan, np.nan, np.inf], dtype=np.float64)
    assert np.array_equal(actual, expected, equal_nan=True)
    source = np.array([1, 2, 3], dtype=np.float64)
    assert float_observations(source) is source


@pytest.mark.parametrize("how", ["first", "last"])
def test_conflicting_selected_order_ties_rejected_for_every_row_permutation(how):
    for rows in permutations([(0, 10), (0, 20), (1, 30)]):
        order, values = zip(*rows, strict=True)
        # Reverse the direction so the tied key is selected by either operation.
        order = [-value for value in order] if how == "last" else order
        frame = pl.DataFrame({"order": order, "value": values})
        with pytest.raises(ValueError, match="conflicting values tied"):
            aggregate(frame, [], "value", how=how, order_by="order")


@pytest.mark.parametrize("how,expected", [("first", 20), ("last", 10)])
def test_explicit_secondary_order_is_permutation_invariant(how, expected):
    rows = [("g", 1, 2, 10), ("g", 1, 1, 20)]
    for ordered in permutations(rows):
        frame = pl.DataFrame(ordered, schema=["g", "order", "tie", "value"], orient="row")
        result = aggregate(frame, ["g"], "value", how=how, order_by=("order", "tie"))
        assert result.to_dicts() == [{"g": "g", "value": expected}]


def test_equal_ties_are_safe_and_nonselected_conflicts_do_not_block_selection():
    frame = pl.DataFrame({"order": [0, 0, 1, 1], "value": [5, 5, 10, 20]})
    assert aggregate(frame, [], "value", how="first", order_by="order").item() == 5


@pytest.mark.parametrize("how", ["first", "last"])
def test_missing_ordering_components_and_values_are_not_selected(how):
    frame = pl.DataFrame({
        "order": [None, float("nan"), 1.0, 1.0, 1.0],
        "tie": [1, 1, None, 1, 2], "value": [100, 200, 300, None, 5],
    })
    assert aggregate(frame, [], "value", how=how, order_by=("order", "tie")).item() == 5


def test_order_contract_normalizes_lists_and_projects_all_tie_break_columns():
    metric = AggregatedMetric(
        dataset="data", value="x", aggregate="first", order_by=["order", "tie"]
    )
    assert metric.order_by == ("order", "tie")
    assert metric.aggregation_columns == ("order", "tie")
    for invalid in ([], ["order", "order"], [""], ""):
        with pytest.raises(ValueError):
            AggregatedMetric(dataset="data", value="x", aggregate="first", order_by=invalid)


def test_temporal_verdict_cannot_depend_on_conflicting_tied_row_order():
    for values in ([100, 0, 50], [0, 100, 50]):
        frame = pl.DataFrame({"year": [1, 1, 2], "order": [0, 0, 1], "value": values})
        with DataContext(frames={"data": frame}) as context:
            findings = CumulativeRule({"series": [{
                "dataset": "data", "value": "value", "aggregate": "first", "order_by": "order",
            }]}).evaluate(context)
        assert findings and all(not finding.assessed for finding in findings)
        assert any("conflicting values tied" in finding.message for finding in findings)
