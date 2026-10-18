"""Independent and metamorphic contracts across public input representations."""

from fractions import Fraction

import polars as pl
import pytest

from qavf import Query, ResourceLimits, validate
from qavf.regression import compare_frames


def test_documented_nested_bounds_exclude_validator_object_addresses():
    from qavf.docgen import _type_name
    from qavf.settings import NumericBound

    assert _type_name(tuple[NumericBound, ...] | None) == "tuple[int | Decimal | float, ...] | None"


@pytest.mark.parametrize("representation", ["frame", "lazy", "file", "query"])
@pytest.mark.parametrize("operator", ["ge", "gt", "le", "lt"])
def test_float_integer_bounds_match_exact_reference(tmp_path, representation, operator):
    values = [float(2**53 - 1), float(2**53), float(2**53 + 2)]
    boundary = 2**53 + 1
    comparisons = {
        "ge": lambda x: x >= boundary,
        "gt": lambda x: x > boundary,
        "le": lambda x: x <= boundary,
        "lt": lambda x: x < boundary,
    }
    expected = sum(not comparisons[operator](Fraction(x)) for x in values)
    frame = pl.DataFrame({"value": values})
    path = tmp_path / "values.parquet"
    frame.write_parquet(path)
    data = {
        "frame": frame,
        "lazy": frame.lazy(),
        "file": path,
        "query": Query("SELECT * FROM read_parquet($path)", parameters={"path": str(path)}),
    }[representation]
    report = validate(data, schema={"columns": {"value": {operator: boundary}}})
    failures = [r for r in report.results if r.rule_id == "INP-003" and r.status.value == "FAIL"]
    assert sum(r.affected_count for r in failures) == expected
    assert all(r.status.value != "ERROR" for r in report.results)


@pytest.mark.parametrize("evidence_rows,evidence_bytes", [(0, 0), (1, 256), (20, 4096)])
def test_evidence_limits_do_not_change_assessment(evidence_rows, evidence_bytes):
    frame = pl.DataFrame({"x": [-1] * 100 + [1] * 100})
    report = validate(
        frame,
        schema={"columns": {"x": {"ge": 0}}},
        resources=ResourceLimits(evidence_rows=evidence_rows, evidence_bytes=evidence_bytes),
    )
    failure = next(r for r in report.results if r.rule_id == "INP-003" and r.status.value == "FAIL")
    assert failure.affected_count == 100
    assert report.verdict() == "failed"
    assert len(failure.affected_records) <= evidence_rows


def test_keyed_regression_is_invariant_to_row_order_and_evidence_limit():
    baseline = pl.DataFrame({"id": [1, 2, 3], "value": [[1, 2], [3, 4], [5, 6]]})
    candidate = pl.DataFrame({"id": [1, 2, 3], "value": [[1, 9], [3, 4], [5, 8]]})
    for left, right in [
        (baseline, candidate),
        (baseline.reverse(), candidate),
        (baseline, candidate.reverse()),
    ]:
        for limit in [0, 1, 10]:
            result = compare_frames(left, right, ["id"], max_differences=limit, rtol=0, atol=0)
            assert result.values_changed == 2
            assert not result.equal
            assert len(result.differences) <= limit
