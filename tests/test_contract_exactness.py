"""Regression oracles for independently assessed, exact native input contracts."""

from decimal import Decimal

import pyarrow as pa
import pytest

from qavf import Dataset, validate


def run(data, schema=None):
    return validate(data, schema=schema, backend="duckdb", config={"engine": {"native_threads": 2}})


def checks(report, rule):
    return [r for r in report.results if r.rule_id == rule and r.status.value != "SKIPPED"]


def constraint(report, column="x"):
    """A column's limits: their own result, or the column summary that passed them."""
    results = checks(report, "INP-003")
    own = [r for r in results if r.check_id == "column.constraints" and r.columns == [column]]
    if own:
        return own[0]
    return next(
        r
        for r in results
        if r.check_id == "column.summary"
        and any(c.check_id == "column.constraints" and c.columns == [column] for c in r.checks)
    )


@pytest.mark.parametrize(
    "child,parent",
    [
        (pa.array([9007199254740993], type=pa.int64()), pa.array([9007199254740992.0])),
        (pa.array(["07"]), pa.array([7])),
    ],
)
def test_incompatible_foreign_keys_never_silently_coerce(child, parent):
    result = run(
        {
            "child": Dataset(
                pa.table({"x": child}), foreign_keys=[{"columns": ["x"], "ref_dataset": "parent"}]
            ),
            "parent": pa.table({"x": parent}),
        }
    )
    errors = checks(result, "INP-005")
    assert len(errors) == 1 and errors[0].status.value == "ERROR"
    assert "cannot be compared exactly" in errors[0].message


def test_foreign_keys_accept_proven_lossless_integer_float_alignment():
    result = run(
        {
            "child": Dataset(
                pa.table({"x": [7]}), foreign_keys=[{"columns": ["x"], "ref_dataset": "parent"}]
            ),
            "parent": pa.table({"x": [7.0]}),
        }
    )
    assert all(r.status.value == "PASS" for r in checks(result, "INP-005"))


@pytest.mark.parametrize(
    "nulls,expected", [("forbid", "FAIL"), ("exempt", "PASS"), ("equal", "FAIL")]
)
def test_undeclared_nan_obeys_key_null_policy(nulls, expected):
    report = run(
        pa.table({"x": [float("nan"), float("nan")]}),
        {"unique_keys": [{"columns": ["x"], "nulls": nulls}]},
    )
    results = checks(report, "INP-004")
    assert len(results) == 1 and results[0].status.value == expected
    if nulls == "forbid":
        assert results[0].affected_count == 2


def test_inferred_floating_values_still_enforce_finite_policy():
    report = run(pa.table({"x": [float("inf")]}), {"columns": {"x": {}}})
    result = next(r for r in checks(report, "INP-003") if r.check_id == "column.finite")
    assert result.status.value == "FAIL" and result.affected_count == 1


@pytest.mark.parametrize(
    "values,dtype,stat,bound,expected",
    [
        ([9007199254740993], "int", "mean", {"ge": 9007199254740993}, "PASS"),
        ([9007199254740993], "int", "mean", {"le": 9007199254740992}, "FAIL"),
        ([9007199254740993], "int", "median", {"ge": 9007199254740993}, "PASS"),
        ([9007199254740993, 9007199254740994], "int", "median", {"le": 9007199254740993}, "FAIL"),
        ([1e16, 1.0, -1e16], "float", "sum", {"ge": 1}, "PASS"),
        ([1e16, 1.0, -1e16], "float", "sum", {"le": 0}, "FAIL"),
        (
            [Decimal("1.000000000000000001")],
            "decimal",
            "mean",
            {"ge": Decimal("1.000000000000000001")},
            "PASS",
        ),
    ],
)
def test_exact_statistics_drive_decisions(values, dtype, stat, bound, expected):
    report = run(pa.table({"x": values}), {"columns": {"x": {"dtype": dtype, stat: bound}}})
    result = constraint(report)
    assert result.status.value == expected, result.message


def test_null_fraction_compares_exact_ratio():
    report = run(
        pa.table({"x": [None, 1, 2]}),
        {"columns": {"x": {"nullable": True, "max_null_fraction": 0.3333333333333333}}},
    )
    result = constraint(report)
    assert result.status.value == "FAIL"
    assert result.metrics["column_problems"] == [
        "missing fraction 0.3333 (null or NaN) exceeds 0.333333"
    ]


def test_bad_condition_does_not_suppress_later_constraints_or_columns():
    report = run(
        pa.table({"x": [1], "y": [-1]}),
        {"columns": {"x": {"checks": [{"where": "unknown > 0", "ge": 0}]}, "y": {"ge": 0}}},
    )
    results = checks(report, "INP-003")
    assert any(r.status.value == "ERROR" and r.columns == ["x"] for r in results)
    assert any(r.status.value == "FAIL" and r.columns == ["y"] for r in results)


def test_bad_fused_predicate_preserves_valid_results():
    # length() does not accept INTEGER; the second predicate must still run.
    report = run(
        pa.table({"x": [1], "y": [-1]}), {"columns": {"x": {"min_length": 2}, "y": {"ge": 0}}}
    )
    results = checks(report, "INP-003")
    assert any(r.status.value == "ERROR" and r.columns == ["x"] for r in results)
    assert any(r.status.value == "FAIL" and r.columns == ["y"] for r in results)


def test_expressions_isolate_errors_and_retain_names():
    report = run(
        Dataset(
            pa.table({"x": [1]}),
            expressions=[
                {"name": "broken", "expr": "unknown > 0"},
                {"name": "A", "expr": "x < 0"},
                {"name": "B", "expr": "x < 0"},
            ],
        )
    )
    results = checks(report, "INP-006")
    assert [r.status.value for r in results] == ["ERROR", "FAIL", "FAIL"]
    assert [r.parameters["name"] for r in results] == ["broken", "A", "B"]


def test_nullable_measurement_does_not_require_unit():
    report = run(
        pa.table(
            {
                "amount": pa.array([None], type=pa.float64()),
                "unit": pa.array([None], type=pa.string()),
            }
        ),
        {
            "columns": {
                "amount": {
                    "dtype": "float",
                    "nullable": True,
                    "unit_column": "unit",
                    "allowed_units": ["kg"],
                }
            }
        },
    )
    assert report.verdict() == "passed"


def test_evidence_retains_condition_dependencies():
    report = run(
        pa.table({"x": [-1, 1], "state": ["active", "inactive"]}),
        {"columns": {"x": {"checks": [{"where": "state='active'", "ge": 0}]}}},
    )
    result = next(r for r in checks(report, "INP-003") if r.status.value == "FAIL")
    assert result.affected_records[0]["state"] == "active"


def test_missing_key_column_has_exact_nonzero_structural_count():
    report = run(pa.table({"y": [1]}), {"primary_key": ["x"]})
    result = checks(report, "INP-004")[0]
    assert result.status.value == "FAIL" and result.affected_count == 1
    assert result.unit == "columns"


def test_bad_foreign_key_does_not_suppress_later_reference():
    report = run(
        {
            "child": Dataset(
                pa.table({"x": [1]}),
                foreign_keys=[
                    {"columns": ["absent"], "ref_dataset": "parent"},
                    {"columns": ["x"], "ref_dataset": "parent"},
                ],
            ),
            "parent": pa.table({"x": [2]}),
        }
    )
    assert [r.status.value for r in checks(report, "INP-005")] == ["ERROR", "FAIL"]


def test_bad_unit_contract_does_not_suppress_later_value_check():
    report = run(
        pa.table({"amount": [1], "unit": ["kg"], "x": [-1]}),
        {
            "columns": {
                "amount": {"unit_column": "unit_code", "allowed_units": ["kg"]},
                "x": {"ge": 0},
            }
        },
    )
    findings = checks(report, "INP-003")
    assert any(r.check_id == "column.units" and r.status.value == "ERROR" for r in findings)
    assert any(r.columns == ["x"] and r.status.value == "FAIL" for r in findings)


def test_native_fixed_width_sum_overflow_uses_bounded_exact_fallback():
    value = Decimal("9" * 38)
    report = run(
        pa.table({"x": pa.array([value, value], type=pa.decimal128(38, 0))}),
        {"columns": {"x": {"dtype": "decimal", "sum": {"ge": 2 * (10**38 - 1)}}}},
    )
    finding = constraint(report)
    assert finding.status.value == "PASS", finding.message
    report = run(
        pa.table({"x": pa.array([value, value], type=pa.decimal128(38, 0))}),
        {"columns": {"x": {"dtype": "decimal", "sum": {"le": 2 * (10**38 - 1) - 1}}}},
    )
    finding = constraint(report)
    assert finding.status.value == "FAIL", finding.message
    assert finding.affected_records[0]["problem"].startswith(f"sum {2 * (10**38 - 1)} outside")


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("parent_value,expected", [("1.250", "PASS"), ("1.251", "FAIL")])
def test_decimal_foreign_keys_share_lossless_common_scale(backend, reverse, parent_value, expected):
    left = pa.table({"x": pa.array([Decimal("1.25")], type=pa.decimal128(18, 2))})
    right = pa.table({"x": pa.array([Decimal(parent_value)], type=pa.decimal128(20, 3))})
    if reverse:
        left, right = right, left
    report = validate(
        {
            "child": Dataset(left, foreign_keys=[{"columns": ["x"], "ref_dataset": "parent"}]),
            "parent": right,
        },
        backend=backend,
        config={"engine": {"native_threads": 2}},
    )
    result = checks(report, "INP-005")[0]
    assert result.status.value == expected, result.message


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_decimal_foreign_keys_refuse_an_unrepresentable_common_domain(backend):
    left = pa.table({"x": pa.array([Decimal("1")], type=pa.decimal128(38, 0))})
    right = pa.table({"x": pa.array([Decimal("0.1")], type=pa.decimal128(38, 38))})
    report = validate(
        {
            "child": Dataset(left, foreign_keys=[{"columns": ["x"], "ref_dataset": "parent"}]),
            "parent": right,
        },
        backend=backend,
        config={"engine": {"native_threads": 2}},
    )
    result = checks(report, "INP-005")[0]
    assert result.status.value == "ERROR", result.message
