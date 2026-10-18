"""Adversarial exactness, bounded evidence, isolation and native-scan contracts."""

import math
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.rules.freshness_checks import FreshnessRule
from qavf.rules.helpers import aggregate
from qavf.rules.input_checks import ColumnCheckRule, SchemaConformanceRule
from qavf.rules.reconciliation_checks import KeyGridRule
from qavf.settings import QAVFSettings
from qavf.testing import rule_results


def results(rule, frames, params=None, datasets=None):
    return rule_results(rule, frames, params, datasets=datasets or {name: {} for name in frames})


def reconcile(a, b, **options):
    return results(
        "REC-001",
        {"a": a, "b": b},
        {
            "reconciliations": [
                {
                    "left": {"dataset": "a", "value": "v"},
                    "right": {"dataset": "b", "value": "v"},
                    **options,
                }
            ]
        },
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([2**53], [2**53 + 1]),
        ([2**62, 2**62], [-(2**63)]),
    ],
)
def test_exact_totals_never_round_or_wrap_to_equal(left, right):
    [result] = reconcile(pl.DataFrame({"v": left}), pl.DataFrame({"v": right}))
    assert result.status is Status.FAIL


@pytest.mark.parametrize("how", ["sum", "min", "max", "first", "last", "none"])
def test_exact_aggregations_preserve_large_integer_values(how):
    frame = pl.DataFrame({"v": [2**53 + 1], "t": [1]})
    kwargs = {"order_by": "t"} if how in ("first", "last") else {}
    assert aggregate(frame, [], "v", how=how, **kwargs)["v"][0] == 2**53 + 1


@pytest.mark.parametrize("value", [None, math.inf, -math.inf])
def test_missing_or_nonfinite_totals_do_not_certify_balance(value):
    frame = pl.DataFrame({"v": [value]}, schema={"v": pl.Float64})
    assert all(r.status is not Status.PASS for r in reconcile(frame, frame))


def test_empty_and_zero_overlap_are_unassessed_and_missing_side_is_distinct():
    empty = pl.DataFrame(schema={"k": pl.Int64, "v": pl.Float64})
    assert reconcile(empty, empty)[0].status is Status.ERROR
    a = pl.DataFrame({"k": [1], "v": [None]}, schema=empty.schema)
    assert reconcile(a, empty, on=["k"])[0].status is Status.FAIL
    a = pl.DataFrame({"k": [1], "v": [2.0]})
    b = pl.DataFrame({"k": [2], "v": [2.0]})
    assert reconcile(a, b, on=["k"], missing_side="ignore")[0].status is Status.ERROR


@pytest.mark.parametrize("reference", [False, True])
def test_expected_keys_are_not_truncated(reference):
    frames = {"d": pl.DataFrame({"k": [1]})}
    dimension = {"values": [1.9]}
    if reference:
        frames["r"] = pl.DataFrame({"k": [1.9]})
        dimension = {"from_dataset": "r"}
    [result] = results(
        "REC-003", frames, {"grids": [{"dataset": "d", "dimensions": {"k": dimension}}]}
    )
    assert result.status is Status.ERROR


@pytest.mark.parametrize(
    ("series", "member"),
    [
        (pl.Series("v", [16777216], dtype=pl.Float32), 16777217),
        (pl.Series("v", [Decimal("1.23")], dtype=pl.Decimal(10, 2)), Decimal("1.234")),
    ],
)
def test_membership_keeps_dtype_precision(series, member):
    frame = series.to_frame()
    for operator, expected in (("isin", Status.FAIL), ("notin", Status.PASS)):
        found = results(
            "INP-003",
            {"d": frame},
            datasets={"d": {"schema": {"columns": {"v": {operator: [member]}}}}},
        )
        assert any(r.status is expected for r in found)
        assert not any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize(
    ("value", "limits"),
    [
        (1e9 + 0.5, {"integral": True}),
        (1e7 + 0.125, {"max_decimals": 2}),
    ],
)
def test_fraction_validation_does_not_relax_with_magnitude(value, limits):
    found = results(
        "INP-003",
        {"d": pl.DataFrame({"v": [value]})},
        datasets={"d": {"schema": {"columns": {"v": limits}}}},
    )
    assert any(r.status is Status.FAIL for r in found)


def test_missing_configured_column_is_accounted_for_without_schema_rule():
    found = results(
        "INP-003",
        {"d": pl.DataFrame({"x": [1]})},
        datasets={"d": {"schema": {"columns": {"x": {"ge": 0}, "absent": {"ge": 0}}}}},
    )
    assert any(r.status is Status.ERROR and "absent" in r.check for r in found)


def test_bad_contract_does_not_erase_neighboring_results():
    frames = {
        "a": pl.DataFrame({"v": [1]}),
        "b": pl.DataFrame({"v": [2]}),
        "bad": pl.DataFrame({"v": ["broken"]}),
    }
    params = {
        "reconciliations": [
            {
                "name": name,
                "left": {"dataset": "a", "value": "v"},
                "right": {"dataset": target, "value": "v"},
            }
            for name, target in (("before", "b"), ("bad", "bad"), ("after", "b"))
        ]
    }
    found = {r.check: r.status for r in results("REC-001", frames, params)}
    assert found == {"before": Status.FAIL, "bad": Status.ERROR, "after": Status.FAIL}


@pytest.mark.parametrize(
    "bad", [pl.Series("batch", [], dtype=pl.String), pl.Series("batch", [math.nan])]
)
def test_every_batch_dataset_must_supply_a_valid_identifier(bad):
    good = pl.Series("batch", ["X"]) if bad.dtype == pl.String else pl.Series("batch", [math.nan])
    found = results(
        "REC-005",
        {"a": good.to_frame(), "b": bad.to_frame()},
        {"batches": [{"datasets": ["a", "b"], "column": "batch"}]},
    )
    assert not any(r.status is Status.PASS for r in found)


def test_huge_grid_uses_exact_cardinality_and_bounded_evidence(monkeypatch):
    frame = pl.DataFrame({"a": [0], "b": [0], "c": [0]})
    context = DataContext(frames={"d": frame}, evidence_limit=3)
    rule = KeyGridRule(
        params={
            "grids": [
                {
                    "dataset": "d",
                    "dimensions": {
                        "a": {"values": tuple(range(10_000))},
                        "b": {"values": tuple(range(10_000))},
                        "c": {"values": tuple(range(100))},
                    },
                }
            ]
        }
    )
    real_join = pl.DataFrame.join

    def no_cross(self, *args, **kwargs):
        assert kwargs.get("how") != "cross"
        return real_join(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "join", no_cross)
    try:
        [finding] = rule.evaluate(context)
        assert finding.affected_count == 10_000_000_000 - 1
        assert finding.affected.height == 3
        assert finding.affected.to_dicts() == [{"a": 0, "b": 0, "c": i} for i in (1, 2, 3)]
    finally:
        context.close()


def test_decimal_aggregate_limits_remain_exact():
    d = Decimal("9007199254740993.01")
    frame = pl.DataFrame({"v": pl.Series([d], dtype=pl.Decimal(30, 2))})
    spec = {"dtype": "decimal", "min": {"ge": d}, "max": {"le": d}, "sum": {"ge": d, "le": d}}
    found = results("INP-003", {"d": frame}, datasets={"d": {"schema": {"columns": {"v": spec}}}})
    assert all(r.status is Status.PASS for r in found)


def test_reserved_user_columns_survive_failure_evidence():
    frame = pl.DataFrame({"v": [-1], "__qavf_row": ["keep"], "__qavf_violates": ["also keep"]})
    found = results(
        "INP-003", {"d": frame}, datasets={"d": {"schema": {"columns": {"v": {"ge": 0}}}}}
    )
    failure = next(r for r in found if r.status is Status.FAIL)
    assert failure.affected_records[0]["__qavf_row"] == "keep"
    assert failure.affected_records[0]["__qavf_violates"] == "also keep"


def test_missing_policies_apply_to_units_and_primary_keys():
    frame = pl.DataFrame({"v": [-999.0], "u": [None]}, schema={"v": pl.Float64, "u": pl.String})
    spec = {"nullable": True, "missing_values": [-999], "unit_column": "u", "unit": "kg"}
    assert all(
        r.status is Status.PASS
        for r in results(
            "INP-003", {"d": frame}, datasets={"d": {"schema": {"columns": {"v": spec}}}}
        )
    )
    found = results(
        "INP-004",
        {"d": pl.DataFrame({"id": [math.nan]})},
        datasets={"d": {"schema": {"primary_key": ["id"]}}},
    )
    assert found[0].status is Status.FAIL


def test_freshness_uses_shared_clock_and_rejects_future_masking():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    context = DataContext(
        frames={"d": pl.DataFrame({"at": [datetime(2020, 1, 1), datetime(2099, 1, 1)]})},
        reference_time=now,
    )
    try:
        rule = FreshnessRule(
            params={"checks": [{"dataset": "d", "column": "at", "max_age": "P1D"}]}
        )
        [finding] = rule.evaluate(context)
        assert not finding.passed and "ahead" in finding.message
        assert finding.affected[0]["reference_time"] == now
    finally:
        context.close()


def test_null_grid_keys_have_row_count_and_evidence():
    found = results(
        "REC-003",
        {"d": pl.DataFrame({"k": [1, None]})},
        {"grids": [{"dataset": "d", "dimensions": {"k": {"values": [1]}}, "unexpected": True}]},
    )
    [failure] = [r for r in found if r.status is Status.FAIL]
    assert failure.affected_count == 1 and failure.unit == "rows"
    assert failure.affected_records == [{"k": None}]


@pytest.mark.parametrize("source_kind", ["lazy", "parquet"])
def test_simple_input_checks_do_not_materialize_large_source(source_kind, tmp_path, monkeypatch):
    frame = pl.DataFrame({"v": [-1, 2, 3]})
    dataset = {"schema": {"columns": {"v": {"ge": 0}}, "min_rows": 3}}
    frames = {"d": frame.lazy()}
    if source_kind == "parquet":
        path = tmp_path / "data.parquet"
        frame.write_parquet(path)
        dataset["path"] = str(path)
        frames = {}
    settings = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "datasets": {"d": dataset}}
    )
    context = DataContext.from_settings(settings, frames=frames)

    def forbidden(*args, **kwargs):
        raise AssertionError("full-frame get is forbidden for this native check")

    monkeypatch.setattr(context, "get", forbidden)
    try:
        assert any(not f.passed and f.assessed for f in ColumnCheckRule().evaluate(context))
        assert all(f.passed for f in SchemaConformanceRule().evaluate(context))
    finally:
        context.close()


def test_column_evaluation_logs_redact_exception_and_check_credentials(monkeypatch, caplog):
    from qavf.rules.input_checks import _column_values

    def broken_statistic(*args, **kwargs):
        raise ValueError("postgres://user:db-secret@host/database password=other-secret")

    monkeypatch.setattr(_column_values, "_statistic", broken_statistic)
    limits = {"checks": [{"name": "password=label-secret", "mean": {"ge": 0}}]}
    found = results(
        "INP-003",
        {"d": pl.DataFrame({"v": [1]})},
        datasets={"d": {"schema": {"columns": {"v": limits}}}},
    )
    assert any(r.status is Status.ERROR for r in found)
    assert "failed" in caplog.text
    assert all(
        secret not in caplog.text for secret in ("db-secret", "other-secret", "label-secret")
    )
