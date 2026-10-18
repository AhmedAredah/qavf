"""Schema-only planning preserves evidence while limiting the loaded projection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import polars as pl
import pytest

from qavf.keys import KeyTypeError
from qavf.regression._plan import comparison_plan


def test_projection_keeps_ordered_metadata_and_full_original_schemas():
    baseline = {
        "removed_z": pl.Int64,
        "value": pl.Float32,
        "key": pl.Int32,
        "incompatible": pl.String,
        "removed_a": pl.String,
        "ignored": pl.String,
    }
    candidate = {
        "added_z": pl.Int64,
        "key": pl.Int64,
        "incompatible": pl.Int64,
        "value": pl.Float64,
        "added_a": pl.String,
    }
    plan = comparison_plan(baseline, candidate, "key", ignore_columns="ignored")

    assert plan.keys == ("key",)
    assert plan.shared == ("value", "key")
    assert plan.values == ("value",)
    assert plan.columns_added == ("added_z", "added_a")
    assert plan.columns_removed == ("removed_z", "removed_a")
    assert plan.dtypes_changed == ("incompatible",)
    assert dict(plan.baseline_schema) == baseline
    assert dict(plan.candidate_schema) == candidate


@pytest.mark.parametrize(
    "policy,shared,changed",
    [
        ("compatible", ("key", "numeric"), ("different",)),
        ("strict", ("key",), ("key", "numeric", "different")),
        ("ignore", ("key", "numeric", "different"), ()),
    ],
)
def test_schema_policy_controls_values_without_losing_keys(policy, shared, changed):
    plan = comparison_plan(
        {"key": pl.Int32, "numeric": pl.Float32, "different": pl.String},
        {"key": pl.Int64, "numeric": pl.Float64, "different": pl.Int64},
        ["key"],
        schema=policy,
    )
    assert plan.shared == shared
    assert plan.dtypes_changed == changed


@pytest.mark.parametrize("container", ["struct", "list", "array"])
@pytest.mark.parametrize("policy", ["compatible", "strict", "ignore"])
def test_nested_structure_changes_need_no_rows(container, policy):
    left, right = pl.Struct({"old": pl.Int64}), pl.Struct({"new": pl.Int64})
    if container == "list":
        left, right = pl.List(left), pl.List(right)
    elif container == "array":
        left, right = pl.Array(left, 2), pl.Array(right, 2)
    plan = comparison_plan(
        {"key": pl.Int64, "nested": left},
        {"key": pl.Int64, "nested": right},
        "key",
        schema=policy,
    )
    assert plan.dtypes_changed == (() if policy == "ignore" else ("nested",))
    assert plan.values == (() if policy == "strict" else ("nested",))


def test_array_shape_and_nested_type_changes_are_distinct():
    plan = comparison_plan(
        {"key": pl.Int64, "shape": pl.Array(pl.Int64, 2), "kind": pl.List(pl.Int64)},
        {"key": pl.Int64, "shape": pl.Array(pl.Int64, 3), "kind": pl.List(pl.String)},
        "key",
    )
    assert plan.dtypes_changed == ("shape", "kind")
    assert plan.values == ("shape",)


def test_plan_never_constructs_or_collects_frames(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planning must inspect schemas without reading values")

    monkeypatch.setattr(pl.LazyFrame, "collect", forbidden)
    monkeypatch.setattr(pl, "DataFrame", forbidden)
    plan = comparison_plan({"key": pl.Int64}, {"key": pl.Int64}, "key")
    assert plan.values == () and plan.shared == ("key",)


def test_plan_detaches_mutable_configuration_and_prevents_reassignment():
    baseline = {"key": pl.Int32, "value": pl.Float64}
    candidate = {"key": pl.Int64, "value": pl.Float64}
    keys = ["key"]
    pair = [0, 1]
    overrides = {"value": pair}
    plan = comparison_plan(baseline, candidate, keys, column_tolerances=overrides)
    baseline.clear()
    candidate["value"] = pl.String
    keys.append("value")
    pair[0] = 10
    overrides.clear()

    assert plan.keys == ("key",)
    assert plan.baseline_schema["key"] == pl.Int32
    assert plan.candidate_schema["value"] == pl.Float64
    assert plan.tolerance("value") == (0, 1)
    with pytest.raises(TypeError):
        plan.baseline_schema["new"] = pl.String
    with pytest.raises(TypeError):
        plan.candidate_schema["value"] = pl.String
    with pytest.raises(TypeError):
        plan.column_tolerances["value"] = (1, 1)
    with pytest.raises(FrozenInstanceError):
        plan.name = "changed"


def test_tolerances_preserve_exact_numbers_and_zero_evidence_bounds():
    exact = Decimal("0.00000000000000000000001")
    plan = comparison_plan(
        {"key": pl.Int64, "value": pl.Decimal(38, 23)},
        {"key": pl.Int64, "value": pl.Decimal(38, 23)},
        "key",
        atol=exact,
        column_tolerances={"value": (0, 0)},
        max_differences=0,
        max_evidence_bytes=0,
        name="measurements",
    )
    assert plan.name == "measurements"
    assert plan.max_differences == plan.max_evidence_bytes == 0
    assert plan.tolerance("value") == (0, 0)
    assert plan.tolerance("unconfigured") == plan.tolerance(None) == (1e-6, exact)


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_missing_keys_are_rejected_from_each_complete_schema(side):
    complete = {"first": pl.Int64, "second": pl.Int64}
    incomplete = {"other": pl.Int64}
    baseline, candidate = (incomplete, complete) if side == "baseline" else (complete, incomplete)
    with pytest.raises(ValueError, match=rf"{side} lacks key column\(s\) second, first"):
        comparison_plan(baseline, candidate, ["second", "first"])


@pytest.mark.parametrize("policy", ["compatible", "strict", "ignore"])
@pytest.mark.parametrize(
    "left,right",
    [
        (pl.Int64, pl.String),
        (pl.Date, pl.String),
        (pl.Datetime("us"), pl.Datetime("us", "UTC")),
        (pl.List(pl.Int64), pl.List(pl.String)),
    ],
)
def test_incompatible_original_key_kinds_are_rejected_for_every_policy(policy, left, right):
    with pytest.raises(KeyTypeError, match="inventory: key 'key' has incompatible types"):
        comparison_plan(
            {"key": left}, {"key": right}, "key", schema=policy, name="inventory"
        )


@pytest.mark.parametrize("left,right", [(pl.Int64, pl.UInt64), (pl.Int64, pl.Float64)])
def test_value_dependent_key_cast_decisions_are_deferred(left, right):
    plan = comparison_plan({"key": left}, {"key": right}, "key")
    assert plan.shared == ("key",)
    assert plan.baseline_schema["key"] == left
    assert plan.candidate_schema["key"] == right


@pytest.mark.parametrize("argument", ["rtol", "atol"])
@pytest.mark.parametrize("invalid", [True, "1", -1, float("nan"), float("inf"), Decimal("NaN")])
def test_invalid_global_tolerances_fail_before_loading(argument, invalid):
    with pytest.raises(ValueError, match=argument):
        comparison_plan({"key": pl.Int64}, {"key": pl.Int64}, "key", **{argument: invalid})


@pytest.mark.parametrize("argument", ["max_differences", "max_evidence_bytes"])
@pytest.mark.parametrize("invalid", [True, -1, 1.5, "1"])
def test_invalid_evidence_limits_fail_before_loading(argument, invalid):
    with pytest.raises(ValueError, match=argument):
        comparison_plan({"key": pl.Int64}, {"key": pl.Int64}, "key", **{argument: invalid})


@pytest.mark.parametrize(
    "options,message",
    [
        ({"keys": []}, "at least one key"),
        ({"keys": ["key", "key"]}, "keys repeat"),
        ({"keys": [""]}, "nonempty strings"),
        ({"keys": [1]}, "nonempty strings"),
        ({"schema": "unknown"}, "unknown schema policy"),
        ({"ignore_columns": ["key"]}, "cannot also be ignored"),
        ({"ignore_columns": ["missing"]}, "in neither table"),
        ({"ignore_columns": [""]}, "nonempty strings"),
        ({"column_tolerances": {"missing": (0, 0)}}, "in neither table"),
        ({"column_tolerances": {"key": (0, 0)}}, "key or ignored"),
        (
            {"ignore_columns": ["value"], "column_tolerances": {"value": (0, 0)}},
            "key or ignored",
        ),
        ({"column_tolerances": []}, "must be a mapping"),
        ({"column_tolerances": {"": (0, 0)}}, "nonempty string columns"),
        ({"column_tolerances": {"value": None}}, "pair"),
        ({"column_tolerances": {"value": (0,)}}, "pair"),
        ({"column_tolerances": {"value": (0, 0, 0)}}, "pair"),
        ({"column_tolerances": {"value": "00"}}, "pair"),
        ({"column_tolerances": {"value": (True, 0)}}, "rtol of 'value'"),
        ({"column_tolerances": {"value": (0, float("inf"))}}, "atol of 'value'"),
    ],
)
def test_invalid_column_and_schema_configuration(options, message):
    arguments = {"keys": "key", **options}
    with pytest.raises(ValueError, match=message):
        comparison_plan(
            {"key": pl.Int64, "value": pl.Int64},
            {"key": pl.Int64, "value": pl.Int64},
            **arguments,
        )
