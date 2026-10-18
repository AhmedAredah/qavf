"""Baseline acceptance integrity and bounded-work regression comparison."""

from __future__ import annotations

import importlib
import json
import logging
import math
from collections import UserDict
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.io import read_table
from qavf.redaction import redact, redact_text
from qavf.regression import accept_baseline, compare_frames, verify_baseline
from qavf.reports.models import RuleResult, evidence_json_size, normalize_evidence
from qavf.rules.regression_checks import BaselineComparisonRule
from qavf.settings import QAVFSettings

comparison = importlib.import_module("qavf.regression._cells")


def frame(value, dtype=None):
    return pl.DataFrame({"k": [1], "v": pl.Series([value], dtype=dtype)})


def result(**fields):
    return RuleResult(
        rule_id="TST-001",
        name="test",
        description="test",
        category="test",
        status="FAIL",
        severity="ERROR",
        message="test",
        **fields,
    )


def run_rule(rule, **settings):
    registry = RuleRegistry()
    registry.register(rule)
    return QARunner(
        QAVFSettings.model_validate({"plugins": {"entry_points": False}, **settings}), registry
    ).run()


@pytest.mark.parametrize(
    "dtype,value,text",
    [
        (pl.Datetime("ns"), 1001, "1970-01-01T00:00:00.000001001"),
        (pl.Date, date(2025, 1, 1), "2025-01-01"),
        (pl.Duration("ns"), 1001, "1001"),
        (pl.Time, 1001, "1001"),
    ],
)
@pytest.mark.parametrize("container", ["list", "struct"])
def test_nested_temporal_semantics_survive_normalization(dtype, value, text, container):
    if container == "list":
        left, right = frame([value], pl.List(dtype)), frame([text], pl.List(pl.String))
    else:
        left = frame({"field": value}, pl.Struct({"field": dtype}))
        right = frame({"field": text}, pl.Struct({"field": pl.String}))
    out = compare_frames(left, right, "k")
    assert not out.equal and out.dtypes_changed == ["v"]
    assert not compare_frames(left, right, "k", schema="ignore").equal


def test_nested_compatible_widening_and_exact_numeric_tolerance():
    assert compare_frames(
        frame([1], pl.List(pl.Int32)), frame([1.0], pl.List(pl.Float64)), "k"
    ).equal
    assert compare_frames(
        frame([1000], pl.List(pl.Duration("ns"))), frame([1], pl.List(pl.Duration("us"))), "k"
    ).equal
    a = frame([Decimal("0.1")], pl.List(pl.Decimal(10, 1)))
    b = frame([0.1], pl.List(pl.Float64))
    assert not compare_frames(a, b, "k", rtol=0, atol=0).equal
    assert compare_frames(a, b, "k", rtol=0, atol=1e-16).equal
    assert compare_frames(
        frame({"x": [1.0, 2.0]}),
        frame({"x": [1.0001, 2.0001]}),
        "k",
        column_tolerances={"v": (0, 0.001)},
    ).equal


def test_accepted_manifest_cannot_hide_nested_type_change(tmp_path):
    path = tmp_path / "accepted.parquet"
    baseline = frame([datetime(2025, 1, 1)], pl.List(pl.Datetime("us")))
    accept_baseline(baseline, path, ["k"])
    registry = RuleRegistry()
    registry.register(BaselineComparisonRule)
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "rules": {
                "REG-001": {
                    "params": {
                        "require_manifest": True,
                        "baselines": [{"dataset": "out", "path": str(path), "keys": ["k"]}],
                    }
                }
            },
        }
    )
    out = QARunner(settings, registry).run(frames={"out": frame(["2025-01-01T00:00:00.000000000"])})
    assert out.get("REG-001").status is Status.FAIL and out.exit_code() == 1


@pytest.mark.parametrize(
    "name", ["state.json", "state.json.gz", "manifest.json", "payload.parquet"]
)
def test_baseline_payload_has_its_own_namespace(name, tmp_path):
    path = tmp_path / name
    for value in (1, 2):
        expected = frame(value)
        accept_baseline(expected, path, ["k"], replace=value == 2)
        assert verify_baseline(path) == []
        assert compare_frames(expected, read_table(path), "k").equal


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("nested", [False, True])
def test_duration_parquet_baseline_round_trip(unit, nested, tmp_path):
    dtype = pl.Duration(unit)
    expected = frame([-1001, 1002], pl.List(dtype)) if nested else frame(-1001, dtype)
    path = tmp_path / "duration.parquet"
    accept_baseline(expected, path, ["k"])
    assert verify_baseline(path) == []
    assert compare_frames(expected, read_table(path), "k", rtol=0, atol=0).equal


@pytest.mark.parametrize(
    "text",
    [
        "password='FAKE_HEAD''FAKE_TAIL' host=db",
        "ATTACH 'dbname=x password=''FAKE_HEAD FAKE_TAIL'''",
        "Driver={SQL};Pwd={FAKE_HEAD; FAKE_TAIL};Server=host",
        "Pwd={FAKE_HEAD}}FAKE_TAIL};Server=host",
        "password='FAKE_HEAD FAKE_TAIL",
        '{"clientSecret": "FAKE_HEAD FAKE_TAIL"}',
        "Authorization: Bearer FAKE_HEAD_FAKE_TAIL",
        "https://host/data?accessToken=FAKE_HEAD_FAKE_TAIL",
    ],
)
def test_complete_credential_values_are_removed(text):
    safe = redact_text(text)
    assert "FAKE_HEAD" not in safe and "FAKE_TAIL" not in safe


def test_credentials_do_not_survive_report_or_log_boundaries(caplog):
    class Secrets(Rule):
        rule_id = "TST-001"
        name = "secrets"
        description = "test"
        category = "test"

        def evaluate(self, ctx):
            raise RuntimeError("password='FAKE_HEAD''FAKE_TAIL' host=db")

    with caplog.at_level(logging.ERROR, logger="qavf"):
        report = run_rule(
            Secrets,
            extensions={
                "service": {
                    "accessToken": "FAKE_TOKEN",
                    "clientSecret": "FAKE_SECRET",
                    "refreshToken": "FAKE_REFRESH",
                }
            },
        )
    assert report.results[0].status is Status.ERROR
    for canary in ("FAKE_HEAD", "FAKE_TAIL", "FAKE_TOKEN", "FAKE_SECRET", "FAKE_REFRESH"):
        assert canary not in report.model_dump_json() and canary not in caplog.text
    assert redact({"ordinary": "visible"}) == {"ordinary": "visible"}


def test_supported_evidence_is_detached_and_immutable():
    values = {1, 2}
    mapping = UserDict({"array": np.array([3, 4]), "set": values})
    out = result(metrics={"mapping": mapping})
    values.add(99)
    mapping["array"][0] = 99
    assert out.metrics["mapping"]["set"] == [1, 2]
    assert out.metrics["mapping"]["array"] == [3, 4]
    with pytest.raises(TypeError):
        out.metrics["mapping"]["set"].append(99)
    raw = {1, 2}
    compared = compare_frames(frame(raw, pl.Object), frame({2, 3}, pl.Object), "k")
    raw.add(99)
    assert compared.differences[0].baseline == [1, 2]


def test_unsupported_or_cyclic_evidence_is_rejected_explicitly():
    class Mutable:
        pass

    with pytest.raises(ValidationError, match="unsupported evidence type"):
        result(metrics={"opaque": Mutable()})
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycles"):
        normalize_evidence(cycle)
    with pytest.raises(ValidationError, match="cycles"):
        result(metrics={"cycle": cycle})
    with pytest.raises(ValueError, match="colliding key"):
        normalize_evidence({1: "a", "1": "b"})
    with pytest.raises(ValidationError, match="colliding key"):
        result(metrics={"keys": {1: "a", "1": "b"}})


@pytest.mark.parametrize(
    "value",
    [
        {"values": [math.nan] * 1000},
        {"values": [{"$float": "NaN"}, {"$qavf_mapping": []}, math.inf, -math.inf]},
        {"values": ['é\\"' * 100], "date": date(2025, 1, 1), "decimal": Decimal("1.25")},
    ],
)
def test_evidence_size_matches_persisted_codec(value):
    out = result(metrics=value)
    actual = json.dumps(
        out.model_dump(mode="json")["metrics"], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert evidence_json_size(value) == len(actual)


def test_nonfinite_evidence_obeys_encoded_budget():
    class Evidence(Rule):
        rule_id = "TST-001"
        name = "evidence"
        description = "test"
        category = "test"

        def evaluate(self, ctx):
            return Finding.fail("test", affected=[{"values": [math.nan] * 1000}], affected_count=17)

    report = run_rule(Evidence, engine={"max_evidence_bytes": 12000})
    out = report.results[0]
    assert out.status is Status.FAIL and out.affected_count == 17
    assert evidence_json_size(out.affected_records) <= 12000
    assert not out.affected_records


def test_zero_evidence_nested_comparison_stops_after_first_changed_integer(monkeypatch):
    count = 0
    original = comparison.numbers_close

    def counted(*args):
        nonlocal count
        count += 1
        return original(*args)

    monkeypatch.setattr(comparison, "numbers_close", counted)
    left = frame(list(range(100_000)), pl.List(pl.Int64))
    right = frame(list(range(1, 100_001)), pl.List(pl.Int64))
    out = compare_frames(left, right, "k", rtol=0, atol=0, max_differences=0)
    assert out.values_changed == 1 and not out.differences and count == 1


def test_wide_comparison_does_not_rebuild_schema_per_column(monkeypatch):
    original = pl.DataFrame.schema.fget
    count = 0

    def schema(frame):
        nonlocal count
        count += 1
        return original(frame)

    data = pl.DataFrame({"k": [1], **{f"v{i}": [1.0] for i in range(300)}})
    monkeypatch.setattr(pl.DataFrame, "schema", property(schema))
    assert compare_frames(data, data, "k").equal
    assert count < 20


@pytest.mark.parametrize("value", [10**500, -(10**500)])
def test_unbounded_python_integer_values_remain_exact(value):
    out = compare_frames(frame(value, pl.Object), frame(value + 1, pl.Object), "k", rtol=0, atol=0)
    assert out.values_changed == 1 and out.max_abs_diff == 1


@pytest.mark.parametrize(
    "left,right",
    [
        ([math.nan], [1.0]),
        ({"x": [math.inf]}, {"x": [2.0]}),
        ([1.0, math.nan], [2.0, math.inf]),
    ],
)
def test_nested_nonfinite_changes_count_cells(left, right):
    out = compare_frames(frame(left), frame(right), "k", rtol=0, atol=0)
    assert out.values_changed == 1 and out.nonfinite_changes == 1 and out.differences[0].path


def test_nonfinite_equal_leaf_is_not_counted_for_another_change():
    out = compare_frames(frame([1.0, math.nan]), frame([2.0, math.nan]), "k", rtol=0, atol=0)
    assert out.values_changed == 1 and out.nonfinite_changes == 0


def test_regression_rule_declares_candidates_and_preserves_opaque_dependencies():
    settings = QAVFSettings()
    rule = BaselineComparisonRule(
        {"baselines": [{"dataset": "out", "path": "a.parquet", "keys": ["k"]}]}
    )
    assert rule.data_requirements(settings).datasets == frozenset({"out"})
    opaque = BaselineComparisonRule(
        {
            "baselines": [
                {"dataset": "out", "source": {"type": "python", "callable": "x:y"}, "keys": ["k"]}
            ]
        }
    )
    assert opaque.data_requirements(settings) is None


def test_added_nested_nonfinite_leaf_counts_once():
    out = compare_frames(frame({"a": 1}), frame({"a": 1, "b": [math.nan]}), "k")
    assert out.values_changed == 1 and out.nonfinite_changes == 1
    assert out.differences[0].path == ["b"]


def test_empty_nested_values_match_when_schema_is_explicitly_ignored():
    assert compare_frames(
        frame([], pl.List(pl.Date)), frame([], pl.List(pl.String)), "k", schema="ignore"
    ).equal
