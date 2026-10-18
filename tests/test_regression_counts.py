"""Exact regression counts with bounded construction and structural schemas."""

from __future__ import annotations

import json

import polars as pl
import pytest

from qavf.audit_codec import evidence_json_size
from qavf.engine.registry import RuleRegistry
from qavf.engine.runner import QARunner
from qavf.regression import accept_baseline, compare_frames, save_baseline, verify_baseline
from qavf.rules.regression_checks import BaselineComparisonRule
from qavf.settings import QAVFSettings


def nested_frame(start=0, size=100_000):
    return pl.DataFrame(
        {"key": [1], "value": pl.Series([range(start, start + size)], dtype=pl.List(pl.Int64))}
    )


@pytest.mark.parametrize("byte_limit,row_limit", [(0, 10), (10_000, 0), (0, 0)])
def test_zero_allowance_does_not_convert_nested_evidence(monkeypatch, byte_limit, row_limit):
    a, b = nested_frame(), nested_frame(1)

    original = pl.Series.to_list

    def bounded(series, *args, **kwargs):
        assert len(series) <= 16, "zero allowance must not convert whole nested cells"
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pl.Series, "to_list", bounded)
    result = compare_frames(
        a,
        b,
        "key",
        rtol=0,
        atol=0,
        max_differences=row_limit,
        max_evidence_bytes=byte_limit,
    )
    assert result.values_changed == result.rows_changed == 1
    assert not result.differences and result.truncated


def test_large_nested_preview_has_changed_leaf_and_complete_counts(monkeypatch):
    a, b = nested_frame(), nested_frame(1)

    original = pl.Series.to_list

    def bounded(series, *args, **kwargs):
        assert len(series) <= 16, "nested previews must not convert a complete Series"
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pl.Series, "to_list", bounded)
    result = compare_frames(a, b, "key", rtol=0, atol=0, max_evidence_bytes=4096)
    [difference] = result.differences
    assert result.values_changed == result.rows_changed == 1
    assert difference.baseline_previewed and difference.candidate_previewed
    assert difference.baseline_length == difference.candidate_length == 100_000
    assert len(difference.baseline) <= 16 and difference.path == [0]
    assert difference.leaf["baseline"] == 0 and difference.leaf["candidate"] == 1
    assert evidence_json_size([d.model_dump(mode="python") for d in result.differences]) <= 4096
    assert json.loads(result.model_dump_json())["differences"][0]["baseline_previewed"]


def test_regression_rule_passes_zero_engine_evidence_allowances(tmp_path, monkeypatch):
    baseline, candidate = nested_frame(), nested_frame(1)
    path = tmp_path / "baseline.parquet"
    accept_baseline(baseline, path, "key")

    original = pl.Series.to_list

    def bounded(series, *args, **kwargs):
        assert len(series) <= 16, "the runner disabled evidence conversion"
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pl.Series, "to_list", bounded)
    registry = RuleRegistry()
    registry.register(BaselineComparisonRule)
    settings = QAVFSettings(
        plugins={"entry_points": False},
        engine={"parallel": False, "max_evidence_bytes": 0, "max_affected_records": 0},
        rules={
            "REG-001": {
                "params": {"baselines": [{"dataset": "out", "path": str(path), "keys": ["key"]}]}
            }
        },
    )
    result = QARunner(settings, registry).run(frames={"out": candidate}).get("REG-001")
    assert result.status.value == "FAIL" and result.affected_count == 1
    assert not result.affected_records


@pytest.mark.parametrize("rows", [0, 1])
@pytest.mark.parametrize("container", ["struct", "list", "array"])
def test_nested_field_replacement_is_detected_without_nonnull_data(rows, container):
    left, right = pl.Struct({"old": pl.Int64}), pl.Struct({"new": pl.Int64})
    if container == "list":
        left, right = pl.List(left), pl.List(right)
    elif container == "array":
        left, right = pl.Array(left, 2), pl.Array(right, 2)
    a = pl.DataFrame(
        {"key": [1] * rows, "value": [None] * rows}, schema={"key": pl.Int64, "value": left}
    )
    b = pl.DataFrame(
        {"key": [1] * rows, "value": [None] * rows}, schema={"key": pl.Int64, "value": right}
    )
    result = compare_frames(a, b, "key")
    assert not result.equal and result.dtypes_changed == ["value"]
    assert result.values_changed == 0
    assert compare_frames(a, b, "key", schema="ignore").equal


def test_array_shape_change_is_detected_on_empty_data():
    a = pl.DataFrame(schema={"key": pl.Int64, "value": pl.Array(pl.Int64, 2)})
    b = pl.DataFrame(schema={"key": pl.Int64, "value": pl.Array(pl.Int64, 3)})
    assert compare_frames(a, b, "key").dtypes_changed == ["value"]


def test_scalar_key_name_has_one_meaning_across_baseline_apis(tmp_path):
    frame = pl.DataFrame({"county": [2, 1], "value": [20, 10]})
    accepted = tmp_path / "accepted.parquet"
    accept_baseline(frame, accepted, "county")
    assert verify_baseline(accepted, "county") == []
    saved = save_baseline(frame, tmp_path / "saved.parquet", "county")
    assert pl.read_parquet(saved)["county"].to_list() == [1, 2]


def test_all_retained_differences_obey_encoded_byte_and_row_caps():
    a = pl.DataFrame({"key": range(100), "value": ["x" * 100_000] * 100})
    b = a.with_columns(pl.lit("y" * 100_000).alias("value"))
    result = compare_frames(a, b, "key", max_differences=3, max_evidence_bytes=2048)
    assert result.values_changed == result.rows_changed == 100
    assert 0 < len(result.differences) <= 3 and result.truncated
    assert all(d.baseline_previewed for d in result.differences)
    assert evidence_json_size([d.model_dump(mode="python") for d in result.differences]) <= 2048
