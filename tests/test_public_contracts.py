"""Exact bounds, nested temporal regression and baseline round trips."""

import json
import math
from datetime import datetime

import polars as pl

from qavf import Query, validate
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.resources import ByteCache
from qavf.engine.runner import QARunner
from qavf.redaction import redact
from qavf.regression import accept_baseline, compare_frames, verify_baseline
from qavf.settings import QAVFSettings
from qavf.testing import rule_results


def test_exact_integer_bound_is_not_rounded_into_float_pass():
    report = validate(
        pl.DataFrame({"x": [float(2**53)]}), schema={"columns": {"x": {"ge": 2**53 + 1}}}
    )
    assert report.verdict() != "passed"


def test_nested_temporal_to_string_is_a_regression():
    a = pl.DataFrame({"id": [1], "v": [[datetime(2020, 1, 1)]]})
    b = pl.DataFrame({"id": [1], "v": [["2020-01-01T00:00:00.000000000"]]})
    assert not compare_frames(a, b, ["id"]).equal


def test_state_json_baseline_roundtrips(tmp_path):
    path = tmp_path / "state.json"
    accept_baseline(pl.DataFrame({"id": [1, 2], "v": [10, 20]}), path, keys=["id"])
    assert verify_baseline(path) == []


def test_same_path_override_preserves_row_budget(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("x\n1\n2\n")
    config = {
        "datasets": {
            "data": {
                "source": {"type": "file", "path": str(path), "row_budget": 1},
                "schema": {"columns": {"x": {"ge": 0}}},
            }
        },
        "plugins": {"entry_points": False},
    }
    assert validate(config=config).verdict() != "passed"
    assert validate(path, config=config).verdict() != "passed"


def test_missing_share_sentinel_does_not_balance():
    results = rule_results(
        "REC-002",
        {"s": pl.DataFrame({"x": [1.0, -999.0, 999.0]})},
        {"shares": [{"dataset": "s", "value": "x"}]},
        datasets={"s": {"schema": {"columns": {"x": {"missing_values": [-999]}}}}},
    )
    assert any(r.status.value in {"FAIL", "ERROR"} for r in results)


def test_native_series_cannot_bypass_byte_cache_limit():
    cache = ByteCache(1024)
    cache["s"] = pl.Series("x", range(1_000_000))
    assert "s" not in cache


def test_common_camelcase_secret_key_is_redacted():
    assert redact({"accessToken": "audit_dummy_secret"})["accessToken"] != "audit_dummy_secret"


def test_unrelated_user_column_cannot_suppress_an_outlier():
    frame = pl.DataFrame(
        {"v": [1.0] * 7 + [100.0], "g": ["a"] * 8, "__qavf_original_median": [100.0] * 8}
    )
    results = rule_results(
        "STA-001",
        {"s": frame},
        {
            "targets": [
                {"dataset": "s", "value": "v", "group_by": ["g"], "min_relative_deviation": 0.1}
            ]
        },
    )
    assert any(r.status.value == "FAIL" and r.affected_count == 1 for r in results)


def test_minimum_int_transition_does_not_pass():
    results = rule_results(
        "TMP-001",
        {"s": pl.DataFrame({"t": [0, 1], "v": [-(2**63), 0]})},
        {"series": [{"dataset": "s", "value": "v", "time_column": "t"}]},
    )
    assert any(r.status.value in {"FAIL", "ERROR"} for r in results)


def test_overflowing_reconciliation_tolerance_does_not_pass():
    results = rule_results(
        "REC-001",
        {"a": pl.DataFrame({"x": [1e308]}), "b": pl.DataFrame({"x": [-1e308]})},
        {
            "reconciliations": [
                {
                    "left": {"dataset": "a", "value": "x"},
                    "right": {"dataset": "b", "value": "x"},
                    "rtol": 1.9,
                }
            ]
        },
    )
    assert any(r.status.value in {"FAIL", "ERROR"} for r in results)


def test_owned_query_supports_eager_statistical_rules():
    report = validate(
        Query("SELECT i AS x FROM range(20) t(i)"),
        config={
            "rules": {"STA-001": {"params": {"targets": [{"dataset": "data", "value": "x"}]}}},
            "engine": {"run_unconfigured_rules": False, "parallel": False},
            "plugins": {"entry_points": False},
        },
    )
    assert all(r.status.value != "ERROR" for r in report.results)


def test_evidence_budget_counts_persisted_float_tags():
    class EvidenceRule(Rule):
        rule_id = "AUD-001"
        name = "Evidence"
        description = "Audit"
        category = "audit"

        def evaluate(self, ctx):
            return Finding.fail("test", affected=[{"values": [math.nan] * 1000}], affected_count=1)

    registry = RuleRegistry()
    registry.register(EvidenceRule)
    settings = QAVFSettings.model_validate(
        {"engine": {"max_evidence_bytes": 12000}, "plugins": {"entry_points": False}}
    )
    result = QARunner(settings, registry).run().results[0]
    encoded = json.dumps(
        result.model_dump(mode="json")["affected_records"], separators=(",", ":")
    ).encode()
    assert len(encoded) <= 12000
