"""Bounded evidence sampling and reproducible run provenance.

Bounded evidence memory with exact counts and reproducible provenance: report
format version, execution-plan digest, baseline identity and fingerprint strength.
"""

from __future__ import annotations

import json
import re
import tracemalloc

import numpy as np
import polars as pl
import pytest

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.reports.models import REPORT_FORMAT_VERSION, QAReport
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)
ROWS = 400_000
LIMIT = 7


def _run(frames, **config) -> QAReport:
    engine = {"parallel": False, "run_unconfigured_rules": False, "max_affected_records": LIMIT}
    engine |= config.pop("engine", {})
    settings = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "engine": engine, **config}
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


@pytest.fixture(scope="module")
def failing() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({"g": rng.integers(0, 40, ROWS), "v": -rng.random(ROWS) - 0.001})


CASES = {
    "INP-003": {
        "rules": {"INP-003": {}},
        "datasets": {"t": {"schema": {"columns": {"v": {"ge": 0}}}}},
    },
    "INP-006": {"rules": {"INP-006": {}}, "datasets": {"t": {"expressions": [{"expr": "v >= 0"}]}}},
    "INP-007": {
        "rules": {"INP-007": {}},
        "datasets": {"t": {}},
        "assertions": [{"name": "negative", "sql": "SELECT * FROM t WHERE v < 0"}],
    },
    "STA-001": {
        "datasets": {"t": {}},
        "rules": {
            "STA-001": {
                "params": {
                    "targets": [{"dataset": "t", "value": "v"}],
                    "method": "zscore",
                    "threshold": 0.5,
                }
            }
        },
    },
}


# =========================================================================
# Evidence memory is bounded; counts stay exact
# =========================================================================
@pytest.mark.parametrize("rule_id", list(CASES))
def test_failure_heavy_runs_keep_only_the_sample(failing, rule_id) -> None:
    config = json.loads(json.dumps(CASES[rule_id]))
    tracemalloc.start()
    try:
        report = _run({"t": failing}, **config)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    [result] = [r for r in report.results if r.failed]
    assert result.status is Status.FAIL
    assert len(result.affected_records) == LIMIT
    if rule_id != "STA-001":
        assert result.affected_count == ROWS  # every row fails
    else:
        assert result.affected_count > LIMIT * 100
    # No per-row Python objects: far below one small object per failing row.
    assert peak < 30_000_000, f"{rule_id}: peak Python allocation {peak / 1e6:.0f} MB"


def test_sampled_rows_are_the_first_rows_and_counts_are_exact() -> None:
    frame = pl.DataFrame({"v": [1.0, -1.0, 2.0, -2.0, -3.0] * 10})
    report = _run(
        {"t": frame},
        rules={"INP-003": {}},
        datasets={"t": {"schema": {"columns": {"v": {"ge": 0}}}}},
    )
    [result] = [r for r in report.results if r.failed]
    assert result.affected_count == 30 and result.affected_rows == 30
    assert [record["_row"] for record in result.affected_records] == [1, 3, 4, 6, 8, 9, 11]


def test_sql_assertion_samples_are_ordered_and_count_everything() -> None:
    frame = pl.DataFrame({"k": [5, 3, 9, 1, 7, 2, 8, 4, 6, 0]})
    report = _run(
        {"t": frame},
        rules={"INP-007": {}},
        datasets={"t": {}},
        assertions=[{"name": "all", "sql": "SELECT k FROM t;"}],
    )
    result = report.get("INP-007")
    assert result.affected_count == 10
    assert [r["k"] for r in result.affected_records] == list(range(LIMIT))


def test_unwrappable_sql_falls_back_and_still_counts() -> None:
    frame = pl.DataFrame({"k": [1, 2, 3]})
    report = _run(
        {"t": frame},
        rules={"INP-007": {}},
        datasets={"t": {}},
        assertions=[{"name": "pivot", "sql": "PIVOT t ON k"}],
    )
    result = report.get("INP-007")
    assert result.status is Status.FAIL and result.affected_count == 1


def test_multiple_violations_per_row_are_joined_in_limit_order() -> None:
    frame = pl.DataFrame({"v": [-1.0, 50.0, -100.0]})
    report = _run(
        {"t": frame},
        rules={"INP-003": {}},
        datasets={"t": {"schema": {"columns": {"v": {"ge": 0, "le": 10, "notin": [-100]}}}}},
    )
    [result] = [r for r in report.results if r.failed]
    reasons = {record["_row"]: record["violates"] for record in result.affected_records}
    assert reasons == {0: "ge 0", 1: "le 10", 2: "ge 0; notin [-100]"}


# =========================================================================
# Reports identify their format, plan, baselines and evidence strength
# =========================================================================
def test_reports_carry_a_format_version() -> None:
    report = _run({"t": pl.DataFrame({"a": [1]})}, datasets={"t": {}}, rules={"INP-001": {}})
    assert json.loads(report.model_dump_json())["format_version"] == REPORT_FORMAT_VERSION


def test_plan_digest_is_deterministic_and_sensitive() -> None:
    def digest(limit: float) -> str:
        return _run(
            {"t": pl.DataFrame({"a": [1]})},
            datasets={"t": {}},
            rules={"STA-001": {"params": {"threshold": limit}}},
        ).metadata.plan_digest

    assert re.fullmatch(r"[0-9a-f]{64}", digest(2.0))
    assert digest(2.0) == digest(2.0) != digest(3.0)


def test_baseline_identity_is_recorded_and_changes_with_content(tmp_path) -> None:
    path = tmp_path / "baseline.csv"
    frame = pl.DataFrame({"k": [1, 2], "v": [1.0, 2.0]})

    def identity() -> dict:
        report = _run(
            {"out": frame},
            datasets={"out": {}},
            rules={
                "REG-001": {
                    "params": {"baselines": [{"dataset": "out", "path": str(path), "keys": ["k"]}]}
                }
            },
        )
        return report.get("REG-001").metrics["baselines"]["out"]

    frame.write_csv(path)
    first = identity()
    assert first["identity"] == "content" and re.fullmatch(r"[0-9a-f]{64}", first["sha256"])
    frame.with_columns(pl.col("v") * 10).write_csv(path)
    assert identity()["sha256"] != first["sha256"]


def test_dataset_fingerprints_state_their_strength(tmp_path) -> None:
    path = tmp_path / "d.csv"
    pl.DataFrame({"a": [1]}).write_csv(path)
    for hash_inputs, expected in ((False, "metadata"), (True, "content")):
        report = _run(
            {},
            datasets={"d": {"path": str(path)}},
            rules={"INP-001": {}},
            report={"hash_inputs": hash_inputs},
        )
        [info] = report.metadata.datasets
        assert info.fingerprint["identity"] == expected
