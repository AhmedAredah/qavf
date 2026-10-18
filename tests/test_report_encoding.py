"""Incremental report encoding, explicit versions and scalar codec admission."""

from __future__ import annotations

import json
import math
import tracemalloc
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from qavf.audit_codec import encode_audit, iter_audit_json, normalize_evidence
from qavf.reports.html import render_html
from qavf.reports.models import QAReport, RuleResult, RunMetadata, SnapshotModel


def report(rows=0, metrics=None):
    now = datetime.now(UTC)
    return QAReport(
        metadata=RunMetadata(
            run_id="test", project="test", qavf_version="test", started_at=now, finished_at=now
        ),
        results=[
            RuleResult(
                rule_id="TEST",
                name="test",
                description="test",
                category="test",
                status="FAIL",
                severity="ERROR",
                message="failed",
                affected_records=[{"a": i, "b": i, "c": i} for i in range(rows)],
                metrics=metrics or {},
            )
        ],
    )


def test_report_stream_does_not_dump_whole_models(monkeypatch, tmp_path):
    value = report(50, {"number": Decimal("1.234"), "nonfinite": math.inf})
    expected = json.loads(value.model_dump_json())

    def forbidden(*args, **kwargs):
        raise AssertionError("streaming must not dump the full model graph")

    monkeypatch.setattr(SnapshotModel, "model_dump", forbidden)
    target = value.write_json(tmp_path / "report.json")
    assert json.loads(target.read_bytes()) == expected
    assert target.stat().st_size == value.json_size_bytes()


def test_streaming_incremental_allocation_does_not_scale_with_evidence_graph():
    value = report(10_000)
    tracemalloc.start()
    size = sum(map(len, value.iter_json_chunks()))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert size > 250_000
    # The former full model graph allocated approximately 3.8 MB here. This
    # deliberately generous allowance covers small interpreter/runtime variation.
    assert peak < 500_000


def test_html_encodes_metrics_only_and_emits_them_once(monkeypatch):
    value = report(10, {"unique_metric_marker": "unique_metric_value"})

    def forbidden(*args, **kwargs):
        raise AssertionError("metrics rendering must not serialize affected records")

    monkeypatch.setattr(RuleResult, "model_dump", forbidden)
    html = render_html(value)
    assert html.count("unique_metric_marker") == 1
    assert html.count("unique_metric_value") == 1


@pytest.mark.parametrize("version", [0, 3, 4, 999])
def test_unknown_report_versions_are_rejected(version):
    raw = report().model_dump(mode="json")
    raw["format_version"] = version
    with pytest.raises(ValueError, match="unsupported QA report version"):
        QAReport.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("kind", [np.longdouble, np.clongdouble])
def test_numpy_nonprogress_scalar_is_explicitly_rejected(kind):
    scalar = kind("1.5")
    if not isinstance(scalar.item(), np.generic):
        pytest.skip("this platform converts the dtype to a Python scalar")
    with pytest.raises(ValueError, match="unsupported evidence NumPy dtype"):
        normalize_evidence(scalar)
    with pytest.raises(ValueError, match="unsupported evidence NumPy dtype"):
        b"".join(iter_audit_json(scalar))


@pytest.mark.parametrize("sort_keys", [False, True])
def test_streaming_codec_matches_existing_canonical_tagged_format(sort_keys):
    payload = {
        "reserved": {"$qavf_mapping": [["literal", 1]]},
        "numbers": [Decimal("1.00000001"), math.inf, math.nan, np.int64(5)],
        "array": np.array([1, 2], dtype="datetime64[ns]"),
        "set": {3, 2, 1},
        "quoted": ('é"\\\n' * 20_000),
        "nested": {"z": 2, "a": 1},
    }
    expected = json.dumps(
        encode_audit(payload),
        sort_keys=sort_keys,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    assert b"".join(iter_audit_json(payload, sort_keys=sort_keys)) == expected
