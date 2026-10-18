"""Audit boundaries: complete redaction, exact acceptance, schema and canonical evidence."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from qavf.audit_codec import canonical_digest, decode_audit, encode_audit, evidence_json_size
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.redaction import redact_text
from qavf.regression import accept_baseline, compare_frames, read_manifest, verify_baseline
from qavf.regression.baselines import manifest_path
from qavf.reports.html import render_html
from qavf.reports.models import (
    DatasetInfo,
    QAReport,
    Recommendation,
    RuleInfo,
    RuleResult,
    RunMetadata,
)
from qavf.settings import QAVFSettings


def result(**changes):
    return RuleResult(
        **{
            "rule_id": "AUDIT",
            "name": "audit",
            "description": "test",
            "category": "test",
            "status": "PASS",
            "severity": "ERROR",
            "message": "ok",
            **changes,
        }
    )


def report(item=None, **metadata):
    now = datetime.now(UTC)
    return QAReport(
        metadata=RunMetadata(
            run_id="audit",
            project="audit",
            qavf_version="test",
            started_at=now,
            finished_at=now,
            **metadata,
        ),
        results=[item or result()],
    )


@pytest.mark.parametrize(
    "text",
    [
        "credentials={'a': {'x': 'CANARY_A'}, 'b': 'CANARY_B'} other=ok",
        'password=["CANARY_A", "CANARY_B"] other=ok',
        "credentials=({'a': ['CANARY_A']}, 'CANARY_B')",
        "credentials={a:{x:CANARY_A},b:CANARY_B}",
        "credentials={'a': 'CANARY_A}bracket', 'b': ['CANARY_B']}",
        "credentials={'a': {'x': 'CANARY_A'}, 'b': 'CANARY_B'",
        "password='CANARY_A''CANARY_B' host=db",
        "ATTACH 'password=''CANARY_A CANARY_B'''",
        "Pwd={CANARY_A}}CANARY_B};Server=host",
        "Pwd={CANARY_A; CANARY_B};Server=host",
        "Authorization = Bearer CANARY_B",
        "Authorization: Basic CANARY_B",
        "password=CANARY_A\\ CANARY_B",
    ],
)
def test_complete_compound_secret_is_masked(text):
    safe = redact_text(text)
    assert "CANARY_A" not in safe and "CANARY_B" not in safe


def test_error_secrets_never_cross_report_or_log_boundary(caplog, tmp_path):
    class Secrets(Rule):
        rule_id = "AUDIT"
        name = "audit"
        description = "test"
        category = "test"

        def evaluate(self, ctx):
            raise RuntimeError("credentials={'a': {'x': 'CANARY_A'}, 'b': 'CANARY_B'}")

    registry = RuleRegistry()
    registry.register(Secrets)
    with caplog.at_level(logging.ERROR, logger="qavf"):
        out = QARunner(QAVFSettings(plugins={"entry_points": False}), registry).run()
    target = out.write_json(tmp_path / "report.json")
    for serialized in (target.read_text(), out.model_dump_json(), render_html(out), caplog.text):
        assert "CANARY_A" not in serialized and "CANARY_B" not in serialized
    assert out.incomplete


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("suffix", [".json", ".jsonl.gz"])
def test_lossy_acceptance_rejects_before_commit_and_preserves_old_pair(value, suffix, tmp_path):
    path = tmp_path / ("accepted" + suffix)
    accept_baseline(pl.DataFrame({"k": [1, 2], "v": [1.0, 2.0]}), path, ["k"])
    before = path.read_bytes(), manifest_path(path).read_bytes()
    with pytest.raises(ValueError, match="does not preserve"):
        accept_baseline(pl.DataFrame({"k": [1, 2], "v": [value, 2.0]}), path, ["k"], replace=True)
    assert (path.read_bytes(), manifest_path(path).read_bytes()) == before
    assert verify_baseline(path) == []
    assert not path.with_name(f".{path.name}.transaction").exists()


def test_typed_baseline_preserves_nonfinite_and_manifest_policy(tmp_path):
    from qavf.io import read_table

    original = pl.DataFrame({"k": [1, 2, 3, 4], "v": [math.nan, math.inf, -math.inf, None]})
    path = tmp_path / "accepted.parquet"
    accept_baseline(original, path, ["k"], comparison={"exact": Decimal("0"), "marker": math.inf})
    assert compare_frames(original, read_table(path), "k", rtol=0, atol=0).equal
    manifest = read_manifest(path)
    assert manifest.comparison == {"exact": Decimal("0"), "marker": math.inf}


@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize(
    "left,right",
    [
        (pl.Int32, pl.Int64),
        (pl.Datetime("ns"), pl.Datetime("us")),
    ],
)
def test_strict_schema_records_key_changes_before_alignment(rows, left, right):
    a = pl.DataFrame({"k": pl.Series([0, 0][:rows], dtype=left)})
    b = pl.DataFrame({"k": pl.Series([0, 0][:rows], dtype=right)})
    if rows:
        # Physical values representing distinct exact keys under both units.
        a = pl.DataFrame({"k": pl.Series([0, 1000], dtype=left)})
        b = pl.DataFrame(
            {"k": pl.Series([0, 1] if left == pl.Datetime("ns") else [0, 1000], dtype=right)}
        )
    assert compare_frames(a, b, "k").equal
    strict = compare_frames(a, b, "k", schema="strict")
    assert not strict.equal and strict.dtypes_changed == ["k"]
    assert strict.rows_added == strict.rows_removed == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"number": math.inf, "missing": None, "nan": math.nan},
        {
            "date": date(2026, 1, 1),
            "datetime": datetime(2026, 1, 1, tzinfo=UTC),
            "time": time(1, 2, 3, fold=1),
            "duration": timedelta(days=-2, microseconds=7),
        },
        {
            "decimal": Decimal("1.00000000000000000001"),
            "binary": b"\xff\x00",
            "path": Path("data/x"),
        },
        {"ordinary": [{"$float": "NaN"}, {"$qavf_value": ["decimal", "1"]}, {"$qavf_mapping": []}]},
    ],
)
def test_every_arbitrary_payload_uses_one_codec_and_round_trip(payload, tmp_path):
    out = report(
        result(metrics=payload, affected_records=[payload]),
        extensions={"plugin": payload},
        config=payload,
        resource_usage=payload,
        query_plans=[payload],
        rules=[RuleInfo(rule_id="AUDIT", severity="ERROR", params=payload, module="test")],
        datasets=[DatasetInfo(name="x", rows=1, columns=["v"], notes=payload, fingerprint=payload)],
    )
    target = out.write_json(tmp_path / "report.json")
    assert target.stat().st_size == out.json_size_bytes()
    assert json.loads(target.read_text()) == json.loads(out.model_dump_json())
    restored = QAReport.model_validate_json(target.read_text())
    observed = [
        restored.results[0].metrics,
        restored.results[0].affected_records[0],
        restored.metadata.extensions["plugin"],
        restored.metadata.config,
        restored.metadata.resource_usage,
        restored.metadata.query_plans[0],
        restored.metadata.rules[0].params,
        restored.metadata.datasets[0].notes,
        restored.metadata.datasets[0].fingerprint,
    ]
    for value in observed:
        assert canonical_digest(value) == canonical_digest(payload)
    serialized = out.results[0].model_dump(mode="json")["metrics"]
    assert evidence_json_size(payload) == len(
        json.dumps(serialized, ensure_ascii=False, separators=(",", ":")).encode()
    )


def test_mapping_keys_do_not_collide_during_either_json_path():
    values = {
        True: "boolean bucket",
        "true": "string bucket",
        None: "missing bucket",
        "null": "text",
    }
    out = report(result(metrics={"buckets": values}))
    expected = {
        "True": "boolean bucket",
        "true": "string bucket",
        "None": "missing bucket",
        "null": "text",
    }
    assert out.results[0].metrics["buckets"] == expected
    assert json.loads(out.model_dump_json())["results"][0]["metrics"]["buckets"] == expected
    assert (
        json.loads(b"".join(out.iter_json_chunks()))["results"][0]["metrics"]["buckets"] == expected
    )
    with pytest.raises(ValueError, match="colliding"):
        result(metrics={"x": {1: "integer", "1": "text"}})


def test_effective_config_digest_distinguishes_nonfinite_and_null():
    class Pass(Rule):
        rule_id = "AUDIT"
        name = "audit"
        description = "test"
        category = "test"

        def evaluate(self, ctx):
            return Finding.ok("ok")

    registry = RuleRegistry()
    registry.register(Pass)
    digests = []
    for value in (math.inf, math.nan, None):
        settings = QAVFSettings(
            plugins={"entry_points": False}, extensions={"plugin": {"value": value}}
        )
        out = QARunner(settings, registry).run()
        digests.append(out.metadata.config_digest)
        decoded = QAReport.model_validate_json(out.model_dump_json())
        assert canonical_digest(decoded.metadata.extensions) == canonical_digest(
            settings.extensions
        )
    assert len(set(digests)) == 3


def test_recommendations_construct_once_per_rule(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recommendations must not copy/freeze each growing prefix")

    monkeypatch.setattr(Recommendation, "model_copy", forbidden)
    out = report()
    out = out.model_copy(
        update={
            "results": [
                result(status="FAIL", check=f"check{i}", recommendation="fix") for i in range(2000)
            ]
        }
    )
    advice = out.recommendations()
    assert len(advice) == 1 and advice[0].checks == [f"check{i}" for i in range(2000)]
    with pytest.raises(TypeError):
        advice[0].checks.append("mutable")


def test_rich_tags_are_data_not_executable_and_reserved_shapes_escape():
    raw = {"$qavf_value": ["not-a-real-type", "user data"]}
    assert decode_audit(encode_audit(raw)) == raw
    with pytest.raises(ValueError, match="unknown or invalid"):
        decode_audit(raw)


def test_temporal_codec_preserves_named_zone_and_fold():
    from zoneinfo import ZoneInfo

    payload = {
        "instant": datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1),
        "clock": time(1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1),
    }
    restored = decode_audit(encode_audit(payload))
    for name in payload:
        assert restored[name].tzinfo.key == "America/New_York"
        assert restored[name].fold == 1
        assert restored[name] == payload[name]
    assert canonical_digest(payload) == canonical_digest(restored)


def test_python_mode_provenance_redacts_path_and_utf8_bytes():
    from qavf.redaction import redact

    out = redact({"path": Path("password=CANARY_A"), "bytes": b"password=CANARY_B"})
    assert isinstance(out["path"], Path) and isinstance(out["bytes"], bytes)
    assert out["bytes"] == b"password=***"
    assert str(out["path"]) == "password=***"
    encoded = json.dumps(encode_audit(out))
    assert "CANARY_A" not in encoded and "CANARY_B" not in encoded


def test_first_lossy_acceptance_creates_no_accepted_pair(tmp_path):
    path = tmp_path / "new.json"
    with pytest.raises(ValueError, match="does not preserve"):
        accept_baseline(pl.DataFrame({"k": [1, 2], "v": [math.nan, 2.0]}), path, "k")
    assert not path.exists() and not manifest_path(path).exists()


def test_numpy_nanosecond_temporals_keep_type_and_unit():
    import numpy as np

    values = {
        "timestamp": np.datetime64(1, "ns"),
        "duration": np.timedelta64(1, "ns"),
        "array": np.array([1, 2], dtype="datetime64[ns]"),
    }
    out = result(metrics=values)
    restored = RuleResult.model_validate_json(out.model_dump_json())
    assert restored.metrics["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert restored.metrics["duration"].dtype == np.dtype("timedelta64[ns]")
    assert restored.metrics["timestamp"].astype("int64") == 1
    assert [v.astype("int64") for v in restored.metrics["array"]] == [1, 2]
    assert canonical_digest(restored.metrics) == canonical_digest(values)
    assert canonical_digest(values["timestamp"]) != canonical_digest(1)
