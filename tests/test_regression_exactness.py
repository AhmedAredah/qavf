"""Public correctness, evidence integrity and bounded scalar comparison contracts."""

from __future__ import annotations

import importlib
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.runner import QARunner
from qavf.engine.severity import Severity, Status
from qavf.io import read_table, write_table
from qavf.redaction import redact_text
from qavf.regression import accept_baseline, compare_frames, verify_baseline
from qavf.regression.models import ComparisonResult
from qavf.reports.html import render_html, write_html
from qavf.reports.models import DatasetInfo, QAReport, RuleResult, RunMetadata
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

comparison = importlib.import_module("qavf.regression._comparison")
baselines = importlib.import_module("qavf.regression.baselines")


def frame(value, dtype=None):
    return pl.DataFrame({"k": [1], "v": pl.Series([value], dtype=dtype)})


def report(result, **metadata):
    now = datetime.now(UTC)
    return QAReport(
        metadata=RunMetadata(
            run_id="test",
            project="test",
            qavf_version="test",
            started_at=now,
            finished_at=now,
            **metadata,
        ),
        results=[result],
    )


def result(**kwargs):
    return RuleResult(
        rule_id="X",
        name="X",
        description="X",
        category="X",
        message="X",
        status=Status.FAIL,
        severity=Severity.WARNING,
        **kwargs,
    )


def run_baselines(data, specs, **params):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {"out": {}},
            "rules": {"REG-001": {"severity": "WARNING", "params": {"baselines": specs, **params}}},
        }
    )
    return QARunner(settings, discovered_rules(entry_points=False)).run(frames={"out": data})


def test_mixed_decimal_float_exact_and_float32_boundary():
    assert not compare_frames(
        frame(Decimal("0.1"), pl.Decimal(10, 1)), frame(0.1), "k", rtol=0, atol=0
    ).equal
    a, b = 1.0, 1.0000001192092896
    tolerance = abs(a - b) / b - 1e-15
    assert not compare_frames(
        frame(a, pl.Float32), frame(b, pl.Float32), ["k"], rtol=tolerance, atol=0
    ).equal


@pytest.mark.parametrize(
    "dtype,a,b",
    [
        (pl.List(pl.Datetime("ns")), [1001], [1002]),
        (pl.Struct({"t": pl.Datetime("ns")}), {"t": 1001}, {"t": 1002}),
        (pl.List(pl.Duration("ns")), [1001], [1002]),
    ],
)
def test_nested_temporal_precision(dtype, a, b):
    out = compare_frames(frame(a, dtype), frame(b, dtype), ["k"], rtol=0, atol=0)
    assert not out.equal and out.values_changed == 1 and out.differences[0].path


def test_wide_temporal_range_and_mixed_units():
    assert not compare_frames(frame(datetime(2500, 1, 1)), frame(datetime(2500, 1, 2)), ["k"]).equal
    assert compare_frames(
        frame(datetime(2500, 1, 1), pl.Datetime("ms")),
        frame(datetime(2500, 1, 1), pl.Datetime("us")),
        ["k"],
    ).equal
    assert compare_frames(
        frame(10**18, pl.Duration("ms")), frame(10**18, pl.Duration("ms")), ["k"]
    ).equal


@pytest.mark.parametrize("key", ["__qavf_b0", "__qavf_c0", "__qavf_brow", "__qavf_crow"])
def test_internal_names_are_ordinary_input_names(key):
    base = pl.DataFrame({key: [1], "v": [1.0]})
    assert compare_frames(base, base.with_columns(pl.col("v") + 1), [key]).values_changed == 1


@pytest.mark.parametrize("a,b", [(-1e308, 1e308), (1e-308, 1e308)])
def test_metric_overflow_does_not_prevent_failure(a, b):
    out = compare_frames(frame(a), frame(b), ["k"], rtol=0, atol=0)
    assert not out.equal and out.values_changed == 1
    assert math.isinf(out.max_abs_diff) or math.isinf(out.max_rel_diff)
    restored = ComparisonResult.model_validate_json(out.model_dump_json())
    assert restored.max_abs_diff == out.max_abs_diff
    if a == -1e308:
        assert out.max_rel_diff == 2.0


def test_scalar_comparison_never_traverses_python_cells(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scalar comparisons must stay out of Python cell traversal")

    monkeypatch.setattr(comparison, "_cell_change", forbidden)
    monkeypatch.setattr(comparison, "_BATCH_ROWS", 7)
    base = pl.DataFrame({"k": range(100), "v": [1.0] * 100, "s": ["a"] * 100})
    candidate = base.with_columns(pl.col("v") + 1, pl.lit("b").alias("s"))
    out = compare_frames(base, candidate, ["k"], max_differences=3)
    assert (out.rows_changed, out.values_changed, len(out.differences), out.truncated) == (
        100,
        200,
        3,
        True,
    )
    zero = compare_frames(base, candidate, ["k"], max_differences=0)
    assert zero.values_changed == 200 and not zero.differences


def test_manifest_source_spelling_and_unassessed_policy(tmp_path):
    path = tmp_path / "b.parquet"
    data = frame(1.0)
    data.write_parquet(path)
    spec = {"dataset": "out", "keys": ["k"], "source": {"type": "file", "path": str(path)}}
    unavailable = run_baselines(data, [spec], require_manifest=True)
    assert (
        unavailable.get("REG-001").status is Status.ERROR
        and unavailable.incomplete
        and unavailable.exit_code() == 1
    )
    accept_baseline(data, path, ["k"], replace=True)
    assert run_baselines(data, [spec], require_manifest=True).get("REG-001").status is Status.PASS
    data.with_columns(pl.col("v") + 1).write_parquet(path)
    assert run_baselines(data, [spec], require_manifest=True).incomplete


def test_multiple_baselines_keep_distinct_identities(tmp_path):
    one, two = tmp_path / "one.parquet", tmp_path / "two.parquet"
    frame(1.0).write_parquet(one)
    frame(2.0).write_parquet(two)
    output = run_baselines(
        frame(1.0), [{"dataset": "out", "keys": ["k"], "path": str(path)} for path in (one, two)]
    )
    checks = output.results_for("REG-001")
    assert len(checks) == 2 and checks[0].check != checks[1].check
    assert checks[0].status is Status.PASS and checks[1].status is Status.FAIL
    assert checks[1].affected_records[0]["comparison"] == checks[1].check


def test_failed_pair_commit_restores_previous_accepted_baseline(tmp_path, monkeypatch):
    path = tmp_path / "baseline.parquet"
    accept_baseline(frame(1.0), path, ["k"])
    before = path.read_bytes()
    real = baselines.os.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        if str(source).endswith("manifest.after") and not failed:
            failed = True
            raise OSError("injected manifest commit failure")
        return real(source, target)

    monkeypatch.setattr(baselines.os, "replace", fail_once)
    with pytest.raises(OSError):
        accept_baseline(frame(2.0), path, ["k"], replace=True)
    assert path.read_bytes() == before and verify_baseline(path) == []
    assert not baselines._transaction_path(path).exists()


def test_concurrent_first_acceptance_has_one_winner(tmp_path):
    path = tmp_path / "baseline.parquet"

    def attempt(value):
        try:
            accept_baseline(frame(value), path, ["k"])
            return "accepted"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(attempt, [1.0, 2.0]))
    assert sorted(outcomes) == ["accepted", "exists"] and verify_baseline(path) == []


def test_reader_recovers_an_interrupted_two_file_commit(tmp_path):
    path = tmp_path / "baseline.parquet"
    accept_baseline(frame(1.0), path, ["k"])
    original = path.read_bytes()
    transaction = baselines._transaction_path(path)
    transaction.mkdir()
    shutil.copy2(path, transaction / "data.before")
    shutil.copy2(baselines.manifest_path(path), transaction / "manifest.before")
    (transaction / "state.json").write_text(json.dumps({"old": [True, True], "committed": False}))
    frame(2.0).write_parquet(path)
    assert verify_baseline(path) == []
    assert path.read_bytes() == original and not transaction.exists()


@pytest.mark.parametrize(
    "value", [{"$float": []}, {"$float": {}}, {"$float": "NaN"}, {"$qavf_mapping": [["x", 2]]}]
)
def test_evidence_codec_escapes_ordinary_tag_shaped_data(value):
    original = result(affected_records=[{"v": value, "number": math.nan}])
    assert original.affected_records[0]["v"] == value
    restored = RuleResult.model_validate_json(original.model_dump_json())
    assert restored.affected_records[0]["v"] == value
    assert math.isnan(restored.affected_records[0]["number"])


def test_report_is_a_detached_immutable_snapshot():
    records = [{"v": [1]}]
    out = report(result(affected_records=records))
    records[0]["v"].append(2)
    assert out.results[0].affected_records[0]["v"] == [1]
    with pytest.raises(TypeError):
        out.results[0].affected_records[0]["v"].append(3)
    with pytest.raises(TypeError):
        out.results.clear()


def test_html_json_acceptance_and_provenance_agree(tmp_path):
    out = report(
        result(metrics={"x": math.inf}),
        fail_on=Severity.WARNING,
        datasets=[
            DatasetInfo(name="lazy", rows=None, columns=["v"], notes={"selection": "verified"})
        ],
        resource_usage={"test": 123},
        query_plans=[{"kind": "native"}],
    )
    assert out.verdict() == "failed"
    html = render_html(out)
    assert "QA failed" in html and "verified" in html and "not materialized" in html
    target = write_html(out, tmp_path / "report.html")
    assert "QA failed" in target.read_text()
    json_path = out.write_json(tmp_path / "report.json")
    assert json_path.stat().st_size == out.json_size_bytes()
    assert json_path.read_bytes() == b"".join(out.iter_json_chunks())
    restored = QAReport.model_validate_json(json_path.read_text())
    assert restored.verdict() == "failed" and math.isinf(restored.results[0].metrics["x"])


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".parquet", ".json", ".jsonl.gz"])
def test_streaming_writes_round_trip(suffix, tmp_path):
    data = pl.DataFrame({"k": [1, 2], "v": [1.0, 2.0]})
    path = write_table(data, tmp_path / ("data" + suffix))
    assert compare_frames(data, read_table(path), ["k"]).equal


@pytest.mark.parametrize(
    "text",
    [
        'password="long secret words" host=db',
        "{'password': 'long secret words'}",
        "Authorization: Bearer secret123",
        "https://secret123@host/a.csv",
    ],
)
def test_redaction_covers_error_and_connection_representations(text):
    redacted = redact_text(text)
    assert "long secret words" not in redacted and "secret123" not in redacted


def test_invalid_schema_policy_is_not_silently_downgraded():
    with pytest.raises(ValueError, match="schema policy"):
        compare_frames(frame(1), frame(1), ["k"], schema="strcit")


def test_table_adapters_share_the_comparison_contract():
    original = frame(1.0)
    for adapted in (original.to_pandas(), original.to_arrow(), original.lazy()):
        assert compare_frames(original, adapted, "k").equal
    reader = original.to_arrow().to_reader()
    with pytest.raises(ValueError, match="one-pass"):
        compare_frames(original, reader, "k")
    # Rejection must not consume a caller-owned stream.
    assert reader.read_next_batch().num_rows == 1
