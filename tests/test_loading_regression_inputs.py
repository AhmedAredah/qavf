"""Regression input policies, projection, and complete accepted-baseline reads."""

from __future__ import annotations

import hashlib

import polars as pl
import pytest

from qavf import Dataset, File, ResourceLimits
from qavf.engine.context import DataContext
from qavf.engine.resources import ResourceLimitError
from qavf.engine.sql_read import SqlReadSession
from qavf.regression import accept_baseline, assert_outputs_match, compare_frames, compare_outputs
from qavf.regression.baselines import read_verified_baseline
from qavf.rules.regression_checks import BaselineComparisonRule
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceError
from qavf.sources.file import FileSource


def pair(tmp_path, *, nested=False):
    base = pl.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    candidate = base.with_columns((pl.col("value") + 1).alias("value"))
    if nested:
        base = base.with_columns(pl.lit([1, 2]).alias("nested"))
        candidate = candidate.with_columns(pl.lit([1, 2]).alias("nested"))
    left, right = tmp_path / "baseline.parquet", tmp_path / "candidate.parquet"
    base.write_parquet(left)
    candidate.write_parquet(right)
    return base, candidate, left, right


@pytest.mark.parametrize("cap", [0, 1, 2])
@pytest.mark.parametrize("nested", [False, True])
def test_public_resource_evidence_caps_preserve_complete_counts(tmp_path, cap, nested):
    _, _, left, right = pair(tmp_path, nested=nested)
    result = compare_outputs(
        File(left),
        File(right),
        "id",
        rtol=0,
        atol=0,
        resources=ResourceLimits(evidence_rows=cap, native_threads=1),
    )
    assert result.rows_changed == result.values_changed == 3
    assert len(result.differences) == cap
    assert result.truncated
    assert result.execution == "native_stream"


@pytest.mark.parametrize("cap", [0, 1])
def test_public_resource_byte_caps_cannot_hide_invalid_explicit_bounds(tmp_path, cap):
    _, _, left, right = pair(tmp_path)
    for argument in ("max_differences", "max_evidence_bytes"):
        with pytest.raises(ValueError, match=argument):
            compare_outputs(
                left,
                right,
                "id",
                **{argument: True},
                resources=ResourceLimits(evidence_rows=cap, evidence_bytes=cap),
            )
    result = compare_outputs(
        left,
        right,
        "id",
        resources=ResourceLimits(evidence_bytes=cap, native_threads=1),
    )
    assert result.values_changed == 3 and result.differences == [] and result.truncated


@pytest.mark.parametrize("function", [compare_outputs, assert_outputs_match])
@pytest.mark.parametrize("resources", [{}, "1GiB", True])
def test_public_resource_argument_rejects_untyped_values(function, resources):
    frame = pl.DataFrame({"id": [1]})
    with pytest.raises(TypeError, match="ResourceLimits"):
        function(frame, frame, "id", resources=resources)


def test_assertion_uses_resource_caps(tmp_path):
    _, _, left, right = pair(tmp_path)
    with pytest.raises(AssertionError, match="and 3 more"):
        assert_outputs_match(
            left,
            right,
            "id",
            resources=ResourceLimits(evidence_rows=0, native_threads=1),
        )


def test_native_resource_failure_cleans_owned_storage_and_never_publishes_result(tmp_path):
    _, _, left, right = pair(tmp_path)
    storage = tmp_path / "temporary"
    with pytest.raises(ResourceLimitError, match=r"memory_budget_bytes is too small.*native byte"):
        compare_outputs(
            left,
            right,
            "id",
            resources=ResourceLimits(
                target_memory=1, max_spill=0, temp_dir=storage, native_threads=1
            ),
        )
    assert list(storage.iterdir()) == []


def test_lazy_projection_checks_materialized_bytes_without_collecting_ignored_payload(tmp_path):
    base = pl.DataFrame({"id": range(1000), "value": range(1000), "payload": ["x" * 4096] * 1000})
    changed = base.with_columns((pl.col("value") + 1).alias("value"))
    resources = ResourceLimits(
        target_memory="256KiB", cache_memory=0, max_spill="64MiB", temp_dir=tmp_path
    )
    result = compare_outputs(
        base.lazy(), changed.lazy(), "id", ignore_columns=["payload"], resources=resources
    )
    assert result.execution == "typed_stream" and result.values_changed == 1000
    with pytest.raises(ResourceLimitError, match="target"):
        compare_outputs(base.lazy(), changed.lazy(), "id", resources=resources)
    assert list(tmp_path.iterdir()) == []


def test_typed_projected_fallback_excludes_ignored_data_and_preserves_exact_values(
    tmp_path, monkeypatch
):
    base = pl.DataFrame(
        {
            "id": [1, 2],
            "value": pl.Series([[1, 2], [3]], dtype=pl.List(pl.Int128)),
            "payload": ["x" * 100_000] * 2,
        }
    )
    changed = base.with_columns(pl.Series("value", [[1, 3], [3]], dtype=pl.List(pl.Int128)))
    left, right = tmp_path / "base.parquet", tmp_path / "candidate.parquet"
    base.write_parquet(left)
    changed.write_parquet(right)
    requested = []
    from qavf.engine import observation

    original = observation.observe

    def project(self, name, columns):
        requested.append(tuple(columns))
        return original(self, name, columns)

    def forbidden(*args, **kwargs):
        raise AssertionError("full file load must not be used for typed projection")

    monkeypatch.setattr(observation, "observe", project)
    monkeypatch.setattr(FileSource, "load", forbidden)
    result = compare_outputs(left, right, "id", ignore_columns=["payload"])
    expected = compare_frames(base, changed, "id", ignore_columns=["payload"], name=right.name)
    assert result.model_dump(exclude={"execution", "fallback_reason"}) == expected.model_dump(
        exclude={"execution", "fallback_reason"}
    )
    assert result.execution == "typed_stream" and result.fallback_reason
    assert requested == [("id", "value"), ("id", "value")]


def test_dataset_load_budget_survives_comparison_normalization(tmp_path):
    _, _, left, right = pair(tmp_path)
    with pytest.raises((SourceError, ValueError), match="row_budget"):
        compare_outputs(Dataset(left, load_policy={"row_budget": 2}), right, "id")


def configured_context(candidate):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "native_threads": 1},
            "datasets": {"output": {"path": str(candidate)}},
        }
    )
    return DataContext.from_settings(settings, threads=1, execution_workers=1)


def baseline_rule(path, *, required=False):
    return BaselineComparisonRule(
        {
            "require_manifest": required,
            "baselines": [
                {"dataset": "output", "path": str(path), "keys": ["id"]},
            ],
        }
    )


def test_reg001_native_path_uses_parent_context_and_verifies_manifest(tmp_path, monkeypatch):
    base, _, left, right = pair(tmp_path)
    accept_baseline(base, left, "id", replace=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("REG-001 must not fully load either native input")

    monkeypatch.setattr(FileSource, "load", forbidden)
    with configured_context(right) as context:
        monkeypatch.setattr(context, "get", forbidden)
        finding = baseline_rule(left, required=True).evaluate(context)[0]
        metrics = finding.metrics["comparisons"]["output"]
        assert not finding.passed and finding.assessed and finding.affected_count == 3
        assert metrics["execution"] == "native_stream"
        assert (
            finding.metrics["baselines"]["output"]["sha256"]
            == hashlib.sha256(left.read_bytes()).hexdigest()
        )
        assert not context.resource_usage()["eager_fallbacks"]


def test_manifest_verification_covers_complete_native_batch_consumption(tmp_path, monkeypatch):
    base, _, left, right = pair(tmp_path)
    accept_baseline(base, left, "id", replace=True)
    original = SqlReadSession.batches

    def mutate_after_read(self, *args, **kwargs):
        yield from original(self, *args, **kwargs)
        with left.open("ab") as handle:
            handle.write(b"changed-after-last-batch")

    monkeypatch.setattr(SqlReadSession, "batches", mutate_after_read)
    with configured_context(right) as context:
        finding = baseline_rule(left, required=True).evaluate(context)[0]
    assert not finding.assessed and not finding.passed
    assert "changed" in finding.message.lower()


def test_verified_baseline_callback_can_return_completed_comparison(tmp_path):
    base, _, left, right = pair(tmp_path)
    accept_baseline(base, left, "id", replace=True)
    result, manifest, identity = read_verified_baseline(
        left,
        ["id"],
        True,
        lambda: compare_outputs(left, right, "id"),
    )
    assert result.values_changed == 3 and result.execution == "native_stream"
    assert manifest is not None and identity["sha256"] == manifest.sha256


def test_native_retry_discards_partial_comparison_counts_and_evidence(tmp_path, monkeypatch):
    from qavf.regression import _comparison

    _, _, left, right = pair(tmp_path)
    original = _comparison.ComparisonAccumulator.matched
    seen = []

    def interrupt_first(self, before, after):
        original(self, before, after)
        seen.append(self)
        if len(seen) == 1:
            raise SourceError("transient after comparison prefix", retryable=True)

    monkeypatch.setattr(_comparison, "_BATCH_ROWS", 2)
    monkeypatch.setattr(_comparison.ComparisonAccumulator, "matched", interrupt_first)
    policy = {"retries": 1, "retry_delay_s": 0}
    result = compare_outputs(
        Dataset(left, load_policy=policy),
        Dataset(right, load_policy=policy),
        "id",
        resources=ResourceLimits(native_threads=1),
    )
    assert result.execution == "native_stream" and result.values_changed == result.rows_changed == 3
    assert [item.key for item in result.differences] == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert seen[0] is not seen[1] and seen[1] is seen[2]


@pytest.mark.parametrize("name", ["b", "c", "projected", "grouped"])
def test_regression_internal_sql_names_do_not_shadow_user_dataset(tmp_path, name):
    from qavf.regression._inputs import compare_context

    _, _, left, right = pair(tmp_path)
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"parallel": False},
            "datasets": {name: {"path": str(right)}},
        }
    )
    with DataContext.from_settings(settings, threads=1) as context:
        result, _ = compare_context(
            context,
            name,
            "id",
            source={"type": "file", "path": str(left)},
        )
    assert result.execution == "native_stream" and result.values_changed == 3


@pytest.mark.parametrize("query_side", ["baseline", "candidate", "both"])
def test_nonrepeatable_sources_use_one_projected_comparison(tmp_path, monkeypatch, query_side):
    from qavf.regression import _inputs

    base, _, left, right = pair(tmp_path)
    base.write_parquet(right)
    query = {
        "type": "query",
        "sql": "SELECT i AS id, CAST(i AS DOUBLE) + random() * 0 AS value FROM range(1, 4) t(i)",
    }
    baseline = query if query_side in {"baseline", "both"} else {"type": "file", "path": str(left)}
    candidate = query if query_side in {"candidate", "both"} else {"path": str(right)}
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"parallel": False},
            "datasets": {"output": {"source": candidate} if "type" in candidate else candidate},
        }
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("a volatile source must never reach multi-query comparison")

    monkeypatch.setattr(_inputs, "compare_native", forbidden)
    with DataContext.from_settings(settings, threads=1) as context:
        result, identity = _inputs.compare_context(
            context,
            "output",
            "id",
            source=baseline,
            fingerprint=True,
        )
    assert result.equal and result.baseline_rows == result.candidate_rows == 3
    assert result.execution == "typed_stream"
    assert "repeatable" in result.fallback_reason
    if query_side == "candidate":
        assert identity is not None and "sha256" in identity
    else:
        assert identity is None  # Queries do not claim a content-addressed identity.
