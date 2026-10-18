"""Reconciliation alias safety across source backends."""

import polars as pl
import pytest

from qavf.engine.severity import Status
from qavf.testing import rule_results


@pytest.mark.parametrize("kind", ["eager", "lazy", "file"])
@pytest.mark.parametrize(
    "key",
    [
        "__qavf_left_total",
        "__qavf_right_total",
        "__qavf_left_present",
        "__qavf_right_present",
        "__qavf_problem",
    ],
)
def test_source_key_names_do_not_collide_with_internal_aliases(kind, key, tmp_path):
    frames = {"a": pl.DataFrame({key: [1], "x": [5]}), "b": pl.DataFrame({"k": [1], "x": [5]})}
    datasets = {"a": {}, "b": {}}
    if kind == "lazy":
        frames = {k: v.lazy() for k, v in frames.items()}
    elif kind == "file":
        for name, frame in frames.items():
            path = tmp_path / f"{name}.parquet"
            frame.write_parquet(path)
            datasets[name] = {"path": str(path)}
        frames = {}
    results = rule_results(
        "REC-001",
        frames,
        {
            "reconciliations": [
                {
                    "left": {"dataset": "a", "value": "x", "keys": [key]},
                    "right": {"dataset": "b", "value": "x"},
                    "on": ["k"],
                }
            ]
        },
        datasets=datasets,
    )
    assert all(r.status is Status.PASS for r in results), [r.message for r in results]


@pytest.mark.parametrize("evidence_limit", [0, 2])
def test_batch_distinct_identifiers_only_collect_bounded_samples(evidence_limit, monkeypatch):
    from qavf.engine.context import DataContext
    from qavf.rules.reconciliation_checks import ConsistentBatchRule
    from qavf.settings import QAVFSettings

    settings = QAVFSettings.model_validate({"datasets": {"a": {}, "b": {}}})
    frames = {
        "a": pl.DataFrame({"batch": range(10000)}).lazy(),
        "b": pl.DataFrame({"batch": [0]}).lazy(),
    }
    with DataContext.from_settings(settings, frames=frames) as ctx:
        ctx.evidence_limit = evidence_limit
        collect = ctx.collect_many
        shapes = []

        def bounded(names, plan):
            result = collect(names, plan)
            shapes.append(result.height)
            assert result.height == 1
            return result

        monkeypatch.setattr(ctx, "collect_many", bounded)
        findings = ConsistentBatchRule(
            {"batches": [{"datasets": ["a", "b"], "column": "batch"}]}
        ).evaluate(ctx)
    assert shapes
    assert any(f.assessed and not f.passed for f in findings)
