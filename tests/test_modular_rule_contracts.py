"""Independent rule contracts survive preparation faults without replay."""

from __future__ import annotations

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules import (
    reconciliation_checks,
    spatial_checks,
    statistical_checks,
    temporal_checks,
)
from qavf.rules._contracts import isolate_contract

CASES = [
    # Shared temporal preparation is used by both native/scoped and fallback reducers.
    (temporal_checks, "_aggregate_raw", temporal_checks.StepChangeRule, "series"),
    (statistical_checks, "normalize_missing", statistical_checks.ColumnOutlierRule, "targets"),
    (
        reconciliation_checks,
        "_context_side_totals",
        reconciliation_checks.TotalsReconcileRule,
        "reconciliations",
    ),
    (spatial_checks, "_build_entity_metric", spatial_checks.EntityOutlierRule, "metrics"),
]


def _parameters(kind, names):
    contracts = []
    for name in names:
        if kind == "reconciliations":
            contracts.append(
                {
                    "left": {"dataset": name, "value": "v"},
                    "right": {"dataset": "control", "value": "v"},
                }
            )
        else:
            spec = {"dataset": name, "value": "v"}
            if kind == "series":
                spec["time_column"] = "t"
            elif kind == "metrics":
                spec["entity_column"] = "t"
            contracts.append(spec)
    return {kind: contracts, **({"method": "iqr"} if kind == "metrics" else {})}


def _frames():
    data = pl.DataFrame({"t": range(8), "v": [1.0] * 7 + [100.0]})
    return {
        "bad": data,
        "good": data,
        "control": data.with_columns(pl.lit(1.0).alias("v")),
    }


@pytest.mark.parametrize(
    "module,seam,rule,kind", CASES, ids=["temporal", "statistical", "reconciliation", "spatial"]
)
@pytest.mark.parametrize("bad_first", [False, True])
def test_preparation_failure_keeps_independent_violation_without_replay(
    monkeypatch, module, seam, rule, kind, bad_first
):
    original = getattr(module, seam)
    failures = []

    def prepare(ctx, contract, *args, **kwargs):
        name = contract if isinstance(contract, str) else contract.dataset
        if name == "bad":
            failures.append(name)
            raise RuntimeError("cannot evaluate fixture")
        return original(ctx, contract, *args, **kwargs)

    monkeypatch.setattr(module, seam, prepare)
    names = ("bad", "good") if bad_first else ("good", "bad")
    with DataContext(frames=_frames()) as ctx:
        findings = rule(_parameters(kind, names)).evaluate(ctx)
    blocked = [f for f in findings if not f.assessed and "bad" in f.datasets]
    assert len(blocked) == 1
    assert "RuntimeError" in blocked[0].message
    assert "cannot evaluate fixture" in blocked[0].message
    assert any(f.assessed and not f.passed and "good" in f.datasets for f in findings)
    assert failures == ["bad"]


@pytest.mark.parametrize(
    "module,seam,rule,kind", CASES, ids=["temporal", "statistical", "reconciliation", "spatial"]
)
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_process_control_exception_propagates_from_rule_preparation(
    monkeypatch, module, seam, rule, kind, exception
):
    def interrupted(*args, **kwargs):
        raise exception("stop the run")

    monkeypatch.setattr(module, seam, interrupted)
    with DataContext(frames=_frames()) as ctx, pytest.raises(exception, match="stop the run"):
        rule(_parameters(kind, ("bad", "good"))).evaluate(ctx)


def test_nested_contract_failure_keeps_group_identity_and_sanitized_diagnostic():
    findings = []
    with (
        isolate_contract("data.value", ("data",), findings),
        isolate_contract("data.value", ("data",), findings, affected=[{"group": "A"}]),
    ):
        raise ValueError("password=secret_fixture cannot be used")
    assert len(findings) == 1
    failure = findings[0]
    assert not failure.assessed and not failure.passed
    assert failure.check == "data.value" and failure.datasets == ("data",)
    assert failure.affected == [{"group": "A"}]
    assert "secret_fixture" not in failure.message
    assert "ValueError at test_modular_rule_contracts.py:" in failure.message
