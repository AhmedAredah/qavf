"""Declared units: unit columns with allowed units, and unit-aware reconciliation."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import ColumnSpec, QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


def _run(frames, datasets, rules):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": datasets,
            "rules": rules,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


def test_rows_in_another_unit_are_violations() -> None:
    frame = pl.DataFrame({"mass": [1.0, 2.0, 3.0, None], "unit": ["kg", "t", "kg", "lb"]})
    datasets = {
        "t": {
            "schema": {"columns": {"mass": {"nullable": True, "unit": "kg", "unit_column": "unit"}}}
        }
    }
    report = _run({"t": frame}, datasets, {"INP-003": {}})
    result = report.get("INP-003", "t.mass: units")
    assert result.status is Status.FAIL and result.affected_count == 1  # the null row is exempt
    allowed = {
        "t": {
            "schema": {
                "columns": {
                    "mass": {"nullable": True, "unit_column": "unit", "allowed_units": ["kg", "t"]}
                }
            }
        }
    }
    assert not _run({"t": frame}, allowed, {"INP-003": {}}).get("INP-003").failed


def test_a_missing_unit_column_is_unassessed() -> None:
    frame = pl.DataFrame({"mass": [1.0]})
    datasets = {"t": {"schema": {"columns": {"mass": {"unit": "kg", "unit_column": "unit"}}}}}
    result = _run({"t": frame}, datasets, {"INP-003": {}}).get("INP-003", "t.mass: units")
    assert result.status is Status.ERROR


@pytest.mark.parametrize(
    "config",
    [{"unit_column": "u"}, {"allowed_units": ["kg"]}, {"unit_column": "u", "allowed_units": []}],
)
def test_unit_settings_are_validated(config) -> None:
    with pytest.raises(ValidationError):
        ColumnSpec.model_validate(config)


def test_reconciliation_refuses_different_units() -> None:
    left = pl.DataFrame({"k": [1], "mass": [1000.0]})
    right = pl.DataFrame({"k": [1], "mass": [1.0]})
    rec = {
        "left": {"dataset": "a", "value": "mass"},
        "right": {"dataset": "b", "value": "mass"},
        "on": ["k"],
    }
    datasets = {
        "a": {"schema": {"columns": {"mass": {"unit": "kg"}}}},
        "b": {"schema": {"columns": {"mass": {"unit": "t"}}}},
    }
    [result] = _run(
        {"a": left, "b": right}, datasets, {"REC-001": {"params": {"reconciliations": [rec]}}}
    ).results_for("REC-001")
    assert result.status is Status.ERROR and "different units" in result.message
    same = {"a": datasets["a"], "b": {"schema": {"columns": {"mass": {"unit": "kg"}}}}}
    report = _run({"a": left, "b": left}, same, {"REC-001": {"params": {"reconciliations": [rec]}}})
    assert report.get("REC-001").status is Status.PASS
