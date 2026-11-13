"""Shared fixtures for the QAVF test suite."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from qavf.demo import Datasets, make_demo_data
from qavf.engine.registry import RuleRegistry
from qavf.testing import discovered_rules

_DEMO = make_demo_data(seed=0)


@pytest.fixture(scope="session")
def registry() -> RuleRegistry:
    """Built-in rules only, so installed plugins cannot affect core tests."""
    return discovered_rules(entry_points=False)


@pytest.fixture
def demo() -> Datasets:
    """A fresh mapping of the clean demo datasets (frames are immutable, so shared)."""
    return dict(_DEMO)


@pytest.fixture
def small() -> dict[str, pl.DataFrame]:
    """A tiny parent/child pair for input-validation tests."""
    return {
        "parent": pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
        "child": pl.DataFrame(
            {
                "id": [1, 1, 2, 3],
                "year": [2020, 2025, 2020, 2020],
                "value": [10.0, 12.0, 20.0, 30.0],
            }
        ),
    }


_STEADY_FREE_DISK = 512 * 1024**3


def _steady_free_disk(_directory: Path) -> int:
    return _STEADY_FREE_DISK


@pytest.fixture(autouse=True)
def steady_free_disk(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give automatic spill planning the same free disk on every machine.

    Automatic spill budgets are sized from free disk space, so a nearly full test
    machine made otherwise deterministic tests fail. Tests that measure the real
    probe opt out with ``@pytest.mark.real_disk``; tests that need another figure
    patch ``qavf.resource_plan._free_disk_bytes`` themselves.
    """
    if request.node.get_closest_marker("real_disk") is None:
        monkeypatch.setattr("qavf.resource_plan._free_disk_bytes", _steady_free_disk)
