"""Shared fixtures for the QAVF test suite."""

from __future__ import annotations

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
