"""Independent coherent-source and relational-domain regression contracts."""

from __future__ import annotations

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules.reconciliation_checks import KeyGridRule, KeySetsMatchRule
from qavf.rules.spatial_checks import CoverageRule


@pytest.mark.parametrize("rule", ["coverage", "key_sets", "grid"])
def test_contract_observes_volatile_keys_once(rule, tmp_path):
    calls = []

    def observed(_):
        value = "a" if not calls else "b"
        calls.append(value)
        return pl.DataFrame({"id": [value]})

    lazy = pl.DataFrame({"id": ["a"]}).lazy().map_batches(observed, schema={"id": pl.String})
    with DataContext(
        frames={"left": lazy, "right": pl.DataFrame({"id": ["b"]})}, temp_directory=tmp_path
    ) as ctx:
        if rule == "coverage":
            results = CoverageRule({"datasets": ["left"], "entity_column": "id"}).evaluate(ctx)
        elif rule == "key_sets":
            results = KeySetsMatchRule(
                {
                    "pairs": [
                        {
                            "left": {"dataset": "left", "columns": ["id"]},
                            "right": {"dataset": "right", "columns": ["id"]},
                        }
                    ]
                }
            ).evaluate(ctx)
        else:
            results = KeyGridRule(
                {"grids": [{"dataset": "left", "dimensions": {"id": {"values": ["a", "b"]}}}]}
            ).evaluate(ctx)
        assert calls == ["a"]
        assert len(results) == 1 and results[0].assessed
        assert results[0].passed is (rule == "coverage")
    assert not list(tmp_path.iterdir())


def test_same_source_key_pair_uses_union_projection_once(tmp_path):
    calls = []
    frame = pl.DataFrame({"a": [1, 2], "b": [1, 3], "unused": ["wide"] * 2})

    def observed(batch):
        calls.append(1)
        return batch

    with DataContext(
        frames={"data": frame.lazy().map_batches(observed)}, temp_directory=tmp_path
    ) as ctx:
        result = KeySetsMatchRule(
            {
                "pairs": [
                    {
                        "left": {"dataset": "data", "columns": ["a"]},
                        "right": {"dataset": "data", "columns": ["b"]},
                    }
                ]
            }
        ).evaluate(ctx)
    assert calls == [1]
    assert len(result) == 1 and result[0].assessed and not result[0].passed
    assert result[0].affected_count == 2


def test_coverage_counts_high_cardinality_domains_with_bounded_evidence(monkeypatch, tmp_path):
    size = 2000
    frame = pl.DataFrame(
        {"id": [*range(size), *range(size - 1)], "period": [0] * size + [1] * (size - 1)}
    )
    original = pl.Series.to_list

    def bounded(series):
        assert len(series) <= 2, "a complete key domain was expanded into Python"
        return original(series)

    monkeypatch.setattr(pl.Series, "to_list", bounded)
    with DataContext(
        frames={"data": frame.lazy()}, evidence_limit=2, temp_directory=tmp_path
    ) as ctx:
        [result] = CoverageRule(
            {"datasets": ["data"], "entity_column": "id", "time_column": "period"}
        ).evaluate(ctx)
    assert result.assessed and not result.passed
    assert result.affected_count == 1
    assert result.affected == [{"dataset": "data", "period": 1, "missing_entities": [size - 1]}]


def test_grid_rank_stream_retains_exact_large_cartesian_cardinality(tmp_path):
    columns = [f"k{index}" for index in range(65)]
    frame = pl.DataFrame({column: [0] for column in columns})
    with DataContext(
        frames={"data": frame.lazy()}, evidence_limit=2, temp_directory=tmp_path
    ) as ctx:
        [result] = KeyGridRule(
            {
                "grids": [
                    {
                        "dataset": "data",
                        "dimensions": {column: {"values": [0, 1]} for column in columns},
                    }
                ]
            }
        ).evaluate(ctx)
    assert result.assessed and not result.passed
    assert result.affected_count == 2**65 - 1
    assert result.total_records == 2**65
    assert len(result.affected) == 2
    affected = result.affected.to_dicts()
    assert affected[0][columns[-1]] == 1
    assert affected[1][columns[-2]] == 1
    assert affected[1][columns[-1]] == 0
