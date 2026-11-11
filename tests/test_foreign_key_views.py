"""Foreign keys may reference a view or an ``across`` copy (review CFG-08).

A view is its parent's rows that match ``where``, with the parent's contract, so a
key present in the parent but excluded by the view is orphaned. Both backends
agree (the parity scenario ``view_references`` compares whole reports); these
tests pin the meaning, the configuration rules and the native plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from qavf import explain, validate
from qavf.settings import QAVFSettings


def _write(directory: Path, name: str, text: str) -> str:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.as_posix()


def _config(directory: Path, ref_dataset: str = "roads.active") -> dict[str, Any]:
    roads = _write(directory, "roads.csv", "id,status,lanes\n1,open,2\n2,closed,2\n3,open,0\n")
    crashes = _write(directory, "crashes.csv", "road_id\n1\n2\n3\n")
    return {
        "datasets": {
            "roads": {
                "path": roads,
                "views": {"active": {"where": "status = 'open'"}, "wide": {"where": "lanes > 1"}},
            },
            "crashes": {
                "path": crashes,
                "foreign_keys": [
                    {"columns": ["road_id"], "ref_dataset": ref_dataset, "ref_columns": ["id"]}
                ],
            },
        }
    }


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_a_key_the_view_excludes_is_orphaned(tmp_path: Path, backend: str) -> None:
    report = validate(config=_config(tmp_path), backend=backend)
    [result] = [r for r in report.results if r.check_id == "key.reference"]
    assert result.status.value == "FAIL"
    assert list(result.datasets) == ["crashes", "roads.active"]
    # Road 2 exists in roads but is closed, so the active view does not hold it.
    assert [record["road_id"] for record in result.affected_records] == [2]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_the_same_keys_pass_against_the_whole_parent(tmp_path: Path, backend: str) -> None:
    report = validate(config=_config(tmp_path, "roads"), backend=backend)
    assert report.verdict() == "passed"


def test_settings_accept_a_view_and_an_across_copy_as_the_referenced_dataset() -> None:
    settings = QAVFSettings.model_validate(
        {
            "datasets": {
                "sites": {
                    "path": "{site}.csv",
                    "across": {"site": ["a", "b"]},
                    "views": {"live": {"where": "x > 0"}},
                },
                "visits": {
                    "path": "visits.csv",
                    "foreign_keys": [
                        {"columns": ["id"], "ref_dataset": "sites@a"},
                        {"columns": ["id"], "ref_dataset": "sites@b.live"},
                    ],
                },
            }
        }
    )
    assert settings.referenced_dataset("sites@b.live") == "sites@b"
    assert settings.referenced_dataset("sites@a") == "sites@a"


@pytest.mark.parametrize("name", ["roads.closed", "ROADS.active", "nowhere"])
def test_an_unknown_referenced_relation_is_an_error(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValidationError, match="foreign key references unknown dataset"):
        QAVFSettings.model_validate(_config(tmp_path, name))


def test_the_native_plan_loads_the_view_s_parent_with_the_columns_its_where_reads(
    tmp_path: Path,
) -> None:
    plan = explain(config=_config(tmp_path), backend="duckdb")
    datasets = {item.name: item for item in plan.datasets}
    assert set(datasets["crashes"].component) == {"crashes", "roads"}
    assert datasets["roads"].retained_columns == ("id", "status")
