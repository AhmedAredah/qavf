"""Check-instance identity: tags, tag policies and ``run.require`` follow each placed item.

Two items may share a display name (two unnamed checks of one kind on a dataset);
their results, tags and requirements must never be confused.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import validate
from qavf.engine.severity import Status
from qavf.reports.models import RuleResult
from qavf.settings import QAVFSettings

BACKENDS = ("duckdb", "polars")
EXPLORATORY = {"tags": {"exploratory": {"fail_on": "never"}}}


def _expressions(*items: dict[str, Any]) -> dict[str, Any]:
    return {"datasets": {"d": {"checks": list(items)}}, "run": EXPLORATORY}


def _required(report: Any) -> list[RuleResult]:
    return [r for r in report.results if (r.check or "").startswith("required: ")]


CRITICAL = {"check": "expression", "expr": "x > 0", "tags": ["critical"]}
LOOSE = {"check": "expression", "expr": "x > -10", "tags": ["exploratory"]}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("order", ["critical_first", "exploratory_first"])
def test_unnamed_checks_of_one_kind_keep_their_own_tags(backend: str, order: str) -> None:
    items = (CRITICAL, LOOSE) if order == "critical_first" else (LOOSE, CRITICAL)
    report = validate(
        {"d": pl.DataFrame({"x": [-1]})}, config=_expressions(*items), backend=backend
    )
    failed = [r for r in report.results if r.status is Status.FAIL]
    assert [r.tags for r in failed] == [["critical"]]
    index = items.index(CRITICAL)
    assert failed[0].item_ids == [f"datasets.d.checks[{index}]"]
    assert report.verdict() == "failed"
    assert report.exit_code() != 0


def test_explicit_name_equal_to_another_items_default_name_is_rejected() -> None:
    renamed = {**LOOSE, "name": "expression on d"}
    with pytest.raises(ValidationError, match="expression on d"):
        QAVFSettings.model_validate(_expressions(CRITICAL, renamed))


def test_unnamed_items_of_one_kind_may_share_their_default_name() -> None:
    QAVFSettings.model_validate(_expressions(CRITICAL, LOOSE))


def test_older_reports_without_item_ids_still_load() -> None:
    result = RuleResult.model_validate(
        {
            "rule_id": "INP-006",
            "name": "n",
            "description": "d",
            "category": "input",
            "status": "PASS",
            "severity": "ERROR",
            "message": "m",
        }
    )
    assert result.item_ids == []


SAME_NAME = {"check": "expression", "name": "x positive", "expr": "x > 0"}


def _two_datasets(second: dict[str, Any], require: list[str]) -> dict[str, Any]:
    return {
        "datasets": {"a": {"checks": [SAME_NAME]}, "b": {"checks": [second]}},
        "run": {"require": require},
    }


FRAMES = {"a": pl.DataFrame({"x": [1]}), "b": pl.DataFrame({"x": [1]})}


@pytest.mark.parametrize("backend", BACKENDS)
def test_requiring_a_shared_name_requires_every_instance(backend: str) -> None:
    config = _two_datasets({**SAME_NAME, "enabled": False}, ["x positive"])
    report = validate(FRAMES, config=config, backend=backend)
    required = _required(report)
    assert [r.status for r in required] == [Status.ERROR]
    assert "datasets.b.checks[0]" in required[0].message
    assert report.verdict() != "passed"


@pytest.mark.parametrize("backend", BACKENDS)
def test_requiring_a_shared_name_is_met_when_every_instance_is_assessed(backend: str) -> None:
    report = validate(FRAMES, config=_two_datasets(SAME_NAME, ["x positive"]), backend=backend)
    assert not _required(report)
    assert report.verdict() == "passed"


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_selection_narrows_a_shared_name_requirement(backend: str) -> None:
    config = _two_datasets({**SAME_NAME, "expr": "x > 5"}, ["x positive"])
    report = validate(FRAMES, config=config, backend=backend, datasets="a")
    assert not _required(report)
    assert report.verdict() == "passed"


def test_a_check_named_like_a_dataset_is_rejected() -> None:
    config = _two_datasets({"check": "expression", "name": "bcheck", "expr": "x > 0"}, ["b"])
    config["datasets"]["a"]["checks"] = [{**SAME_NAME, "name": "b"}]
    with pytest.raises(ValidationError, match="dataset"):
        QAVFSettings.model_validate(config)


FILTERED = {"check": "expression", "expr": "x < 10", "where": "x > 0"}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("item", "values", "verdict"),
    [
        (FILTERED, [1, 2], "passed"),
        (FILTERED, [1, 20], "failed"),
        ({"check": "expression", "expr": "x  <  10"}, [1, 2], "passed"),
    ],
    ids=["filtered_pass", "filtered_fail", "spaced_pass"],
)
def test_required_unnamed_expression_is_assessed(
    backend: str, item: dict[str, Any], values: list[int], verdict: str
) -> None:
    config = {"datasets": {"d": {"checks": [item]}}, "run": {"require": ["expression on d"]}}
    report = validate({"d": pl.DataFrame({"x": values})}, config=config, backend=backend)
    assert not _required(report)
    assert report.verdict() == verdict


def _view_collision(physical: str) -> dict[str, Any]:
    return {
        "datasets": {
            "roads": {"path": "roads.csv", "views": {"active": {"where": "id = 1"}}},
            physical: {"path": "active.csv"},
        }
    }


@pytest.mark.parametrize("physical", ["roads.active", "ROADS.Active"])
def test_a_view_and_a_dataset_of_one_name_are_rejected(physical: str) -> None:
    with pytest.raises(ValidationError) as caught:
        QAVFSettings.model_validate(_view_collision(physical))
    message = str(caught.value)
    assert "datasets.roads.views.active" in message
    assert f"datasets.{physical}" in message


def test_two_datasets_differing_only_in_case_are_rejected() -> None:
    config = {"datasets": {"Roads": {"path": "a.csv"}, "roads": {"path": "b.csv"}}}
    with pytest.raises(ValidationError, match="Roads"):
        QAVFSettings.model_validate(config)


def test_an_across_copy_colliding_with_a_dataset_is_rejected() -> None:
    config = {
        "datasets": {
            "counts": {"path": "{site}.csv", "across": {"site": ["a"]}},
            "COUNTS@A": {"path": "other.csv"},
        }
    }
    with pytest.raises(ValidationError, match="counts@a"):
        QAVFSettings.model_validate(config)


def test_dotted_dataset_names_stay_usable_when_unambiguous() -> None:
    config = {
        "datasets": {
            "roads": {"path": "roads.csv", "views": {"active": {"where": "id = 1"}}},
            "roads.archive": {"path": "archive.csv"},
        }
    }
    QAVFSettings.model_validate(config)
