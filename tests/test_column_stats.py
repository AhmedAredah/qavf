"""QAVF 0.7 column limits: ``between``, statistics under ``stats``, ``conditions``.

Old column keys (``mean:`` beside the limits, a column's ``checks``) are rejected
with a pointer to ``qavf migrate``, which moves them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from qavf import validate
from qavf.checks.legacy import LegacyConfigurationError
from qavf.engine.severity import Status
from qavf.migrate import ColumnMigration, Migrator
from qavf.settings import ColumnSpec, QAVFSettings, load_settings

BACKENDS = ("duckdb", "polars")

_VALUES = "k,x,g\n1,1,p\n2,2,p\n3,3,q\n4,4,q\n5,10,q\n"
"""``x``: min 1, max 10, mean 4, median 3, sum 20, five distinct values."""


def _config(tmp_path: Path, column: dict[str, Any]) -> dict[str, Any]:
    path = tmp_path / "values.csv"
    path.write_text(_VALUES, encoding="utf-8")
    columns = {"k": {"dtype": "int"}, "x": {"dtype": "int", **column}, "g": {"dtype": "str"}}
    return {"datasets": {"values": {"path": str(path), "schema": {"columns": columns}}}}


def _column_results(report: Any) -> dict[str, tuple[Status, str]]:
    """Each column check's status and message by label; passing checks are listed in
    one summary result."""
    outcomes = {}
    for result in report.results:
        if result.rule_id != "INP-003":
            continue
        for label in result.metrics.get("passed_checks") or ():
            outcomes[label] = (Status.PASS, result.message)
        if result.check:
            outcomes[result.check] = (result.status, result.message)
    return outcomes


# --------------------------------------------------------------------- between
def test_between_is_ge_and_le() -> None:
    spec = ColumnSpec.model_validate({"dtype": "int", "between": [0, 10]})
    assert (spec.ge, spec.le, spec.between) == (0, 10, None)
    (check,) = spec.all_checks()
    assert check.constraints() == {"ge": 0, "le": 10}


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ({"between": [0, 10], "ge": 1}, "drop 'ge' or 'between'"),
        ({"between": [0, 10], "lt": 5}, "drop 'lt' or 'between'"),
        ({"between": [10, 0]}, re.escape("'between: [10, 0]' leaves no allowed value")),
        ({"between": [1]}, "at least 2 items"),
        ({"between": [1, 2, 3]}, "at most 2 items"),
        ({"stats": {"mean": {"between": [1, 2], "le": 3}}}, "drop 'le' or 'between'"),
        ({"stats": {"mean": {"between": [5, 1]}}}, "leaves no allowed value"),
        ({"stats": {"median": {"gt": 5, "lt": 5}}}, "leave no allowed value"),
        ({"stats": {"std": {"lt": 0}}}, "'std' can never be negative"),
        ({"stats": {"distinct_count": {"le": -1}}}, "'distinct_count' can never be negative"),
        ({"stats": {}}, "'stats' must bound at least one"),
        ({"stats": {"mean": {}}}, "set 'ge', 'le', 'gt', 'lt' or 'between'"),
        ({"stats": {"mode": {"ge": 1}}}, "Extra inputs are not permitted"),
        ({"dtype": "str", "stats": {"mean": {"ge": 1}}}, "the 'mean' statistic needs a numeric"),
    ],
)
def test_invalid_limits_are_rejected(column: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ColumnSpec.model_validate({"dtype": "int", **column})


def test_between_takes_dates() -> None:
    spec = ColumnSpec.model_validate({"dtype": "date", "between": ["2020-01-01", "2020-12-31"]})
    assert (spec.ge.isoformat()[:10], spec.le.isoformat()[:10]) == ("2020-01-01", "2020-12-31")


# ----------------------------------------------------------------------- stats
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("stats", "passes"),
    [
        ({"min": {"ge": 1}}, True),
        ({"min": {"gt": 1}}, False),
        ({"max": {"between": [10, 10]}}, True),
        ({"max": {"lt": 10}}, False),
        ({"mean": {"between": [3, 5]}}, True),
        ({"mean": {"gt": 4}}, False),
        ({"median": {"le": 3}}, True),
        ({"median": {"lt": 3}}, False),
        ({"std": {"gt": 0}}, True),
        ({"sum": {"between": [20, 20]}}, True),
        ({"sum": {"gt": 20}}, False),
        ({"distinct_count": {"ge": 5}}, True),
        ({"distinct_count": {"lt": 5}}, False),
    ],
)
def test_each_statistic_takes_every_bound(
    tmp_path: Path, backend: str, stats: dict[str, Any], passes: bool
) -> None:
    report = validate(config=_config(tmp_path, {"stats": stats}), backend=backend)
    ((status, message),) = [
        outcome
        for label, outcome in _column_results(report).items()
        if label.startswith("values.x") and not label.endswith("not null")
    ]
    assert (status is Status.PASS) == passes, message
    if not passes:
        (name,) = stats
        assert f"{name} " in message and "outside" in message


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_exclusive_bound_is_shown_as_an_open_interval(tmp_path: Path, backend: str) -> None:
    report = validate(config=_config(tmp_path, {"stats": {"mean": {"gt": 4}}}), backend=backend)
    messages = [message for _, message in _column_results(report).values()]
    assert any("mean 4 outside (4, ]" in message for message in messages), messages


@pytest.mark.parametrize("backend", BACKENDS)
def test_conditions_have_their_own_severity_and_rows(tmp_path: Path, backend: str) -> None:
    column = {
        "between": [1, 10],
        "conditions": [
            {"name": "small q", "where": "g = 'q'", "le": 4, "severity": "WARNING"},
            {"name": "p total", "where": "g = 'p'", "stats": {"sum": {"between": [3, 3]}}},
        ],
    }
    report = validate(config=_config(tmp_path, column), backend=backend)
    small = next(r for r in report.results if (r.check or "").endswith("small q"))
    assert small.status is Status.FAIL and small.severity.value == "WARNING"
    assert small.affected_count == 1
    assert _column_results(report)["values.x: p total"][0] is Status.PASS


# ------------------------------------------------------------- old column keys
@pytest.mark.parametrize(
    ("column", "key"),
    [
        ({"mean": {"ge": 1}}, "mean"),
        ({"distinct_count": {"le": 3}}, "distinct_count"),
        ({"checks": [{"le": 3}]}, "checks"),
    ],
)
def test_old_column_keys_point_to_migrate(tmp_path: Path, column: dict, key: str) -> None:
    raw = {"datasets": {"d": {"schema": {"columns": {"x": {"dtype": "int", **column}}}}}}
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(LegacyConfigurationError, match="qavf migrate") as error:
        load_settings(path)
    assert error.value.key == f"datasets.d.schema.columns.x.{key}"
    reason = "conditions" if key == "checks" else "stats"
    assert reason in str(error.value)
    with pytest.raises(ValidationError, match="qavf migrate"):
        ColumnSpec.model_validate({"dtype": "int", **column})


def test_old_statistics_inside_old_checks_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"'min' is a QAVF 0.6 column setting"):
        ColumnSpec.model_validate({"conditions": [{"min": {"ge": 0}}]})


# --------------------------------------------------------------------- migrate
def _outcome(report: Any) -> list[tuple[Any, ...]]:
    return sorted(
        (r.rule_id, str(r.check), r.status.value, r.affected_count, r.message)
        for r in report.results
    )


def test_migrate_moves_statistics_and_renames_checks() -> None:
    old = {
        "dtype": "int",
        "ge": 0,
        "min": {"ge": 0},
        "nullable": True,
        "mean": {"le": 9},
        "checks": [{"name": "c", "where": "g = 'q'", "sum": {"ge": 1}, "le": 5}],
    }
    assert ColumnMigration().column(old) == {
        "dtype": "int",
        "ge": 0,
        "stats": {"min": {"ge": 0}, "mean": {"le": 9}},
        "nullable": True,
        "conditions": [{"name": "c", "where": "g = 'q'", "stats": {"sum": {"ge": 1}}, "le": 5}],
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_migrated_column_limits_report_as_written_by_hand(tmp_path: Path, backend: str) -> None:
    old_column = {
        "ge": 1,
        "mean": {"ge": 3, "le": 5},
        "max": {"le": 9},
        "checks": [{"name": "q", "where": "g = 'q'", "le": 4, "std": {"le": 1}}],
    }
    new_column = {
        "ge": 1,
        "stats": {"mean": {"between": [3, 5]}, "max": {"le": 9}},
        "conditions": [{"name": "q", "where": "g = 'q'", "le": 4, "stats": {"std": {"le": 1}}}],
    }
    old = _config(tmp_path, old_column)
    migration = Migrator().migrate(old)
    assert migration.changed
    migrated = validate(config=migration.config, backend=backend)
    written = validate(config=_config(tmp_path, new_column), backend=backend)
    assert _outcome(migrated) == _outcome(written)
    assert any(status is Status.FAIL for status, _ in _column_results(migrated).values())


def test_new_configurations_have_nothing_to_migrate(tmp_path: Path) -> None:
    config = _config(tmp_path, {"between": [1, 10], "stats": {"mean": {"ge": 1}}})
    assert not Migrator().migrate(config).changed
    QAVFSettings.model_validate(config)
