"""Plugin packaging, connection targets and constraint validation.

Private plugin packages, connection targets, contradictory or invalid
constraints, test-helper registries and user columns that share names
with diagnostic columns.
"""

from __future__ import annotations

import math
import sys
import textwrap
from datetime import date, datetime
from typing import ClassVar

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.registry import Finding, Rule, RuleRegistry, default_registry
from qavf.engine.runner import ConfigurationError, QARunner
from qavf.engine.severity import Status
from qavf.rules.helpers import tag
from qavf.settings import (
    Bounds,
    ColumnSpec,
    ConnectionSpec,
    ForeignKeySpec,
    QAVFSettings,
    SchemaSpec,
)
from qavf.sources.base import SourceEnvironment
from qavf.testing import discovered_rules, rule_results, run_rules


# =========================================================================
# Private plugin packages are never imported
# =========================================================================
@pytest.fixture
def plugin_package(tmp_path, monkeypatch):
    root = tmp_path / "audit_plugin_e09"
    (root / "_private" / "deep").mkdir(parents=True)
    (root / "public").mkdir()
    rule = textwrap.dedent(
        """
        from typing import ClassVar
        from qavf.engine.registry import Finding, Rule

        class {cls}(Rule):
            rule_id: ClassVar[str] = "{rule_id}"
            name: ClassVar[str] = "n"
            description: ClassVar[str] = "d"
            category: ClassVar[str] = "c"

            def evaluate(self, ctx):
                return Finding.ok("ok")
        """
    )
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "_private" / "__init__.py").write_text(
        "raise RuntimeError('private package executed')\n", encoding="utf-8"
    )
    (root / "_private" / "deep" / "__init__.py").write_text("", encoding="utf-8")
    (root / "_private" / "deep" / "rules.py").write_text(
        rule.format(cls="Hidden", rule_id="PKG-HIDDEN"), encoding="utf-8"
    )
    (root / "public" / "__init__.py").write_text("", encoding="utf-8")
    (root / "public" / "rules.py").write_text(
        rule.format(cls="Visible", rule_id="PKG-VISIBLE"), encoding="utf-8"
    )
    (root / "public" / "_helpers.py").write_text(
        "raise RuntimeError('private module executed')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "audit_plugin_e09"
    for name in [m for m in sys.modules if m.startswith("audit_plugin_e09")]:
        del sys.modules[name]


def test_private_subpackages_are_not_imported(plugin_package) -> None:
    registry = RuleRegistry()
    found = {rule.rule_id for rule in registry.discover_package(plugin_package)}
    assert found == {"PKG-VISIBLE"}
    assert f"{plugin_package}._private" not in sys.modules


# =========================================================================
# A connection uses the target its type requires
# =========================================================================
@pytest.mark.parametrize(
    "config",
    [
        {"type": "postgres", "uri": "host=db dbname=qa", "path": "wrong.db"},
        {"type": "mysql", "uri": "host=db database=qa", "path": "wrong.db"},
        {"type": "duckdb", "path": "x.duckdb", "uri": "host=db"},
        {"type": "sqlite", "path": "x.sqlite", "uri": "host=db"},
    ],
)
def test_connections_reject_the_other_target(config) -> None:
    with pytest.raises(ValidationError, match=r"takes '(path|uri)', not"):
        ConnectionSpec.model_validate(config)


def test_remote_connections_attach_their_uri(tmp_path) -> None:
    spec = ConnectionSpec(type="postgres", uri="host=db dbname=qa")
    sql = spec.attach_sql("db", SourceEnvironment(base_dir=tmp_path))
    assert "'host=db dbname=qa'" in sql and "TYPE postgres" in sql


# =========================================================================
# Contradictory and structurally invalid constraints are rejected
# =========================================================================
@pytest.mark.parametrize(
    ("model", "config"),
    [
        (Bounds, {"ge": math.nan}),
        (Bounds, {"le": math.inf}),
        (ColumnSpec, {"ge": math.nan}),
        (ColumnSpec, {"lt": -math.inf}),
        (ColumnSpec, {"ge": 10, "le": 1}),
        (ColumnSpec, {"gt": 5, "lt": 5}),
        (ColumnSpec, {"ge": 5, "lt": 5}),
        (ColumnSpec, {"gt": 5, "le": 5}),
        (ColumnSpec, {"ge": date(2021, 1, 1), "le": date(2020, 1, 1)}),
        (ColumnSpec, {"ge": 0, "le": date(2020, 1, 1)}),
        (ColumnSpec, {"min_length": 10, "max_length": 1}),
        (ColumnSpec, {"isin": []}),
        (ColumnSpec, {"regex": "("}),
        (ColumnSpec, {"checks": [{"ge": 3, "le": 2}]}),
        (ColumnSpec, {"dtype": "str", "ge": 0}),
        (ColumnSpec, {"dtype": "int", "regex": "x"}),
        (ColumnSpec, {"dtype": "float", "max_length": 3}),
        (ColumnSpec, {"dtype": "str", "mean": {"ge": 0}}),
        (ColumnSpec, {"dtype": "int", "ge": date(2020, 1, 1)}),
        (ColumnSpec, {"dtype": "date", "le": 5}),
        (ColumnSpec, {"std": {"le": -1}}),
        (SchemaSpec, {"min_rows": 10, "max_rows": 1}),
        (ForeignKeySpec, {"columns": [], "ref_dataset": "r"}),
        (ForeignKeySpec, {"columns": ["a", "a"], "ref_dataset": "r"}),
        (ForeignKeySpec, {"columns": ["a", "b"], "ref_dataset": "r", "ref_columns": ["x", "x"]}),
    ],
)
def test_invalid_constraints_are_rejected(model, config) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(config)


@pytest.mark.parametrize(
    ("model", "config"),
    [
        (Bounds, {"ge": 1, "le": 1}),
        (ColumnSpec, {"ge": 5, "le": 5}),
        (ColumnSpec, {"gt": 0, "lt": 1}),
        (ColumnSpec, {"ge": 0, "gt": 10, "le": 20}),
        (ColumnSpec, {"min_length": 3, "max_length": 3}),
        (ColumnSpec, {"dtype": "float", "ge": 0, "le": 1}),
        (ColumnSpec, {"dtype": "datetime", "ge": date(2020, 1, 1), "le": datetime(2021, 1, 1)}),
        (ColumnSpec, {"dtype": "str", "regex": r"[A-Z]{2}\d+", "distinct_count": {"le": 5}}),
        (ColumnSpec, {"ge": 2**53 + 1, "le": 2**53 + 1}),
        (ColumnSpec, {"notin": []}),
        (SchemaSpec, {"min_rows": 3, "max_rows": 3}),
        (ForeignKeySpec, {"columns": ["a"], "ref_dataset": "r"}),
    ],
)
def test_valid_boundary_constraints_are_accepted(model, config) -> None:
    model.model_validate(config)


# =========================================================================
# Test helpers use exactly the registry they are given
# =========================================================================
def test_an_explicitly_empty_registry_runs_nothing() -> None:
    report = run_rules({}, registry=RuleRegistry())
    assert report.results == []
    with pytest.raises(ConfigurationError, match="unknown rule"):
        rule_results("INP-001", {}, registry=RuleRegistry())


class _Late(Rule):
    rule_id: ClassVar[str] = "PKG-LATE"
    name: ClassVar[str] = "Late"
    description: ClassVar[str] = "Registered after discovery."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx):
        return Finding.ok("ok")


def test_late_registrations_are_discovered() -> None:
    assert "PKG-LATE" not in {rule.rule_id for rule in discovered_rules(entry_points=False)}
    default_registry.register(_Late)
    try:
        assert "PKG-LATE" in {rule.rule_id for rule in discovered_rules(entry_points=False)}
    finally:
        default_registry._rules.pop("PKG-LATE", None)
    assert "PKG-LATE" not in {rule.rule_id for rule in discovered_rules(entry_points=False)}


def test_discovered_registries_are_independent_snapshots() -> None:
    first = discovered_rules(entry_points=False)
    first.register(_Late)
    assert "PKG-LATE" not in {rule.rule_id for rule in discovered_rules(entry_points=False)}


# =========================================================================
# User columns may share names with diagnostic columns
# =========================================================================
def _settings(**config) -> QAVFSettings:
    return QAVFSettings.model_validate({"plugins": {"entry_points": False}, **config})


def _run(registry, frames, **config):
    return QARunner(_settings(**config), registry).run(frames=frames)


def test_user_row_column_survives_null_and_limit_failures(registry) -> None:
    frame = pl.DataFrame({"_row": [100, 101, 102], "v": [1.0, None, -5.0]})
    report = _run(
        registry,
        {"t": frame},
        datasets={"t": {"schema": {"columns": {"v": {"ge": 0}}}}},
    )
    failed = {r.check: r for r in report.results if r.rule_id == "INP-003" and r.failed}
    assert {r.status for r in failed.values()} == {Status.FAIL}
    null_record = failed["t.v: not null"].affected_records[0]
    assert null_record["_row"] == 101 and null_record["qavf__row"] == 1
    limit_record = failed["t.v: ge 0"].affected_records[0]
    assert limit_record["_row"] == 102 and limit_record["qavf__row"] == 2


def test_user_violates_column_is_preserved(registry) -> None:
    frame = pl.DataFrame({"violates": ["orig-a", "orig-b"], "v": [None, -1.0]})
    report = _run(
        registry,
        {"t": frame},
        datasets={"t": {"schema": {"columns": {"v": {"ge": 0}}}}},
    )
    failed = {r.check: r for r in report.results if r.rule_id == "INP-003" and r.failed}
    null_record = failed["t.v: not null"].affected_records[0]
    assert null_record["violates"] == "orig-a" and null_record["qavf_violates"] == "missing"
    limit_record = failed["t.v: ge 0"].affected_records[0]
    assert limit_record["violates"] == "orig-b" and "ge 0" in limit_record["qavf_violates"]


def test_primary_key_named_occurrences(registry) -> None:
    frame = pl.DataFrame({"occurrences": [1, 1, 2]})
    report = _run(
        registry,
        {"t": frame},
        datasets={
            "t": {"schema": {"primary_key": ["occurrences"], "columns": {"occurrences": {}}}}
        },
    )
    result = report.get("INP-004")
    assert result.status is Status.FAIL and result.affected_rows == 2
    assert result.affected_records == [{"occurrences": 1, "qavf_occurrences": 2}]


@pytest.mark.parametrize("column", ["rows", "_position"])
def test_foreign_key_columns_with_internal_names(registry, column) -> None:
    report = _run(
        registry,
        {"parent": pl.DataFrame({column: [1]}), "child": pl.DataFrame({column: [1, 2, 2]})},
        datasets={
            "parent": {},
            "child": {"foreign_keys": [{"columns": [column], "ref_dataset": "parent"}]},
        },
    )
    result = report.get("INP-005")
    assert result.status is Status.FAIL and result.affected_count == 1
    assert result.affected_rows == 2
    [record] = result.affected_records
    assert record[column] == 2


def test_tag_never_overwrites_user_columns() -> None:
    frame = pl.DataFrame({"dataset": ["mine"], "x": [1]})
    tagged = tag(frame, dataset="t")
    assert tagged["dataset"].to_list() == ["mine"]
    assert tagged["qavf_dataset"].to_list() == ["t"]
