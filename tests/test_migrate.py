"""QAVF 0.6 configurations: rejected with a pointer to ``qavf migrate``, which converts them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from qavf.checks.legacy import LegacyConfigurationError
from qavf.cli import main
from qavf.demo import demo_config, write_demo_project
from qavf.engine.runner import QARunner
from qavf.migrate import Migrator, migrate_file
from qavf.reports.models import RuleResult
from qavf.settings import QAVFSettings, load_settings, settings_from_text

OLD_DEMO = Path(__file__).parent / "configs" / "qavf06_demo.yaml"
"""The demo configuration as QAVF 0.6.2 shipped it."""


def _old_demo(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    write_demo_project(root)
    old = root / "old.yaml"
    old.write_text(OLD_DEMO.read_text(encoding="utf-8"), encoding="utf-8")
    return old


def _outcome(results: list[RuleResult]) -> list[tuple[Any, ...]]:
    return sorted(
        (r.rule_id, r.status.value, r.severity.value, r.affected_count, r.message)
        for r in results
        if r.status.value != "SKIPPED"
    )


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ({"rules": {"TMP-001": {"enabled": False}}}, "rules"),
        ({"assertions": [{"name": "a", "sql": "SELECT 1"}]}, "assertions"),
        ({"datasets": {"d": {"expressions": [{"expr": "x > 0"}]}}}, "datasets.d.expressions"),
        ({"engine": {"run_unconfigured_rules": False}}, "engine.run_unconfigured_rules"),
        ({"engine": {"include_categories": ["input"]}}, "engine.include_categories"),
        ({"engine": {"exclude_categories": ["temporal"]}}, "engine.exclude_categories"),
        ({"engine": {"fail_on": "WARNING"}}, "engine.fail_on"),
        ({"engine": {"fail_on_incomplete": False}}, "engine.fail_on_incomplete"),
        ({"engine": {"require": ["INP-004"]}}, "engine.require"),
    ],
)
def test_old_configurations_fail_fast_naming_the_key_and_the_converter(
    tmp_path: Path, raw: dict[str, Any], key: str
) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(LegacyConfigurationError) as error:
        load_settings(path)
    assert error.value.key == key
    assert f"'{key}'" in str(error.value) and "qavf migrate" in str(error.value)
    assert str(path.resolve()) in str(error.value)
    with pytest.raises(LegacyConfigurationError):
        settings_from_text(path.read_text(encoding="utf-8"), path)
    with pytest.raises(ValidationError, match="qavf migrate"):
        QAVFSettings.model_validate(raw)


def test_run_reports_an_old_configuration_as_a_usage_error(tmp_path: Path, capsys) -> None:
    path = tmp_path / "old.yaml"
    path.write_text("rules: {TMP-001: {enabled: false}}\n", encoding="utf-8")
    assert main(["run", "-c", str(path)]) == 2
    assert "qavf migrate" in capsys.readouterr().err


def test_old_demo_converts_and_gives_the_same_results(tmp_path: Path) -> None:
    old = _old_demo(tmp_path)
    new = old.with_name("qavf.yaml")
    migration = migrate_file(old, new)
    assert migration.changed
    text = new.read_text(encoding="utf-8")
    assert text.startswith("# yaml-language-server")
    converted = yaml.safe_load(text)
    for key in ("rules", "assertions"):
        assert key not in converted
    counts = converted["datasets"]["counts"]["checks"]
    assert [item["check"] for item in counts] == [
        "change_rate",
        "trend",
        "trend_outliers",
        "missing_periods",
        "matches_baseline",
    ]
    assert counts[0]["max"] == 0.15 and "max_change_rate" not in counts[0]
    assert counts[1]["expect"] == "increasing"
    assert counts[3]["expected"] == list(range(2016, 2025))
    assert counts[4]["rtol"] == 1e-9
    crashes = [item["check"] for item in converted["datasets"]["crashes"]["checks"]]
    assert crashes == ["expression", "expression", "entity_outliers", "group_comparison"]
    assert [item["check"] for item in converted["checks"]] == ["sql", "coverage"]
    migrated = QARunner(load_settings(new)).run()
    shipped = old.parent / "shipped.yaml"
    shipped.write_text(demo_config(), encoding="utf-8")
    assert _outcome(QARunner(load_settings(shipped)).run().results) == _outcome(migrated.results)


def test_migrate_check_reports_changes_without_writing(tmp_path: Path, capsys) -> None:
    old = _old_demo(tmp_path)
    before = old.read_text(encoding="utf-8")
    assert main(["migrate", str(old), "--check"]) == 1
    out = capsys.readouterr().out
    assert "rules.TMP-001 -> 1 change_rate check(s)" in out
    assert old.read_text(encoding="utf-8") == before
    new = tmp_path / "new.yaml"
    assert main(["migrate", str(old), "-o", str(new)]) == 0
    assert main(["migrate", str(new), "--check"]) == 0


def test_rule_level_settings_reach_every_generated_item() -> None:
    raw = {
        "datasets": {"a": {}, "b": {}},
        "rules": {
            "TMP-004": {
                "severity": "WARNING",
                "enabled": False,
                "params": {
                    "expected_periods": [1, 2],
                    "series": [
                        {"dataset": "a", "value": "x", "time_column": "t"},
                        {"dataset": "b", "value": "y", "time_column": "t", "group_by": ["g"]},
                    ],
                },
            },
            "REC-001": {
                "params": {
                    "reconciliations": [
                        {
                            "name": "totals",
                            "left": {"dataset": "a", "value": "x", "keys": ["k1"]},
                            "right": {"dataset": "b", "value": "y", "keys": ["k2"]},
                            "on": ["k"],
                        }
                    ]
                }
            },
            "STA-002": {
                "params": {"comparisons": [{"reference": "a", "dataset": "b", "column": "x"}]}
            },
            "INP-003": {"params": {"datasets": ["a"]}},
        },
        "engine": {
            "require": ["TMP-004", "INP-003[a.x: ge 0]"],
            "exclude_categories": ["spatial"],
            "fail_on": "WARNING",
        },
    }
    migration = Migrator().migrate(raw)
    config = migration.config
    a_item, b_item = config["datasets"]["a"]["checks"][0], config["datasets"]["b"]["checks"][0]
    for item in (a_item, b_item):
        assert item["check"] == "missing_periods"
        assert item["expected"] == [1, 2]
        assert item["severity"] == "WARNING" and item["enabled"] is False
    assert b_item["series"] == "y_by_g"
    assert config["datasets"]["b"]["series"] == {
        "y_by_g": {"value": "y", "time_column": "t", "group_by": ["g"]}
    }
    totals, drift = config["checks"]
    assert totals == {
        "check": "totals_match",
        "name": "totals",
        "datasets": ["a", "b"],
        "keys": ["k"],
        "sides": {"a": {"value": "x", "keys": ["k1"]}, "b": {"value": "y", "keys": ["k2"]}},
    }
    assert drift == {"check": "drift", "datasets": ["a", "b"], "column": "x"}
    assert config["run"] == {
        "require": ["missing_periods on a", "missing_periods on b", "a"],
        "fail_on": "WARNING",
    }
    assert "engine" not in config or not config["engine"]
    assert any("--skip-tags spatial" in note for note in migration.notes)
    settings = QAVFSettings.model_validate(config)
    assert settings.run.require == ["missing_periods on a", "missing_periods on b", "a"]


def test_identical_series_and_metrics_merge_into_named_entries() -> None:
    series = {"dataset": "a", "value": "x", "time_column": "t", "group_by": ["g"]}
    other = {**series, "aggregate": "mean"}
    metric = {
        "dataset": "a",
        "value": "x",
        "entity_column": "g",
        "normalize_by": {"dataset": "b", "value": "y"},
    }
    raw = {
        "datasets": {"a": {}, "b": {}},
        "rules": {
            "TMP-001": {"params": {"series": [series, other]}},
            "TMP-004": {"params": {"series": [series], "expected_periods": [1, 2]}},
            "SPA-001": {"params": {"metrics": [metric]}},
            "SPA-002": {"params": {"metrics": [metric], "groups": {"n": ["g1"]}}},
        },
    }
    config = Migrator().migrate(raw).config
    dataset = config["datasets"]["a"]
    assert dataset["series"] == {
        "x_by_g": {"value": "x", "time_column": "t", "group_by": ["g"]},
        "x_by_g_2": {"value": "x", "time_column": "t", "group_by": ["g"], "aggregate": "mean"},
    }
    assert [item.get("series") for item in dataset["checks"]] == [
        "x_by_g",
        "x_by_g_2",
        "x_by_g",
        None,
        None,
    ]
    assert dataset["metrics"] == {"x_by_g": {"value": "x", "entity": "g", "per": "b.y"}}
    assert [item.get("metric") for item in dataset["checks"][3:]] == ["x_by_g", "x_by_g"]
    QAVFSettings.model_validate(config)


def test_drop_disabled_leaves_disabled_rules_out() -> None:
    raw = {
        "datasets": {"a": {}},
        "rules": {
            "TMP-001": {
                "enabled": False,
                "params": {"series": [{"dataset": "a", "value": "x"}]},
            }
        },
    }
    assert "checks" not in Migrator(drop_disabled=True).migrate(raw).config["datasets"]["a"]
    assert Migrator().migrate(raw).config["datasets"]["a"]["checks"][0]["enabled"] is False


def test_new_configurations_are_left_unchanged() -> None:
    raw = {"datasets": {"a": {"checks": [{"check": "expression", "expr": "x > 0"}]}}}
    migration = Migrator().migrate(raw)
    assert not migration.changed
    assert migration.config == raw
