"""Safe dependency lifetimes and explicit declarations for opaque plugins."""

import traceback
from concurrent.futures import ThreadPoolExecutor

import pytest

from qavf import DataRequirements, Finding, QARunner, Rule
from qavf.engine.planning import build_plan
from qavf.engine.registry import RuleRegistry
from qavf.settings import QAVFSettings


class _Declared(Rule):
    rule_id = "TEST-A"
    name = "Declared reader"
    description = "A plugin with an explicit complete dependency contract."
    category = "test"

    def data_requirements(self, settings):
        return DataRequirements(frozenset({"a"}), columns={"a": ("x",)})

    def evaluate(self, ctx):
        return Finding.ok("read")


class _Both(_Declared):
    rule_id = "TEST-B"

    def data_requirements(self, settings):
        return DataRequirements(frozenset({"a", "b"}))


class _Opaque(_Declared):
    rule_id = "TEST-O"
    requires = ("a",)

    def data_requirements(self, settings):
        return None


def test_last_consumer_waits_for_every_actual_consumer():
    plan = build_plan([_Declared(), _Both(), _Opaque()], QAVFSettings(), ["a", "b", "unused"])
    assert plan.complete("TEST-A") == set()
    assert plan.complete("TEST-O") == {"unused"}
    assert plan.complete("TEST-B") == {"a", "b"}
    assert plan.complete("TEST-B") == set()
    assert plan.describe()[-1]["dependency_mode"] == "opaque"
    assert plan.describe()[-1]["datasets"] == ["a", "b", "unused"]


def test_parallel_completion_releases_each_dataset_once():
    rules = [_Declared(), _Both(), _Opaque()]
    plan = build_plan(rules, QAVFSettings(), ["a", "b"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        releases = list(pool.map(plan.complete, [rule.rule_id for rule in rules] * 5))
    assert sum("a" in result for result in releases) == 1
    assert sum("b" in result for result in releases) == 1


def test_builtin_nested_dataset_references_are_kept():
    settings = QAVFSettings.model_validate({
        "datasets": {"counts": {}, "sites": {}, "baseline": {}, "unrelated": {}},
        "engine": {"run_unconfigured_rules": False},
        "plugins": {"entry_points": False},
        "rules": {
            "TMP-004": {"params": {
                "series": [{"dataset": "counts", "value": "x", "group_by": ["id"]}],
                "groups_from": "sites",
            }},
            "STA-002": {"params": {
                "comparisons": [{"dataset": "counts", "reference": "baseline", "column": "x"}]
            }},
        },
    })
    rules = QARunner(settings).plan()
    plan = build_plan(rules, settings, settings.datasets)
    descriptions = {entry["rule_id"]: entry for entry in plan.describe()}
    assert descriptions["TMP-004"]["datasets"] == ["counts", "sites"]
    assert descriptions["STA-002"]["datasets"] == ["baseline", "counts"]
    assert plan.complete("TMP-004") == {"sites"}
    assert plan.complete("STA-002") == {"baseline", "counts"}


def test_arbitrary_input_sql_retains_all_datasets_conservatively():
    settings = QAVFSettings.model_validate({
        "datasets": {"a": {}, "b": {}},
        "engine": {"run_unconfigured_rules": False},
        "rules": {"INP-007": {}},
    })
    plan = build_plan(QARunner(settings).plan(), settings, settings.datasets)
    assert plan.describe()[0]["datasets"] == ["a", "b"]
    assert plan.describe()[0]["dependency_mode"] == "opaque"


def test_invalid_column_dependency_cannot_authorize_early_release():
    with pytest.raises(ValueError, match="declared datasets"):
        DataRequirements(frozenset({"a"}), columns={"b": ("x",)})
    with pytest.raises(ValueError, match="collection"):
        DataRequirements("a")


def test_plugin_failure_logs_are_redacted(monkeypatch, caplog):
    class BrokenEntry:
        name = "broken"

        def load(self):
            raise RuntimeError("https://user:secret@example.org/x?token=hidden")

    monkeypatch.setattr("qavf.engine.registry.entry_points", lambda **kwargs: [BrokenEntry()])
    registry = RuleRegistry()
    registry.discover_entry_points()
    assert "secret" not in caplog.text
    assert "hidden" not in caplog.text
    assert "secret" not in registry.plugins["broken"]
    assert "hidden" not in registry.plugins["broken"]


def test_configured_plugin_import_failure_has_no_raw_exception_traceback(tmp_path):
    plugin = tmp_path / "leaky.py"
    plugin.write_text(
        'raise RuntimeError("https://user:secret@example.org/x?token=hidden")\n'
    )
    with pytest.raises(ImportError) as caught:
        RuleRegistry().discover_path(plugin)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "secret" not in rendered
    assert "hidden" not in rendered
    assert "RuntimeError" in rendered
