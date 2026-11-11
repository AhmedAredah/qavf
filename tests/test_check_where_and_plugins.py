"""``where`` is taken by every dataset check (natively, or through a hidden view of
the dataset), and a plugin check binds to the dataset it is placed on (review
findings R6 and R11)."""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl
import pytest

from qavf.checks.catalog import CATALOG, definition_for
from qavf.engine.compilation import ConfigurationError
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.runner import QARunner, build_source_registry
from qavf.scaffold import config_json_schema
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules


class _AcmeParams(RuleParams):
    dataset: str | None = None
    payload: Any = None
    where: str | None = None


class _AcmeRule(Rule):
    rule_id: ClassVar[str] = "ACME-1"
    check_name: ClassVar[str] = "acme"
    name: ClassVar[str] = "Acme"
    description: ClassVar[str] = "A plugin check for the tests."
    category: ClassVar[str] = "custom"
    Params: ClassVar[type[RuleParams]] = _AcmeParams

    def evaluate(self, ctx):
        return Finding.ok(f"target {self.params.dataset} where {self.params.where}")


class _PlainParams(RuleParams):
    dataset: str | None = None


class _PlainRule(_AcmeRule):
    rule_id: ClassVar[str] = "ACME-2"
    check_name: ClassVar[str] = "plain"
    Params: ClassVar[type[RuleParams]] = _PlainParams

    def evaluate(self, ctx):
        return Finding.ok(
            f"{ctx.get(str(self.params.dataset)).height} row(s) of {self.params.dataset}"
        )


class _BareParams(RuleParams):
    payload: Any = None


class _BareRule(_AcmeRule):
    rule_id: ClassVar[str] = "ACME-3"
    check_name: ClassVar[str] = "bare"
    Params: ClassVar[type[RuleParams]] = _BareParams


def _registry():
    registry = discovered_rules(entry_points=False)
    registry.register(_AcmeRule)
    registry.register(_PlainRule)
    registry.register(_BareRule)
    return registry


def _run(item: dict[str, Any]):
    settings = QAVFSettings.model_validate(
        {"plugins": {"entry_points": False}, "datasets": {"a": {"checks": [item]}, "b": {}}}
    )
    frames = {"a": pl.DataFrame({"x": [1]}), "b": pl.DataFrame({"x": [1]})}
    return settings, QARunner(settings, registry=_registry()).run(frames=frames)


@pytest.mark.parametrize(
    "check", ["totals_match", "key_sets_match", "drift", "coverage", "batch_consistent"]
)
def test_top_level_where_points_at_views(check):
    item = {"check": check, "datasets": ["d", "e"], "where": "x > 0"}
    with pytest.raises(ValueError, match=r"top-level \w+ check: one condition is ambiguous"):
        QAVFSettings.model_validate({"datasets": {"d": {}, "e": {}}, "checks": [item]})


def test_every_dataset_check_advertises_where():
    native = {name for name, d in CATALOG.items() if d.configurable and d.supports_where}
    assert native == {"expression", "compare"}
    for name, definition in CATALOG.items():
        if definition.configurable:
            dataset_scoped = definition.placement == "dataset"
            assert definition.takes_where == (dataset_scoped or name == "compare"), name
    schema = config_json_schema(_registry(), build_source_registry(QAVFSettings()))
    definitions = schema["$defs"]
    assert "where" in definitions["ExpressionItem"]["properties"]
    assert "where" not in definitions["SqlItem"]["properties"]
    assert "where" in definitions["Check_compare"]["properties"]
    assert "where" not in definitions["Check_totals_match"]["properties"]
    assert "where" in definitions["Check_fresh"]["properties"]
    assert "min_applicable_rows" in definitions["Check_fresh"]["properties"]
    assert "min_applicable_rows" not in definitions["Check_compare"]["properties"]
    assert "where" in definitions["Check_acme"]["properties"]
    assert "where" in definitions["Check_plain"]["properties"]
    assert "where" not in definitions["Check_bare"]["properties"]


def test_a_plugin_declaring_where_receives_it():
    _, report = _run({"check": "acme", "where": "x > 0"})
    (result,) = report.results_for("ACME-1")
    assert result.message == "target a where x > 0"


def test_a_plugin_naming_its_dataset_reads_the_filtered_rows():
    frames = {"a": pl.DataFrame({"x": [1, -1, 2]}), "b": pl.DataFrame({"x": [1]})}
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "datasets": {"a": {"checks": [{"check": "plain", "where": "x > 0"}]}, "b": {}},
        }
    )
    report = QARunner(settings, registry=_registry()).run(frames=frames)
    (result,) = report.results_for("ACME-2")
    assert result.message.startswith("2 row(s) of a Assessed 2 of 3 row(s) of a")
    assert result.datasets == ["a"]
    assert result.parameters["where"] == "x > 0"
    assert "__where_" not in str(result.model_dump())


def test_a_plugin_naming_no_dataset_rejects_where():
    with pytest.raises(ConfigurationError, match="'where' is not supported by the bare check"):
        _run({"check": "bare", "where": "x > 0"})


def test_a_plugin_naming_another_dataset_than_its_placement_is_rejected():
    with pytest.raises(ConfigurationError, match="placed on dataset 'a' but names dataset 'b'"):
        _run({"check": "acme", "dataset": "b"})


def test_a_plugin_result_names_its_placed_dataset_and_keeps_its_settings():
    settings, report = _run({"check": "acme", "dataset": "a", "payload": {"k": [1]}})
    (result,) = report.results_for("ACME-1")
    assert result.datasets == ["a"]
    assert result.item_name == "acme on a"
    item = settings.datasets["a"].checks[0]
    definition = definition_for(_AcmeRule)
    params = definition.compile(item, _scope("a"))
    params["payload"]["k"].append(2)
    assert item.settings["payload"] == {"k": [1]}


def _scope(dataset: str):
    from qavf.checks.placement import NamedBlocks, Scope

    settings = QAVFSettings.model_validate({"datasets": {dataset: {}}})
    return Scope(f"datasets.{dataset}.checks[0]", dataset, NamedBlocks.of(settings))
