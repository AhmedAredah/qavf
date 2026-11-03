"""Rule and source registries reject invalid classes at registration (A3) and look
entries up directly instead of copying the registry (A4)."""

from __future__ import annotations

import abc
from typing import Any, ClassVar

import pytest

from qavf.engine.registry import Rule, RuleRegistry
from qavf.plugins.contracts import ExtensionContract
from qavf.sources.base import DataSource, SourceRegistry
from qavf.sources.parameters import SourceParams

pytest.importorskip("polars")


class NotARule:
    rule_id = "X1"
    check_name = "x1"
    name = "x"
    description = "d"
    category = "c"


class IntParamsRule(Rule):
    rule_id = "X2"
    check_name = "x2"
    name = "x2"
    description = "d"
    category = "c"
    Params: ClassVar[Any] = int

    def evaluate(self, context: Any) -> Any:
        return None


class NotASource:
    kind = "fake"


class AbstractSource(DataSource[SourceParams]):
    kind: ClassVar[str] = "abstract_fake"
    Params: ClassVar[type[SourceParams]] = SourceParams

    @abc.abstractmethod
    def extra(self) -> None: ...


class NamelessSource(DataSource[SourceParams]):
    kind: ClassVar[str] = ""
    Params: ClassVar[type[SourceParams]] = SourceParams

    def load(self) -> Any:
        return None


@pytest.mark.parametrize(
    ("rule", "message"),
    [(NotARule, "must subclass Rule"), (IntParamsRule, "Params must be a pydantic model")],
)
def test_rule_registry_rejects_invalid_classes_unchanged(rule: type, message: str) -> None:
    registry = RuleRegistry()
    with pytest.raises(TypeError, match=message):
        registry.register(rule)
    assert len(registry) == 0 and rule.rule_id not in registry


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (NotASource, "must subclass DataSource"),
        (AbstractSource, "is abstract"),
        (NamelessSource, "has no source type"),
    ],
)
def test_source_registry_rejects_invalid_classes_unchanged(source: type, message: str) -> None:
    registry = SourceRegistry()
    with pytest.raises(TypeError, match=message):
        registry.register(source)
    assert len(registry) == 0


def test_the_contract_names_the_class_and_rejects_non_classes() -> None:
    contract = ExtensionContract("rule", Rule, "rule_id", "id")
    with pytest.raises(TypeError, match="must subclass Rule"):
        contract.check("not a class")


def test_valid_builtin_rules_and_sources_still_register() -> None:
    from qavf.engine.compilation import build_registry, build_source_registry
    from qavf.settings import QAVFSettings

    settings = QAVFSettings(plugins={"entry_points": False})
    rules = build_registry(settings)
    sources = build_source_registry(settings)
    assert len(rules) > 10 and "file" in sources
    rule = next(iter(rules))
    assert rules.get(rule.rule_id) is rule and rule.rule_id in rules
    assert 42 not in rules and "NOPE-1" not in rules
    with pytest.raises(KeyError, match="unknown rule id"):
        rules.get("NOPE-1")
    with pytest.raises(KeyError, match="unknown source type"):
        sources.get("nope")


def test_lookups_do_not_copy_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from qavf.engine.compilation import build_registry, build_source_registry
    from qavf.settings import QAVFSettings

    settings = QAVFSettings(plugins={"entry_points": False})
    rules, sources = build_registry(settings), build_source_registry(settings)
    rule = next(iter(rules))
    monkeypatch.setattr(RuleRegistry, "_visible", _no_copy)
    monkeypatch.setattr(SourceRegistry, "_visible", _no_copy)
    assert rules.get(rule.rule_id) is rule and rule.rule_id in rules
    assert sources.get("file") is not None and "file" in sources


def _no_copy(self: object) -> dict[str, Any]:
    raise AssertionError("a lookup copied the whole registry")
