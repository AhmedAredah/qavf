"""Comparison extension points: in-code registration and entry-point discovery."""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import polars as pl
import pytest
import yaml
from pydantic import Field, ValidationError

from qavf import QARunner, QAVFSettings, compare_datasets
from qavf.compare import (
    COMPARATORS,
    LEVELS,
    MEASURES,
    AlignmentStep,
    CheckResult,
    Comparator,
    ComparisonLevel,
    DatasetComparison,
    LevelResult,
    Measure,
    register_alignment_step,
    register_comparator,
    register_level,
    register_measure,
)
from qavf.compare.registry import ComponentRegistry, entry_point_scope, registries
from qavf.engine.compilation import ConfigurationError
from qavf.engine.registry import RuleParams
from qavf.rules._grouped import grouped_reduce


@pytest.fixture(autouse=True)
def restore_registries():
    for registry in registries().values():
        registry.discover(use_entry_points=False)
    saved = {
        name: (dict(r._components), dict(r._installed), dict(r.plugins), r._entry_points_loaded)
        for name, r in registries().items()
    }
    yield
    for name, registry in registries().items():
        components, installed, plugins, loaded = saved[name]
        registry._components, registry._installed = components, installed
        registry.plugins, registry._entry_points_loaded = plugins, loaded


class RangeParams(RuleParams):
    scale: int = 1


class Range(Measure):
    """Largest minus smallest value, times ``scale``."""

    name: ClassVar[str] = "range"
    Params: ClassVar[type[RuleParams]] = RangeParams

    def compute(self, data):
        value = pl.col(data.column)
        spread = (value.max() - value.min()) * self.params.scale
        return grouped_reduce(data.relation, data.keys, {data.alias: spread})


class RowParity(ComparisonLevel):
    """Both sides have an even (or both an odd) number of rows."""

    name: ClassVar[str] = "row_parity"

    class Result(LevelResult):
        parities: dict[str, int] = Field(default_factory=dict)

    def evaluate(self, inputs, name):
        parities = {side: inputs.rows(side) % 2 for side in ("baseline", "candidate")}
        same = len(set(parities.values())) == 1
        check = CheckResult(check="parity", passed=same, message=f"parities {parities}")
        return self.Result(level=self.name, name=name, checks=[check], parities=parities)


class Doubled(Comparator):
    name: ClassVar[str] = "doubled_diff"

    def compare(self, baseline, candidate, context):
        return 2 * (candidate - baseline)


class Upper(AlignmentStep):
    name: ClassVar[str] = "upper"

    class Params(RuleParams):
        column: str

    def apply(self, plan, context):
        return plan.with_columns(pl.col(self.params.column).str.to_uppercase())


def test_in_code_registration_extends_every_registry():
    register_measure(Range)
    register_level(RowParity)
    register_comparator(Doubled)
    register_alignment_step(Upper)
    baseline = pl.DataFrame({"g": ["a", "a"], "v": [1, 5], "s": ["x", "y"]})
    candidate = pl.DataFrame({"g": ["a", "a", "a"], "v": [1, 9, 4], "s": ["X", "Y", "Z"]})
    result = compare_datasets(
        baseline,
        candidate,
        steps=[{"step": "upper", "column": "s"}],
        levels=[
            "row_parity",
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": [
                    {
                        "measure": "range",
                        "column": "v",
                        "scale": 10,
                        "comparators": ["doubled_diff"],
                    },
                    "distinct_count(s)",
                ],
            },
        ],
    )
    parity = result.level("row_parity")
    assert isinstance(parity, RowParity.Result)
    assert parity.parities == {"baseline": 0, "candidate": 1}
    spread = next(c for c in result.level("groups").checks if c.check == "range(v)")
    (record,) = spread.affected_records
    assert (record["baseline"], record["candidate"], record["doubled_diff"]) == (40, 80, 80)
    distinct = next(c for c in result.level("groups").checks if c.check == "distinct_count(s)")
    assert distinct.affected_records[0]["baseline"] == 2  # upper-cased baseline: X, Y
    assert result.alignment == ["upper: {'column': 's'}"]
    # The plugin's typed result survives JSON transport.
    restored = DatasetComparison.model_validate_json(result.model_dump_json())
    assert restored.level("row_parity").parities == parity.parities
    with pytest.raises(ValidationError, match="invalid measure 'range'"):
        compare_datasets(
            baseline,
            candidate,
            levels=[
                {"level": "groups", "measures": [{"measure": "range", "column": "v", "scale": "x"}]}
            ],
        )


def test_registration_rejects_invalid_and_conflicting_components():
    with pytest.raises(TypeError, match="subclass"):
        register_measure(Doubled)
    with pytest.raises(TypeError, match="abstract"):
        register_measure(Measure)

    class Other(Measure):
        name: ClassVar[str] = "sum"

        def compute(self, data):
            raise AssertionError

    with pytest.raises(ValueError, match="duplicate measure 'sum'"):
        register_measure(Other)


def _entry(name, load):
    return SimpleNamespace(name=name, load=load)


def test_entry_points_load_classes_and_modules(monkeypatch):
    module = ModuleType("qavf_fake_compare_plugin")
    Range.__module__ = module.__name__
    module.Range = Range
    try:
        monkeypatch.setattr(
            "qavf.compare.registry.entry_points",
            lambda group: {
                "qavf.compare.measures": [_entry("spread", lambda: module)],
                "qavf.compare.comparators": [_entry("double", lambda: Doubled)],
            }.get(group, []),
        )
        for registry in registries().values():
            registry.refresh()
        # Installed components resolve only where plugins are enabled.
        assert "range" not in MEASURES and "doubled_diff" not in COMPARATORS
        with entry_point_scope(True):
            assert "range" in MEASURES and "doubled_diff" in COMPARATORS
        assert MEASURES.plugins == {"spread": "loaded"}
        assert COMPARATORS.plugins == {"double": "loaded"}
    finally:
        Range.__module__ = __name__


def test_broken_entry_points_are_recorded_and_leave_nothing_behind(monkeypatch):
    def half_registers():
        register_level(RowParity)
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(
        "qavf.compare.registry.entry_points",
        lambda group: (
            [
                _entry("broken", half_registers),
                _entry("twice", lambda: Doubled),
                _entry("twice", lambda: Doubled),
                _entry("notacomponent", lambda: 42),
            ]
            if group == "qavf.compare.levels"
            else []
        ),
    )
    LEVELS.refresh()
    with entry_point_scope(True):
        assert "row_parity" not in LEVELS
    assert LEVELS.plugins["broken"].startswith("error: ImportError")
    assert "duplicate entry-point name" in LEVELS.plugins["twice"]
    assert LEVELS.plugins["notacomponent"].startswith("error:")
    # Built-ins are unaffected and comparisons still run.
    frame = pl.DataFrame({"k": [1]})
    assert compare_datasets(frame, frame, keys="k").passed


def test_required_comparison_plugins_stop_the_run(monkeypatch):
    monkeypatch.setattr(
        "qavf.compare.registry.entry_points",
        lambda group: (
            [_entry("pack", lambda: (_ for _ in ()).throw(ImportError("gone")))]
            if group == "qavf.compare.measures"
            else []
        ),
    )
    for registry in registries().values():
        registry.refresh()
    for required in (["pack"], ["compare.measures:pack"]):
        settings = QAVFSettings.model_validate({"plugins": {"required": required}})
        runner = QARunner(settings)
        assert runner.plugin_status()["compare.measures:pack"].startswith("error")
        with pytest.raises(ConfigurationError, match=r"compare\.measures:pack"):
            runner.plan()
    settings = QAVFSettings.model_validate({"plugins": {"required": ["compare.levels:pack"]}})
    with pytest.raises(ConfigurationError, match="not installed"):
        QARunner(settings).plan()


def test_generic_registry_is_reusable_for_new_component_kinds():
    class Base(ComparisonLevel):
        pass

    registry: ComponentRegistry = ComponentRegistry("probe", "qavf.compare.probe", Base, ())
    registry.discover(use_entry_points=False)
    assert len(registry) == 0
    with pytest.raises(ValueError, match="unknown probe 'x'; available: none"):
        registry.create("x")


DOCS = Path(__file__).resolve().parents[1] / "docs"


def _blocks(path, language, section=None):
    text = (DOCS / path).read_text(encoding="utf-8")
    if section is not None:
        start = text.index(f"## {section}\n")
        end = text.find("\n## ", start + 1)
        text = text[start : end if end != -1 else None]
    return re.findall(rf"```{language}\n(.*?)```", text, flags=re.DOTALL)


def test_documented_plugin_example_runs():
    code = _blocks("plugins.md", "python", "Comparison components")[0]
    configuration = _blocks("plugins.md", "yaml", "Comparison components")[0]
    namespace: dict = {"__name__": "qavf_documented_plugin"}
    exec(compile(code, "plugins.md", "exec"), namespace)
    levels = yaml.safe_load(configuration)["levels"]
    frame = pl.DataFrame({"county": ["a", "a"], "aadt": [1, 4]})
    result = compare_datasets(frame, frame.with_columns(aadt=pl.Series([1, 9])), levels=levels)
    assert result.level("row_parity").passed
    (check,) = [c for c in result.level("groups").checks if c.check == "range(aadt)"]
    record = check.affected_records[0]
    assert (record["baseline"], record["candidate"]) == (6, 16)


def test_documented_comparison_configuration_validates():
    from qavf.rules.comparison_checks import DatasetComparisonRule

    (configuration,) = _blocks("comparison.md", "yaml")
    settings = QAVFSettings.model_validate(yaml.safe_load(configuration))
    rule = DatasetComparisonRule(settings.rules["CMP-001"].params)
    assert rule.configuration_errors(settings) == []
