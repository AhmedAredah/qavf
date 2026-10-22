"""``plugins.entry_points`` gates installed comparison plugins exactly like rules.

A temporary distribution on ``sys.path`` advertises a comparison level under
``qavf.compare.levels``; isolated workers inherit the path, so the same plugin
is visible to them. Importing the plugin module appends its process id to a
marker file, so each test can prove whether (and in which process) plugin
code ran.
"""

from __future__ import annotations

import os
import sys

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import (
    ExecutionLimitError,
    File,
    QARunner,
    QAVFSettings,
    ResourceLimits,
    Status,
    compare_datasets,
    validate,
)
from qavf.compare.registry import registries
from qavf.engine.compilation import ConfigurationError
from qavf.execution.os_limits import SUPPORTED_PLATFORMS

MODULE = "qavf_gating_fake_level"
PLUGIN = '''
import os
from typing import ClassVar

from qavf.compare import CheckResult, ComparisonLevel, LevelResult

with open(os.environ["QAVF_GATING_MARKER"], "a", encoding="utf-8") as marker:
    marker.write(f"{os.getpid()}\\n")


class FakeParity(ComparisonLevel):
    """Passes when both sides have rows of the same parity."""

    name: ClassVar[str] = "fake_parity"

    def evaluate(self, inputs, name):
        same = inputs.rows("baseline") % 2 == inputs.rows("candidate") % 2
        check = CheckResult(check="parity", passed=same, message="parity")
        return LevelResult(level=self.name, name=name, checks=[check])
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Install the fake distribution for this test only; return the marker path."""
    site = tmp_path / "site"
    info = site / "qavf_gating_fake-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: qavf-gating-fake\nVersion: 1.0\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(
        f"[qavf.compare.levels]\nfake = {MODULE}:FakeParity\n", encoding="utf-8"
    )
    (site / f"{MODULE}.py").write_text(PLUGIN, encoding="utf-8")
    marker = tmp_path / "imported.txt"
    monkeypatch.setenv("QAVF_GATING_MARKER", str(marker))
    monkeypatch.syspath_prepend(str(site))
    saved = {
        name: (dict(r._installed), dict(r.plugins), r._entry_points_loaded)
        for name, r in registries().items()
    }
    for registry in registries().values():
        registry.discover()
        registry._installed, registry.plugins = {}, {}
        registry._entry_points_loaded = False
    sys.modules.pop(MODULE, None)
    yield marker
    sys.modules.pop(MODULE, None)
    for name, registry in registries().items():
        registry._installed, registry.plugins, registry._entry_points_loaded = saved[name]


def _imported(marker) -> list[int]:
    return [int(line) for line in marker.read_text().split()] if marker.exists() else []


FRAME = pl.DataFrame({"k": [1, 2], "v": [1, 2]})


def _config(entry_points: bool, **plugins):
    return {
        "rules": {
            "CMP-001": {
                "params": {
                    "comparisons": [{"baseline": "a", "candidate": "b", "levels": ["fake_parity"]}]
                }
            }
        },
        "plugins": {"entry_points": entry_points, **plugins},
        "engine": {"run_unconfigured_rules": False},
    }


def test_disabled_plugins_are_never_imported_by_compare_datasets(plugin):
    with pytest.raises(ValidationError, match="unknown comparison level 'fake_parity'"):
        compare_datasets(FRAME, FRAME, levels=["fake_parity"], entry_points=False)
    # Built-in lookups (and their entire comparison) import no plugin code either.
    assert compare_datasets(FRAME, FRAME, keys="k", entry_points=False).passed
    temporal = [{"level": "temporal", "time_column": "k", "measures": ["sum(v)"]}]
    assert compare_datasets(FRAME, FRAME, levels=temporal, entry_points=False).passed
    assert MODULE not in sys.modules and _imported(plugin) == []


def test_enabled_plugins_resolve_in_compare_datasets(plugin):
    result = compare_datasets(FRAME, FRAME, levels=["fake_parity"])
    assert result.level("fake_parity").passed
    assert MODULE in sys.modules and _imported(plugin) == [os.getpid()]
    # Visibility is scoped: a later disabled call does not see the loaded plugin.
    with pytest.raises(ValidationError, match="plugins are disabled"):
        compare_datasets(FRAME, FRAME, levels=["fake_parity"], entry_points=False)


def test_cmp001_obeys_plugins_entry_points(plugin):
    with pytest.raises(ConfigurationError, match="unknown comparison level 'fake_parity'"):
        validate({"a": FRAME, "b": FRAME}, config=_config(False))
    report = validate(
        {"a": FRAME, "b": FRAME},
        config=_config(False, required=[])
        | {
            "rules": {
                "CMP-001": {
                    "params": {"comparisons": [{"baseline": "a", "candidate": "b", "keys": ["k"]}]}
                }
            }
        },
    )
    assert report.verdict() == "passed"
    assert not any(key.startswith("compare.") for key in report.metadata.plugins)
    assert MODULE not in sys.modules and _imported(plugin) == []

    report = validate({"a": FRAME, "b": FRAME}, config=_config(True))
    (result,) = [r for r in report.results if r.rule_id == "CMP-001"]
    assert result.status is Status.PASS and result.check == "a vs b: fake_parity: parity"
    assert report.metadata.plugins["compare.levels:fake"] == "loaded"
    assert _imported(plugin) == [os.getpid()]


def test_required_comparison_plugin_respects_entry_points(plugin):
    required = {"required": ["compare.levels:fake"]}
    settings = QAVFSettings.model_validate({"plugins": {"entry_points": False, **required}})
    with pytest.raises(ConfigurationError, match="not installed"):
        QARunner(settings).plan()
    assert _imported(plugin) == []
    settings = QAVFSettings.model_validate({"plugins": {"entry_points": True, **required}})
    QARunner(settings).plan()
    assert _imported(plugin) == [os.getpid()]


@pytest.mark.skipif(sys.platform not in SUPPORTED_PLATFORMS, reason="no isolated execution")
def test_isolated_workers_honour_the_same_choice(plugin, tmp_path):
    path = tmp_path / "data.parquet"
    FRAME.write_parquet(path)
    limits = ResourceLimits(isolate=True, wall_time=120)
    disabled = compare_datasets(
        File(path), File(path), keys="k", resources=limits, entry_points=False
    )
    assert disabled.passed and disabled.execution_limits["execution_mode"] == "isolated"
    assert _imported(plugin) == []  # neither the parent nor the worker imported it

    enabled = compare_datasets(File(path), File(path), levels=["fake_parity"], resources=limits)
    assert enabled.level("fake_parity").passed
    processes = _imported(plugin)
    assert os.getpid() in processes  # the parent validated the configuration
    assert any(pid != os.getpid() for pid in processes)  # the worker re-validated it

    config = _config(True)
    config["datasets"] = {"a": {"path": str(path)}, "b": {"path": str(path)}}
    report = validate(config=config, resources=limits)
    (result,) = [r for r in report.results if r.rule_id == "CMP-001"]
    assert result.status is Status.PASS
    before = len(_imported(plugin))
    config["plugins"]["entry_points"] = False
    # The worker compiles the configuration; its configuration error is reported
    # through the isolation boundary, and no process imported the plugin.
    with pytest.raises(ExecutionLimitError, match="unknown comparison level 'fake_parity'"):
        validate(config=config, resources=limits)
    assert len(_imported(plugin)) == before
