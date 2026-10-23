"""Built-in provenance avoids an unrelated installed-package scan."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import PurePosixPath
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from qavf.engine import runner
from qavf.engine.registry import Rule
from qavf.rules.input_checks.expressions import ExpressionRule
from qavf.rules.reconciliation_checks import TotalsReconcileRule


@pytest.fixture(autouse=True)
def clear_provenance_caches():
    caches = (runner._builtin_distribution, runner._distribution, runner._packages_distributions)
    for function in caches:
        function.cache_clear()
    yield
    for function in caches:
        function.cache_clear()


def _metadata(*, declared=None, files=("qavf/__init__.py",), version="9.8.7"):
    return SimpleNamespace(
        metadata={"Name": "qavf"},
        version=version,
        files=[PurePosixPath(path) for path in files],
        read_text=lambda name: declared if name == "top_level.txt" else None,
    )


def _unexpected_scan():
    pytest.fail("verified built-ins must not scan unrelated distributions")


@pytest.mark.parametrize("declared", [None, "qavf\n"])
def test_builtin_wheel_records_installed_version_without_global_scan(monkeypatch, declared):
    lookups = []

    def distribution(name):
        lookups.append(name)
        return _metadata(declared=declared)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(importlib.metadata, "packages_distributions", _unexpected_scan)
    assert runner._rule_info(TotalsReconcileRule()).distribution == "qavf==9.8.7"
    assert runner._rule_info(ExpressionRule()).distribution == "qavf==9.8.7"
    assert lookups == ["qavf"]


@pytest.mark.parametrize(
    ("declared", "files"),
    [
        (None, ("_editable_impl_qavf.pth", "qavf-9.8.7.dist-info/METADATA")),
        (None, ()),
        ("some_other_package", ("qavf/__init__.py",)),
    ],
)
def test_canonical_metadata_without_package_claim_preserves_unknown_provenance(
    monkeypatch, declared, files
):
    monkeypatch.setattr(
        importlib.metadata, "distribution", lambda name: _metadata(declared=declared, files=files)
    )
    monkeypatch.setattr(importlib.metadata, "packages_distributions", _unexpected_scan)
    assert runner._rule_info(TotalsReconcileRule()).distribution is None


@pytest.mark.parametrize("providers", [[], ["alternate-qavf"]])
def test_source_checkout_without_canonical_metadata_keeps_generic_fallback(
    monkeypatch, providers
):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    scans = []

    def packages():
        scans.append(True)
        return {"qavf": providers}

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    monkeypatch.setattr(importlib.metadata, "packages_distributions", packages)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.3.4")
    expected = "alternate-qavf==2.3.4" if providers else None
    assert runner._rule_info(TotalsReconcileRule()).distribution == expected
    assert scans == [True]


def test_declared_package_claim_does_not_enumerate_distribution_files(monkeypatch):
    class Metadata:
        metadata: ClassVar[dict[str, str]] = {"Name": "qavf"}
        version = "9.8.7"

        def read_text(self, name):
            return "qavf\n"

        @property
        def files(self):
            pytest.fail("declared top-level metadata takes precedence")

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Metadata())
    monkeypatch.setattr(importlib.metadata, "packages_distributions", _unexpected_scan)
    assert runner._rule_info(TotalsReconcileRule()).distribution == "qavf==9.8.7"


def test_unreadable_canonical_metadata_retains_optional_generic_lookup(monkeypatch):
    def unreadable(name):
        raise OSError("metadata cannot be read")

    monkeypatch.setattr(importlib.metadata, "distribution", unreadable)
    monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: {})
    assert runner._rule_info(TotalsReconcileRule()).distribution is None


def test_matching_module_prefix_without_actual_export_is_not_a_builtin(monkeypatch):
    forged = type("ForgedRule", (Rule,), {"__module__": TotalsReconcileRule.__module__})
    monkeypatch.setattr(runner, "_distribution", lambda name: "plugin-provider==1")
    monkeypatch.setattr(runner, "_builtin_distribution", _unexpected_scan)
    assert runner._rule_distribution(forged) == "plugin-provider==1"


@pytest.mark.parametrize("origin", [None, "/external/plugins/shadow.py"])
def test_exported_shadow_module_outside_package_is_not_a_builtin(monkeypatch, origin):
    name = "qavf.rules.shadow"
    module = ModuleType(name)
    module.__spec__ = SimpleNamespace(origin=origin)
    shadow = type("ShadowRule", (Rule,), {"__module__": name})
    module.ShadowRule = shadow
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runner, "_distribution", lambda name: "shadow-provider==1")
    monkeypatch.setattr(runner, "_builtin_distribution", _unexpected_scan)
    assert runner._rule_distribution(shadow) == "shadow-provider==1"


def test_plugin_namespace_keeps_first_available_distribution(monkeypatch):
    plugin = type("PluginRule", (Rule,), {"__module__": "shared_namespace.rules"})
    versions = []

    def version(name):
        versions.append(name)
        if name == "uninstalled-provider":
            raise importlib.metadata.PackageNotFoundError(name)
        return "4.5.6"

    monkeypatch.setattr(runner, "_builtin_distribution", _unexpected_scan)
    monkeypatch.setattr(
        importlib.metadata,
        "packages_distributions",
        lambda: {"shared_namespace": ["uninstalled-provider", "second-provider"]},
    )
    monkeypatch.setattr(importlib.metadata, "version", version)
    assert runner._rule_distribution(plugin) == "second-provider==4.5.6"
    assert versions == ["uninstalled-provider", "second-provider"]


def test_builtin_subclass_defined_by_plugin_keeps_plugin_provenance(monkeypatch):
    plugin = type("SpecializedReconciliation", (TotalsReconcileRule,), {"__module__": "plugin"})
    monkeypatch.setattr(runner, "_builtin_distribution", _unexpected_scan)
    monkeypatch.setattr(runner, "_distribution", lambda name: f"{name}-distribution==1")
    assert runner._rule_distribution(plugin) == "plugin-distribution==1"


@pytest.mark.parametrize("claims_package", [True, False])
def test_verified_builtins_use_canonical_policy_when_another_distribution_shares_namespace(
    monkeypatch, claims_package
):
    files = ("qavf/__init__.py",) if claims_package else ("_editable_impl_qavf.pth",)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _metadata(files=files))
    monkeypatch.setattr(
        importlib.metadata,
        "packages_distributions",
        lambda: {"qavf": ["alternate-qavf", "qavf"]},
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.3.4")
    assert runner._distribution(TotalsReconcileRule.__module__) == "alternate-qavf==2.3.4"
    expected = "qavf==9.8.7" if claims_package else None
    assert runner._rule_info(TotalsReconcileRule()).distribution == expected
