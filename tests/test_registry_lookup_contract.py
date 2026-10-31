"""Keyed registry reads preserve transactional and visibility contracts."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from qavf.engine.registry import RuleParams as HistoricalParams
from qavf.parameters import RuleParams
from qavf.plugins.registry import Component, ComponentRegistry
from qavf.plugins.transactions import LOCK, mapping, transaction, visibility


class Original(Component):
    name = "same"


class Installed(Component):
    name = "same"


class Extra(Component):
    name = "extra"


def registry():
    result = ComponentRegistry("test", "qavf.test.lookup", Component, ())
    result.register(Original)
    with LOCK:
        result._installed = {"same": Installed, "extra": Extra}
    return result


def test_neutral_parameters_preserve_historical_identity():
    assert HistoricalParams is RuleParams


def test_direct_reads_preserve_visibility_and_shadowing():
    entries = registry()
    with visibility(False):
        assert entries.get("same") is Original
        assert "extra" not in entries
        assert len(entries) == 1
        with pytest.raises(KeyError, match="plugins are disabled"):
            entries.get("extra")
    with visibility(True):
        assert entries.get("same") is Original
        assert entries.get("extra") is Extra
        assert len(entries) == 2
        assert entries.names() == ["extra", "same"]


def test_staged_lookup_is_private_and_rolls_back():
    entries = registry()
    with pytest.raises(RuntimeError, match="rollback"), transaction():
        with LOCK:
            mapping(entries, "_components")["same"] = Installed
            mapping(entries, "_components")["extra"] = Extra
        assert entries.get("same") is Installed
        assert "extra" in entries and len(entries) == 2
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(entries.get, "same").result() is Original
        raise RuntimeError("rollback")
    assert entries.get("same") is Original
    assert "extra" not in entries
    assert len(entries) == 1


def test_iteration_remains_a_snapshot_after_registration():
    entries = registry()
    before = iter(entries)
    entries.register(Extra)
    assert list(before) == [Original]
    assert entries.names() == ["extra", "same"]
