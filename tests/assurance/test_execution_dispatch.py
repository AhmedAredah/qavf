"""Paired checks keep their backend contract while another consumer fills the cache."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from qavf import Query, Status
from qavf.engine import observation as observation_module
from qavf.engine.context import DataContext

from .support.contracts import run_checks

pytestmark = pytest.mark.assurance


def _materialize_during_dispatch(monkeypatch, transition: str) -> list[str]:
    """Inject a valid cache transition at an exact interleaving, without sleeps."""
    observed = []
    if transition.startswith("dispatch-"):
        dataset = transition.removeprefix("dispatch-")
        original = DataContext.execution_kind

        def execution_kind(self, name):
            kind = original(self, name)
            if name == dataset and kind == "native_sql" and not observed:
                # Another consumer can finish materializing after this read but
                # before the paired operation constructs either relation.
                self.get(name)
                observed.append(name)
            return kind

        monkeypatch.setattr(DataContext, "execution_kind", execution_kind)
    else:
        original_observe = observation_module.observe

        @contextmanager
        def observe(self, name, *args, **kwargs):
            with original_observe(self, name, *args, **kwargs) as observation:
                if name == "child" and not observed:
                    # The left observation exists; another consumer fills the
                    # right cache before its coherent observation is captured.
                    self.get("parent")
                    observed.append("parent")
                yield observation

        monkeypatch.setattr(observation_module, "observe", observe)
    return observed


@pytest.mark.parametrize("rule_id", ["INP-005", "REC-004"])
@pytest.mark.parametrize("transition", ["dispatch-child", "dispatch-parent", "between-relations"])
@pytest.mark.parametrize("different", [False, True])
def test_paired_verdicts_survive_materialization_transition(
    monkeypatch, rule_id, transition, different
):
    last = 4 if different else 3
    inputs = {
        "child": Query(f"SELECT * FROM (VALUES (1), (2), ({last})) AS values_table(id)"),
        "parent": Query("SELECT * FROM (VALUES (1), (2), (3)) AS values_table(id)"),
    }
    datasets = {"child": {}, "parent": {}}
    params = {}
    if rule_id == "INP-005":
        datasets["child"]["foreign_keys"] = [{"columns": ["id"], "ref_dataset": "parent"}]
    else:
        params["pairs"] = [{
            "left": {"dataset": "child", "columns": ["id"]},
            "right": {"dataset": "parent", "columns": ["id"]},
        }]
    observed = _materialize_during_dispatch(monkeypatch, transition)
    results = run_checks(inputs, rule_id, datasets=datasets, params=params)
    assert len(observed) == 1, "the materialization interleaving must actually be exercised"
    assert len(results) == 1
    assert results[0].status is (Status.FAIL if different else Status.PASS)
    if different:
        assert results[0].affected_count == (1 if rule_id == "INP-005" else 2)
