"""Primary keys, parallel stability and failure isolation.

Parallel stability on pinned dependencies, primary-key contracts,
all-missing observations and run-wide failure isolation.
"""

from __future__ import annotations

import sys
import textwrap
from typing import ClassVar

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf.demo import write_demo_project
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings, load_settings
from qavf.testing import make_settings, rule_results, run_rule


# =========================================================================
# Parallel execution is stable
# =========================================================================
def test_repeated_parallel_runs_of_the_clean_demo_have_no_errors(tmp_path) -> None:
    settings = load_settings(write_demo_project(tmp_path, clean=True))
    settings.engine.parallel, settings.engine.max_workers = True, 8
    for _ in range(3):
        report = QARunner(settings).run()
        assert [r.label for r in report.results if r.status is Status.ERROR] == []
        assert report.verdict() == "passed"


# =========================================================================
# Primary keys are enforced as declared
# =========================================================================
def _keyed(frame: pl.DataFrame, primary_key: list[str], columns: dict | None = None) -> tuple:
    datasets = {"t": {"schema": {"primary_key": primary_key, "columns": columns or {}}}}
    return {"t": frame}, datasets


def test_missing_primary_key_column_fails(registry) -> None:
    frames, datasets = _keyed(pl.DataFrame({"x": [1]}), ["id"], {"x": {}})
    result = run_rule("INP-004", frames, datasets=datasets, registry=registry)
    assert result.status is Status.FAIL and "id" in result.message
    report = QARunner(make_settings(datasets=datasets), registry).run(frames=frames)
    assert report.exit_code() == 1


def test_null_primary_key_values_fail(registry) -> None:
    frames, datasets = _keyed(pl.DataFrame({"id": [1, None], "y": ["a", None]}), ["id"])
    result = run_rule("INP-004", frames, datasets=datasets, registry=registry)
    assert result.status is Status.FAIL and "null" in result.message
    composite = pl.DataFrame({"a": [1, 1, None], "b": [None, 2, 3]})
    frames, datasets = _keyed(composite, ["a", "b"])
    result = run_rule("INP-004", frames, datasets=datasets, registry=registry)
    assert result.status is Status.FAIL and result.affected_count == 2


def test_contradictory_primary_key_declarations_are_rejected() -> None:
    for schema, match in [
        ({"primary_key": ["id", "id"]}, "repeats"),
        ({"primary_key": ["id"], "columns": {"id": {"nullable": True}}}, "nullable"),
        ({"primary_key": ["id"], "columns": {"id": {"required": False}}}, "required"),
    ]:
        with pytest.raises(ValidationError, match=match):
            QAVFSettings.model_validate({"datasets": {"t": {"schema": schema}}})


# =========================================================================
# No usable value is not zero
# =========================================================================
def test_all_null_period_is_a_missing_period(registry) -> None:
    frame = pl.DataFrame({"year": [2020, 2021], "g": ["a", "a"], "v": [1.0, None]})
    params = {
        "series": [{"dataset": "t", "value": "v", "group_by": ["g"]}],
        "expected_periods": [2020, 2021],
    }
    result = run_rule("TMP-004", {"t": frame}, params, registry=registry)
    assert result.status is Status.FAIL
    assert result.affected_records[0]["missing_periods"] == [2021]


def test_all_null_period_is_not_treated_as_a_drop_to_zero(registry) -> None:
    frame = pl.DataFrame({"year": [2020, 2021, 2022], "v": [100.0, None, 101.0]})
    results = rule_results(
        "TMP-001", {"t": frame}, {"series": [{"dataset": "t", "value": "v"}]}, registry=registry
    )
    assert not any(r.status is Status.FAIL for r in results)  # no fake 100 -> 0 -> 101
    assert any(r.status is Status.ERROR and "missing" in r.message for r in results)


def test_entity_without_any_value_is_reported_not_dropped(registry) -> None:
    frame = pl.DataFrame({"e": list("abcdef"), "v": [10.0, 10.1, 9.9, 10.2, 9.8, None]})
    params = {"metrics": [{"dataset": "t", "value": "v", "entity_column": "e"}]}
    for rule_id in ("SPA-001", "SPA-002"):
        results = rule_results(rule_id, {"t": frame}, params, registry=registry)
        (blocked,) = [r for r in results if r.status is Status.ERROR]
        assert [r["e"] for r in blocked.affected_records] == ["f"]


def test_total_keeps_partial_sums_and_marks_empty_groups() -> None:
    from qavf.rules.helpers import total

    frame = pl.DataFrame({"g": ["a", "a", "b"], "v": [1.0, None, None]})
    summed = total(frame, ["g"], "v").sort("g")
    assert summed["v"].to_list() == [1.0, None]


# =========================================================================
# One failure never aborts the run
# =========================================================================
def test_source_conversion_errors_become_load_failures(tmp_path, monkeypatch) -> None:
    (tmp_path / "bad_loaders.py").write_text(
        textwrap.dedent(
            """
            import pandas as pd
            import polars as pl

            def mixed():
                return pd.DataFrame({"x": [1, "bad"]})

            def lazy():
                return pl.LazyFrame({"a": [1]}).select(pl.col("missing"))
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    datasets = {
        "mixed": {"source": {"type": "python", "callable": "bad_loaders:mixed"}},
        "lazy": {"source": {"type": "python", "callable": "bad_loaders:lazy"}},
    }
    settings = make_settings(datasets=datasets)
    settings.plugins.entry_points = False
    report = QARunner(settings).run()  # must not raise
    failures = {
        r.check for r in report.results if r.rule_id == "INP-001" and r.status is Status.FAIL
    }
    assert failures == {"mixed", "lazy"}
    sys.modules.pop("bad_loaders", None)


def test_rename_collisions_are_reported_not_crashes(tmp_path) -> None:
    # In memory: a clear configuration error when the context is built.
    settings = make_settings(datasets={"t": {"rename": {"x": "y"}}})
    with pytest.raises(ValueError, match="rename"):
        DataContext(frames={"t": pl.DataFrame({"x": [1], "y": [2]})}, specs=settings.datasets)
    # From a source: a load failure reported by INP-001, while the run completes.
    pl.DataFrame({"x": [1], "y": [2]}).write_csv(tmp_path / "t.csv")
    settings = make_settings(
        datasets={"t": {"path": str(tmp_path / "t.csv"), "rename": {"x": "y"}}}
    )
    settings.plugins.entry_points = False
    report = QARunner(settings).run()
    failed = report.get("INP-001", check="t")
    assert failed.status is Status.FAIL and "rename" in failed.message


@pytest.mark.parametrize(
    "source",
    [
        {"type": "query", "sql": "   "},
        {"type": "file", "path": ""},
        {"type": "python", "callable": " "},
        {"type": "database", "connection": "db", "table": ""},
    ],
)
def test_blank_source_parameters_are_rejected(source) -> None:
    from qavf.sources import SourceEnvironment, SourceRegistry

    with pytest.raises(ValueError):
        SourceRegistry().discover(use_entry_points=False).build(source, SourceEnvironment())


class _ZeroDimMetric(Rule):
    rule_id: ClassVar[str] = "T-ZERO-D"
    name: ClassVar[str] = "Zero-dimensional metric"
    description: ClassVar[str] = "Returns a 0-D NumPy array as a metric."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.ok("fine", metrics={"value": np.array(7.0)})


class _BadEvidence(Rule):
    rule_id: ClassVar[str] = "T-BAD-EVIDENCE"
    name: ClassVar[str] = "Unserialisable evidence"
    description: ClassVar[str] = "Returns evidence the engine cannot convert."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.fail("bad", affected=[{"x": object()}], metrics={"m": object()})


class _Fine(Rule):
    rule_id: ClassVar[str] = "T-FINE"
    name: ClassVar[str] = "Fine"
    description: ClassVar[str] = "Always passes."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx: DataContext) -> Finding:
        return Finding.ok("fine")


def test_result_conversion_errors_are_isolated_per_rule() -> None:
    registry = RuleRegistry()
    for rule in (_ZeroDimMetric, _BadEvidence, _Fine):
        registry.register(rule)
    report = QARunner(make_settings(), registry).run(frames={})
    assert report.get("T-ZERO-D").status is Status.PASS
    assert report.get("T-ZERO-D").metrics == {"value": 7.0}
    assert report.get("T-FINE").status is Status.PASS
    bad = report.get("T-BAD-EVIDENCE")
    # Optional unsupported evidence cannot erase an already assessed failure.
    assert bad.status is Status.FAIL and "truncated or omitted" in bad.message
    assert bad.affected_records == [] and bad.metrics == {}
    report.model_dump_json()  # the whole report still serialises
