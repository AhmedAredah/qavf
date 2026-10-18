"""Report anchors, counts and provenance.

Unique report anchors, count units and row impact, complete
run provenance in JSON, unambiguous canonical columns and metadata
that never loads data.
"""

from __future__ import annotations

import json
import re
import time
from typing import ClassVar

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import ConfigurationError, QARunner
from qavf.reports.html import render_html
from qavf.reports.models import QAReport
from qavf.settings import QAVFSettings
from qavf.sources.base import DataSource, SourceEnvironment, SourceParams


def _settings(**config) -> QAVFSettings:
    return QAVFSettings.model_validate({"plugins": {"entry_points": False}, **config})


def _run(registry, frames, **config) -> QAReport:
    return QARunner(_settings(**config), registry).run(frames=frames)


# =========================================================================
# Every result has its own HTML anchor
# =========================================================================
def test_report_anchors_are_unique(registry) -> None:
    checks = [
        {"name": "ge 0", "ge": 0},
        {"name": "ge -0", "ge": 0},
        {"name": "ge (0)", "ge": 0},
    ]
    report = _run(
        registry,
        {"a": pl.DataFrame({"x": [-1]})},
        datasets={"a": {"schema": {"columns": {"x": {"checks": checks}}}}},
    )
    page = render_html(report)
    ids = re.findall(r'\bid="([^"]+)"', page)
    assert len(ids) == len(set(ids))
    finding_ids = re.findall(r'<article class="card finding" id="([^"]+)"', page)
    assert len(finding_ids) == 3
    # Every link in the results table points at an existing finding.
    links = re.findall(r'href="#([^"]+)"', page)
    assert set(links) <= set(ids) and len(set(links)) == 3


def test_anchors_are_stable_between_renders(registry) -> None:
    report = _run(
        registry,
        {"a": pl.DataFrame({"x": [-1]})},
        datasets={
            "a": {
                "schema": {
                    "columns": {
                        "x": {"checks": [{"name": "ge 0", "ge": 0}, {"name": "ge -0", "ge": 0}]}
                    }
                }
            }
        },
    )
    first = re.findall(r'<article class="card finding" id="([^"]+)"', render_html(report))
    second = re.findall(r'<article class="card finding" id="([^"]+)"', render_html(report))
    assert first == second


# =========================================================================
# Counts state their unit; row impact is separate
# =========================================================================
def test_duplicate_keys_count_keys_and_report_rows_involved(registry) -> None:
    report = _run(
        registry,
        {"k": pl.DataFrame({"id": [1, 1, 1, 2]})},
        datasets={"k": {"schema": {"primary_key": ["id"], "columns": {"id": {}}}}},
    )
    result = report.get("INP-004")
    assert (result.unit, result.affected_count, result.total_records) == ("keys", 1, 2)
    assert result.affected_rows == 3
    assert result.affected_fraction == pytest.approx(0.5)


def test_column_level_problems_have_no_row_denominator(registry) -> None:
    limits = {"min": {"ge": 10}, "max": {"le": 0}, "mean": {"ge": 10}, "sum": {"ge": 10}}
    report = _run(
        registry,
        {"g": pl.DataFrame({"v": [5]})},
        datasets={"g": {"schema": {"columns": {"v": limits}}}},
    )
    [result] = [r for r in report.results if r.failed]
    assert result.unit == "problems" and result.affected_count == 4
    assert result.total_records is None and result.affected_fraction is None


def test_row_level_violations_count_rows(registry) -> None:
    report = _run(
        registry,
        {"g": pl.DataFrame({"v": [-1, 2, -3, 4]})},
        datasets={"g": {"schema": {"columns": {"v": {"ge": 0}}}}},
    )
    [result] = [r for r in report.results if r.failed]
    assert (result.unit, result.affected_count, result.total_records) == ("rows", 2, 4)
    assert result.affected_rows == 2 and result.affected_fraction == pytest.approx(0.5)


def test_orphaned_keys_count_keys_and_rows(registry) -> None:
    report = _run(
        registry,
        {
            "parent": pl.DataFrame({"id": [1]}),
            "child": pl.DataFrame({"pid": [1, 2, 2, 3]}),
        },
        datasets={
            "parent": {},
            "child": {
                "foreign_keys": [
                    {"columns": ["pid"], "ref_dataset": "parent", "ref_columns": ["id"]}
                ]
            },
        },
    )
    result = report.get("INP-005")
    assert (result.unit, result.affected_count, result.total_records) == ("keys", 2, 3)
    assert result.affected_rows == 3


def test_built_in_results_with_a_count_declare_a_unit(tmp_path, registry) -> None:
    from qavf.demo import write_demo_project
    from qavf.settings import load_settings

    report = QARunner(load_settings(write_demo_project(tmp_path)), registry).run()
    counted = [r for r in report.results if r.affected_count or r.total_records]
    assert counted and all(r.unit for r in counted), [r.label for r in counted if not r.unit]
    fractions = [r.affected_fraction for r in counted if r.affected_fraction is not None]
    assert all(0 <= f <= 1 for f in fractions)


def test_json_round_trip_keeps_units(registry) -> None:
    report = _run(
        registry,
        {"k": pl.DataFrame({"id": [1, 1, 2]})},
        datasets={"k": {"schema": {"primary_key": ["id"], "columns": {"id": {}}}}},
    )
    restored = QAReport.model_validate_json(report.model_dump_json())
    result = restored.get("INP-004")
    assert (result.unit, result.affected_rows) == ("keys", 2)


# =========================================================================
# JSON reports carry the acceptance policy, verdict and configuration
# =========================================================================
def test_json_report_records_verdict_and_policy(registry) -> None:
    report = _run(
        registry,
        {"g": pl.DataFrame({"v": [-1]})},
        datasets={"g": {"schema": {"columns": {"v": {"ge": 0}}}}},
        engine={"fail_on": "critical"},
    )
    data = json.loads(report.model_dump_json())
    acceptance = data["acceptance"]
    assert acceptance["verdict"] == report.verdict() == "passed"
    assert acceptance["exit_code"] == report.exit_code() == 0
    assert acceptance["fail_on"] == "CRITICAL" or acceptance["fail_on"].lower() == "critical"
    assert acceptance["fail_on_incomplete"] is True


def test_json_report_records_effective_configuration_and_rules(registry) -> None:
    report = _run(
        registry,
        {"g": pl.DataFrame({"v": [1]})},
        datasets={"g": {"schema": {"columns": {"v": {"ge": 0}}}}},
        rules={"STA-001": {"enabled": False}},
    )
    metadata = json.loads(report.model_dump_json())["metadata"]
    config = metadata["config"]
    assert config["datasets"]["g"]["schema"]["columns"]["v"]["ge"] == 0
    assert config["rules"]["STA-001"]["enabled"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", metadata["config_digest"])
    rules = {rule["rule_id"]: rule for rule in metadata["rules"]}
    assert "STA-001" not in rules and "INP-003" in rules
    assert rules["INP-003"]["severity"] and "params" in rules["INP-003"]
    assert rules["INP-003"]["module"].startswith("qavf.rules")
    environment = metadata["environment"]
    assert {"python", "platform", "polars", "duckdb", "qavf"} <= set(environment)


def test_configuration_digest_is_deterministic_and_sensitive(registry) -> None:
    def digest(limit) -> str:
        return _run(
            registry,
            {"g": pl.DataFrame({"v": [1]})},
            datasets={"g": {"schema": {"columns": {"v": {"ge": limit}}}}},
        ).metadata.config_digest

    assert digest(0) == digest(0) != digest(1)


def test_recorded_configuration_is_redacted(registry) -> None:
    settings = _settings(
        connections={
            "pg": {"type": "postgres", "uri": "host=db user=qa password=hunter2 dbname=x"},
            "url": {"type": "mysql", "uri": "mysql://qa:s3cret@db:3306/x"},
        },
        datasets={
            "remote": {
                "required": False,
                "source": {
                    "type": "file",
                    "path": "https://example.com/d.csv?X-Amz-Signature=abc123&token=zzz&a=1",
                },
            }
        },
    )
    # Recording the configuration needs no data: run nothing, so nothing is fetched.
    # (Secrets inside load-error messages are covered with source provenance.)
    settings.engine.run_unconfigured_rules = False
    report = QARunner(settings, registry).run(frames={})
    text = report.model_dump_json()
    for secret in ("hunter2", "s3cret", "abc123", "zzz"):
        assert secret not in text, secret
    assert "a=1" in text and "user=qa" in text


def test_datasets_record_types_renames_and_file_fingerprints(tmp_path, registry) -> None:
    path = tmp_path / "d.csv"
    pl.DataFrame({"Old": [1, 2]}).write_csv(path)
    settings = _settings(
        datasets={"d": {"path": str(path), "rename": {"Old": "new"}}},
        report={"hash_inputs": True},
    )
    report = QARunner(settings, registry).run()
    [info] = report.metadata.datasets
    assert info.dtypes == {"new": "Int64"} and info.renamed == {"Old": "new"}
    fingerprint = info.fingerprint or {}
    assert fingerprint["bytes"] == path.stat().st_size and "modified" in fingerprint
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint["sha256"])


# =========================================================================
# Canonical column names must be unambiguous
# =========================================================================
CANONICAL = {
    "engine": {"canonical_columns": ["Value"]},
    "datasets": {"t": {"schema": {"columns": {"Value": {"le": 10}}}}},
}


@pytest.mark.parametrize("order", [["VALUE", "value"], ["value", "VALUE"], ["Value", "value"]])
def test_ambiguous_canonical_columns_are_rejected(registry, order) -> None:
    frame = pl.DataFrame({"VALUE": [1], "value": [999], "Value": [5]}).select(order)
    with pytest.raises(ConfigurationError, match="Value"):
        _run(registry, {"t": frame}, **CANONICAL)


class _FrameParams(SourceParams):
    type: str = "frame"


class _FrameSource(DataSource):
    """A source (like a Python callable or plugin) that keeps column names as given.
    DuckDB file readers de-duplicate case-insensitive names themselves (``value_1``)."""

    kind: ClassVar[str] = "frame"
    Params: ClassVar[type[SourceParams]] = _FrameParams

    def load(self) -> pl.DataFrame:
        return pl.DataFrame({"VALUE": [1], "value": [999]})


def test_ambiguous_canonical_columns_from_a_source_fail_loading(registry) -> None:
    settings = _settings(engine=CANONICAL["engine"], datasets=CANONICAL["datasets"])
    ctx = DataContext(
        specs=settings.datasets,
        sources={"t": _FrameSource(_FrameParams(), SourceEnvironment())},
        canonical_columns=["Value"],
    )
    report = QARunner(settings, registry).run(ctx=ctx)
    assert report.verdict() != "passed"
    assert "canonical" in report.metadata.unavailable_datasets["t"]
    inp001 = report.get("INP-001")
    assert inp001.failed and "canonical" in inp001.message


def test_explicit_rename_disambiguates(registry) -> None:
    frame = pl.DataFrame({"VALUE": [1], "value": [999]})
    config = {
        **CANONICAL,
        "datasets": {"t": {**CANONICAL["datasets"]["t"], "rename": {"value": "value_raw"}}},
    }
    report = _run(registry, {"t": frame}, **config)
    assert report.verdict() == "passed"
    [info] = report.metadata.datasets
    assert set(info.columns) == {"Value", "value_raw"}
    assert info.renamed == {"VALUE": "Value", "value": "value_raw"}


def test_duplicate_canonical_spellings_are_rejected() -> None:
    with pytest.raises(ValidationError, match="canonical"):
        _settings(engine={"canonical_columns": ["Value", "VALUE"]})


def test_duplicate_rename_targets_are_rejected() -> None:
    with pytest.raises(ValidationError, match="rename"):
        _settings(datasets={"t": {"rename": {"a": "x", "b": "x"}}})


# =========================================================================
# Run metadata never loads data, and timing covers the run
# =========================================================================
class _SlowParams(SourceParams):
    type: str = "slow"


class _SlowSource(DataSource):
    kind: ClassVar[str] = "slow"
    Params: ClassVar[type[SourceParams]] = _SlowParams
    loads: ClassVar[int] = 0

    def load(self) -> pl.DataFrame:
        type(self).loads += 1
        time.sleep(0.05)
        return pl.DataFrame({"a": [1]})


class _Noop(Rule):
    rule_id: ClassVar[str] = "NOOP-1"
    name: ClassVar[str] = "Noop"
    description: ClassVar[str] = "Reads nothing."
    category: ClassVar[str] = "test"

    def evaluate(self, ctx):
        return Finding.ok("ok")


def test_metadata_does_not_load_unused_sources() -> None:
    registry = RuleRegistry()
    registry.register(_Noop)
    settings = _settings(rules={"NOOP-1": {}})
    ctx = DataContext(sources={"unused": _SlowSource(_SlowParams(), SourceEnvironment())})
    _SlowSource.loads = 0
    report = QARunner(settings, registry).run(ctx=ctx)
    assert _SlowSource.loads == 0
    assert report.metadata.datasets == []
    assert report.metadata.unused_datasets == ["unused"]


def test_metadata_lists_unavailable_datasets_with_reasons(tmp_path, registry) -> None:
    report = _run(
        registry,
        None,
        datasets={"gone": {"path": str(tmp_path / "missing.csv"), "required": False}},
    )
    assert "missing.csv" in report.metadata.unavailable_datasets["gone"]
    assert report.metadata.datasets == [] and report.metadata.unused_datasets == []
