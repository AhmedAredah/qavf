"""CMP-001 through validate(), configuration files, the CLI and isolated execution."""

from __future__ import annotations

import json
import sys

import polars as pl
import pytest
import yaml

from qavf import File, QAReport, ResourceLimits, Severity, Status, compare_datasets, validate
from qavf.cli import EXIT_OK, EXIT_QA_FAILED, EXIT_USAGE, main
from qavf.engine.compilation import ConfigurationError
from qavf.execution.os_limits import SUPPORTED_PLATFORMS
from qavf.reports.html import render_html

V1 = pl.DataFrame(
    {
        "SiteID": ["s1", "s2", "s3", "s4"],
        "county": ["a", "a", "b", "b"],
        "aadt_k": [1.2, 3.4, 5.0, 2.0],
        "fc": [1, 2, 2, 3],
    }
)
V2 = pl.DataFrame(
    {
        "site": ["s1", "s2", "s3", "s4"],
        "county": ["a", "a", "b", "b"],
        "aadt": [1200.0, 3400.0, 5000.0, 2000.0],
        "fc": ["interstate", "arterial", "arterial", "local"],
    }
)


def _config(**comparison):
    spec = {
        "name": "aadt v1 vs v2",
        "baseline": "v1",
        "candidate": "v2",
        "keys": ["site"],
        "columns": {"aadt_k": "aadt"},
        "units": {"aadt": {"factor": 1000}},
        "recode": {"fc": {"values": {"1": "interstate", "2": "arterial", "3": "local"}}},
        "decision": "material",  # the rule's plumbing, not significance (four sites)
        "levels": [
            "structure",
            "coverage",
            "records",
            {
                "level": "groups",
                "severity": "WARNING",
                "group_by": ["county"],
                "measures": ["count", "sum(aadt)"],
            },
        ],
        **comparison,
    }
    return {
        # A dataset-level rename is applied before the comparison sees the data.
        "datasets": {"v1": {"rename": {"SiteID": "site"}}, "v2": {}},
        "rules": {"CMP-001": {"params": {"comparisons": [spec]}}},
        "engine": {"run_unconfigured_rules": False},
    }


def _results(report):
    return {result.check: result for result in report.results if result.rule_id == "CMP-001"}


def test_cmp001_passes_after_alignment_and_reports_every_check():
    report = validate({"v1": V1, "v2": V2}, config=_config())
    results = _results(report)
    assert report.verdict() == "passed", [r.message for r in results.values()]
    assert "aadt v1 vs v2: records: values" in results
    assert "aadt v1 vs v2: groups: sum(aadt)" in results
    assert all(result.status is Status.PASS for result in results.values())


def test_cmp001_failures_keep_level_severity_evidence_and_reports():
    changed = V2.with_columns(pl.Series("aadt", [1200.0, 3500.0, 5000.0, 2000.0]))
    report = validate({"v1": V1, "v2": changed}, config=_config())
    results = _results(report)
    values = results["aadt v1 vs v2: records: values"]
    assert values.status is Status.FAIL and values.severity is Severity.ERROR
    assert values.affected_count == 1 and values.affected_records[0]["site"] == "s2"
    grouped = results["aadt v1 vs v2: groups: sum(aadt)"]
    assert grouped.status is Status.FAIL and grouped.severity is Severity.WARNING
    assert report.verdict() == "failed"
    # JSON and HTML reports carry the comparison unchanged.
    restored = QAReport.model_validate_json(b"".join(report.iter_json_chunks()))
    assert restored.get("CMP-001", values.check).affected_count == 1
    assert "records: values" in render_html(report)


def test_cmp001_unassessable_checks_are_errors_and_satisfy_no_requirement():
    config = _config(columns={"not_there": "aadt"})
    config["engine"]["require"] = ["CMP-001[aadt v1 vs v2: records: values]"]
    report = validate({"v1": V1, "v2": V2}, config=config)
    results = _results(report)
    assert {r.status for r in results.values()} == {Status.ERROR}
    assert any("lacks mapped column" in r.message for r in results.values())
    assert report.incomplete and report.verdict() == "incomplete"


def test_cmp001_unavailable_dataset_is_not_assessed():
    config = _config()
    config["datasets"]["v2"] = {"required": False, "path": "does/not/exist.parquet"}
    report = validate({"v1": V1}, config=config)
    assert {r.status for r in _results(report).values()} == {Status.ERROR}


def test_cmp001_configuration_errors_fail_before_any_data_is_read():
    config = _config(baseline="nope")
    with pytest.raises(ConfigurationError, match="unknown dataset 'nope'"):
        validate({"v1": V1, "v2": V2}, config=config)
    config = _config(levels=["records"], keys=[])
    with pytest.raises(ConfigurationError, match="needs the comparison's 'keys'"):
        validate({"v1": V1, "v2": V2}, config=config)


def test_cmp001_from_a_configuration_file_and_the_cli(tmp_path):
    V1.write_parquet(tmp_path / "v1.parquet")
    V2.write_parquet(tmp_path / "v2.parquet")
    config = _config()
    config["datasets"]["v1"]["path"] = "v1.parquet"
    config["datasets"]["v2"]["path"] = "v2.parquet"
    config["report"] = {"output_dir": "out", "html": True, "json": True}
    path = tmp_path / "qavf.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "-c", str(path)]) == EXIT_OK
    data = json.loads((tmp_path / "out" / "qa_report.json").read_text(encoding="utf-8"))
    assert any(r["rule_id"] == "CMP-001" for r in data["results"])
    V2.with_columns(pl.col("aadt") * 2).write_parquet(tmp_path / "v2.parquet")
    assert main(["run", "-c", str(path)]) == EXIT_QA_FAILED
    config["rules"]["CMP-001"]["params"]["comparisons"][0]["levels"] = ["mystery"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "-c", str(path)]) == EXIT_USAGE


def test_compare_datasets_accepts_files_and_matches_the_in_process_result(tmp_path):
    V1.rename({"SiteID": "site"}).write_parquet(tmp_path / "v1.parquet")
    V2.write_parquet(tmp_path / "v2.parquet")
    options = {
        "keys": ["site"],
        "columns": {"aadt_k": "aadt"},
        "units": {"aadt": {"factor": 1000}},
        "recode": {"fc": {"values": {"1": "interstate", "2": "arterial", "3": "local"}}},
    }
    local = compare_datasets(tmp_path / "v1.parquet", File(tmp_path / "v2.parquet"), **options)
    assert local.passed, local.summary()
    if sys.platform not in SUPPORTED_PLATFORMS:
        return
    isolated = compare_datasets(
        File(tmp_path / "v1.parquet"),
        File(tmp_path / "v2.parquet"),
        resources=ResourceLimits(isolate=True, wall_time=120),
        **options,
    )
    assert isolated.execution_limits["execution_mode"] == "isolated"
    assert isolated.model_dump(exclude={"execution_limits"}) == local.model_dump(
        exclude={"execution_limits"}
    )
