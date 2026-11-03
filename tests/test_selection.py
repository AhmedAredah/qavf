"""Run-time selection (--only, --tags, --skip-tags, --dataset), tag policies and console text."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from qavf import validate
from qavf.checks.selection import NOT_SELECTED, Selection, SelectionError, narrow
from qavf.cli import ConsoleText, main
from qavf.settings import QAVFSettings

SERIES = {"value": "aadt", "group_by": ["site_id"], "time_column": "year"}
COUNTS = pl.DataFrame(
    {
        "site_id": ["S1", "S1", "S1", "S2", "S2"],
        "year": [2016, 2017, 2019, 2016, 2019],
        "aadt": [100, 110, 120, 50, 60],
    }
)
CRASHES = pl.DataFrame({"site_id": ["S1", "S2"], "year": [2016, 2016], "crashes": [3, -1]})


def _config() -> dict[str, Any]:
    return {
        "engine": {"parallel": False},
        "datasets": {
            "counts": {
                "schema": {"columns": {"aadt": {"dtype": "int", "ge": 0}}},
                "checks": [
                    {
                        "check": "missing_periods",
                        "name": "no missing years",
                        "series": SERIES,
                        "expected": "2016..2019",
                        "tags": ["nightly"],
                    },
                    {"check": "change_rate", "series": SERIES, "max": 0.05, "tags": ["slow"]},
                    {"check": "expression", "expr": "aadt > 50", "name": "positive"},
                ],
            },
            "crashes": {
                "schema": {"columns": {"crashes": {"dtype": "int", "ge": 0}}},
                "checks": [{"check": "expression", "expr": "crashes >= 0", "name": "counted"}],
            },
        },
    }


def _run(**selection: Any) -> Any:
    return validate({"counts": COUNTS, "crashes": CRASHES}, config=_config(), **selection)


def _ran(report: Any) -> set[str]:
    return {result.item_name for result in report.results if result.item_name}


def _rules(report: Any) -> set[str]:
    return {result.rule_id for result in report.results}


def test_tags_select_items_and_built_in_families() -> None:
    assert _ran(_run(tags=["nightly"])) == {"no missing years"}
    assert _rules(_run(tags="temporal")) == {"TMP-001", "TMP-004"}
    report = _run(tags="input")
    assert _ran(report) == {"positive", "counted"}
    assert {"INP-002", "INP-003"} <= _rules(report)


def test_skip_tags_leave_checks_out_and_report_why() -> None:
    report = _run(skip_tags=["slow", "input"])
    assert _ran(report) == {"no missing years"}
    assert report.metadata.excluded_rules["change_rate on counts"] == NOT_SELECTED
    assert report.metadata.excluded_rules["INP-003"] == NOT_SELECTED


def test_only_runs_named_items_without_schema_contracts() -> None:
    report = _run(only=["positive"])
    assert _ran(report) == {"positive"}
    assert _rules(report) == {"INP-006"}


def test_dataset_runs_its_checks_and_its_schema_contracts() -> None:
    report = _run(datasets="crashes")
    assert _ran(report) == {"counted"}
    columns = [r for r in report.results if r.rule_id == "INP-003"]
    assert columns and all(r.datasets == ["crashes"] for r in columns if r.datasets)
    assert {r.rule_id for r in report.results} == {"INP-001", "INP-002", "INP-003", "INP-006"} | (
        {"INP-004", "INP-005", "INP-009"} & _rules(report)
    )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"tags": ["nope"]}, "the selection --tags nope matches no check"),
        ({"only": ["nope"]}, "no check is named 'nope'"),
        ({"datasets": ["nope"]}, "unknown dataset 'nope' selected"),
        ({"only": ["positive"], "tags": ["nightly"]}, "matches no check"),
    ],
)
def test_selecting_nothing_is_an_error(selection: dict[str, Any], message: str) -> None:
    with pytest.raises(SelectionError, match=message):
        _run(**selection)


def test_narrowing_keeps_requirements_of_selected_checks_only() -> None:
    config = _config()
    config["run"] = {"require": ["no missing years", "positive", "counts", "crashes"]}
    settings = QAVFSettings.model_validate(config)
    narrowed = narrow(settings, Selection.of(datasets=["counts"], skip_tags=["nightly"]))
    assert narrowed.run.require == ["positive", "counts"]
    assert settings.run.require == ["no missing years", "positive", "counts", "crashes"]
    assert narrow(settings, Selection()) is settings


def test_tag_policy_keeps_tagged_failures_from_failing_the_run() -> None:
    config = _config()
    config["datasets"]["crashes"]["checks"][0]["tags"] = ["exploratory"]
    data = {"counts": COUNTS, "crashes": CRASHES}
    strict = validate(data, config=config, only="counted")
    assert strict.verdict() == "failed"
    config["run"] = {"tags": {"exploratory": {"fail_on": "never"}}}
    lenient = validate(data, config=config, only="counted")
    assert lenient.metadata.tag_fail_on == {"exploratory": "never"}
    assert lenient.results_for("INP-006")[0].tags == ["exploratory"]
    assert lenient.verdict() == "passed"
    # A family tag works as a policy key too; the strictest applicable policy wins.
    config["run"] = {"tags": {"exploratory": {"fail_on": "never"}, "input": {"fail_on": "ERROR"}}}
    assert validate(data, config=config, only="counted").verdict() == ("failed")


def test_tag_policy_names_must_be_words() -> None:
    with pytest.raises(ValueError, match="single nonempty word"):
        QAVFSettings.model_validate({"run": {"tags": {"two words": {"fail_on": "never"}}}})


def test_cli_selects_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    counts = tmp_path / "counts.csv"
    COUNTS.write_csv(counts)
    config = tmp_path / "qa.yaml"
    config.write_text(
        f"""
engine: {{parallel: false}}
report: {{html: false, json: false}}
datasets:
  counts:
    path: {counts.as_posix()}
    checks:
      - {{check: expression, expr: aadt > 0, name: positive, tags: [nightly]}}
      - {{check: missing_periods, name: no missing years, expected: 2016..2019,
         series: {{value: aadt, group_by: [site_id], time_column: year}}}}
""",
        encoding="utf-8",
    )
    assert main(["run", "-c", str(config), "--only", "no missing years"]) == 1
    out = capsys.readouterr().out
    assert "1 results: 0 passed, 1 failed" in out
    assert main(["run", "-c", str(config), "--tags", "nightly", "--skip-tags", "temporal"]) == 0
    assert "1 results: 1 passed" in capsys.readouterr().out
    assert main(["run", "-c", str(config), "--tags", "nope"]) == 2
    assert "matches no check" in capsys.readouterr().err


def test_console_text_falls_back_to_ascii_on_a_legacy_code_page() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")
    console = ConsoleText(stream)
    console.prepare()
    assert not console.unicode
    stream.write(console("missing_periods · counts → ☃\n"))
    stream.flush()
    assert raw.getvalue() == b"missing_periods - counts -> ?\n"


def test_console_text_keeps_symbols_on_utf8() -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    assert ConsoleText(stream)("a · b") == "a · b"


def test_isolated_runs_receive_the_selection(tmp_path: Path) -> None:
    from qavf.execution.protocol import read_request, request_bytes

    settings = narrow(QAVFSettings.model_validate(_config()), Selection.of(datasets="crashes"))
    request = tmp_path / "request.json"
    request.write_bytes(request_bytes(settings, {}))
    received, _, _ = read_request(request)
    assert received.selection == settings.selection
    assert received.selection is not None
    assert received.selection.contract_datasets() == ["crashes"]
