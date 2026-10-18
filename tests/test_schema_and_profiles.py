"""Configuration schema, scaffolds and profiling.

JSON evidence for non-finite values, schema/runtime agreement,
scaffolds with required plugin parameters, profiling precision and margins and CLI
handling of malformed configuration and filesystem errors.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

import jsonschema
import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from qavf.cli import EXIT_USAGE, main
from qavf.engine.registry import Finding, Rule, RuleParams, RuleRegistry
from qavf.engine.runner import QARunner, build_source_registry
from qavf.engine.severity import Severity, Status
from qavf.reports.models import QAReport, RuleResult, RunMetadata
from qavf.scaffold import config_json_schema, profile_frame, render_config
from qavf.settings import QAVFSettings, load_settings

BIG = 2**53 + 1  # 9007199254740993: not representable as a float


# =========================================================================
# JSON keeps NaN and infinity distinct from null
# =========================================================================
def _report(records: list[dict], metrics: dict) -> QAReport:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    result = RuleResult(
        rule_id="X",
        name="X",
        description="d",
        category="c",
        status=Status.FAIL,
        severity=Severity.ERROR,
        message="m",
        affected_records=records,
        metrics=metrics,
    )
    metadata = RunMetadata(
        run_id="r", project="p", qavf_version="0", started_at=now, finished_at=now
    )
    return QAReport(metadata=metadata, results=[result])


def test_non_finite_values_round_trip_through_json(tmp_path) -> None:
    records = [
        {
            "nan": math.nan,
            "inf": math.inf,
            "ninf": -math.inf,
            "none": None,
            "np": np.float64("nan"),
            "x": 1.5,
        }
    ]
    metrics = {"nested": {"values": [math.nan, 2.0, math.inf]}, "plain": 3}
    report = _report(records, metrics)
    text = report.model_dump_json()
    # Standard JSON: no NaN/Infinity literals, and parseable by a strict parser.
    json.loads(text, parse_constant=lambda name: pytest.fail(f"non-standard literal {name}"))
    record = json.loads(text)["results"][0]["affected_records"][0]
    assert record["nan"] == {"$float": "NaN"} and record["inf"] == {"$float": "Infinity"}
    assert (
        record["ninf"] == {"$float": "-Infinity"} and record["none"] is None and record["x"] == 1.5
    )

    path = report.write_json(tmp_path / "r.json")
    restored = QAReport.model_validate_json(path.read_text(encoding="utf-8"))
    back = restored.results[0].affected_records[0]
    assert math.isnan(back["nan"]) and back["inf"] == math.inf and back["ninf"] == -math.inf
    assert back["none"] is None and math.isnan(back["np"])
    nested = restored.results[0].metrics["nested"]["values"]
    assert math.isnan(nested[0]) and nested[1:] == [2.0, math.inf]


# =========================================================================
# The generated JSON Schema agrees with runtime validation
# =========================================================================
class _NeedsLimit(RuleParams):
    limit: int
    """The required limit."""


class _PluginRule(Rule):
    rule_id: ClassVar[str] = "PLUG-1"
    name: ClassVar[str] = "Plugin rule"
    description: ClassVar[str] = "Needs a limit."
    category: ClassVar[str] = "plugin"
    Params: ClassVar[type[RuleParams]] = _NeedsLimit

    def evaluate(self, ctx):
        return Finding.ok("ok")


@pytest.fixture
def plugin_registry(registry) -> RuleRegistry:
    combined = RuleRegistry()
    for rule in [*registry, _PluginRule]:
        combined.register(rule)
    return combined


def _schema_accepts(schema, document) -> bool:
    return not list(jsonschema.Draft202012Validator(schema).iter_errors(document))


def _runtime_accepts(registry, document) -> bool:
    try:
        settings = QAVFSettings.model_validate(document)
        QARunner(settings, registry).plan()
    except (ValidationError, ValueError):
        return False
    return True


@pytest.mark.parametrize(
    "document",
    [
        {"rules": {"TMP-001": None}},
        {"rules": {"TMP-001": {"params": None}}},
        {"rules": {"TMP-001": {"severity": None}}},
        {"rules": {"TMP-001": {"params": {"max_change_rate": 0.1}}}},
        {"rules": {"TMP-001": {"params": {"max_change_rate": "fast"}}}},
        {"rules": {"PLUG-1": {"enabled": False}}},
        {"rules": {"PLUG-1": {}}},
        {"rules": {"PLUG-1": {"enabled": True}}},
        {"rules": {"PLUG-1": {"params": {"limit": 3}}}},
        {"datasets": {"t": {"source": None}}},
    ],
)
def test_schema_and_runtime_agree(document, plugin_registry) -> None:
    schema = config_json_schema(plugin_registry, build_source_registry(QAVFSettings()))
    assert _schema_accepts(schema, document) == _runtime_accepts(plugin_registry, document), (
        document
    )


# =========================================================================
# A generated configuration always loads and plans
# =========================================================================
def test_scaffold_with_required_plugin_parameters_plans(tmp_path, plugin_registry) -> None:
    text = render_config(plugin_registry, None)
    block = text[text.index("  PLUG-1:") :].split("\n\n")[0]
    assert (
        "enabled: false" in block
        and "limit" in text[text.index("# PLUG-1") : text.index("  PLUG-1:")]
    )
    path = tmp_path / "qavf.yaml"
    path.write_text(text, encoding="utf-8")
    settings = load_settings(path)
    assert QARunner(settings, plugin_registry).plan()
    schema = config_json_schema(plugin_registry, build_source_registry(QAVFSettings()))
    assert _schema_accepts(schema, json.loads(json.dumps(yaml.safe_load(text), default=str)))


# =========================================================================
# Profiling keeps exact bounds and validates margins
# =========================================================================
def test_profiled_integer_bounds_are_exact() -> None:
    frame = pl.DataFrame({"n": [BIG, BIG + 2]})
    column = profile_frame(frame)["columns"]["n"]
    assert column == {"dtype": "int", "ge": BIG, "le": BIG + 2}
    assert all(type(column[key]) is int for key in ("ge", "le"))
    widened = profile_frame(frame, margin=0.5)["columns"]["n"]
    assert widened["ge"] == BIG - 1 and widened["le"] == BIG + 3


@pytest.mark.parametrize("margin", [-1.0, math.inf, math.nan])
def test_invalid_margins_are_rejected(margin) -> None:
    with pytest.raises(ValueError, match="margin"):
        profile_frame(pl.DataFrame({"n": list(range(11))}), margin=margin)


def test_cli_rejects_invalid_margins(tmp_path, capsys) -> None:
    pl.DataFrame({"n": [1, 2]}).write_csv(tmp_path / "d.csv")
    with pytest.raises(SystemExit) as exit_info:
        main(["profile", str(tmp_path / "d.csv"), "--margin", "-1"])
    assert exit_info.value.code == EXIT_USAGE
    assert "margin" in capsys.readouterr().err


def test_date_and_datetime_bounds_are_widened_by_the_margin() -> None:
    frame = pl.DataFrame(
        {
            "d": [date(2020, 1, 1), date(2020, 1, 11)],
            "t": [datetime(2020, 1, 1), datetime(2020, 1, 1, 10)],
        }
    )
    exact = profile_frame(frame)["columns"]
    assert (exact["d"]["ge"], exact["d"]["le"]) == (date(2020, 1, 1), date(2020, 1, 11))
    wide = profile_frame(frame, margin=0.5)["columns"]
    assert (wide["d"]["ge"], wide["d"]["le"]) == (date(2019, 12, 27), date(2020, 1, 16))
    assert wide["t"]["ge"] == datetime(2020, 1, 1) - timedelta(hours=5)
    assert wide["t"]["le"] == datetime(2020, 1, 1, 15)


# =========================================================================
# The CLI reports configuration and filesystem errors cleanly
# =========================================================================
@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("bad.yaml", "datasets: [unclosed\n", "invalid YAML"),
        ("bad.json", "{not json", "invalid JSON"),
        ("tabs.yaml", "datasets:\n\tt: {}\n", "invalid YAML"),
    ],
)
def test_malformed_configuration_exits_with_usage_error(
    tmp_path, capsys, name, content, match
) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    assert main(["run", "-c", str(path)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert match in err and "line" in err and "Traceback" not in err
    with pytest.raises(ValueError, match=match):
        load_settings(path)


def test_unwritable_report_location_exits_with_usage_error(tmp_path, capsys) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    path = tmp_path / "qavf.yaml"
    config = {"report": {"output_dir": str(blocker / "sub")}, "plugins": {"entry_points": False}}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["run", "-c", str(path)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err
