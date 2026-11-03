"""Every catalog check compiles to its rule's parameters and gives the rule's results.

Each case pairs a check item with the QAVF 0.6 rule configuration it replaces.
The item must compile to parameters equal to the old ones, and a run of the
item must report exactly what the rule reports when it is given the old
parameters directly, apart from the new identity labels.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from qavf.checks.catalog import BY_RULE, CATALOG
from qavf.checks.compiler import compile_checks
from qavf.engine.registry import Rule
from qavf.engine.runner import QARunner
from qavf.reports.models import RuleResult
from qavf.settings import ExpressionCheck, QAVFSettings, SqlAssertion
from qavf.testing import discovered_rules

SERIES = {"value": "aadt", "group_by": ["site_id"], "time_column": "year"}
OLD_SERIES = {"dataset": "counts", **SERIES}
METRIC = {"value": "aadt", "entity": "site_id", "group_by": ["year"]}
OLD_METRIC = {
    "dataset": "counts",
    "value": "aadt",
    "entity_column": "site_id",
    "group_by": ["year"],
}
BASELINE = "baseline.csv"

# (check item, where it goes, rule id, old rule parameters)
CASES: list[tuple[dict[str, Any], str | None, str, dict[str, Any]]] = [
    (
        {"check": "fresh", "column": "seen", "max_age": 86_400, "now": "2024-06-01T00:00:00"},
        "counts",
        "INP-008",
        {
            "checks": [{"dataset": "counts", "column": "seen", "max_age": 86_400}],
            "now": "2024-06-01T00:00:00",
        },
    ),
    (
        {"check": "change_rate", "series": SERIES, "max": 0.2, "min_value": 10},
        "counts",
        "TMP-001",
        {"series": [OLD_SERIES], "max_change_rate": 0.2, "min_value": 10},
    ),
    (
        {"check": "trend", "series": SERIES, "expect": "increasing", "min_points": 4},
        "counts",
        "TMP-002",
        {"series": [{**OLD_SERIES, "expected_trend": "increasing"}], "min_points": 4},
    ),
    (
        {"check": "trend_outliers", "series": SERIES, "alpha": 0.01},
        "counts",
        "TMP-003",
        {"series": [OLD_SERIES], "alpha": 0.01},
    ),
    (
        {"check": "missing_periods", "series": SERIES, "expected": [2016, 2017, 2018, 2019]},
        "counts",
        "TMP-004",
        {"series": [OLD_SERIES], "expected_periods": [2016, 2017, 2018, 2019]},
    ),
    (
        {"check": "stale_values", "series": SERIES, "max_repeats": 2},
        "counts",
        "TMP-005",
        {"series": [OLD_SERIES], "max_repeats": 2},
    ),
    (
        {"check": "cumulative", "series": SERIES, "allow_reset_to_zero": True},
        "counts",
        "TMP-006",
        {"series": [OLD_SERIES], "allow_reset_to_zero": True},
    ),
    (
        {"check": "seasonal", "series": SERIES, "season_length": 2, "min_cycles": 3},
        "counts",
        "TMP-007",
        {"series": [OLD_SERIES], "season_length": 2, "min_cycles": 3},
    ),
    (
        {"check": "level_shift", "series": SERIES, "min_segment": 2},
        "counts",
        "TMP-008",
        {"series": [OLD_SERIES], "min_segment": 2},
    ),
    (
        {"check": "outliers", "column": "aadt", "group_by": ["year"], "method": "iqr"},
        "counts",
        "STA-001",
        {
            "targets": [
                {"dataset": "counts", "value": "aadt", "group_by": ["year"], "method": "iqr"}
            ]
        },
    ),
    (
        {"check": "drift", "datasets": ["other", "counts"], "column": "aadt", "bins": 5},
        None,
        "STA-002",
        {"comparisons": [{"reference": "other", "dataset": "counts", "column": "aadt", "bins": 5}]},
    ),
    (
        {
            "check": "totals_match",
            "datasets": ["counts", "other"],
            "value": "aadt",
            "keys": ["site_id"],
            "atol": 1,
        },
        None,
        "REC-001",
        {
            "reconciliations": [
                {
                    "left": {"dataset": "counts", "value": "aadt"},
                    "right": {"dataset": "other", "value": "aadt"},
                    "on": ["site_id"],
                    "atol": 1,
                }
            ]
        },
    ),
    (
        {"check": "shares", "value": "share", "group_by": ["year"]},
        "counts",
        "REC-002",
        {"shares": [{"dataset": "counts", "value": "share", "group_by": ["year"]}]},
    ),
    (
        {"check": "key_grid", "dimensions": {"year": {"values": [2016, 2017, 2018, 2019]}}},
        "counts",
        "REC-003",
        {
            "grids": [
                {"dataset": "counts", "dimensions": {"year": {"values": [2016, 2017, 2018, 2019]}}}
            ]
        },
    ),
    (
        {"check": "key_sets_match", "datasets": ["counts", "other"], "keys": ["site_id", "year"]},
        None,
        "REC-004",
        {
            "pairs": [
                {
                    "left": {"dataset": "counts", "columns": ["site_id", "year"]},
                    "right": {"dataset": "other", "columns": ["site_id", "year"]},
                }
            ]
        },
    ),
    (
        {"check": "batch_consistent", "datasets": ["counts", "other"], "column": "batch"},
        None,
        "REC-005",
        {"batches": [{"datasets": ["counts", "other"], "column": "batch"}]},
    ),
    (
        {"check": "coordinates", "latitude": "lat", "longitude": "lon"},
        "counts",
        "GEO-001",
        {"points": [{"dataset": "counts", "latitude": "lat", "longitude": "lon"}]},
    ),
    (
        {"check": "entity_outliers", "metric": METRIC, "min_entities": 3},
        "counts",
        "SPA-001",
        {"metrics": [OLD_METRIC], "min_entities": 3},
    ),
    (
        {
            "check": "group_comparison",
            "metric": METRIC,
            "groups": {"A": ["S1", "S2"], "B": ["S3", "S4"]},
            "min_entities": 2,
        },
        "counts",
        "SPA-002",
        {
            "metrics": [OLD_METRIC],
            "groups": {"A": ["S1", "S2"], "B": ["S3", "S4"]},
            "min_entities": 2,
        },
    ),
    (
        {
            "check": "coverage",
            "datasets": ["counts", "other"],
            "entity": "site_id",
            "time_column": "year",
        },
        None,
        "SPA-003",
        {"datasets": ["counts", "other"], "entity_column": "site_id", "time_column": "year"},
    ),
    (
        {"check": "matches_baseline", "path": BASELINE, "keys": ["site_id", "year"], "rtol": 0.01},
        "counts",
        "REG-001",
        {
            "baselines": [
                {"dataset": "counts", "path": BASELINE, "keys": ["site_id", "year"], "rtol": 0.01}
            ]
        },
    ),
    (
        {"check": "compare", "datasets": ["other", "counts"], "keys": ["site_id", "year"]},
        None,
        "CMP-001",
        {
            "comparisons": [
                {"baseline": "other", "candidate": "counts", "keys": ["site_id", "year"]}
            ]
        },
    ),
]

IDENTITY_FIELDS = {
    "check_name",
    "item_name",
    "item_ids",
    "execution_time_s",
    "load_time_s",
    "description",
}


def _frames() -> dict[str, pl.DataFrame]:
    rows = [
        (site, year, aadt)
        for site, base in (("S1", 100), ("S2", 220), ("S3", 90), ("S4", 400))
        for year, aadt in zip(
            (2016, 2017, 2018, 2019), (base, base + 10, base + 5, base * 2), strict=True
        )
    ]
    counts = pl.DataFrame(
        {
            "site_id": [r[0] for r in rows],
            "year": [r[1] for r in rows],
            "aadt": [r[2] for r in rows],
            "share": [0.25] * len(rows),
            "lat": [30.0 + i / 100 for i in range(len(rows))],
            "lon": [-97.0] * (len(rows) - 1) + [200.0],
            "seen": [datetime(2024, 5, 1)] * len(rows),
            "batch": ["b1"] * len(rows),
        }
    )
    other = counts.with_columns(
        pl.when(pl.col("site_id") == "S2").then(pl.col("aadt") + 1).otherwise(pl.col("aadt"))
    ).filter(pl.col("site_id") != "S4")
    return {"counts": counts, "other": other}


def _settings(tmp_path: Path, item: dict[str, Any] | None, where: str | None) -> QAVFSettings:
    frames = _frames()
    frames["counts"].write_csv(tmp_path / BASELINE)
    datasets: dict[str, Any] = {name: {} for name in frames}
    raw: dict[str, Any] = {"datasets": datasets, "engine": {"parallel": False}}
    if item is not None and where is not None:
        datasets[where]["checks"] = [item]
    elif item is not None:
        raw["checks"] = [item]
    settings = QAVFSettings.model_validate(raw)
    settings.source_path = tmp_path / "qavf.yaml"
    return settings


class _FixedPlan(QARunner):
    """Runs exactly the given rule instances: the rule configured by its old parameters."""

    def __init__(self, settings: QAVFSettings, rules: list[Rule]) -> None:
        super().__init__(settings, registry=discovered_rules())
        self.fixed = rules

    def plan(self) -> list[Rule]:
        self.excluded = {}
        return list(self.fixed)


def _outcome(results: list[RuleResult], rule_id: str) -> list[dict[str, Any]]:
    """The results of ``rule_id`` without identity labels; evidence rows in any order
    (some rules sample unordered sets)."""
    outcome = []
    for result in results:
        if result.rule_id == rule_id:
            dumped = result.model_dump(mode="json", exclude=IDENTITY_FIELDS)
            dumped["affected_records"] = sorted(map(repr, dumped["affected_records"]))
            outcome.append(dumped)
    return outcome


def test_cases_cover_every_configurable_check() -> None:
    covered = {item["check"] for item, *_ in CASES} | {"expression", "sql"}
    configurable = {name for name, definition in CATALOG.items() if definition.configurable}
    assert covered == configurable


def test_every_rule_has_its_catalog_name() -> None:
    for rule_cls in discovered_rules(entry_points=False):
        definition = BY_RULE[rule_cls.rule_id]
        assert definition.name == rule_cls.check_name
        assert CATALOG[rule_cls.check_name].rule_cls is rule_cls


@pytest.mark.parametrize(
    ("item", "where", "rule_id", "old"), CASES, ids=[c[0]["check"] for c in CASES]
)
def test_check_item_compiles_to_the_old_parameters(
    tmp_path: Path, item: dict[str, Any], where: str | None, rule_id: str, old: dict[str, Any]
) -> None:
    settings = _settings(tmp_path, item, where)
    (compiled,) = compile_checks(settings)
    assert compiled.rule_id == rule_id
    params = BY_RULE[rule_id].rule_cls.Params
    assert params.model_validate(compiled.params) == params.model_validate(old)


@pytest.mark.parametrize(
    ("item", "where", "rule_id", "old"), CASES, ids=[c[0]["check"] for c in CASES]
)
def test_check_item_reports_what_the_old_rule_reports(
    tmp_path: Path, item: dict[str, Any], where: str | None, rule_id: str, old: dict[str, Any]
) -> None:
    frames = _frames()
    new = QARunner(_settings(tmp_path, item, where), registry=discovered_rules())
    new_results = new.run(frames=frames).results
    rule_cls = BY_RULE[rule_id].rule_cls
    previous = _FixedPlan(_settings(tmp_path, None, None), [rule_cls(params=old)])
    old_results = previous.run(frames=frames).results
    assert _outcome(new_results, rule_id) == _outcome(old_results, rule_id)
    assert _outcome(new_results, rule_id)
    for result in new_results:
        if result.rule_id == rule_id:
            assert result.check_name == item["check"]
            assert result.item_name is not None
            assert result.label.startswith(f"{item['check']} · ")


def test_expression_and_sql_items_compile_to_the_old_models(tmp_path: Path) -> None:
    raw = {
        "datasets": {
            "counts": {
                "checks": [
                    {"check": "expression", "expr": "aadt >= 0", "name": "positive"},
                    {"check": "expression", "expr": "share <= 1", "null_as": "fail"},
                ]
            }
        },
        "checks": [{"check": "sql", "name": "none", "sql": "SELECT 1 WHERE false"}],
    }
    settings = QAVFSettings.model_validate(raw)
    assert settings.datasets["counts"].expressions == [
        ExpressionCheck(expr="aadt >= 0", name="positive"),
        ExpressionCheck(expr="share <= 1", null_as="fail"),
    ]
    assert settings.assertions == [
        SqlAssertion(name="none", sql="SELECT 1 WHERE false", item_id="checks[0]")
    ]
    assert [check.item_id for check in settings.datasets["counts"].expressions_of("counts")] == [
        "datasets.counts.checks[0]",
        "datasets.counts.checks[1]",
    ]
