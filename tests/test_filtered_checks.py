"""``where`` on every dataset check (review finding R6).

A dataset check item whose check does not filter rows itself reads a hidden view
of its dataset (``<dataset>.__where_<digest>``). Its results must equal those of
the same check placed on a dataset holding exactly the matching rows, name the
real dataset (never the hidden view), report the population, and refuse to pass
on an empty filtered population.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import explain, validate
from qavf.checks.selection import Selection, narrow
from qavf.engine.compilation import ConfigurationError
from qavf.engine.derived_names import WHERE_VIEW_PREFIX, where_view_name
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.runner import QARunner
from qavf.optional import OptionalDependencyError
from qavf.profiles import apply_profile
from qavf.redaction import exception_details
from qavf.reports.models import RuleResult
from qavf.settings import QAVFSettings, load_settings
from qavf.testing import discovered_rules

WHERE = "keep = 1"
SERIES = {"value": "aadt", "group_by": ["site_id"], "time_column": "year"}
SCOPE_METRICS = ("population_rows", "applicable_rows", "excluded_rows")


def _kept() -> pl.DataFrame:
    rows = [
        (site, year, aadt)
        for site, base in (("S1", 100), ("S2", 220), ("S3", 90), ("S4", 400))
        for year, aadt in zip(
            (2016, 2017, 2018, 2019), (base, base + 10, base + 5, base * 2), strict=True
        )
    ]
    return pl.DataFrame(
        {
            "site_id": [r[0] for r in rows],
            "year": [r[1] for r in rows],
            "aadt": [r[2] for r in rows],
            "share": [0.25] * len(rows),
            "lat": [30.0 + i / 100 for i in range(len(rows))],
            "lon": [-97.0] * (len(rows) - 1) + [200.0],
            "seen": [datetime(2024, 5, 1)] * len(rows),
            "keep": [1] * len(rows),
        }
    )


def _excluded() -> pl.DataFrame:
    """Rows the condition excludes; each would change every check's outcome."""
    return pl.DataFrame(
        {
            "site_id": ["S9"] * 4,
            "year": [2016, 2017, 2018, 2020],
            "aadt": [1, 90_000, 1, 90_000],
            "share": [0.5] * 4,
            "lat": [95.0] * 4,
            "lon": [500.0] * 4,
            "seen": [datetime(2030, 1, 1)] * 4,
            "keep": [0] * 4,
        }
    )


def _frames() -> dict[str, pl.DataFrame]:
    kept = _kept()
    return {"alpha": pl.concat([kept, _excluded()]), "beta": kept}


def _alpha() -> dict[str, pl.DataFrame]:
    return {"alpha": _frames()["alpha"]}


class _LevelParams(RuleParams):
    dataset: str | None = None
    level: int = 0


class _LevelRule(Rule):
    rule_id: ClassVar[str] = "LVL-1"
    check_name: ClassVar[str] = "levels"
    name: ClassVar[str] = "Levels"
    description: ClassVar[str] = "A plugin check for the tests."
    category: ClassVar[str] = "custom"
    Params: ClassVar[type[RuleParams]] = _LevelParams

    def evaluate(self, ctx: Any) -> Finding:
        return Finding.ok("ok")


ITEMS: list[dict[str, Any]] = [
    {"check": "fresh", "column": "seen", "max_age": "30d", "now": "2024-05-02T00:00:00"},
    {"check": "outliers", "column": "aadt"},
    {"check": "change_rate", "series": SERIES, "max": 0.2},
    {"check": "shares", "value": "share", "group_by": ["year"]},
    {
        "check": "key_grid",
        "dimensions": {
            "site_id": {"values": ["S1", "S9"]},
            "year": {"values": [2016, 2017, 2018]},
        },
    },
    {"check": "coordinates", "latitude": "lat", "longitude": "lon"},
    {"check": "matches_baseline", "keys": ["site_id", "year"], "rtol": 0.01},
]


def _config(item: dict[str, Any], **filtered: Any) -> dict[str, Any]:
    """``alpha`` checked with ``where``; ``beta`` (its matching rows) checked plainly."""
    return {
        "datasets": {
            "alpha": {"checks": [{**item, "where": WHERE, **filtered}]},
            "beta": {"checks": [item]},
        },
        "engine": {"parallel": False},
    }


def _results(report: Any, dataset: str) -> list[RuleResult]:
    return [r for r in report.results if r.item_ids == [f"datasets.{dataset}.checks[0]"]]


def _comparable(result: RuleResult) -> str:
    """The outcome of ``result`` with its dataset's name, and the filter's own
    parameters, metrics and message suffix, left out."""
    dumped = result.model_dump(
        mode="json",
        include={
            "rule_id",
            "check_id",
            "status",
            "severity",
            "affected_count",
            "affected_count_exact",
            "total_records",
            "unit",
            "affected_rows",
            "columns",
            "checks",
        },
    )
    dumped["affected_records"] = sorted(map(repr, result.affected_records))
    dumped["metrics"] = {k: v for k, v in result.metrics.items() if k not in SCOPE_METRICS}
    dumped["parameters"] = {
        k: v for k, v in result.parameters.items() if k not in ("where", "min_applicable_rows")
    }
    dumped["message"] = result.message.split(" Assessed ")[0]
    return (
        json.dumps(dumped, sort_keys=True, default=str).replace("alpha", "DS").replace("beta", "DS")
    )


def _with_baseline(item: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    if item["check"] != "matches_baseline":
        return item
    baseline = tmp_path / "baseline.csv"
    expected = _kept().with_columns(
        pl.when(pl.col("site_id") == "S2").then(pl.col("aadt") + 50).otherwise(pl.col("aadt"))
    )
    expected.drop("seen").write_csv(baseline)
    return {**item, "path": str(baseline)}


@pytest.mark.parametrize("item", ITEMS, ids=[item["check"] for item in ITEMS])
def test_a_filtered_check_reports_what_it_reports_on_the_matching_rows(
    item: dict[str, Any], tmp_path: Path
) -> None:
    item = _with_baseline(item, tmp_path)
    report = validate(_frames(), config=_config(item), backend="polars")
    filtered, plain = _results(report, "alpha"), _results(report, "beta")
    assert filtered and len(filtered) == len(plain)
    for left, right in zip(filtered, plain, strict=True):
        assert _comparable(left) == _comparable(right)
        assert left.datasets == ["alpha"]
        assert left.parameters["where"] == WHERE
        assert left.metrics["population_rows"] == 20
        assert left.metrics["applicable_rows"] == 16
        assert left.metrics["excluded_rows"] == 4
        assert "Assessed 16 of 20 row(s) of alpha matching 'where: keep = 1'." in left.message


@pytest.mark.parametrize("item", ITEMS, ids=[item["check"] for item in ITEMS])
def test_the_where_matters(item: dict[str, Any], tmp_path: Path) -> None:
    """Without the condition, the excluded rows change the outcome (the probes' R6)."""
    item = _with_baseline(item, tmp_path)
    frames = _alpha()
    config = {"datasets": {"alpha": {"checks": [item]}}, "engine": {"parallel": False}}
    unfiltered = _results(validate(frames, config=config, backend="polars"), "alpha")
    filtered = _results(validate(frames, config=_config(item), backend="polars"), "alpha")
    assert [_comparable(r) for r in unfiltered] != [_comparable(r) for r in filtered]


def test_results_never_show_the_hidden_view(tmp_path: Path) -> None:
    hidden = where_view_name("alpha", WHERE)
    for item in ITEMS:
        report = validate(
            _frames(), config=_config(_with_baseline(item, tmp_path)), backend="polars"
        )
        dumped = report.model_dump_json()
        assert WHERE_VIEW_PREFIX not in dumped, item["check"]
        assert hidden not in dumped
        assert all(
            not name.startswith("alpha.") for name in (d.name for d in report.metadata.datasets)
        )


def test_an_empty_filtered_population_is_not_assessed() -> None:
    item = {"check": "fresh", "column": "seen", "max_age": "30d", "where": "keep = 7"}
    report = validate(_alpha(), config={"datasets": {"alpha": {"checks": [item]}}})
    (result,) = _results(report, "alpha")
    assert result.status.value == "ERROR"
    assert "no row of alpha matches 'where: keep = 7'" in result.message
    assert result.metrics == {"population_rows": 20, "applicable_rows": 0, "excluded_rows": 20}
    assert report.exit_code() != 0


@pytest.mark.parametrize(("minimum", "status"), [(16, "PASS"), (17, "ERROR")])
def test_min_applicable_rows(minimum: int, status: str) -> None:
    item = {"check": "fresh", "column": "seen", "max_age": "30d", "now": "2024-05-02T00:00:00"}
    report = validate(
        _frames(), config=_config(item, min_applicable_rows=minimum), backend="polars"
    )
    (result,) = _results(report, "alpha")
    assert result.status.value == status
    assert result.parameters["min_applicable_rows"] == minimum
    if status == "ERROR":
        assert "only 16 row(s) match 'where: keep = 1', fewer than min_applicable_rows 17" in (
            result.message
        )


def test_min_applicable_rows_goes_with_where() -> None:
    item = {"check": "fresh", "column": "seen", "max_age": "30d", "min_applicable_rows": 2}
    with pytest.raises(ValidationError, match="'min_applicable_rows' goes with 'where'"):
        QAVFSettings.model_validate({"datasets": {"alpha": {"checks": [item]}}})


def test_defaults_give_min_applicable_rows_to_filtered_items() -> None:
    item = {"check": "fresh", "column": "seen", "max_age": "30d", "now": "2024-05-02T00:00:00"}
    config = {
        "defaults": {"fresh": {"min_applicable_rows": 17}},
        "datasets": {"alpha": {"checks": [{**item, "where": WHERE}]}},
        "engine": {"parallel": False},
    }
    alpha = {"alpha": _frames()["alpha"]}
    report = validate(alpha, config=config, backend="polars")
    (result,) = _results(report, "alpha")
    assert result.status.value == "ERROR"
    assert result.parameters["min_applicable_rows"] == 17
    # An item's own setting wins over the default.
    config["datasets"]["alpha"]["checks"][0]["min_applicable_rows"] = 16  # type: ignore[index]
    (result,) = _results(validate(alpha, config=config, backend="polars"), "alpha")
    assert result.status.value == "PASS"


def test_defaulted_min_applicable_rows_goes_with_where() -> None:
    item = {"check": "fresh", "column": "seen", "max_age": "30d"}
    config = {
        "defaults": {"fresh": {"min_applicable_rows": 2}},
        "datasets": {"alpha": {"checks": [item]}},
    }
    with pytest.raises(ValidationError, match="'min_applicable_rows' goes with 'where'"):
        QAVFSettings.model_validate(config)


def test_defaults_refuse_min_applicable_rows_for_checks_not_filtered() -> None:
    config = {"defaults": {"drift": {"min_applicable_rows": 2}}}
    with pytest.raises(ValidationError, match=r"defaults.drift.min_applicable_rows: not a setting"):
        QAVFSettings.model_validate(config)


def test_items_with_one_condition_share_one_hidden_view() -> None:
    settings = QAVFSettings.model_validate(
        {
            "datasets": {
                "alpha": {
                    "checks": [
                        {"check": "fresh", "column": "seen", "max_age": "30d", "where": WHERE},
                        {"check": "outliers", "column": "aadt", "where": WHERE},
                        {"check": "outliers", "column": "lat", "where": "keep = 0"},
                        {"check": "expression", "expr": "aadt > 0", "where": "lat > 0"},
                    ]
                },
                "beta": {"checks": [{"check": "outliers", "column": "aadt", "where": WHERE}]},
            }
        }
    )
    hidden = settings.where_views()
    assert set(hidden) == {
        where_view_name("alpha", WHERE),
        where_view_name("alpha", "keep = 0"),
        where_view_name("beta", WHERE),
    }
    parent, spec, origin = hidden[where_view_name("alpha", WHERE)]
    assert (parent, spec.where, origin) == ("alpha", WHERE, "datasets.alpha.checks[0].where")
    assert settings.dataset_parents()[where_view_name("beta", WHERE)] == ("beta",)
    assert settings.views() == {}


def test_explain_shows_the_dependency_and_the_columns_where_reads() -> None:
    item = {"check": "outliers", "column": "aadt", "where": "keep = 1 AND lat > 0"}
    plan = explain(_alpha(), config={"datasets": {"alpha": {"checks": [item]}}}, backend="polars")
    (rule,) = [r for r in plan.rules if r.item_name == "outliers on alpha"]
    assert rule.datasets == ("alpha",)
    assert rule.where == "keep = 1 AND lat > 0"
    assert rule.where_columns == ("keep", "lat")
    assert WHERE_VIEW_PREFIX not in plan.model_dump_json()


def test_the_run_plan_keeps_the_dataset_with_the_hidden_view() -> None:
    from qavf.engine.compilation import compile_rules
    from qavf.engine.planning import build_plan
    from qavf.engine.registry import RuleRegistry
    from qavf.sources.base import SourceRegistry

    item = {"check": "outliers", "column": "aadt", "where": WHERE}
    settings = QAVFSettings.model_validate({"datasets": {"alpha": {"checks": [item]}}})
    registry = RuleRegistry().discover(packages=("qavf.rules",), use_entry_points=False)
    sources = SourceRegistry().discover(use_entry_points=False)
    rules = compile_rules(settings, registry, sources).rules
    (rule,) = [r for r in rules if r.binding is not None]
    hidden = where_view_name("alpha", WHERE)
    assert rule.params.model_dump()["targets"][0]["dataset"] == hidden
    plan = build_plan(rules, settings, ["alpha"])
    (described,) = [d for d in plan.describe() if d.get("where")]
    assert described["datasets"] == ["alpha"]
    assert described["where_columns"] == ["keep"]
    assert hidden in plan.complete(rule.key)


def test_an_unparsable_where_fails_at_load_with_its_origin(tmp_path: Path) -> None:
    path = tmp_path / "qavf.yaml"
    path.write_text(
        "datasets:\n"
        "  alpha:\n"
        "    checks:\n"
        "      - check: fresh\n"
        "        column: seen\n"
        "        max_age: 30d\n"
        "        where: 'keep ='\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as error:
        load_settings(path)
    details = exception_details(error.value)
    assert "datasets.alpha.checks[0].where: invalid where condition" in details
    assert "qavf.yaml:7" in details


def test_an_override_may_change_where() -> None:
    settings = QAVFSettings.model_validate(
        {
            "datasets": {
                "alpha": {
                    "checks": [
                        {"check": "fresh", "name": "recent", "column": "seen", "max_age": "30d"}
                    ]
                }
            },
            "overrides": {"checks": {"recent": {"where": WHERE}}},
        }
    )
    assert settings.datasets["alpha"].checks[0].where == WHERE
    assert where_view_name("alpha", WHERE) in settings.where_views()


@pytest.mark.parametrize("name", ["alpha.__where_0123456789ab", "x.__WHERE_y"])
def test_the_hidden_prefix_is_reserved_for_datasets(name: str) -> None:
    with pytest.raises(ValidationError, match="reserved"):
        QAVFSettings.model_validate({"datasets": {name: {}}})


def test_the_hidden_prefix_is_reserved_for_views() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        QAVFSettings.model_validate(
            {"datasets": {"alpha": {"views": {"__where_0123456789ab": {"where": WHERE}}}}}
        )


def test_a_top_level_check_cannot_name_a_hidden_view() -> None:
    hidden = where_view_name("alpha", WHERE)
    item = {"check": "key_sets_match", "datasets": ["beta", hidden], "keys": ["site_id"]}
    with pytest.raises(ValidationError, match="unknown dataset"):
        QAVFSettings.model_validate(
            {
                "datasets": {
                    "alpha": {"checks": [{"check": "outliers", "column": "aadt", "where": WHERE}]},
                    "beta": {},
                },
                "checks": [item],
            }
        )


def test_the_native_backend_refuses_a_filtered_polars_check() -> None:
    item = {"check": "fresh", "column": "seen", "max_age": "30d", "where": WHERE}
    with pytest.raises(
        OptionalDependencyError, match="the fresh check requires the optional Polars"
    ):
        validate(_alpha(), config={"datasets": {"alpha": {"checks": [item]}}}, backend="duckdb")


def test_a_filtered_expression_keeps_its_native_where() -> None:
    item = {"check": "expression", "expr": "aadt < 1000", "where": WHERE}
    settings = QAVFSettings.model_validate({"datasets": {"alpha": {"checks": [item]}}})
    assert settings.where_views() == {}
    for backend in ("polars", "duckdb"):
        report = validate(
            _alpha(), config={"datasets": {"alpha": {"checks": [item]}}}, backend=backend
        )
        (result,) = _results(report, "alpha")
        assert result.status.value == "PASS", backend


# Configuration origins survive run-time selection and profiles (phase-4 leftover).


def _selected_config(tmp_path: Path) -> Path:
    path = tmp_path / "qavf.yaml"
    path.write_text(
        "datasets:\n"
        "  alpha:\n"
        "    checks:\n"
        "      - check: fresh\n"
        "        column: seen\n"
        "        max_age: 30d\n"
        "  beta: {}\n"
        "checks:\n"
        "  - check: sql\n"
        "    name: nothing\n"
        "    sql: SELECT * FROM alpha WHERE false\n",
        encoding="utf-8",
    )
    return path


def test_selection_and_profiles_keep_the_origins(tmp_path: Path) -> None:
    settings = load_settings(_selected_config(tmp_path))
    assert settings.origins is not None
    narrowed = narrow(settings, Selection.of(datasets="alpha"))
    assert narrowed.origins is settings.origins
    assert apply_profile(settings, "validity").origins is settings.origins


def test_an_error_after_selection_names_its_origin(tmp_path: Path) -> None:
    path = tmp_path / "qavf.yaml"
    path.write_text(
        "plugins:\n"
        "  entry_points: false\n"
        "datasets:\n"
        "  alpha:\n"
        "    checks:\n"
        "      - check: levels\n"
        "        level: high\n"
        "  beta: {}\n",
        encoding="utf-8",
    )
    registry = discovered_rules(entry_points=False)
    registry.register(_LevelRule)
    settings = narrow(load_settings(path), Selection.of(datasets="alpha"))
    with pytest.raises(ConfigurationError) as error:
        QARunner(settings, registry=registry).plan()
    details = exception_details(error.value)
    assert "datasets.alpha.checks[0].level" in details
    assert "qavf.yaml:7" in details


class _FixedPlan(QARunner):
    """Runs exactly the given rule instances."""

    def __init__(self, settings: QAVFSettings, rules: list[Rule]) -> None:
        super().__init__(settings, registry=discovered_rules(entry_points=False))
        self.fixed = rules

    def plan(self) -> list[Rule]:
        self.excluded = {}
        return list(self.fixed)


@pytest.mark.parametrize("item", ITEMS, ids=[item["check"] for item in ITEMS])
def test_a_filtered_check_reports_what_its_rule_reports_on_a_declared_view(
    item: dict[str, Any], tmp_path: Path
) -> None:
    """The same rule, configured on a declared view with the same condition, finds the
    same: a dataset check cannot be placed on a view, so its rule is built directly."""
    item = _with_baseline(item, tmp_path)
    raw: dict[str, Any] = {
        "datasets": {"alpha": {"views": {"recent": {"where": WHERE}}, "checks": [item]}},
        "engine": {"parallel": False},
    }
    filtered_raw = {
        **raw,
        "datasets": {"alpha": {**raw["datasets"]["alpha"], "checks": [{**item, "where": WHERE}]}},
    }
    filtered = _results(validate(_alpha(), config=filtered_raw, backend="polars"), "alpha")
    settings = QAVFSettings.model_validate(filtered_raw)
    runner = QARunner(settings, registry=discovered_rules(entry_points=False))
    (compiled,) = [rule for rule in runner.plan() if rule.binding is not None]
    hidden = where_view_name("alpha", WHERE)
    dumped = json.dumps(compiled.params.model_dump(mode="json", exclude_unset=True))
    params = json.loads(dumped.replace(hidden, "alpha.recent"))
    on_view = type(compiled)(params=params)
    view_results = _FixedPlan(QAVFSettings.model_validate(raw), [on_view]).run(frames=_alpha())
    expected = [r for r in view_results.results if r.rule_id == compiled.rule_id]
    assert filtered and len(filtered) == len(expected)
    for left, right in zip(filtered, expected, strict=True):
        assert _comparable(left) == _comparable(right).replace("DS.recent", "DS")
