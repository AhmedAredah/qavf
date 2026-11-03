"""Configuration reuse: named series and metrics, views, defaults, include, vars, ranges."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.checks.compiler import compile_checks
from qavf.engine.runner import QARunner
from qavf.settings import QAVFSettings, load_settings, settings_from_text
from qavf.testing import discovered_rules

SERIES = {"value": "aadt", "group_by": ["site_id"], "time_column": "year"}
COUNTS = pl.DataFrame(
    {
        "site_id": ["S1", "S1", "S1", "S2", "S2"],
        "year": [2016, 2017, 2019, 2016, 2019],
        "aadt": [100, 110, 120, 50, 60],
    }
)
CRASHES = pl.DataFrame(
    {"site_id": ["S1", "S1", "S2"], "year": [2016, 2017, 2016], "crashes": [3, 4, 1]}
)


def _settings(raw: dict[str, Any]) -> QAVFSettings:
    return QAVFSettings.model_validate({"engine": {"parallel": False}, **raw})


def _params(settings: QAVFSettings) -> list[dict[str, Any]]:
    return [check.params for check in compile_checks(settings)]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------- series, metrics


def test_a_named_series_compiles_like_the_inline_mapping() -> None:
    inline = _settings(
        {"datasets": {"counts": {"checks": [{"check": "change_rate", "series": SERIES}]}}}
    )
    named = _settings(
        {
            "datasets": {
                "counts": {
                    "series": {"aadt_by_site": SERIES},
                    "checks": [{"check": "change_rate", "series": "aadt_by_site"}],
                }
            }
        }
    )
    assert _params(named) == _params(inline)


def test_a_series_of_another_dataset_is_reached_by_its_qualified_name() -> None:
    settings = _settings(
        {
            "datasets": {
                "counts": {"series": {"aadt_by_site": SERIES}},
                "counts_2025": {"checks": [{"check": "trend", "series": "counts.aadt_by_site"}]},
            }
        }
    )
    (params,) = _params(settings)
    assert params["series"] == [{**SERIES, "dataset": "counts_2025"}]


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("nope", "datasets.counts.checks[0].series: unknown series 'nope'"),
        ("other.x", "datasets.counts.checks[0].series: unknown dataset 'other'"),
        ("counts.x", "datasets.counts.checks[0].series: dataset 'counts' defines no series 'x'"),
    ],
)
def test_an_unknown_series_is_a_load_error_with_its_path(reference: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _settings(
            {
                "datasets": {
                    "counts": {
                        "series": {"aadt_by_site": SERIES},
                        "checks": [{"check": "change_rate", "series": reference}],
                    }
                }
            }
        )


def test_a_named_metric_takes_its_normalizer_as_per() -> None:
    metric = {"value": "crashes", "entity": "site_id", "group_by": ["year"], "per": "counts.aadt"}
    settings = _settings(
        {
            "datasets": {
                "counts": {},
                "crashes": {
                    "metrics": {"crash_rate": metric},
                    "checks": [
                        {"check": "entity_outliers", "metric": "crash_rate"},
                        {"check": "group_comparison", "metric": "crash_rate", "groups": {}},
                    ],
                },
            }
        }
    )
    first = _params(settings)[0]["metrics"][0]
    assert first["entity_column"] == "site_id"
    assert first["normalize_by"] == {"dataset": "counts", "value": "aadt"}


def test_an_unknown_metric_is_a_load_error() -> None:
    with pytest.raises(ValidationError, match=r"checks\[0\]\.metric: unknown metric 'rate'"):
        _settings(
            {"datasets": {"crashes": {"checks": [{"check": "entity_outliers", "metric": "rate"}]}}}
        )


# --------------------------------------------------------------------------- views


def _view_config() -> dict[str, Any]:
    return {
        "datasets": {
            "counts": {
                "views": {"recent": {"where": "year >= 2017", "columns": ["site_id", "year"]}}
            },
            "crashes": {},
        },
        "checks": [
            {
                "check": "key_sets_match",
                "name": "recent keys",
                "datasets": ["counts.recent", "crashes"],
                "keys": ["site_id", "year"],
            }
        ],
    }


def test_a_view_is_a_filtered_dataset_with_its_parent_columns() -> None:
    settings = _settings(_view_config())
    report = QARunner(settings, registry=discovered_rules()).run(
        frames={"counts": COUNTS, "crashes": CRASHES}
    )
    (result,) = report.results_for("REC-004")
    assert result.item_name == "recent keys"
    assert result.status.value == "FAIL"
    # counts.recent holds (S1, 2017), (S1, 2019), (S2, 2019); crashes adds the 2016 keys.
    assert result.affected_count == 4
    info = {d.name: d for d in report.metadata.datasets}
    assert info["counts.recent"].rows == 3
    assert info["counts.recent"].columns == ["site_id", "year"]
    assert "counts.recent" not in report.metadata.unused_datasets


def test_an_unknown_view_is_a_load_error() -> None:
    config = _view_config()
    config["checks"][0]["datasets"] = ["counts.old", "crashes"]
    with pytest.raises(
        ValidationError, match=r"checks\[0\]\.datasets: unknown dataset 'counts.old'"
    ):
        _settings(config)


def test_a_view_condition_is_checked_when_loaded() -> None:
    with pytest.raises(ValidationError, match="invalid view condition"):
        _settings({"datasets": {"counts": {"views": {"bad": {"where": "year >="}}}}})


# ------------------------------------------------------------------------ defaults


def test_defaults_fill_settings_an_item_omits() -> None:
    settings = _settings(
        {
            "defaults": {"change_rate": {"max": 0.3, "severity": "WARNING"}},
            "datasets": {
                "counts": {
                    "checks": [
                        {"check": "change_rate", "series": SERIES},
                        {"check": "change_rate", "name": "own", "series": SERIES, "max": 0.1},
                    ]
                }
            },
        }
    )
    first, second = compile_checks(settings)
    assert first.params["max_change_rate"] == 0.3
    assert second.params["max_change_rate"] == 0.1
    assert first.item.severity is not None and first.item.severity.value == "WARNING"


@pytest.mark.parametrize(
    ("defaults", "message"),
    [
        ({"change_rate": {"maxx": 1}}, "defaults.change_rate.maxx: not a setting"),
        ({"expression": {"exprr": "x"}}, "defaults.expression.exprr: not a setting"),
    ],
)
def test_defaults_must_be_settings_of_their_check(defaults: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _settings({"defaults": defaults})


def test_defaults_for_an_unknown_check_fail_when_the_run_is_planned() -> None:
    from qavf.checks.compiler import CheckCompilationError

    settings = _settings({"defaults": {"no_such_check": {"x": 1}}})
    with pytest.raises(CheckCompilationError, match="defaults.no_such_check: unknown check"):
        compile_checks(settings, discovered_rules())


# -------------------------------------------------------------------------- ranges


@pytest.mark.parametrize(
    ("expected", "values"),
    [
        ("2016..2019", [2016, 2017, 2018, 2019]),
        ({"from": 2016, "to": 2020, "step": 2}, [2016, 2018, 2020]),
        ([2016, 2019], [2016, 2019]),
    ],
)
def test_ranges_expand_and_lists_stay_as_written(expected: Any, values: list[int]) -> None:
    settings = _settings(
        {
            "datasets": {
                "counts": {
                    "checks": [{"check": "missing_periods", "series": SERIES, "expected": expected}]
                }
            }
        }
    )
    assert _params(settings)[0]["expected_periods"] == values


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ("2020..2016", "'from' must not exceed 'to'"),
        ({"from": 2016, "to": 2020, "step": 0}, "step must be positive"),
        ({"from": "a", "to": 2020}, "from must be a whole number"),
        ("0..1000000", "more than 100,000"),
    ],
)
def test_a_bad_range_is_a_load_error(expected: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=r"checks\[0\]\.expected: .*" + message):
        _settings(
            {
                "datasets": {
                    "counts": {
                        "checks": [
                            {"check": "missing_periods", "series": SERIES, "expected": expected}
                        ]
                    }
                }
            }
        )


# ---------------------------------------------------------------------------- vars


def test_vars_substitute_whole_values_and_inside_text(tmp_path: Path) -> None:
    config = _write(
        tmp_path / "qa.yaml",
        """
vars:
  years: 2016..2018
  limit: 0.2
  site: S1
datasets:
  counts:
    checks:
      - {check: missing_periods, series: {value: aadt, group_by: [site_id], time_column: year},
         expected: '${vars.years}', description: "site ${vars.site} only"}
      - {check: change_rate, series: {value: aadt, group_by: [site_id], time_column: year},
         max: '${vars.limit}'}
""",
    )
    settings = load_settings(config)
    periods, change = _params(settings)
    assert periods["expected_periods"] == [2016, 2017, 2018]
    assert settings.datasets["counts"].checks[0].description == "site S1 only"
    assert change["max_change_rate"] == 0.2


def test_an_unknown_variable_is_an_error(tmp_path: Path) -> None:
    config = _write(tmp_path / "qa.yaml", "vars: {a: 1}\nproject: '${vars.b}'\n")
    with pytest.raises(ValueError, match=r"project: unknown variable \$\{vars.b\} \(defined: a\)"):
        load_settings(config)


def test_variables_may_refer_to_each_other_but_not_in_a_cycle(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.yaml", "vars: {a: x, b: '${vars.a}y'}\nproject: '${vars.b}'\n")
    assert load_settings(good).project == "xy"
    bad = _write(tmp_path / "bad.yaml", "vars: {a: '${vars.b}', b: '${vars.a}'}\n")
    with pytest.raises(ValueError, match="cycle"):
        load_settings(bad)


def test_vars_work_in_configuration_mappings() -> None:
    settings = _settings({"vars": {"name": "demo"}, "project": "QA ${vars.name}"})
    assert settings.project == "QA demo"


# ------------------------------------------------------------------------- include


SHARED = """
vars: {years: 2016..2017, owner: shared}
defaults:
  change_rate: {max: 0.5}
datasets:
  sites: {path: sites.csv}
checks:
  - {check: sql, name: shared query, sql: SELECT 1 WHERE false}
"""


def test_included_files_merge_first_and_the_including_file_wins(tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir()
    _write(tmp_path / "shared" / "base.yaml", SHARED)
    config = _write(
        tmp_path / "qa.yaml",
        """
include: [shared/base.yaml]
vars: {owner: main}
project: '${vars.owner}'
datasets:
  counts:
    path: counts.csv
    checks:
      - {check: missing_periods, series: {value: aadt, group_by: [site_id], time_column: year},
         expected: '${vars.years}'}
checks:
  - {check: sql, name: own query, sql: SELECT 1 WHERE false}
""",
    )
    settings = load_settings(config)
    assert settings.project == "main"
    assert list(settings.datasets) == ["sites", "counts"]
    assert [item.name for item in settings.checks] == ["shared query", "own query"]
    assert settings.defaults == {"change_rate": {"max": 0.5}}
    assert _params(settings)[0]["expected_periods"] == [2016, 2017]
    assert settings.include == []


def test_include_works_for_unsaved_text(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", SHARED)
    settings = settings_from_text("include: base.yaml\n", tmp_path / "unsaved.yaml")
    assert list(settings.datasets) == ["sites"]


def test_a_dataset_defined_in_two_files_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", SHARED)
    config = _write(
        tmp_path / "qa.yaml", "include: [base.yaml]\ndatasets:\n  sites: {path: x.csv}\n"
    )
    with pytest.raises(ValueError, match="datasets 'sites' is already defined by an included file"):
        load_settings(config)


def test_a_named_check_repeated_across_files_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", SHARED)
    config = _write(
        tmp_path / "qa.yaml",
        "include: [base.yaml]\nchecks:\n  - {check: sql, name: shared query, sql: SELECT 1}\n",
    )
    with pytest.raises(ValueError, match="'shared query' is already defined by an included file"):
        load_settings(config)


def test_an_include_cycle_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "include: [b.yaml]\n")
    _write(tmp_path / "b.yaml", "include: [a.yaml]\n")
    with pytest.raises(ValueError, match=r"include cycle: .*a\.yaml -> .*b\.yaml -> .*a\.yaml"):
        load_settings(tmp_path / "a.yaml")


def test_a_missing_include_is_an_error(tmp_path: Path) -> None:
    config = _write(tmp_path / "qa.yaml", "include: [nope.yaml]\n")
    with pytest.raises(ValueError, match="cannot include 'nope.yaml'"):
        load_settings(config)


def test_include_in_a_mapping_is_rejected() -> None:
    with pytest.raises(ValidationError, match="include is resolved when a configuration file"):
        _settings({"include": ["base.yaml"]})


def test_validate_reads_included_files(tmp_path: Path) -> None:
    from qavf import validate

    _write(
        tmp_path / "base.yaml",
        "datasets:\n  counts:\n    checks:\n      - {check: expression, expr: aadt > 0}\n",
    )
    config = _write(tmp_path / "qa.yaml", "include: [base.yaml]\nengine: {parallel: false}\n")
    report = validate({"counts": COUNTS}, config=config)
    assert [r.status.value for r in report.results_for("INP-006")] == ["PASS"]
