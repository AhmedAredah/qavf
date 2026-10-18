"""Configuration-driven validation rules (INP-*)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from qavf.engine.severity import Severity, Status
from qavf.testing import rule_results, run_rule

SCHEMAS = {
    "parent": {"schema": {"primary_key": ["id"], "columns": {"id": {"dtype": "int"}}}},
    "child": {
        "schema": {
            "primary_key": ["id", "year"],
            "columns": {
                "id": {"dtype": "int"},
                "year": {"dtype": "int", "isin": [2020, 2025]},
                "value": {"dtype": "float", "ge": 0, "le": 1000},
            },
        },
        "foreign_keys": [{"columns": ["id"], "ref_dataset": "parent"}],
    },
}
INPUT_RULES = [f"INP-00{i}" for i in range(1, 8)]


def _child(small, **schema_columns) -> dict:
    """A datasets mapping whose child schema has the given column specs."""
    return {"child": {"schema": {"columns": schema_columns}}}


def _failures(results):
    return [r for r in results if r.status is Status.FAIL]


@pytest.mark.parametrize("rule_id", INPUT_RULES[1:5])
def test_clean_inputs_pass(rule_id: str, small, registry) -> None:
    result = run_rule(rule_id, small, datasets=SCHEMAS, registry=registry)
    assert result.status is Status.PASS, result.message


@pytest.mark.parametrize("rule_id", INPUT_RULES)
def test_rules_skip_without_configuration(rule_id: str, small, registry) -> None:
    assert run_rule(rule_id, small, registry=registry).status is Status.SKIPPED


# ----------------------------------------------------------------- INP-001
def test_dataset_availability(tmp_path, small, registry) -> None:
    datasets = {
        **SCHEMAS,
        "required_missing": {"path": str(tmp_path / "absent.csv")},
        "optional_missing": {"path": str(tmp_path / "absent2.csv"), "required": False},
        "optional_broken": {
            "source": {"type": "query", "sql": "SELECT * FROM nope"},
            "required": False,
        },
    }
    results = rule_results("INP-001", small, datasets=datasets, registry=registry)
    by_check = {r.check: r for r in results}
    assert by_check["required_missing"].severity is Severity.CRITICAL
    assert by_check["optional_broken"].severity is Severity.WARNING
    assert "optional_missing" not in by_check  # optional and simply absent: fine
    assert any(r.status is Status.PASS for r in results)


# ----------------------------------------------------------------- INP-002
def test_schema_conformance(small, registry) -> None:
    child = (
        small["child"]
        .drop("year")
        .with_columns(pl.col("value").cast(pl.String), pl.lit(1).alias("surprise"))
    )
    datasets = {
        **SCHEMAS,
        "child": {"schema": {**SCHEMAS["child"]["schema"], "strict": True, "min_rows": 10}},
    }
    result = run_rule("INP-002", {**small, "child": child}, datasets=datasets, registry=registry)
    assert result.status is Status.FAIL and result.check == "child"
    problems = {(r["column"], r["problem"]) for r in result.affected_records}
    assert ("year", "missing required column") in problems
    assert ("surprise", "undeclared column") in problems
    assert ("value", "expected float, found String") in problems
    assert (None, "4 rows, expected at least 10") in problems


def test_logical_types_including_dates() -> None:
    frame = pl.DataFrame(
        {"i": [1], "f": [1], "d": [date(2020, 1, 1)], "t": [date(2020, 1, 1)], "b": [True]}
    ).with_columns(pl.col("t").cast(pl.Datetime))
    datasets = {
        "x": {
            "schema": {
                "columns": {
                    "i": {"dtype": "int"},
                    "f": {"dtype": "float"},
                    "d": {"dtype": "date"},
                    "t": {"dtype": "datetime"},
                    "b": {"dtype": "bool"},
                },
                "max_rows": 1,
            }
        }
    }
    assert run_rule("INP-002", {"x": frame}, datasets=datasets).status is Status.PASS
    wrong = {"x": {"schema": {"columns": {"d": {"dtype": "int"}}, "min_rows": 2}}}
    problems = {
        r["problem"] for r in run_rule("INP-002", {"x": frame}, datasets=wrong).affected_records
    }
    assert problems == {"expected int, found Date", "1 rows, expected at least 2"}


# ----------------------------------------------------------------- INP-003
def test_each_column_check_is_graded_at_its_own_severity(small) -> None:
    frame = small["child"].with_columns(pl.Series("value", [-5.0, 12.0, 2000.0, 30.0]))
    datasets = _child(
        small,
        value={
            "ge": 0,
            "checks": [
                {"le": 1000, "severity": "WARNING", "name": "plausible"},
                {"le": 5000, "severity": "CRITICAL", "name": "hard ceiling"},
            ],
        },
    )
    results = rule_results("INP-003", {**small, "child": frame}, datasets=datasets)
    failed = {r.check: r for r in _failures(results)}
    assert set(failed) == {"child.value: ge 0", "child.value: plausible"}
    assert failed["child.value: ge 0"].severity is Severity.ERROR
    assert failed["child.value: plausible"].severity is Severity.WARNING
    record = failed["child.value: ge 0"].affected_records[0]
    assert record["_row"] == 0 and record["value"] == -5.0 and record["violates"] == "ge 0"
    passed = [r for r in results if r.status is Status.PASS]
    assert len(passed) == 1 and "column check" in passed[0].message  # hard ceiling + not null


def test_combined_limits_report_each_row_once_with_reasons() -> None:
    frame = pl.DataFrame({"x": [-1.0, 5.0, 50.0, None]})
    datasets = {"t": {"schema": {"columns": {"x": {"nullable": True, "ge": 0, "lt": 10}}}}}
    (failed,) = _failures(rule_results("INP-003", {"t": frame}, datasets=datasets))
    assert failed.affected_count == 2 and failed.total_records == 4
    assert [r["violates"] for r in failed.affected_records] == ["ge 0", "lt 10"]


def test_nulls_and_null_fractions() -> None:
    frame = pl.DataFrame({"a": [1, None, None, 4], "b": [1, None, 3, 4]})
    datasets = {
        "t": {
            "schema": {
                "columns": {
                    "a": {},
                    "b": {"nullable": True, "max_null_fraction": 0.3},
                }
            }
        }
    }
    failed = {
        r.check: r for r in _failures(rule_results("INP-003", {"t": frame}, datasets=datasets))
    }
    assert failed["t.a: not null"].affected_count == 2
    assert "t.b: max_null_fraction 0.3" not in failed  # 25 % is within 30 %
    datasets["t"]["schema"]["columns"]["b"]["max_null_fraction"] = 0.1
    failed = {r.check for r in _failures(rule_results("INP-003", {"t": frame}, datasets=datasets))}
    assert "t.b: max_null_fraction 0.1" in failed


def test_text_limits() -> None:
    frame = pl.DataFrame({"code": ["AB12", "ab12", "AB1", None], "kind": ["x", "y", "-999", "x"]})
    datasets = {
        "t": {
            "schema": {
                "columns": {
                    "code": {"nullable": True, "regex": "[A-Z]{2}\\d{2}"},
                    "kind": {"notin": ["-999"], "checks": [{"min_length": 2, "name": "long"}]},
                }
            }
        }
    }
    failed = {
        r.check: r for r in _failures(rule_results("INP-003", {"t": frame}, datasets=datasets))
    }
    assert [r["code"] for r in failed["t.code: regex [A-Z]{2}\\d{2}"].affected_records] == [
        "ab12",
        "AB1",
    ]
    assert failed["t.kind: notin [-999]"].affected_count == 1
    assert failed["t.kind: long"].affected_count == 3


def test_date_bounds_and_type_mismatches() -> None:
    frame = pl.DataFrame({"d": [date(2020, 1, 1), date(2031, 1, 1)], "n": [1, 2]})
    datasets = {
        "t": {
            "schema": {
                "columns": {
                    "d": {"le": date(2030, 12, 31)},
                    "n": {
                        "checks": [
                            {"ge": date(2020, 1, 1), "name": "date on number"},
                            {"regex": "x", "name": "regex on number"},
                        ]
                    },
                }
            }
        }
    }
    failed = {
        r.check: r for r in _failures(rule_results("INP-003", {"t": frame}, datasets=datasets))
    }
    assert failed["t.d: le 2030-12-31"].affected_records[0]["d"] == date(2031, 1, 1)
    # A limit that cannot apply to the column's type is not assessed, never a violation.
    results = {r.check: r for r in rule_results("INP-003", {"t": frame}, datasets=datasets)}
    for check in ("t.n: date on number", "t.n: regex on number"):
        assert results[check].status is Status.ERROR
        assert "Not assessed" in results[check].message
        assert "cannot apply" in results[check].message


def test_aggregate_limits_and_uniqueness() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 3.0], "s": ["a", "b", "b", "c"]})
    datasets = {
        "t": {
            "schema": {
                "columns": {
                    "x": {
                        "checks": [
                            {"mean": {"ge": 1, "le": 3}, "name": "mean ok"},
                            {"max": {"le": 2.5}, "name": "max too high"},
                            {"sum": {"ge": 100}, "name": "sum too low"},
                            {"unique": True, "name": "unique"},
                        ]
                    },
                    "s": {
                        "checks": [
                            {"distinct_count": {"ge": 3, "le": 3}, "name": "three kinds"},
                            {"mean": {"ge": 0}, "name": "bad"},
                        ]
                    },
                }
            }
        }
    }
    failed = {
        r.check: r for r in _failures(rule_results("INP-003", {"t": frame}, datasets=datasets))
    }
    assert set(failed) == {"t.x: max too high", "t.x: sum too low", "t.x: unique"}
    assert "max 3 outside [, 2.5]" in failed["t.x: max too high"].message
    assert "2 values are not unique" in failed["t.x: unique"].message
    # A statistic that needs numbers, on text, cannot be assessed: not a data violation.
    results = {r.check: r for r in rule_results("INP-003", {"t": frame}, datasets=datasets)}
    assert results["t.s: bad"].status is Status.ERROR
    assert "requires a numeric column" in results["t.s: bad"].message


def test_invalid_check_definitions_are_rejected() -> None:
    from pydantic import ValidationError

    from qavf.settings import Bounds, ColumnCheck

    with pytest.raises(ValidationError, match="at least one limit"):
        ColumnCheck(severity="WARNING")
    with pytest.raises(ValidationError, match="must not exceed"):
        Bounds(ge=2, le=1)


# ----------------------------------------------------------------- INP-004/005
def test_duplicate_primary_keys(small) -> None:
    child = pl.concat([small["child"], small["child"].head(1), small["child"].head(1)])
    result = run_rule("INP-004", {**small, "child": child}, datasets=SCHEMAS)
    assert result.status is Status.FAIL and result.check == "child"
    assert result.affected_records[0]["occurrences"] == 3


def test_referential_integrity_with_per_key_severity(small) -> None:
    orphaned = pl.concat(
        [small["child"], pl.DataFrame({"id": [9], "year": [2020], "value": [1.0]})]
    )
    datasets = {**SCHEMAS, "child": {**SCHEMAS["child"]}}
    datasets["child"]["foreign_keys"] = [
        {"columns": ["id"], "ref_dataset": "parent", "severity": "CRITICAL"}
    ]
    result = run_rule("INP-005", {**small, "child": orphaned}, datasets=datasets)
    assert result.status is Status.FAIL and result.severity is Severity.CRITICAL
    assert [(r["id"], r["rows"]) for r in result.affected_records] == [(9, 1)]
    as_float = {
        "parent": small["parent"].with_columns(pl.col("id").cast(pl.Float64)),
        "child": small["child"],
    }
    assert run_rule("INP-005", as_float, datasets=SCHEMAS).status is Status.PASS


# ----------------------------------------------------------------- INP-006/007
def test_row_expressions(demo) -> None:
    crashes = demo["crashes"].with_columns(
        pl.when(pl.col("site_id") == "S01")
        .then(pl.col("crashes") + 100)
        .otherwise(pl.col("fatal"))
        .alias("fatal")
    )
    datasets = {
        "crashes": {
            "expressions": [
                {"expr": "fatal <= crashes", "name": "fatal within total", "severity": "CRITICAL"},
                {"expr": "site_id IN (SELECT site_id FROM sites)", "name": "known site"},
                {"expr": "no_such_column > 0", "name": "broken"},
            ]
        }
    }
    results = rule_results("INP-006", {**demo, "crashes": crashes}, datasets=datasets)
    failed = {r.check: r for r in _failures(results)}
    assert failed["crashes: fatal within total"].severity is Severity.CRITICAL
    assert failed["crashes: fatal within total"].affected_count == 9  # one per year for S01
    assert "crashes: known site" not in failed  # cross-dataset subquery holds
    (broken,) = [r for r in results if r.status is Status.ERROR]  # not assessed, not failed
    assert broken.check == "crashes: broken" and "could not be evaluated" in broken.message


def test_expression_null_handling() -> None:
    frame = pl.DataFrame({"a": [1, None], "b": [0, 0]})
    lenient = {"t": {"expressions": [{"expr": "a > b"}]}}
    assert run_rule("INP-006", {"t": frame}, datasets=lenient).status is Status.PASS
    strict = {"t": {"expressions": [{"expr": "a > b", "null_as": "fail"}]}}
    assert run_rule("INP-006", {"t": frame}, datasets=strict).affected_count == 1


def test_sql_assertions(demo) -> None:
    assertions = [
        {
            "name": "sites have crashes",
            "sql": "SELECT site_id FROM sites WHERE site_id NOT IN (SELECT site_id FROM crashes)",
        },
        {"name": "big years", "sql": "SELECT * FROM counts WHERE aadt > 50000", "severity": "INFO"},
        {"name": "typo", "sql": "SELECT * FROM nowhere"},
    ]
    results = rule_results("INP-007", demo, assertions=assertions)
    failed = {r.check: r for r in _failures(results)}
    assert set(failed) == {"big years"}
    assert failed["big years"].severity is Severity.INFO and failed["big years"].affected_count > 0
    assert failed["big years"].datasets == ["counts"]
    (typo,) = [r for r in results if r.status is Status.ERROR]  # not assessed, not failed
    assert typo.check == "typo" and "could not be evaluated" in typo.message
