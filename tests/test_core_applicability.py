"""Differential applicability behavior, including portable native execution."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qavf import Dataset, File, Loader, Query, validate


def inputs(kind, tmp_path, data):
    if kind == "arrow":
        return pa.table(data)
    if kind == "pandas":
        return pd.DataFrame(data)
    if kind == "polars":
        return pl.DataFrame(data)
    if kind == "lazy":
        return pl.DataFrame(data).lazy()
    if kind == "loader":
        return Loader(lambda: pa.table(data))
    if kind in {"csv", "parquet"}:
        path = tmp_path / f"values.{kind}"
        if kind == "csv":
            pl.DataFrame(data).write_csv(path)
        else:
            pq.write_table(pa.table(data), path)
        return path
    return Query(
        "SELECT * FROM (VALUES ('N/A', '-'), ('07', '8'), ('N/A', '1'), (NULL, '-')) t(x,y)"
    )


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize(
    "kind", ["arrow", "pandas", "polars", "lazy", "loader", "csv", "parquet", "query"]
)
def test_groups_and_exact_populations_across_backends(backend, kind, tmp_path):
    data = {"x": ["N/A", "07", "N/A", None], "y": ["-", "8", "1", "-"]}
    value = inputs(kind, tmp_path, data)
    report = validate(
        value,
        schema={
            "columns": {
                "x": {"dtype": "int", "na_values": ["N/A"], "nullable": True, "ge": 0},
                "y": {"dtype": "int", "na_values": ["-"], "nullable": True, "ge": 0},
            },
            "na_groups": [["x", "y"]],
        },
        config={"engine": {"max_affected_records": 1}},
        profile="validity",
        backend=backend,
    )
    errors = [result.message for result in report.results if result.status.value == "ERROR"]
    assert not errors, errors
    populations = [
        result for result in report.results if result.check_id == "applicability.population"
    ]
    assert {tuple(result.columns): result.metrics for result in populations} == {
        ("x",): {"na_count": 2, "applicable_count": 2},
        ("y",): {"na_count": 2, "applicable_count": 2},
    }
    group = next(
        result for result in report.results if result.check_id == "applicability.all_or_none"
    )
    assert group.affected_count == 2 and group.total_records == 4 and group.affected_count_exact
    assert len(group.affected_records) == 1
    assert group.parameters == {"na_values": {"x": ["N/A"], "y": ["-"]}}


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_na_whitespace_grammar_and_options(backend):
    report = validate(
        pa.table({"x": ["\u00a0N/A\u00a0", "+07.0", "8"]}),
        backend=backend,
        schema={
            "columns": {
                "x": {
                    "dtype": "int",
                    "na_values": ["N/A"],
                    "trim_whitespace": "unicode",
                    "number_grammar": {"int_decimal": True},
                    "isin": [7, 8],
                    "each_allowed_once": True,
                }
            }
        },
    )
    assert report.verdict() == "passed", [
        (result.rule_id, result.message)
        for result in report.results
        if result.status.value in {"FAIL", "ERROR"}
    ]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_unit_checks_do_not_reject_valid_na(backend):
    report = validate(
        pa.table({"x": ["N/A", "7"], "unit": ["not measured", "km"]}),
        backend=backend,
        schema={
            "columns": {
                "x": {
                    "dtype": "int",
                    "na_values": ["N/A"],
                    "unit_column": "unit",
                    "unit": "km",
                }
            }
        },
    )
    assert report.verdict() == "passed", [
        (result.rule_id, result.message)
        for result in report.results
        if result.status.value in {"FAIL", "ERROR"}
    ]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("statistic", ["sum", "mean", "min", "max"])
def test_all_na_aggregates_cannot_claim_pass(backend, statistic):
    report = validate(
        pa.table({"x": ["N/A", "N/A"]}),
        backend=backend,
        schema={
            "columns": {
                "x": {
                    "dtype": "int",
                    "na_values": ["N/A"],
                    statistic: {"ge": 0},
                }
            }
        },
    )
    incomplete = [
        result
        for result in report.results
        if result.rule_id == "INP-003" and result.status.value == "ERROR"
    ]
    assert incomplete, [
        (result.status.value, result.check_id, result.message) for result in report.results
    ]
    assert all(result.metrics.get("na_count") == 2 for result in incomplete)


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_na_scope_requires_enough_applicable_rows(backend):
    report = validate(
        pa.table({"x": ["N/A", "7"], "selected": [True, False]}),
        backend=backend,
        schema={
            "columns": {
                "x": {
                    "dtype": "int",
                    "na_values": ["N/A"],
                    "checks": [{"ge": 0, "where": "selected = TRUE", "min_applicable_rows": 1}],
                }
            }
        },
    )
    assert any(result.status.value == "ERROR" for result in report.results), [
        (result.status.value, result.message) for result in report.results
    ]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_raw_na_expression_is_not_silently_null(backend):
    report = validate(
        Dataset(
            pa.table({"x": ["N/A", "7"]}),
            schema={
                "columns": {
                    "x": {
                        "dtype": "int",
                        "na_values": ["N/A"],
                    }
                }
            },
            expressions=[{"expr": "x IS NOT NULL", "null_as": "fail"}],
        ),
        backend=backend,
    )
    assert report.verdict() == "passed", [
        (result.rule_id, result.message)
        for result in report.results
        if result.status.value in {"FAIL", "ERROR"}
    ]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize(
    "option, marker", [("nullstr", "N/A"), ("NULLSTR", "N/A"), ("nullstr", " N/A ")]
)
def test_file_descriptor_cannot_erase_na(backend, tmp_path, option, marker):
    path = tmp_path / "values.csv"
    path.write_text(f"x\n{marker}\n7\n")
    try:
        report = validate(
            File(path, options={option: marker}),
            backend=backend,
            schema={"columns": {"x": {"dtype": "int", "nullable": True, "na_values": ["N/A"]}}},
        )
    except ValueError as exc:
        assert "nullstr would erase" in str(exc)
        return
    assert report.verdict() != "passed"


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_empty_applicability_groups_have_exact_zero_counts(backend):
    report = validate(
        pa.table({"x": pa.array([], type=pa.string()), "y": pa.array([], type=pa.string())}),
        backend=backend,
        schema={
            "min_rows": 0,
            "columns": {name: {"dtype": "int", "na_values": ["N/A"]} for name in ["x", "y"]},
            "na_groups": [["x", "y"]],
        },
    )
    assert report.verdict() == "passed"
    groups = [result for result in report.results if result.rule_id == "INP-009"]
    assert len(groups) == 3 and all(result.total_records == 0 for result in groups)


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_na_tokens_do_not_match_already_typed_numbers(backend):
    report = validate(
        pa.table({"x": [7, 8]}), backend=backend, schema={"columns": {"x": {"na_values": ["7"]}}}
    )
    population = next(
        result for result in report.results if result.check_id == "applicability.population"
    )
    assert population.metrics == {"na_count": 0, "applicable_count": 2}


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_all_na_is_not_a_missing_fraction_failure(backend):
    report = validate(
        pa.table({"x": ["N/A", "N/A"]}),
        backend=backend,
        schema={
            "columns": {
                "x": {
                    "dtype": "int",
                    "na_values": ["N/A"],
                    "max_null_fraction": 0,
                }
            }
        },
    )
    assert report.verdict() == "passed"


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_csv_applicability_survives_column_rename(backend, tmp_path):
    path = tmp_path / "renamed.csv"
    path.write_text("raw\nN/A\n7\n")
    report = validate(
        Dataset(
            path,
            rename={"raw": "x"},
            schema={
                "columns": {
                    "x": {"dtype": "int", "na_values": ["N/A"]},
                }
            },
        ),
        backend=backend,
    )
    assert report.verdict() == "passed", [
        result.message for result in report.results if result.status.value == "ERROR"
    ]
    population = next(
        result for result in report.results if result.check_id == "applicability.population"
    )
    assert population.columns == ["x"] and population.metrics["na_count"] == 1
