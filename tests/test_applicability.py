"""N/A is a distinct, opt-in state across source adapters and validation checks."""

from __future__ import annotations

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from qavf import Query, validate
from qavf.applicability import NotApplicablePolicy
from qavf.engine.severity import Status
from qavf.settings import ColumnSpec, DatasetSpec, SchemaSpec
from qavf.testing import rule_results


def schema():
    return {
        "columns": {
            "x": {"dtype": "int", "na_values": ["N/A"], "ge": 0, "max_null_fraction": 0.25},
            "y": {"dtype": "int", "na_values": ["-"], "ge": 0},
        },
        "na_groups": [["x", "y"]],
    }


@pytest.mark.parametrize("source", ["polars", "lazy", "arrow", "pandas", "csv", "parquet", "query"])
def test_applicability_across_inputs(source, tmp_path):
    data = {"x": ["N/A", "7", None, "-1"], "y": ["-", "8", "2", "3"]}
    if source == "polars":
        value = pl.DataFrame(data)
    elif source == "lazy":
        value = pl.DataFrame(data).lazy()
    elif source == "arrow":
        value = pa.table(data)
    elif source == "pandas":
        value = pd.DataFrame(data)
    elif source == "csv":
        value = tmp_path / "values.csv"
        value.write_text("x,y\nN/A,-\n7,8\n,2\n-1,3\n")
    elif source == "parquet":
        value = tmp_path / "values.parquet"
        pq.write_table(pa.table(data), value)
    else:
        value = Query(
            "SELECT * FROM (VALUES ('N/A', '-'), ('7', '8'), (NULL, '2'), ('-1', '3')) t(x, y)"
        )
    report = validate(value, schema=schema(), profile="validity", backend="polars")
    results = report.results
    assert all(result.status is Status.PASS for result in results if result.rule_id == "INP-002")
    failures = [
        result for result in results if result.rule_id == "INP-003" and result.status is Status.FAIL
    ]
    assert sorted(result.affected_count for result in failures) == [1, 2]
    assert all(result.total_records == 3 for result in failures)
    populations = [result for result in results if result.check_id == "applicability.population"]
    assert [result.metrics["na_count"] for result in populations] == [1, 1]
    groups = [result for result in results if result.check_id == "applicability.all_or_none"]
    assert len(groups) == 1 and groups[0].status is Status.PASS


def test_partial_group_counts_exact_with_bounded_evidence():
    frame = pl.DataFrame({"x": ["N/A", "N/A", None, "bad", "1"], "y": ["-", "1", "-", "-", None]})
    results = rule_results(
        "INP-009", {"t": frame}, datasets={"t": {"schema": schema()}}, max_affected_records=1
    )
    group = next(result for result in results if result.check_id == "applicability.all_or_none")
    assert group.affected_count == 3
    assert group.total_records == 5
    assert len(group.affected_records) == 1
    assert group.columns == ["x", "y"]
    assert group.parameters["na_values"] == {"x": ["N/A"], "y": ["-"]}


def test_na_does_not_become_invalid_or_missing():
    frame = pl.DataFrame({"x": ["N/A", "bad", None, "7"]})
    datasets = {
        "t": {
            "schema": {
                "columns": {"x": {"dtype": "int", "na_values": ["N/A"], "max_null_fraction": 0.5}}
            }
        }
    }
    invalid = rule_results("INP-002", {"t": frame}, datasets=datasets)
    assert sum(result.affected_count or 0 for result in invalid) == 1
    limits = rule_results("INP-003", {"t": frame}, datasets=datasets)
    failed = [result for result in limits if result.status is Status.FAIL]
    assert len(failed) == 2  # nullability and missing fraction on 3 applicable cells
    assert all(result.affected_count == 2 and result.total_records == 3 for result in failed)


def test_all_na_numeric_statistics_are_undefined():
    datasets = {
        "t": {
            "schema": {
                "columns": {
                    "x": {"dtype": "int", "na_values": ["N/A"], "stats": {"sum": {"ge": 0}}}
                }
            }
        }
    }
    results = rule_results("INP-003", {"t": pl.DataFrame({"x": ["N/A", "N/A"]})}, datasets=datasets)
    undefined = next(result for result in results if result.status is Status.ERROR)
    assert "undefined" in undefined.message and "every value is N/A" in undefined.message
    assert undefined.metrics["na_count"] == 2


@pytest.mark.parametrize(
    "values",
    [
        {"na_values": ["N/A"], "missing_values": [" N/A "]},
        {"na_values": ["N/A", " N/A "]},
        {"na_values": [" "]},
        {"dtype": "bool", "na_values": ["TRUE"]},
        {"dtype": "int", "na_values": ["07"]},
        {"dtype": "str", "na_values": ["N/A"], "isin": ["N/A"]},
        {"dtype": "str", "na_values": ["N/A"], "isin": ["n/a"], "normalize": ["case"]},
        {"dtype": "str", "na_values": ["N/A"], "notin": ["N/A"]},
        {"dtype": "date", "na_values": ["2026-01-01"]},
    ],
)
def test_ambiguous_tokens_rejected(values):
    with pytest.raises(ValidationError, match="na_values"):
        ColumnSpec.model_validate(values)


@pytest.mark.parametrize("groups", [[["x"]], [["x", "x"]], [["x", "z"]], [["x", "y"], ["y", "x"]]])
def test_invalid_groups_rejected(groups):
    with pytest.raises(ValidationError):
        SchemaSpec.model_validate({**schema(), "na_groups": groups})


@pytest.mark.parametrize(
    "trim,token,expected",
    [
        ("ascii", " N/A ", True),
        ("ascii", "\u00a0N/A\u00a0", False),
        ("unicode", "\u00a0N/A\u00a0", True),
        ("none", " N/A ", False),
    ],
)
def test_token_compilers_agree(trim, token, expected):
    policy = NotApplicablePolicy(("N/A",), trim)
    with duckdb.connect() as con:
        con.register("data", pa.table({"x": [token, None]}))
        actual = con.execute(f"SELECT {policy.sql('x')} FROM data").fetchall()
    assert actual == [(expected,), (False,)]
    assert pl.DataFrame({"x": [token, None]}).select(policy.polars("x")).to_series().to_list() == [
        expected,
        False,
    ]
    assert policy.matches(token) is expected


def test_csv_cannot_erase_na_tokens():
    spec = DatasetSpec.model_validate(
        {
            "source": {"type": "file", "path": "x.csv", "options": {"nullstr": ["N/A"]}},
            "schema": schema(),
        }
    )
    with pytest.raises(ValueError, match="nullstr would erase"):
        spec.source_config()


def test_na_columns_are_read_as_text_and_other_columns_still_convert():
    spec = DatasetSpec.model_validate(
        {
            "path": "x.csv",
            "schema": {
                "columns": {"x": {"dtype": "int", "na_values": ["N/A"]}, "y": {"dtype": "int"}}
            },
        }
    )
    config = spec.source_config()
    assert config["text_columns"] == ["x"]
    assert set(config["declared_columns"]) == {"y"}


def test_default_na_spelling_remains_invalid():
    results = rule_results(
        "INP-002",
        {"t": pl.DataFrame({"x": ["N/A"]})},
        datasets={"t": {"schema": {"columns": {"x": {"dtype": "int"}}}}},
    )
    assert any(result.status is Status.FAIL for result in results)


def test_unconfigured_applicability_does_not_assess_unavailable_dataset(tmp_path):
    results = rule_results(
        "INP-009",
        {},
        datasets={
            "t": {
                "path": str(tmp_path / "absent.csv"),
                "schema": {"columns": {"x": {"dtype": "int"}}},
            }
        },
    )
    assert len(results) == 1 and results[0].status is Status.SKIPPED


@pytest.mark.parametrize("extra", [{}, {"_row": [101, 102, 103, 104]}])
def test_applicable_filter_preserves_original_evidence_row(extra):
    frame = pl.DataFrame({"x": ["N/A", "7", "N/A", "-1"], **extra})
    results = rule_results(
        "INP-003",
        {"t": frame},
        datasets={
            "t": {
                "schema": {
                    "evidence_columns": "all",
                    "columns": {"x": {"dtype": "int", "na_values": ["N/A"], "ge": 0}},
                }
            }
        },
    )
    result = next(result for result in results if result.status is Status.FAIL)
    assert result.affected_records[0]["qavf__row" if extra else "_row"] == 3
    assert not any("applicability_row" in key for key in result.affected_records[0])
    if extra:
        assert result.affected_records[0]["_row"] == 104
