"""Key joins retain complete grouped relations in the backend until reduction."""

from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from qavf import Query, Status, validate
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation, Observations
from qavf.rules._execution import KeyRelation, align_key_relations


def _inputs(kind):
    if kind == "native":
        return {
            "child": Query("SELECT range AS id FROM range(10002)"),
            "parent": Query("SELECT range AS id FROM range(10000)"),
        }
    child = pl.LazyFrame().select(pl.int_range(0, 10002).alias("id"))
    parent = pl.LazyFrame().select(pl.int_range(0, 10000).alias("id"))
    if kind == "eager":
        child, parent = child.collect(), parent.collect()
    return {"child": child, "parent": parent}


@pytest.mark.parametrize("kind", ["native", "lazy", "eager"])
@pytest.mark.parametrize("rule", ["INP-005", "REC-004"])
def test_key_join_transfers_only_summary_and_bounded_sample(kind, rule):
    config = {
        "datasets": {"child": {}, "parent": {}},
        "plugins": {"entry_points": False},
        "engine": {"parallel": False, "run_unconfigured_rules": False, "max_affected_records": 1},
        "rules": {rule: {}},
    }
    if rule == "INP-005":
        config["datasets"]["child"]["foreign_keys"] = [
            {"columns": ["id"], "ref_dataset": "parent", "ref_columns": ["id"]},
        ]
    else:
        config["rules"][rule] = {
            "params": {
                "pairs": [
                    {
                        "left": {"dataset": "child", "columns": ["id"]},
                        "right": {"dataset": "parent", "columns": ["id"]},
                    }
                ]
            }
        }
    returned = []
    original_sql, original_collect = DataContext.sql, DataContext.collect_many
    original_observed_collect = Observation.collect
    original_combined_collect = Observations.collect

    def record(frame):
        returned.append(frame.height)
        # Summaries must not hide a complete key table inside a one-row list.
        for column in frame:
            if isinstance(column.dtype, pl.List):
                assert (column.list.len().max() or 0) <= 1
        return frame

    def sql(self, *args, **kwargs):
        frame = original_sql(self, *args, **kwargs)
        return record(frame)

    def collect(self, *args, **kwargs):
        frame = original_collect(self, *args, **kwargs)
        return record(frame)

    def observed_collect(self, *args, **kwargs):
        return record(original_observed_collect(self, *args, **kwargs))

    def combined_collect(self, *args, **kwargs):
        return record(original_combined_collect(self, *args, **kwargs))

    inputs = _inputs(kind)
    with (
        patch.object(DataContext, "sql", sql),
        patch.object(DataContext, "collect_many", collect),
        patch.object(Observation, "collect", observed_collect),
        patch.object(Observations, "collect", combined_collect),
        patch(
            "qavf.rules.input_checks.references.key_frequencies",
            side_effect=AssertionError("eager keys"),
        ),
        patch.object(DataContext, "project", side_effect=AssertionError("eager keys")),
    ):
        report = validate(inputs, config=config)
    findings = [result for result in report.results if result.rule_id == rule]
    assert len(findings) == 1
    assert findings[0].status is Status.FAIL, findings[0].message
    assert findings[0].affected_count == 2
    assert len(findings[0].affected_records) == 1
    assert returned and max(returned) == 1
    if rule == "INP-005":
        assert findings[0].affected_rows == 2
        assert findings[0].total_records == 10002


def test_mixed_backend_foreign_key_keeps_projected_fallback_correct():
    report = validate(
        {
            "child": Query("SELECT * FROM (VALUES (1), (3), (3)) t(id)"),
            "parent": pl.DataFrame({"id": [1, 2]}).lazy(),
        },
        config={
            "datasets": {
                "child": {"foreign_keys": [{"columns": ["id"], "ref_dataset": "parent"}]},
                "parent": {},
            },
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "rules": {"INP-005": {}},
        },
    )
    finding = report.results[0]
    assert finding.status is Status.FAIL
    assert finding.affected_count == 1 and finding.affected_rows == 2


@pytest.mark.parametrize("kind", ["native", "lazy"])
def test_key_relations_preserve_lossless_numeric_alignment(kind):
    inputs = (
        {
            "child": Query("SELECT 7::UINTEGER AS id"),
            "parent": Query("SELECT 7::BIGINT AS code"),
        }
        if kind == "native"
        else {
            "child": pl.DataFrame({"id": pl.Series([7], dtype=pl.UInt32)}).lazy(),
            "parent": pl.DataFrame({"code": pl.Series([7], dtype=pl.Int64)}).lazy(),
        }
    )
    report = validate(
        inputs,
        config={
            "datasets": {
                "child": {
                    "foreign_keys": [
                        {
                            "columns": ["id"],
                            "ref_dataset": "parent",
                            "ref_columns": ["code"],
                        }
                    ]
                },
                "parent": {},
            },
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "rules": {"INP-005": {}},
        },
    )
    assert report.exit_code() == 0, [result.message for result in report.results]


@pytest.mark.parametrize("dtype", [pl.String(), pl.Int64(), pl.Date()])
@pytest.mark.parametrize("backend", ["native", "lazy"])
def test_empty_null_key_relation_aligns_to_supported_counterpart(dtype, backend):
    with DataContext() as ctx:
        left = KeyRelation(
            ctx,
            (),
            ("key",),
            "rows",
            {"key": pl.Null(), "rows": pl.UInt64()},
            sql="SELECT NULL AS key, 1::UBIGINT AS rows WHERE false"
            if backend == "native"
            else None,
            lazy=pl.DataFrame(schema={"key": pl.Null, "rows": pl.UInt64}).lazy()
            if backend == "lazy"
            else None,
        )
        right = KeyRelation(
            ctx,
            (),
            ("key",),
            "rows",
            {"key": dtype, "rows": pl.UInt64()},
            sql="SELECT NULL AS key, 1::UBIGINT AS rows WHERE false"
            if backend == "native"
            else None,
            lazy=pl.DataFrame(schema={"key": dtype, "rows": pl.UInt64}).lazy()
            if backend == "lazy"
            else None,
        )
        aligned, _ = align_key_relations(left, right)
        summary = aligned.summarize(1)
        assert summary.keys == summary.rows == 0
        assert summary.sample.schema["key"] == dtype


@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("limit", [0, 2])
def test_native_key_summary_excludes_padding_from_all_outputs(rows, limit):
    with DataContext() as ctx:
        relation = KeyRelation(
            ctx,
            (),
            ("__qavf_summary_padding",),
            "rows",
            {"__qavf_summary_padding": pl.Int64(), "rows": pl.UInt64()},
            sql=f"SELECT range AS __qavf_summary_padding, 2::UBIGINT AS rows FROM range({rows})",
        )
        summary = relation.summarize(limit)
    assert summary.keys == rows
    assert summary.rows == 2 * rows
    assert summary.sample.height == min(rows, limit)
    assert summary.sample.columns == ["__qavf_summary_padding", "rows"]
    assert summary.sample["rows"].to_list() == [2] * min(rows, limit)


def test_native_key_summary_evaluates_real_relation_once():
    calls = []

    def observed(value):
        calls.append(value)
        return value

    with DataContext() as ctx, duckdb.connect() as native:
        native.create_function(
            "observed",
            observed,
            parameters=["BIGINT"],
            return_type="BIGINT",
            side_effects=True,
        )
        relation = KeyRelation(
            ctx,
            (),
            ("key",),
            "rows",
            {"key": pl.Int64(), "rows": pl.UInt64()},
            sql="SELECT observed(range) AS key, 2::UBIGINT AS rows FROM range(10)",
        )
        with patch.object(ctx, "sql", side_effect=lambda sql, **_: native.execute(sql).pl()):
            summary = relation.summarize(3)
    assert sorted(calls) == list(range(10))
    assert summary.keys == 10 and summary.rows == 20
    assert summary.sample.height == 3
