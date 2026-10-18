"""Execution shape and semantic parity for independent column check plans."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from qavf import ResourceLimits, validate
from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.rules.input_checks import ColumnCheckRule, DuplicateKeyRule
from qavf.settings import QAVFSettings
from qavf.testing import rule_results


def adapt(frame, kind, tmp_path):
    if kind in ("file", "query"):
        path = tmp_path / f"{kind}.parquet"
        frame.write_parquet(path)
        spec = (
            {"path": str(path)}
            if kind == "file"
            else {"source": {"type": "query", "sql": f"SELECT * FROM read_parquet('{path}')"}}
        )
        return {}, {"d": spec}
    return {"d": frame.lazy() if kind == "lazy" else frame}, {"d": {}}


@pytest.mark.parametrize("kind", ["lazy", "file", "query"])
def test_common_checks_return_reductions_under_small_materialization_budget(kind, tmp_path):
    frame = pl.DataFrame({"x": range(10000), "payload": ["z" * 2000] * 10000})
    frames, datasets = adapt(frame, kind, tmp_path)
    datasets["d"]["schema"] = {
        "columns": {"x": {"ge": 0, "unique": True, "missing_values": [-999]}}
    }
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "datasets": datasets,
            "rules": {"INP-003": {}},
        }
    )
    report = validate(
        frames or None,
        config=settings,
        resources=ResourceLimits(target_memory="16MB", max_workers=1, native_threads=1),
    )
    assert all(r.status is Status.PASS for r in report.results_for("INP-003"))


@pytest.mark.parametrize("kind", ["eager", "lazy", "file", "query"])
def test_missing_regex_where_membership_and_aggregate_semantics(kind, tmp_path):
    frame = pl.DataFrame(
        {"x": [1, -999, 3, 3], "text": ["A", " ", "BAD", "B"], "scope": [1, 1, 0, 1]}
    )
    frames, datasets = adapt(frame, kind, tmp_path)
    datasets["d"]["schema"] = {
        "columns": {
            "x": {
                "nullable": True,
                "missing_values": [-999],
                "checks": [
                    {"name": "sum", "sum": {"ge": 7, "le": 7}},
                    {
                        "name": "mean",
                        "mean": {"ge": "2.333333333333333333", "le": "2.333333333333333334"},
                    },
                    {"name": "allowed", "isin": [1, 3]},
                    {"name": "distinct", "distinct_count": {"ge": 2, "le": 2}},
                    {"name": "unique", "unique": True},
                ],
            },
            "text": {
                "nullable": True,
                "blank_as_missing": True,
                "checks": [
                    {
                        "name": "letters",
                        "regex": "[A-Z]",
                        "where": "scope = 1",
                        "min_applicable_rows": 3,
                    },
                    {"name": "missing", "max_null_fraction": 0.25},
                ],
            },
        }
    }
    results = rule_results("INP-003", frames, datasets=datasets)
    assert not any(r.status is Status.ERROR for r in results), [r.message for r in results]
    assert [(r.check, r.affected_count) for r in results if r.status is Status.FAIL] == [
        ("d.x: unique", 1)
    ]


@pytest.mark.parametrize("kind", ["eager", "lazy", "file", "query"])
def test_explicit_nan_forbidden_remains_visible_after_normalizing_missing(kind, tmp_path):
    frames, datasets = adapt(pl.DataFrame({"x": [float("nan"), 1.0]}), kind, tmp_path)
    datasets["d"]["schema"] = {"columns": {"x": {"nullable": True, "notin": [float("nan")]}}}
    results = rule_results("INP-003", frames, datasets=datasets)
    assert not any(r.status is Status.ERROR for r in results), [r.message for r in results]
    assert sum(r.affected_count for r in results if r.status is Status.FAIL) == 1


@pytest.mark.parametrize("kind", ["lazy", "file", "query"])
def test_exact_float_statistics_do_not_change_sibling_projection(kind, tmp_path, monkeypatch):
    frame = pl.DataFrame({"x": [1e16, 1.0, -1e16], "y": [1, 2, 3], "payload": ["q" * 100000] * 3})
    frames, datasets = adapt(frame, kind, tmp_path)
    datasets["d"]["schema"] = {"columns": {"x": {"sum": {"ge": 1, "le": 1}}, "y": {"ge": 0}}}
    settings = QAVFSettings.model_validate({"datasets": datasets})
    with DataContext.from_settings(settings, frames=frames) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *a: pytest.fail("full dataset fallback"))
        found = ColumnCheckRule().evaluate(ctx)
    assert all(f.passed for f in found), [f.message for f in found]


@pytest.mark.parametrize("constraint", ["isin", "notin", "missing_values"])
@pytest.mark.parametrize("dtype", [pl.Date, pl.Datetime("us")])
def test_aware_membership_and_markers_reject_naive_columns(constraint, dtype):
    stamp = datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=2)))
    frame = pl.DataFrame({"x": [datetime(2019, 12, 31, 22)]}).with_columns(pl.col("x").cast(dtype))
    found = rule_results(
        "INP-003",
        {"d": frame},
        datasets={"d": {"schema": {"columns": {"x": {constraint: [stamp]}}}}},
    )
    assert any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize("kind", ["lazy", "file"])
def test_duplicate_key_boundary_is_scalar_or_bounded_sample(kind, tmp_path, monkeypatch):
    frames, datasets = adapt(pl.DataFrame({"id": [*range(10000), 9999]}), kind, tmp_path)
    datasets["d"]["schema"] = {"primary_key": ["id"]}
    settings = QAVFSettings.model_validate({"datasets": datasets})
    with DataContext.from_settings(settings, frames=frames) as ctx:
        ctx.evidence_limit = 1
        collect, sql, collect_many = ctx.collect, ctx.sql, ctx.collect_many
        shapes = []

        def collect_checked(name, plan):
            frame = collect(name, plan)
            shapes.append(frame.height)
            assert frame.height <= 2
            return frame

        def sql_checked(query, *args, **kwargs):
            frame = sql(query, *args, **kwargs)
            shapes.append(frame.height)
            assert frame.height <= 2
            return frame

        def collect_many_checked(names, plan):
            frame = collect_many(names, plan)
            shapes.append(frame.height)
            assert frame.height <= 2
            return frame

        monkeypatch.setattr(ctx, "collect_many", collect_many_checked)
        monkeypatch.setattr(ctx, "collect", collect_checked)
        monkeypatch.setattr(ctx, "sql", sql_checked)
        found = DuplicateKeyRule().evaluate(ctx)
    assert shapes
    assert any(not f.passed and f.assessed and f.affected_count == 1 for f in found), [
        f.message for f in found
    ]
