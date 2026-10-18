"""Temporal identity, exact configuration and projected QA execution contracts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.rules.freshness_checks import FreshnessRule
from qavf.rules.input_checks import DuplicateKeyRule, ReferentialIntegrityRule
from qavf.rules.reconciliation_checks import ShareSpec, TotalsReconcileRule
from qavf.settings import QAVFSettings
from qavf.testing import rule_results


def inputs(frame, kind, tmp_path):
    if kind == "file":
        path = tmp_path / "data.parquet"
        frame.write_parquet(path)
        return {}, {"d": {"path": str(path)}}
    if kind == "query":
        path = tmp_path / "query.parquet"
        frame.write_parquet(path)
        return {}, {
            "d": {
                "source": {
                    "type": "query",
                    "sql": f"SELECT * FROM read_parquet('{path}')",
                }
            }
        }
    return {"d": frame.lazy() if kind == "lazy" else frame}, {"d": {}}


@pytest.mark.parametrize("kind", ["eager", "lazy", "file", "query"])
@pytest.mark.parametrize(
    "constraint,failed",
    [
        ({"le": datetime(1970, 1, 1)}, 2),
        ({"gt": datetime(1970, 1, 1)}, 0),
        ({"isin": [datetime(1970, 1, 1)]}, 2),
        ({"notin": [datetime(1970, 1, 1)]}, 0),
    ],
)
def test_nanosecond_bounds_and_membership_preserve_identity(kind, constraint, failed, tmp_path):
    frame = pl.DataFrame({"x": pl.Series([1, 2], dtype=pl.Datetime("ns"))})
    frames, datasets = inputs(frame, kind, tmp_path)
    datasets["d"]["schema"] = {"columns": {"x": constraint}}
    results = rule_results("INP-003", frames, datasets=datasets)
    assert not any(r.status is Status.ERROR for r in results)
    assert sum(r.affected_count for r in results if r.status is Status.FAIL) == failed
    if failed:
        evidence = next(r.affected_records for r in results if r.status is Status.FAIL)
        assert evidence[0]["x"].endswith("000000001")


@pytest.mark.parametrize("dtype", [pl.Datetime("ns"), pl.Duration("ns"), pl.Time])
def test_nanosecond_batch_and_grid_keys_remain_distinct(dtype):
    a = pl.DataFrame({"k": pl.Series([1], dtype=dtype)})
    b = pl.DataFrame({"k": pl.Series([2], dtype=dtype)})
    found = rule_results(
        "REC-005", {"a": a, "b": b}, {"batches": [{"datasets": ["a", "b"], "column": "k"}]}
    )
    assert any(r.status is Status.FAIL for r in found)
    params = {"grids": [{"dataset": "a", "dimensions": {"k": {"from_dataset": "r"}}}]}
    assert all(r.status is Status.PASS for r in rule_results("REC-003", {"a": a, "r": a}, params))
    found = rule_results("REC-003", {"a": b, "r": a}, params)
    assert any(r.status is Status.FAIL and r.affected_count == 1 for r in found)
    assert not any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize("kind", ["eager", "lazy", "file", "query"])
def test_future_nanosecond_is_not_fresh(kind, tmp_path):
    frame = pl.DataFrame({"x": pl.Series([1], dtype=pl.Datetime("ns"))})
    frames, datasets = inputs(frame, kind, tmp_path)
    found = rule_results(
        "INP-008",
        frames,
        {
            "now": datetime(1970, 1, 1, tzinfo=UTC),
            "checks": [{"dataset": "d", "column": "x", "max_age": "P1D"}],
        },
        datasets=datasets,
    )
    assert any(r.status is Status.FAIL and "ahead" in r.message for r in found)
    assert not any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize("dtype", [pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")])
def test_temporal_bounds_outside_column_range_and_between_ticks(dtype):
    frame = pl.DataFrame({"x": pl.Series([0, 1], dtype=dtype)})
    bound = datetime(1970, 1, 1) + timedelta(microseconds=1)
    for operator, expected in [
        ("lt", 1 if dtype.time_unit != "ns" else 0),
        ("ge", 1 if dtype.time_unit != "ns" else 2),
    ]:
        found = rule_results(
            "INP-003",
            {"d": frame},
            datasets={
                "d": {"schema": {"columns": {"x": {operator: bound}}}},
            },
        )
        assert sum(r.affected_count for r in found if r.status is Status.FAIL) == expected
    found = rule_results(
        "INP-003",
        {"d": frame},
        datasets={"d": {"schema": {"columns": {"x": {"le": datetime(9999, 1, 1)}}}}},
    )
    assert all(r.status is Status.PASS for r in found)


@pytest.mark.parametrize("target", [2**53 + 1, Decimal("1.00000000000000000001")])
def test_share_target_preserves_exact_numeric_type(target):
    assert type(ShareSpec(dataset="d", value="x", total=target).total) is type(target)
    params = {"shares": [{"dataset": "d", "value": "x", "total": target, "atol": 0}]}
    equal = rule_results("REC-002", {"d": pl.DataFrame({"x": [target]})}, params)
    unequal = rule_results(
        "REC-002", {"d": pl.DataFrame({"x": [type(target)(float(target))]})}, params
    )
    assert all(r.status is Status.PASS for r in equal)
    assert any(r.status is Status.FAIL for r in unequal)


@pytest.mark.parametrize("kind", ["lazy", "file"])
def test_key_checks_project_and_reduce_without_get(kind, tmp_path, monkeypatch):
    frame = pl.DataFrame({"id": [1, 1, 9], "unused": ["x" * 10000] * 3})
    frames, datasets = inputs(frame, kind, tmp_path)
    datasets["d"]["schema"] = {"primary_key": ["id"]}
    datasets["parent"] = {}
    datasets["d"]["foreign_keys"] = [{"columns": ["id"], "ref_dataset": "parent"}]
    frames["parent"] = pl.DataFrame({"id": [1, 2]}).lazy()
    settings = QAVFSettings.model_validate({"datasets": datasets})
    with DataContext.from_settings(settings, frames=frames) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *a: pytest.fail("full input materialized"))
        dupes = DuplicateKeyRule().evaluate(ctx)
        assert any(not f.passed and f.affected_count == 1 for f in dupes)
        foreign = ReferentialIntegrityRule().evaluate(ctx)
        assert any(not f.passed and f.affected_count == 1 for f in foreign)


def test_freshness_lazy_uses_context_collection(monkeypatch):
    with DataContext(frames={"d": pl.DataFrame({"x": [datetime(2020, 1, 1)]}).lazy()}) as ctx:
        calls = []
        collect = ctx.collect

        def tracked(name, plan):
            calls.append(name)
            return collect(name, plan)

        monkeypatch.setattr(ctx, "collect", tracked)
        found = FreshnessRule(
            params={
                "now": datetime(2020, 1, 1),
                "checks": [{"dataset": "d", "column": "x", "max_age": "P1D"}],
            }
        ).evaluate(ctx)
        assert all(f.passed for f in found) and calls == ["d"]


@pytest.mark.parametrize("kind", ["lazy", "file", "query"])
@pytest.mark.parametrize("count_only", [False, True])
def test_reconciliation_reduces_at_source_without_full_load(
    kind, count_only, tmp_path, monkeypatch
):
    frames, datasets = {}, {}
    for name, frame in {
        "a": pl.DataFrame(
            {"k": [1, 1, 2], "v": [2**53 + 1, 2**53 + 1, 1], "unused": ["x" * 10000] * 3}
        ),
        "b": pl.DataFrame(
            {"k": [1, 2], "v": [2 if count_only else 2**54 + 2, 1], "unused": ["x" * 10000] * 2}
        ),
    }.items():
        if kind == "lazy":
            frames[name], datasets[name] = frame.lazy(), {}
        else:
            path = tmp_path / f"{name}.parquet"
            frame.write_parquet(path)
            datasets[name] = (
                {"path": str(path)}
                if kind == "file"
                else {"source": {"type": "query", "sql": f"SELECT * FROM read_parquet('{path}')"}}
            )
    settings = QAVFSettings.model_validate({"datasets": datasets})
    with DataContext.from_settings(settings, frames=frames) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *a: pytest.fail("full dataset materialized"))
        found = TotalsReconcileRule(
            params={
                "reconciliations": [
                    {
                        "left": {"dataset": "a", **({} if count_only else {"value": "v"})},
                        "right": {"dataset": "b", "value": "v"},
                        "on": ["k"],
                    }
                ]
            }
        ).evaluate(ctx)
        assert all(f.passed for f in found), [f.message for f in found]


@pytest.mark.parametrize("zone,origin", [("UTC", 0), ("America/New_York", 18_000_000_000_000)])
def test_zoned_nanosecond_bounds_keep_wall_clock_policy(zone, origin):
    frame = pl.DataFrame({"x": pl.Series([origin + 1], dtype=pl.Datetime("ns", zone))})
    found = rule_results(
        "INP-003",
        {"d": frame},
        datasets={"d": {"schema": {"columns": {"x": {"le": datetime(1970, 1, 1)}}}}},
    )
    assert any(r.status is Status.FAIL and r.affected_count == 1 for r in found)
    assert not any(r.status is Status.ERROR for r in found)


def test_nanosecond_missing_marker_does_not_remove_nearby_instants():
    frame = pl.DataFrame({"x": pl.Series([0, 1], dtype=pl.Datetime("ns"))})
    found = rule_results(
        "INP-003",
        {"d": frame},
        datasets={
            "d": {
                "schema": {
                    "columns": {
                        "x": {
                            "nullable": True,
                            "missing_values": [datetime(1970, 1, 1)],
                            "le": datetime(1970, 1, 1),
                        }
                    }
                }
            }
        },
    )
    assert any(r.status is Status.FAIL and r.affected_count == 1 for r in found)


def test_aggregate_checks_use_stable_shared_reducers():
    for values, bounds in [
        ([1e16, 1.0, -1e16], {"sum": {"ge": 1, "le": 1}}),
        ([1e308, 1e-300, -1e308], {"mean": {"ge": 1e-301}}),
        ([1e16, 1e16 + 2], {"std": {"ge": 1.3, "le": 1.5}}),
    ]:
        found = rule_results(
            "INP-003",
            {"d": pl.DataFrame({"x": values})},
            datasets={
                "d": {"schema": {"columns": {"x": bounds}}},
            },
        )
        assert all(r.status is Status.PASS for r in found), [r.message for r in found]


@pytest.mark.parametrize("kind", ["eager", "lazy", "file", "query"])
@pytest.mark.parametrize("grouped", [False, True])
def test_float_reconciliation_compares_exact_sum_before_rounding(kind, grouped, tmp_path):
    frames, datasets = {}, {}
    for name, values in (("a", [1e16, 1.0]), ("b", [1e16])):
        frame = pl.DataFrame({"k": [1] * len(values), "x": values})
        if kind in ("eager", "lazy"):
            frames[name] = frame.lazy() if kind == "lazy" else frame
            datasets[name] = {}
        else:
            path = tmp_path / f"{name}.parquet"
            frame.write_parquet(path)
            datasets[name] = (
                {"path": str(path)}
                if kind == "file"
                else {"source": {"type": "query", "sql": f"SELECT * FROM read_parquet('{path}')"}}
            )
    params = {
        "reconciliations": [
            {
                "left": {"dataset": "a", "value": "x"},
                "right": {"dataset": "b", "value": "x"},
                "on": ["k"] if grouped else [],
                "rtol": 0,
                "atol": 0,
            }
        ]
    }
    found = rule_results("REC-001", frames, params, datasets=datasets)
    assert any(r.status is Status.FAIL for r in found), [r.message for r in found]
    assert not any(r.status is Status.ERROR for r in found)
    sample = next(r.affected_records for r in found if r.status is Status.FAIL)
    assert sample[0]["a_total"] == "10000000000000001"
    params["reconciliations"][0].pop("rtol")
    found = rule_results("REC-001", frames, params, datasets=datasets)
    assert all(r.status is Status.PASS for r in found)


def test_exact_float_comparison_preserves_tolerance_and_overflow_semantics():
    for atol, expected in ((0, Status.FAIL), (1e-16, Status.PASS)):
        params = {
            "reconciliations": [
                {
                    "left": {"dataset": "a", "value": "x"},
                    "right": {"dataset": "b", "value": "x"},
                    "rtol": 0,
                    "atol": atol,
                }
            ]
        }
        found = rule_results(
            "REC-001",
            {"a": pl.DataFrame({"x": [0.1, 0.2]}), "b": pl.DataFrame({"x": [0.3]})},
            params,
        )
        assert all(r.status is expected for r in found)
    params["reconciliations"][0]["atol"] = 0
    found = rule_results(
        "REC-001",
        {"a": pl.DataFrame({"x": [1e308, 1e308]}), "b": pl.DataFrame({"x": [1e308, 1e308]})},
        params,
    )
    assert all(r.status is Status.PASS for r in found)


@pytest.mark.parametrize("grouped", [False, True])
def test_float_share_sum_compares_exact_value_to_exact_target(grouped):
    params = {
        "shares": [
            {
                "dataset": "d",
                "value": "x",
                "group_by": ["k"] if grouped else [],
                "total": 10**16,
                "atol": 0,
            }
        ]
    }
    found = rule_results("REC-002", {"d": pl.DataFrame({"k": [1, 1], "x": [1e16, 1.0]})}, params)
    assert any(r.status is Status.FAIL for r in found), [r.message for r in found]
    params["shares"][0]["total"] = 10**16 + 1
    found = rule_results("REC-002", {"d": pl.DataFrame({"k": [1, 1], "x": [1e16, 1.0]})}, params)
    assert all(r.status is Status.PASS for r in found)


@pytest.mark.parametrize("grouped", [False, True])
def test_nonfinite_float_reconciliation_remains_an_assessable_error(grouped):
    found = rule_results(
        "REC-001",
        {
            "a": pl.DataFrame({"k": [1, 1], "x": [1.0, float("inf")]}),
            "b": pl.DataFrame({"k": [1], "x": [1.0]}),
        },
        {
            "reconciliations": [
                {
                    "left": {"dataset": "a", "value": "x"},
                    "right": {"dataset": "b", "value": "x"},
                    "on": ["k"] if grouped else [],
                }
            ]
        },
    )
    assert all(r.status is Status.ERROR for r in found)
    assert "non-finite" in found[0].message
