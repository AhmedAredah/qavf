"""Independent boundary fixtures and backend invariants for input checks."""

from datetime import UTC, datetime
from decimal import Decimal, localcontext

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.rules.freshness_checks import FreshnessRule
from qavf.rules.helpers import aggregate, align_keys, exact_sum_expr
from qavf.rules.input_checks import ColumnCheckRule
from qavf.rules.reconciliation_checks import _missing_grid_sample
from qavf.settings import QAVFSettings
from qavf.testing import rule_results


def results(rule, frames, params=None, datasets=None):
    return rule_results(rule, frames, params, datasets=datasets or {})


def input_result(frame, limits, kind="eager", tmp_path=None):
    spec = {"schema": {"columns": {"x": limits}}}
    frames = {"d": frame.lazy() if kind == "lazy" else frame}
    if kind == "parquet":
        path = tmp_path / "data.parquet"
        frame.write_parquet(path)
        spec["path"] = str(path)
        frames = {}
    return results("INP-003", frames, datasets={"d": spec})


@pytest.mark.parametrize("precision", [6, 28, 100])
def test_int128_overflow_guard_does_not_depend_on_decimal_context(precision):
    value = 34028236692093846346337460743176821146
    series = pl.Series("x", [value] * 5, dtype=pl.Int128)
    with localcontext() as context:
        context.prec = precision
        with pytest.raises(ValueError, match="accumulator"):
            exact_sum_expr(series)
    found = input_result(series.to_frame(), {"sum": {"le": 0}})
    assert any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize("kind", ["eager", "lazy", "parquet"])
@pytest.mark.parametrize(
    "operator,value,bound",
    [
        ("ge", 2**53, 2**53 + 1),
        ("le", 2**53 + 4, 2**53 + 3),
        ("gt", 2**53 + 4, 2**53 + 4),
        ("lt", 2**53, 2**53),
        ("ge", -(2**53 + 4), -(2**53 + 3)),
    ],
)
def test_exact_float_bound_backend_parity(kind, operator, value, bound, tmp_path):
    found = input_result(pl.DataFrame({"x": [float(value)]}), {operator: bound}, kind, tmp_path)
    assert any(r.status is Status.FAIL for r in found)
    assert not any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize("kind", ["eager", "lazy", "parquet"])
def test_decimal_bound_keeps_fractional_identity(kind, tmp_path):
    frame = pl.DataFrame(
        {"x": pl.Series([Decimal("90.000000000000000001")], dtype=pl.Decimal(38, 18))}
    )
    assert any(r.status is Status.FAIL for r in input_result(frame, {"le": 90.0}, kind, tmp_path))


@pytest.mark.parametrize("limits", [{"mean": {"le": 2**53}}, {"median": {"le": 2**53}}])
def test_exact_large_integer_location_statistics(limits):
    assert any(
        r.status is Status.FAIL for r in input_result(pl.DataFrame({"x": [2**53 + 1]}), limits)
    )


@pytest.mark.parametrize("offset", [0, 2**53, 2**100])
def test_integer_standard_deviation_is_translation_invariant(offset):
    frame = pl.DataFrame({"x": pl.Series([offset, offset + 1], dtype=pl.Int128)})
    assert any(r.status is Status.FAIL for r in input_result(frame, {"std": {"le": 0}}))


def test_finite_extreme_std_is_scaled_before_variance():
    frame = pl.DataFrame({"x": [0.0, 1e308]})
    assert any(r.status is Status.FAIL for r in input_result(frame, {"std": {"ge": 1.5e308}}))
    assert all(
        r.status is Status.PASS for r in input_result(frame, {"std": {"ge": 7e307, "le": 8e307}})
    )


@pytest.mark.parametrize("how", ["mean", "weighted_mean"])
def test_large_aggregate_precision_is_checked_before_and_after_reduction(how):
    for values in [[2**53, 2**53 + 1], [2**53, 2**53 + 2]]:
        frame = pl.DataFrame({"x": values, "w": [1, 1]})
        with pytest.raises(ValueError, match=r"(precision|represent)"):
            aggregate(frame, [], "x", how=how, weight="w" if how == "weighted_mean" else None)


def test_relative_tolerance_cannot_pass_through_infinity():
    found = results(
        "REC-001",
        {"a": pl.DataFrame({"x": [1e308]}), "b": pl.DataFrame({"x": [-1e308]})},
        {
            "reconciliations": [
                {
                    "left": {"dataset": "a", "value": "x"},
                    "right": {"dataset": "b", "value": "x"},
                    "rtol": 1.9,
                }
            ]
        },
    )
    assert found[0].status is Status.FAIL


def test_missing_markers_apply_to_shares_and_batches():
    specs = {
        name: {"schema": {"columns": {"x": {"missing_values": [-999]}}}} for name in ["a", "b"]
    }
    found = results(
        "REC-005",
        {name: pl.DataFrame({"x": [-999]}) for name in specs},
        {"batches": [{"datasets": ["a", "b"], "column": "x"}]},
        specs,
    )
    assert found[0].status is Status.FAIL
    found = results(
        "REC-002",
        {"a": pl.DataFrame({"x": [1.0, -999.0, 999.0]})},
        {"shares": [{"dataset": "a", "value": "x"}]},
        specs,
    )
    assert found[0].status is Status.FAIL


@pytest.mark.parametrize("value,places", [(1e308, 2), (1.25, 400), (0.1, 309)])
def test_decimal_places_do_not_overflow(value, places):
    assert all(
        r.status is Status.PASS
        for r in input_result(pl.DataFrame({"x": [value]}), {"max_decimals": places})
    )


@pytest.mark.parametrize(
    "name", ["__qavf_value", "__qavf_metric", "__qavf_group", "__qavf_group_median"]
)
def test_spatial_internal_names_preserve_entity_identity(name):
    found = results(
        "SPA-002",
        {"s": pl.DataFrame({name: [1, 2, 3], "v": [1.0, 1.0, 100.0]})},
        {"metrics": [{"dataset": "s", "entity_column": name, "value": "v"}]},
    )
    failure = next(r for r in found if r.status is Status.FAIL)
    assert failure.affected_records[0][name] == 3


def test_mixed_int128_keys_do_not_narrow():
    a, b = align_keys(
        pl.DataFrame({"k": pl.Series([2**100], dtype=pl.Int128)}), pl.DataFrame({"k": [1]}), ["k"]
    )
    assert a["k"][0] == 2**100 and b["k"][0] == 1


@pytest.mark.parametrize("value", [float("nan"), -999.0])
def test_missing_spatial_domains_are_unassessed(value):
    found = results(
        "SPA-003",
        {"s": pl.DataFrame({"k": [value]})},
        {"entity_column": "k"},
        {"s": {"schema": {"columns": {"k": {"missing_values": [-999]}}}}},
    )
    assert found[0].status is Status.ERROR


def test_spatial_partition_scope_is_explicit():
    frame = pl.DataFrame({"e": [1, 2, 3, 4, 5, 6], "y": [2020] * 3 + [2021] * 3, "v": [1.0] * 6})
    params = {
        "metrics": [{"dataset": "s", "entity_column": "e", "value": "v", "group_by": ["y"]}],
        "groups": {"A": [1, 2, 3], "B": [4, 5, 6]},
    }
    found = results("SPA-002", {"s": frame}, {**params, "require_all_groups_per_partition": True})
    assert any(r.status is Status.ERROR and r.affected_count == 2 for r in found)
    observed = results("SPA-002", {"s": frame}, params)
    assert all(r.status is Status.PASS for r in observed)


def test_freshness_preserves_lazy_projection(monkeypatch):
    with DataContext(
        frames={
            "d": pl.DataFrame({"at": [datetime(2026, 1, 1)], "irrelevant": ["x" * 10000]}).lazy()
        },
        reference_time=datetime(2026, 1, 1, tzinfo=UTC),
    ) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *args: pytest.fail("freshness materialized input"))
        found = FreshnessRule(
            params={"checks": [{"dataset": "d", "column": "at", "max_age": "P1D"}]}
        ).evaluate(ctx)
        assert all(f.passed for f in found)
        assert not ctx.is_materialized("d")


def test_simple_lazy_checks_share_one_reduction_scan(monkeypatch):
    settings = QAVFSettings.model_validate(
        {"datasets": {"d": {"schema": {"columns": {"x": {"ge": 0, "le": 10}, "y": {"ge": 0}}}}}}
    )
    calls = []
    collect = pl.LazyFrame.collect

    def counted(self, *args, **kwargs):
        calls.append(1)
        return collect(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", counted)
    with DataContext.from_settings(
        settings, frames={"d": pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).lazy()}
    ) as ctx:
        assert all(f.passed for f in ColumnCheckRule().evaluate(ctx))
    assert len(calls) == 1


def test_zero_grid_evidence_does_not_sort_or_enumerate(monkeypatch):
    monkeypatch.setattr(pl.DataFrame, "sort", lambda *args, **kwargs: pytest.fail("unneeded sort"))
    empty = _missing_grid_sample(pl.DataFrame({"k": [1]}), {"k": pl.Series([1, 2])}, 0)
    assert empty.height == 0 and empty.columns == ["k"]


def test_spatial_membership_does_not_truncate_configured_ids():
    found = results(
        "SPA-002",
        {"s": pl.DataFrame({"e": [1, 2, 3], "v": [1.0, 1.0, 1.0]})},
        {
            "metrics": [{"dataset": "s", "entity_column": "e", "value": "v"}],
            "groups": {"A": [1.9, 2.9, 3.9]},
        },
    )
    assert any(r.status is Status.ERROR for r in found)
    assert not any(r.status is Status.PASS for r in found)


def test_spatial_bad_target_preserves_previous_failure():
    found = results(
        "SPA-002",
        {
            "good": pl.DataFrame({"e": [1, 2, 3], "v": [1.0, 1.0, 100.0]}),
            "bad": pl.DataFrame({"e": [1, 2, 3], "v": ["x", "y", "z"]}),
        },
        {
            "metrics": [
                {"dataset": name, "entity_column": "e", "value": "v"} for name in ["good", "bad"]
            ]
        },
    )
    assert any(r.status is Status.FAIL for r in found)
    assert any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize("kind", ["eager", "lazy", "parquet"])
def test_float32_bound_is_compared_without_rounding(kind, tmp_path):
    frame = pl.DataFrame({"x": pl.Series([1.0], dtype=pl.Float32)})
    assert any(
        r.status is Status.FAIL for r in input_result(frame, {"ge": 1.0000000001}, kind, tmp_path)
    )


def test_freshness_filters_declared_missing_timestamp_without_materializing(monkeypatch):
    marker = datetime(2099, 1, 1)
    settings = QAVFSettings.model_validate(
        {"datasets": {"d": {"schema": {"columns": {"at": {"missing_values": [marker]}}}}}}
    )
    with DataContext.from_settings(
        settings, frames={"d": pl.DataFrame({"at": [datetime(2026, 1, 1), marker]}).lazy()}
    ) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *args: pytest.fail("unexpected eager read"))
        found = FreshnessRule(
            params={
                "now": datetime(2026, 1, 1, tzinfo=UTC),
                "checks": [{"dataset": "d", "column": "at", "max_age": "P1D"}],
            }
        ).evaluate(ctx)
        assert all(f.passed for f in found)


def test_coverage_zero_evidence_still_counts_missing_slices():
    from qavf.rules.spatial_checks import CoverageRule

    with DataContext(frames={"s": pl.DataFrame({"e": [1]})}, evidence_limit=0) as ctx:
        [found] = CoverageRule(params={"entity_column": "e", "expected_entities": [1, 2]}).evaluate(
            ctx
        )
        assert not found.passed and found.affected_count == 1 and found.affected == []


def test_decimal_places_do_not_lose_large_value_fraction():
    frame = pl.DataFrame({"x": [45035996273705.125]})
    assert any(r.status is Status.FAIL for r in input_result(frame, {"max_decimals": 2}))


def test_native_freshness_does_not_load_wide_frame(tmp_path, monkeypatch):
    path = tmp_path / "fresh.parquet"
    pl.DataFrame({"at": [datetime(2026, 1, 1)], "unused": ["x" * 10000]}).write_parquet(path)
    settings = QAVFSettings.model_validate({"datasets": {"d": {"path": str(path)}}})
    with DataContext.from_settings(settings) as ctx:
        monkeypatch.setattr(ctx, "get", lambda *args: pytest.fail("native max materialized"))
        found = FreshnessRule(
            params={
                "now": datetime(2026, 1, 1, tzinfo=UTC),
                "checks": [{"dataset": "d", "column": "at", "max_age": "P1D"}],
            }
        ).evaluate(ctx)
        assert all(f.passed for f in found)
        assert not ctx.is_materialized("d")


def test_fused_native_counts_use_one_query(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).write_parquet(path)
    settings = QAVFSettings.model_validate(
        {
            "datasets": {
                "d": {"path": str(path), "schema": {"columns": {"x": {"ge": 0}, "y": {"le": 10}}}}
            }
        }
    )
    with DataContext.from_settings(settings) as ctx:
        queries = []
        sql = ctx.sql

        def tracked(query, *args, **kwargs):
            queries.append(query)
            return sql(query, *args, **kwargs)

        monkeypatch.setattr(ctx, "sql", tracked)
        monkeypatch.setattr(ctx, "get", lambda *args: pytest.fail("native checks materialized"))
        assert all(f.passed for f in ColumnCheckRule().evaluate(ctx))
        assert sum("count(*) AS rows" in query for query in queries) == 1


@pytest.mark.parametrize(
    "entities,periods", [([True], [1]), ([1], [True]), ([1], [Decimal("NaN")]), ([True, 1], [1])]
)
def test_coverage_rejects_incompatible_or_missing_expected_identities(entities, periods):
    found = results(
        "SPA-003",
        {"s": pl.DataFrame({"e": [1], "t": [1]})},
        {
            "entity_column": "e",
            "time_column": "t",
            "expected_entities": entities,
            "expected_periods": periods,
        },
    )
    assert any(r.status is Status.ERROR for r in found)
    assert not any(r.status is Status.PASS for r in found)


def test_coverage_inferred_domains_do_not_conflate_boolean_and_integer():
    found = results(
        "SPA-003",
        {"a": pl.DataFrame({"e": [True]}), "b": pl.DataFrame({"e": [1]})},
        {"entity_column": "e"},
    )
    assert any(r.status is Status.ERROR for r in found)


@pytest.mark.parametrize(
    "values,divisors",
    [
        ([2**53, 2**53, 2**54 + 1], [1, 1, 1]),
        ([2**54, 2**54, 2**54 + 2], [2, 2, 2]),
    ],
)
def test_spatial_normalization_checks_precision_before_and_after_division(values, divisors):
    found = results(
        "SPA-002",
        {
            "s": pl.DataFrame({"e": [1, 2, 3], "v": values}),
            "d": pl.DataFrame({"e": [1, 2, 3], "n": divisors}),
        },
        {
            "metrics": [
                {
                    "dataset": "s",
                    "entity_column": "e",
                    "value": "v",
                    "normalize_by": {"dataset": "d", "value": "n"},
                }
            ]
        },
    )
    assert any(r.status is Status.ERROR for r in found)
    assert not any(r.status is Status.PASS for r in found)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coverage_never_merges_submicrosecond_period_identities(seed):
    frame = pl.DataFrame({"e": [1, 2, 1], "t": pl.Series([1, 1, 2], dtype=pl.Datetime("ns"))})
    found = results(
        "SPA-003",
        {"s": frame.sample(fraction=1, shuffle=True, seed=seed)},
        {"entity_column": "e", "time_column": "t", "expected_entities": [1, 2]},
    )
    assert any(r.status is Status.ERROR and "precision" in r.message for r in found)
    assert not any(r.status is Status.PASS for r in found)
