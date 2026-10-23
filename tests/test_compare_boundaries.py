"""End-to-end comparison boundaries: grouping, matching and statistical admission."""

import polars as pl
import pytest

from qavf import ResourceLimits, compare_datasets
from qavf.compare.levels import LevelInputs
from qavf.compare.results import (
    CheckResult,
    ColumnChanges,
    DatasetComparison,
    RecordsResult,
    Transition,
)
from qavf.engine.observation import Observation


def compare(b, c, **options):
    return compare_datasets(b, c, entry_points=False, **options)


@pytest.mark.parametrize("factor", [1.0, 10.0])
def test_trend_difference_test_never_certifies_equivalence(factor):
    t = list(range(1, 31))
    b = pl.DataFrame({"t": t, "v": [float(x) for x in t]})
    c = b.with_columns((pl.col("v") * factor).alias("v"))
    result = compare(
        b,
        c,
        decision="equivalence",
        levels=[
            {
                "level": "series_shape",
                "time_column": "t",
                "measure": "mean(v)",
                "correlation": None,
                "peak": None,
                "lag": None,
                "level_shift": None,
                "trend": {"dependence": "none", "rtol": 0.01},
            }
        ],
    )
    check = result.levels[0].checks[-1]
    assert not check.assessed and not result.passed
    assert "no equivalence form" in check.message


@pytest.mark.parametrize("shift", [0, 1])
def test_lag_never_certifies_an_unsupported_equivalence(shift):
    t = list(range(30))
    values = [float(i * i % 17) for i in t]
    candidate = values if not shift else [values[0], *values[:-1]]
    b = pl.DataFrame({"t": t, "v": values})
    c = pl.DataFrame({"t": t, "v": candidate})
    result = compare(
        b,
        c,
        decision="equivalence",
        levels=[
            {
                "level": "series_shape",
                "time_column": "t",
                "measure": "mean(v)",
                "correlation": None,
                "peak": None,
                "trend": None,
                "level_shift": None,
            }
        ],
    )
    check = result.levels[0].checks[-1]
    assert not check.assessed and not result.passed
    assert "lag has no equivalence form" in check.message


def test_agreement_preserves_member_columns_that_also_identify_records():
    b = pl.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "id": [1, 2, 3] * 2, "v": [1.0, 2.0, 3.0] * 2})
    c = b.with_columns((pl.col("v") + pl.when(pl.col("g") == "a").then(1).otherwise(-1)).alias("v"))
    result = compare(
        b,
        c,
        keys=["g", "id"],
        decision="material",
        levels=[
            {
                "level": "relationships",
                "group_by": ["g"],
                "agreement": {
                    "columns": ["v"],
                    "min_ccc": None,
                    "max_slope_deviation": None,
                    "max_abs_bias": 0.5,
                },
            }
        ],
    )
    level = result.levels[0]
    assert result.assessed and not result.passed
    assert level.checks[0].affected_count == level.checks[0].total == 2
    assert [(x.group["g"], x.bias) for x in level.agreement] == [("a", 1.0), ("b", -1.0)]


def test_agreement_uses_the_records_null_key_matching_policy():
    b = pl.DataFrame({"id": [None, 1, 2, 3], "v": [1000.0, 1.0, 2.0, 3.0]})
    c = pl.DataFrame({"id": [None, 1, 2, 3], "v": [-1000.0, 1.0, 2.0, 3.0]})
    result = compare(
        b,
        c,
        keys="id",
        decision="material",
        levels=[
            {
                "level": "relationships",
                "agreement": {
                    "columns": ["v"],
                    "min_ccc": None,
                    "max_slope_deviation": None,
                    "max_abs_bias": 1,
                },
            }
        ],
    )
    level = result.levels[0]
    assert result.assessed and not result.passed
    assert level.matched == level.agreement[0].pairs == 4
    assert level.agreement[0].bias == -500.0


@pytest.mark.parametrize("values", [[[1.9, 999]], {"01": 999, "1": 999}])
def test_recode_refuses_lossy_or_colliding_old_categories(values):
    b = pl.DataFrame({"id": [1], "code": [1]})
    c = pl.DataFrame({"id": [1], "code": [999]})
    result = compare(
        b,
        c,
        keys="id",
        levels=["records"],
        decision="material",
        recode={"code": {"values": values, "unmapped": "error"}},
    )
    assert not result.assessed and not result.passed
    assert "cannot all be represented" in result.error


def test_recode_preserves_documented_text_key_parsing():
    b = pl.DataFrame({"id": [1], "code": [1]})
    c = pl.DataFrame({"id": [1], "code": ["mapped"]})
    result = compare(
        b,
        c,
        keys="id",
        levels=["records"],
        decision="material",
        recode={"code": {"values": {"01": "mapped"}, "unmapped": "error"}},
    )
    assert result.passed


def test_contribution_checks_offsetting_one_sided_members():
    b = pl.DataFrame({"g": ["same", "old"], "x": [1, 100]})
    c = pl.DataFrame({"g": ["same", "new"], "x": [1, 100]})
    result = compare(
        b,
        c,
        decision="material",
        levels=[
            {
                "level": "contribution",
                "measure": "sum(x)",
                "dimensions": ["g"],
            }
        ],
    )
    total, members = result.levels[0].checks
    assert total.passed and members.assessed and not members.passed
    assert members.total == 3 and members.affected_count == 2
    assert {r["g"] for r in members.affected_records} == {"old", "new"}


@pytest.mark.parametrize("decision", ["material", "material_and_significant", "equivalence"])
def test_undefined_mean_is_not_an_assessed_comparison(decision):
    b = pl.DataFrame({"g": ["a", "a"], "x": pl.Series([None, None], dtype=pl.Float64)})
    result = compare(
        b,
        b,
        decision=decision,
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": ["mean(x)"],
            }
        ],
    )
    check = result.levels[0].checks[-1]
    assert not check.assessed and not result.passed
    assert check.metrics["not_measured"] == 1


def test_invalid_weighted_means_do_not_certify_unchanged_values():
    b = pl.DataFrame({"g": ["a", "a"], "x": [1.0, 2.0], "w": [-1.0, -1.0]})
    c = b.with_columns((pl.col("x") * 100).alias("x"))
    result = compare(
        b,
        c,
        decision="material",
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": [{"measure": "weighted_mean", "column": "x", "weight": "w"}],
            }
        ],
    )
    assert not result.assessed and not result.passed
    assert "undefined measure" in result.levels[0].checks[-1].message


def test_contribution_does_not_invent_zero_for_an_undefined_rate_numerator():
    b = pl.DataFrame(
        {
            "g": ["a", "a"],
            "n": pl.Series([None, None], dtype=pl.Float64),
            "d": [1, 1],
        }
    )
    result = compare(
        b,
        b,
        decision="material",
        levels=[
            {
                "level": "contribution",
                "dimensions": ["g"],
                "rates": ["n/d"],
            }
        ],
    )
    check = result.levels[0].checks[-1]
    assert not check.assessed and not result.passed
    assert "undefined numerator" in check.message


def test_contribution_preserves_undefined_totals_and_omits_unfounded_decomposition():
    b = pl.DataFrame({"g": ["a"], "x": pl.Series([None], dtype=pl.Float64)})
    c = pl.DataFrame({"g": ["a"], "x": [100.0]})
    result = compare(
        b,
        c,
        decision="material",
        levels=[{"level": "contribution", "measure": "sum(x)", "dimensions": ["g"]}],
    )
    level = result.levels[0]
    assert not result.assessed and not result.passed
    assert level.baseline_total is None and level.candidate_total == 100
    assert level.total_change is None
    assert not level.dimensions


def test_record_values_with_no_comparable_columns_are_not_assessed():
    b = pl.DataFrame({"id": [1, 2, 3], "v": [1, 2, 3]})
    c = pl.DataFrame({"id": [1, 2, 3], "v": ["bad", "worse", "wrong"]})
    result = compare(b, c, keys="id", levels=["records"], decision="material")
    assert not result.assessed and not result.passed
    assert result.levels[0].checks[1].metrics["not_compared"] == ["v"]
    keys_only = b.select("id")
    assert compare(keys_only, keys_only, keys="id", levels=["records"], decision="material").passed


def test_transition_detail_respects_evidence_bytes():
    b = pl.DataFrame({"id": [1, 2, 3], "cat": ["B" * 100_000] * 3})
    c = pl.DataFrame({"id": [1, 2, 3], "cat": ["C" * 100_000] * 3})
    result = compare(
        b,
        c,
        keys="id",
        levels=["records"],
        decision="material",
        resources=ResourceLimits(evidence_bytes=1024),
    )
    column = result.levels[0].columns[0]
    assert column.changed == 3 and column.distinct_transitions == 1
    assert not column.transitions
    assert result.json_size_bytes() < 4096


def test_zero_evidence_rows_leaves_transition_counts_without_examples():
    b = pl.DataFrame({"id": [1], "cat": ["before"]})
    c = pl.DataFrame({"id": [1], "cat": ["after"]})
    result = compare(
        b,
        c,
        keys="id",
        levels=["records"],
        decision="material",
        resources=ResourceLimits(evidence_rows=0),
    )
    column = result.levels[0].columns[0]
    assert column.changed == 1 and column.distinct_transitions == 1
    assert not column.transitions


def test_json_budget_can_remove_transitions_without_losing_counts(tmp_path):
    result = DatasetComparison(
        name="test",
        baseline="b",
        candidate="c",
        levels=[
            RecordsResult(
                level="records",
                name="records",
                values_changed=3,
                columns=[
                    ColumnChanges(
                        column="cat",
                        changed=3,
                        distinct_transitions=1,
                        transitions=[
                            Transition(baseline="B" * 100_000, candidate="C" * 100_000, count=3),
                        ],
                    ),
                ],
                checks=[
                    CheckResult(check="values", passed=False, message="3 changed", affected_count=3)
                ],
            )
        ],
    )
    target = result.write_json(tmp_path / "bounded.json", max_bytes=4096)
    restored = DatasetComparison.model_validate_json(target.read_bytes())
    assert len(target.read_bytes()) <= 4096
    assert restored.levels[0].columns[0].changed == 3
    assert not restored.levels[0].columns[0].transitions
    assert restored.omitted == ["per-member level details"]
    assert not restored.passed and restored.assessed


@pytest.mark.parametrize("width", [1, 4])
def test_record_shift_reuses_join_without_redundant_count(monkeypatch, width):
    collect, derived = LevelInputs.collect, Observation.derived
    joined = {"collected": 0, "derived": 0}

    def trace_collect(self, plan):
        joined["collected"] += "INNER JOIN" in plan.explain(optimized=False)
        return collect(self, plan)

    def trace_derived(self, build):
        def trace(plan):
            query = build(plan)
            joined["derived"] += "INNER JOIN" in query.explain(optimized=False)
            return query

        return derived(self, trace)

    monkeypatch.setattr(LevelInputs, "collect", trace_collect)
    monkeypatch.setattr(Observation, "derived", trace_derived)
    b = pl.DataFrame(
        {"id": list(range(20)), **{f"x{i}": [float(j) for j in range(20)] for i in range(width)}}
    )
    result = compare(
        b,
        b,
        keys="id",
        levels=[
            {
                "level": "records",
                "top_n": 0,
                "max_transitions": 0,
            }
        ],
    )
    assert result.passed
    assert joined["collected"] + joined["derived"] == 2
    assert joined["derived"] == (1 if width > 1 else 0)
