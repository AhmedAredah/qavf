"""Rule observations, grading and evidence against independent oracles."""

from fractions import Fraction

import polars as pl
import pytest

from qavf import validate
from qavf.engine.context import DataContext
from qavf.rules.input_checks import (
    ColumnCheckRule,
    DuplicateKeyRule,
    ExpressionRule,
    ReferentialIntegrityRule,
)
from qavf.rules.reconciliation_checks import TotalsReconcileRule
from qavf.rules.spatial_checks import GroupComparisonRule
from qavf.rules.temporal_checks import MissingPeriodRule, TrendDirectionRule
from qavf.settings import DatasetSpec
from qavf.sources.base import SourceEnvironment
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams


def spec(value):
    return DatasetSpec.model_validate(value)


def changing(first, second, calls):
    def read(_):
        calls.append(1)
        return first if len(calls) == 1 else second

    return first.lazy().map_batches(read, schema=first.schema)


@pytest.mark.parametrize("cache_bytes", [0, 268435456])
@pytest.mark.parametrize("contract", ["foreign_key", "primary_alternate", "columns", "totals"])
def test_compound_contract_uses_one_owned_observation(contract, cache_bytes, tmp_path):
    calls = []
    specs = {}
    if contract == "foreign_key":
        first, second = pl.DataFrame({"id": [1]}), pl.DataFrame({"id": [2]})
        frames = {"d": changing(first, second, calls), "parent": pl.DataFrame({"id": [2]})}
        specs = {
            "d": spec(
                {
                    "foreign_keys": [
                        {"columns": ["id"], "ref_dataset": "parent", "ref_columns": ["id"]}
                    ]
                }
            )
        }
        rule = ReferentialIntegrityRule()
    elif contract == "primary_alternate":
        first = pl.DataFrame({"id": [1, 2], "alt": [1, 1]})
        second = pl.DataFrame({"id": [1, 1], "alt": [1, 2]})
        frames = {"d": changing(first, second, calls)}
        specs = {"d": spec({"schema": {"primary_key": ["id"], "unique_keys": [["alt"]]}})}
        rule = DuplicateKeyRule()
    elif contract == "columns":
        first, second = pl.DataFrame({"x": [-1]}), pl.DataFrame({"x": [1]})
        frames = {"d": changing(first, second, calls)}
        specs = {
            "d": spec(
                {
                    "schema": {
                        "columns": {
                            "x": {
                                "nullable": True,
                                "checks": [
                                    {"name": "positive", "ge": 0},
                                    {"name": "negative", "where": "abs(x) >= 0", "le": 0},
                                ],
                            }
                        }
                    }
                }
            )
        }
        rule = ColumnCheckRule()
    else:
        first, second = pl.DataFrame({"a": [1], "b": [2]}), pl.DataFrame({"a": [2], "b": [1]})
        frames = {"d": changing(first, second, calls)}
        rule = TotalsReconcileRule(
            {
                "reconciliations": [
                    {
                        "left": {"dataset": "d", "value": "a"},
                        "right": {"dataset": "d", "value": "b"},
                    }
                ]
            }
        )
    owned = tmp_path / "owned"
    with DataContext(
        frames=frames, specs=specs, cache_max_bytes=cache_bytes, temp_directory=owned
    ) as ctx:
        findings = rule.evaluate(ctx)
        assert calls == [1]
        failures = [f for f in findings if f.assessed and not f.passed]
        assert len(failures) == 1 and failures[0].affected_count == 1
        assert all(f.assessed for f in findings), [f.message for f in findings]
    assert not list(owned.iterdir())


def test_count_and_evidence_describe_same_observation():
    calls = []
    first, second = pl.DataFrame({"x": [-1]}), pl.DataFrame({"x": [1]})
    with DataContext(
        frames={"d": changing(first, second, calls)},
        specs={"d": spec({"schema": {"columns": {"x": {"ge": 0}}}})},
    ) as ctx:
        findings = ColumnCheckRule().evaluate(ctx)
    failure = next(f for f in findings if f.assessed and not f.passed)
    assert calls == [1]
    assert failure.affected_count == 1
    assert failure.affected["x"].to_list() == [-1]


def test_coherent_column_evidence_projects_contracts_and_primary_key():
    frame = pl.DataFrame({"id": [7], "x": [-1], "payload": ["irrelevant"]})
    with DataContext(
        frames={"d": frame.lazy()},
        specs={"d": spec({"schema": {"primary_key": ["id"], "columns": {"x": {"ge": 0}}}})},
    ) as ctx:
        findings = ColumnCheckRule().evaluate(ctx)
    failure = next(f for f in findings if f.assessed and not f.passed)
    assert failure.affected["id"].to_list() == [7]
    assert failure.affected["x"].to_list() == [-1]
    assert "payload" not in failure.affected.columns


@pytest.mark.parametrize("alpha,expected", [(None, 100), (0.5, 100), (0.05, 0)])
def test_uncorrected_trend_retains_only_capped_evidence(monkeypatch, alpha, expected):
    from qavf.rules import temporal_checks

    calls = []
    annotate = temporal_checks.annotate

    def record(*args, **kwargs):
        calls.append(1)
        return annotate(*args, **kwargs)

    monkeypatch.setattr(temporal_checks, "annotate", record)
    monkeypatch.setattr(
        temporal_checks, "adjust", lambda *args: pytest.fail("unneeded family retention")
    )
    frame = pl.DataFrame(
        {"g": [g for g in range(100) for _ in range(3)], "t": [1, 2, 3] * 100, "v": [3, 2, 1] * 100}
    )
    with DataContext(frames={"d": frame}, evidence_limit=2) as ctx:
        [finding] = TrendDirectionRule(
            {
                "alpha": alpha,
                "series": [
                    {
                        "dataset": "d",
                        "value": "v",
                        "time_column": "t",
                        "group_by": ["g"],
                        "expected_trend": "increasing",
                    }
                ],
            }
        ).evaluate(ctx)
    assert finding.assessed and (finding.affected_count or 0) == expected
    assert finding.passed == (expected == 0)
    assert len(calls) == min(2, expected)
    if expected:
        assert [row["g"] for row in finding.affected] == [0, 1]
        assert all(row["p_value"] == row["adjusted_p"] == 0.333333 for row in finding.affected)


@pytest.mark.parametrize("lazy", [False, True])
def test_empty_period_expectation_is_unassessed_but_explicit_periods_fail(lazy):
    frame = pl.DataFrame(schema={"t": pl.Int64, "v": pl.Int64})
    with DataContext(frames={"d": frame.lazy() if lazy else frame}) as ctx:
        params = {"series": [{"dataset": "d", "value": "v", "time_column": "t"}]}
        inferred = MissingPeriodRule(params).evaluate(ctx)
        explicit = MissingPeriodRule({**params, "expected_periods": [1, 2]}).evaluate(ctx)
    assert all(not f.assessed for f in inferred)
    assert len(explicit) == 1 and explicit[0].assessed and not explicit[0].passed
    assert explicit[0].affected[0]["missing_periods"] == [1, 2]


@pytest.mark.parametrize(
    "values,limit", [([-1e308, -1e308, 1.7e308, 1.7e308], 6), ([1e308] * 4, 2)]
)
def test_group_median_matches_exact_middle_pair(values, limit):
    ordered = sorted(map(Fraction, values))
    median = (ordered[1] + ordered[2]) / 2
    expected = [
        i
        for i, value in enumerate(values)
        if not Fraction(1, limit) <= Fraction(value) / median <= limit
    ]
    with DataContext(frames={"d": pl.DataFrame({"id": range(4), "v": values})}) as ctx:
        findings = GroupComparisonRule(
            {
                "max_ratio": limit,
                "metrics": [
                    {"dataset": "d", "value": "v", "entity_column": "id", "aggregate": "none"}
                ],
            }
        ).evaluate(ctx)
    assert all(f.assessed for f in findings), [f.message for f in findings]
    failures = [f for f in findings if not f.passed]
    assert bool(failures) == bool(expected)
    if expected:
        assert failures[0].affected_count == len(expected)
        assert [r["id"] for r in failures[0].affected] == expected
        assert all(Fraction(r["group_median"]) == median for r in failures[0].affected)


@pytest.mark.parametrize(
    "unusable,reason",
    [([0.0, 5e-324], "groups too small"), ([0.0, 0.0, 5e-324, 5e-324], "group medians")],
)
def test_unusable_group_median_preserves_sibling_violation(unusable, reason):
    frame = pl.DataFrame(
        {
            "id": range(len(unusable) + 4),
            "g": ["unusable"] * len(unusable) + ["good"] * 4,
            "v": [*unusable, 10.0, 10.0, 10.0, 100.0],
        }
    )
    params = {
        "max_ratio": 3,
        "min_entities": 3,
        "metrics": [
            {
                "dataset": "d",
                "value": "v",
                "entity_column": "id",
                "group_by": ["g"],
                "aggregate": "none",
            }
        ],
    }
    with DataContext(frames={"d": frame}) as ctx:
        findings = GroupComparisonRule(params).evaluate(ctx)
    [blocked] = [f for f in findings if not f.assessed]
    assert blocked.check.endswith(reason) and blocked.affected_count == 1
    [failure] = [f for f in findings if f.assessed and not f.passed]
    assert failure.affected_count == 1 and failure.total_records == 4
    assert failure.affected[0]["id"] == len(unusable) + 3
    assert failure.affected[0]["g"] == "good"


def test_expression_input_failure_preserves_other_dataset_violation():
    def broken(_):
        raise RuntimeError("deterministic decoder failure")

    frames = {
        "bad": pl.DataFrame({"x": [1]}).lazy().map_batches(broken),
        "good": pl.DataFrame({"x": [-1]}),
    }
    config = {
        "plugins": {"entry_points": False},
        "engine": {"parallel": False, "run_unconfigured_rules": False},
        "datasets": {
            name: {"expressions": [{"expr": "x > 0", "name": "positive"}]} for name in frames
        },
        "rules": {"INP-006": {}},
    }
    report = validate(frames, config=config)
    results = {r.check: str(r.status) for r in report.results_for("INP-006")}
    assert results == {"bad: positive": "ERROR", "good: positive": "FAIL"}


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_expression_process_control_is_not_swallowed(exception):
    def stopped(_):
        raise exception("stop")

    with (
        DataContext(
            frames={"d": pl.DataFrame({"x": [1]}).lazy().map_batches(stopped)},
            specs={"d": spec({"expressions": [{"expr": "x > 0"}]})},
        ) as ctx,
        pytest.raises(exception),
    ):
        ExpressionRule().evaluate(ctx)


def test_native_key_evidence_is_stable_across_worker_counts(tmp_path):
    path = tmp_path / "keys.parquet"
    pl.DataFrame({"id": list(range(50000)) * 2}).write_parquet(path)
    samples = []
    for threads in [1, 2, 2]:
        with DataContext(
            sources={"d": FileSource(FileParams(path=str(path)), SourceEnvironment())},
            env=SourceEnvironment(threads=threads),
            specs={"d": spec({"schema": {"primary_key": ["id"]}})},
            evidence_limit=5,
        ) as ctx:
            [failure] = DuplicateKeyRule().evaluate(ctx)
            assert failure.affected_count == 50000
            samples.append(failure.affected["id"].to_list())
    assert samples == [[0, 1, 2, 3, 4]] * 3
