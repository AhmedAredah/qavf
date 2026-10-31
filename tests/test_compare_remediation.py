"""Comparison remediation contracts: exact work, bounded evidence and native reuse."""

from fractions import Fraction

import numpy as np
import polars as pl
import pytest

from qavf import ResourceLimits, compare_datasets
from qavf.audit_codec import evidence_json_size
from qavf.compare.evidence import bound_level
from qavf.compare.results import CheckResult, DistributionMember, DistributionResult
from qavf.engine.sql_read import NativeReadUnavailable
from qavf.rules.helpers import BoundedEvidence, LargestEvidence


def compare(baseline, candidate, **options):
    return compare_datasets(baseline, candidate, entry_points=False, **options)


def outcomes(result):
    return [
        (c.check, c.passed, c.assessed, c.affected_count, c.total)
        for level in result.levels
        for c in level.checks
    ]


@pytest.mark.parametrize("size", [0, 1, 2, 12, 60, 200])
def test_prefix_and_ranked_collectors_include_container_framing(size):
    rows = [{"key": "\N{SNOWMAN}" * n} for n in range(1, 20)]
    prefix = BoundedEvidence(30, byte_limit=size)
    leaders = LargestEvidence(30, byte_limit=size)
    for index, row in enumerate(rows):
        prefix.append(row)
        leaders.add(index, index, row)
    assert prefix.count == len(rows)
    for kept in (prefix.records, leaders.records):
        assert not kept or evidence_json_size(kept) <= size
    assert prefix.records == rows[: len(prefix.records)]


def test_plugin_publication_bounds_nested_details_and_preserves_complete_counts():
    original = DistributionResult(
        level="distribution",
        name="plugin",
        members=[
            DistributionMember(
                column="x",
                kind="numeric",
                group={"g": "huge" * 1000},
                baseline_n=100,
                candidate_n=100,
            )
        ],
        checks=[
            CheckResult(
                check="x",
                passed=False,
                message="changed",
                affected_count=42,
                affected_records=[{"g": "huge" * 1000}],
            )
        ],
    )
    bounded = bound_level(original, 10, 0)
    assert bounded.members == [] and bounded.details_truncated
    assert bounded.checks[0].affected_count == 42 and bounded.checks[0].truncated
    assert original.members and original.checks[0].affected_records


@pytest.mark.parametrize("kind", ["Datetime", "Duration", "Time"])
def test_contribution_and_rate_keep_submicrosecond_keys(kind):
    dtype = {"Datetime": pl.Datetime("ns"), "Duration": pl.Duration("ns"), "Time": pl.Time}[kind]
    key = pl.Series("g", [1, 2], dtype=pl.Int64).cast(dtype)
    baseline = pl.DataFrame({"g": key, "x": [10, 20], "n": [100, 100]})
    candidate = baseline.with_columns(pl.Series("x", [11, 24]))
    result = compare(
        baseline,
        candidate,
        decision="material",
        levels=[
            {
                "level": "contribution",
                "measure": "sum(x)",
                "dimensions": ["g"],
                "rates": ["x / n"],
            }
        ],
    )
    level = result.levels[0]
    assert level.assessed
    assert len({str(c.group["g"]) for c in level.dimensions[0].contributions}) == 2
    assert len({str(c.group["g"]) for c in level.rates[0].strata}) == 2
    assert sum(Fraction(c.exact_change) for c in level.dimensions[0].contributions) == 5


@pytest.mark.parametrize("cache", [0, "8MiB"])
def test_native_records_reuse_aligned_observations_and_match_typed_fallback(monkeypatch, cache):
    import qavf.compare.levels.records as records

    original = records.compare_observations
    calls = []

    def counted(inputs, plan):
        calls.append(True)
        return original(inputs, plan)

    baseline = pl.DataFrame({"k": [1, 2, 3], "x": [1.25, 2.5, 3.75], "tag": ["a", "b", None]})
    candidate = pl.DataFrame({"k": [1, 2, 4], "x": [2.5, 5.5, 8.0], "tag": ["b", "b", None]})
    options = dict(
        keys="k",
        levels=["records"],
        decision="material",
        units={"x": {"factor": 2}},
        resources=ResourceLimits(cache_memory=cache),
    )
    monkeypatch.setattr(records, "compare_observations", counted)
    native = compare(baseline, candidate, **options)
    assert calls and native.levels[0].assessed

    def unavailable(*args):
        raise NativeReadUnavailable("test requests exact fallback")

    monkeypatch.setattr(records, "compare_observations", unavailable)
    fallback = compare(baseline, candidate, **options)
    assert native.levels[0].execution == "native_stream"
    assert fallback.levels[0].execution == "typed_stream"
    assert native.model_dump(exclude={"levels": {"__all__": {"execution"}}}) == fallback.model_dump(
        exclude={"levels": {"__all__": {"execution"}}}
    )


def test_zoned_ns_records_fall_back_before_arrow_conversion(monkeypatch):
    from qavf.engine.observation import Observation

    def forbidden(*args):
        raise AssertionError("ns must not bind through Arrow/DuckDB")

    monkeypatch.setattr(Observation, "native_binding", forbidden)
    key = pl.Series("k", [1, 2], dtype=pl.Int64).cast(pl.Datetime("ns", "UTC"))
    baseline = pl.DataFrame({"k": key, "x": [1, 2]})
    result = compare(
        baseline,
        baseline.with_columns(pl.Series("x", [1, 3])),
        keys="k",
        levels=["records"],
        decision="material",
    )
    assert result.levels[0].assessed and result.levels[0].rows_changed == 1


def test_compatible_exact_measures_share_reduction_and_match_independent_execution(monkeypatch):
    import qavf.compare.measures as measures

    counts = {"sum": 0, "quantiles": 0}
    exact, quantiles = measures.exact_grouped, measures.grouped_quantiles

    def sum_count(*args, **kwargs):
        counts["sum"] += 1
        return exact(*args, **kwargs)

    def quantile_count(*args, **kwargs):
        counts["quantiles"] += 1
        return quantiles(*args, **kwargs)

    monkeypatch.setattr(measures, "exact_grouped", sum_count)
    monkeypatch.setattr(measures, "grouped_quantiles", quantile_count)
    baseline = pl.DataFrame(
        {"g": ["a"] * 4 + ["b"] * 4, "x": [1e16, 1.0, -1e16, 2.0, None, 0.1, 0.2, 0.3]}
    )
    candidate = baseline.with_columns((pl.col("x") + 1).alias("x"))
    options = dict(
        decision="material",
        levels=[
            {
                "level": "groups",
                "group_by": ["g"],
                "measures": [
                    "sum(x)",
                    "mean(x)",
                    "median(x)",
                    {"measure": "quantile", "column": "x", "p": 0.9},
                ],
            }
        ],
    )
    fused = compare(baseline, candidate, **options)
    assert fused.levels[0].assessed and counts == {"sum": 2, "quantiles": 2}

    def independent(requests):
        for measure, data in requests:
            yield measure.compute(data)

    monkeypatch.setattr(measures, "shared_measure_results", independent)
    assert compare(baseline, candidate, **options) == fused


@pytest.mark.parametrize("kind", ["categorical", "numeric"])
def test_zero_evidence_keeps_distribution_decisions_and_family_counts(kind):
    g = [f"{i}-" + "G" * 1000 for i in range(12) for _ in range(8)]
    values = (["a"] * 4 + ["b"] * 4) * 12 if kind == "categorical" else list(range(8)) * 12
    baseline = pl.DataFrame({"g": g, "x": values})
    candidate = baseline.with_columns(pl.lit("c" if kind == "categorical" else 42).alias("x"))
    options = dict(levels=[{"level": "distribution", "columns": ["x"], "group_by": ["g"]}])
    complete = compare(baseline, candidate, **options)
    limited = compare(baseline, candidate, resources=ResourceLimits(evidence_bytes=0), **options)
    assert outcomes(limited) == outcomes(complete)
    assert limited.levels[0].members == []
    for check in limited.levels[0].checks:
        assert check.affected_records == [] and check.metrics == {}
        if check.significance is not None:
            assert check.significance.tests == []
            assert check.significance.family_size == 12


def test_completed_distribution_test_releases_sample_closure():
    import weakref

    from qavf.compare.levels.distribution import _completed_test
    from qavf.compare.significance import Inference
    from qavf.compare.spec import ComparisonSpec, InferenceParams
    from qavf.inference.results import not_assessed

    data = np.arange(10000.0)
    reference = weakref.ref(data)
    inference = Inference.of(
        ComparisonSpec(baseline="b", candidate="c", decision="significant"), InferenceParams()
    )
    completed = _completed_test(
        lambda values=data: not_assessed("example", str(len(values))), inference, True
    )
    del data
    assert reference() is None and completed().method == "example"


def test_nested_contribution_details_share_one_level_allocation():
    from qavf.compare.results import ContributionResult, DimensionContribution, MemberContribution

    member = MemberContribution(
        group={"g": "large" * 50}, baseline=1, candidate=2, change=1, exact_change="1"
    )
    one = DimensionContribution(dimension="g", keys=["g"], members=1, contributions=[member])
    level = ContributionResult(level="contribution", name="test", dimensions=[one, one])
    limit = evidence_json_size({"contributions": [member.model_dump(mode="python")]})
    bounded = bound_level(level, 5, limit)
    assert len(bounded.dimensions[0].contributions) == 1
    assert bounded.dimensions[1].contributions == []
    assert bounded.details_truncated


@pytest.mark.parametrize("limit", [0, 1, 40, 128, 500])
def test_check_optional_object_including_field_framing_obeys_limit(limit):
    from qavf.compare.evidence import bound_check

    check = CheckResult(
        check="test",
        passed=False,
        message="changed",
        affected_count=10,
        affected_records=[{"g": "☃" * 12} for _ in range(10)],
        metrics={"optional": "large" * 20},
    )
    bounded = bound_check(check, 10, limit)
    payload = {
        name: value
        for name, value in {
            "affected_records": bounded.affected_records,
            "metrics": bounded.metrics,
        }.items()
        if value
    }
    assert not payload or evidence_json_size(payload) <= limit
    assert bounded.affected_count == 10 and not bounded.passed


@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_zero_evidence_does_not_construct_distribution_detail(monkeypatch, kind):
    import qavf.compare.levels.distribution as distribution

    def forbidden(**kwargs):
        raise AssertionError("zero evidence constructed a rich detail model")

    monkeypatch.setattr(distribution, "DistributionMember", forbidden)
    values = [1, 2, 3, 4] if kind == "numeric" else ["a", "a", "b", "b"]
    frame = pl.DataFrame({"g": ["one"] * 4, "x": values})
    result = compare(
        frame,
        frame,
        decision="material",
        resources=ResourceLimits(evidence_bytes=0),
        levels=[{"level": "distribution", "columns": ["x"], "group_by": ["g"]}],
    )
    assert result.passed and result.levels[0].members == []
