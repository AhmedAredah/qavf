"""Public rule contracts across the built-in rule modules."""

from fractions import Fraction

import polars as pl
import pytest

from qavf.compare.results import CheckResult, DatasetComparison, LevelResult
from qavf.compare.spec import ComparisonSpec
from qavf.engine.context import DataContext
from qavf.inference.results import not_assessed
from qavf.rules import temporal_checks
from qavf.rules._outlier_batches import _median
from qavf.rules.comparison_checks import comparison_findings
from qavf.rules.input_checks import ColumnCheckRule, DuplicateKeyRule, ExpressionRule
from qavf.rules.statistical_checks import ColumnOutlierRule
from qavf.rules.temporal_checks import SeasonalAnomalyRule, TrendDirectionRule
from qavf.settings import DatasetSpec
from qavf.sources.base import SourceEnvironment
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource


def context(frame, schema, backend, tmp_path, calls, evidence_limit=2):
    options = {
        "specs": {"d": DatasetSpec.model_validate({"schema": schema})},
        "evidence_limit": evidence_limit,
    }
    if backend == "native":
        path = tmp_path / "input.parquet"
        frame.write_parquet(path)
        options["sources"] = {"d": FileSource(FileParams(path=str(path)), SourceEnvironment())}
    elif backend == "lazy":

        def observed(data):
            calls.append(1)
            return data

        options["frames"] = {"d": frame.lazy().map_batches(observed, schema=frame.schema)}
    else:
        options["frames"] = {"d": frame}
    return DataContext(**options)


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
@pytest.mark.parametrize("unique_rows", [False, True])
def test_unusable_primary_key_preserves_other_key_contracts(backend, unique_rows, tmp_path):
    calls = []
    frame = pl.DataFrame({"id": [1, 1], "alt": [3, 3]})
    schema = {
        "primary_key": ["id"],
        "unique_keys": [["alt"]],
        "unique_rows": unique_rows,
        "columns": {"id": {"missing_values": ["oops"]}},
    }
    with context(frame, schema, backend, tmp_path, calls) as ctx:
        findings = DuplicateKeyRule().evaluate(ctx)
    assert sum(not f.assessed for f in findings) == 1
    failures = {f.check: f for f in findings if f.assessed and not f.passed}
    assert failures["d: unique (alt)"].affected_count == 1
    assert failures["d: unique (alt)"].affected_rows == 2
    if unique_rows:
        assert failures["d: duplicate rows"].affected_count == 2
    if backend == "lazy":
        assert calls == [1]


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
@pytest.mark.parametrize("evidence_limit", [0, 2])
@pytest.mark.parametrize("units", [[["kg"], ["kg"]], [b"\xff", b"\xff"]])
def test_unusable_units_preserve_sibling_failure(backend, evidence_limit, units, tmp_path):
    calls = []
    frame = pl.DataFrame({"x": [1, 2], "u": units, "bad": [-1, -2]})
    schema = {"columns": {"x": {"unit": "kg", "unit_column": "u"}, "bad": {"ge": 0}}}
    with context(frame, schema, backend, tmp_path, calls, evidence_limit) as ctx:
        findings = ColumnCheckRule().evaluate(ctx)
    assert [f.check for f in findings if not f.assessed] == ["d.x: units"]
    [failure] = [f for f in findings if f.assessed and not f.passed]
    assert failure.check == "d.bad: ge 0" and failure.affected_count == 2
    assert failure.total_records == 2
    assert (0 if failure.affected is None else len(failure.affected)) == evidence_limit
    if backend == "lazy":
        assert calls == [1]


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
def test_nontext_units_remain_supported_when_cast_is_valid(backend, tmp_path):
    calls = []
    frame = pl.DataFrame({"x": [1, 2], "u": [1, 2]})
    schema = {"columns": {"x": {"unit_column": "u", "allowed_units": ["1"]}}}
    with context(frame, schema, backend, tmp_path, calls) as ctx:
        findings = ColumnCheckRule().evaluate(ctx)
    assert all(f.assessed for f in findings)
    [failure] = [f for f in findings if not f.passed]
    assert failure.check == "d.x: units" and failure.affected_count == 1
    if backend == "lazy":
        assert calls == [1]


def test_unresolved_trend_is_not_counted_as_success_and_siblings_continue(monkeypatch):
    original = temporal_checks.kendall_tau
    calls = []

    def unresolved(x, y):
        calls.append(tuple(y))
        return (
            not_assessed("kendall_tau", "controlled unresolved inference")
            if y[0] == 1
            else original(x, y)
        )

    monkeypatch.setattr(temporal_checks, "kendall_tau", unresolved)
    frame = pl.DataFrame(
        {
            "g": ["unresolved"] * 3 + ["bad"] * 3 + ["constant"] * 3,
            "t": [1, 2, 3] * 3,
            "v": [1, 2, 3, 3, 2, 1, 5, 5, 5],
        }
    )
    with DataContext(frames={"d": frame}) as ctx:
        findings = TrendDirectionRule(
            {
                "series": [
                    {
                        "dataset": "d",
                        "value": "v",
                        "time_column": "t",
                        "group_by": ["g"],
                        "expected_trend": "increasing",
                    }
                ]
            }
        ).evaluate(ctx)
    [blocked] = [f for f in findings if not f.assessed]
    assert "controlled unresolved inference" in blocked.message
    assert blocked.affected == [{"g": "unresolved"}]
    [failure] = [f for f in findings if f.assessed and not f.passed]
    assert failure.affected_count == 1 and failure.total_records == 2
    assert failure.affected[0]["g"] == "bad"
    assert len(calls) == 2


@pytest.mark.parametrize("sign", [1, -1])
def test_outlier_materiality_median_rounds_only_once(sign):
    s = 5e-324
    values = sorted(sign * s * v for v in [0, 1, 1, 1, 2, 2, 10, 20])
    expected = float((Fraction(values[3]) + Fraction(values[4])) / 2)
    assert _median(iter((v, 1) for v in values), len(values)) == expected
    with DataContext(frames={"d": pl.DataFrame({"x": values})}) as ctx:
        findings = ColumnOutlierRule(
            {"targets": [{"dataset": "d", "value": "x", "min_relative_deviation": 15}]}
        ).evaluate(ctx)
    assert all(f.assessed and f.passed for f in findings)


@pytest.mark.parametrize("floor", [0, 1])
def test_outlier_midpoint_underflow_is_required_only_for_materiality(floor):
    s = 5e-324
    values = [s * v for v in [0, 0, 0, 0, 1, 1, 10, 20]]
    with DataContext(frames={"d": pl.DataFrame({"x": values})}) as ctx:
        findings = ColumnOutlierRule(
            {"targets": [{"dataset": "d", "value": "x", "min_relative_deviation": floor}]}
        ).evaluate(ctx)
    if floor:
        assert not any(f.assessed for f in findings)
        assert any("nonzero finite Float64" in f.message for f in findings)
    else:
        [failure] = findings
        assert failure.assessed and not failure.passed and failure.affected_count == 2


def test_comparison_identity_preserves_labels_and_same_key_plugin_payload():
    marker = "_qavf_comparison_identity"
    check = CheckResult(
        check="check: suffix",
        passed=False,
        message="different",
        metrics={marker: {"plugin": "value"}, "count": 3},
    )
    result = DatasetComparison(
        name="a: b",
        baseline="a",
        candidate="b",
        levels=[LevelResult(level="structure", name="level: suffix", checks=[check])],
    )
    [finding] = comparison_findings(result, ComparisonSpec(baseline="a", candidate="b"))
    assert finding.check == "a: b: level: suffix: check: suffix"
    assert finding.metrics[marker] == {
        "name": "a: b",
        "level": "level: suffix",
        "check": "check: suffix",
        "metric": {"plugin": "value"},
    }
    assert finding.metrics["count"] == 3


def test_unrepresentable_season_median_preserves_other_season_failure():
    smallest = 5e-324
    frame = pl.DataFrame(
        {
            "g": ["unusable"] * 8 + ["good"] * 8,
            "t": list(range(0, 16, 2)) * 2,
            "v": [smallest * n for n in [0, 0, 0, 0, 1, 1, 10, 20]]
            + [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
        }
    )
    with DataContext(frames={"d": frame}) as ctx:
        findings = SeasonalAnomalyRule(
            {
                "season_length": 2,
                "min_relative_deviation": 15,
                "series": [{"dataset": "d", "value": "v", "time_column": "t", "group_by": ["g"]}],
            }
        ).evaluate(ctx)
    [blocked] = [f for f in findings if not f.assessed]
    assert "nonzero finite Float64" in blocked.message
    assert blocked.affected == [{"g": "unusable", "season": 0}]
    [failure] = [f for f in findings if f.assessed and not f.passed]
    assert failure.affected_count == 1 and failure.total_records == 8
    assert failure.affected[0]["g"] == "good" and failure.affected[0]["value"] == 100.0


def test_unresolved_trend_summary_does_not_claim_too_few_periods(monkeypatch):
    monkeypatch.setattr(
        temporal_checks,
        "kendall_tau",
        lambda *_: not_assessed("kendall_tau", "controlled unresolved inference"),
    )
    with DataContext(frames={"d": pl.DataFrame({"t": [1, 2, 3], "v": [1, 2, 3]})}) as ctx:
        findings = TrendDirectionRule(
            {
                "series": [
                    {
                        "dataset": "d",
                        "value": "v",
                        "time_column": "t",
                        "expected_trend": "increasing",
                    }
                ]
            }
        ).evaluate(ctx)
    assert all(not f.assessed for f in findings)
    assert "no group could be assessed" in findings[-1].message
    assert findings[-1].total_records == 0


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
def test_binary_escape_cannot_be_accepted_as_a_text_unit(backend, tmp_path):
    frame = pl.DataFrame({"x": [1], "u": [b"\xff"], "bad": [-1]})
    schema = {"columns": {"x": {"unit_column": "u", "allowed_units": [r"\xFF"]}, "bad": {"ge": 0}}}
    with context(frame, schema, backend, tmp_path, []) as ctx:
        findings = ColumnCheckRule().evaluate(ctx)
    assert [f.check for f in findings if not f.assessed] == ["d.x: units"]
    [failure] = [f for f in findings if f.assessed and not f.passed]
    assert failure.check == "d.bad: ge 0" and failure.affected_count == 1


def test_disabled_seasonal_materiality_keeps_assessment_and_exact_median_evidence():
    smallest = 5e-324
    values = [smallest * n for n in [0, 0, 0, 0, 1, 1, 10, 20]]
    with DataContext(frames={"d": pl.DataFrame({"v": values, "t": list(range(0, 16, 2))})}) as ctx:
        findings = SeasonalAnomalyRule(
            {
                "season_length": 2,
                "min_relative_deviation": 0,
                "series": [{"dataset": "d", "value": "v", "time_column": "t"}],
            }
        ).evaluate(ctx)
    [failure] = findings
    assert failure.assessed and not failure.passed and failure.affected_count == 2
    assert failure.total_records == 8
    assert all(Fraction(row["season_median"]) == Fraction(smallest) / 2 for row in failure.affected)


@pytest.mark.parametrize(
    "required,selected,unassessed",
    [(True, False, True), (False, False, False), (False, True, True)],
)
@pytest.mark.parametrize("kind", ["no_source", "missing_file", "broken_query"])
def test_expression_observation_preserves_missing_input_policy(
    required, selected, unassessed, kind, tmp_path
):
    sources = {}
    if kind == "missing_file":
        sources["d"] = FileSource(
            FileParams(path=str(tmp_path / "missing.parquet")), SourceEnvironment()
        )
    elif kind == "broken_query":
        sources["d"] = QuerySource(
            QueryParams(sql="SELECT * FROM absent_table"), SourceEnvironment()
        )
    spec = DatasetSpec.model_validate({"required": required, "expressions": [{"expr": "x > 0"}]})
    with DataContext(sources=sources, specs={"d": spec}) as ctx:
        findings = ExpressionRule({"datasets": ["d"]} if selected else {}).evaluate(ctx)
    [finding] = findings
    assert finding.assessed is (not unassessed)
    assert finding.skipped is (not unassessed)
    if unassessed:
        assert finding.check == "d: x > 0"


def test_expression_known_source_failure_is_not_retried(monkeypatch):
    from qavf.engine import connections
    from qavf.sources import query

    calls = []

    def broken():
        calls.append(1)
        raise ValueError("controlled source failure")

    connect = connections.connect

    def instrumented(*args, **kwargs):
        con = connect(*args, **kwargs)
        con.create_function("broken_source", broken, [], "BIGINT", side_effects=True)
        return con

    monkeypatch.setattr(connections, "connect", instrumented)
    monkeypatch.setattr(query, "connect", instrumented)
    source = QuerySource(QueryParams(sql="SELECT broken_source() AS x"), SourceEnvironment())
    spec = DatasetSpec.model_validate({"expressions": [{"expr": "x > 0"}]})
    with DataContext(sources={"d": source}, specs={"d": spec}) as ctx:
        # Force a genuine prior load failure on every supported DuckDB version.
        with pytest.raises(Exception, match="controlled source failure"):
            ctx.get("d")
        assert calls == [1]
        [finding] = ExpressionRule().evaluate(ctx)
        assert not finding.assessed and not finding.skipped
        assert "controlled source failure" in finding.message
        assert calls == [1]
