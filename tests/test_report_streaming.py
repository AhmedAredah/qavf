"""Streaming report construction and typed omissions."""

from __future__ import annotations

from typing import Any

import pytest
from markupsafe import Markup

from qavf.compare import results
from qavf.compare.results import CheckResult, DatasetComparison, LevelResult
from qavf.engine.resources import ResourceLimitError
from qavf.inference.results import Significance, TestResult
from qavf.reports.comparison import ChartBudget, comparison_view, iter_comparison
from qavf.reports.html import _environment
from qavf.reports.presentation import HtmlLimits


def _check(**options: Any) -> CheckResult:
    return CheckResult(check="check", passed=True, message="checked", **options)


def _comparison(*levels: LevelResult) -> DatasetComparison:
    return DatasetComparison(
        name="review", baseline="baseline", candidate="candidate", levels=list(levels)
    )


def _significance() -> Significance:
    return Significance(
        method="probe",
        decision="significant",
        alpha=0.05,
        correction="none",
        family_size=7,
        significant=3,
        failed=3,
        not_assessed=2,
        tests=[TestResult(method="probe", p_value=0.01, notes=["long" * 12_500])],
    )


@pytest.mark.parametrize("with_pass", [False, True])
def test_empty_assessment_is_never_rendered_as_passing(with_pass: bool) -> None:
    empty = LevelResult(level="custom", name="empty")
    levels = [empty]
    if with_pass:
        levels.append(LevelResult(level="custom", name="assessed", checks=[_check()]))
    result = _comparison(*levels)
    view = comparison_view(result, ChartBudget(HtmlLimits()))
    assert not result.assessed and not result.passed
    assert view.verdict == view.levels[0].verdict == "not assessed"
    assert "not assessed" in result.summary()
    assert "v-not-assessed" in result.render_html()


def test_no_levels_and_unassessed_checks_have_truthful_summaries() -> None:
    assert "not assessed" in _comparison().summary()
    assert (
        "not assessed"
        in _comparison(
            LevelResult(
                level="custom",
                name="skipped",
                checks=[_check(assessed=False)],
            )
        ).summary()
    )
    failed = CheckResult(check="failure", passed=False, message="failed")
    result = _comparison(
        LevelResult(level="custom", name="level", checks=[failed, _check(assessed=False)])
    )
    assert result.verdict == "failed"
    assert "differs" in result.summary()


def test_chart_builders_are_lazy_cached_and_stop_after_byte_budget() -> None:
    calls: list[str] = []
    budget = ChartBudget(HtmlLimits(max_chart_bytes=20))

    def build(label: str, size: int) -> Markup:
        calls.append(label)
        return Markup("x" * size)

    first = budget.chart("first", 1, lambda: build("first", 10))
    oversized = budget.chart("oversized", 1, lambda: build("oversized", 15))
    later = budget.chart("later", 1, lambda: build("later", 1))
    assert calls == []
    assert first.svg == "x" * 10 and first.note is None
    assert calls == ["first"]
    assert oversized.svg is None and "budget" in (oversized.note or "")
    assert later.svg is None and "budget" in (later.note or "")
    assert calls == ["first", "oversized"]


def test_chart_marks_refuse_without_calling_builder() -> None:
    budget = ChartBudget(HtmlLimits(max_chart_points=1))
    chart = budget.chart("large", 2, lambda: pytest.fail("builder must not run"))
    assert chart.svg is None and "2 marks" in (chart.note or "")


def test_streaming_limit_stops_before_later_chart_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    from qavf.reports import comparison

    calls: list[str] = []

    def charts(level: LevelResult, budget: ChartBudget) -> list[Any]:
        def build() -> Markup:
            calls.append(level.name)
            return Markup("<svg></svg>")

        return [budget.chart(level.name, 1, build)]

    monkeypatch.setitem(comparison._CHARTS, "custom", charts)
    first = LevelResult(
        level="custom",
        name="first",
        checks=[CheckResult(check="large", passed=True, message="x" * 50_000)],
    )
    later = LevelResult(level="custom", name="later", checks=[_check()])
    stream = iter_comparison(_comparison(first, later), limits=HtmlLimits(max_output_bytes=30_000))
    assert calls == []
    with pytest.raises(ResourceLimitError, match="output limit"):
        list(stream)
    assert calls == ["first"]


def test_comparison_template_does_not_buffer_whole_section() -> None:
    result = _comparison(
        LevelResult(
            level="custom",
            name="custom",
            checks=[
                CheckResult(check=str(i), passed=False, message="x" * 10_000) for i in range(200)
            ],
        )
    )
    limits = HtmlLimits(max_output_bytes=16_384)
    context = {
        "title": "review",
        "comparisons": [comparison_view(result, ChartBudget(limits))],
        "qavf_version": "review",
    }
    sizes = [
        len(chunk.encode())
        for chunk in _environment(limits)
        .get_template("comparison_report.html.j2")
        .generate(**context)
    ]
    assert sum(sizes) > 2_000_000
    assert max(sizes) < 16_384


@pytest.mark.parametrize(
    "kind,container,member",
    [
        ("records", "columns", lambda sig: results.ColumnChanges(column="x", significance=sig)),
        (
            "groups",
            "measures",
            lambda sig: results.MeasureComparison(label="count", measure="count", significance=sig),
        ),
        (
            "temporal",
            "measures",
            lambda sig: results.MeasureComparison(label="count", measure="count", significance=sig),
        ),
        (
            "spatial",
            "measures",
            lambda sig: results.MeasureComparison(label="count", measure="count", significance=sig),
        ),
        (
            "shape",
            "assessments",
            lambda sig: results.ShapeAssessment(check="shape", significance=sig),
        ),
        (
            "distribution",
            "columns",
            lambda sig: results.ColumnDistribution(column="x", kind="numeric", significance=sig),
        ),
        (
            "contribution",
            "dimensions",
            lambda sig: results.DimensionContribution(dimension="zone", significance=sig),
        ),
    ],
)
def test_bounding_removes_all_nested_family_tests_without_losing_counts(
    kind, container, member
) -> None:
    significance = _significance()
    cls = getattr(results, f"{kind.title()}Result")
    level = cls(
        level="series_shape" if kind == "shape" else kind,
        name=kind,
        checks=[_check(significance=significance)],
        **{container: [member(significance)]},
    )
    result = _comparison(level)
    bounded = result.bounded(10_000)
    assert bounded.json_size_bytes() < 10_000
    assert bounded.omitted == ["tests listed in each significance summary"]
    nested = getattr(bounded.levels[0], container)[0].significance
    assert nested.tests == [] and bounded.levels[0].checks[0].significance.tests == []
    assert (nested.family_size, nested.significant, nested.failed, nested.not_assessed) == (
        7,
        3,
        3,
        2,
    )
    assert bounded.verdict == result.verdict
    restored = DatasetComparison.model_validate_json(b"".join(bounded.iter_json_chunks()))
    assert restored == bounded
    assert type(restored.levels[0]) is cls


def test_bounding_preserves_required_plugin_test_fields() -> None:
    class PluginLevel(LevelResult):
        required_test: TestResult
        tests: list[TestResult]
        nested: dict[str, tuple[Significance, ...]]

    test = TestResult(method="probe", p_value=0.2)
    level = PluginLevel(
        level="custom",
        name="plugin",
        checks=[_check()],
        required_test=test,
        tests=[test],
        nested={"family": (_significance(),)},
    )
    bounded = _comparison(level).bounded(1)
    kept = bounded.levels[0]
    assert isinstance(kept, PluginLevel)
    assert kept.required_test == test and kept.tests == [test]
    assert kept.nested["family"][0].tests == []
    assert PluginLevel.model_validate_json(kept.model_dump_json()) == kept


def test_disclosures_only_name_actual_removals_and_keep_prior_truncation() -> None:
    result = _comparison(
        LevelResult(level="custom", name="custom", checks=[_check(truncated=True)])
    )
    bounded = result.bounded(0)
    assert bounded.omitted == []
    assert bounded.levels[0].checks[0].truncated
    with_records = _comparison(
        LevelResult(
            level="custom",
            name="custom",
            checks=[_check(affected_records=[{"id": 1}])],
        )
    ).bounded(0)
    assert with_records.omitted == ["check evidence records"]
    assert with_records.levels[0].checks[0].truncated
    assert with_records.bounded(0) == with_records
