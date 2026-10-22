"""The dataset comparison section of HTML reports, and standalone comparison pages.

A report shows one section per comparison whenever it holds ``CMP-001``
results: a summary card (verdict, a verdict per level with its counts, the
alignment notes), then a drill-down per level with each check's message,
evidence and significance (p, adjusted p, effect and interval), and inline SVG
charts built from the typed :class:`~qavf.compare.results.DatasetComparison`
that ``CMP-001`` keeps with its first result (``metrics["comparison"]``):

* a waterfall of the contributions to the total change (and of a rate's mix,
  rate and interaction effects);
* a heat map of relative change by group (or zone) and bucket;
* paired bars of each side's distribution over shared bins, and quantile deltas;
* baseline and candidate series lines per bucket;
* a choropleth of each zone's change from the simplified zone outlines.

Charts are bounded: one with more marks than ``HtmlLimits.max_chart_points``,
or beyond the document's ``max_chart_bytes``, is replaced by an explicit
"omitted: too large" note, and a map without outlines says why (no spatial
extension, no geometry). When a report kept only the findings (the detail did
not fit the evidence limits), the section is built from them, without charts,
and says so. :func:`render_comparison` renders a ``DatasetComparison`` from
``compare_datasets`` as its own page with the same templates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markupsafe import Markup

from qavf.reports import charts
from qavf.reports.presentation import HtmlLimits

if TYPE_CHECKING:
    from qavf.compare.results import DatasetComparison, LevelResult
    from qavf.inference.results import Significance
    from qavf.reports.models import QAReport, RuleResult

COMPARISON_METRIC = "comparison"
"""Metrics key of the first ``CMP-001`` result of a comparison holding its typed result."""
_WATERFALL_MEMBERS = 12
_SERIES_GROUPS = 3
_DISTRIBUTION_MEMBERS = 4


@dataclass(frozen=True, slots=True)
class TestRow:
    """One test of a check's family, for the drill-down table."""

    label: str
    method: str
    p_value: float | None
    adjusted_p: float | None
    effect: float | None
    effect_name: str | None
    ci_low: float | None
    ci_high: float | None
    significant: bool | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class CheckView:
    label: str
    verdict: str
    """``passed``, ``failed`` or ``not assessed``."""
    message: str
    affected: str
    records: list[dict[str, Any]]
    truncated: bool
    family: str | None = None
    """The family's summary: test, policy, correction and counts."""
    tests: list[TestRow] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChartView:
    title: str
    svg: Markup | None = None
    note: str | None = None
    """Why the chart is not drawn, or what it leaves out."""


@dataclass(frozen=True, slots=True)
class LevelView:
    name: str
    level: str
    verdict: str
    counts: dict[str, int]
    checks: list[CheckView]
    charts: list[ChartView] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ComparisonView:
    name: str
    anchor: str
    baseline: str
    candidate: str
    rows: str
    verdict: str
    counts: dict[str, int]
    alignment: list[str]
    levels: list[LevelView]
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class ChartBudget:
    """Admits charts within the per-chart mark limit and the document's byte budget."""

    def __init__(self, limits: HtmlLimits) -> None:
        self.limits = limits
        self.used = 0

    def chart(
        self, title: str, marks: int, build: Callable[[], Markup], note: str | None = None
    ) -> ChartView:
        if marks > self.limits.max_chart_points:
            return ChartView(
                title,
                note=f"Chart omitted: too large ({marks:,} marks; the limit is "
                f"{self.limits.max_chart_points:,}).",
            )
        svg = build()
        size = len(str(svg).encode("utf-8"))
        if self.used + size > self.limits.max_chart_bytes:
            return ChartView(
                title,
                note=f"Chart omitted: too large (the document's "
                f"{self.limits.max_chart_bytes:,}-byte chart budget is used).",
            )
        self.used += size
        return ChartView(title, svg, note)


def partial_chart(chart: Callable[..., Markup], *args: Any) -> Callable[[], Markup]:
    """A chart drawn only once the budget admits it."""
    return lambda: chart(*args)


def _text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:.6g}"
    return "—" if value is None else str(value)


def _slug(text: str) -> str:
    return "cmp-" + (re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "comparison")


def _relative(baseline: Any, candidate: Any) -> float | None:
    if not isinstance(baseline, int | float) or not isinstance(candidate, int | float):
        return None
    if not baseline or not math.isfinite(baseline) or not math.isfinite(candidate):
        return None
    return (candidate - baseline) / abs(baseline)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


# ------------------------------------------------------------------ checks
def _family(significance: Significance | None) -> tuple[str | None, list[TestRow]]:
    if significance is None:
        return None, []
    how = f"{significance.method}; {significance.decision.replace('_', ' ')}"
    if significance.applicable:
        how += (
            f", {significance.correction}, alpha={significance.alpha:g}: family of "
            f"{significance.family_size:,}, {significance.significant:,} significant, "
            f"{significance.failed:,} failing, {significance.not_assessed:,} not assessed"
        )
    else:
        how += f" ({significance.reason})"
    rows = [
        TestRow(
            test.label or "—", test.method, test.p_value, test.adjusted_p, test.effect,
            test.effect_name, test.ci_low, test.ci_high, test.significant, test.reason,
        )
        for test in significance.tests
    ]  # fmt: skip
    return how, rows


def _verdict(passed: bool, assessed: bool) -> str:
    return "not assessed" if not assessed else ("passed" if passed else "failed")


def _counts(verdicts: Iterable[str]) -> dict[str, int]:
    found = {"passed": 0, "failed": 0, "not assessed": 0}
    for verdict in verdicts:
        found[verdict] += 1
    return found


def _rolled(counts: Mapping[str, int]) -> str:
    if counts["failed"]:
        return "failed"
    return "not assessed" if counts["not assessed"] else "passed"


def _affected(count: int, total: int | None, unit: str | None) -> str:
    text = f"{count:,}" + (f" {unit}" if unit else "")
    return text + (f" of {total:,}" if total else "")


# ------------------------------------------------------------------ charts
def _waterfall(
    budget: ChartBudget, title: str, start: float, end: float, changes: Sequence[tuple[str, float]],
    note: str | None,
) -> ChartView:  # fmt: skip
    steps = [charts.WaterfallStep("baseline", start, "total")]
    steps += [charts.WaterfallStep(label, value) for label, value in changes]
    steps.append(charts.WaterfallStep("candidate", end, "total"))
    return budget.chart(title, len(steps), lambda: charts.waterfall(steps, title), note)


def _member(group: Mapping[str, Any], keys: Sequence[str]) -> str:
    return ", ".join(_text(group.get(key)) for key in keys) or "all"


def _contribution_charts(level: Any, budget: ChartBudget) -> list[ChartView]:
    views = []
    for dimension in level.dimensions:
        shown = dimension.contributions[:_WATERFALL_MEMBERS]
        if not shown or level.baseline_total is None or level.candidate_total is None:
            continue
        changes = [(_member(c.group, dimension.keys), float(c.change)) for c in shown]
        rest = float(dimension.total_change) - sum(value for _, value in changes)
        others = dimension.members - len(shown)
        if others:
            changes.append((f"{others:,} other members", rest))
        title = f"Contributions to the change of {level.measure} by {dimension.dimension}"
        note = None
        if dimension.minimal_set is not None:
            note = (
                f"{dimension.minimal_set:,} of {dimension.members:,} members explain "
                f"{(dimension.explained or 0) * 100:.3g}% of the change."
            )
        start, end = float(level.baseline_total), float(level.candidate_total)
        views.append(_waterfall(budget, title, start, end, changes, note))
        if len(dimension.keys) == 2:
            rows, columns = dimension.keys
            cells = {
                (_text(c.group.get(rows)), _text(c.group.get(columns))): _relative(
                    c.baseline, c.candidate
                )
                for c in dimension.contributions
            }
            partial = (
                f"Shows the {len(dimension.contributions):,} largest of "
                f"{dimension.members:,} members."
                if dimension.truncated
                else None
            )
            views.append(_heat(budget, f"Relative change by {rows} and {columns}", cells, partial))
    for rate in level.rates:
        title = f"{rate.name}: mix and rate effects over {rate.dimension}"
        parts = [("mix", rate.mix), ("rate", rate.rate)]
        if rate.method == "three_term":
            parts.append(("interaction", rate.interaction))
        views.append(
            _waterfall(budget, title, rate.baseline_rate, rate.candidate_rate, parts, None)
        )
    return views


def _heat(
    budget: ChartBudget, title: str, cells: Mapping[tuple[str, str], float | None],
    note: str | None,
) -> ChartView:  # fmt: skip
    rows = list(dict.fromkeys(row for row, _ in cells))
    columns = sorted(dict.fromkeys(column for _, column in cells))
    rows.sort()
    labels = len(rows) * len(columns) <= 300
    return budget.chart(
        title, len(rows) * len(columns),
        lambda: charts.heat_map(rows, columns, cells, title, labels=labels), note,
    )  # fmt: skip


def _series_charts(level: Any, budget: ChartBudget) -> list[ChartView]:
    views = []
    for values in level.values:
        if not values.points:
            continue
        *groups, bucket = values.keys
        partial = (
            f"Shows the first {len(values.points):,} of {values.total:,} values."
            if values.truncated
            else None
        )
        series: dict[str, list[dict[str, Any]]] = {}
        for record in values.points:
            series.setdefault(_member(record, groups), []).append(record)
        if groups:
            cells = {
                (_member(r, groups), _text(r.get(bucket))): _relative(r["baseline"], r["candidate"])
                for r in values.points
            }
            title = f"{values.measure}: relative change by {', '.join(groups)} and {bucket}"
            views.append(_heat(budget, title, cells, partial))
        for name, records in list(series.items())[:_SERIES_GROUPS]:
            labels = [_text(r.get(bucket)) for r in records]
            b = [_number(r["baseline"]) for r in records]
            c = [_number(r["candidate"]) for r in records]
            title = f"{values.measure} per {bucket}" + (f" ({name})" if groups else "")
            views.append(
                budget.chart(
                    title, 2 * len(labels), partial_chart(charts.series_lines, labels, b, c, title),
                    partial,
                )
            )  # fmt: skip
        if groups and len(series) > _SERIES_GROUPS:
            views.append(
                ChartView(
                    f"{values.measure} per {bucket}",
                    note=f"Series lines show the first {_SERIES_GROUPS} of "
                    f"{len(series):,} groups; the heat map shows them all.",
                )
            )
    return views


def _spatial_charts(level: Any, budget: ChartBudget) -> list[ChartView]:
    outlines = level.outlines
    if outlines is None or not level.values:
        return []
    chosen = next((v for v in level.values if v.measure == level.pattern_measure), level.values[0])
    title = f"{chosen.measure}: relative change per zone"
    if outlines.note is not None or not outlines.paths:
        return [ChartView(title, note=f"Map omitted: {outlines.note or 'no zone outlines'}.")]
    column = chosen.keys[0]
    values = {_text(r.get(column)): _relative(r["baseline"], r["candidate"]) for r in chosen.points}
    labels = len(outlines.paths) <= 40
    marks = outlines.vertices + len(outlines.paths)
    note = (
        f"Shows {len(chosen.points):,} of {chosen.total:,} zones' values."
        if chosen.truncated
        else None
    )
    return [
        budget.chart(
            title, marks,
            lambda: charts.choropleth(
                outlines.paths, outlines.centres, values, (outlines.width, outlines.height),
                title, labels=labels,
            ), note,
        )
    ]  # fmt: skip


def _distribution_charts(level: Any, budget: ChartBudget) -> list[ChartView]:
    views = []
    for member in level.members[:_DISTRIBUTION_MEMBERS]:
        where = _member(member.group, list(member.group)) if member.group else "all rows"
        if member.bins:
            title = f"{member.column} ({where}): share per bin"
            labels = [b.label for b in member.bins]
            b = [b.baseline for b in member.bins]
            c = [b.candidate for b in member.bins]
            views.append(
                budget.chart(
                    title, 2 * len(labels),
                    partial_chart(charts.paired_bars, labels, b, c, title),
                )
            )  # fmt: skip
        if member.quantiles:
            title = f"{member.column} ({where}): quantile deltas"
            labels = [f"p{q.p * 100:g}" for q in member.quantiles]
            values = [q.delta for q in member.quantiles]
            views.append(
                budget.chart(
                    title, len(labels), partial_chart(charts.deltas, labels, values, title),
                )
            )  # fmt: skip
    if len(level.members) > _DISTRIBUTION_MEMBERS:
        views.append(
            ChartView(
                "Distributions",
                note=f"Charts show the first {_DISTRIBUTION_MEMBERS} of {len(level.members):,} "
                "measured members; the checks list them all.",
            )
        )
    return views


_CHARTS: dict[str, Callable[[Any, ChartBudget], list[ChartView]]] = {
    "contribution": _contribution_charts,
    "temporal": _series_charts,
    "spatial": _spatial_charts,
    "distribution": _distribution_charts,
}


def _facts(level: LevelResult) -> list[str]:
    """A few typed details of a level for its drill-down header."""
    found = []
    for name, label in (
        ("group_by", "grouped by"), ("bucket", "bucketed by"), ("zones", "zones from"),
        ("measure", "measure"), ("matched", "matched records"), ("total_change", "total change"),
        ("keys", "keys"), ("time_column", "time column"),
    ):  # fmt: skip
        value = getattr(level, name, None)
        if value not in (None, "", []):
            shown = ", ".join(map(str, value)) if isinstance(value, list) else _text(value)
            found.append(f"{label}: {shown}")
    return found


# ------------------------------------------------------------------- views
def comparison_view(
    result: DatasetComparison, budget: ChartBudget, *, charts_note: str | None = None
) -> ComparisonView:
    """The summary card, drill-down and charts of one typed comparison."""
    levels = []
    for level in result.levels:
        checks = []
        for check in level.checks:
            family, tests = _family(check.significance)
            checks.append(
                CheckView(
                    label=check.check,
                    verdict=_verdict(check.passed, check.assessed),
                    message=check.message,
                    affected=_affected(check.affected_count, check.total, check.unit),
                    records=[dict(record) for record in check.affected_records],
                    truncated=check.truncated,
                    family=family,
                    tests=tests,
                )
            )
        counts = _counts(c.verdict for c in checks)
        build = _CHARTS.get(level.level)
        drawn = build(level, budget) if build is not None else []
        levels.append(
            LevelView(
                level.name, level.level, _rolled(counts), counts, checks, drawn, _facts(level)
            )
        )
    counts = _counts(c.verdict for level in levels for c in level.checks)
    rows = (
        f"{result.baseline_rows:,} vs {result.candidate_rows:,} aligned rows"
        if result.baseline_rows is not None and result.candidate_rows is not None
        else "not aligned"
    )
    notes = (
        [f"Omitted to fit the byte budget: {', '.join(result.omitted)}."] if result.omitted else []
    )
    if charts_note:
        notes.append(charts_note)
    verdict = "not assessed" if result.error else _rolled(counts)
    return ComparisonView(
        name=result.name,
        anchor=_slug(result.name),
        baseline=result.baseline,
        candidate=result.candidate,
        rows=rows,
        verdict=verdict,
        counts=counts,
        alignment=list(result.alignment),
        levels=levels,
        error=result.error,
        notes=notes,
    )


def _from_results(name: str, results: list[RuleResult]) -> ComparisonView:
    """A comparison's section from its findings alone (no typed detail, no charts)."""
    from qavf.engine.severity import Status

    by_level: dict[str, list[CheckView]] = {}
    for result in results:
        assert result.check is not None
        _, level, check = ([*result.check.split(": ", 2), "", ""])[:3]
        verdict = {Status.PASS: "passed", Status.FAIL: "failed"}.get(result.status, "not assessed")
        metrics = result.metrics
        family = None
        if "test" in metrics:
            family = (
                f"{metrics['test']}; {str(metrics.get('decision', '')).replace('_', ' ')}, "
                f"{metrics.get('correction')}, alpha={metrics.get('alpha')}: family of "
                f"{metrics.get('family_size')}, {metrics.get('significant')} significant, "
                f"{metrics.get('not_assessed')} not assessed"
            )
        by_level.setdefault(level, []).append(
            CheckView(
                label=check or level,
                verdict=verdict,
                message=result.message,
                affected=_affected(result.affected_count, result.total_records, result.unit),
                records=[dict(record) for record in result.affected_records],
                truncated=result.affected_count > len(result.affected_records),
                family=family,
            )
        )
    levels = [
        LevelView(level, level, _rolled(_counts(c.verdict for c in checks)),
                  _counts(c.verdict for c in checks), checks)
        for level, checks in by_level.items()
    ]  # fmt: skip
    counts = _counts(c.verdict for level in levels for c in level.checks)
    datasets = results[0].datasets
    return ComparisonView(
        name=name,
        anchor=_slug(name),
        baseline=datasets[0] if datasets else "",
        candidate=datasets[-1] if datasets else "",
        rows="see the datasets table",
        verdict=_rolled(counts),
        counts=counts,
        alignment=[],
        levels=levels,
        notes=[
            "Charts and per-member detail omitted: the comparison's typed result did not fit "
            "the evidence limits (engine.max_evidence_bytes or engine.max_report_bytes); the "
            "findings are shown."
        ],
    )


def report_views(report: QAReport, limits: HtmlLimits | None = None) -> list[ComparisonView]:
    """The comparison sections of a report: one per comparison of its ``CMP-001`` results."""
    from qavf.compare.results import DatasetComparison

    budget = ChartBudget(limits or HtmlLimits())
    found = [r for r in report.results if r.rule_id == "CMP-001" and r.check]
    views: list[ComparisonView] = []
    covered: list[str] = []
    for result in found:
        payload = result.metrics.get(COMPARISON_METRIC)
        if not isinstance(payload, Mapping):
            continue
        try:
            typed = DatasetComparison.model_validate(payload)
        except ValueError:
            continue
        views.append(comparison_view(typed, budget))
        covered.append(typed.name)
    remaining: dict[str, list[RuleResult]] = {}
    for result in found:
        assert result.check is not None
        if any(result.check.startswith(f"{name}: ") for name in covered):
            continue
        name = result.check.split(": ", 1)[0]
        remaining.setdefault(name, []).append(result)
    views.extend(_from_results(name, results) for name, results in remaining.items())
    return views


# -------------------------------------------------------------- standalone
def render_comparison(
    result: DatasetComparison, title: str | None = None, *, limits: HtmlLimits | None = None
) -> str:
    """``result`` as a self-contained HTML page (the report's comparison section)."""
    return "".join(chunk.decode("utf-8") for chunk in iter_comparison(result, title, limits=limits))


def iter_comparison(
    result: DatasetComparison, title: str | None = None, *, limits: HtmlLimits | None = None
) -> Iterable[bytes]:
    from qavf._version import __version__
    from qavf.reports.html import iter_template

    limits = limits or HtmlLimits()
    charts.reset_ids()
    view = comparison_view(result, ChartBudget(limits))
    context = {
        "title": title or f"Dataset comparison: {result.name}",
        "comparisons": [view],
        "qavf_version": __version__,
    }
    return iter_template("comparison_report.html.j2", context, limits)


def write_comparison(
    result: DatasetComparison,
    path: str | Path,
    title: str | None = None,
    *,
    limits: HtmlLimits | None = None,
) -> Path:
    """Write :func:`render_comparison` to ``path`` atomically, creating parent directories."""
    from qavf.io import atomic_stream

    target = Path(path)
    with atomic_stream(target) as handle:
        for chunk in iter_comparison(result, title, limits=limits):
            handle.write(chunk)
    return target
