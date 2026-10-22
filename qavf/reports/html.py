"""Render a :class:`~qavf.reports.models.QAReport` as a self-contained HTML page."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup

from qavf.audit_codec import iter_audit_json
from qavf.engine.severity import Severity, Status
from qavf.reports.charts import BarRow, Segment, horizontal_bars, reset_ids
from qavf.reports.models import (
    QAReport,
    Recommendation,
    ReportSummary,
    RuleResult,
    SnapshotModel,
    _iter_snapshot_json,
)
from qavf.reports.presentation import HtmlLimits, bounded_cell, evidence_table

logger = logging.getLogger(__name__)

_STATUS_ORDER = (Status.PASS, Status.FAIL, Status.ERROR, Status.SKIPPED)
_SEVERITY_ORDER = (Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO)
_SLOWEST_RULES_SHOWN = 8


def _format_value(value: Any) -> str:
    """Compact, human-friendly rendering of one cell in an affected-records table."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
        if value.is_integer() and abs(value) < 1e15:
            return f"{value:,.0f}"
        return f"{value:,.6g}"
    if isinstance(value, list | tuple):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    return str(value)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def _anchors(results: list[RuleResult]) -> dict[int, str]:
    """A unique, stable HTML id per result (keyed by ``id(result)``).

    The id is the rule id plus a readable slug of the check. Slugs drop
    punctuation and signs, so checks whose slugs coincide (``ge 0`` and
    ``ge -0``) get a short digest of the exact check text appended; results
    identical in rule and check are numbered in report order.
    """
    base: dict[int, str] = {}
    texts: defaultdict[str, set[str]] = defaultdict(set)
    for result in results:
        rule = re.sub(r"[^A-Za-z0-9_-]+", "-", result.rule_id).strip("-") or "rule"
        slug = _slug(result.check) if result.check else ""
        base[id(result)] = f"{rule}-{slug}" if slug else rule
        texts[base[id(result)]].add(result.check or "")
    anchors: dict[int, str] = {}
    seen: Counter[str] = Counter()
    for result in results:
        anchor = base[id(result)]
        if len(texts[anchor]) > 1:
            digest = hashlib.sha256((result.check or "").encode("utf-8")).hexdigest()[:8]
            anchor = f"{anchor}-{digest}"
        seen[anchor] += 1
        anchors[id(result)] = anchor if seen[anchor] == 1 else f"{anchor}-{seen[anchor]}"
    return anchors


def _affected(result: RuleResult) -> str:
    """``3 keys of 2,000 (5 rows)``: the count, its unit, denominator and row impact."""
    text = f"{result.affected_count:,}"
    if result.unit:
        text += f" {result.unit}"
    if result.total_records:
        text += f" of {result.total_records:,}"
    if result.affected_rows is not None and result.unit != "rows":
        text += f" ({result.affected_rows:,} row{'' if result.affected_rows == 1 else 's'})"
    return text


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{int(seconds // 60)} min {seconds % 60:.0f} s"


def _status_chart(summary: ReportSummary) -> Markup:
    rows = [
        BarRow(
            label=category,
            segments=[
                Segment(status.value, counts.get(status, 0), f"s-{status.value.lower()}")
                for status in _STATUS_ORDER
            ],
        )
        for category, counts in sorted(summary.by_category.items())
    ]
    return horizontal_bars(rows, "Rule results by category and status")


def _severity_chart(summary: ReportSummary) -> Markup:
    counts = summary.failures_by_severity
    rows = [
        BarRow(
            label=severity.value.title(),
            segments=[
                Segment("failed rules", counts.get(severity, 0), f"sev-{severity.value.lower()}")
            ],
        )
        for severity in _SEVERITY_ORDER
        if counts.get(severity, 0)
    ]
    return horizontal_bars(rows, "Failed rules by severity")


def _timing_chart(results: list[RuleResult]) -> Markup:
    slowest = sorted(results, key=lambda r: r.execution_time_s, reverse=True)[:_SLOWEST_RULES_SHOWN]
    rows = [
        BarRow(
            label=f"{result.rule_id} {result.name}",
            segments=[Segment("execution time (s)", result.execution_time_s, "time")],
            value_label=_duration(result.execution_time_s),
        )
        for result in slowest
        if result.execution_time_s > 0
    ]
    return horizontal_bars(rows, "Slowest rules by execution time")


def _environment(limits: HtmlLimits) -> Environment:
    env = Environment(
        loader=PackageLoader("qavf.reports", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["cell"] = lambda value: bounded_cell(value, _format_value)
    env.filters["affected"] = _affected
    env.filters["duration"] = _duration
    env.filters["number"] = _number
    env.globals["evidence_table"] = lambda records: evidence_table(records, limits)
    env.globals["audit_chunks"] = _audit_chunks
    env.globals["shown_metrics"] = _shown_metrics
    return env


def _number(value: Any) -> str:
    """A p-value, effect or interval bound in a significance table."""
    if value is None:
        return "—"
    if isinstance(value, float) and math.isfinite(value) and value and abs(value) < 1e-3:
        return f"{value:.2e}"
    return _format_value(value)


def _shown_metrics(result: RuleResult) -> dict[str, Any]:
    """A result's metrics without the typed comparison (shown in its own section)."""
    from qavf.reports.comparison import COMPARISON_METRIC

    return {k: v for k, v in result.metrics.items() if k != COMPARISON_METRIC}


def iter_template(
    name: str, context: dict[str, Any], limits: HtmlLimits, env: Environment | None = None
) -> Iterator[bytes]:
    """Render template ``name`` in bounded UTF-8 chunks within ``limits.max_output_bytes``.

    Raises:
        ResourceLimitError: once the page exceeds the output limit.
    """
    from qavf.engine.resources import ResourceLimitError

    written = 0
    for text in (env or _environment(limits)).get_template(name).generate(**context):
        for start in range(0, len(text), 16_384):
            chunk = text[start : start + 16_384].encode("utf-8")
            written += len(chunk)
            if written > limits.max_output_bytes:
                raise ResourceLimitError(
                    f"HTML report exceeds its {limits.max_output_bytes:,}-byte output limit"
                )
            yield chunk


def _audit_chunks(value: Any) -> Iterator[str]:
    chunks = (
        _iter_snapshot_json(value) if isinstance(value, SnapshotModel) else iter_audit_json(value)
    )
    for chunk in chunks:
        yield chunk.decode("utf-8")


def _presentation(report: QAReport, title: str, fail_on: Severity | None) -> dict[str, Any]:
    """Compute summary/order/acceptance once for both output APIs."""
    effective = report.metadata.fail_on if fail_on is None else fail_on
    summary = report.summary
    verdict = (
        "failed"
        if report.should_fail(effective)
        else "incomplete"
        if summary.errored or not summary.assessed
        else "passed"
    )
    failures = report.failures()
    advice: dict[str, RuleResult] = {}
    checks: dict[str, list[str]] = {}
    for result in failures:
        if result.recommendation:
            advice.setdefault(result.rule_id, result)
            if result.check:
                checks.setdefault(result.rule_id, []).append(result.check)
    recommendations = [
        Recommendation(
            rule_id=key,
            name=result.name,
            severity=result.severity,
            text=result.recommendation,
            checks=checks.get(key, []),
        )
        for key, result in advice.items()
    ]
    return dict(
        title=title,
        report=report,
        summary=summary,
        metadata=report.metadata,
        verdict=verdict,
        fail_on=effective,
        failures=failures,
        recommendations=recommendations,
        status_chart=_status_chart(summary),
        severity_chart=_severity_chart(summary),
        timing_chart=_timing_chart(report.results),
        statuses=_STATUS_ORDER,
        severities=_SEVERITY_ORDER,
    )


def iter_html(
    report: QAReport,
    title: str = "QA Report",
    fail_on: Severity | None = None,
    *,
    limits: HtmlLimits | None = None,
) -> Iterator[bytes]:
    """Yield bounded HTML fragments; no evidence-table macro buffers the table."""
    from qavf.reports.comparison import report_views

    limits = limits or HtmlLimits()
    env = _environment(limits)
    anchors = _anchors(report.results)
    env.filters["anchor"] = lambda result: anchors[id(result)]
    reset_ids()
    context = _presentation(report, title, fail_on)
    context["comparisons"] = report_views(report, limits)
    yield from iter_template("report.html.j2", context, limits, env)


def render_html(
    report: QAReport,
    title: str = "QA Report",
    fail_on: Severity | None = None,
    *,
    limits: HtmlLimits | None = None,
) -> str:
    """Render ``report`` to an HTML string.

    Args:
        report: The QA run to present.
        title: Page heading.
        fail_on: Severity at which the run counts as failed; drives the verdict banner.
    """
    from io import BytesIO

    result = BytesIO()
    for chunk in iter_html(report, title, fail_on, limits=limits):
        result.write(chunk)
    return result.getvalue().decode("utf-8")


def write_html(
    report: QAReport,
    path: str | Path,
    title: str = "QA Report",
    fail_on: Severity | None = None,
    *,
    limits: HtmlLimits | None = None,
) -> Path:
    """Render ``report`` and write it to ``path``, creating parent directories."""
    from qavf.io import atomic_stream

    target = Path(path)
    with atomic_stream(target) as handle:
        for chunk in iter_html(report, title, fail_on, limits=limits):
            handle.write(chunk)
    logger.info("Wrote HTML report to %s", target)
    return target
