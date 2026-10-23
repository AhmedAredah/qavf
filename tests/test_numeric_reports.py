"""Numeric edge cases in rules and reports."""

from __future__ import annotations

import math
import re
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import polars as pl
import pytest

from qavf.audit_codec import iter_audit_json
from qavf.compare.results import CheckResult, DatasetComparison, MemberValues, TemporalResult
from qavf.engine.severity import Severity, Status
from qavf.key_codec import canonical_key
from qavf.numeric import checked_float, exact_rational, numeric_close
from qavf.profiling import profile_frame
from qavf.reports.charts import WaterfallStep, series_lines, waterfall
from qavf.reports.comparison import _relative, render_comparison, report_views
from qavf.reports.html import render_html
from qavf.reports.models import QAReport, RuleResult, RunMetadata
from qavf.reports.presentation import HtmlLimits
from qavf.stats import check_threshold
from qavf.typed_values import checked_cast


def _result(**updates):
    values = dict(
        rule_id="TEST",
        name="test",
        description="test",
        category="category",
        status=Status.PASS,
        severity=Severity.ERROR,
        message="done",
    )
    values.update(updates)
    return RuleResult(**values)


def _report(results):
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return QAReport(
        metadata=RunMetadata(
            run_id="test", project="test", qavf_version="0.5.0", started_at=now, finished_at=now
        ),
        results=results,
    )


def _comparison(name="comparison"):
    return DatasetComparison(
        name=name,
        baseline="b",
        candidate="c",
        levels=[
            TemporalResult(
                name="series",
                level="temporal",
                checks=[CheckResult(check="values", passed=True, message="equal")],
                values=[
                    MemberValues(
                        measure="x",
                        keys=["day"],
                        total=2,
                        points=[
                            {"day": "first", "baseline": -1e308, "candidate": -1e308},
                            {"day": "last", "baseline": 1e308, "candidate": 1e308},
                        ],
                    )
                ],
            )
        ],
    )


@pytest.mark.parametrize("margin,span", [(0.1, 10), (0.07, 100), (0.14, 50), (0.5, 3)])
def test_profile_margin_matches_decimal_spelling_for_all_numeric_storage(margin, span):
    exact_widen = Fraction(str(margin)) * span
    for values in ([0, span], [0.0, float(span)], [Decimal(0), Decimal(span)]):
        bounds = profile_frame(pl.DataFrame({"x": values}), margin=margin)["columns"]["x"]
        if isinstance(values[0], int):
            assert bounds["ge"] == -math.ceil(exact_widen)
            assert bounds["le"] == span + math.ceil(exact_widen)
        else:
            assert Fraction(str(bounds["ge"])) == -exact_widen
            assert Fraction(str(bounds["le"])) == span + exact_widen


def test_profile_date_and_datetime_margins_round_outward_in_their_units():
    first = date(2024, 1, 1)
    bounds = profile_frame(pl.DataFrame({"x": [first, first + timedelta(days=100)]}), margin=0.07)[
        "columns"
    ]["x"]
    assert bounds["ge"] == first - timedelta(days=7)
    assert bounds["le"] == first + timedelta(days=107)
    first_time = datetime(2024, 1, 1)
    bounds = profile_frame(
        pl.DataFrame({"x": [first_time, first_time + timedelta(microseconds=1)]}), margin=0.1
    )["columns"]["x"]
    assert bounds["ge"] == first_time - timedelta(microseconds=1)
    assert bounds["le"] == first_time + timedelta(microseconds=2)


@pytest.mark.parametrize("margin", [True, 10**400])
def test_profile_rejects_invalid_margin_with_value_error(margin):
    with pytest.raises(ValueError, match="margin"):
        profile_frame(pl.DataFrame({"x": [0, 10]}), margin=margin)


@pytest.mark.parametrize(
    "value,expected,lost",
    [
        ("2024-01-01", "2024-01-01T00:00:00-05:00", False),
        ("2024-01-01T00:00:00", "2024-01-01T00:00:00-05:00", False),
        ("2024-01-01T00:00:00Z", "2023-12-31T19:00:00-05:00", False),
        ("2024-01-01T00:00:00-02:00", "2023-12-31T21:00:00-05:00", False),
        ("2024-01-01T00:00:00+02:00", "2023-12-31T17:00:00-05:00", False),
        ("2024-03-10T02:30:00", None, True),
        ("2024-11-03T01:30:00", None, True),
        ("2024-01-01T00:00:00.1234567", "2024-01-01T00:00:00.123456-05:00", True),
    ],
)
def test_checked_datetime_cast_honors_timezone_and_loss(value, expected, lost):
    target = pl.Datetime("us", "America/New_York")
    converted, flag = checked_cast(pl.col("x"), pl.String, target)
    got = pl.DataFrame({"x": [value]}).select(converted.alias("value"), flag.alias("lost"))
    assert got.schema["value"] == target
    assert got["lost"].to_list() == [lost]
    actual = got["value"][0]
    assert (actual.isoformat() if actual is not None else None) == expected


def test_key_encoder_streams_until_budget_without_copying_all_values():
    class CountingList(list):
        visited = 0

        def __iter__(self):
            for value in super().__iter__():
                self.visited += 1
                yield value

    values = CountingList(range(10_000))
    with pytest.raises(ValueError, match="bounded work"):
        canonical_key(values, max_bytes=8)
    assert values.visited <= 5
    assert canonical_key([{"z": [-0.0], "a": (1, 2)}]) == canonical_key([{"a": [1, 2], "z": [0.0]}])
    assert b"".join(iter_audit_json([-0.0])) == b"[-0.0]"
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycles"):
        canonical_key(cycle)


@pytest.mark.parametrize("maximum", [False, -1, 1.0])
def test_key_encoder_validates_its_byte_limit(maximum):
    with pytest.raises(ValueError, match="max_bytes"):
        canonical_key([], max_bytes=maximum)


def test_finite_extreme_series_have_valid_coordinates_and_relative_changes():
    svg = str(series_lines(["a", "b", "c"], [-1e308, 0.0, 1e308], [None, math.nan, math.inf], "x"))
    assert 'd="M60.0 174.0 L305.0 99.0 L550.0 24.0"' in svg
    assert not re.search(r"\b(?:nan|inf)\b", svg.lower())
    assert _relative(-1e308, 1e308) == 2.0
    assert _relative(1e308, -1e308) == -2.0
    assert _relative(0, 1) is None
    html = render_comparison(_comparison())
    assert not re.search(r'(?:d|x|y|cx|cy|width|height)="[^\"]*(?:nan|inf)', html)
    omitted = waterfall([WaterfallStep("first", 1e308, "total"), WaterfallStep("last", 1e308)], "x")
    assert "Chart omitted" in omitted


def test_summary_charts_obey_mark_and_document_byte_limits():
    report = _report([_result(category=f"category-{i}") for i in range(3)])
    html = render_html(report, limits=HtmlLimits(max_chart_points=1))
    assert "3 marks; the limit is 1" in html
    assert "Rule results by category and status" not in html
    html = render_html(report, limits=HtmlLimits(max_chart_bytes=1))
    assert "<svg" not in html
    assert "1-byte chart budget" in html


def test_summary_and_comparison_share_one_chart_budget():
    comparison = _comparison()
    result = _result(
        rule_id="CMP-001",
        check="comparison: series: values",
        metrics={"comparison": comparison.model_dump()},
    )
    report = _report([result])
    html = render_html(report)
    summary_svg = re.search(r'<svg class="chart".*?</svg>', html, re.S).group()
    # The summary fits exactly; the comparison then has no remaining allowance.
    bounded = render_html(report, limits=HtmlLimits(max_chart_bytes=len(summary_svg.encode())))
    assert bounded.count("<svg") == 1
    assert "chart budget is used" in bounded


def test_structured_comparison_identity_preserves_delimiters_and_prefixes():
    def finding(name, level, check, **extra):
        metrics = {"_qavf_comparison_identity": {"name": name, "level": level, "check": check}}
        metrics.update(extra)
        return _result(rule_id="CMP-001", check=f"{name}: {level}: {check}", metrics=metrics)

    report = _report(
        [
            finding("a", "series", "values", comparison=_comparison("a").model_dump()),
            finding("a: b", "level: with delimiter", "check: with delimiter"),
            finding("a b", "level", "check"),
            finding("a-b-2", "level", "check"),
        ]
    )
    views = report_views(report)
    assert [view.name for view in views] == ["a", "a: b", "a b", "a-b-2"]
    assert views[1].levels[0].name == "level: with delimiter"
    assert views[1].levels[0].checks[0].label == "check: with delimiter"
    assert len({view.anchor for view in views}) == len(views)
    html = render_html(report)
    assert 'id="cmp-a-b"' in html and 'id="cmp-a-b-2"' in html


def test_exact_numeric_helpers_reject_undefined_values_without_internal_exceptions():
    assert exact_rational("1/0") is None
    assert not numeric_close("1/0", 1)
    assert exact_rational("7/2") == Fraction(7, 2)
    assert checked_float(2**100) == float(2**100)
    with pytest.raises(ValueError, match="Float64"):
        checked_float(10**400)
    with pytest.raises(ValueError, match="threshold"):
        check_threshold(10**400)


def test_snapshot_text_cannot_carry_html_trust_into_reports():
    from markupsafe import Markup

    text = "<script>unsafe()</script>"

    class Untrusted(str):
        def __str__(self):
            raise AssertionError("snapshot must bypass caller __str__")

        def __html__(self):
            return text

    class TrustedDecimal(Decimal):
        def __html__(self):
            return text

    report = _report(
        [
            _result(
                metrics={"text": Markup(text), "custom": Untrusted(text)},
                affected_records=[{"value": Markup(text)}],
            )
        ]
    )
    metadata = report.metadata.model_copy(
        update={"extensions": {"plugin": {"value": Markup(text), "number": TrustedDecimal("1.25")}}}
    )
    report = report.model_copy(update={"metadata": metadata})
    assert type(report.metadata.extensions["plugin"]["value"]) is str
    assert type(report.results[0].metrics["text"]) is str
    assert type(report.results[0].metrics["custom"]) is str
    html = render_html(report)
    assert text not in html
    assert "&lt;script&gt;unsafe()&lt;/script&gt;" in html
    assert report.metadata.extensions["plugin"]["value"] == text


def test_result_anchors_are_unique_when_a_natural_id_matches_a_generated_suffix():
    report = _report([_result(rule_id=name, status=Status.FAIL) for name in ("a", "a", "a-2")])
    html = render_html(report)
    ids = re.findall(r'<article class="card finding" id="([^"]+)"', html)
    assert len(ids) == len(set(ids)) == 3
    assert all(f'href="#{anchor}"' in html for anchor in ids)


@pytest.mark.parametrize("reverse", [False, True])
def test_profile_timezone_folds_compare_instants_across_batch_boundaries(reverse):
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    early = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=0)
    late = datetime(2024, 11, 3, 1, 30, tzinfo=zone, fold=1)
    first, last = (late, early) if reverse else (early, late)
    values = pl.Series([first] * 65_536 + [last], dtype=pl.Datetime("us", zone.key))
    frame = pl.DataFrame({"x": values})
    bounds = profile_frame(frame)["columns"]["x"]
    assert bounds["ge"].astimezone(UTC) == early.astimezone(UTC)
    assert bounds["le"].astimezone(UTC) == late.astimezone(UTC)
    bounds = profile_frame(frame, margin=0.1)["columns"]["x"]
    assert bounds["ge"].astimezone(UTC) == early.astimezone(UTC) - timedelta(minutes=6)
    assert bounds["le"].astimezone(UTC) == late.astimezone(UTC) + timedelta(minutes=6)
    assert bounds["ge"].tzinfo == zone and bounds["le"].tzinfo == zone
