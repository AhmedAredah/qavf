"""Dependency-free inline SVG charts for the HTML report.

Charts are drawn as plain SVG so the report stays a single self-contained
file: horizontal (stacked) bars for the run summary and, for dataset
comparisons, a contribution waterfall, a heat map, series lines, paired bars,
diverging deltas and a choropleth of zone outlines. Colours are never set
here: every mark carries a CSS class and the report stylesheet maps classes to
light/dark theme colours. Nothing depends on colour alone: values are written
as text labels, the candidate's marks are dashed or hatched, decreases carry
a hatch pattern, and each mark has a ``<title>`` with its exact value.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from markupsafe import Markup, escape

_WIDTH = 420
_LABEL_WIDTH = 150
_VALUE_WIDTH = 60
_ROW_HEIGHT = 30
_BAR_HEIGHT = 14
_SEGMENT_GAP = 2
_MAX_LABEL_CHARS = 22


@dataclass(frozen=True, slots=True)
class Segment:
    """One coloured part of a bar."""

    name: str
    value: float
    css_class: str


@dataclass(frozen=True, slots=True)
class BarRow:
    """One labelled bar made of one or more segments."""

    label: str
    segments: Sequence[Segment]
    value_label: str | None = None
    """Text at the end of the bar; the formatted total when omitted."""

    @property
    def total(self) -> float:
        """Sum of the segment values."""
        return sum(segment.value for segment in self.segments)


def _shorten(text: str) -> str:
    return text if len(text) <= _MAX_LABEL_CHARS else text[: _MAX_LABEL_CHARS - 1] + "…"


def _format(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def horizontal_bars(rows: Sequence[BarRow], aria_label: str) -> Markup:
    """Render ``rows`` as a (stacked) horizontal bar chart.

    Returns an empty-state paragraph when there is nothing to draw.
    """
    rows = [row for row in rows if row.segments]
    scale_max = max((row.total for row in rows), default=0.0)
    if not rows or scale_max <= 0:
        return Markup('<p class="empty">Nothing to show.</p>')

    bar_area = _WIDTH - _LABEL_WIDTH - _VALUE_WIDTH
    height = len(rows) * _ROW_HEIGHT + 6
    parts: list[str] = [
        f'<svg class="chart" viewBox="0 0 {_WIDTH} {height}" role="img" '
        f'aria-label="{escape(aria_label)}" preserveAspectRatio="xMinYMin meet">',
        f'<line class="axis" x1="{_LABEL_WIDTH}" y1="0" x2="{_LABEL_WIDTH}" y2="{height}"/>',
    ]
    for index, row in enumerate(rows):
        top = index * _ROW_HEIGHT + 3
        middle = top + _ROW_HEIGHT / 2
        parts.append(
            f'<text class="label" x="{_LABEL_WIDTH - 10}" y="{middle:.1f}" '
            f'text-anchor="end" dominant-baseline="central">'
            f"<title>{escape(row.label)}</title>{escape(_shorten(row.label))}</text>"
        )
        x = float(_LABEL_WIDTH)
        visible = [segment for segment in row.segments if segment.value > 0]
        for segment in visible:
            width = max(segment.value / scale_max * bar_area, 2.0)
            drawn = max(width - (_SEGMENT_GAP if len(visible) > 1 else 0), 1.5)
            parts.append(
                f'<rect class="bar {escape(segment.css_class)}" x="{x:.1f}" '
                f'y="{middle - _BAR_HEIGHT / 2:.1f}" width="{drawn:.1f}" '
                f'height="{_BAR_HEIGHT}" rx="3">'
                f"<title>{escape(row.label)} · {escape(segment.name)}: "
                f"{escape(_format(segment.value))}</title></rect>"
            )
            x += width
        parts.append(
            f'<text class="value" x="{x + 8:.1f}" y="{middle:.1f}" '
            f'dominant-baseline="central">'
            f"{escape(row.value_label or _format(row.total))}</text>"
        )
    parts.append("</svg>")
    return Markup("".join(parts))


# ------------------------------------------------------ comparison charts
_PLOT_WIDTH = 560
_EMPTY = Markup('<p class="empty">Nothing to show.</p>')
_ids = itertools.count(1)


def reset_ids() -> None:
    """Number pattern ids from 1 again, so one document renders deterministically."""
    global _ids
    _ids = itertools.count(1)


def signed(value: float | None, percent: bool = False) -> str:
    """``+1.23k``-style compact text of a change (``-`` for none)."""
    if value is None or not math.isfinite(value):
        return "—"
    if percent:
        return f"{value * 100:+.3g}%"
    magnitude = abs(value)
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
            return f"{value / limit:+.3g}{suffix}"
    return f"{value:+.3g}"


def plain(value: float | None) -> str:
    """Compact text of a level (no sign)."""
    text = signed(value)
    return text[1:] if text.startswith("+") else text


def _open(width: float, height: float, aria: str, css: str = "chart cmp") -> tuple[str, str]:
    """The ``<svg>`` start with its hatch pattern, and the pattern's id."""
    hatch = f"hatch-{next(_ids)}"
    return (
        f'<svg class="{css}" viewBox="0 0 {width:g} {height:g}" role="img" '
        f'aria-label="{escape(aria)}" preserveAspectRatio="xMinYMin meet">'
        f'<defs><pattern id="{hatch}" width="6" height="6" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><line class="hatch" x1="0" y1="0" x2="0" y2="6"/>'
        f"</pattern></defs>",
        hatch,
    )


def _text(x: float, y: float, text: str, css: str = "label", anchor: str = "start") -> str:
    return (
        f'<text class="{css}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'dominant-baseline="central">{escape(text)}</text>'
    )


def diverging_class(value: float | None, scale: float) -> str:
    """``div-d2`` ... ``div-u2``: five classes of a change against ``scale`` (``na``: none)."""
    if value is None or not math.isfinite(value):
        return "na"
    if not scale:
        return "div-z"
    share = value / scale
    if share <= -0.6:
        return "div-d2"
    if share < -0.2:
        return "div-d1"
    if share <= 0.2:
        return "div-z"
    return "div-u1" if share < 0.6 else "div-u2"


def _legend(scale: float, percent: bool) -> list[tuple[str, str]]:
    cuts = [-scale, -0.6 * scale, -0.2 * scale, 0.2 * scale, 0.6 * scale, scale]
    names = ["div-d2", "div-d1", "div-z", "div-u1", "div-u2"]
    return [
        (name, f"{signed(low, percent)} to {signed(high, percent)}")
        for name, low, high in zip(names, cuts, cuts[1:], strict=False)
    ]


_LEGEND_ROW = 16
_LEGEND_HEIGHT = 5 * _LEGEND_ROW + 6
"""Height of the five-class legend, stacked one class per row."""


def _legend_marks(entries: Sequence[tuple[str, str]], top: float, hatch: str) -> str:
    parts = []
    for index, (css, text) in enumerate(entries):
        y = top + index * _LEGEND_ROW
        parts.append(f'<rect class="{css}" x="0" y="{y:.1f}" width="12" height="12"/>')
        if css.startswith("div-d"):
            parts.append(f'<rect fill="url(#{hatch})" x="0" y="{y:.1f}" width="12" height="12"/>')
        parts.append(_text(18, y + 6, text, "legend-text"))
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class WaterfallStep:
    """A bar of a waterfall: a ``total`` from zero, or a ``change`` from the running total."""

    label: str
    value: float
    kind: str = "change"


def waterfall(steps: Sequence[WaterfallStep], aria: str) -> Markup:
    """Baseline total, each member's change, candidate total: horizontal bars.

    Increases are solid, decreases hatched, and every bar is labelled with its
    signed value, so the chart reads without colour.
    """
    if not steps:
        return _EMPTY
    running = 0.0
    spans = []
    for step in steps:
        start = 0.0 if step.kind == "total" else running
        end = step.value if step.kind == "total" else running + step.value
        spans.append((min(start, end), max(start, end), start, end))
        running = end
    # The axis spans the running totals, not zero: changes small against the totals
    # stay visible. Totals are drawn from the axis start, which is labelled.
    ends = [s[3] for s in spans]
    low, high = min(ends), max(ends)
    margin = (high - low) * 0.1 or abs(high) * 0.1 or 1.0
    low = 0.0 if low >= 0 and low - margin <= 0 else low - margin
    high = high + margin if high > 0 or low < 0 else 0.0
    span = (high - low) or 1.0
    left, area = 170.0, _PLOT_WIDTH - 170.0 - 70.0
    height = len(steps) * 24 + 26
    svg, hatch = _open(_PLOT_WIDTH, height, aria)
    parts = [svg]
    parts.append(f'<line class="axis" x1="{left}" y1="0" x2="{left}" y2="{height - 16}"/>')
    parts.append(_text(left, height - 8, f"axis from {plain(low)}", "tick"))
    spans = [(max(lo, low), max(hi, low), start, end) for lo, hi, start, end in spans]
    for index, (step, (lo, hi, start, end)) in enumerate(zip(steps, spans, strict=True)):
        top = index * 24 + 5
        x = left + (lo - low) / span * area
        width = max((hi - lo) / span * area, 1.5)
        rising = end >= start
        css = "wf-total" if step.kind == "total" else ("wf-up" if rising else "wf-down")
        value = plain(step.value) if step.kind == "total" else signed(step.value)
        title = f"{step.label}: {value}"
        parts.append(_text(left - 8, top + 7, _shorten_to(step.label, 26), anchor="end"))
        parts.append(
            f'<rect class="bar {css}" x="{x:.1f}" y="{top:.1f}" width="{width:.1f}" '
            f'height="14"><title>{escape(title)}</title></rect>'
        )
        if css == "wf-down":
            parts.append(
                f'<rect fill="url(#{hatch})" x="{x:.1f}" y="{top:.1f}" width="{width:.1f}" '
                f'height="14"/>'
            )
        parts.append(_text(x + width + 4, top + 7, value, "value"))
    parts.append("</svg>")
    return Markup("".join(parts))


def _shorten_to(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def heat_map(
    rows: Sequence[str],
    columns: Sequence[str],
    cells: Mapping[tuple[str, str], float | None],
    aria: str,
    *,
    percent: bool = True,
    labels: bool = True,
) -> Markup:
    """A ``rows x columns`` grid of changes in five diverging classes, with a legend.

    With ``labels`` each cell shows its value; decreases are hatched either way.
    """
    if not rows or not columns:
        return _EMPTY
    finite = [abs(v) for v in cells.values() if v is not None and math.isfinite(v)]
    scale = max(finite, default=0.0)
    left, top = 110.0, 40.0
    cell = max(12.0, min(48.0, (_PLOT_WIDTH - left) / len(columns)))
    rowh = 18.0 if labels else 12.0
    width = left + cell * len(columns)
    height = top + rowh * len(rows) + 14 + _LEGEND_HEIGHT
    svg, hatch = _open(width, height, aria)
    parts = [svg]
    step = max(1, math.ceil(len(columns) / max(1, (width - left) // 60)))
    for j, column in enumerate(columns):
        if j % step == 0:
            parts.append(_text(left + j * cell + 2, top - 10, _shorten_to(column, 12), "tick"))
    for i, row in enumerate(rows):
        y = top + i * rowh
        parts.append(_text(left - 6, y + rowh / 2, _shorten_to(row, 16), anchor="end"))
        for j, column in enumerate(columns):
            value = cells.get((row, column))
            css = diverging_class(value, scale)
            x = left + j * cell
            title = f"{row} · {column}: {signed(value, percent)}"
            parts.append(
                f'<rect class="cell {css}" x="{x:.1f}" y="{y:.1f}" width="{cell - 1:.1f}" '
                f'height="{rowh - 1:.1f}"><title>{escape(title)}</title></rect>'
            )
            if css.startswith("div-d"):
                parts.append(
                    f'<rect fill="url(#{hatch})" x="{x:.1f}" y="{y:.1f}" '
                    f'width="{cell - 1:.1f}" height="{rowh - 1:.1f}"/>'
                )
            if labels and cell >= 34 and value is not None:
                parts.append(
                    _text(x + (cell - 1) / 2, y + rowh / 2, signed(value, percent),
                          "cell-text", "middle")
                )  # fmt: skip
    parts.append(_legend_marks(_legend(scale, percent), height - _LEGEND_HEIGHT, hatch))
    parts.append("</svg>")
    return Markup("".join(parts))


def series_lines(
    labels: Sequence[str],
    baseline: Sequence[float | None],
    candidate: Sequence[float | None],
    aria: str,
) -> Markup:
    """Baseline (solid, circles) and candidate (dashed, squares) per bucket."""
    values = [v for v in (*baseline, *candidate) if v is not None and math.isfinite(v)]
    if not labels or not values:
        return _EMPTY
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    left, top, plot_h = 60.0, 24.0, 150.0
    area = _PLOT_WIDTH - left - 10
    height = top + plot_h + 36
    svg, _ = _open(_PLOT_WIDTH, height, aria)
    parts = [svg]

    def x_of(i: int) -> float:
        return left + (i / (len(labels) - 1) * area if len(labels) > 1 else area / 2)

    def y_of(v: float) -> float:
        return top + plot_h - (v - low) / span * plot_h

    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(_text(left - 6, top, plain(high), "tick", "end"))
    parts.append(_text(left - 6, top + plot_h, plain(low), "tick", "end"))
    for i in sorted({0, len(labels) // 2, len(labels) - 1}):
        parts.append(_text(x_of(i), top + plot_h + 14, _shorten_to(labels[i], 14), "tick",
                           "middle"))  # fmt: skip
    for series, css, marker in (
        (baseline, "line-base", "circle"),
        (candidate, "line-cand", "rect"),
    ):
        points = [(x_of(i), y_of(v)) for i, v in enumerate(series) if v is not None]
        if not points:
            continue
        path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points)
        parts.append(f'<path class="{css}" d="{path}"/>')
        name = "baseline" if css == "line-base" else "candidate"
        if len(points) <= 120:
            present = [(i, v) for i, v in enumerate(series) if v is not None]
            for (x, y), (i, v) in zip(points, present, strict=True):
                title = f"{labels[i]} · {name}: {plain(v)}"
                shape = (
                    f'<circle class="{css}-mark" cx="{x:.1f}" cy="{y:.1f}" r="2.6">'
                    if marker == "circle"
                    else f'<rect class="{css}-mark" x="{x - 2.4:.1f}" y="{y - 2.4:.1f}" '
                    f'width="4.8" height="4.8">'
                )
                closing = "</circle>" if marker == "circle" else "</rect>"
                parts.append(f"{shape}<title>{escape(title)}</title>{closing}")
    parts.append(
        f'<line class="line-base" x1="{left}" y1="8" x2="{left + 24}" y2="8"/>'
        + _text(left + 30, 8, "baseline (solid)", "legend-text")
        + f'<line class="line-cand" x1="{left + 150}" y1="8" x2="{left + 174}" y2="8"/>'
        + _text(left + 180, 8, "candidate (dashed)", "legend-text")
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def paired_bars(
    labels: Sequence[str], baseline: Sequence[float], candidate: Sequence[float], aria: str
) -> Markup:
    """Each side's share per bin or category: baseline solid, candidate hatched."""
    if not labels:
        return _EMPTY
    high = max([*baseline, *candidate, 0.0]) or 1.0
    left, top, plot_h = 46.0, 24.0, 130.0
    group = (_PLOT_WIDTH - left - 10) / len(labels)
    bar = max(min(group / 2 - 1, 22.0), 1.0)
    height = top + plot_h + 40
    svg, hatch = _open(_PLOT_WIDTH, height, aria)
    parts = [svg]
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{_PLOT_WIDTH - 10}" '
                 f'y2="{top + plot_h}"/>')  # fmt: skip
    parts.append(_text(left - 6, top, f"{high * 100:.3g}%", "tick", "end"))
    step = max(1, math.ceil(len(labels) / 8))
    for i, label in enumerate(labels):
        x = left + i * group + (group - 2 * bar) / 2
        for k, (value, css) in enumerate(((baseline[i], "bar-base"), (candidate[i], "bar-cand"))):
            h = value / high * plot_h
            title = f"{label} · {'baseline' if k == 0 else 'candidate'}: {value * 100:.3g}%"
            rect = (
                f'x="{x + k * bar:.1f}" y="{top + plot_h - h:.1f}" width="{bar:.1f}" '
                f'height="{h:.1f}"'
            )
            parts.append(f'<rect class="{css}" {rect}><title>{escape(title)}</title></rect>')
            if k:
                parts.append(f'<rect fill="url(#{hatch})" {rect}/>')
        if i % step == 0:
            parts.append(
                _text(x + bar, top + plot_h + 12, _shorten_to(label, 12), "tick", "middle")
            )
    parts.append(
        f'<rect class="bar-base" x="{left}" y="4" width="12" height="10"/>'
        + _text(left + 16, 9, "baseline", "legend-text")
        + f'<rect class="bar-cand" x="{left + 90}" y="4" width="12" height="10"/>'
        + f'<rect fill="url(#{hatch})" x="{left + 90}" y="4" width="12" height="10"/>'
        + _text(left + 106, 9, "candidate (hatched)", "legend-text")
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def deltas(labels: Sequence[str], values: Sequence[float], aria: str) -> Markup:
    """Signed changes as bars around zero (quantile deltas), each labelled."""
    if not labels:
        return _EMPTY
    scale = max((abs(v) for v in values), default=0.0) or 1.0
    left, area = 90.0, _PLOT_WIDTH - 90.0 - 80.0
    middle = left + area / 2
    height = len(labels) * 22 + 10
    svg, hatch = _open(_PLOT_WIDTH, height, aria)
    parts = [svg, f'<line class="axis" x1="{middle}" y1="0" x2="{middle}" y2="{height}"/>']
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        top = i * 22 + 5
        width = max(abs(value) / scale * area / 2, 1.0)
        x = middle if value >= 0 else middle - width
        css = "wf-up" if value >= 0 else "wf-down"
        parts.append(_text(left - 6, top + 6, label, anchor="end"))
        parts.append(
            f'<rect class="bar {css}" x="{x:.1f}" y="{top:.1f}" width="{width:.1f}" height="12">'
            f"<title>{escape(f'{label}: {signed(value)}')}</title></rect>"
        )
        if value < 0:
            parts.append(
                f'<rect fill="url(#{hatch})" x="{x:.1f}" y="{top:.1f}" width="{width:.1f}" '
                f'height="12"/>'
            )
        end = x + width + 4 if value >= 0 else x - 4
        parts.append(_text(end, top + 6, signed(value), "value", "start" if value >= 0 else "end"))
    parts.append("</svg>")
    return Markup("".join(parts))


def choropleth(
    paths: Mapping[str, str],
    centres: Mapping[str, Sequence[float]],
    values: Mapping[str, float | None],
    size: tuple[float, float],
    aria: str,
    *,
    percent: bool = True,
    labels: bool = True,
) -> Markup:
    """Zones filled by their change in five diverging classes; decreases hatched.

    ``paths`` are SVG path data in a ``size`` view box (north up). Zones without a
    value are outlined only. With ``labels`` each zone's change is written inside it.
    """
    if not paths:
        return _EMPTY
    finite = [abs(v) for v in values.values() if v is not None and math.isfinite(v)]
    scale = max(finite, default=0.0)
    width, height = size
    zoom = max(width / _PLOT_WIDTH, 1.0)
    svg, hatch = _open(width, height + (_LEGEND_HEIGHT + 16) * zoom, aria, "chart cmp map")
    parts = [svg]
    for zone, path in paths.items():
        value = values.get(zone)
        css = diverging_class(value, scale)
        title = f"{zone}: {signed(value, percent)}"
        parts.append(
            f'<path class="zone {css}" d="{escape(path)}"><title>{escape(title)}</title></path>'
        )
        if css.startswith("div-d"):
            parts.append(f'<path fill="url(#{hatch})" d="{escape(path)}"/>')
    if labels:
        for zone, (x, y) in centres.items():
            text = f"{zone} {signed(values.get(zone), percent)}"
            parts.append(
                f'<text class="map-text" x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="{11 * zoom:.3g}">{escape(text)}</text>'
            )
    scale_text = _legend(scale, percent)
    parts.append(
        f'<g transform="scale({zoom:.4g})">'
        + _legend_marks(scale_text, height / zoom + 12, hatch)
        + "</g>"
    )
    parts.append("</svg>")
    return Markup("".join(parts))
