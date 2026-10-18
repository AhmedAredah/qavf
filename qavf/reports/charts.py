"""Dependency-free inline SVG charts for the HTML report.

Charts are horizontal bar charts drawn as plain SVG so the report stays a
single self-contained file. Colours are never set here: every mark carries a
CSS class and the report stylesheet maps classes to light/dark theme colours.
Each mark has a ``<title>`` so hovering shows its exact value.
"""

from __future__ import annotations

from collections.abc import Sequence
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
