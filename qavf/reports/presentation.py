"""Bounded presentation decisions shared by string and streamed HTML output."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HtmlLimits:
    """HTML-specific limits, independent of the canonical JSON report budget."""

    max_cells: int = 10_000
    max_columns: int = 64
    max_output_bytes: int = 16_777_216
    max_chart_points: int = 2_000
    """Most marks (bars, cells, points, zones) one chart draws; a larger one is omitted."""
    max_chart_bytes: int = 2_097_152
    """Bytes all charts of one document may take; later charts are omitted beyond it."""

    def __post_init__(self) -> None:
        for name in (
            "max_cells", "max_columns", "max_output_bytes", "max_chart_points", "max_chart_bytes",
        ):  # fmt: skip
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"HtmlLimits.{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class EvidenceTable:
    records: list[dict[str, Any]]
    columns: tuple[str, ...]
    sparse: bool
    maximum: int
    omitted: bool

    def rows(self) -> Iterator[tuple[Any, ...]]:
        if self.sparse:
            remaining = self.maximum
            for index, record in enumerate(self.records, start=1):
                for key, value in record.items():
                    if not remaining:
                        return
                    yield index, key, value
                    remaining -= 1
        else:
            for record in self.records[: self.maximum]:
                yield tuple(record.get(column) for column in self.columns)


def evidence_table(records: list[dict[str, Any]], limits: HtmlLimits) -> EvidenceTable:
    """Switch heterogeneous evidence to sparse rows before a dense cross-product."""
    columns: dict[str, None] = {}
    populated = sum(len(record) for record in records)
    for record in records:
        for key in record:
            columns[key] = None
            if len(columns) > limits.max_columns:
                break
        if len(columns) > limits.max_columns:
            break
    sparse = len(columns) > limits.max_columns or (
        len(records) * len(columns) > max(1, populated) * 3
    )
    if sparse:
        maximum = limits.max_cells // 3
        return EvidenceTable(
            records, ("Record", "Field", "Value"), True, maximum, populated > maximum
        )
    maximum = limits.max_cells // max(1, len(columns))
    return EvidenceTable(records, tuple(columns), False, maximum, len(records) > maximum)


def bounded_cell(value: Any, formatter, maximum: int = 1024) -> str:
    """Preview cell trees without constructing a complete nested display string."""
    parts: list[str] = []
    remaining = maximum
    truncated = False

    def append(text: str) -> None:
        nonlocal remaining, truncated
        if len(text) > remaining:
            truncated = True
        parts.append(text[:remaining])
        remaining -= min(remaining, len(text))

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal truncated
        if not remaining or depth >= 8:
            truncated = True
            return
        if isinstance(item, Mapping):
            for index, (key, child) in enumerate(item.items()):
                if not remaining:
                    truncated = True
                    break
                if index:
                    append(", ")
                append(str(key))
                append("=")
                visit(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                if not remaining:
                    truncated = True
                    break
                if index:
                    append(", ")
                visit(child, depth + 1)
        elif isinstance(item, (str, bytes)):
            if isinstance(item, bytes):
                selected = item[:remaining]
                truncated |= len(item) > len(selected)
                append(repr(selected))
            else:
                append(item)
        else:
            append(formatter(item))

    visit(value)
    return "".join(parts) + ("… [preview]" if truncated else "")
