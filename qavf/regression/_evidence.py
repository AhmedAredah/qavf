"""Complete change aggregates with count- and byte-bounded retained evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from qavf.audit_codec import iter_audit_json
from qavf.regression._cells import _MISSING
from qavf.regression.models import Difference


@dataclass
class _Changes:
    """Complete aggregates and bounded evidence; never stores a full diff graph."""

    limit: int
    byte_limit: int
    differences: list[Difference] = field(default_factory=list)
    bytes_used: int = 2
    exhausted: bool = False
    values: int = 0
    columns: dict[str, int] = field(default_factory=dict)
    """Complete changed-value counts per column; ``values`` is their sum."""
    rows: int = 0
    nonfinite: int = 0
    max_abs: float | None = None
    max_rel: float | None = None

    @property
    def remaining(self) -> int:
        if self.exhausted or self.bytes_used >= self.byte_limit:
            return 0
        return max(0, self.limit - len(self.differences))

    def record(self, **fields: Any) -> None:
        if not self.remaining:
            return
        available = self.byte_limit - self.bytes_used - bool(self.differences)
        # Admission runs before freezing: huge strings/keys are streamed only
        # until the allowance is exhausted, never copied into an audit snapshot.
        size = 0
        for chunk in iter_audit_json(fields):
            size += len(chunk)
            if size > available:
                self.exhausted = True
                return
        difference = Difference(**fields)
        size = 0
        for chunk in iter_audit_json(difference.model_dump(mode="python")):
            size += len(chunk)
            if size > available:
                self.exhausted = True
                return
        self.differences.append(difference)
        self.bytes_used += size + (len(self.differences) > 1)

    def changed(self, column: str, count: int = 1) -> None:
        """Count changed values of one column; evidence is recorded separately."""
        self.values += count
        self.columns[column] = self.columns.get(column, 0) + count

    def key(self, frame: pl.DataFrame, position: int, keys: list[str]) -> dict[str, Any] | None:
        if not self.remaining:
            return None
        selected = frame.select(keys).slice(position, 1)
        if selected.estimated_size() > self.byte_limit - self.bytes_used:
            self.exhausted = True
            return None
        return selected.row(0, named=True)

    def maxima(self, absolute: float | None, relative: float | None) -> None:
        if absolute is not None:
            self.max_abs = absolute if self.max_abs is None else max(self.max_abs, absolute)
        if relative is not None:
            self.max_rel = relative if self.max_rel is None else max(self.max_rel, relative)


def _preview(value: Any) -> tuple[Any, bool]:
    """Convert at most a small bounded subtree, never a complete large Series."""
    remaining = 64
    truncated = False

    def visit(item: Any, depth: int = 0) -> Any:
        nonlocal remaining, truncated
        if remaining <= 0 or depth >= 8:
            truncated = True
            return "<preview omitted>"
        remaining -= 1
        if isinstance(item, str) and len(item) > 256:
            truncated = True
            return item[:256] + "…"
        if isinstance(item, bytes) and len(item) > 256:
            truncated = True
            return item[:256]
        if isinstance(item, Mapping):
            out = {}
            for index, (key, child) in enumerate(item.items()):
                if index == 16 or remaining <= 0:
                    truncated = True
                    break
                out[key] = visit(child, depth + 1)
            return out
        if isinstance(item, (list, tuple, pl.Series)):
            sequence = []
            for index in range(min(len(item), 16)):
                if remaining <= 0:
                    break
                sequence.append(visit(item[index], depth + 1))
            truncated |= len(sequence) != len(item)
            return sequence
        # Sets are supported Object-cell evidence; only a bounded selection is
        # retained, then the audit codec applies its deterministic representation.
        if isinstance(item, (set, frozenset)):
            if len(item) <= 16:
                selected = [visit(child, depth + 1) for child in item]
                return sorted(
                    selected, key=lambda child: b"".join(iter_audit_json(child, sort_keys=True))
                )
            truncated = True
            return "<set preview omitted>"
        return item

    result = visit(value)
    return result, truncated


def _capture_native(a: pl.Series, b: pl.Series, position: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, series in (("baseline", a), ("candidate", b)):
        cell = series.slice(position, 1)
        if series.dtype == pl.String and cell.estimated_size() > 256:
            result[name] = cell.str.slice(0, 256)[0]
            result[name + "_previewed"] = True
            result[name + "_length"] = cell.str.len_chars()[0]
        elif series.dtype == pl.Binary and cell.estimated_size() > 256:
            result[name] = "<binary preview omitted>"
            result[name + "_previewed"] = True
            result[name + "_length"] = cell.bin.size()[0]
        else:
            result[name] = cell[0]
    return result


def _capture_cells(before: Any, after: Any, path: list[Any]) -> dict[str, Any]:
    left, left_cut = _preview(before)
    right, right_cut = _preview(after)
    result = {
        "baseline": left,
        "candidate": right,
        "baseline_previewed": left_cut,
        "candidate_previewed": right_cut,
    }
    if not (left_cut or right_cut):
        return result
    for name, value in (("baseline", before), ("candidate", after)):
        if isinstance(value, (Mapping, list, tuple, pl.Series, str, bytes, set, frozenset)):
            result[name + "_length"] = len(value)
    leaf: dict[str, Any] = {}
    for name, value in (("baseline", before), ("candidate", after)):
        for part in path:
            try:
                value = value[part]
            except (KeyError, IndexError, TypeError):
                value = _MISSING
                break
        leaf[name + "_present"] = value is not _MISSING
        if value is not _MISSING:
            leaf[name], leaf[name + "_previewed"] = _preview(value)
    result["leaf"] = leaf
    return result
