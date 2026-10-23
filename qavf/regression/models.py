"""Immutable, serializable public models for output comparison results."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, ValidationInfo, field_serializer, model_validator

from qavf.audit_codec import decode_audit as decode_non_finite
from qavf.audit_codec import encode_audit as encode_non_finite
from qavf.snapshots import SnapshotModel

Tolerance = int | float | Decimal


def _format_tolerance(value: Tolerance) -> str:
    return str(value) if isinstance(value, int) else format(value, "g")


DifferenceKind = Literal[
    "value_changed", "row_added", "row_removed", "column_added", "column_removed", "dtype_changed"
]
SchemaPolicy = Literal["compatible", "strict", "ignore"]
ComparisonExecution = Literal["in_memory", "native_stream", "typed_stream", "projected_fallback"]
"""How column types are compared: ``compatible`` reports changes of kind (number, text,
boolean, date, datetime with or without time zone, ...) but allows numeric widening
and similar; ``strict`` reports any dtype difference; ``ignore`` compares values only."""


class Difference(SnapshotModel):
    """One discrepancy between baseline and candidate."""

    kind: DifferenceKind
    key: dict[str, Any] = Field(default_factory=dict)
    """Key column values identifying the row (empty for column-level differences)."""
    column: str | None = None
    path: list[Any] = Field(default_factory=list)
    """Location inside a nested cell (list index / struct field), when the change is nested."""
    baseline: Any = None
    candidate: Any = None
    baseline_previewed: bool = False
    candidate_previewed: bool = False
    baseline_length: int | None = None
    candidate_length: int | None = None
    leaf: dict[str, Any] = Field(default_factory=dict)
    """Bounded values at ``path`` when the displayed cells are previews."""
    abs_diff: float | None = None
    rel_diff: float | None = None
    rtol: Tolerance | None = None
    """Relative tolerance applied to this value (its column's own, or the table's)."""
    atol: Tolerance | None = None
    """Absolute tolerance applied to this value."""

    @field_serializer(
        "key",
        "baseline",
        "candidate",
        "leaf",
        "abs_diff",
        "rel_diff",
        "rtol",
        "atol",
        when_used="json",
    )
    def _encode(self, value: Any) -> Any:
        return encode_non_finite(value)

    @model_validator(mode="before")
    @classmethod
    def _decode(cls, data: Any, info: ValidationInfo) -> Any:
        if info.mode == "json" and isinstance(data, dict):
            data = dict(data)
            for key in (
                "key",
                "baseline",
                "candidate",
                "leaf",
                "abs_diff",
                "rel_diff",
                "rtol",
                "atol",
            ):
                if key in data:
                    data[key] = decode_non_finite(data[key])
        return data


class ComparisonResult(SnapshotModel):
    """Outcome of comparing one candidate table with its baseline."""

    execution_limits: dict[str, Any] = Field(default_factory=dict)
    """Effective isolated worker controls; empty for in-process comparisons."""
    name: str
    execution: ComparisonExecution = "in_memory"
    """Selected path: native batches, exact indexed typed batches, or explicit eager fallback."""
    fallback_reason: str | None = None
    """Why an input could not use the native streamed comparison, when applicable."""
    keys: list[str]
    rtol: Tolerance
    atol: Tolerance
    column_tolerances: dict[str, tuple[Tolerance, Tolerance]] = Field(default_factory=dict)
    """Columns compared with their own ``(rtol, atol)`` instead of the table's."""
    column_map: dict[str, str] = Field(default_factory=dict)
    """Baseline columns compared under candidate names (``baseline: candidate``)."""
    baseline_rows: int
    candidate_rows: int
    rows_added: int = 0
    rows_removed: int = 0
    rows_changed: int = 0
    values_changed: int = 0
    columns_added: list[str] = Field(default_factory=list)
    columns_removed: list[str] = Field(default_factory=list)
    dtypes_changed: list[str] = Field(default_factory=list)
    """Shared columns whose kind or nested structure changed. Incompatible kinds
    are not value-compared; structural changes retain useful cell-level evidence."""
    max_abs_diff: float | None = None
    """Largest absolute difference among changes between finite numbers."""
    max_rel_diff: float | None = None
    """Largest relative difference among changes between finite numbers."""
    column_changes: dict[str, int] = Field(default_factory=dict)
    """Changed values per compared column, in comparison order; unchanged columns
    are omitted. The counts sum to ``values_changed``."""
    nonfinite_changes: int = 0
    """Changed values where either side is NaN or infinite; they have no difference
    size and are excluded from the maxima."""
    differences: list[Difference] = Field(default_factory=list)
    """Individual differences, capped at ``max_differences``."""
    truncated: bool = False

    @field_serializer(
        "max_abs_diff", "max_rel_diff", "rtol", "atol", "column_tolerances", when_used="json"
    )
    def _encode(self, value: Any) -> Any:
        return encode_non_finite(value)

    @model_validator(mode="before")
    @classmethod
    def _decode(cls, data: Any, info: ValidationInfo) -> Any:
        if info.mode == "json" and isinstance(data, dict):
            data = dict(data)
            for key in ("max_abs_diff", "max_rel_diff", "rtol", "atol", "column_tolerances"):
                if key in data:
                    data[key] = decode_non_finite(data[key])
        return data

    @property
    def total_differences(self) -> int:
        """All cell, row, column and type differences, including omitted evidence."""
        return (
            self.values_changed
            + self.rows_added
            + self.rows_removed
            + len(self.columns_added)
            + len(self.columns_removed)
            + len(self.dtypes_changed)
        )

    @property
    def equal(self) -> bool:
        """True when the candidate matches the baseline within tolerance."""
        return not (
            self.rows_added
            or self.rows_removed
            or self.values_changed
            or self.columns_added
            or self.columns_removed
            or self.dtypes_changed
        )

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.equal:
            return (
                f"{self.name}: matches baseline ({self.baseline_rows:,} rows, "
                f"rtol={_format_tolerance(self.rtol)}, atol={_format_tolerance(self.atol)})"
            )
        parts = []
        if self.values_changed:
            parts.append(
                f"{self.values_changed:,} value(s) changed in {self.rows_changed:,} row(s)"
            )
        if self.rows_added:
            parts.append(f"{self.rows_added:,} row(s) added")
        if self.rows_removed:
            parts.append(f"{self.rows_removed:,} row(s) removed")
        if self.columns_added:
            parts.append(f"column(s) added: {', '.join(self.columns_added)}")
        if self.columns_removed:
            parts.append(f"column(s) removed: {', '.join(self.columns_removed)}")
        if self.dtypes_changed:
            parts.append(f"column type(s) changed: {', '.join(self.dtypes_changed)}")
        if self.max_rel_diff is not None:
            parts.append(f"max relative difference {self.max_rel_diff:.3g}")
        if self.nonfinite_changes:
            parts.append(f"{self.nonfinite_changes:,} change(s) to or from NaN/infinity")
        return f"{self.name}: differs from baseline: " + "; ".join(parts)
