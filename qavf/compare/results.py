"""Immutable, serializable results of a dataset comparison.

Each level reports typed details plus graded checks; a check becomes one CMP-001
finding. Evidence is bounded by the run's record and byte limits while counts
stay complete. Values of arbitrary type (keys, categories) live in
``affected_records``/``metrics`` or in fields with audit encoding, so a result
survives JSON transport from an isolated worker with its types intact.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import Field, SerializeAsAny, ValidationInfo, field_serializer, field_validator

from qavf.reports.models import SnapshotModel, encode_non_finite


class Significance(SnapshotModel):
    """Slot for a statistical-significance judgement (not computed in this release)."""

    method: str
    statistic: float | None = None
    p_value: float | None = None
    alpha: float | None = None
    significant: bool | None = None


class CheckResult(SnapshotModel):
    """One graded check of a level; ``assessed=False`` is never a pass."""

    audit_fields: ClassVar[tuple[str, ...]] = ("affected_records", "metrics")

    check: str
    passed: bool
    assessed: bool = True
    message: str
    affected_count: int = 0
    """Complete count of problems, in ``unit``; evidence may hold fewer."""
    total: int | None = None
    """Items examined, in ``unit``."""
    unit: str | None = None
    affected_records: list[dict[str, Any]] = Field(default_factory=list)
    """Bounded, deterministically ordered evidence."""
    truncated: bool = False
    """Whether counted problems were left out of ``affected_records``."""
    metrics: dict[str, Any] = Field(default_factory=dict)


class LevelResult(SnapshotModel):
    """A level's checks; subclasses add the level's typed details."""

    level: str
    """Registered level name, which selects the result type when reading JSON."""
    name: str
    """Report label (several levels of one kind need distinct names)."""
    checks: list[CheckResult] = Field(default_factory=list)
    significance: Significance | None = None

    @property
    def assessed(self) -> bool:
        return all(check.assessed for check in self.checks)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class _Values(SnapshotModel):
    """Encode non-finite and rich values in ``baseline``/``candidate`` fields."""

    audit_fields: ClassVar[tuple[str, ...]] = ("baseline", "candidate")

    @field_serializer("baseline", "candidate", when_used="json", check_fields=False)
    def _encode_values(self, value: Any) -> Any:
        return encode_non_finite(value)


# ---------------------------------------------------------------- structure
class TypeChange(SnapshotModel):
    column: str
    baseline: str
    candidate: str


class UnitChange(SnapshotModel):
    column: str
    baseline: str | None
    candidate: str | None


class NullabilityChange(SnapshotModel):
    column: str
    baseline_missing: int
    candidate_missing: int
    baseline_declared: bool | None = None
    """Declared ``nullable`` in the baseline schema, when declared."""
    candidate_declared: bool | None = None


class RenameSuggestion(SnapshotModel):
    """A removed and an added column that may be one renamed column. Never applied."""

    removed: str
    added: str
    score: float
    name_similarity: float
    value_similarity: float | None = None
    """Jaccard similarity of bounded sorted samples of distinct values."""


class StructureResult(LevelResult):
    baseline_rows: int = 0
    candidate_rows: int = 0
    columns_added: list[str] = Field(default_factory=list)
    columns_removed: list[str] = Field(default_factory=list)
    type_changes: list[TypeChange] = Field(default_factory=list)
    unit_changes: list[UnitChange] = Field(default_factory=list)
    nullability_changes: list[NullabilityChange] = Field(default_factory=list)
    rename_suggestions: list[RenameSuggestion] = Field(default_factory=list)
    duplicate_keys: dict[str, int] = Field(default_factory=dict)
    """Key values repeated on each side (groups of rows sharing one key)."""


# ----------------------------------------------------------------- coverage
class NullRateChange(SnapshotModel):
    column: str
    baseline_rate: float | None
    candidate_rate: float | None
    change: float | None


class CoverageResult(LevelResult):
    keys: list[str] = Field(default_factory=list)
    baseline_keys: int = 0
    candidate_keys: int = 0
    shared_keys: int = 0
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    rows_only_in_baseline: int = 0
    rows_only_in_candidate: int = 0
    null_key_rows: dict[str, int] = Field(default_factory=dict)
    group_by: list[str] = Field(default_factory=list)
    null_rates: list[NullRateChange] = Field(default_factory=list)


# ------------------------------------------------------------------ records
class Transition(_Values):
    """How often one baseline value became one candidate value."""

    baseline: Any = None
    candidate: Any = None
    count: int


class ColumnChanges(SnapshotModel):
    column: str
    changed: int = 0
    """Values that differ beyond tolerance (the regression engine's decision)."""
    rate: float | None = None
    """``changed`` over matched rows."""
    max_abs_diff: float | None = None
    """Largest absolute difference among differing finite pairs, within tolerance or not."""
    max_rel_diff: float | None = None
    null_to_value: int = 0
    value_to_null: int = 0
    nan_to_value: int = 0
    value_to_nan: int = 0
    transitions: list[Transition] = Field(default_factory=list)
    """Most frequent ``baseline -> candidate`` changes of a categorical column."""
    distinct_transitions: int | None = None
    """All distinct changes of a categorical column, including those not listed."""

    @field_serializer("max_abs_diff", "max_rel_diff", when_used="json")
    def _encode(self, value: Any) -> Any:
        return encode_non_finite(value)


class RecordsResult(LevelResult):
    keys: list[str] = Field(default_factory=list)
    matched: int = 0
    rows_added: int = 0
    rows_removed: int = 0
    rows_changed: int = 0
    rows_unchanged: int = 0
    values_changed: int = 0
    columns: list[ColumnChanges] = Field(default_factory=list)
    execution: str | None = None
    """How the keyed comparison ran (see :class:`qavf.regression.ComparisonResult`)."""


# ------------------------------------------------------------------- groups
class MeasureComparison(SnapshotModel):
    label: str
    measure: str
    column: str | None = None
    comparators: list[str] = Field(default_factory=list)
    compared: int = 0
    """Groups present on both sides."""
    material: int = 0
    """Groups whose values differ beyond the tolerance."""
    rtol: float = 0.0
    atol: float = 0.0
    significance: Significance | None = None


class RankChange(SnapshotModel):
    measure: str
    top_k: int
    entered: int = 0
    """Groups in the candidate's top ``k`` but not the baseline's."""
    left: int = 0
    moved: int = 0
    """Groups in both top lists at a different rank."""


class GroupsResult(LevelResult):
    group_by: list[str] = Field(default_factory=list)
    baseline_groups: int = 0
    candidate_groups: int = 0
    only_in_baseline: int = 0
    only_in_candidate: int = 0
    measures: list[MeasureComparison] = Field(default_factory=list)
    rank: RankChange | None = None


# --------------------------------------------------------------- comparison
class DatasetComparison(SnapshotModel):
    """Outcome of comparing a candidate dataset with its baseline."""

    name: str
    baseline: str
    candidate: str
    keys: list[str] = Field(default_factory=list)
    column_map: dict[str, str] = Field(default_factory=dict)
    alignment: list[str] = Field(default_factory=list)
    """The alignment steps applied, in order."""
    baseline_rows: int | None = None
    """Rows after alignment (``None`` when alignment did not complete)."""
    candidate_rows: int | None = None
    error: str | None = None
    """Why nothing could be compared (unavailable data, refused alignment)."""
    levels: list[SerializeAsAny[LevelResult]] = Field(default_factory=list)
    execution_limits: dict[str, Any] = Field(default_factory=dict)
    """Effective isolated worker controls; empty for in-process comparisons."""

    @field_validator("levels", mode="before")
    @classmethod
    def _typed_levels(cls, value: Any, info: ValidationInfo) -> Any:
        """Read each level with its registered result type (plugins included).

        JSON input is re-read as JSON so audit-encoded evidence decodes exactly.
        """
        if not isinstance(value, list | tuple):
            return value
        from qavf.compare.levels import LEVELS

        typed = []
        for item in value:
            if isinstance(item, dict) and item.get("level") in LEVELS:
                result = LEVELS.get(item["level"]).Result
                item = (
                    result.model_validate_json(json.dumps(item))
                    if info.mode == "json"
                    else result.model_validate(item)
                )
            typed.append(item)
        return typed

    @property
    def assessed(self) -> bool:
        """Every configured check was assessed."""
        return self.error is None and all(level.assessed for level in self.levels)

    @property
    def passed(self) -> bool:
        """Every check was assessed and passed."""
        return self.assessed and all(level.passed for level in self.levels)

    def level(self, name: str) -> LevelResult:
        """The level reported as ``name``.

        Raises:
            KeyError: if no level has that name.
        """
        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(f"no level named {name!r}")

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.error is not None:
            return f"{self.name}: not assessed: {self.error}"
        failed = [
            f"{level.name}: {check.check}"
            for level in self.levels
            for check in level.checks
            if not check.passed
        ]
        if not failed:
            return f"{self.name}: datasets agree ({len(self.levels)} level(s) compared)"
        return f"{self.name}: differs: " + "; ".join(failed)
