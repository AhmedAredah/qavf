"""Comparison levels: each compares the aligned sides at one level of detail.

A level receives both aligned relations (coherent observations, read natively)
and returns a typed :class:`~qavf.compare.results.LevelResult` whose checks
become findings. Raising marks the level not assessed; it never passes.

Writing a level::

    @register_level
    class RowCount(ComparisonLevel):
        name = "row_count"

        def evaluate(self, inputs, name):
            rows = {side: inputs.rows(side) for side in ("baseline", "candidate")}
            same = rows["baseline"] == rows["candidate"]
            return LevelResult(level=self.name, name=name, checks=[
                CheckResult(check="rows", passed=same, message=f"rows {rows}"),
            ])
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from qavf.compare.registry import Component, ComponentRegistry
from qavf.compare.results import CheckResult, LevelResult
from qavf.compare.spec import ComparisonSpec, Side
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.rules._grouped import ProjectedRelation, grouped_reduce
from qavf.rules.helpers import BoundedEvidence, missing_expr

SIDES: tuple[Side, Side] = ("baseline", "candidate")


@dataclass(frozen=True)
class LevelInputs:
    """Both aligned sides of one comparison, valid while the comparison runs."""

    ctx: DataContext
    spec: ComparisonSpec
    observations: Mapping[Side, Observation]
    datasets: Mapping[Side, str]
    units: Mapping[Side, Mapping[str, str | None]]
    """Declared column units after alignment."""
    nullable: Mapping[Side, Mapping[str, bool]]
    """Declared ``nullable`` flags after alignment."""
    row_counts: Mapping[Side, int]

    def schema(self, side: Side) -> dict[str, pl.DataType]:
        return self.observations[side].schema()

    def scan(self, side: Side) -> pl.LazyFrame:
        return self.observations[side].scan()

    def collect(self, plan: pl.LazyFrame) -> pl.DataFrame:
        """Collect a plan over either or both sides under the run's result limits."""
        for observation in self.observations.values():
            observation._check()
        return self.observations["baseline"].collect(plan)

    def rows(self, side: Side) -> int:
        return self.row_counts[side]

    def relation(self, side: Side) -> ProjectedRelation:
        """A side as a scoped relation for the shared grouped reductions."""
        observation = self.observations[side]
        return ProjectedRelation(
            self.ctx,
            self.datasets[side],
            observation.schema(),
            lazy=observation.scan(),
            observation=observation,
            collector=observation.collect,
        )

    def missing_counts(self, side: Side, columns: list[str]) -> dict[str, int]:
        """Null-or-NaN values per column on one side, in one native pass."""
        relation = self.relation(side)
        empty = pl.DataFrame(schema=relation.schema)
        reductions = {c: missing_expr(empty, c).sum().cast(pl.UInt64) for c in columns}
        frame = grouped_reduce(relation, (), reductions)
        return {c: int(frame[c][0] or 0) for c in columns}

    def evidence(self) -> BoundedEvidence:
        """A record collector bounded by the run's evidence limits."""
        return BoundedEvidence(self.ctx.evidence_limit, byte_limit=self.ctx.evidence_bytes)

    def compared_columns(self) -> list[str]:
        """Columns on both sides, in baseline order, without ignored columns."""
        candidate = self.schema("candidate")
        ignored = set(self.spec.ignore_columns)
        return [c for c in self.schema("baseline") if c in candidate and c not in ignored]


class ComparisonLevel(Component):
    """Compare the aligned sides at one level; ``evaluate`` returns a typed result."""

    Result: ClassVar[type[LevelResult]] = LevelResult
    """Result model, also used to read the level back from JSON."""
    requires_keys: ClassVar[bool] = False

    def configuration_errors(self, spec: ComparisonSpec) -> list[str]:
        """Problems visible in the configuration alone, reported when it loads."""
        if self.requires_keys and not spec.keys:
            return ["needs the comparison's 'keys'"]
        return []

    def required_columns(self) -> frozenset[str] | None:
        """Aligned columns read; None preserves all columns for schema checks/plugins."""
        return None

    def datasets(self) -> frozenset[str]:
        """Further configured datasets the level reads (e.g. a zone layer)."""
        return frozenset()

    @abstractmethod
    def evaluate(self, inputs: LevelInputs, name: str) -> LevelResult:
        """Compare the sides; raise for anything that cannot be assessed."""


LEVELS: ComponentRegistry[ComparisonLevel] = ComponentRegistry(
    "comparison level",
    "qavf.compare.levels",
    ComparisonLevel,
    builtins=(
        "qavf.compare.levels.structure",
        "qavf.compare.levels.coverage",
        "qavf.compare.levels.records",
        "qavf.compare.levels.groups",
        "qavf.compare.levels.temporal",
        "qavf.compare.levels.shape",
        "qavf.compare.levels.spatial",
        "qavf.compare.levels.locations",
        "qavf.compare.levels.distribution",
        "qavf.compare.levels.relationships",
        "qavf.compare.levels.contribution",
    ),
)
register_level = LEVELS.register


def graded(
    check: str,
    problems: int,
    ok: str,
    failed: str,
    *,
    evidence: BoundedEvidence | None = None,
    total: int | None = None,
    unit: str | None = None,
    allowed: bool = False,
    metrics: Mapping[str, object] | None = None,
) -> CheckResult:
    """A check that fails when it found problems that are not ``allowed``."""
    passed = not problems or allowed
    return CheckResult(
        check=check,
        passed=passed,
        message=ok if not problems else failed,
        affected_count=problems,
        total=total,
        unit=unit,
        affected_records=evidence.records if evidence is not None else [],
        truncated=evidence.truncated if evidence is not None else False,
        metrics=dict(metrics or {}),
    )


def not_assessed(check: str, reason: str) -> CheckResult:
    return CheckResult(
        check=check, passed=False, assessed=False, message=f"Not assessed: {reason}."
    )


__all__ = [
    "LEVELS",
    "SIDES",
    "ComparisonLevel",
    "LevelInputs",
    "graded",
    "not_assessed",
    "register_level",
]
