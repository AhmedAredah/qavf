"""The run's evidence bounds and admission budgets, behind the data context.

:class:`AdmissionPolicy` decides what a run may keep: how much evidence a rule
retains per result (it satisfies the :class:`~qavf.engine.ports.EvidencePolicy`
port), whether a completed result fits the memory target, and whether a dataset
stays within its ``row_budget``. The data context delegates its checks here; it
still owns when they run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.duckdb_results import first_row
from qavf.engine.resources import ResourceLimitError, retained_size
from qavf.sources.base import SourceError

if TYPE_CHECKING:
    from qavf.engine.resources import RunResources
    from qavf.settings import DatasetSpec
    from qavf.sources.base import DataSource


class AdmissionPolicy:
    """Evidence bounds, the memory target for results and per-dataset row budgets.

    Args:
        resources: The run's resources (their memory target, read when checked).
        specs: The run's dataset specifications (their load-policy row budgets).
        evidence_limit: Offending records kept per result.
        evidence_bytes: Bytes of evidence kept per result.
    """

    def __init__(
        self,
        resources: RunResources,
        specs: Mapping[str, DatasetSpec],
        evidence_limit: int,
        evidence_bytes: int,
    ) -> None:
        self._resources = resources
        self._specs = specs
        self.evidence_limit = evidence_limit
        """Offending records kept per result (``engine.max_affected_records``)."""
        self.evidence_bytes = evidence_bytes
        """Bytes of evidence kept per result (``engine.max_evidence_bytes``)."""

    def check_materialization(self, name: str, frame: pl.DataFrame) -> None:
        """Admit a completed result against the memory target."""
        if self._resources.memory_bytes is None:
            return
        self.check_retained_bytes(name, retained_size(frame))

    def check_retained_bytes(self, name: str, size: int) -> None:
        """Admit a measured retained result, including incrementally built output."""
        target = self._resources.memory_bytes
        if target is None:
            return
        if size > target:
            raise ResourceLimitError(
                f"{name} materialized {size:,} bytes, exceeding memory_budget_bytes={target:,}; "
                "use a projected scan or increase the target"
            )

    def dataset_row_budget(self, name: str) -> int | None:
        """``name``'s declared row budget (its effective load policy)."""
        spec = self._specs.get(name)
        return spec.effective_load_policy().row_budget if spec is not None else None

    def source_row_budget(self, name: str, source: DataSource | None) -> int | None:
        """The row budget of ``name`` read through ``source`` (its parameters),
        else its declared one."""
        if source is not None:
            return source.params.row_budget
        return self.dataset_row_budget(name)

    def check_rows(self, name: str, frame: pl.DataFrame) -> None:
        """Refuse a materialized dataset past its declared row budget."""
        self.admit_rows(name, self.dataset_row_budget(name), frame.height)

    @staticmethod
    def admit_rows(name: str, budget: int | None, rows: int) -> None:
        """Refuse ``name`` past ``budget`` (``rows`` read so far)."""
        if budget is not None and rows > budget:
            raise SourceError(f"dataset {name!r} has more than {budget:,} rows (row_budget)")

    @classmethod
    def admit_relation(cls, name: str, relation: Any, budget: int | None) -> None:
        """Refuse a native relation past ``budget``, counting at most one row past it."""
        if budget is None:
            return
        count_row = first_row(relation.limit(budget + 1).aggregate("count(*)"))
        if count_row is not None:
            cls.admit_rows(name, budget, count_row[0])


__all__ = ["AdmissionPolicy"]
