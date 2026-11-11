"""The rows a filtered dataset check item assesses, and how its results show them.

A dataset check item with ``where`` whose check does not filter rows itself is
compiled to read a hidden view of its dataset (:class:`~qavf.engine.derived_names.RowFilter`).
Before its rule runs, :class:`FilteredPopulation` counts the dataset's rows and the
matching ones: too few matching rows (none, or fewer than ``min_applicable_rows``)
make the check not assessed, as for a filtered ``expression``. :class:`FilteredResult`
then shows every result as the dataset, never the hidden view, with the condition
among its parameters and the population, applicable and excluded rows among its
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.derived_names import RowFilter
from qavf.input_rules import expression_scope, filtered_scope, filtered_too_few

if TYPE_CHECKING:
    from qavf.engine.ports import DatasetReductions
    from qavf.reports.models import CheckIdentity, RuleResult


def _row_count(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.select(pl.len().alias("rows"))


def _rows_of(ctx: DatasetReductions, name: str) -> int:
    return int(ctx.reduce(name, _row_count).item())


@dataclass(frozen=True, slots=True)
class FilteredPopulation:
    """The rows of a filtered item's dataset, and those its ``where`` selects."""

    rows: RowFilter
    population: int
    applicable: int

    @classmethod
    def count(cls, rows: RowFilter, ctx: DatasetReductions) -> FilteredPopulation:
        """Count the dataset's rows and the hidden view's."""
        return cls(rows, _rows_of(ctx, rows.dataset), _rows_of(ctx, rows.view))

    def shortfall(self, label: str) -> str | None:
        """Why the check is not assessed (too few matching rows), or ``None``."""
        if self.applicable >= self.rows.minimum:
            return None
        rows = self.rows
        return filtered_too_few(label, rows.dataset, self.applicable, rows.where, rows.minimum)

    def metrics(self) -> dict[str, int]:
        """The population, applicable and excluded rows (as a filtered expression's)."""
        return expression_scope(self.population, self.applicable)

    def scoped(self, result: RuleResult, *, described: bool = True) -> RuleResult:
        """``result`` with the counts in its metrics and, when ``described``, its message."""
        rows = self.rows
        update: dict[str, Any] = {"metrics": {**result.metrics, **self.metrics()}}
        if described:
            scope = filtered_scope(rows.dataset, rows.where, self.population, self.applicable)
            update["message"] = f"{result.message} {scope}".strip()
        return result.model_copy(update=update)


class FilteredResult:
    """Shows a filtered item's results as its dataset, never the hidden view it read."""

    def __init__(self, rows: RowFilter) -> None:
        self.rows = rows

    def shown(self, result: RuleResult) -> RuleResult:
        rows = self.rows
        return result.model_copy(
            update={
                # Results always name the dataset a filtered item is placed on.
                "datasets": rows.shown_datasets(result.datasets) or [rows.dataset],
                "check": rows.shown(result.check),
                "check_id": rows.shown(result.check_id),
                "message": rows.shown(result.message),
                "parameters": {**rows.shown(result.parameters), **rows.parameters()},
                "affected_records": rows.shown(result.affected_records),
                "metrics": rows.shown(result.metrics),
                "checks": [self._check(check) for check in result.checks],
            }
        )

    def _check(self, check: CheckIdentity) -> CheckIdentity:
        rows = self.rows
        update: dict[str, Any] = {
            "check_id": rows.shown(check.check_id),
            "datasets": rows.shown_datasets(check.datasets),
            "parameters": rows.shown(check.parameters),
        }
        return check.model_copy(update=update)


__all__ = ["FilteredPopulation", "FilteredResult"]
