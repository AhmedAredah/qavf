"""Dataset selection, missing-value policies and row-count contracts."""

from __future__ import annotations

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.rules.helpers import (
    missing_expr,
)
from qavf.settings import (
    QAVFSettings,
    SchemaSpec,
)
from qavf.sources.sql import quote_identifier

CATEGORY = "input"


class DatasetSelection(RuleParams):
    """Restrict a rule to some datasets; all configured datasets when omitted."""

    datasets: tuple[str, ...] | None = None
    """Datasets to examine; every configured dataset when omitted."""


class _SelectsDatasets(Rule):
    """A rule whose ``datasets`` selection must name configured datasets: a typo would
    otherwise select nothing and the rule would pass without checking anything."""

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        selected = (
            set(self.params.datasets)
            if self.params.datasets is not None
            else set(settings.datasets)
        )
        return DataRequirements(frozenset(selected))

    def configuration_errors(self, settings: QAVFSettings) -> list[str]:
        selection = getattr(self.params, "datasets", None) or ()
        unknown = sorted(name for name in selection if name not in settings.datasets)
        if not unknown:
            return []
        known = ", ".join(sorted(settings.datasets)) or "none"
        return [f"'datasets' names unknown dataset(s) {', '.join(unknown)} (configured: {known})"]


def _blocked(ctx: DataContext, dataset: str, what: str) -> Finding:
    """A finding for a check that could not run because ``dataset`` is unavailable."""
    return Finding.unassessed(
        f"Not assessed: {what} on {dataset!r}: {ctx.unavailable_reason(dataset)}.",
        check=dataset,
        datasets=(dataset,),
    )


def _must_assess(ctx: DataContext, dataset: str, selection: tuple[str, ...] | None) -> bool:
    """Whether an unavailable ``dataset`` means configured checks went unassessed.

    Required datasets, and datasets a rule was explicitly pointed at, must be
    assessed; an optional dataset that is simply absent is not an error.
    """
    spec = ctx.specs.get(dataset)
    if selection is not None and dataset in selection:
        return True
    return spec is None or spec.required


def _schemas(
    ctx: DataContext, selection: tuple[str, ...] | None, what: str
) -> tuple[dict[str, SchemaSpec], list[Finding]]:
    """Schemas of the available datasets to examine, and findings for blocked ones."""
    available: dict[str, SchemaSpec] = {}
    blocked: list[Finding] = []
    for name, spec in ctx.specs.items():
        if spec.schema_spec is None or (selection is not None and name not in selection):
            continue
        if ctx.has(name):
            available[name] = spec.schema_spec
        elif _must_assess(ctx, name, selection):
            blocked.append(_blocked(ctx, name, what))
    return available, blocked


def _missing_markers(schema: SchemaSpec, frame: pl.DataFrame) -> dict[str, pl.Expr]:
    """Per column, an expression that is true for configured sentinels and blanks."""
    markers: dict[str, pl.Expr] = {}
    for column, spec in schema.columns.items():
        if column in frame.columns and (spec.missing_values or spec.blank_as_missing):
            markers[column] = missing_expr(frame, column, spec)
    return markers


def _key_missing(
    frame: pl.DataFrame, columns: list[str], schema: SchemaSpec | None
) -> tuple[pl.Expr, pl.Expr]:
    markers = _missing_markers(schema, frame) if schema is not None else {}
    masks = [missing_expr(frame, c) | markers.get(c, pl.lit(False)) for c in columns]
    return pl.any_horizontal(masks), pl.all_horizontal(masks)


def _row_count(ctx: DataContext, dataset: str) -> int:
    if ctx.execution_kind(dataset) == "native_sql":
        return int(ctx.sql(f"SELECT count(*) FROM {quote_identifier(dataset)}").item())
    return int(ctx.reduce(dataset, lambda frame: frame.select(pl.len())).item())
