"""Admit independent statistical contracts before sharing source projections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

from qavf.engine.registry import Finding
from qavf.rules._contracts import isolate_contract
from qavf.rules.helpers import usable

if TYPE_CHECKING:
    from qavf.engine.context import DataContext
    from qavf.rules.statistical_checks import DriftComparison, OutlierTarget


def outlier_targets(
    ctx: DataContext, targets: Sequence[OutlierTarget], findings: list[Finding]
) -> list[OutlierTarget]:
    admitted = []
    for target in targets:
        with isolate_contract(target.label, (target.dataset,), findings):
            reason = usable(ctx, target.dataset, (*target.group_by, target.value))
            if reason is None:
                schema = ctx.schema(target.dataset)
                if not schema[target.value].is_numeric():
                    reason = f"statistical values must be numeric, found {schema[target.value]}"
                elif any(schema[key] == pl.Object for key in target.group_by):
                    reason = "opaque Object columns cannot be statistical grouping keys"
            if reason is not None:
                findings.append(
                    Finding.unassessed(
                        f"Not assessed: target {target.label}: {reason}.",
                        check=target.label,
                        datasets=(target.dataset,),
                    )
                )
            else:
                admitted.append(target)
    return admitted


def drift_comparisons(
    ctx: DataContext, comparisons: Sequence[DriftComparison], findings: list[Finding]
) -> list[DriftComparison]:
    admitted = []
    for comparison in comparisons:
        names = (comparison.dataset, comparison.reference)
        with isolate_contract(comparison.label(), names, findings):
            reason = None
            for name in names:
                reason = usable(ctx, name, (comparison.column,))
                if reason is None and ctx.schema(name)[comparison.column] == pl.Object:
                    reason = "opaque Object columns cannot be distribution observations"
                if reason is not None:
                    break
            if reason is not None:
                findings.append(
                    Finding.unassessed(
                        f"Not assessed: {comparison.label()}: {reason}.",
                        check=comparison.label(),
                        datasets=names,
                    )
                )
            else:
                admitted.append(comparison)
    return admitted
