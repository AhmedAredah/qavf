"""Public dataset comparison over the same input forms as :func:`qavf.validate`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qavf.compare.results import DatasetComparison
from qavf.compare.spec import ComparisonSpec
from qavf.inputs.resources import ResourceLimits

_SIDES = ("baseline", "candidate")


def compare_datasets(
    baseline: object,
    candidate: object,
    *,
    keys: str | Sequence[str] = (),
    columns: Mapping[str, str] | None = None,
    levels: Sequence[str | Mapping[str, Any]] | None = None,
    name: str | None = None,
    resources: ResourceLimits | None = None,
    **options: Any,
) -> DatasetComparison:
    """Compare a candidate dataset with its baseline at several levels.

    ``baseline`` and ``candidate`` may be paths, :class:`~qavf.File`,
    :class:`~qavf.Dataset` (its schema's missing-value markers and units apply),
    :class:`~qavf.Query`, :class:`~qavf.Loader` or frames. ``columns`` maps
    baseline names to candidate names; every other column reference, including
    ``keys``, uses candidate names. ``levels`` lists level names or
    ``{level: name, ...parameters}`` mappings; ``structure``, ``coverage`` and
    ``records`` by default (``structure`` alone without keys). Further options
    are the fields of :class:`~qavf.compare.ComparisonSpec` (``cast``,
    ``units``, ``recode``, ``derive``, ``steps``, ``baseline_where``,
    ``candidate_where``, ``where``, ``ignore_columns``, ``rtol``, ``atol``,
    ``tolerances``, ``min_rows``).

    ``resources`` applies the same :class:`~qavf.ResourceLimits` as validation,
    including isolated execution for transportable file and query inputs.

    Raises:
        pydantic.ValidationError: for an invalid comparison configuration.
        SourceError: if an input is missing or its source policy cannot be satisfied.
    """
    from qavf.inputs.normalize import prepare

    spec = ComparisonSpec.model_validate(
        {
            **options,
            "name": name or "comparison",
            "baseline": "baseline",
            "candidate": "candidate",
            "keys": [keys] if isinstance(keys, str) else list(keys),
            "columns": dict(columns or {}),
            "levels": None if levels is None else list(levels),
        }
    )
    for value in (baseline, candidate):
        if callable(getattr(value, "read_next_batch", None)):
            raise ValueError(
                "one-pass Arrow readers must be materialized explicitly before comparison"
            )
    settings, frames = prepare(
        dict(zip(_SIDES, (baseline, candidate), strict=True)),
        schema=None,
        config={"engine": {"parallel": False}, "plugins": {"entry_points": False}},
        resources=resources,
    )
    if settings.engine.execution_mode == "isolated":
        from qavf.execution.dispatch import compare_datasets_isolated

        return compare_datasets_isolated(settings, frames, spec)
    from qavf.compare.engine import compare_in_context
    from qavf.engine.context import DataContext

    with DataContext.from_settings(
        settings, frames, threads=settings.engine.native_threads, execution_workers=1
    ) as context:
        result = compare_in_context(context, spec)
        if failures := context.verify_inputs():
            raise ValueError(f"comparison source identity changed: {failures}")
        return result
