"""Regression comparison of generated outputs against stored baselines.

Tables are compared *by key*, not by row position, so a reordered output is not
a regression. Numeric values are compared with relative and absolute
tolerances, so floating-point noise from a library upgrade is not one either.
Integers and decimals are compared exactly (with rational arithmetic) against
the tolerance's decimal spelling; floats with ``math.isclose``. Temporal values
are always exact.
Rows are aligned by key, and bounded columnar batches compute scalar counts
and evidence. Native file comparisons keep the join inside the SQL engine.
A short-circuit traversal compares nested residual cells;
scalar changes never build a Python representation of the full input tables.

Typical use in a pytest regression test::

    from qavf.regression import assert_outputs_match

    def test_inventory_matches_baseline(tmp_path):
        run_my_utility(output=tmp_path / "inventory.csv")
        assert_outputs_match(
            "tests/baselines/inventory.csv",
            tmp_path / "inventory.csv",
            keys=["countyID", "yearID", "pollutantID"],
            rtol=1e-6,
        )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from qavf.inputs.resources import ResourceLimits
from qavf.io import write_table
from qavf.keys import KeyTypeError, align_keys
from qavf.regression._columns import column_names
from qavf.regression._comparison import _compare_aligned
from qavf.regression._plan import ComparisonPlan, check_evidence_limits, comparison_plan
from qavf.regression._schema import _encode_temporal
from qavf.regression.models import ComparisonExecution, ComparisonResult, SchemaPolicy
from qavf.sources.frames import as_polars, supports_frame


def _check_no_nan_keys(frame: pl.DataFrame, keys: Sequence[str], side: str, name: str) -> None:
    """NaN is not equal to itself, so it cannot identify a row (null keys match nulls)."""
    for key in keys:
        if frame[key].dtype.is_float() and bool(frame[key].is_nan().any()):
            raise ValueError(
                f"{name}: {side} key column {key!r} contains NaN, which cannot identify a "
                f"row; replace it (or use null) before comparing"
            )


def _check_unique(frame: pl.DataFrame, keys: Sequence[str], side: str, name: str) -> None:
    duplicates = frame.select(keys).is_duplicated().sum()
    if duplicates:
        raise ValueError(
            f"{name}: {side} has {duplicates} row(s) with duplicate key ({', '.join(keys)}); "
            f"keys must identify rows uniquely"
        )


def compare_frames(
    baseline: Any,
    candidate: Any,
    keys: Sequence[str],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    ignore_columns: Sequence[str] = (),
    max_differences: int = 1000,
    max_evidence_bytes: int = 1 << 20,
    name: str = "output",
    column_tolerances: Mapping[str, tuple[float, float]] | None = None,
    schema: SchemaPolicy = "compatible",
    column_map: Mapping[str, str] | None = None,
) -> ComparisonResult:
    """Compare two tables by key with numeric tolerance.

    Keys are matched exactly: key columns of different types are aligned only
    when that is lossless, never with a tolerance. Values are compared with
    the tolerance, using exact arithmetic for integers. Nested values (lists,
    structs) are compared element by element; any change inside a cell counts
    as one changed value for that cell.

    Args:
        baseline: Accepted reference output (Polars, pandas or Arrow table).
        candidate: Newly generated output in any supported table representation.
            LazyFrames are explicitly collected; this API is an in-memory comparison.
        keys: Columns that identify a row uniquely in both tables.
        rtol: Relative tolerance for numeric values (finite, >= 0).
        atol: Absolute tolerance for numeric values (finite, >= 0).
        ignore_columns: Columns excluded from the comparison (timestamps, run ids).
            Each must exist in at least one table and must not be a key.
        max_differences: Cap on retained differences; zero disables evidence.
            Counts are always complete.
        max_evidence_bytes: Encoded byte cap for the differences list; zero disables
            evidence. Large nested cells are previewed, with values at the changed path.
        name: Label used in messages.
        column_tolerances: ``column -> (rtol, atol)`` for columns that need their own
            tolerance (``(0, 0)`` compares a column exactly); others use ``rtol``/``atol``.
        schema: How column types are compared (:data:`SchemaPolicy`).
        column_map: ``baseline name -> candidate name`` for renamed columns. The
            baseline is compared under candidate names; ``keys``, ``ignore_columns``
            and ``column_tolerances`` use candidate names. A mapped column the
            baseline lacks is an error, never an empty comparison.

    Raises:
        ValueError: for invalid arguments, if a key column is missing, keys are not
            unique, or key columns cannot be compared exactly
            (:class:`~qavf.keys.KeyTypeError`).
    """
    baseline, candidate = _frame(baseline), _frame(candidate)
    plan = comparison_plan(
        baseline.schema,
        candidate.schema,
        keys,
        rtol=rtol,
        atol=atol,
        ignore_columns=ignore_columns,
        max_differences=max_differences,
        max_evidence_bytes=max_evidence_bytes,
        name=name,
        column_tolerances=column_tolerances,
        schema=schema,
        column_map=column_map,
    )
    return _compare_projected(plan, baseline.rename(dict(plan.column_map)), candidate)


def _compare_projected(
    plan: ComparisonPlan,
    baseline: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    execution: ComparisonExecution = "in_memory",
    fallback_reason: str | None = None,
) -> ComparisonResult:
    """Compare owned projections against the complete original schema contract."""
    keys = list(plan.keys)
    _check_no_nan_keys(baseline, keys, "baseline", plan.name)
    _check_no_nan_keys(candidate, keys, "candidate", plan.name)
    _check_unique(baseline, keys, "baseline", plan.name)
    _check_unique(candidate, keys, "candidate", plan.name)
    base = _encode_temporal(baseline.select(plan.shared), keys)
    cand = _encode_temporal(candidate.select(plan.shared), keys)
    try:
        base, cand = align_keys(base, cand, keys)
    except KeyTypeError as exc:
        raise KeyTypeError(f"{plan.name}: {exc}") from exc
    return _compare_aligned(plan, base, cand, execution=execution, fallback_reason=fallback_reason)


def _frame(source: Any) -> pl.DataFrame:
    if callable(getattr(source, "read_next_batch", None)):
        raise ValueError("one-pass Arrow readers must be materialized explicitly before comparison")
    return as_polars(source)


def compare_outputs(
    baseline: Any,
    candidate: Any,
    keys: Sequence[str],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    ignore_columns: Sequence[str] = (),
    max_differences: int = 1000,
    max_evidence_bytes: int = 1 << 20,
    name: str | None = None,
    column_tolerances: Mapping[str, tuple[float, float]] | None = None,
    schema: SchemaPolicy = "compatible",
    resources: ResourceLimits | None = None,
    column_map: Mapping[str, str] | None = None,
) -> ComparisonResult:
    """Compare a candidate output with its baseline; either may be a frame or a file path.

    Verified local files with native-safe types, including supported nested
    types, are compared as bounded joined batches. Other typed sources are
    snapshotted once and aligned through a disk index; only bounded result
    batches are decoded. Opaque Object columns retain an explicit projected
    fallback. File, Dataset, Query and Loader descriptors use the common input
    boundary. The result identifies the execution path and fallback reason.
    ``resources`` accepts the same :class:`~qavf.ResourceLimits` policy as validation.
    Native work uses its memory/spill limits; exact typed snapshots and index
    pages share the owned spill budget. Decoded batch sizes are checked against
    the memory target. These are planning targets, not an RSS cap.
    Evidence limits cap the explicit comparison limits. Isolated execution applies
    OS/deadline controls and a serialized-result cap to transportable file/query
    inputs. In-process worker/report settings do not change this single comparison.
    See :func:`compare_frames` for the other arguments.

    Raises:
        SourceError: if an input is missing or its source policy cannot be satisfied.
        ResourceLimitError: if an owned allocation exceeds its resource target.
    """
    if name is None:
        name = Path(candidate).name if isinstance(candidate, str | Path) else "output"
    if resources is not None and not isinstance(resources, ResourceLimits):
        raise TypeError("resources must be a ResourceLimits instance")
    check_evidence_limits(max_differences, max_evidence_bytes, name)
    overrides = resources.engine_overrides() if resources is not None else {}
    options: dict[str, Any] = {
        "rtol": rtol,
        "atol": atol,
        "ignore_columns": ignore_columns,
        "max_differences": min(
            max_differences, overrides.get("max_affected_records", max_differences)
        ),
        "max_evidence_bytes": min(
            max_evidence_bytes, overrides.get("max_evidence_bytes", max_evidence_bytes)
        ),
        "name": name,
        "column_tolerances": column_tolerances,
        "schema": schema,
        "column_map": dict(column_map) if isinstance(column_map, Mapping) else column_map,
    }
    # Evidence policy is already applied above; an empty/evidence-only policy
    # does not require copying caller-owned eager tables into a disk index.
    # Actual allocation, source, threading and isolation policies still use
    # the managed context and its execution planner.
    managed_policy = {
        key: value
        for key, value in overrides.items()
        if key not in {"max_affected_records", "max_evidence_bytes"}
        and not (key == "execution_mode" and value == "in_process")
    }
    if managed_policy or not all(
        supports_frame(value) and not isinstance(value, pl.LazyFrame)
        for value in (baseline, candidate)
    ):
        from qavf.regression._inputs import compare_inputs

        return compare_inputs(baseline, candidate, keys, resources=resources, **options)
    return compare_frames(baseline, candidate, keys, **options)


def assert_outputs_match(
    baseline: Any,
    candidate: Any,
    keys: Sequence[str],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    ignore_columns: Sequence[str] = (),
    name: str | None = None,
    show: int = 10,
    column_tolerances: Mapping[str, tuple[float, float]] | None = None,
    schema: SchemaPolicy = "compatible",
    resources: ResourceLimits | None = None,
    column_map: Mapping[str, str] | None = None,
) -> ComparisonResult:
    """Assert that ``candidate`` matches ``baseline``; designed for pytest.

    Raises:
        AssertionError: with a summary and the first ``show`` differences.
    """
    result = compare_outputs(
        baseline,
        candidate,
        keys,
        rtol=rtol,
        atol=atol,
        ignore_columns=ignore_columns,
        name=name,
        column_tolerances=column_tolerances,
        schema=schema,
        resources=resources,
        column_map=column_map,
    )
    if not result.equal:
        lines = [result.summary()]
        for difference in result.differences[:show]:
            where = ", ".join(f"{k}={v}" for k, v in difference.key.items())
            if difference.kind == "value_changed":
                lines.append(
                    f"  [{where}] {difference.column}: "
                    f"{difference.baseline!r} -> {difference.candidate!r}"
                )
            elif difference.kind == "dtype_changed":
                lines.append(
                    f"  dtype_changed: {difference.column}: "
                    f"{difference.baseline} -> {difference.candidate}"
                )
            else:
                lines.append(f"  {difference.kind}: {where or difference.column}")
        # Every kind of difference counts, column- and type-level ones included.
        hidden = result.total_differences - min(show, len(result.differences))
        if hidden > 0:
            lines.append(f"  ... and {hidden:,} more")
        raise AssertionError("\n".join(lines))
    return result


def save_baseline(frame: pl.DataFrame, path: str | Path, keys: Sequence[str] = ()) -> Path:
    """Write ``frame`` as a baseline file, sorted by ``keys`` for stable diffs in review."""
    return write_table(frame.sort(column_names(keys)) if keys else frame, path)
