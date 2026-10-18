"""Immutable comparison decisions made from schemas before reading table values."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

import polars as pl

from qavf.keys import KeyTypeError
from qavf.regression._columns import column_names
from qavf.regression._numeric import _finite
from qavf.regression._schema import _structure_changed, _type_changed
from qavf.regression.models import SchemaPolicy


@dataclass(frozen=True, slots=True)
class ComparisonPlan:
    """Validated arguments, full schema snapshots, and the required projection.

    ``shared`` follows baseline column order and includes all keys. Incompatible
    value types need no value reads: their changes remain in ``dtypes_changed``.
    Nested structural changes are retained even when the tables have no rows.
    Key uniqueness, NaN detection, and lossless casts require values and are
    deliberately left to execution.
    """

    keys: tuple[str, ...]
    values: tuple[str, ...]
    shared: tuple[str, ...]
    baseline_schema: Mapping[str, pl.DataType]
    candidate_schema: Mapping[str, pl.DataType]
    columns_added: tuple[str, ...]
    columns_removed: tuple[str, ...]
    dtypes_changed: tuple[str, ...]
    name: str
    rtol: float
    atol: float
    max_differences: int
    max_evidence_bytes: int
    column_tolerances: Mapping[str, tuple[float, float]]

    def tolerance(self, column: str | None) -> tuple[float, float]:
        """Return a column override or the comparison-wide tolerance."""
        if column is None:
            return self.rtol, self.atol
        return self.column_tolerances.get(column, (self.rtol, self.atol))


def _check_tolerance(tolerance: float, label: str, name: str) -> None:
    if isinstance(tolerance, bool) or not isinstance(tolerance, int | float | Decimal):
        raise ValueError(f"{name}: {label} must be a number, got {tolerance!r}")
    if not (_finite(tolerance) and tolerance >= 0):
        raise ValueError(f"{name}: {label} must be a finite number >= 0, got {tolerance!r}")


def _column_tolerances(
    supplied: Mapping[str, tuple[float, float]] | None, name: str
) -> dict[str, tuple[float, float]]:
    """Validate and detach overrides, including any caller-owned pair lists."""
    if supplied is None:
        return {}
    if not isinstance(supplied, Mapping):
        raise ValueError(f"{name}: column_tolerances must be a mapping")
    result = {}
    for column, pair in supplied.items():
        if not isinstance(column, str) or not column:
            raise ValueError(f"{name}: column tolerances must name nonempty string columns")
        if isinstance(pair, str | bytes) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"{name}: tolerance for {column!r} must be an (rtol, atol) pair")
        column_rtol, column_atol = pair
        _check_tolerance(column_rtol, f"rtol of {column!r}", name)
        _check_tolerance(column_atol, f"atol of {column!r}", name)
        result[column] = (column_rtol, column_atol)
    return result


def _check_columns(
    keys: tuple[str, ...],
    ignored: set[str],
    columns: set[str],
    tolerated: set[str],
    name: str,
) -> None:
    repeated = sorted(key for key, count in Counter(keys).items() if count > 1)
    if repeated:
        raise ValueError(f"{name}: keys repeat column(s) {', '.join(repeated)}")
    ignored_keys = sorted(set(keys) & ignored)
    if ignored_keys:
        raise ValueError(f"{name}: key column(s) {', '.join(ignored_keys)} cannot also be ignored")
    if unknown := sorted(ignored - columns):
        raise ValueError(
            f"{name}: ignore_columns names column(s) in neither table: {', '.join(unknown)}"
        )
    if unknown := sorted(tolerated - columns):
        raise ValueError(
            f"{name}: column tolerances name column(s) in neither table: {', '.join(unknown)}"
        )
    if clash := sorted(tolerated & (set(keys) | ignored)):
        raise ValueError(
            f"{name}: column tolerances cannot apply to key or ignored column(s): "
            f"{', '.join(clash)}"
        )


def check_evidence_limits(max_differences: int, max_evidence_bytes: int, name: str) -> None:
    """Validate explicit evidence arguments before resource caps can reduce them."""
    if isinstance(max_differences, bool) or not isinstance(max_differences, int):
        raise ValueError(f"{name}: max_differences must be an integer")
    if max_differences < 0:
        raise ValueError(f"{name}: max_differences must be >= 0, got {max_differences}")
    if (
        isinstance(max_evidence_bytes, bool)
        or not isinstance(max_evidence_bytes, int)
        or max_evidence_bytes < 0
    ):
        raise ValueError(f"{name}: max_evidence_bytes must be an integer >= 0")


def comparison_plan(
    baseline_schema: Mapping[str, pl.DataType],
    candidate_schema: Mapping[str, pl.DataType],
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
) -> ComparisonPlan:
    """Plan comparison from complete original schemas without materializing data.

    Numeric key widths may differ; execution must verify any proposed key cast
    against the original values. Incompatible original key kinds are rejected
    here, regardless of the selected value-schema policy.
    """
    if schema not in ("compatible", "strict", "ignore"):
        raise ValueError(f"{name}: unknown schema policy {schema!r}")
    selected_keys = tuple(column_names(keys))
    if not selected_keys:
        raise ValueError(f"{name}: at least one key column is required")
    ignored = set(column_names(ignore_columns))
    _check_tolerance(rtol, "rtol", name)
    _check_tolerance(atol, "atol", name)
    per_column = _column_tolerances(column_tolerances, name)
    check_evidence_limits(max_differences, max_evidence_bytes, name)
    _check_columns(
        selected_keys,
        ignored,
        set(baseline_schema) | set(candidate_schema),
        set(per_column),
        name,
    )
    # Copy before deriving any projection: full original schemas remain available
    # for type checks even when execution excludes an incompatible value column.
    baseline, candidate = dict(baseline_schema), dict(candidate_schema)
    for side, columns in (("baseline", baseline), ("candidate", candidate)):
        if absent := [key for key in selected_keys if key not in columns]:
            raise ValueError(f"{name}: {side} lacks key column(s) {', '.join(absent)}")
    for key in selected_keys:
        if _type_changed(baseline[key], candidate[key], "compatible"):
            raise KeyTypeError(f"{name}: key {key!r} has incompatible types")

    base_columns = tuple(column for column in baseline if column not in ignored)
    cand_columns = tuple(column for column in candidate if column not in ignored)
    base_names, cand_names = set(base_columns), set(cand_columns)
    shared = tuple(column for column in base_columns if column in cand_names)
    key_names = set(selected_keys)
    changed_types = {
        column
        for column in shared
        if (column not in key_names or schema == "strict")
        and _type_changed(baseline[column], candidate[column], schema)
    }
    dtypes_changed = tuple(
        column
        for column in shared
        if (column not in key_names or schema == "strict")
        and (
            column in changed_types
            or (schema != "ignore" and _structure_changed(baseline[column], candidate[column]))
        )
    )
    shared = tuple(
        column for column in shared if column in key_names or column not in changed_types
    )
    return ComparisonPlan(
        keys=selected_keys,
        values=tuple(column for column in shared if column not in key_names),
        shared=shared,
        baseline_schema=MappingProxyType(baseline),
        candidate_schema=MappingProxyType(candidate),
        columns_added=tuple(column for column in cand_columns if column not in base_names),
        columns_removed=tuple(column for column in base_columns if column not in cand_names),
        dtypes_changed=dtypes_changed,
        name=name,
        rtol=rtol,
        atol=atol,
        max_differences=max_differences,
        max_evidence_bytes=max_evidence_bytes,
        column_tolerances=MappingProxyType(per_column),
    )
