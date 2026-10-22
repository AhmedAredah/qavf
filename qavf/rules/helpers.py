"""Helpers for writing rules: dataset checks, aggregation and result shaping.

Public API for rule authors; the built-in rules use the same functions.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal

import polars as pl
from pydantic import model_validator

from qavf.aggregation import exact_sum_expr as exact_sum_expr
from qavf.aggregation import narrow_integer, numeric_expr
from qavf.audit_codec import evidence_json_size
from qavf.engine.context import DataContext, stable_order
from qavf.engine.registry import Finding, RuleParams
from qavf.keys import KeyTypeError as KeyTypeError
from qavf.keys import align_keys as align_keys
from qavf.numeric import (
    checked_float,
    stable_mean_scalar,
    stable_weighted_mean,
)
from qavf.numeric import exact_number as exact_number
from qavf.numeric import numeric_close as numeric_close
from qavf.ordering import OrderBy as OrderBy
from qavf.ordering import ordered_candidates
from qavf.ordering import ordering_columns as ordering_columns


class MetricSpec(RuleParams):
    """A numeric column of a dataset, optionally broken down by group columns."""

    dataset: str
    """Dataset name."""
    value: str
    """Numeric column."""
    group_by: tuple[str, ...] = ()
    """Columns to break the metric down by."""

    @property
    def label(self) -> str:
        """Short ``dataset.column`` label for messages."""
        return f"{self.dataset}.{self.value}"


Aggregate = Literal["sum", "mean", "count", "min", "max", "first", "last", "weighted_mean", "none"]
"""How values are combined per group (and period): see :func:`aggregate`."""


def check_aggregation(how: str, weight: str | None, order_by: OrderBy) -> None:
    """Reject aggregation settings that lack (or do not use) their extra column."""
    if (how == "weighted_mean") != (weight is not None):
        raise ValueError("'weight' is required for, and only used by, aggregate: weighted_mean")
    if (how in ("first", "last")) != (order_by is not None):
        raise ValueError("'order_by' is required for, and only used by, aggregate: first/last")
    ordering_columns(order_by)


class AggregatedMetric(MetricSpec):
    """A metric whose values are combined per group by an explicit :data:`Aggregate`.

    The default ``sum`` suits additive quantities (counts, masses, totals). Rates,
    concentrations, shares and averages are not additive: use ``mean`` (or
    ``weighted_mean`` with a ``weight`` column), ``min``/``max``, ``first``/``last``
    by ``order_by``, ``count``, or ``none`` when the data already has one row per
    group and period.
    """

    aggregate: Aggregate = "sum"
    weight: str | None = None
    """Weight column for ``weighted_mean``."""
    order_by: OrderBy = None
    """Column or lexicographic column list for ``first``/``last``. Conflicting
    values tied at the selected key are rejected; add an explicit tie-break column."""

    @model_validator(mode="after")
    def _aggregation(self) -> AggregatedMetric:
        check_aggregation(self.aggregate, self.weight, self.order_by)
        return self

    @property
    def label(self) -> str:
        """``dataset.column``, with the aggregation when it is not a sum."""
        base = f"{self.dataset}.{self.value}"
        return base if self.aggregate == "sum" else f"{base} ({self.aggregate})"

    @property
    def aggregation_columns(self) -> tuple[str, ...]:
        """Extra columns the aggregation reads (weight, ordering)."""
        weight = (self.weight,) if self.weight is not None else ()
        return (*weight, *ordering_columns(self.order_by))


def missing_columns(frame: pl.DataFrame, columns: Iterable[str]) -> list[str]:
    """Names from ``columns`` that ``frame`` lacks, in the order given."""
    present = set(frame.columns)
    return [column for column in columns if column not in present]


def usable(ctx: DataContext, dataset: str, columns: Iterable[str]) -> str | None:
    """Explain why ``dataset`` cannot serve ``columns``; ``None`` when it can."""
    if not ctx.has(dataset):
        return f"dataset {dataset!r} not available ({ctx.unavailable_reason(dataset)})"
    available = ctx.schema(dataset)
    absent = [column for column in columns if column not in available]
    if absent:
        return f"dataset {dataset!r} lacks column(s) {', '.join(absent)}"
    return None


def total(
    frame: pl.DataFrame, keys: Sequence[str], value: str, alias: str | None = None
) -> pl.DataFrame:
    """Sum ``value`` within ``keys`` (a one-row frame when ``keys`` is empty).

    A group with no usable value (every input null or NaN) sums to null, not to
    zero: "no measurement" must stay distinguishable from a measured zero.
    Groups with some usable values sum those values.
    """
    return aggregate(frame, keys, value, alias)


def aggregate(
    frame: pl.DataFrame,
    keys: Sequence[str],
    value: str,
    alias: str | None = None,
    how: Aggregate = "sum",
    weight: str | None = None,
    order_by: OrderBy = None,
) -> pl.DataFrame:
    """Combine ``value`` within ``keys`` by ``how`` (one row when ``keys`` is empty).

    Null and NaN are missing and never take part. A group with no usable value is
    null ("no measurement"), never zero, except for ``count``, which counts the
    usable values. ``weighted_mean`` uses rows whose value and weight are both
    usable; a group with a negative weight or no positive total weight is null.
    ``first``/``last`` take the usable value with the lowest/highest lexicographic
    ``order_by``. Conflicting values tied at that key are rejected; equal-valued
    ties are safe. Null/NaN ordering keys are missing. Add secondary ordering
    columns to make an otherwise ambiguous selection explicit.
    ``none`` takes the group's value as is: callers must ensure (see
    :func:`duplicated_groups`) that each group has one row.
    """
    check_aggregation(how, weight, order_by)
    dtype = frame.schema[value]
    if not dtype.is_numeric() and dtype != pl.Null:
        raise ValueError(f"aggregation requires numeric values, found {dtype}")
    x = numeric_expr(value, dtype)
    tie_count = None
    if how in ("mean", "weighted_mean"):
        # Validate exact inputs before a floating reduction can erase distinctions.
        for column in [value, *([str(weight)] if weight is not None else [])]:
            data = frame[column].drop_nulls()
            if data.dtype.is_integer() or data.dtype.is_decimal():
                for item in data.unique():
                    checked_float(item)
    match how:
        case "sum":
            expr = exact_sum_expr(frame[value])
        case "mean":
            expr = x.cast(pl.Float64).map_batches(
                stable_mean_scalar, return_dtype=pl.Float64, returns_scalar=True
            )
        case "count":
            expr = x.count().cast(pl.UInt64)
        case "min":
            expr = x.min()
        case "max":
            expr = x.max()
        case "first" | "last":
            candidates = ordered_candidates(
                x, ordering_columns(order_by), frame.schema, last=how == "last"
            )
            expr = candidates.first()
            tie_count = candidates.n_unique()
        case "weighted_mean":
            expr = pl.map_groups(
                [x, pl.col(str(weight))],
                lambda series: stable_weighted_mean(series[0], series[1]),
                return_dtype=pl.Float64,
                returns_scalar=True,
            )
        case "none":
            expr = x.first()
        case _:
            raise ValueError(f"unknown aggregation {how!r}")
    output = alias or value
    named = expr.alias(output)
    tie_column = fresh_internal("ordered_tie_values", [*frame.columns, output])
    expressions = [named]
    if tie_count is not None:
        expressions.append(tie_count.alias(tie_column))
    try:
        result = (
            frame.select(expressions)
            if not keys
            else frame.group_by(list(keys), maintain_order=True).agg(expressions)
        )
    except pl.exceptions.PanicException as exc:
        # Older supported Polars wraps Python UDF validation failures in a
        # native panic wrapper. Preserve the public error/assessment boundary.
        raise ValueError(f"numeric aggregation could not be assessed: {exc}") from exc
    if tie_count is not None:
        if (result[tie_column] > 1).any():
            raise ValueError(
                "ordered aggregation has conflicting values tied at the selected order_by key; "
                "provide additional order_by columns to break the tie explicitly"
            )
        result = result.drop(tie_column)
    if how in ("mean", "weighted_mean") and not dtype.is_float():
        extrema = frame[value].drop_nulls()
        if not extrema.is_empty() and any(
            isinstance(v, int | float | Decimal) and exact_number(v).copy_abs() >= 2**52
            for v in (extrema.min(), extrema.max())
        ):
            groups = frame.partition_by(list(keys), maintain_order=True) if keys else [frame]
            for group, computed in zip(groups, result[output], strict=True):
                if computed is None:
                    continue
                validation_weights = group[str(weight)] if weight is not None else [1] * len(group)
                pairs = [
                    (Fraction(v), Fraction(w))
                    for v, w in zip(group[value], validation_weights, strict=True)
                    if v is not None
                    and w is not None
                    and not (isinstance(v, float) and math.isnan(v))
                    and not (isinstance(w, float) and math.isnan(w))
                ]
                denominator = sum((w for _, w in pairs), Fraction())
                if (
                    denominator
                    and Fraction(computed)
                    != sum((v * w for v, w in pairs), Fraction()) / denominator
                ):
                    raise ValueError("aggregated mean cannot preserve exact large-value precision")
    return narrow_integer(result, output) if how == "sum" and dtype.is_integer() else result


def duplicated_groups(frame: pl.DataFrame, keys: Sequence[str]) -> int:
    """How many ``keys`` combinations occur on more than one row."""
    if not keys:
        return int(frame.height > 1)
    return frame.group_by(list(keys)).len().filter(pl.col("len") > 1).height


def tag(frame: pl.DataFrame, **constants: str) -> pl.DataFrame:
    """Prepend literal string columns (e.g. ``dataset="inventory"``) to ``frame``.

    A data column with the same name is kept; the tag gets a ``qavf_`` prefix.
    """
    taken = set(frame.columns)
    literals = [pl.lit(value).alias(_unclaimed(name, taken)) for name, value in constants.items()]
    return frame.select(*literals, pl.all())


def stack(frames: Sequence[pl.DataFrame]) -> pl.DataFrame | None:
    """Vertically combine frames with differing columns; ``None`` when all are empty."""
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return None
    return pl.concat(non_empty, how="diagonal_relaxed")


#: Relative slack when a computed value is compared with a configured limit. A value
#: exceeds its limit only by more than binary rounding of the computation can explain,
#: so a change of exactly 10% passes ``max_change_rate: 0.1`` although
#: ``110 / 100 - 1`` is ``0.10000000000000009`` in floating point.
BOUNDARY_RTOL = 1e-9


def above(value: pl.Expr, limit: float) -> pl.Expr:
    """``value > limit``, beyond rounding noise (:data:`BOUNDARY_RTOL`)."""
    return value > limit + abs(limit) * BOUNDARY_RTOL


def below(value: pl.Expr, limit: float) -> pl.Expr:
    """``value < limit``, beyond rounding noise (:data:`BOUNDARY_RTOL`)."""
    return value < limit - abs(limit) * BOUNDARY_RTOL


def internal(name: str) -> str:
    """Preferred internal prefix; use ``fresh_internal`` beside user columns."""
    return f"__qavf_{name}"


def _unclaimed(name: str, taken: set[str]) -> str:
    while name in taken:
        name = f"qavf_{name}"
    taken.add(name)
    return name


def present(frame: pl.DataFrame, names: Mapping[str, str]) -> pl.DataFrame:
    """Rename internal columns to readable names for evidence, never onto a user column.

    ``names`` maps internal names (see :func:`internal`) to readable ones; a
    readable name the frame already uses gets a ``qavf_`` prefix instead.
    """
    taken = set(frame.columns) - set(names)
    renames = {
        source: _unclaimed(target, taken)
        for source, target in names.items()
        if source in frame.columns
    }
    return frame.rename(renames)


def annotate(row: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    """An evidence record: ``fields`` first, then the data ``row``.

    The data keeps its own column names; an annotation whose name the row
    already uses is prefixed with ``qavf_`` rather than overwriting the data.
    """
    taken = set(row)
    record = {_unclaimed(name, taken): value for name, value in fields.items()}
    record.update(row)
    return record


def plural(count: int, noun: str) -> str:
    """``"1 row"`` / ``"3 rows"`` (``series`` is its own plural)."""
    return f"{count:,} {noun}{'' if count == 1 or noun.endswith('series') else 's'}"


def shared[T](
    ctx: DataContext,
    key: Any,
    blocked: list[Finding],
    compute: Callable[[list[Finding]], T],
) -> T:
    """Compute ``compute(findings)`` once per ``key`` in a run and share the result.

    For work several rules repeat (aggregating the same series, building the same
    entity metric). Findings ``compute`` records (e.g. "not assessed") are kept with
    the result and appended to every caller's ``blocked``, so each rule still
    reports them. ``key`` must contain everything the result depends on.
    """

    def run() -> tuple[T, tuple[Finding, ...]]:
        found: list[Finding] = []
        value = compute(found)
        return value, tuple(found)

    value, found = ctx.memo(("qavf.shared", key), run)
    blocked.extend(found)
    return value


def deterministic(frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` in a stable order: sorted by every column that can be sorted.

    For results whose row order is not guaranteed (SQL with ``GROUP BY`` or joins
    runs in parallel), so the evidence sample a report keeps is the same on every
    run of the same data. Nested columns (lists, structs) are not sort keys.
    """
    return stable_order(frame)


def outcome(
    failures: Sequence[Finding],
    passed: int | Sequence[str],
    ok_message: str,
    nothing_message: str,
    *,
    summary_metrics: Mapping[str, Any] | None = None,
    **ok_kwargs: Any,
) -> list[Finding]:
    """Shape the findings of a rule that grades several checks separately.

    Returns the failures, plus one passing summary finding when anything passed,
    or a single skip when nothing was checked at all. Pass the labels of the
    checks that passed (rather than a count) to record each of them in the
    summary's ``passed_checks`` metric, so the report accounts for every check.
    ``summary_metrics`` describe the entire evaluation, including exempt rows.
    They survive an all-failure outcome on its first finding; when some checks
    pass, they belong to the existing passing summary. No new verdict is added.
    """
    findings = list(failures)
    if passed:
        if not isinstance(passed, int):
            metrics = {**ok_kwargs.pop("metrics", {}), "passed_checks": list(passed)}
            ok_kwargs = {**ok_kwargs, "metrics": metrics}
        findings.append(Finding.ok(ok_message, **ok_kwargs))
    if not findings:
        findings.append(Finding.skip(nothing_message))
    if summary_metrics:
        index = -1 if passed else 0
        findings[index] = replace(
            findings[index], metrics={**findings[index].metrics, **summary_metrics}
        )
    return findings


def fresh_internal(name: str, columns: Iterable[str]) -> str:
    """Allocate a name without shadowing even case-insensitive SQL identifiers."""
    taken = {column.casefold() for column in columns}
    candidate = internal(name)
    while candidate.casefold() in taken:
        candidate = f"qavf_{candidate}"
    return candidate


def missing_expr(frame: pl.DataFrame, column: str, spec: Any = None) -> pl.Expr:
    """Null or NaN, without asking unsupported dtypes to evaluate is_nan."""
    expr = pl.col(column).is_null()
    if frame.schema[column].is_float():
        expr = expr | pl.col(column).is_nan()
    if spec is not None:
        from qavf.rules._predicates import is_text, members, membership

        if spec.missing_values:
            values, foreign = members(spec.missing_values, frame.schema[column])
            if foreign:
                raise ValueError(f"missing markers for {column!r} have incompatible types")
            if values:
                expr = expr | membership(values, frame.schema[column])(column).fill_null(False)
        if spec.blank_as_missing and is_text(frame.schema[column]):
            expr = expr | (pl.col(column).cast(pl.String).str.strip_chars() == "").fill_null(False)
    return expr


def normalize_missing(
    ctx: DataContext,
    dataset: str,
    frame: pl.DataFrame,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Apply the dataset's declared missing-value contract to a projected frame."""
    dataset_spec = ctx.specs.get(dataset)
    schema = dataset_spec.schema_spec if dataset_spec is not None else None
    specs = schema.columns if schema is not None else {}
    return frame.with_columns(
        pl.when(missing_expr(frame, column, specs.get(column)))
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column in (frame.columns if columns is None else columns)
    )


@dataclass(slots=True)
class LargestEvidence:
    """Retain the ``limit`` records of largest magnitude; ties keep the earliest ordinal.

    Callers test :meth:`admits` before building a record, so evidence for values
    that cannot enter the bounded heap is never constructed.
    """

    limit: int
    _heap: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)

    def admits(self, magnitude: float, ordinal: int) -> bool:
        if not self.limit:
            return False
        return len(self._heap) < self.limit or (magnitude, -ordinal) > self._heap[0][:2]

    def add(self, magnitude: float, ordinal: int, record: dict[str, Any]) -> None:
        if not self.admits(magnitude, ordinal):
            return
        item = (magnitude, -ordinal, record)
        if len(self._heap) < self.limit:
            heapq.heappush(self._heap, item)
        else:
            heapq.heapreplace(self._heap, item)

    @property
    def records(self) -> list[dict[str, Any]]:
        """Retained records, largest magnitude first, then in ordinal order."""
        return [row for _, _, row in sorted(self._heap, key=lambda item: item[:2], reverse=True)]


@dataclass(slots=True)
class BoundedEvidence:
    """Count all findings while retaining at most ``limit`` Python records.

    With ``byte_limit``, records are also admitted only while their encoded
    audit size fits; ``truncated`` then reports that counted records were omitted.
    """

    limit: int
    records: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0
    byte_limit: int | None = None
    bytes_used: int = 0

    @property
    def truncated(self) -> bool:
        """Whether counted records were left out of :attr:`records`."""
        return self.count > len(self.records)

    def _admit(self, record: Mapping[str, Any]) -> bool:
        if len(self.records) >= self.limit:
            return False
        if self.byte_limit is None:
            return True
        size = evidence_json_size(dict(record))
        if self.bytes_used + size > self.byte_limit:
            self.limit = len(self.records)  # later, smaller records stay out: order is kept
            return False
        self.bytes_used += size
        return True

    def append(self, record: Mapping[str, Any]) -> None:
        self.count += 1
        if self._admit(record):
            self.records.append(dict(record))

    def extend(self, records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            self.append(record)

    def add_frame(self, frame: pl.DataFrame) -> None:
        self.count += frame.height
        remaining = max(0, self.limit - len(self.records))
        if self.byte_limit is None:
            self.records.extend(frame.head(remaining).to_dicts())
            return
        for record in frame.head(remaining).iter_rows(named=True):
            if not self._admit(record):
                break
            self.records.append(record)
