"""One-pass profile state bounded by column count and retained categories."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, cast

import polars as pl

from .values import _datetime_bounds, _dtype_name, _instant, _plain, _widen_dates, _widen_numbers


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Draft contract and exact dimensions of the completely observed input."""

    schema: dict[str, Any]
    rows: int
    columns: int


@dataclass(slots=True)
class _CategoryBudget:
    maximum: int
    used: int = 0

    def reserve(self, amount: int) -> bool:
        if self.used + amount > self.maximum:
            return False
        self.used += amount
        return True


@dataclass(slots=True)
class _Column:
    dtype: pl.DataType
    category_limit: int
    category_budget: _CategoryBudget
    category_value_bytes: int
    category_bytes: int = 0
    category_reason: str | None = None
    count: int = 0
    missing: int = 0
    positive_infinity: bool = False
    negative_infinity: bool = False
    low: Any = None
    high: Any = None
    min_length: int | None = None
    max_length: int | None = None
    rounded_datetime: bool = False
    categories: set[str] | None = field(default_factory=set)

    def update(self, series: pl.Series, *, bounds: bool) -> None:
        values = series.drop_nulls()
        if series.dtype.is_float():
            values = values.drop_nans()
        self.count += len(values)
        self.missing += len(series) - len(values)
        kind = _dtype_name(self.dtype)
        if series.dtype.is_float():
            self.positive_infinity |= bool((values == float("inf")).any())
            self.negative_infinity |= bool((values == -float("inf")).any())
            values = values.filter(values.is_finite())
        if values.is_empty():
            return
        if kind == "str":
            text = values.cast(pl.String)
            lengths = text.str.len_chars()
            low_length, high_length = int(lengths.min()), int(lengths.max())  # type: ignore[arg-type]
            self.min_length = (
                low_length if self.min_length is None else min(self.min_length, low_length)
            )
            self.max_length = (
                high_length if self.max_length is None else max(self.max_length, high_length)
            )
            if self.categories is not None:
                self._retain_categories(text)
        elif bounds and kind in ("int", "float", "decimal", "date", "datetime"):
            low: Any
            high: Any
            if kind == "datetime":
                low, high, rounded = _datetime_bounds(values)
                self.rounded_datetime |= rounded
            else:
                low, high = values.min(), values.max()
            self.low = low if self.low is None else min(self.low, low, key=_instant)
            self.high = high if self.high is None else max(self.high, high, key=_instant)

    def _discard_categories(self, reason: str | None = None) -> None:
        self.categories = None
        self.category_budget.used -= self.category_bytes
        self.category_bytes = 0
        self.category_reason = reason

    def _retain_categories(self, text: pl.Series) -> None:
        assert self.categories is not None
        if self.category_limit == 0:
            self._discard_categories()
            return
        if self.category_value_bytes == 0:
            self._discard_categories("category value byte limit")
            return
        # Reject oversized payloads while still columnar; do not copy them into
        # Python merely to discover that the category budget cannot retain them.
        if cast(int, text.str.len_bytes().max() or 0) > self.category_value_bytes:
            self._discard_categories("category value byte limit")
            return
        candidates = text.unique().head(self.category_limit + 1)
        if len(candidates) > self.category_limit:
            self._discard_categories()
            return
        for value in candidates:
            if value in self.categories:
                continue
            if len(self.categories) == self.category_limit:
                self._discard_categories()
                return
            # Include Python text and conservative hash-set entry overhead;
            # encoded bytes also matter for the resulting draft schema.
            size = max(sys.getsizeof(value), len(value.encode("utf-8"))) + 72
            if not self.category_budget.reserve(size):
                self._discard_categories("profile category byte budget")
                return
            self.category_bytes += size
            self.categories.add(value)

    def contract(self, margin: float, *, bounds: bool) -> dict[str, Any]:
        result: dict[str, Any] = {}
        kind = _dtype_name(self.dtype)
        if kind:
            result["dtype"] = kind
        if self.missing:
            result["nullable"] = True
        if self.positive_infinity or self.negative_infinity:
            result["allow_infinite"] = True
        if self.low is not None and bounds:
            low, high = self.low, self.high
            if isinstance(low, int | float | Decimal) and isinstance(high, int | float | Decimal):
                if margin:
                    low, high = _widen_numbers(low, high, margin)
            elif isinstance(low, date) and isinstance(high, date) and margin:
                low, high = _widen_dates(low, high, margin)
            # A bound on an observed infinite side would reject the original data.
            if not self.negative_infinity:
                result["ge"] = _plain(low)
            if not self.positive_infinity:
                result["le"] = _plain(high)
            if self.rounded_datetime:
                result["description"] = (
                    "Observed nanosecond datetime bounds were rounded outwards "
                    "to configuration datetime precision (one microsecond)."
                )
        if kind == "str" and self.count:
            if self.categories is not None and len(self.categories) < max(2, self.count // 2):
                result["isin"] = sorted(self.categories)
            else:
                result["min_length"], result["max_length"] = self.min_length, self.max_length
        if self.category_reason is not None:
            result["description"] = (
                "Allowed-value enumeration omitted after exceeding the "
                f"{self.category_reason}; exact string lengths and missingness are retained."
            )
        return result


class ProfileAccumulator:
    """Retain extrema/counts plus at most ``max_categories`` labels per column."""

    def __init__(
        self,
        schema: dict[str, pl.DataType],
        *,
        max_categories: int,
        bounds: bool,
        max_category_bytes: int = 1 << 20,
        max_category_value_bytes: int = 64 << 10,
    ) -> None:
        self.category_budget = _CategoryBudget(max_category_bytes)
        self.columns = {
            name: _Column(
                dtype,
                max_categories,
                self.category_budget,
                min(max_category_value_bytes, max_category_bytes),
            )
            for name, dtype in schema.items()
        }
        self.bounds = bounds
        self.rows = 0

    def update(self, frame: pl.DataFrame) -> None:
        self.rows += frame.height
        for name, state in self.columns.items():
            state.update(frame[name], bounds=self.bounds)

    def finish(self, margin: float) -> ProfileResult:
        columns = {
            name: state.contract(margin, bounds=self.bounds) for name, state in self.columns.items()
        }
        return ProfileResult(
            {"columns": columns, "primary_key": [], "min_rows": 1}, self.rows, len(columns)
        )
