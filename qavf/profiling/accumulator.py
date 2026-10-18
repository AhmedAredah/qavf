"""One-pass profile state bounded by column count and retained categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

import polars as pl

from .values import _datetime_bounds, _dtype_name, _plain, _widen_dates, _widen_numbers


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Draft contract and exact dimensions of the completely observed input."""

    schema: dict[str, Any]
    rows: int
    columns: int


@dataclass(slots=True)
class _Column:
    dtype: pl.DataType
    category_limit: int
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
                # Never retain a high-cardinality column's complete unique set.
                self.categories.update(text.unique().head(self.category_limit + 1).to_list())
                if len(self.categories) > self.category_limit:
                    self.categories = None
        elif bounds and kind in ("int", "float", "decimal", "date", "datetime"):
            low: Any
            high: Any
            if kind == "datetime":
                low, high, rounded = _datetime_bounds(values)
                self.rounded_datetime |= rounded
            else:
                low, high = values.min(), values.max()
            self.low = low if self.low is None else min(self.low, low)
            self.high = high if self.high is None else max(self.high, high)

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
        return result


class ProfileAccumulator:
    """Retain extrema/counts plus at most ``max_categories`` labels per column."""

    def __init__(
        self, schema: dict[str, pl.DataType], *, max_categories: int, bounds: bool
    ) -> None:
        self.columns = {name: _Column(dtype, max_categories) for name, dtype in schema.items()}
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
