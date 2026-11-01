"""Values of a declared logical type in a column that was read as text.

A reader that infers types from a sample leaves a whole column as text when one
cell is not a value of its type (``12x`` in a count, ``N/A`` in a date). INP-002
reports exactly those cells, and INP-003 checks its limits on the rest. Both use
:class:`TextConversion`, so they always agree on which cells are invalid, and
both evaluate it lazily over one observation rather than a loaded column.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from qavf.applicability import NotApplicablePolicy
from qavf.settings import ColumnSpec
from qavf.text_types import polars_stripped, polars_value
from qavf.value_policy import is_text

CONVERTIBLE = ("int", "float", "decimal", "bool", "date", "datetime")
"""Logical types whose values can be recognized in text."""


def converts_text(spec: ColumnSpec, actual: pl.DataType) -> bool:
    """Whether a column declared as ``spec`` was read as text that may hold its values."""
    return (
        (spec.dtype in CONVERTIBLE and is_text(actual))
        or (spec.dtype == "decimal" and spec.decimal_precision is not None and actual.is_decimal())
        or (
            spec.dtype == "datetime"
            and isinstance(actual, pl.Datetime)
            and (spec.timezone is not None or spec.datetime_unit != "us")
        )
    )


def described(dtype: str) -> str:
    """``an int``, ``a date``: the declared type as a noun phrase."""
    return f"{'an' if dtype == 'int' else 'a'} {dtype}"


def typed_spec(spec: ColumnSpec) -> ColumnSpec:
    """``spec`` for the converted values: text markers have already been applied."""
    typed: list[Any] = [value for value in spec.missing_values if not isinstance(value, str)]
    if spec.timezone:
        from qavf.text_policies import temporal_literal

        typed = [temporal_literal(value, spec) for value in typed]
    return spec.model_copy(update={"missing_values": typed, "blank_as_missing": False})


class DecimalText:
    """Decimal text converted exactly at one column scale, never rounded.

    A column holds one Decimal type, so its scale is the most significant
    fractional digits any value needs (trailing zeros need none). A value whose
    significant fraction exceeds that scale, or whose integer digits do not fit
    beside it within :data:`PRECISION` digits, is invalid rather than rounded.
    """

    PATTERN = r"^[+-]?(\d+\.?\d*|\.\d+)$"
    PRECISION = 38

    @classmethod
    def fraction_digits(cls, text: pl.Expr) -> pl.Expr:
        """Significant fractional digits of each decimal text (0 for none)."""
        fraction = text.str.extract(r"\.(\d*?)0*$", 1)
        return fraction.str.len_chars().fill_null(0)

    @classmethod
    def integer_digits(cls, text: pl.Expr) -> pl.Expr:
        """Integer digits after leading zeros."""
        return text.str.extract(r"^[+-]?0*(\d*)", 1).str.len_chars().fill_null(0)

    @classmethod
    def scale(cls, text: pl.Expr) -> pl.Expr:
        """The column scale, as a reduction over the decimal texts of ``text``: the
        widest significant fraction, narrowed so the widest integer part still fits."""
        decimal = text.str.contains(cls.PATTERN) & (
            cls.integer_digits(text) + cls.fraction_digits(text) <= cls.PRECISION
        )
        fraction = pl.when(decimal).then(cls.fraction_digits(text)).max().fill_null(0)
        integer = pl.when(decimal).then(cls.integer_digits(text)).max().fill_null(0)
        return pl.min_horizontal(fraction, cls.PRECISION - integer).clip(0, cls.PRECISION)

    @classmethod
    def value(cls, text: pl.Expr, scale: int, precision: int = 38) -> pl.Expr:
        exact = (
            text.str.contains(cls.PATTERN)
            & (cls.fraction_digits(text) <= scale)
            & (cls.integer_digits(text) + scale <= precision)
        )
        return pl.when(exact).then(text).cast(pl.Decimal(precision, scale), strict=False)


class TextConversion:
    """How one text column converts to its declared type, as Polars expressions.

    Surrounding spaces are ignored; text ``missing_values`` (and blank text with
    ``blank_as_missing``) are missing, never invalid. Typed markers such as -999
    apply afterwards, to the converted values (see :func:`typed_spec`).
    """

    def __init__(self, column: str, spec: ColumnSpec, physical: pl.DataType | None = None) -> None:
        if spec.dtype not in CONVERTIBLE:
            raise ValueError(f"values cannot be recognized as {spec.dtype} in text")
        self.column = column
        self.spec = spec
        self.physical = physical

    def scale(self) -> pl.Expr | None:
        """A reduction giving the column's decimal scale; ``None`` for other types."""
        if self.spec.dtype != "decimal":
            return None
        if self.spec.decimal_scale is not None:
            return None
        return DecimalText.scale(self._present()).alias(self.column)

    def value(self, scale: int | None = None) -> pl.Expr:
        """The converted value: null where missing or invalid."""
        return self._converted(scale)[0].alias(self.column)

    def invalid(self, scale: int | None = None) -> pl.Expr:
        """Whether a cell holds text that is neither missing nor of the declared type."""
        value, missing = self._converted(scale)
        invalid = ~missing & value.is_null()
        if self.spec.na_values:
            invalid = invalid & ~NotApplicablePolicy.from_spec(self.spec).polars(self.column)
        return invalid

    def _converted(self, scale: int | None) -> tuple[pl.Expr, pl.Expr]:
        if isinstance(self.physical, pl.Datetime):
            value = pl.col(self.column)
            if self.physical.time_zone is not None:
                value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
            ticks = {"ns": 1, "us": 1000, "ms": 1_000_000}
            ratio = ticks[self.spec.datetime_unit] // ticks[self.physical.time_unit]
            if ratio > 1:
                value = pl.when(value.cast(pl.Int64) % ratio == 0).then(value)
            multiplier = ticks[self.physical.time_unit] // ticks[self.spec.datetime_unit]
            if multiplier > 1:
                minimum = -((2**63) // multiplier)
                maximum = (2**63 - 1) // multiplier
                value = pl.when(value.cast(pl.Int64).is_between(minimum, maximum)).then(value)
            return value.dt.cast_time_unit(self.spec.datetime_unit), pl.col(self.column).is_null()
        text = pl.col(self.column).cast(pl.String)
        typed = self.spec.text_type()
        if typed is not None:
            return polars_value(text, typed)
        scale = self.spec.decimal_scale if self.spec.decimal_scale is not None else scale
        if scale is None:
            raise ValueError("a decimal conversion needs its column scale")
        missing = (
            pl.col(self.column).is_null()
            if self.physical is not None and self.physical.is_decimal()
            else self._missing(text)
        )
        value = DecimalText.value(self._present(), scale, self.spec.decimal_precision or 38)
        return pl.when(missing).then(None).otherwise(value), missing

    def _missing(self, text: pl.Expr) -> pl.Expr:
        missing = text.is_null()
        markers = [value for value in self.spec.missing_values if isinstance(value, str)]
        if markers:
            missing = missing | text.is_in(pl.lit(pl.Series(markers, dtype=pl.String)).implode())
        if self.spec.blank_as_missing:
            missing = missing | (polars_stripped(text, self.spec.trim_whitespace) == "")
        return missing.fill_null(True)

    def _present(self) -> pl.Expr:
        text = pl.col(self.column).cast(pl.String)
        if self.physical is not None and self.physical.is_decimal():
            # A physical decimal has already been parsed; a lexical locale
            # cannot reinterpret its canonical numeric representation.
            return text
        stripped = polars_stripped(text, self.spec.trim_whitespace)
        pattern = self.spec.number_grammar.pattern("decimal")
        present = (
            pl.when(~self._missing(text) & stripped.str.contains(f"^(?:{pattern})$"))
            .then(stripped)
            .otherwise(None)
        )
        return self.spec.number_grammar.polars_canonical(present)


class ConvertedPlan:
    """A lazy plan with text columns replaced by their declared-type values."""

    def __init__(self, conversions: Sequence[TextConversion], scales: Mapping[str, int]) -> None:
        self.conversions = list(conversions)
        self.scales = dict(scales)

    @classmethod
    def resolved(
        cls,
        conversions: Sequence[TextConversion],
        reduce: Callable[[Callable[[pl.LazyFrame], pl.LazyFrame]], pl.DataFrame],
    ) -> ConvertedPlan:
        """Resolve decimal scales with one reduction; ``reduce`` collects a plan's
        reduction (an observation's ``reduce``)."""
        reductions = [r for c in conversions if (r := c.scale()) is not None]
        scales: dict[str, int] = {}
        if reductions:
            row = reduce(ScaleReduction(reductions)).row(0, named=True)
            scales = {name: int(value) for name, value in row.items()}
        return cls(conversions, scales)

    def scale(self, column: str) -> int | None:
        return self.scales.get(column)

    def __call__(self, plan: pl.LazyFrame) -> pl.LazyFrame:
        return plan.with_columns(c.value(self.scales.get(c.column)) for c in self.conversions)


class ScaleReduction:
    """The plan reducing a source to its decimal columns' scales."""

    def __init__(self, reductions: Sequence[pl.Expr]) -> None:
        self.reductions = list(reductions)

    def __call__(self, plan: pl.LazyFrame) -> pl.LazyFrame:
        return plan.select(self.reductions)


@dataclass(frozen=True, slots=True)
class Converted:
    """A text column interpreted as its declared type.

    ``values`` is null where a cell is missing or invalid; ``invalid`` marks cells
    that hold text which is neither missing nor a value of the declared type.
    """

    values: pl.Series
    invalid: pl.Series


def convert(series: pl.Series, spec: ColumnSpec) -> Converted:
    """Convert an in-memory text series with :class:`TextConversion`."""
    frame = series.rename("text").to_frame()
    conversion = TextConversion("text", spec, series.dtype)
    reduction = conversion.scale()
    scale = int(frame.select(reduction).item()) if reduction is not None else None
    out = frame.select(
        conversion.value(scale).alias("value"), conversion.invalid(scale).alias("invalid")
    )
    return Converted(out["value"].alias(series.name), out["invalid"].alias(series.name))
