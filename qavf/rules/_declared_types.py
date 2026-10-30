"""Values of a declared logical type in a column that was read as text.

A reader that infers types from a sample leaves a whole column as text when one
cell is not a value of its type (``12x`` in a count, ``N/A`` in a date). INP-002
reports exactly those cells, and INP-003 checks its limits on the rest. Both use
:func:`convert`, so they always agree on which cells are invalid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from qavf.settings import ColumnSpec
from qavf.text_types import polars_value
from qavf.value_policy import is_text

CONVERTIBLE = ("int", "float", "decimal", "bool", "date", "datetime")
"""Logical types whose values can be recognized in text."""

_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)
_DECIMAL = r"^[+-]?(\d+\.?\d*|\.\d+)$"
_MAX_DECIMAL_DIGITS = 38


def converts_text(spec: ColumnSpec, actual: pl.DataType) -> bool:
    """Whether a column declared as ``spec`` was read as text that may hold its values."""
    return spec.dtype in CONVERTIBLE and is_text(actual)


def described(dtype: str) -> str:
    """``an int``, ``a date``: the declared type as a noun phrase."""
    return f"{'an' if dtype == 'int' else 'a'} {dtype}"


@dataclass(frozen=True, slots=True)
class Converted:
    """A text column interpreted as its declared type.

    ``values`` is null where a cell is missing or invalid; ``invalid`` marks cells
    that hold text which is neither missing nor a value of the declared type.
    """

    values: pl.Series
    invalid: pl.Series


def convert(series: pl.Series, spec: ColumnSpec) -> Converted:
    """Interpret text as ``spec.dtype``; surrounding whitespace is ignored.

    Text markers in ``missing_values`` (and blank text with ``blank_as_missing``)
    are missing, never invalid. Typed markers such as ``-999`` apply afterwards,
    to the converted values (see :func:`typed_spec`).
    """
    if spec.dtype not in CONVERTIBLE:
        raise ValueError(f"values cannot be recognized as {spec.dtype} in text")
    text = series.cast(pl.String)
    frame = pl.DataFrame({"text": text})
    typed = spec.text_type()
    if typed is not None:
        # The same definition a typed CSV read uses, so both agree on every value.
        value, missing = polars_value(pl.col("text"), typed)
        out = frame.select(value.alias("value"), missing.alias("missing"))
        invalid = ~out["missing"] & out["value"].is_null()
        return Converted(out["value"].alias(series.name), invalid.alias(series.name))
    missing = pl.col("text").is_null()
    markers = [value for value in spec.missing_values if isinstance(value, str)]
    if markers:
        missing = missing | pl.col("text").is_in(pl.lit(pl.Series(markers)).implode())
    stripped = pl.col("text").str.strip_chars()
    if spec.blank_as_missing:
        missing = missing | (stripped == "")
    present = pl.when(missing).then(None).otherwise(stripped)
    flags = frame.select(missing.alias("missing"), present.alias("present"))
    values = flags.select(_parse(pl.col("present"), spec.dtype, flags["present"])).to_series()
    invalid = ~flags["missing"] & values.is_null()
    return Converted(values.alias(series.name), invalid.alias(series.name))


def typed_spec(spec: ColumnSpec) -> ColumnSpec:
    """``spec`` for the converted values: text markers have already been applied."""
    typed: list[Any] = [value for value in spec.missing_values if not isinstance(value, str)]
    return spec.model_copy(update={"missing_values": typed, "blank_as_missing": False})


def _parse(text: pl.Expr, dtype: str, values: pl.Series) -> pl.Expr:
    if dtype == "int":
        return text.cast(pl.Int64, strict=False)
    if dtype == "float":
        return text.cast(pl.Float64, strict=False)
    if dtype == "decimal":
        valid = values.str.contains(_DECIMAL)
        fractions = values.filter(valid).str.split(".").list.get(1, null_on_oob=True)
        widest = fractions.str.len_chars().max()
        scale = min(widest if isinstance(widest, int) else 0, _MAX_DECIMAL_DIGITS)
        return (
            pl.when(text.str.contains(_DECIMAL))
            .then(text)
            .cast(pl.Decimal(_MAX_DECIMAL_DIGITS, scale), strict=False)
        )
    if dtype == "bool":
        lowered = text.str.to_lowercase()
        return pl.when(lowered == "true").then(True).when(lowered == "false").then(False)
    if dtype == "date":
        return text.str.to_date("%Y-%m-%d", strict=False)
    return pl.coalesce(
        [text.str.to_datetime(f, strict=False, time_unit="us") for f in _DATETIME_FORMATS]
    )
