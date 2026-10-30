"""Values of a declared type written as text: one definition for every engine.

CSV reads, frames and column checks recognize a declared type in text the same
way. Surrounding spaces are ignored, text markers mean missing, and a value
must match its type's pattern in full before it is converted. Polars and DuckDB
test the same patterns first, because their own casts differ (DuckDB rounds
``2.5`` to an integer and reads ``0x10`` as 16); both then accept exactly the
same text.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

import polars as pl
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

TextDType = Literal["int", "float", "bool", "date", "datetime"]
TEXT_DTYPES: tuple[TextDType, ...] = ("int", "float", "bool", "date", "datetime")
"""Types recognized in text by both engines (``decimal`` needs the data's scale)."""

WHITESPACE = " "
"""Surrounding characters ignored in text values: spaces, which DuckDB's ``trim``
removes at no measurable cost (trimming a character set costs more than the read)."""
_PATTERNS: dict[str, str] = {
    "int": r"[+-]?[0-9]+",
    "float": r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?|[+-]?(inf|infinity|nan)",
    "date": r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
    "datetime": (
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}([ T][0-9]{2}:[0-9]{2}(:[0-9]{2}(\.[0-9]{1,6})?)?)?"
    ),
}
_CASELESS = frozenset({"float"})
_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)
_FORMAT_DIRECTIVE = re.compile(r"%(.)")
_DATE_DIRECTIVES = frozenset("YymdbBj%")
_TIME_DIRECTIVES = frozenset("HIMSp")


def check_format(fmt: str, dtype: str) -> None:
    """Accept only directives both engines parse identically.

    ``%Y %y %m %d %b %B %j`` for dates, plus ``%H %I %M %S %p`` for datetimes;
    ``%%`` is a literal percent sign.
    """
    allowed = _DATE_DIRECTIVES | (_TIME_DIRECTIVES if dtype == "datetime" else frozenset())
    used = _FORMAT_DIRECTIVE.findall(fmt)
    unknown = sorted({f"%{d}" for d in used if d not in allowed})
    if unknown:
        raise ValueError(
            f"format {fmt!r} uses {', '.join(unknown)}; a {dtype} format may use "
            f"{' '.join('%' + d for d in sorted(allowed))}"
        )
    if fmt.count("%") != 2 * fmt.count("%%") + len([d for d in used if d != "%"]):
        raise ValueError(f"format {fmt!r} ends with an incomplete directive")


def _word(value: Any) -> Any:
    """A boolean word as written; numbers such as ``1`` are words too."""
    if isinstance(value, bool):
        raise ValueError(
            "YAML reads unquoted yes, no, on, off, true and false as booleans; quote the "
            "word as it is written in the data, e.g. ['yes']"
        )
    return str(value) if isinstance(value, int | float) else value


Word = Annotated[str, BeforeValidator(_word)]
"""Text meaning true or false in a ``bool`` column."""


class TextType(BaseModel):
    """How a column's declared type is written as text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dtype: TextDType
    missing: list[str] = Field(default_factory=list)
    """Text that means missing, compared exactly as written."""
    blank_as_missing: bool = False
    format: str | None = None
    """For ``date``/``datetime``: how values are written, e.g. ``%m/%d/%Y``."""
    true_values: list[Word] = Field(default_factory=lambda: ["true"])
    """For ``bool``: text meaning true, compared without case or surrounding space."""
    false_values: list[Word] = Field(default_factory=lambda: ["false"])

    @model_validator(mode="after")
    def _consistent(self) -> TextType:
        if self.format is not None:
            if self.dtype not in ("date", "datetime"):
                raise ValueError(f"'format' applies to date and datetime, not {self.dtype}")
            check_format(self.format, self.dtype)
        truthy = {value.strip(WHITESPACE).lower() for value in self.true_values}
        falsy = {value.strip(WHITESPACE).lower() for value in self.false_values}
        if not truthy or not falsy:
            raise ValueError("true_values and false_values must not be empty")
        if truthy & falsy:
            raise ValueError(f"{sorted(truthy & falsy)} cannot mean both true and false")
        return self


def polars_value(text: pl.Expr, spec: TextType) -> tuple[pl.Expr, pl.Expr]:
    """``(value, missing)`` for a text expression: the value is null when missing or
    when the text is not a value of ``spec.dtype``."""
    missing = text.is_null()
    if spec.missing:
        missing = missing | text.is_in(pl.lit(pl.Series(spec.missing, dtype=pl.String)).implode())
    stripped = text.str.strip_chars(WHITESPACE)
    if spec.blank_as_missing:
        missing = missing | (stripped == "")
    missing = missing.fill_null(True)
    parsed = _polars_parsed(stripped, spec)
    return pl.when(missing).then(None).otherwise(parsed), missing


def _polars_parsed(text: pl.Expr, spec: TextType) -> pl.Expr:
    if spec.dtype == "bool":
        lowered = text.str.to_lowercase()
        truthy = [value.strip(WHITESPACE).lower() for value in spec.true_values]
        falsy = [value.strip(WHITESPACE).lower() for value in spec.false_values]
        return (
            pl.when(lowered.is_in(pl.lit(pl.Series(truthy)).implode()))
            .then(True)
            .when(lowered.is_in(pl.lit(pl.Series(falsy)).implode()))
            .then(False)
        )
    if spec.format is not None:
        # A written date has digits; DuckDB alone would read "-infinity" as one.
        dated = pl.when(text.str.contains("[0-9]")).then(text)
        if spec.dtype == "date":
            return dated.str.to_date(spec.format, strict=False)
        return dated.str.to_datetime(spec.format, strict=False, time_unit="us")
    flags = "(?i)" if spec.dtype in _CASELESS else ""
    gated = pl.when(text.str.contains(f"{flags}^(?:{_PATTERNS[spec.dtype]})$")).then(text)
    if spec.dtype == "int":
        return gated.cast(pl.Int64, strict=False)
    if spec.dtype == "float":
        return gated.cast(pl.Float64, strict=False)
    if spec.dtype == "date":
        return gated.str.to_date("%Y-%m-%d", strict=False)
    return pl.coalesce(
        [gated.str.to_datetime(f, strict=False, time_unit="us") for f in _DATETIME_FORMATS]
    )


def sql_stripped(column: str) -> str:
    """The quoted text ``column`` without surrounding whitespace, as in Polars."""
    return f"trim({column})"


def sql_value(column: str, spec: TextType, stripped: str | None = None) -> tuple[str, str]:
    """``(value, invalid)`` DuckDB expressions over the quoted text ``column``.

    ``invalid`` is true where the text is neither missing nor a value. Pass the
    column's :func:`sql_stripped` text, computed once, as ``stripped``: DuckDB
    evaluates a repeated expression again at every reference.
    """
    stripped = stripped or sql_stripped(column)
    missing = [f"{column} IS NULL"]
    if spec.missing:
        missing.append(f"{column} IN ({', '.join(_literal(v) for v in spec.missing)})")
    if spec.blank_as_missing:
        missing.append(f"{stripped} = ''")
    is_missing = " OR ".join(f"({part})" for part in missing)
    value = f"CASE WHEN {is_missing} THEN NULL ELSE {_sql_parsed(stripped, spec)} END"
    return value, f"(NOT ({is_missing}) AND ({value}) IS NULL)"


def _sql_parsed(text: str, spec: TextType) -> str:
    if spec.dtype == "bool":
        truthy = ", ".join(_literal(v.strip(WHITESPACE).lower()) for v in spec.true_values)
        falsy = ", ".join(_literal(v.strip(WHITESPACE).lower()) for v in spec.false_values)
        return (
            f"CASE WHEN lower({text}) IN ({truthy}) THEN TRUE "
            f"WHEN lower({text}) IN ({falsy}) THEN FALSE END"
        )
    if spec.format is not None:
        parsed = (
            f"CASE WHEN regexp_matches({text}, '[0-9]') "
            f"THEN try_strptime({text}, {_literal(spec.format)}) END"
        )
        return f"CAST({parsed} AS DATE)" if spec.dtype == "date" else parsed
    options = ", 'i'" if spec.dtype in _CASELESS else ""
    target = {"int": "BIGINT", "float": "DOUBLE", "date": "DATE", "datetime": "TIMESTAMP"}
    pattern = _literal(_PATTERNS[spec.dtype])
    return (
        f"CASE WHEN regexp_full_match({text}, {pattern}{options}) "
        f"THEN TRY_CAST({text} AS {target[spec.dtype]}) END"
    )


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
