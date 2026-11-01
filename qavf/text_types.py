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
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    import polars as pl
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from qavf.number_grammar import NumberGrammar, Whitespace, trim_text, whitespace_chars

TextDType = Literal["int", "float", "bool", "date", "datetime"]
TEXT_DTYPES: tuple[TextDType, ...] = ("int", "float", "bool", "date", "datetime")
"""Types recognized in text by both engines (``decimal`` needs the data's scale)."""

WHITESPACE = " "
"""Legacy surrounding-space policy; retained for imports from earlier versions.
Use :func:`polars_stripped` or :func:`sql_stripped` for configurable trimming.
"""
_PATTERNS: dict[str, str] = {
    "int": r"[+-]?[0-9]+",
    "float": r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?|[+-]?(inf|infinity|nan)",
    "date": r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
    "datetime": (r"[0-9]{4}-[0-9]{2}-[0-9]{2}([ T][0-9]{2}:[0-9]{2}(:[0-9]{2}(\.[0-9]{1,6})?)?)?"),
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
    number_grammar: NumberGrammar = Field(default_factory=NumberGrammar)
    trim_whitespace: Whitespace = "ascii"
    missing: list[str] = Field(default_factory=list)
    """Text that means missing, compared exactly as written."""
    blank_as_missing: bool = False
    format: str | None = None
    """For ``date``/``datetime``: how values are written, e.g. ``%m/%d/%Y``."""
    true_values: list[Word] = Field(default_factory=lambda: ["true"])
    """For ``bool``: text meaning true, compared without case or surrounding space."""
    false_values: list[Word] = Field(default_factory=lambda: ["false"])
    datetime_unit: Literal["ms", "us", "ns"] = "us"
    timezone: str | None = None
    ambiguous: Literal["raise", "earliest", "latest", "null"] = "raise"
    nonexistent: Literal["raise", "null"] = "raise"

    @property
    def governed_datetime(self) -> bool:
        return self.dtype == "datetime" and (
            self.datetime_unit != "us" or self.timezone is not None
        )

    @property
    def sql_datetime_type(self) -> str:
        return {"ms": "TIMESTAMP_MS", "us": "TIMESTAMP", "ns": "TIMESTAMP_NS"}[self.datetime_unit]

    @model_validator(mode="after")
    def _consistent(self) -> TextType:
        if self.timezone is not None:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"unknown IANA timezone {self.timezone!r}") from exc
        if self.format is not None:
            if self.dtype not in ("date", "datetime"):
                raise ValueError(f"'format' applies to date and datetime, not {self.dtype}")
            check_format(self.format, self.dtype)
        truthy = {trim_text(value, self.trim_whitespace).lower() for value in self.true_values}
        falsy = {trim_text(value, self.trim_whitespace).lower() for value in self.false_values}
        if not truthy or not falsy:
            raise ValueError("true_values and false_values must not be empty")
        if truthy & falsy:
            raise ValueError(f"{sorted(truthy & falsy)} cannot mean both true and false")
        return self


def polars_value(text: pl.Expr, spec: TextType) -> tuple[pl.Expr, pl.Expr]:
    """``(value, missing)`` for a text expression: the value is null when missing or
    when the text is not a value of ``spec.dtype``."""
    import polars as pl

    missing = text.is_null()
    if spec.missing:
        missing = missing | text.is_in(pl.lit(pl.Series(spec.missing, dtype=pl.String)).implode())
    stripped = polars_stripped(text, spec.trim_whitespace)
    if spec.blank_as_missing:
        missing = missing | (stripped == "")
    missing = missing.fill_null(True)
    parsed = _polars_parsed(stripped, spec)
    return pl.when(missing).then(None).otherwise(parsed), missing


def _governed_datetime_text(convert: Any, spec: TextType, value: Any) -> Any:
    """Render one value through the governed datetime text policy of a spec."""
    return convert(
        value,
        spec.datetime_unit,
        spec.timezone or "",
        spec.ambiguous,
        spec.nonexistent,
        spec.format or "",
    )


def _polars_parsed(text: pl.Expr, spec: TextType) -> pl.Expr:
    import polars as pl

    if spec.governed_datetime:
        from qavf.text_policies import datetime_text

        return text.map_elements(
            partial(_governed_datetime_text, datetime_text, spec),
            return_dtype=pl.String,
        ).str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit=spec.datetime_unit, strict=False)

    if spec.dtype == "bool":
        lowered = text.str.to_lowercase()
        truthy = [trim_text(value, spec.trim_whitespace).lower() for value in spec.true_values]
        falsy = [trim_text(value, spec.trim_whitespace).lower() for value in spec.false_values]
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
    pattern = (
        spec.number_grammar.pattern(spec.dtype)
        if spec.dtype in ("int", "float")
        else _PATTERNS[spec.dtype]
    )
    gated = pl.when(text.str.contains(f"{flags}^(?:{pattern})$")).then(text).otherwise(None)
    if spec.dtype in ("int", "float"):
        gated = spec.number_grammar.polars_canonical(gated)
    if spec.dtype == "int" and spec.number_grammar.int_decimal:
        gated = gated.str.replace(r"\.0*$", "")
    if spec.dtype == "int":
        return gated.cast(pl.Int64, strict=False)
    if spec.dtype == "float":
        return gated.cast(pl.Float64, strict=False)
    if spec.dtype == "date":
        return gated.str.to_date("%Y-%m-%d", strict=False)
    return pl.coalesce(
        [gated.str.to_datetime(f, strict=False, time_unit="us") for f in _DATETIME_FORMATS]
    )


def polars_stripped(text: pl.Expr, policy: Whitespace = "ascii") -> pl.Expr:
    """Strip exactly the shared character set, with no engine-dependent defaults."""
    chars = whitespace_chars(policy)
    return text.str.strip_chars(chars) if chars else text


def sql_stripped(column: str, policy: Whitespace = "ascii") -> str:
    """The quoted text ``column`` without configured surrounding whitespace."""
    chars = whitespace_chars(policy)
    return f"trim({column}, {_literal(chars)})" if chars else column


def sql_prepared(column: str, spec: TextType) -> str:
    """Reusable parsing input; ASCII numbers need no character-set trim pass.

    Their regex admits surrounding ASCII spaces on the original text, then the
    numeric cast handles those spaces. Checking the raw spelling first prevents
    DuckDB's broader whitespace/casting rules from admitting Unicode spaces.
    """
    if spec.trim_whitespace == "ascii" and spec.dtype in ("int", "float"):
        return column
    return sql_stripped(column, spec.trim_whitespace)


def sql_value(column: str, spec: TextType, stripped: str | None = None) -> tuple[str, str]:
    """``(value, invalid)`` DuckDB expressions over the quoted text ``column``.

    ``invalid`` is true where the text is neither missing nor a value. Pass the
    column's :func:`sql_prepared` text, computed once, as ``stripped``: DuckDB
    evaluates a repeated expression again at every reference. ASCII numeric
    admission always examines the original column, even if a caller supplies
    a stripped expression, so normalization cannot hide invalid whitespace.
    """
    ascii_numeric = spec.trim_whitespace == "ascii" and spec.dtype in ("int", "float")
    stripped = column if ascii_numeric else stripped or sql_prepared(column, spec)
    missing = [f"{column} IS NULL"]
    if spec.missing:
        missing.append(f"{column} IN ({', '.join(_literal(v) for v in spec.missing)})")
    if spec.blank_as_missing:
        missing.append(
            f"regexp_full_match({column}, ' *')" if ascii_numeric else f"{stripped} = ''"
        )
    is_missing = " OR ".join(f"({part})" for part in missing)
    value = (
        f"CASE WHEN {is_missing} THEN NULL "
        f"ELSE {_sql_parsed(stripped, spec, ascii_spaces=ascii_numeric)} END"
    )
    return value, f"(NOT ({is_missing}) AND ({value}) IS NULL)"


def _sql_parsed(text: str, spec: TextType, *, ascii_spaces: bool = False) -> str:
    if spec.governed_datetime:
        arguments = [
            spec.datetime_unit,
            spec.timezone or "",
            spec.ambiguous,
            spec.nonexistent,
            spec.format or "",
        ]
        return (
            f"try_cast(__qavf_datetime_text({text}, "
            f"{', '.join(_literal(value) for value in arguments)}) AS {spec.sql_datetime_type})"
        )
    if spec.dtype == "bool":
        truthy = ", ".join(
            _literal(trim_text(v, spec.trim_whitespace).lower()) for v in spec.true_values
        )
        falsy = ", ".join(
            _literal(trim_text(v, spec.trim_whitespace).lower()) for v in spec.false_values
        )
        # DuckDB simple lowercase differs from Unicode full lowercase (e.g. İ).
        # Keep the common ASCII path native; opt into the shared scalar policy
        # only when Unicode words can match a non-ASCII spelling.
        words = [
            trim_text(v, spec.trim_whitespace).lower() for v in spec.true_values + spec.false_values
        ]
        if all(word.isascii() for word in words):
            # Unicode full lowercase has one unconditional expanding mapping:
            # İ -> i + combining dot. Protect it before native simple lowercase
            # so it cannot spuriously match the ASCII word "i". Contextual Greek
            # sigma cannot match an ASCII domain; ordinary booleans need no UDF.
            prepared = f"replace({text}, 'İ', 'i̇')" if any("i" in word for word in words) else text
            lowered = f"lower({prepared})"
        else:
            lowered = (
                f"CASE WHEN regexp_full_match({text}, '[\\x00-\\x7f]*') "
                f"THEN lower({text}) ELSE __qavf_unicode_lower({text}) END"
            )
        return (
            f"CASE WHEN {lowered} IN ({truthy}) THEN TRUE "
            f"WHEN {lowered} IN ({falsy}) THEN FALSE END"
        )
    if spec.format is not None:
        parsed = (
            f"CASE WHEN regexp_matches({text}, '[0-9]') "
            f"THEN try_strptime({text}, {_literal(spec.format)}) END"
        )
        return f"CAST({parsed} AS DATE)" if spec.dtype == "date" else parsed
    options = ", 'i'" if spec.dtype in _CASELESS else ""
    target = {"int": "BIGINT", "float": "DOUBLE", "date": "DATE", "datetime": "TIMESTAMP"}
    pattern = (
        spec.number_grammar.pattern(spec.dtype)
        if spec.dtype in ("int", "float")
        else _PATTERNS[spec.dtype]
    )
    pattern = _literal(f" *(?:{pattern}) *" if ascii_spaces else pattern)
    parsed = text
    if spec.dtype in ("int", "float"):
        parsed = spec.number_grammar.sql_canonical(parsed)
    if spec.dtype == "int" and spec.number_grammar.int_decimal:
        if ascii_spaces:
            # Only reached after the raw ASCII-space grammar has passed. The
            # faster default trim therefore cannot admit other whitespace.
            parsed = f"trim({parsed})"
        parsed = f"regexp_replace({parsed}, '\\.0*$', '')"
    return (
        f"CASE WHEN regexp_full_match({text}, {pattern}{options}) "
        f"THEN TRY_CAST({parsed} AS {target[spec.dtype]}) END"
    )


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
