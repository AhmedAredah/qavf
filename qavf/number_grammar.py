"""Engine-independent lexical policy for numbers and surrounding whitespace.

Grammar admission precedes conversion. In particular, admitting an integral
decimal never routes an integer through a float or a rounding cast.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

Whitespace = Literal["ascii", "unicode", "none"]
# A fixed set shared by Python, DuckDB and Polars, including Python's four
# ASCII information separators. Engine-dependent default trim sets differ.
UNICODE_WHITESPACE = (
    " \t\n\r\v\f\x1c\x1d\x1e\x1f\x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
# ``normalize: [trim]`` predates configurable type parsing and used Rust's
# Unicode White_Space property (which excludes Python's information separators).
COMPARISON_WHITESPACE = UNICODE_WHITESPACE.translate(dict.fromkeys(map(ord, "\x1c\x1d\x1e\x1f")))


def whitespace_chars(policy: Whitespace = "ascii") -> str:
    """Characters stripped by the policy; ``ascii`` preserves the legacy space-only rule."""
    return {"ascii": " ", "unicode": UNICODE_WHITESPACE, "none": ""}[policy]


def trim_text(value: str, policy: Whitespace = "ascii") -> str:
    chars = whitespace_chars(policy)
    return value.strip(chars) if chars else value


def exact_decimal(value: int | float | Decimal, precision: int, scale: int) -> Decimal | None:
    """A literal on an exact fixed-point storage grid, without context rounding."""
    if isinstance(value, Decimal) and not value.is_finite():
        return None
    try:
        coefficient = Fraction(value) * 10**scale
    except (ValueError, OverflowError):
        return None
    if coefficient.denominator != 1 or abs(coefficient) >= 10**precision:
        return None
    integer = int(coefficient)
    return Decimal((int(integer < 0), tuple(map(int, str(abs(integer)))), -scale))


def _sql_text(value):
    """Quote a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


class NumberGrammar(BaseModel):
    """Optional numeric spellings. Defaults retain QAVF 0.6 lexical rules.

    ``int_decimal`` admits an integer followed by a decimal point and only
    zeros (``7.0``, ``+07.``). ``underscores`` admits one separator between
    digits. ``exponent`` controls float exponents; integers and exact decimal
    columns continue to use decimal notation. ``special_values`` admits float
    nan/inf spellings; missingness and ``allow_infinite`` still apply afterwards.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    int_decimal: bool = False
    underscores: bool = False
    exponent: bool = True
    special_values: bool = True
    decimal_separator: str = "."
    grouping_separator: str | None = None

    @model_validator(mode="after")
    def _separators(self):
        forbidden = "0123456789+-eE_"
        for name in ("decimal_separator", "grouping_separator"):
            value = getattr(self, name)
            if value is not None and (
                len(value) != 1 or value in forbidden or value.isalnum() or value in "\r\n\t"
            ):
                raise ValueError(f"{name} must be one nonnumeric punctuation/space character")
        if self.decimal_separator == self.grouping_separator:
            raise ValueError("decimal_separator and grouping_separator must differ")
        if self.decimal_separator.isspace():
            raise ValueError("decimal_separator cannot be whitespace")
        if self.grouping_separator is not None and self.underscores:
            raise ValueError("grouping_separator and underscores cannot both be enabled")
        return self

    def pattern(self, dtype: str) -> str:
        digits = r"[0-9](?:_?[0-9])*" if self.underscores else r"[0-9]+"
        integer = digits
        if self.grouping_separator is not None:
            group = re.escape(self.grouping_separator)
            integer = rf"(?:[0-9]+|[0-9]{{1,3}}(?:{group}[0-9]{{3}})+)"
        point = re.escape(self.decimal_separator)
        zeros = r"0(?:_?0)*" if self.underscores else r"0+"
        if dtype == "int":
            suffix = rf"(?:{point}(?:{zeros})?)?" if self.int_decimal else ""
            return rf"[+-]?{integer}{suffix}"
        decimal = rf"(?:{integer}(?:{point}(?:{digits})?)?|{point}{digits})"
        if dtype == "decimal":
            return rf"[+-]?{decimal}"
        if dtype != "float":
            raise ValueError(f"no numeric grammar for {dtype!r}")
        exponent = rf"(?:[eE][+-]?{digits})?" if self.exponent else ""
        finite = rf"[+-]?{decimal}{exponent}"
        return rf"(?:{finite}|[+-]?(?:inf|infinity|nan))" if self.special_values else finite

    def parse(self, text: str, dtype: str, trim_whitespace: Whitespace = "ascii") -> Any:
        """Parse a configuration literal exactly, raising ValueError on invalid grammar.

        Range admission belongs to the physical column type. Returning a Python
        integer here preserves all digits even beyond signed 64-bit storage.
        """
        value = trim_text(text, trim_whitespace)
        if re.fullmatch(self.pattern(dtype), value, re.IGNORECASE) is None:
            raise ValueError(f"{text!r} does not match the configured {dtype} grammar")
        value = self.canonical(value)
        if dtype == "int":
            return int(value.split(".", 1)[0])
        return Decimal(value) if dtype == "decimal" else float(value)

    def canonical(self, value: str) -> str:
        """Normalize an already admitted spelling; never call before grammar admission."""
        if self.underscores:
            value = value.replace("_", "")
        if self.grouping_separator is not None:
            value = value.replace(self.grouping_separator, "")
        return (
            value.replace(self.decimal_separator, ".") if self.decimal_separator != "." else value
        )

    def sql_canonical(self, expression: str) -> str:
        for old, new in self.replacements():
            expression = f"replace({expression}, {_sql_text(old)}, {_sql_text(new)})"
        return expression

    def polars_canonical(self, expression):
        for old, new in self.replacements():
            expression = expression.str.replace_all(old, new, literal=True)
        return expression

    def replacements(self):
        if self.underscores:
            yield "_", ""
        if self.grouping_separator is not None:
            yield self.grouping_separator, ""
        if self.decimal_separator != ".":
            yield self.decimal_separator, "."


def converted_options(
    values: list[Any], dtype: str, grammar: NumberGrammar, whitespace: Whitespace = "ascii"
) -> list[Any]:
    """Convert numeric string options once during contract planning.

    Invalid literals stay foreign values, allowing the ordinary compatibility
    diagnostic to explain them rather than silently removing options.
    """
    result = []
    for value in values:
        if isinstance(value, str):
            with suppress(ValueError):
                value = grammar.parse(value, dtype, whitespace)
        result.append(value)
    return result


@dataclass(frozen=True, slots=True)
class AllowedValueDomain:
    """Distinct configured values under the contract's equality policy.

    Domain cardinality plus membership and uniqueness proves exact coverage;
    it adds no per-row Python work and does not materialize a source column.
    """

    values: tuple[Any, ...]

    @classmethod
    def resolved(cls, values: list[Any], normalize: Sequence[str]) -> AllowedValueDomain:
        normalized = []
        for value in values:
            if isinstance(value, str):
                from qavf.text_policies import normalize_scalar

                value = normalize_scalar(value, normalize)
            normalized.append(value)
        try:
            return cls(tuple(dict.fromkeys(normalized)))
        except TypeError as exc:
            raise ValueError("'each_allowed_once' requires scalar allowed values") from exc

    @property
    def cardinality(self) -> int:
        return len(self.values)
