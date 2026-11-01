"""Backend-independent recognition of explicitly not-applicable text tokens.

Applicability is independent of missingness and type conversion. This module has
no dataframe dependency until a caller explicitly requests its Polars compiler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
from typing import Any, Literal

from qavf.number_grammar import trim_text, whitespace_chars


def _sql_text(value: str) -> str:
    """Quote a SQL string literal without importing DuckDB, readers or Polars."""
    return "'" + value.replace("'", "''") + "'"


def _comparable_token(policy: NotApplicablePolicy, case: bool, value: str) -> str:
    """Normalize a token as a check compares it, folding case when requested."""
    normalized_value = policy.normalize(value)
    return normalized_value.lower() if case else normalized_value


@dataclass(frozen=True, slots=True)
class NotApplicablePolicy:
    """Recognize configured tokens without conflating them with SQL NULL.

    Tokens are case-sensitive. Surrounding whitespace follows the column's
    conversion policy. A null input is always false, including with empty tokens.
    """

    tokens: tuple[str, ...] = ()
    trim_whitespace: Literal["ascii", "unicode", "none"] = "ascii"

    @classmethod
    def from_spec(cls, spec: Any) -> NotApplicablePolicy:
        return cls(tuple(spec.na_values), getattr(spec, "trim_whitespace", "ascii"))

    def normalize(self, value: str) -> str:
        return trim_text(value, self.trim_whitespace)

    @property
    def normalized_tokens(self) -> tuple[str, ...]:
        return tuple(self.normalize(value) for value in self.tokens)

    def matches(self, value: Any) -> bool:
        return isinstance(value, str) and self.normalize(value) in self.normalized_tokens

    def sql(self, expression: str) -> str:
        """Compile a SQL expression (already quoted by its caller)."""
        if not self.tokens:
            return "FALSE"

        value = f"CAST(({expression}) AS VARCHAR)"
        if self.trim_whitespace == "ascii":
            value = f"trim({value}, ' ')"
        elif self.trim_whitespace == "unicode":
            chars = whitespace_chars("unicode")
            value = f"trim({value}, {_sql_text(chars)})"
        tokens = ", ".join(_sql_text(token) for token in self.normalized_tokens)
        text_type = (
            f"(typeof({expression}) = 'VARCHAR' OR starts_with(typeof({expression}), 'ENUM('))"
        )
        return f"({text_type} AND coalesce(({value} IN ({tokens})), FALSE))"

    def polars(self, column: str, dtype: Any = None):
        """Compile a mask lazily; null is never an N/A token."""
        import polars as pl

        if not self.tokens or (
            dtype is not None
            and dtype != pl.String
            and not isinstance(dtype, pl.Categorical | pl.Enum)
        ):
            return pl.lit(False)
        value = pl.col(column).cast(pl.String, strict=False)
        if self.trim_whitespace != "none":
            value = value.str.strip_chars(whitespace_chars(self.trim_whitespace))
        return value.is_in(
            pl.lit(pl.Series(self.normalized_tokens, dtype=pl.String)).implode()
        ).fill_null(False)

    def validate(self, spec: Any) -> None:
        """Reject ambiguous declarations before a source is read."""
        if not self.tokens:
            return
        tokens = self.normalized_tokens
        if len(tokens) != len(set(tokens)):
            raise ValueError("na_values repeats a token after whitespace normalization")
        if "" in tokens:
            raise ValueError("na_values cannot contain empty or whitespace-only tokens")
        missing = {self.normalize(value) for value in spec.missing_values if isinstance(value, str)}
        if overlap := set(tokens) & missing:
            raise ValueError(
                f"na_values overlaps missing_values after normalization: {sorted(overlap)!r}"
            )
        if spec.dtype == "bool":
            boolean = {
                self.normalize(value).lower() for value in [*spec.true_values, *spec.false_values]
            }
            if any(token.lower() in boolean for token in tokens):
                raise ValueError("na_values overlaps a true_values or false_values token")
        if spec.dtype in {"int", "float", "decimal"}:
            for token in tokens:
                try:
                    spec.number_grammar.parse(token, spec.dtype, self.trim_whitespace)
                except ValueError:
                    continue
                raise ValueError(f"na_values token {token!r} overlaps a valid {spec.dtype} value")
        if spec.dtype in {"date", "datetime"}:
            from qavf.text_types import _PATTERNS

            for token in tokens:
                try:
                    if spec.format:
                        datetime.strptime(token, spec.format)
                    else:
                        if re.fullmatch(_PATTERNS[spec.dtype], token) is None:
                            continue
                        if spec.dtype == "date":
                            date.fromisoformat(token)
                        else:
                            datetime.fromisoformat(token)
                except ValueError:
                    continue
                raise ValueError(f"na_values token {token!r} overlaps a valid {spec.dtype} value")
        for check in spec.all_checks():
            normalized = partial(_comparable_token, self, "case" in check.normalize)
            reserved = {normalized(value) for value in self.tokens}
            for name in ("isin", "notin"):
                if any(
                    isinstance(value, str) and normalized(value) in reserved
                    for value in (getattr(check, name) or [])
                ):
                    raise ValueError(f"na_values overlaps a value in {name}")


@dataclass(frozen=True, slots=True)
class AllOrNoneApplicability:
    """A row is inconsistent precisely when some, but not all, cells are N/A."""

    columns: tuple[str, ...]
    policies: tuple[NotApplicablePolicy, ...]

    @classmethod
    def from_schema(cls, columns: list[str], schema: Any) -> AllOrNoneApplicability:
        return cls(
            tuple(columns),
            tuple(NotApplicablePolicy.from_spec(schema.columns[name]) for name in columns),
        )

    def sql(self) -> str:
        masks = [
            policy.sql('"' + column.replace('"', '""') + '"')
            for column, policy in zip(self.columns, self.policies, strict=True)
        ]
        return f"({' OR '.join(masks)}) AND NOT ({' AND '.join(masks)})"

    def polars(self, actual: Any = None):
        import polars as pl

        masks = [
            policy.polars(column, actual.get(column) if actual is not None else None)
            for column, policy in zip(self.columns, self.policies, strict=True)
        ]
        return pl.any_horizontal(masks) & ~pl.all_horizontal(masks)
