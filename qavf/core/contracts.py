"""Lightweight dependency and canonical-key policies for native contracts."""

from __future__ import annotations

from collections.abc import Iterable


def condition_dependencies(expression: str | None, columns: Iterable[str]) -> tuple[str, ...]:
    """Use parser-proven dependencies, retaining every column for unsafe syntax."""
    if not expression:
        return ()
    from qavf.core.planning import expression_columns

    dependencies = expression_columns(expression)
    if dependencies is None:
        return tuple(columns)
    tokens = {column.casefold() for column in dependencies}
    return tuple(column for column in columns if column.casefold() in tokens)


def canonical_key(expression: str, dtype: str) -> str:
    """IEEE NaN is missing for key identity, including undeclared columns."""
    if dtype in {"FLOAT", "REAL", "DOUBLE"}:
        return f"CASE WHEN isnan({expression}) THEN NULL ELSE {expression} END"
    return expression
