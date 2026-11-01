"""Lossless native key alignment without an optional dataframe dependency."""

from __future__ import annotations

import re
from collections.abc import Callable

INTEGERS = frozenset(
    (
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
    )
)
FLOATS = frozenset(("FLOAT", "REAL", "DOUBLE"))


def decimal_key_domain(left, right):
    """Return a common (precision, scale), refusing fixed-width identity loss."""
    scale = max(left[1], right[1])
    width = max((precision or 38) - places for precision, places in (left, right)) + scale
    return (width, scale) if width <= 38 else None


def key_casts(
    column: str, left: str, right: str, exact_float: Callable[[], bool]
) -> tuple[str | None, str | None]:
    """Plan a common domain only when every original identity is preserved."""
    if left == right:
        return None, None
    if left in {"NULL", '"NULL"'}:
        return right, None
    if right in {"NULL", '"NULL"'}:
        return None, left
    # Signed 128 cannot represent every unsigned 128 value.
    if left in INTEGERS and right in INTEGERS and "UHUGEINT" not in (left, right):
        return "HUGEINT", "HUGEINT"
    if left in FLOATS and right in FLOATS:
        return "DOUBLE", "DOUBLE"
    if left in INTEGERS | FLOATS and right in INTEGERS | FLOATS and exact_float():
        return "DOUBLE", "DOUBLE"
    decimals = [re.fullmatch(r"DECIMAL\((\d+),\s*(\d+)\)", t) for t in (left, right)]
    if all(decimals):
        parts = [tuple(map(int, d.groups())) for d in decimals if d is not None]
        domain = decimal_key_domain(*parts)
        if domain is not None:
            width, scale = domain
            target = f"DECIMAL({width},{scale})"
            return target, target
    raise ValueError(
        f"key column {column!r} cannot be compared exactly: {left} vs {right}; "
        "declare compatible logical types or cast losslessly in the source"
    )
