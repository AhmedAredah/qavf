"""Conservative equivalence boundary between Rust regex and DuckDB's RE2."""

from __future__ import annotations


def sql_compatible(pattern: str) -> bool:
    """Whether this pattern belongs to the shared ASCII-expression subset.

    Input text may remain Unicode. Unicode classes, Perl shorthand classes,
    flags, escapes with engine-specific meanings and class set operations use
    the canonical Polars evaluator. Successful compilation alone proves no
    semantic equivalence between regex engines.
    """
    if not pattern.isascii() or any(operator in pattern for operator in ("&&", "--", "~~")):
        return False
    index = 0
    in_class = False
    while index < len(pattern):
        if pattern[index] == "\\":
            index += 1
            if index == len(pattern) or pattern[index] not in r"\.[]{}()*+?^$|-/":
                return False
        elif pattern.startswith("(?", index) and not pattern.startswith("(?:", index):
            return False
        elif pattern[index] == "[":
            if in_class:
                return False
            in_class = True
        elif pattern[index] == "]":
            in_class = False
        index += 1
    return True
