"""Backend-independent syntax admission and explicit regex dialect boundaries."""

from __future__ import annotations

import functools

from qavf.rules._regex import sql_compatible


@functools.lru_cache(maxsize=1024)
def check_regex_syntax(pattern: str) -> None:
    """Check shared syntax without selecting an engine from installed packages.

    Nonportable expressions receive their definitive compilation in the selected
    backend. This structural admission deliberately does not reinterpret Unicode
    classes or Rust escapes with another engine's matching semantics.
    """
    if sql_compatible(pattern):
        import duckdb

        from qavf.sources.sql import _parser

        try:
            _parser().execute("SELECT regexp_full_match('', ?)", [pattern])
        except duckdb.Error as exc:
            raise ValueError(f"invalid regex {pattern!r}: {exc}") from None
        return
    groups = 0
    classes = 0
    escaped = False
    for char in pattern:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "[":
            classes += 1
        elif char == "]" and classes:
            classes -= 1
        elif not classes:
            if char == "(":
                groups += 1
            elif char == ")":
                groups -= 1
                if groups < 0:
                    raise ValueError(f"invalid regex {pattern!r}: unmatched parenthesis")
    if escaped or groups or classes:
        raise ValueError(f"invalid regex {pattern!r}: unclosed group, class, or escape")


def require_native_regex(pattern: str) -> None:
    """Never silently substitute RE2's ASCII classes for Rust Unicode classes."""
    if not sql_compatible(pattern):
        raise ValueError(
            "Regex is outside the shared Rust/RE2 subset supported by backend='duckdb'; "
            "select backend='polars' with qavf[polars] for Unicode classes, flags, "
            "or Rust-specific escapes, or use an explicit portable character class."
        )
