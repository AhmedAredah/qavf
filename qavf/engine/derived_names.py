"""Names of the datasets a check relating many datasets derives, and how users see them.

Derived datasets (:mod:`qavf.engine.derived`) are named after the check item's
place, ``<role>~checks[2]`` or ``<member>~checks[2]``, so they never clash with
configured datasets. Results show them by role (``consensus``, ``pooled``) or by
member instead. Imports neither Polars nor Pydantic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


def derived_name(role: str, path: str, member: str | None = None) -> str:
    """The name of a dataset a check item at ``path`` derives: ``consensus~checks[2]``,
    ``pooled~checks[2]``, or ``<member>~checks[2]`` for a member without its keys that
    have no consensus."""
    return f"{member if role == 'member' else role}~{path}"


_DERIVED = re.compile(r"([^\s'\"`(\[{,;]+)~checks\[\d+\]")
"""A derived dataset name inside text: ``<role or member>~checks[<n>]``."""
_ROLES = {"full": "consensus", "consensus": "consensus", "pooled": "pooled"}
"""How each derived role is shown; a member's rows are shown as the member."""


def shown_name(name: str) -> str:
    """The name users see for a dataset: a derived one by its role or member
    (``consensus~checks[0]`` is ``consensus``, ``counts@c~checks[0]`` is ``counts@c``)."""
    return _DERIVED.sub(_shown_match, name)


def _shown_match(match: re.Match[str]) -> str:
    prefix = match.group(1)
    return _ROLES.get(prefix, prefix)


def shown_datasets(names: Iterable[str]) -> list[str]:
    """The configured datasets among ``names``, derived ones shown as their member;
    the derived roles (consensus, pooled) only when no member is named."""
    names = list(names)
    members = [shown_name(name) for name in names if _role(name) is None]
    shown = members or [shown_name(name) for name in names]
    return list(dict.fromkeys(shown))


def is_derived(name: str) -> bool:
    """Whether ``name`` is a dataset a check derived from its members."""
    return _DERIVED.fullmatch(name) is not None


def _role(name: str) -> str | None:
    match = _DERIVED.fullmatch(name)
    return match.group(1) if match and match.group(1) in _ROLES else None


def shown_value(value: Any) -> Any:
    """``value`` (text, or JSON-like lists and mappings) with every derived dataset
    name, in keys and text, shown as users see it."""
    if isinstance(value, str):
        return shown_name(value) if "~" in value else value
    if isinstance(value, Mapping):
        return {shown_value(key): shown_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        items = [shown_value(item) for item in value]
        return items if isinstance(value, list) else tuple(items)
    return value


__all__ = ["derived_name", "is_derived", "shown_datasets", "shown_name", "shown_value"]
