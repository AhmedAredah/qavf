"""Names of the datasets a check relating many datasets derives, and how users see them.

Derived datasets (:mod:`qavf.engine.derived`) are named after the check item's
place, ``<role>~checks[2]`` or ``<member>~checks[2]``, so they never clash with
configured datasets. Results show them by role (``consensus``, ``pooled``) or by
member instead.

A dataset check item with ``where`` whose check does not filter rows itself reads
a hidden view of its dataset, ``<dataset>.__where_<digest of the condition>``
(:class:`RowFilter`); results show it as the dataset, with ``where`` among the
parameters. Imports neither Polars nor Pydantic.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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


WHERE_VIEW_PREFIX = "__where_"
"""Reserved: views named so are the hidden views of filtered check items."""
_WHERE_VIEW = re.compile(r".+\." + WHERE_VIEW_PREFIX + r"[0-9a-f]{12}")


def where_view_name(dataset: str, condition: str) -> str:
    """The hidden view of ``dataset`` holding the rows matching ``condition``; every
    item of the dataset with that condition reads the same view."""
    digest = hashlib.sha256(condition.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}.{WHERE_VIEW_PREFIX}{digest}"


def is_where_view(name: str) -> bool:
    """Whether ``name`` is the hidden view of a filtered check item."""
    return _WHERE_VIEW.fullmatch(name) is not None


@dataclass(frozen=True, slots=True)
class RowFilter:
    """The rows a filtered dataset check item assesses: those of its dataset matching
    ``where``, read through a hidden view (:func:`where_view_name`)."""

    dataset: str
    where: str
    min_applicable_rows: int = 0
    """Fewer matching rows make the check not assessed; an empty population is never
    assessed, whatever the minimum."""

    @property
    def view(self) -> str:
        return where_view_name(self.dataset, self.where)

    @property
    def minimum(self) -> int:
        """The fewest matching rows the check is assessed on (at least one)."""
        return max(1, self.min_applicable_rows)

    def shown(self, value: Any) -> Any:
        """``value`` (text, or JSON-like lists and mappings) with the hidden view, in
        keys and text, shown as its dataset."""
        if isinstance(value, str):
            return value.replace(self.view, self.dataset)
        if isinstance(value, Mapping):
            return {self.shown(key): self.shown(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            items = [self.shown(item) for item in value]
            return items if isinstance(value, list) else tuple(items)
        return value

    def shown_datasets(self, names: Iterable[str]) -> list[str]:
        """``names`` with the hidden view shown as its dataset, once."""
        return list(dict.fromkeys(self.shown(name) for name in names))

    def parameters(self) -> dict[str, Any]:
        """The settings selecting the rows, shown among a result's parameters."""
        shown: dict[str, Any] = {"where": self.where}
        if self.min_applicable_rows:
            shown["min_applicable_rows"] = self.min_applicable_rows
        return shown


__all__ = [
    "WHERE_VIEW_PREFIX",
    "RowFilter",
    "derived_name",
    "is_derived",
    "is_where_view",
    "shown_datasets",
    "shown_name",
    "shown_value",
    "where_view_name",
]
