"""Where check items are placed, their names, and checks that need no rule code.

Everything here works without the optional Polars backend: the native DuckDB
backend validates ``expression`` and ``sql`` items with it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from qavf.checks.items import CheckItem

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings


class CheckSettingError(ValueError):
    """A check item setting that cannot be turned into rule parameters."""

    def __init__(self, field: str | None, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


@dataclass(frozen=True, slots=True)
class Scope:
    """Where a check item is placed."""

    path: str
    """The item's location, e.g. ``datasets.counts.checks[2]``."""
    dataset: str | None = None
    """The dataset a dataset-level item tests; ``None`` at the top level."""

    def at(self, field: str | None) -> str:
        """The path of one of the item's settings."""
        return f"{self.path}.{field}" if field else self.path


@dataclass(frozen=True, slots=True)
class PlacedItem:
    """A check item and where it is placed."""

    item: CheckItem
    scope: Scope

    @property
    def path(self) -> str:
        return self.scope.path

    @property
    def datasets(self) -> tuple[str, ...]:
        """The dataset an item is placed on, or the ``datasets`` a top-level item names."""
        if self.scope.dataset is not None:
            return (self.scope.dataset,)
        return related_datasets(self.item)

    @property
    def name(self) -> str:
        """The item's name, or its default ``<check> on <dataset>``."""
        return self.item.effective_name(self.datasets)


def related_datasets(item: CheckItem) -> tuple[str, ...]:
    """The datasets a top-level item names in ``datasets``."""
    named = item.settings.get("datasets")
    if isinstance(named, list | tuple):
        return tuple(name for name in named if isinstance(name, str))
    return ()


def placed_items(settings: QAVFSettings) -> Iterator[PlacedItem]:
    """Every check item, dataset lists first, with its location."""
    for dataset, spec in settings.datasets.items():
        for index, item in enumerate(spec.checks):
            yield PlacedItem(item, Scope(f"datasets.{dataset}.checks[{index}]", dataset))
    for index, item in enumerate(settings.checks):
        yield PlacedItem(item, Scope(f"checks[{index}]"))


def _location(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc if not isinstance(part, int))


def expression_errors(item: CheckItem, scope: Scope) -> list[str]:
    """Problems with an ``expression`` item."""
    from qavf.settings import expression_check

    if scope.dataset is None:
        return [f"{scope.at('check')}: expression goes in a dataset's checks list"]
    settings = item.settings
    unknown = sorted(set(settings) - {"expr", "null_as"})
    if unknown:
        return [f"{scope.at(unknown[0])}: unknown setting of the expression check"]
    if not isinstance(settings.get("expr"), str) or not settings["expr"].strip():
        return [f"{scope.at('expr')}: the expression check needs an 'expr' condition"]
    try:
        expression_check(item)
    except ValidationError as exc:
        return [f"{scope.at(_location(e['loc']))}: {e['msg']}" for e in exc.errors()]
    return []


def sql_errors(item: CheckItem, scope: Scope) -> list[str]:
    """Problems with a ``sql`` item."""
    if scope.dataset is not None:
        return [f"{scope.at('check')}: sql goes in the top-level checks list"]
    settings = item.settings
    unknown = sorted(set(settings) - {"sql"})
    if unknown:
        return [f"{scope.at(unknown[0])}: unknown setting of the sql check"]
    if item.name is None:
        return [f"{scope.at('name')}: a sql check needs a name"]
    if item.where is not None:
        return [f"{scope.at('where')}: put the condition in the query's WHERE clause"]
    if not isinstance(settings.get("sql"), str) or not settings["sql"].strip():
        return [f"{scope.at('sql')}: the sql check needs a 'sql' query"]
    return []


ROW_CHECKS = {"expression": expression_errors, "sql": sql_errors}
"""Checks evaluated by both backends, validated without the rule implementations."""


def name_errors(settings: QAVFSettings) -> list[str]:
    """Explicit item names repeated within one list, and unmatched ``run.require``."""
    problems = []
    seen: dict[str | None, set[str]] = {}
    for placed in placed_items(settings):
        name = placed.item.name
        if name is None:
            continue
        names = seen.setdefault(placed.scope.dataset, set())
        if name in names:
            problems.append(f"{placed.scope.at('name')}: {name!r} is already used in this list")
        names.add(name)
    known = {placed.name for placed in placed_items(settings)} | set(settings.datasets)
    problems.extend(
        f"run.require[{index}]: {entry!r} matches no check name or dataset"
        for index, entry in enumerate(settings.run.require)
        if entry not in known
    )
    for placed in placed_items(settings):
        if placed.scope.dataset is None:
            problems.extend(
                f"{placed.scope.at('datasets')}: unknown dataset {name!r}"
                for name in related_datasets(placed.item)
                if name not in settings.datasets
            )
    return problems


def row_check_errors(settings: QAVFSettings) -> list[str]:
    """Problems with ``expression``/``sql`` items, names and ``run.require``."""
    problems = []
    for placed in placed_items(settings):
        validate = ROW_CHECKS.get(placed.item.check)
        if validate is not None:
            problems.extend(validate(placed.item, placed.scope))
    return problems + name_errors(settings)


__all__ = [
    "ROW_CHECKS",
    "CheckSettingError",
    "PlacedItem",
    "Scope",
    "expression_errors",
    "name_errors",
    "placed_items",
    "related_datasets",
    "row_check_errors",
    "sql_errors",
]
