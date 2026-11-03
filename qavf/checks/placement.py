"""Where check items are placed, their names, and checks that need no rule code.

Everything here works without the optional Polars backend: the native DuckDB
backend validates ``expression`` and ``sql`` items with it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from qavf.checks.items import CheckItem, dataset_item_id, top_level_item_id
from qavf.checks.reuse import default_problems

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings


class CheckSettingError(ValueError):
    """A check item setting that cannot be turned into rule parameters."""

    def __init__(self, field: str | None, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


BLOCK_SECTIONS = {"series": "series", "metric": "metrics"}
"""The dataset section holding the named blocks a check setting refers to."""


@dataclass(frozen=True, slots=True)
class NamedBlocks:
    """The named ``series``, ``metrics`` and ``views`` every dataset defines."""

    blocks: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]] = field(
        default_factory=dict
    )
    """Section (``series``/``metrics``) -> dataset -> name -> definition."""
    views: Mapping[str, Any] = field(default_factory=dict)
    """Qualified view name (``counts.recent``) -> its definition."""
    datasets: frozenset[str] = frozenset()

    @classmethod
    def of(cls, settings: QAVFSettings) -> NamedBlocks:
        blocks = {
            "series": {name: spec.series for name, spec in settings.datasets.items()},
            "metrics": {name: spec.metrics for name, spec in settings.datasets.items()},
        }
        views = {name: spec for name, (_, spec) in settings.views().items()}
        return cls(blocks, views, frozenset(settings.datasets))

    def resolve(self, setting: str, reference: str, dataset: str | None) -> dict[str, Any]:
        """The definition ``reference`` names, looked up on ``dataset`` first.

        ``other.name`` reaches the block of dataset ``other``.

        Raises:
            CheckSettingError: when nothing of that name is defined.
        """
        section = BLOCK_SECTIONS[setting]
        defined = self.blocks.get(section, {})
        own = defined.get(dataset or "", {})
        if reference in own:
            return dict(own[reference])
        owner, _, name = reference.partition(".")
        if name and name in defined.get(owner, {}):
            return dict(defined[owner][name])
        if name and owner not in self.datasets:
            message = f"unknown dataset {owner!r} in {setting} {reference!r}"
        elif name:
            message = f"dataset {owner!r} defines no {setting} {name!r}"
        else:
            names = ", ".join(sorted(own)) or "none"
            message = (
                f"unknown {setting} {reference!r}; dataset {dataset!r} defines {section}: {names}"
            )
        raise CheckSettingError(setting, message)

    def view_parent(self, name: str) -> str | None:
        """The dataset a qualified view name belongs to; ``None`` if it names no view."""
        return name.partition(".")[0] if name in self.views else None

    def known_dataset(self, name: str) -> bool:
        """Whether ``name`` is a dataset or a view (``counts.recent``)."""
        return name in self.datasets or name in self.views


@dataclass(frozen=True, slots=True)
class Scope:
    """Where a check item is placed."""

    path: str
    """The item's location, e.g. ``datasets.counts.checks[2]``."""
    dataset: str | None = None
    """The dataset a dataset-level item tests; ``None`` at the top level."""
    blocks: NamedBlocks = field(default_factory=NamedBlocks, compare=False)
    """The named series, metrics and views the item may refer to."""

    def at(self, field: str | None) -> str:
        """The path of one of the item's settings."""
        return f"{self.path}.{field}" if field else self.path

    def resolve(self, setting: str, reference: str) -> dict[str, Any]:
        """The named ``series``/``metric`` a setting refers to (see :class:`NamedBlocks`)."""
        return self.blocks.resolve(setting, reference, self.dataset)


@dataclass(frozen=True, slots=True)
class PlacedItem:
    """A check item and where it is placed."""

    item: CheckItem
    scope: Scope

    @property
    def path(self) -> str:
        """The item's location, which is also its instance id."""
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
    blocks = NamedBlocks.of(settings)
    for dataset, spec in settings.datasets.items():
        for index, item in enumerate(spec.checks):
            yield PlacedItem(item, Scope(dataset_item_id(dataset, index), dataset, blocks))
    for index, item in enumerate(settings.checks):
        yield PlacedItem(item, Scope(top_level_item_id(index), None, blocks))


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

ROW_CHECK_SETTINGS: dict[str, tuple[str, ...]] = {
    "expression": ("expr", "null_as"),
    "sql": ("sql",),
}
"""The own settings of the ``expression`` and ``sql`` checks."""


def list_name_errors(items: Iterable[PlacedItem]) -> list[str]:
    """Explicit names in one ``checks`` list that repeat another item's name or
    default name. Unnamed items may share a default name: results are told apart
    by instance id."""
    placed = list(items)
    defaults = {item.name: item.path for item in reversed(placed) if item.item.name is None}
    problems = []
    named: set[str] = set()
    for item in placed:
        name = item.item.name
        if name is None:
            continue
        if name in named:
            problems.append(f"{item.scope.at('name')}: {name!r} is already used in this list")
        elif name in defaults:
            problems.append(
                f"{item.scope.at('name')}: {name!r} is the default name of {defaults[name]}; "
                "give one of them another name"
            )
        named.add(name)
    return problems


def name_errors(settings: QAVFSettings) -> list[str]:
    """Item names repeated within one list, item names that are dataset names, and
    unmatched ``run.require``."""
    problems = []
    lists: dict[str | None, list[PlacedItem]] = {}
    for placed in placed_items(settings):
        lists.setdefault(placed.scope.dataset, []).append(placed)
        if placed.item.name is not None and placed.item.name in settings.datasets:
            problems.append(
                f"{placed.scope.at('name')}: {placed.item.name!r} is the name of a dataset; "
                "a check needs a name of its own so run.require and --only are unambiguous"
            )
    for items in lists.values():
        problems.extend(list_name_errors(items))
    blocks = NamedBlocks.of(settings)
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
                if not blocks.known_dataset(name)
            )
    return problems


def row_check_errors(settings: QAVFSettings) -> list[str]:
    """Problems with ``expression``/``sql`` items, names and ``run.require``."""
    problems = []
    for placed in placed_items(settings):
        validate = ROW_CHECKS.get(placed.item.check)
        if validate is not None:
            problems.extend(validate(placed.item, placed.scope))
    problems.extend(default_problems(settings.defaults, ROW_CHECK_SETTINGS))
    return problems + name_errors(settings)


__all__ = [
    "BLOCK_SECTIONS",
    "ROW_CHECKS",
    "ROW_CHECK_SETTINGS",
    "CheckSettingError",
    "NamedBlocks",
    "PlacedItem",
    "Scope",
    "expression_errors",
    "list_name_errors",
    "name_errors",
    "placed_items",
    "related_datasets",
    "row_check_errors",
    "sql_errors",
]
