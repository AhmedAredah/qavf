"""Run-time selection of checks: ``--only``, ``--tags``, ``--skip-tags`` and ``--dataset``.

A selection narrows a configuration without editing it. Check items it leaves
out are switched off for the run (and reported as not selected); schema
contracts carry the implicit tag ``input`` and run only for the selected
datasets. Built-in checks carry their family (``temporal``, ``spatial``, …) as
an implicit tag, so ``--tags temporal`` works without configuration.

Like :mod:`qavf.checks.placement`, this module imports no rule code, so the
native DuckDB backend can apply a selection.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qavf.checks.across import ALL_COPIES, COPY_SEPARATOR, expand_dataset_names
from qavf.checks.placement import ROW_CHECKS, PlacedItem, placed_items

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings

CONTRACT_TAG = "input"
"""The implicit tag of schema contracts (their family)."""

NOT_SELECTED = "not selected for this run"


class SelectionError(ValueError):
    """A selection that names unknown checks or datasets, or selects nothing."""


def builtin_families() -> dict[str, str]:
    """The family (implicit tag) of every built-in check, by check name.

    Without the optional Polars backend only the row checks are known.
    """
    families = dict.fromkeys(ROW_CHECKS, CONTRACT_TAG)
    try:
        from qavf.checks.catalog import CATALOG
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("polars"):
            raise
        return families
    return families | {name: d.rule_cls.category for name, d in CATALOG.items()}


def _words(value: str | Iterable[str] | None, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    words = (value,) if isinstance(value, str) else tuple(value)
    for word in words:
        if not isinstance(word, str) or not word.strip():
            raise SelectionError(f"{what} must be nonempty names, got {word!r}")
    return words


@dataclass(frozen=True, slots=True)
class Selection:
    """Which configured checks a run evaluates; empty selects everything."""

    only: tuple[str, ...] = ()
    """Item names; when given, only these items run."""
    tags: tuple[str, ...] = ()
    """When given, only checks carrying one of these tags run."""
    skip_tags: tuple[str, ...] = ()
    """Checks carrying one of these tags do not run."""
    datasets: tuple[str, ...] = ()
    """When given, only checks on these datasets, and their schema contracts, run."""

    @classmethod
    def of(
        cls,
        only: str | Iterable[str] | None = None,
        tags: str | Iterable[str] | None = None,
        skip_tags: str | Iterable[str] | None = None,
        datasets: str | Iterable[str] | None = None,
    ) -> Selection:
        """A selection from API arguments (a single name or a list of names each)."""
        return cls(
            _words(only, "only"),
            _words(tags, "tags"),
            _words(skip_tags, "skip_tags"),
            _words(datasets, "datasets"),
        )

    @property
    def active(self) -> bool:
        return bool(self.only or self.tags or self.skip_tags or self.datasets)

    def describe(self) -> str:
        """The selection as command options, e.g. ``--tags nightly --dataset counts``."""
        parts = [
            f"{option} {name!r}" if " " in name else f"{option} {name}"
            for option, names in (
                ("--only", self.only),
                ("--tags", self.tags),
                ("--skip-tags", self.skip_tags),
                ("--dataset", self.datasets),
            )
            for name in names
        ]
        return " ".join(parts)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "only": list(self.only),
            "tags": list(self.tags),
            "skip_tags": list(self.skip_tags),
            "datasets": list(self.datasets),
        }

    def _tagged(self, tags: frozenset[str]) -> bool:
        if self.tags and not tags & set(self.tags):
            return False
        return not tags & set(self.skip_tags)

    def selects(self, placed: PlacedItem, family: str | None) -> bool:
        """Whether the item runs under this selection."""
        if self.only and placed.name not in self.only:
            return False
        tags = frozenset(placed.item.tags) | ({family} if family else frozenset())
        if not self._tagged(tags):
            return False
        if self.datasets:
            datasets = placed.datasets
            if not datasets:
                return False  # a SQL check may read any dataset
            blocks = placed.scope.blocks
            return all((blocks.view_parent(name) or name) in self.datasets for name in datasets)
        return True

    @property
    def contracts(self) -> bool:
        """Whether schema contracts run (``--only`` selects check items alone)."""
        return not self.only and self._tagged(frozenset({CONTRACT_TAG}))

    def contract_runs(self, dataset: str | None) -> bool:
        """Whether schema contracts run for ``dataset`` (``None``: run-wide contracts)."""
        if not self.contracts:
            return False
        return dataset is None or not self.datasets or dataset in self.datasets


@dataclass(frozen=True, slots=True)
class AppliedSelection:
    """The selection a narrowed configuration runs under."""

    selection: Selection
    not_selected: frozenset[str] = field(default_factory=frozenset)
    """Locations (``datasets.counts.checks[2]``) of the items the selection switched off."""

    def contract_runs(self, dataset: str | None) -> bool:
        return self.selection.contract_runs(dataset)

    def contract_datasets(self) -> list[str] | None:
        """The datasets schema contracts are restricted to (``None``: every dataset)."""
        return list(self.selection.datasets) if self.selection.datasets else None


def narrow(
    settings: QAVFSettings, selection: Selection, families: Mapping[str, str] | None = None
) -> QAVFSettings:
    """A copy of ``settings`` running only what ``selection`` selects.

    Items left out are switched off; ``run.require`` keeps only selected entries.
    The copy carries the selection (:attr:`QAVFSettings.selection`) so the run restricts
    schema contracts and reports why items did not run.

    Raises:
        SelectionError: for an unknown item name or dataset, or when nothing is selected.
    """
    if not selection.active:
        return settings
    if families is None:
        families = builtin_families()
    if any(COPY_SEPARATOR + ALL_COPIES in name for name in selection.datasets):
        try:
            copies = expand_dataset_names(selection.datasets, settings.datasets, "--dataset")
        except ValueError as exc:
            raise SelectionError(str(exc).removeprefix("--dataset: ")) from None
        selection = dataclasses.replace(selection, datasets=tuple(copies))
    unknown = [name for name in selection.datasets if name not in settings.datasets]
    if unknown:
        known = ", ".join(settings.datasets) or "none"
        raise SelectionError(f"unknown dataset {unknown[0]!r} selected (datasets: {known})")
    placed = list(placed_items(settings))
    names = {item.name for item in placed}
    missing = [name for name in selection.only if name not in names]
    if missing:
        shown = ", ".join(repr(name) for name in sorted(names)[:10]) or "none"
        raise SelectionError(f"no check is named {missing[0]!r} (names: {shown})")
    copied = settings.model_copy(deep=True)
    copied_items = list(placed_items(copied))
    selected: set[str] = set()
    switched_off: set[str] = set()
    for original, item in zip(placed, copied_items, strict=True):
        if not original.item.enabled:
            continue
        if selection.selects(original, families.get(original.item.check)):
            selected.add(original.name)
        else:
            item.item.enabled = False
            switched_off.add(original.path)
    contracts = selection.contracts and bool(settings.datasets)
    if not selected and not contracts:
        raise SelectionError(f"the selection {selection.describe()} matches no check")
    selected_datasets = set(selection.datasets or settings.datasets) if contracts else set()
    copied.run.require = [
        entry
        for entry in settings.run.require
        if entry in selected or (entry in settings.datasets and entry in selected_datasets)
    ]
    copied.apply_selection(AppliedSelection(selection, frozenset(switched_off)))
    copied.adopt_origins(settings.origins)
    if settings.run.partial_families == "refuse":
        _refuse_partial_families(copied, selection)
    return copied


def _refuse_partial_families(settings: QAVFSettings, selection: Selection) -> None:
    """Raise for a selection running only some members of an inference family.

    Raises:
        SelectionError: under ``run.partial_families: refuse``.
    """
    from qavf.checks.families import FamilyPlans

    partial = FamilyPlans(settings).partial()
    if partial:
        family = partial[0]
        raise SelectionError(
            f"the selection {selection.describe()} {family.selected()} of inference family "
            f"{family.name!r} (not run: {', '.join(family.left_out)}), and "
            "run.partial_families is refuse; select every member or none"
        )


class ItemTags:
    """The explicit tags of every check item, by instance id (never by name, which
    two items may share)."""

    def __init__(self, settings: QAVFSettings) -> None:
        self.tags = {
            placed.path: tuple(placed.item.tags)
            for placed in placed_items(settings)
            if placed.item.tags
        }

    def of(self, item_ids: Sequence[str]) -> list[str]:
        """The tags of the one item ``item_ids`` names; none for a summary of several."""
        if len(item_ids) != 1:
            return []
        return list(self.tags.get(item_ids[0], ()))


__all__ = [
    "CONTRACT_TAG",
    "NOT_SELECTED",
    "AppliedSelection",
    "ItemTags",
    "Selection",
    "SelectionError",
    "builtin_families",
    "narrow",
]
