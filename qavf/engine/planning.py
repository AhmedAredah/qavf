"""Conservative dataset lifetimes for configured rules.

This is a dependency ledger, not a replacement query optimizer. Native engines
still optimize scans. An undeclared plugin retains every dataset until it has
actually finished; uncertainty must never cause an input to be released early.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qavf.engine.derived_names import RowFilter
    from qavf.engine.registry import Rule
    from qavf.settings import QAVFSettings


@dataclass(frozen=True, slots=True)
class DataRequirements:
    """Complete dataset dependencies explicitly declared by a rule author.

    ``columns`` contains optional complete column declarations for inspection;
    omitted datasets mean all/unknown columns. These hints do not authorize
    projecting away columns from other checks. Rules must declare every dataset
    they may access, including datasets reached through SQL or helper calls.
    """

    datasets: frozenset[str]
    columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.datasets, str):
            raise ValueError("data requirements need a collection of dataset names")
        names = frozenset(self.datasets)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("data requirements need nonempty dataset names")
        if set(self.columns) - names:
            raise ValueError("column requirements must refer to declared datasets")
        columns = {}
        for name, values in self.columns.items():
            if isinstance(values, str):
                raise ValueError("column requirements must be sequences of nonempty names")
            collected = tuple(values)
            if any(not isinstance(v, str) or not v for v in collected):
                raise ValueError("column requirements must be sequences of nonempty names")
            columns[name] = tuple(dict.fromkeys(collected))
        object.__setattr__(self, "datasets", names)
        object.__setattr__(self, "columns", MappingProxyType(columns))


@dataclass(frozen=True, slots=True)
class _RuleDependencies:
    key: str
    """The rule instance's key: its check item, or its rule id."""
    rule_id: str
    requirements: DataRequirements
    mode: str
    rows: RowFilter | None = None
    """The rows a filtered check item reads through a hidden view of its dataset."""
    where_columns: tuple[str, ...] | None = None
    """The dataset columns the item's ``where`` reads (``None``: not proven)."""

    def describe(self) -> dict[str, Any]:
        """Data-free details; a hidden view is shown as its dataset (whose columns
        include the view's) with the item's ``where``."""
        datasets = sorted(self.requirements.datasets)
        columns = {name: list(cols) for name, cols in self.requirements.columns.items()}
        described: dict[str, Any] = {"rule_id": self.rule_id}
        if self.rows is not None:
            columns.pop(self.rows.view, None)
            datasets = sorted(self.rows.shown_datasets(datasets))
        described |= {"datasets": datasets, "dependency_mode": self.mode, "columns": columns}
        if self.rows is not None:
            described["where"] = self.rows.where
            described["where_columns"] = (
                list(self.where_columns) if self.where_columns is not None else None
            )
        return described


class ExecutionPlan:
    """Thread-safe reference counts released only after actual rule completion."""

    def __init__(self, dependencies: Sequence[_RuleDependencies]) -> None:
        self._rules = {entry.key: entry for entry in dependencies}
        if len(self._rules) != len(dependencies):
            raise ValueError("an execution plan cannot repeat rule instances")
        self._remaining: Counter[str] = Counter(
            name for entry in dependencies for name in entry.requirements.datasets
        )
        self._completed: set[str] = set()
        self._lock = threading.Lock()

    def complete(self, key: str) -> set[str]:
        """Datasets whose last planned consumer just stopped executing.

        Calling this twice is harmless. A timeout notification is not completion
        while its worker continues running; callers must release in the actual
        worker's ``finally`` block, not the reporting/deadline thread.
        """
        with self._lock:
            if key in self._completed:
                return set()
            entry = self._rules[key]
            self._completed.add(key)
            released = set()
            for name in entry.requirements.datasets:
                self._remaining[name] -= 1
                if self._remaining[name] == 0:
                    released.add(name)
            return released

    def describe(self) -> list[dict[str, Any]]:
        """Stable, data-free dependency details suitable for a run report."""
        return [entry.describe() for entry in self._rules.values()]


def _ancestors(datasets: Iterable[str], parents: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """Every dataset ``datasets`` are read from, directly or through other derived ones."""
    found: set[str] = set()
    pending = [parent for name in datasets for parent in parents.get(name, ())]
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            pending.extend(parents.get(name, ()))
    return frozenset(found)


class FilteredReads:
    """The columns of a dataset a filtered check item reads through its hidden view:
    those the rule declares for the view, and those the item's ``where`` reads."""

    def __init__(self, settings: QAVFSettings) -> None:
        from qavf.sql_dependencies import PredicateDependencies

        hidden = settings.where_views()
        self.views = {name: (parent, spec.where) for name, (parent, spec, _) in hidden.items()}
        self.reads = PredicateDependencies(settings.datasets)

    def columns(self, requirements: DataRequirements) -> dict[str, tuple[str, ...]]:
        """``requirements.columns`` with each hidden view's parent dataset's columns.

        A parent the rule declares itself, a view without declared columns, or a
        condition whose reads are not proven leaves the parent's columns unknown.
        """
        columns = dict(requirements.columns)
        for name in sorted(requirements.datasets & self.views.keys()):
            parent, where = self.views[name]
            declared = requirements.columns.get(name)
            if declared is None or parent in requirements.datasets or where is None:
                continue
            reads = self.where_columns(parent, where)
            if reads is not None:
                read = (*columns.get(parent, ()), *declared, *reads)
                columns[parent] = tuple(dict.fromkeys(read))
        return columns

    def where_columns(self, dataset: str, where: str) -> tuple[str, ...] | None:
        """The columns of ``dataset`` that ``where`` reads (``None``: not proven)."""
        reads = self.reads.of(where, dataset).columns
        return tuple(sorted(reads)) if reads is not None else None


def build_plan(
    rules: Sequence[Rule], settings: QAVFSettings, dataset_names: Iterable[str]
) -> ExecutionPlan:
    """Plan safe dataset lifetimes without reading input data.

    Complete declarations belong to each rule. Input-rule schemas, foreign
    keys and arbitrary SQL may cross datasets, so undeclared scope remains
    conservative. Unknown plugins retain all declared datasets; ``requires``
    alone is an availability condition, not a complete dependency declaration.
    """
    universe = frozenset(dataset_names)
    parents = settings.dataset_parents()
    filtered = FilteredReads(settings)
    dependencies = []
    for rule in rules:
        explicit = rule.data_requirements(settings)
        if explicit is not None:
            if not isinstance(explicit, DataRequirements):
                raise TypeError(f"{rule.rule_id}.data_requirements must return DataRequirements")
            requirements = DataRequirements(
                explicit.datasets | frozenset(rule.requires), explicit.columns
            )
            mode = "declared"
        else:
            requirements = DataRequirements(universe | frozenset(rule.requires))
            mode = "opaque"
        viewed = _ancestors(requirements.datasets, parents)
        if viewed:
            # A view (or a derived dataset) is read from its parents, which must stay
            # loaded while it is used.
            requirements = DataRequirements(
                requirements.datasets | viewed, filtered.columns(requirements)
            )
        rows = rule.binding.rows if rule.binding is not None else None
        reads = filtered.where_columns(rows.dataset, rows.where) if rows is not None else None
        dependencies.append(
            _RuleDependencies(rule.key, rule.rule_id, requirements, mode, rows, reads)
        )
    return ExecutionPlan(dependencies)


__all__ = ["DataRequirements", "ExecutionPlan", "FilteredReads", "build_plan"]
