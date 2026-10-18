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
    rule_id: str
    requirements: DataRequirements
    mode: str


class ExecutionPlan:
    """Thread-safe reference counts released only after actual rule completion."""

    def __init__(self, dependencies: Sequence[_RuleDependencies]) -> None:
        self._rules = {entry.rule_id: entry for entry in dependencies}
        if len(self._rules) != len(dependencies):
            raise ValueError("an execution plan cannot repeat rule IDs")
        self._remaining: Counter[str] = Counter(
            name for entry in dependencies for name in entry.requirements.datasets
        )
        self._completed: set[str] = set()
        self._lock = threading.Lock()

    def complete(self, rule_id: str) -> set[str]:
        """Datasets whose last planned consumer just stopped executing.

        Calling this twice is harmless. A timeout notification is not completion
        while its worker continues running; callers must release in the actual
        worker's ``finally`` block, not the reporting/deadline thread.
        """
        with self._lock:
            if rule_id in self._completed:
                return set()
            entry = self._rules[rule_id]
            self._completed.add(rule_id)
            released = set()
            for name in entry.requirements.datasets:
                self._remaining[name] -= 1
                if self._remaining[name] == 0:
                    released.add(name)
            return released

    def describe(self) -> list[dict[str, Any]]:
        """Stable, data-free dependency details suitable for a run report."""
        return [
            {
                "rule_id": entry.rule_id,
                "datasets": sorted(entry.requirements.datasets),
                "dependency_mode": entry.mode,
                "columns": {name: list(cols) for name, cols in entry.requirements.columns.items()},
            }
            for entry in self._rules.values()
        ]


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
        dependencies.append(_RuleDependencies(rule.rule_id, requirements, mode))
    return ExecutionPlan(dependencies)


__all__ = ["DataRequirements", "ExecutionPlan", "build_plan"]
