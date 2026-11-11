"""Inference families: check items whose p-values are corrected together.

An item with ``inference: {family: <name>}`` belongs to that family. The run
corrects the tests of every member it runs as one family (one multiple-testing
correction over all of their p-values), whatever the items are named or how a
suite splits them. A selection (``--only``, ``--tags``, ``--dataset``) that runs
only some members corrects the family over those members alone, which results
and ``qavf explain`` disclose as ``selected N of M members`` (or refuses, with
``run.partial_families: refuse``).

Like :mod:`qavf.checks.selection`, this module imports no rule code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qavf.checks.placement import ROW_CHECKS, CheckSettingError, PlacedItem, placed_items

if TYPE_CHECKING:
    from qavf.checks.catalog import CheckDefinition
    from qavf.settings import QAVFSettings


@dataclass(frozen=True, slots=True)
class FamilyPlan:
    """The members of one inference family: those the run evaluates, and every member
    the configuration declares (enabled, or switched off only by the selection)."""

    name: str
    members: tuple[str, ...]
    """Instance ids of the members this run evaluates, in configuration order."""
    configured: tuple[str, ...]
    """Instance ids of every member of the configured suite."""

    @property
    def partial(self) -> bool:
        """Whether the selection left members out."""
        return len(self.members) < len(self.configured)

    @property
    def left_out(self) -> tuple[str, ...]:
        """Members the selection did not run."""
        running = set(self.members)
        return tuple(member for member in self.configured if member not in running)

    def selected(self) -> str:
        """``selected N of M members``."""
        return f"selected {len(self.members)} of {len(self.configured)} members"

    def describe(self) -> str:
        """One line for ``qavf explain``: the family, its members and the selection."""
        text = f"{self.name}: {', '.join(self.members) or 'no member runs'}"
        if self.partial:
            text += f" ({self.selected()}; not run: {', '.join(self.left_out)})"
        return text


class FamilyPlans:
    """The inference families of a configuration, by name, in order of first member."""

    def __init__(self, settings: QAVFSettings) -> None:
        selection = settings.selection
        self._not_selected = selection.not_selected if selection is not None else frozenset()
        self.plans = self._plans(placed_items(settings))

    def _plans(self, placed: Iterable[PlacedItem]) -> dict[str, FamilyPlan]:
        members: dict[str, list[str]] = {}
        configured: dict[str, list[str]] = {}
        for item in placed:
            inference = item.item.inference
            if inference is None:
                continue
            if item.item.enabled:
                members.setdefault(inference.family, []).append(item.path)
            elif item.path not in self._not_selected:
                continue
            configured.setdefault(inference.family, []).append(item.path)
        return {
            name: FamilyPlan(name, tuple(members.get(name, ())), tuple(paths))
            for name, paths in configured.items()
        }

    def family_of(self, path: str) -> FamilyPlan | None:
        """The family the item at ``path`` is a member of."""
        for plan in self.plans.values():
            if path in plan.configured:
                return plan
        return None

    def partial(self) -> list[FamilyPlan]:
        """Families the selection runs only some members of (at least one)."""
        return [plan for plan in self.plans.values() if plan.partial and plan.members]


@dataclass(frozen=True, slots=True)
class _Member:
    """A family member as configured: where it is and the corrections it applies."""

    placed: PlacedItem
    corrections: frozenset[str]

    @property
    def at(self) -> str:
        return self.placed.scope.at("inference.family")


class FamilyValidation:
    """Load-time problems with ``inference``: an item whose check runs no correction,
    and members of one family that do not agree on a single correction."""

    def __init__(self, settings: QAVFSettings, definitions: Mapping[str, CheckDefinition]) -> None:
        self.settings = settings
        self.definitions = definitions
        selection = settings.selection
        self._not_selected = selection.not_selected if selection is not None else frozenset()

    def errors(self) -> list[str]:
        """Each problem as ``path: message``."""
        problems: list[str] = []
        families: dict[str, list[_Member]] = {}
        for placed in placed_items(self.settings):
            inference = placed.item.inference
            definition = self.definitions.get(placed.item.check)
            if inference is None or definition is None or placed.item.check in ROW_CHECKS:
                continue
            if not definition.supports_inference:
                problems.append(f"{placed.scope.at('inference')}: {self._unsupported(placed)}")
                continue
            if not placed.item.enabled and placed.path not in self._not_selected:
                continue
            corrections = self._corrections(placed, definition)
            if corrections is not None:
                families.setdefault(inference.family, []).append(_Member(placed, corrections))
        for name, members in families.items():
            problems.extend(self._family_errors(name, members))
        return problems

    def _unsupported(self, placed: PlacedItem) -> str:
        supported = sorted(name for name, d in self.definitions.items() if d.supports_inference)
        return (
            f"the {placed.item.check} check runs no multiple-testing correction, so it "
            f"cannot join an inference family; 'inference' is taken by {', '.join(supported)}"
        )

    @staticmethod
    def _corrections(placed: PlacedItem, definition: CheckDefinition) -> frozenset[str] | None:
        """The corrections the item applies; ``None`` when its settings are invalid
        (reported where the item is validated)."""
        try:
            params = definition.compile_item(placed.item, placed.scope)
            rule = definition.rule_cls(params=params)
        except (CheckSettingError, ValueError, TypeError):
            return None
        corrections: frozenset[str] = rule.family_corrections()  # type: ignore[attr-defined]
        return corrections

    @staticmethod
    def _family_errors(name: str, members: list[_Member]) -> list[str]:
        problems = []
        agreed: tuple[str, _Member] | None = None
        for member in members:
            corrections = sorted(member.corrections)
            if not corrections:
                problems.append(
                    f"{member.at}: inference family {name!r}: this item runs no significance "
                    "test (every level decides by materiality alone)"
                )
            elif len(corrections) > 1:
                problems.append(
                    f"{member.at}: inference family {name!r}: this item applies the "
                    f"corrections {' and '.join(corrections)}; a member applies one"
                )
            elif corrections[0] == "none":
                problems.append(
                    f"{member.at}: inference family {name!r}: this item's correction is "
                    "'none', which corrects nothing; set the correction the family applies "
                    "(bonferroni, holm, bh or by)"
                )
            elif agreed is None:
                agreed = (corrections[0], member)
            elif corrections[0] != agreed[0]:
                problems.append(
                    f"{member.at}: inference family {name!r} is corrected with {agreed[0]} "
                    f"({agreed[1].placed.path}), but this item applies {corrections[0]}; "
                    "the members of a family must agree on one correction"
                )
        return problems


__all__ = ["FamilyPlan", "FamilyPlans", "FamilyValidation"]
