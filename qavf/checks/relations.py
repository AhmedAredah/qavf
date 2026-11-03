"""One namespace for every relation a run reads by name.

Datasets (``across`` copies included), their views (``<dataset>.<view>``) and the
datasets checks derive (``consensus~checks[2]``) are all tables of one SQL
namespace, where names are case-insensitive. Dataset names may contain dots, so
a view ``roads.active`` and a dataset ``roads.active`` would be one table: such
collisions are rejected when the configuration is loaded, naming both
declarations, instead of one relation silently replacing the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings


@dataclass(frozen=True, slots=True)
class Relation:
    """A named relation and where the configuration declares it."""

    kind: str
    """``dataset``, ``view`` or ``derived dataset``."""
    name: str
    origin: str
    """The declaration's location, e.g. ``datasets.roads.views.active``."""

    def describe(self) -> str:
        return f"{self.kind} {self.name!r} ({self.origin})"


class RelationNamespace:
    """The relations of a configuration, each name (ignoring case) declared once."""

    def __init__(self) -> None:
        self._relations: dict[str, Relation] = {}

    @classmethod
    def of(cls, settings: QAVFSettings) -> RelationNamespace:
        """Every dataset, view and derived dataset of ``settings``.

        Raises:
            ValueError: when two of them have one name, exactly or ignoring case.
        """
        namespace = cls()
        for name in settings.datasets:
            namespace.declare(Relation("dataset", name, f"datasets.{name}"))
        for name, (parent, _) in settings.views().items():
            view = name.removeprefix(f"{parent}.")
            namespace.declare(Relation("view", name, f"datasets.{parent}.views.{view}"))
        for name in settings.derived_datasets():
            origin = name.rpartition("~")[2]
            namespace.declare(Relation("derived dataset", name, origin))
        return namespace

    def declare(self, relation: Relation) -> None:
        """Add ``relation``.

        Raises:
            ValueError: when a relation of that name (ignoring case) is declared.
        """
        other = self._relations.setdefault(relation.name.lower(), relation)
        if other is relation:
            return
        same = "the same name" if other.name == relation.name else "names differing only in case"
        raise ValueError(
            f"{other.describe()} and {relation.describe()} have {same}; SQL reads both as "
            "one table, so one would silently replace the other. Rename one."
        )

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._relations


__all__ = ["Relation", "RelationNamespace"]
