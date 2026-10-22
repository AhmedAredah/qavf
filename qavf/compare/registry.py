"""Named comparison components and their discovery from code and entry points.

Alignment steps, measures, comparators and levels each live in one
:class:`ComponentRegistry` reading a ``qavf.compare.*`` entry-point group. The
registry, its discovery transactions and the ``plugins.entry_points`` scope are
the shared machinery of :mod:`qavf.plugins.registry`, which statistical tests
(:mod:`qavf.inference`) use as well; this module keeps the comparison names.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any

from qavf.plugins import registry as _shared
from qavf.plugins.registry import (
    Component,
    entry_point_scope,
    entry_points_enabled,
    load_entry_points,
    plugin_status,
)


class ComponentRegistry[C: Component](_shared.ComponentRegistry[C]):
    """A comparison registry; its entry points are read through this module."""

    def entries(self) -> list[EntryPoint]:
        return list(entry_points(group=self.group))


def registries() -> dict[str, _shared.ComponentRegistry[Any]]:
    """Every comparison registry, keyed by its ``plugins.required`` qualifier."""
    from qavf.compare.align import ALIGNMENT_STEPS
    from qavf.compare.comparators import COMPARATORS
    from qavf.compare.levels import LEVELS
    from qavf.compare.measures import MEASURES

    return {
        "compare.alignment": ALIGNMENT_STEPS,
        "compare.comparators": COMPARATORS,
        "compare.levels": LEVELS,
        "compare.measures": MEASURES,
    }


__all__ = [
    "Component",
    "ComponentRegistry",
    "entry_point_scope",
    "entry_points_enabled",
    "load_entry_points",
    "plugin_status",
    "registries",
]
