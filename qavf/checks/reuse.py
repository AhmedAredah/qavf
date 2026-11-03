"""Check-item reuse applied before validation: ``defaults`` and value ranges.

* ``defaults: {change_rate: {max: 0.15}}`` gives every ``change_rate`` item the
  settings it omits; an item's own settings win.
* A check setting written as ``{from: 2016, to: 2024, step: 1}`` or as the
  string ``2016..2024`` expands to every value of that range. A plain list such
  as ``[2016, 2024]`` always means just those values.

Like :mod:`qavf.checks.items`, this module imports no rule code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from qavf.checks.items import COMMON_FIELDS

DEFAULTABLE_COMMON_FIELDS = frozenset({"severity", "where", "tags", "enabled"})
"""Common item fields ``defaults`` may set; ``check``, ``name`` and ``description``
identify one item and have no default."""

MAX_RANGE_VALUES = 100_000
"""Values a range may expand to."""

_RANGE_TEXT = re.compile(r"\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*")
_RANGE_KEYS = (frozenset({"from", "to"}), frozenset({"from", "to", "step"}))


class ValueRange:
    """An inclusive integer range: ``{from, to, step}`` or ``"from..to"``."""

    def __init__(self, start: Any, stop: Any, step: Any = 1) -> None:
        for label, value in (("from", start), ("to", stop), ("step", step)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"range {label} must be a whole number, got {value!r}")
        if step <= 0:
            raise ValueError(f"range step must be positive, got {step}")
        if stop < start:
            raise ValueError(f"range from {start} to {stop} is empty; 'from' must not exceed 'to'")
        count = (stop - start) // step + 1
        if count > MAX_RANGE_VALUES:
            raise ValueError(
                f"range from {start} to {stop} has {count:,} values, more than {MAX_RANGE_VALUES:,}"
            )
        self.start, self.stop, self.step = start, stop, step

    @classmethod
    def parse(cls, value: Any) -> ValueRange | None:
        """The range ``value`` spells, or ``None`` when it is not a range."""
        if isinstance(value, str):
            match = _RANGE_TEXT.fullmatch(value)
            return cls(int(match.group(1)), int(match.group(2))) if match else None
        if isinstance(value, Mapping) and frozenset(value) in _RANGE_KEYS:
            return cls(value["from"], value["to"], value.get("step", 1))
        return None

    def values(self) -> list[int]:
        return list(range(self.start, self.stop + 1, self.step))


class ItemPreparation:
    """Applies ``defaults`` and expands ranges in the check items of a raw configuration."""

    def __init__(self, defaults: Any) -> None:
        self.defaults: Mapping[str, Any] = defaults if isinstance(defaults, Mapping) else {}

    @classmethod
    def apply(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """``raw`` with prepared copies of its check items (validated items are kept)."""
        preparation = cls(raw.get("defaults"))
        result = dict(raw)
        datasets = raw.get("datasets")
        if isinstance(datasets, Mapping):
            result["datasets"] = {
                name: preparation._dataset(name, spec) for name, spec in datasets.items()
            }
        if isinstance(raw.get("checks"), list):
            result["checks"] = preparation._items(raw["checks"], "checks")
        return result

    def _dataset(self, name: str, spec: Any) -> Any:
        if not isinstance(spec, Mapping) or not isinstance(spec.get("checks"), list):
            return spec
        return {**spec, "checks": self._items(spec["checks"], f"datasets.{name}.checks")}

    def _items(self, items: list[Any], path: str) -> list[Any]:
        return [self._item(item, f"{path}[{index}]") for index, item in enumerate(items)]

    def _item(self, item: Any, path: str) -> Any:
        if not isinstance(item, Mapping):
            return item  # a validated item, or an invalid one the model reports
        prepared = dict(item)
        check = item.get("check")
        defaults = self.defaults.get(check) if isinstance(check, str) else None
        if isinstance(defaults, Mapping):
            for key, value in defaults.items():
                if key in DEFAULTABLE_COMMON_FIELDS or key not in COMMON_FIELDS:
                    prepared.setdefault(key, value)
        for key, value in prepared.items():
            if key in COMMON_FIELDS:
                continue
            try:
                expanded = ValueRange.parse(value)
            except ValueError as exc:
                raise ValueError(f"{path}.{key}: {exc}") from None
            if expanded is not None:
                prepared[key] = expanded.values()
        return prepared


def default_problems(
    defaults: Mapping[str, Mapping[str, Any]], settings_of: Mapping[str, Iterable[str]]
) -> list[str]:
    """Problems with ``defaults``: keys that are not settings of their check.

    ``settings_of`` maps each known check name to its setting names; checks it
    does not know are left for the run's registry to judge.
    """
    problems: list[str] = []
    for check, values in defaults.items():
        known = settings_of.get(check)
        if known is None:
            continue
        allowed = set(known) | DEFAULTABLE_COMMON_FIELDS
        problems.extend(
            f"defaults.{check}.{key}: not a setting of the {check} check"
            for key in values
            if key not in allowed
        )
    return problems


__all__ = [
    "DEFAULTABLE_COMMON_FIELDS",
    "MAX_RANGE_VALUES",
    "ItemPreparation",
    "ValueRange",
    "default_problems",
]
