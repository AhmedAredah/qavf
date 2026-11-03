"""Check-item reuse applied before validation: ``defaults`` and value ranges.

* ``defaults: {change_rate: {max: 0.15}}`` gives every ``change_rate`` item the
  settings it omits; an item's own settings win.
* A check setting declared as a list, written as ``{from: 2016, to: 2024, step: 1}``
  or as the string ``2016..2024``, expands to every value of that range, also in
  nested settings (``key_grid``'s ``dimensions.<column>.values``). A plain list
  such as ``[2016, 2024]`` always means just those values. Settings of other types
  (a column name, a threshold, a plugin's setting) keep the value as written.

Like :mod:`qavf.checks.items`, this module imports no rule code; the check catalog
that declares each setting's type is imported when items are prepared.
"""

from __future__ import annotations

import re
import types
import typing
from collections.abc import Iterable, Mapping
from typing import Annotated, Any

from pydantic import BaseModel

from qavf.checks.items import COMMON_FIELDS, CheckItem

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


_SEQUENCES = (list, tuple, set, frozenset)


class TypedRanges:
    """Expands ranges in a raw check item only where its model declares a sequence.

    The walk follows the item model's fields (and a model's ``path_parts`` hook,
    as :mod:`qavf.config_paths` does), into nested models, lists of models and
    mappings of models. A value is expanded when its field accepts a sequence and
    no other accepted type (a string, a mapping, ``Any``) takes the value as written.
    A check the catalog does not know (a plugin's) keeps its settings as written.
    """

    def __init__(self) -> None:
        self._memo: dict[tuple[int, type[BaseModel]], Any] = {}
        self._kept: list[Any] = []  # keeps memoised originals alive, so ids stay unique

    def item(self, item: dict[str, Any], path: str) -> dict[str, Any]:
        """``item`` with the ranges of its own (non-common) settings expanded."""
        result = dict(item)
        for _, model in item_parts(item):
            fields = _field_index(model)
            for key, value in item.items():
                field = fields.get(key) if isinstance(key, str) else None
                if field is not None and key not in COMMON_FIELDS:
                    result[key] = self.value(value, field.annotation, f"{path}.{key}")
        return result

    def model(self, raw: Any, model: type[BaseModel], path: str) -> Any:
        """``raw`` (the input of ``model``) with its sequence ranges expanded."""
        if not isinstance(raw, Mapping):
            return raw
        key = (id(raw), model)
        if key in self._memo:
            return self._memo[key]
        result = dict(raw)
        self._memo[key] = result
        self._kept.append(raw)
        for part, described in _parts(model, raw):
            if part is not None:
                result[part] = self.model(result.get(part), described, f"{path}.{part}")
                continue
            fields = _field_index(described)
            for name, value in list(result.items()):
                field = fields.get(name) if isinstance(name, str) else None
                if field is not None:
                    result[name] = self.value(value, field.annotation, f"{path}.{name}")
        return result

    def value(self, value: Any, annotation: Any, path: str) -> Any:
        """``value`` of a field typed ``annotation``, with its ranges expanded."""
        arms = _arms(annotation)
        if any(_is_sequence(arm) for arm in arms) and not any(
            _takes_literally(arm, value) for arm in arms
        ):
            try:
                expanded = ValueRange.parse(value)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from None
            if expanded is not None:
                return expanded.values()
        for arm in arms:
            nested = self._nested(value, arm, path)
            if nested is not value:
                return nested
        return value

    def _nested(self, value: Any, arm: Any, path: str) -> Any:
        origin = typing.get_origin(arm) or arm
        arguments = typing.get_args(arm)
        if isinstance(origin, type) and issubclass(origin, BaseModel):
            return self.model(value, origin, path) if isinstance(value, Mapping) else value
        if origin in _SEQUENCES and isinstance(value, list | tuple) and arguments:
            element = arguments[0]
            return [self.value(item, element, f"{path}[{i}]") for i, item in enumerate(value)]
        if origin in (dict, Mapping) and isinstance(value, Mapping) and len(arguments) == 2:
            return {k: self.value(item, arguments[1], f"{path}.{k}") for k, item in value.items()}
        return value


def item_parts(item: Mapping[str, Any]) -> tuple[tuple[str | None, type[BaseModel]], ...]:
    """The model of a raw check item's settings, or none when the catalog (which
    needs the Polars backend) or the check is unavailable."""
    try:
        return CheckItem.path_parts(item)
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("polars"):
            raise
        return ()


def _stripped(annotation: Any) -> Any:
    while typing.get_origin(annotation) is Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _arms(annotation: Any) -> tuple[Any, ...]:
    """The members of a union annotation (the annotation itself otherwise)."""
    annotation = _stripped(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return tuple(_stripped(arm) for arm in typing.get_args(annotation))
    return (annotation,)


def _is_sequence(arm: Any) -> bool:
    return (typing.get_origin(arm) or arm) in _SEQUENCES


def _takes_literally(arm: Any, value: Any) -> bool:
    """Whether a non-sequence type ``arm`` accepts ``value`` as it is written."""
    if arm is Any:
        return True
    if typing.get_origin(arm) is typing.Literal:
        return value in typing.get_args(arm)
    origin = typing.get_origin(arm) or arm
    if isinstance(value, str):
        return origin is str
    if isinstance(value, Mapping):
        return origin in (dict, Mapping) or (
            isinstance(origin, type) and issubclass(origin, BaseModel)
        )
    return False


def _parts(model: type[BaseModel], raw: Mapping[str, Any]) -> Iterable[tuple[Any, Any]]:
    hook = getattr(model, "path_parts", None)
    if hook is None:
        return ((None, model),)
    parts: Iterable[tuple[Any, Any]] = hook(raw)
    return parts


def _field_index(model: type[BaseModel]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        index[name] = field
        if isinstance(field.alias, str):
            index[field.alias] = field
    return index


class ItemPreparation:
    """Applies ``defaults``, expands ranges and reads a single ``datasets`` string as a
    one-element list in the check items of a raw configuration."""

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
        prepared = TypedRanges().item(prepared, path)
        if isinstance(prepared.get("datasets"), str):
            prepared["datasets"] = [prepared["datasets"]]
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
    "TypedRanges",
    "ValueRange",
    "default_problems",
]
