"""Explicit ``overrides``: patches applied to known datasets and check instances.

::

    overrides:
      checks:
        no missing years: {max: 0.2, severity: WARNING}      # a check item's name
        datasets.counts.checks[2]: {enabled: false}          # or its instance id
      datasets:
        counts: {schema: {min_rows: 100}}
        counts@houston: {load_policy: {retries: 2}}          # an across copy
      resources: {target_memory: 4GiB}                       # the resources section

Overrides apply after ``include``, ``vars``, ``defaults`` and ``across`` expansion
and before validation, so they patch the configuration as it will run. Mappings
merge recursively; lists and scalars replace; ``null`` is an error, since an
override never deletes a setting. A target must exist: an unknown one is an error
naming the closest match, and a check name that several instances share is an
error asking for the instance id. ``resources`` patches the ``resources`` section
as a whole (it need not be written before). Each file's ``overrides`` apply in include
order, the root configuration's last, and every applied override is recorded
(:class:`AppliedOverride`).

Like :mod:`qavf.composition`, this module works on raw mappings and imports no
settings models.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

OVERRIDES_KEY = "overrides"
ORIGIN_KEY = "origin"
"""The file a layer of overrides comes from, when files were included."""
_SECTIONS = ("datasets", "checks")
"""Datasets apply first: a dataset override may replace a dataset's check list."""
RESOURCES_SECTION = "resources"
"""The section patched as a whole: its patch is the settings, not targets."""


@dataclass(frozen=True, slots=True)
class AppliedOverride:
    """One override as applied: its target, the instance it resolved to and what it set."""

    section: str
    """``datasets``, ``checks`` or ``resources``."""
    target: str
    """The target as written: a dataset name, a check name or an instance id."""
    instance: str
    """The resolved location, e.g. ``datasets.counts`` or ``datasets.counts.checks[2]``."""
    fields: tuple[str, ...]
    """The settings it set, as dotted paths within the target (``schema.min_rows``)."""
    origin: str | None = None
    """The file that declared it (``None``: the configuration itself, without includes)."""
    line: int | None = None
    """The line its target was written on, when the configuration was read from a file."""

    def describe(self) -> str:
        """One line for ``qavf explain``."""
        named = self.target not in (self.instance, self.instance.removeprefix("datasets."))
        written = f" ({self.target!r})" if named else ""
        line = f":{self.line}" if self.line is not None else ""
        if self.origin:
            source = f" from {self.origin}{line}"
        else:
            source = f" (line {self.line})" if self.line is not None else ""
        return f"{self.instance}{written}: {', '.join(self.fields)}{source}"


@dataclass(frozen=True, slots=True)
class OverrideLayer:
    """One file's ``overrides`` section."""

    origin: str | None
    sections: Mapping[str, Mapping[str, Mapping[str, Any]]]
    resources: Mapping[str, Any] | None = None
    """The patch of the ``resources`` section, when the layer has one."""

    @classmethod
    def parse(cls, raw: Any, origin: str | None = None) -> OverrideLayer:
        """Read one layer, rejecting unknown sections, non-mapping patches and ``null``."""
        where = f"overrides ({origin})" if origin else "overrides"
        if isinstance(raw, BaseModel):
            raw = raw.model_dump(by_alias=True, exclude_unset=True)
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"{where} must be a mapping with 'checks', 'datasets' and/or 'resources'"
            )
        layer = dict(raw)
        written = layer.pop(ORIGIN_KEY, None)
        origin = origin if origin is not None else written
        resources = layer.pop(RESOURCES_SECTION, None)
        unknown = sorted(str(key) for key in layer if key not in _SECTIONS)
        if unknown:
            raise ValueError(
                f"{where}: unknown section {unknown[0]!r}; overrides take 'checks', "
                "'datasets' and 'resources'"
            )
        if resources is not None:
            resources = _plain(resources)
            if not isinstance(resources, Mapping) or not resources:
                raise ValueError(
                    f"{where}.{RESOURCES_SECTION} must be a non-empty mapping of resource limits"
                )
            _reject_null(resources, f"{where}.{RESOURCES_SECTION}")
        sections: dict[str, Mapping[str, Mapping[str, Any]]] = {}
        for section in _SECTIONS:
            patches = layer.get(section) or {}
            if not isinstance(patches, Mapping):
                raise ValueError(f"{where}.{section} must map each target to its settings")
            for target, patch in patches.items():
                location = f"{where}.{section}.{target}"
                if not isinstance(patch, Mapping) or not patch:
                    raise ValueError(f"{location} must be a non-empty mapping of settings")
                _reject_null(patch, location)
            sections[section] = patches
        return cls(origin if isinstance(origin, str) else None, sections, resources)

    @classmethod
    def all_of(cls, raw: Any) -> list[OverrideLayer]:
        """Every layer of an ``overrides`` value: one mapping, or a list of layers
        (one per file, in the order they apply) as included files produce."""
        if raw is None:
            return []
        if isinstance(raw, list | tuple):
            return [cls.parse(item) for item in raw]
        return [cls.parse(raw)]


class OverridePatch:
    """Merges one override into its target: mappings recursively, other values replace."""

    def __init__(self) -> None:
        self.fields: list[str] = []

    def merge(self, base: Any, patch: Mapping[str, Any], path: str = "") -> dict[str, Any]:
        """``patch`` merged over ``base``; :attr:`fields` collects the settings set."""
        result = dict(_mapping(base))
        for key, value in patch.items():
            location = f"{path}{key}"
            current = result.get(key)
            if isinstance(value, Mapping) and isinstance(_plain(current), Mapping):
                result[key] = self.merge(current, value, f"{location}.")
            else:
                result[key] = value
                self.fields.append(location)
        return result


class CheckInstances:
    """The check items of a raw configuration by instance id and by name."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.locations: dict[str, tuple[tuple[str, ...], int]] = {}
        """Instance id -> (path of the list holding it, index in that list)."""
        self.names: dict[str, list[str]] = {}
        checks = raw.get("checks")
        if isinstance(checks, list):
            self._index(("checks",), "checks", checks)
        datasets = raw.get("datasets")
        if isinstance(datasets, Mapping):
            for name, spec in datasets.items():
                items = _plain(spec).get("checks") if isinstance(_plain(spec), Mapping) else None
                if isinstance(items, list):
                    self._index(("datasets", name, "checks"), f"datasets.{name}.checks", items)

    def _index(self, path: tuple[str, ...], prefix: str, items: list[Any]) -> None:
        for index, item in enumerate(items):
            instance = f"{prefix}[{index}]"
            self.locations[instance] = (path, index)
            name = _plain(item).get("name") if isinstance(_plain(item), Mapping) else None
            if isinstance(name, str):
                self.names.setdefault(name, []).append(instance)

    def resolve(self, target: str, where: str) -> str:
        """The instance id ``target`` (an id or a name) names.

        Raises:
            ValueError: for an unknown target (naming the closest match) or a name
                that several instances share (listing their ids).
        """
        if target in self.locations:
            return target
        matches = self.names.get(target, [])
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(
                f"{where}: check name {target!r} names {len(matches)} instances "
                f"({', '.join(matches)}); override one by its instance id"
            )
        raise ValueError(
            f"{where}: no check is named {target!r} or has that instance id"
            f"{_closest(target, [*self.names, *self.locations])}"
        )


class OverrideApplication:
    """Applies every layer of ``overrides`` to a raw configuration, in order."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.applied: list[AppliedOverride] = []

    @classmethod
    def apply(cls, raw: dict[str, Any]) -> tuple[dict[str, Any], tuple[AppliedOverride, ...]]:
        """``raw`` without ``overrides``, patched by them, and the record of each applied."""
        if OVERRIDES_KEY not in raw:
            return raw, ()
        layers = OverrideLayer.all_of(raw[OVERRIDES_KEY])
        application = cls({key: value for key, value in raw.items() if key != OVERRIDES_KEY})
        for layer in layers:
            application._layer(layer)
        return application.raw, tuple(application.applied)

    def _layer(self, layer: OverrideLayer) -> None:
        where = f"overrides ({layer.origin})" if layer.origin else "overrides"
        for target, patch in layer.sections["datasets"].items():
            self._dataset(layer.origin, f"{where}.datasets.{target}", target, patch)
        instances = CheckInstances(self.raw)
        for target, patch in layer.sections["checks"].items():
            location = f"{where}.checks.{target}"
            self._check(layer.origin, instances.resolve(target, location), target, patch)
        if layer.resources is not None:
            self._resources(layer.origin, layer.resources)

    def _dataset(
        self, origin: str | None, where: str, target: str, patch: Mapping[str, Any]
    ) -> None:
        datasets = self.raw.get("datasets")
        known = list(datasets) if isinstance(datasets, Mapping) else []
        if target not in known:
            raise ValueError(f"{where}: no dataset is named {target!r}{_closest(target, known)}")
        assert isinstance(datasets, Mapping)
        merge = OverridePatch()
        self.raw["datasets"] = {**datasets, target: merge.merge(datasets[target], patch)}
        self._record("datasets", target, f"datasets.{target}", merge.fields, origin)

    def _resources(self, origin: str | None, patch: Mapping[str, Any]) -> None:
        merge = OverridePatch()
        self.raw[RESOURCES_SECTION] = merge.merge(self.raw.get(RESOURCES_SECTION), patch)
        self._record(RESOURCES_SECTION, RESOURCES_SECTION, RESOURCES_SECTION, merge.fields, origin)

    def _check(
        self, origin: str | None, instance: str, target: str, patch: Mapping[str, Any]
    ) -> None:
        path, index = CheckInstances(self.raw).locations[instance]
        items = list(self._container(path))
        merge = OverridePatch()
        items[index] = merge.merge(items[index], patch)
        self._replace(path, items)
        self._record("checks", target, instance, merge.fields, origin)

    def _container(self, path: tuple[str, ...]) -> list[Any]:
        value: Any = self.raw
        for key in path:
            value = _plain(value)[key]
        assert isinstance(value, list)
        return value

    def _replace(self, path: tuple[str, ...], items: list[Any]) -> None:
        """Set the list at ``path`` to ``items``, copying each mapping on the way so
        shared parts of the configuration are never changed in place."""
        if path == ("checks",):
            self.raw["checks"] = items
            return
        _, dataset, key = path
        datasets = dict(self.raw["datasets"])
        datasets[dataset] = {**_mapping(datasets[dataset]), key: items}
        self.raw["datasets"] = datasets

    def _record(
        self, section: str, target: str, instance: str, fields: Iterable[str], origin: str | None
    ) -> None:
        self.applied.append(AppliedOverride(section, target, instance, tuple(fields), origin))


def _reject_null(patch: Mapping[str, Any], where: str) -> None:
    for key, value in patch.items():
        if value is None:
            raise ValueError(
                f"{where}.{key}: null is not allowed; an override sets or merges a "
                "value and never deletes one"
            )
        if isinstance(value, Mapping):
            _reject_null(value, f"{where}.{key}")


def _plain(value: Any) -> Any:
    """A model as the mapping it was validated from; anything else unchanged."""
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, exclude_unset=True)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    plain = _plain(value)
    return plain if isinstance(plain, Mapping) else {}


def _closest(target: str, candidates: Iterable[str]) -> str:
    """``; did you mean 'x'?`` for the closest candidate, or nothing."""
    close = difflib.get_close_matches(target, list(candidates), n=1, cutoff=0.5)
    return f"; did you mean {close[0]!r}?" if close else ""


__all__ = [
    "ORIGIN_KEY",
    "OVERRIDES_KEY",
    "RESOURCES_SECTION",
    "AppliedOverride",
    "CheckInstances",
    "OverrideApplication",
    "OverrideLayer",
    "OverridePatch",
]
