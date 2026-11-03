"""Which configuration settings are file paths: one typed declaration for every use.

A setting is a file path because its model field says so, never because of the
key it is written under. Fields are marked with :data:`FilePath` (one local file
or glob), :data:`BaselinePath` (an accepted baseline: the file and its acceptance
manifest) or :data:`SourceSpec` (a source specification whose ``path``/``paths``
name its files). :class:`ConfigPaths` walks a raw configuration mapping along
the fields of a model and visits only those values; mappings of user data
(``rename``, ``schema``, ``views``, ``series``, ``where``, ...) are not visited,
so a column that happens to be named ``path`` stays a column name.

The same walk serves three purposes:

* rebasing the relative paths of an included file (:mod:`qavf.composition`);
* rebasing a migrated configuration written to another directory (``qavf migrate``);
* the declared-input manifest that output destinations may never replace
  (:class:`qavf.io.ProtectedInputs`).

A model whose fields depend on a value (a check item's ``check``, a comparison
level's ``level``) provides ``path_parts(raw)``: the ``(key, model)`` pairs that
describe ``raw`` (``key`` ``None`` for ``raw`` itself).
"""

from __future__ import annotations

import enum
import os
import types
import typing
from collections.abc import Callable, Iterable, Mapping
from pathlib import PurePath
from typing import Annotated, Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo


class PathRole(enum.Enum):
    """What a path-valued setting names."""

    FILE = "file"
    """A local file (or a glob of files) the run reads."""
    BASELINE = "baseline"
    """An accepted baseline file; its acceptance manifest is read with it."""
    SOURCE = "source"
    """A source specification mapping; its ``path``/``paths`` name its files."""


FilePath = Annotated[str, PathRole.FILE]
"""A setting naming a local file, relative to the configuration file."""
BaselinePath = Annotated[str, PathRole.BASELINE]
"""A setting naming an accepted baseline file, relative to the configuration file."""
SourceSpec = Annotated[dict[str, Any], PathRole.SOURCE]
"""A source specification (``{type: file, path: ...}``, ``{type: query, ...}``, ...)."""

SOURCE_PATH_KEYS = ("path", "paths")
"""The keys of a source specification that name files."""

PathVisitor = Callable[[str, PathRole], str]
"""Receives one path value and its role; returns the value to keep."""

PathPart = tuple[str | None, type[BaseModel]]
"""``(key, model)``: ``raw[key]`` (``raw`` itself for ``None``) is described by ``model``."""


class ConfigPaths:
    """Visits the file-path settings of raw configuration mappings.

    The walk returns a copy with each visited value replaced by the visitor's
    result; everything else is kept as it is (shared YAML aliases stay shared).
    """

    def __init__(self, visit: PathVisitor) -> None:
        self.visit = visit
        self._memo: dict[tuple[int, Any], Any] = {}
        self._kept: list[Any] = []  # keeps memoised originals alive, so ids stay unique

    def model(self, raw: Any, model: type[BaseModel]) -> Any:
        """``raw`` (the input of ``model``) with its file-path settings visited."""
        if not isinstance(raw, Mapping):
            return raw
        key = (id(raw), model)
        if key in self._memo:
            return self._memo[key]
        result = dict(raw)
        self._memo[key] = result
        self._kept.append(raw)
        for part, described in _parts(model, raw):
            if part is None:
                if described is not model:
                    result.update(self.model(result, described))
                else:
                    self._fields(result, model)
            elif isinstance(result.get(part), Mapping):
                result[part] = self.model(result[part], described)
        return result

    def _fields(self, result: dict[str, Any], model: type[BaseModel]) -> None:
        fields = _field_index(model)
        for key, value in list(result.items()):
            field = fields.get(key) if isinstance(key, str) else None
            if field is not None:
                result[key] = self.value(value, field.annotation, tuple(field.metadata))

    def value(self, value: Any, annotation: Any, metadata: tuple[Any, ...] = ()) -> Any:
        """``value`` of a field typed ``annotation`` with its file paths visited."""
        role = _role(metadata) or _role(_annotated_metadata(annotation))
        if role is not None:
            return self._role(value, role)
        annotation = _stripped(annotation)
        origin = typing.get_origin(annotation)
        if origin in (typing.Union, types.UnionType):
            return self._union(value, typing.get_args(annotation))
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self.model(value, annotation)
        if origin in (list, tuple, set, frozenset) and isinstance(value, list | tuple):
            return self._sequence(value, typing.get_args(annotation))
        if origin in (dict, Mapping) and isinstance(value, Mapping):
            arguments = typing.get_args(annotation)
            if len(arguments) == 2 and _may_hold_paths(arguments[1]):
                return {key: self.value(item, arguments[1]) for key, item in value.items()}
        return value

    def _role(self, value: Any, role: PathRole) -> Any:
        if role is PathRole.SOURCE:
            return self.source(value)
        if isinstance(value, str | os.PathLike):
            return self.visit(os.fspath(value), role)
        if isinstance(value, list | tuple):
            return [self._role(item, role) for item in value]
        return value

    def source(self, value: Any) -> Any:
        """A source specification with its ``path``/``paths`` visited."""
        if not isinstance(value, Mapping):
            return value
        result = dict(value)
        for key in SOURCE_PATH_KEYS:
            if key in result:
                result[key] = self._role(result[key], PathRole.FILE)
        return result

    def _union(self, value: Any, arms: tuple[Any, ...]) -> Any:
        for arm in arms:
            if _accepts(arm, value):
                return self.value(value, arm)
        return value

    def _sequence(self, value: list[Any] | tuple[Any, ...], arguments: tuple[Any, ...]) -> Any:
        if not arguments:
            return value
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            items = [self.value(item, arguments[0]) for item in value]
        elif len(arguments) == len(value):
            items = [self.value(item, arm) for item, arm in zip(value, arguments, strict=True)]
        elif len(arguments) == 1:
            items = [self.value(item, arguments[0]) for item in value]
        else:
            return value
        return items if isinstance(value, list) else tuple(items)


class PathCollector:
    """A :data:`PathVisitor` that records every visited path and keeps it unchanged."""

    def __init__(self) -> None:
        self.found: list[tuple[str, PathRole]] = []

    def __call__(self, value: str, role: PathRole) -> str:
        self.found.append((value, role))
        return value


def declared_paths(raw: Mapping[str, Any], model: type[BaseModel]) -> list[tuple[str, PathRole]]:
    """Every file-path setting of ``raw`` (the input of ``model``), with its role."""
    collector = PathCollector()
    ConfigPaths(collector).model(raw, model)
    return collector.found


def _parts(model: type[BaseModel], raw: Mapping[str, Any]) -> Iterable[PathPart]:
    hook = getattr(model, "path_parts", None)
    if hook is None:
        return ((None, model),)
    parts: Iterable[PathPart] = hook(raw)
    return parts


def _field_index(model: type[BaseModel]) -> dict[str, FieldInfo]:
    index: dict[str, FieldInfo] = {}
    for name, field in model.model_fields.items():
        index[name] = field
        if isinstance(field.alias, str):
            index[field.alias] = field
    return index


def _role(metadata: Iterable[Any]) -> PathRole | None:
    for item in metadata:
        if isinstance(item, PathRole):
            return item
    return None


def _annotated_metadata(annotation: Any) -> tuple[Any, ...]:
    if typing.get_origin(annotation) is Annotated:
        return tuple(annotation.__metadata__)
    return ()


def _stripped(annotation: Any) -> Any:
    while typing.get_origin(annotation) is Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _accepts(arm: Any, value: Any) -> bool:
    """Whether a union member ``arm`` is the one that describes ``value``."""
    if _role(_annotated_metadata(arm)) is PathRole.SOURCE:
        return isinstance(value, Mapping)
    target = _stripped(arm)
    origin = typing.get_origin(target) or target
    if target is Any:
        return False
    if isinstance(origin, type) and issubclass(origin, BaseModel):
        return isinstance(value, Mapping)
    if origin in (list, tuple, set, frozenset):
        return isinstance(value, list | tuple)
    if origin in (dict, Mapping):
        return isinstance(value, Mapping)
    if origin in (str, PurePath) or (isinstance(origin, type) and issubclass(origin, PurePath)):
        return isinstance(value, str | os.PathLike)
    return False


def _may_hold_paths(annotation: Any) -> bool:
    """Whether values typed ``annotation`` can contain a file-path setting at all."""
    if _role(_annotated_metadata(annotation)) is not None:
        return True
    target = _stripped(annotation)
    if isinstance(target, type) and issubclass(target, BaseModel):
        return True
    return any(_may_hold_paths(argument) for argument in typing.get_args(target))


__all__ = [
    "SOURCE_PATH_KEYS",
    "BaselinePath",
    "ConfigPaths",
    "FilePath",
    "PathCollector",
    "PathRole",
    "PathVisitor",
    "SourceSpec",
    "declared_paths",
]
