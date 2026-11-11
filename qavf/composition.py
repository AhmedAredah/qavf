"""Configuration reuse resolved before validation: ``include`` files and ``vars``.

* ``include`` lists other YAML/JSON files, relative to the including file. They
  merge first, in order; the including file wins on conflicting values. A
  dataset defined twice, or an explicitly named top-level check repeated across
  files, is an error rather than a silent override. Include cycles are errors.
  Relative file paths written in an included file (a source's ``path``, a
  connection's database file, a baseline or zone-layer file, plugin ``paths``)
  are relative to that file, like its own ``include`` entries
  (:class:`IncludedPaths`), whether written literally or through a variable the
  file defines itself (:class:`PathReferences`). ``include`` entries may use
  ``${ENV}`` / ``${ENV:-default}``, but not ``${vars.name}``: they are read before
  any file's ``vars`` merge. A file reached through more than one include path
  (a diamond) is merged once.
* ``vars`` holds named values; ``${vars.name}`` in any string value refers to
  one. A value that is exactly a reference takes the variable's value (a list
  stays a list); inside a longer string it is spelled out. An unknown variable
  is an error.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qavf.config_paths import ConfigPaths, PathRole
from qavf.configuration import ConfigurationGraph, expand_env

if TYPE_CHECKING:
    from pydantic import BaseModel

INCLUDE_KEY = "include"
VARS_KEY = "vars"

_VAR_REFERENCE = re.compile(r"(?<!\$)\$\{vars\.([^}]*)\}")
"""``${vars.name}``; ``$${vars.name}`` is literal text (see :class:`LiteralReferences`)."""
_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNIQUE_SECTIONS = ("datasets",)
"""Top-level mappings whose entries may be defined in one file only."""
_CONCATENATED_SECTIONS = ("checks",)
"""Top-level lists that collect the entries of every file."""

Parser = Callable[[str, Path], dict[str, Any]]
"""Parses one file's text (without expanding references) as the content of a path."""


class IncludeResolver:
    """Merges the files a configuration includes into one mapping."""

    def __init__(self, parse: Parser) -> None:
        self._parse = parse
        self.included: list[Path] = []
        """Every file merged in so far (resolved), in the order they were read."""

    def resolve(self, raw: dict[str, Any], source: Path) -> dict[str, Any]:
        """``raw`` (the content of ``source``) with its included files merged in.

        Each file is measured before it is merged (:class:`ConfigurationGraph`), so a
        file whose shared parts expand beyond the limit fails before any copy is made.
        """
        root = source.resolve()
        ConfigurationGraph().check(raw)
        return self._resolve(raw, root, (root,))

    def _resolve(
        self,
        raw: dict[str, Any],
        source: Path,
        chain: tuple[Path, ...],
        owned: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """``raw`` (the content of ``source``) merged over its own included files.

        ``owned`` names the variables an including file defines: they are its to
        set, so an included file's path that uses one is left for the merged value.
        """
        entries = raw.pop(INCLUDE_KEY, None)
        if entries is None:
            return raw
        if isinstance(entries, str):
            entries = [entries]
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ValueError(f"{source}: include must be a file path or a list of paths")
        owned = owned | _variable_names(raw.get(VARS_KEY))
        merged: dict[str, Any] = {}
        for entry in entries:
            target = self._target(entry, source)
            if target in chain:
                cycle = " -> ".join(str(step) for step in (*chain, target))
                raise ValueError(f"include cycle: {cycle}")
            if target in self.included:
                continue  # a diamond: this file is already merged once
            try:
                text = target.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"{source}: cannot include {entry!r}: {exc.strerror}") from None
            self.included.append(target)
            content = self._parse(text, target)
            ConfigurationGraph().check(content)
            references = PathReferences(content.get(VARS_KEY), owned)
            parsed = IncludedPaths(target.parent, chain[0].parent, references).rebase(content)
            included = self._resolve(parsed, target, (*chain, target), owned)
            merged = ConfigMerge(str(target)).merge(merged, included)
        return ConfigMerge(str(source)).merge(merged, raw)

    @staticmethod
    def _target(entry: str, source: Path) -> Path:
        """The file an ``include`` entry names, its ``${ENV}`` references expanded."""
        try:
            expanded: str = expand_env(entry)
        except ValueError as exc:
            raise ValueError(f"{source}: include {entry!r}: {exc}") from None
        path = Path(expanded)
        return (path if path.is_absolute() else source.parent / path).resolve()


class PathReferences:
    """Substitutes an included file's own references in its file paths before rebasing.

    A path means the same whether it is written literally or through a variable
    the file defines (``vars: {input_file: roads.csv}``, ``path:
    ${vars.input_file}``): its own variables and ``${ENV}`` references are
    substituted first, then the result is rebased like literal text. A variable
    the file does not define, or one an including file also defines (``owned``),
    belongs to the configuration that sets it: the reference is kept and the
    merged value is resolved relative to the root configuration.
    """

    def __init__(self, variables: Any, owned: frozenset[str] = frozenset()) -> None:
        local = variables if isinstance(variables, Mapping) else {}
        self._variables = {
            name: value
            for name, value in local.items()
            if isinstance(name, str) and _VAR_NAME.fullmatch(name) and name not in owned
        }

    def substitute(self, value: str) -> str | list[str]:
        """``value`` with this file's references substituted, or ``value`` itself when
        it refers to a variable this file does not own."""
        names = _VAR_REFERENCE.findall(value)
        if names and not all(name in self._variables for name in names):
            return value
        substituted: Any = value
        if names:
            try:
                substituted = VariableSubstitution(self._variables).substitute(value, "path")
            except ValueError:
                return value  # it refers, through a variable, to one set elsewhere
        if isinstance(substituted, list) and all(isinstance(item, str) for item in substituted):
            return [str(expand_env(item)) for item in substituted]
        if not isinstance(substituted, str):
            return value
        expanded: str = expand_env(substituted)
        return expanded


class IncludedPaths:
    """Rewrites the relative file paths of an included file for the root configuration.

    Paths resolve against the root configuration's directory once files merge, so a
    relative file path written in an included file is rewritten to the same file
    relative to the root (absolute when no relative form exists, e.g. on another
    drive). With ``references`` (:class:`PathReferences`), the file's own variables
    and ``${ENV}`` references are substituted first. Absolute paths, URLs and paths
    that still start with a ``${...}`` reference are kept as written. SQL text is not
    rewritten: file paths inside queries stay relative to
    the root configuration.

    Only the settings typed as file paths are rewritten (:mod:`qavf.config_paths`):
    a dataset's ``path`` and its source's ``path``/``paths``, a connection's
    database file, check settings such as a baseline ``path`` or a zone layer's
    ``path``, and ``plugins.paths``. Column names, renames and other user data are
    never rewritten, whatever their key.
    """

    def __init__(
        self, directory: Path, root: Path, references: PathReferences | None = None
    ) -> None:
        self.directory = directory
        self.root = root
        self.references = references

    def rebase(self, raw: dict[str, Any], model: type[BaseModel] | None = None) -> dict[str, Any]:
        """``raw`` (an included file's content) with its relative paths rebased.

        ``model`` describes ``raw`` (a QAVF configuration by default).
        """
        if self.directory == self.root:
            return raw
        if model is None:
            from qavf.settings import QAVFSettings

            model = QAVFSettings
        rebased: dict[str, Any] = ConfigPaths(self.visit).model(raw, model)
        return rebased

    def visit(self, value: str, role: PathRole) -> str | list[str]:
        """A :data:`~qavf.config_paths.PathVisitor` rebasing each path."""
        if self.references is None:
            return self.path(value)
        substituted = self.references.substitute(value)
        if isinstance(substituted, list):
            return [self.path(item) for item in substituted]
        return self.path(substituted)

    def path(self, value: str) -> str:
        """``value`` relative to the root directory, when it is a relative file path."""
        if not value or "://" in value or value.startswith(("${", "~")):
            return value
        if Path(value).is_absolute() or value.startswith(("/", "\\")):
            return value
        target = os.path.normpath(self.directory / value)
        try:
            return Path(os.path.relpath(target, self.root)).as_posix()
        except ValueError:  # another drive (Windows): no relative form
            return Path(target).as_posix()


class ConfigMerge:
    """Merges one file's configuration over what earlier files defined."""

    def __init__(self, origin: str) -> None:
        self.origin = origin
        """The file whose content is merged in, for error messages."""
        self._deep = DeepMerge()

    def merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """``override`` merged over ``base``: the override wins on conflicting values."""
        result = dict(base)
        for key, value in override.items():
            if key not in result:
                result[key] = value
            elif key in _UNIQUE_SECTIONS:
                result[key] = self._unique(key, result[key], value)
            elif key in _CONCATENATED_SECTIONS:
                result[key] = self._concatenated(key, result[key], value)
            else:
                result[key] = self._deep.merge(result[key], value)
        return result

    def _unique(self, section: str, base: Any, override: Any) -> Any:
        if not isinstance(base, Mapping) or not isinstance(override, Mapping):
            return override
        repeated = sorted(set(base) & set(override), key=str)
        if repeated:
            raise ValueError(
                f"{self.origin}: {section} {repeated[0]!r} is already defined by an included "
                "file; each dataset is defined once"
            )
        return {**base, **override}

    def _concatenated(self, section: str, base: Any, override: Any) -> Any:
        if not isinstance(base, list) or not isinstance(override, list):
            return override
        named = {_item_name(item) for item in base} - {None}
        for item in override:
            name = _item_name(item)
            if name is not None and name in named:
                raise ValueError(
                    f"{self.origin}: the {section} entry named {name!r} is already defined "
                    "by an included file; give each check its own name"
                )
        return [*base, *override]


def _item_name(item: Any) -> str | None:
    name = item.get("name") if isinstance(item, Mapping) else None
    return name if isinstance(name, str) else None


class DeepMerge:
    """Merges mappings recursively, each pair of shared parts once.

    YAML anchors share one part in many places; merging occurrence by occurrence
    would copy it exponentially often, so the merge of a ``(base, override)`` pair
    is memoized and shared parts stay shared in the result.
    """

    def __init__(self) -> None:
        self._memo: dict[tuple[int, int], Any] = {}
        self._kept: list[Any] = []  # keeps memoised inputs alive, so ids stay unique

    def merge(self, base: Any, override: Any) -> Any:
        """``override`` merged over ``base`` (the override wins on other values)."""
        if not (isinstance(base, Mapping) and isinstance(override, Mapping)):
            return override
        key = (id(base), id(override))
        if key in self._memo:
            return self._memo[key]
        merged = dict(base)
        self._memo[key] = merged
        self._kept.append((base, override))
        for name, value in override.items():
            merged[name] = self.merge(merged[name], value) if name in merged else value
        return merged


class VariableSubstitution:
    """Replaces ``${vars.name}`` references with the configuration's ``vars``.

    Variables may refer to other variables; a cycle is an error. Shared parts
    (YAML anchors) are substituted once and stay shared.
    """

    def __init__(self, variables: Any) -> None:
        if variables is None:
            variables = {}
        if not isinstance(variables, Mapping):
            raise ValueError("vars must be a mapping of names to values")
        for name in variables:
            if not isinstance(name, str) or not _VAR_NAME.fullmatch(name):
                raise ValueError(f"vars: {name!r} is not a valid variable name")
        self._raw = dict(variables)
        self._resolved: dict[str, Any] = {}
        self._resolving: list[str] = []
        self._memo: dict[int, Any] = {}

    @classmethod
    def apply(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """``raw`` with every reference substituted (unchanged without references)."""
        if VARS_KEY not in raw and not _mentions_vars(raw):
            return raw
        substitution = cls(raw.get(VARS_KEY))
        result: dict[str, Any] = {}
        try:
            for key, value in raw.items():
                result[key] = (
                    {name: substitution.value(name) for name in substitution._raw}
                    if key == VARS_KEY
                    else substitution.substitute(value, key)
                )
        except RecursionError as exc:
            raise ValueError("configuration is nested too deeply or refers to itself") from exc
        return result

    def value(self, name: str) -> Any:
        """The value of variable ``name``, its own references substituted."""
        if name in self._resolved:
            return self._resolved[name]
        if name not in self._raw:
            known = ", ".join(sorted(self._raw)) or "none are defined"
            raise ValueError(f"unknown variable ${{vars.{name}}} (defined: {known})")
        if name in self._resolving:
            cycle = " -> ".join([*self._resolving[self._resolving.index(name) :], name])
            raise ValueError(f"vars refer to each other in a cycle: {cycle}")
        self._resolving.append(name)
        try:
            resolved = self.substitute(self._raw[name], f"vars.{name}")
        finally:
            self._resolving.pop()
        self._resolved[name] = resolved
        return resolved

    def substitute(self, item: Any, path: str) -> Any:
        """``item`` with every reference substituted; ``path`` locates errors."""
        if isinstance(item, str):
            return self._text(item, path)
        if not isinstance(item, dict | list):
            return item
        identity = id(item)
        if identity in self._memo:
            return self._memo[identity]
        if isinstance(item, dict):
            result: Any = {
                key: self.substitute(child, f"{path}.{key}") for key, child in item.items()
            }
        else:
            result = [self.substitute(child, f"{path}[{i}]") for i, child in enumerate(item)]
        self._memo[identity] = result
        return result

    def _text(self, text: str, path: str) -> Any:
        whole = _VAR_REFERENCE.fullmatch(text)
        if whole is not None:
            return copy.deepcopy(self._named(whole.group(1), path))
        return _VAR_REFERENCE.sub(self._spelled_out(path), text)

    def _named(self, name: str, path: str) -> Any:
        try:
            return self.value(name)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None

    def _spelled_out(self, path: str) -> _SpelledOut:
        return _SpelledOut(self, path)


class _SpelledOut:
    """Replaces one reference inside a longer string with the variable's text."""

    def __init__(self, substitution: VariableSubstitution, path: str) -> None:
        self.substitution = substitution
        self.path = path

    def __call__(self, match: re.Match[str]) -> str:
        value = self.substitution._named(match.group(1), self.path)
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str | int | float):
            return str(value)
        raise ValueError(
            f"{self.path}: ${{vars.{match.group(1)}}} holds a {type(value).__name__}; only "
            "a value that is exactly the reference can take a list or mapping"
        )


def _variable_names(variables: Any) -> frozenset[str]:
    """The names a file's ``vars`` mapping defines."""
    if not isinstance(variables, Mapping):
        return frozenset()
    return frozenset(name for name in variables if isinstance(name, str))


def _mentions_vars(value: Any) -> bool:
    """Whether any string in ``value`` refers to a variable (cheap pre-check)."""
    stack = [value]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if "${vars." in item:
                return True
        elif isinstance(item, dict | list) and id(item) not in seen:
            seen.add(id(item))
            stack.extend(item.values() if isinstance(item, dict) else item)
    return False


__all__ = [
    "INCLUDE_KEY",
    "VARS_KEY",
    "ConfigMerge",
    "IncludeResolver",
    "IncludedPaths",
    "PathReferences",
    "VariableSubstitution",
]
