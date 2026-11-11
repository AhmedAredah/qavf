"""Where each configuration value was written: its file, line and column.

Parsing records the location of every mapping key and sequence item, keyed by
its path in the configuration (``("datasets", "x", "schema", "min_rows")``), in
an :class:`OriginIndex`. Each resolution step keeps the index pointing at the
place a value was written:

* ``include`` merges the index of each file with its content, so an included
  file's values keep its name (a top-level ``checks`` list shifts the indexes of
  the entries a later file appends);
* ``${vars.name}`` and ``${ENV}`` substitution keep the location where the
  reference was written and note it ("via vars.limit", "via ${HOST}"); a value
  that is a whole variable reaches the variable's own definition for its parts;
* ``defaults`` give the settings an item omits the location in ``defaults``
  ("from defaults.change_rate"); ``across`` copies point at the dataset they copy
  ("from across copy counts@houston"); ``overrides`` point at the override that
  set them (:class:`ReuseOrigins`).

The index is compact: one small record per key or item written, and an alias
(a path prefix standing for another) where a part is reused, so shared YAML
anchors, copies and defaults are never copied entry by entry. A path the index
does not know simply has no origin.

This module reads YAML nodes and plain mappings only and imports no settings model.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from qavf.overrides import AppliedOverride

ConfigPath = tuple[str | int, ...]
"""A location in a configuration: mapping keys and sequence indexes."""
ORIGINS_CONTEXT = "qavf_origins"
"""The validation-context key carrying an :class:`OriginIndex` into ``QAVFSettings``."""
EXCEPTION_ATTRIBUTE = "qavf_origins"
"""The attribute an error raised while loading a configuration carries its index in."""
_MAX_ALIAS_DEPTH = 32
_PATH_PART = re.compile(r"\[(\d+)\]|([^.\[\]]+)")
_MENTIONED_PATH = re.compile(
    r"(?<![\w.@-])(?:datasets|checks|connections|secrets|defaults|vars|run|engine|report|"
    r"plugins|extensions|overrides|project|resources)(?:\.[^\s:;,'\"()\[\]]+|\[\d+\])*"
)


@dataclass(frozen=True, slots=True)
class Origin:
    """Where one value was written."""

    file: str
    """The file, relative to the root configuration's directory where it can be."""
    line: int
    """1-based line."""
    column: int
    """1-based column."""
    note: str | None = None
    """How the value got here, when not literally (``via vars.limit``, ``from defaults.x``)."""

    def location(self) -> str:
        """``file:line``."""
        return f"{self.file}:{self.line}"

    def describe(self) -> str:
        """``file:line:column`` with the note, for ``qavf resolve --origins``."""
        note = f" ({self.note})" if self.note else ""
        return f"{self.file}:{self.line}:{self.column}{note}"

    def noted(self, note: str | None) -> Origin:
        """This origin with ``note`` added after any note it has."""
        if not note:
            return self
        return replace(self, note=f"{self.note}; {note}" if self.note else note)


def format_path(path: Sequence[str | int]) -> str:
    """``datasets.x.checks[0].max`` for ``("datasets", "x", "checks", 0, "max")``."""
    text = ""
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else str(part))
    return text


def parse_path(text: str) -> ConfigPath:
    """The parts of a dotted path such as ``datasets.x.checks[2]`` (best effort: a key
    holding a dot reads as two keys)."""
    return tuple(
        int(index) if index else name for index, name in _PATH_PART.findall(text) if index or name
    )


class OriginIndex:
    """Configuration path -> :class:`Origin`, with aliases for reused parts."""

    __slots__ = ("_aliases", "_entries")

    def __init__(self) -> None:
        self._entries: dict[ConfigPath, Origin] = {}
        self._aliases: dict[ConfigPath, tuple[ConfigPath, str | None]] = {}

    @classmethod
    def of(cls, carrier: object) -> OriginIndex | None:
        """The index a validation context or an error carries, if any."""
        if isinstance(carrier, Mapping):
            found = carrier.get(ORIGINS_CONTEXT)
        else:
            found = getattr(carrier, EXCEPTION_ATTRIBUTE, None)
        return found if isinstance(found, OriginIndex) else None

    @property
    def size(self) -> int:
        """Entries and aliases held."""
        return len(self._entries) + len(self._aliases)

    def record(self, path: ConfigPath, origin: Origin) -> None:
        self._entries[path] = origin

    def alias(self, path: ConfigPath, source: ConfigPath, note: str | None = None) -> None:
        """Values under ``path`` come from the same place as those under ``source``."""
        if path != source:
            self._aliases[path] = (source, note)

    def annotate(self, path: ConfigPath, note: str) -> None:
        """Add ``note`` to the origin recorded at ``path``."""
        origin = self._entries.get(path)
        if origin is not None:
            self._entries[path] = origin.noted(note)

    def redirect(self, path: ConfigPath, source: ConfigPath, note: str | None = None) -> None:
        """``path`` and everything under it now come from ``source`` (a value replaced)."""
        self.drop(path, keep_root=False)
        origin = self.find(source)
        if origin is not None:
            self._entries[path] = origin.noted(note)
        self.alias(path, source, note)

    def find(self, path: Sequence[str | int], depth: int = 0) -> Origin | None:
        """Where the value at ``path`` was written, or ``None``."""
        key = tuple(path)
        origin = self._entries.get(key)
        if origin is not None or depth > _MAX_ALIAS_DEPTH:
            return origin
        for cut in range(len(key), 0, -1):
            alias = self._aliases.get(key[:cut])
            if alias is not None:
                source, note = alias
                found = self.find(source + key[cut:], depth + 1)
                return found.noted(note) if found is not None else None
        return None

    def locate(self, path: Sequence[str | int]) -> Origin | None:
        """:meth:`find`, or the origin of the nearest enclosing value (for an error
        about a setting that is missing or not written)."""
        key = tuple(path)
        for cut in range(len(key), 0, -1):
            found = self.find(key[:cut])
            if found is not None:
                return found
        return None

    def mentioned(self, text: str) -> Origin | None:
        """The origin of the most specific configuration path ``text`` names (an error
        message): the one with the most parts, the first of those."""
        best: tuple[int, Origin] | None = None
        for match in _MENTIONED_PATH.finditer(text):
            path = parse_path(match.group(0).rstrip("."))
            found = self.locate(path)
            if found is not None and (best is None or len(path) > best[0]):
                best = (len(path), found)
        return best[1] if best is not None else None

    def pop_subtree(self, prefix: ConfigPath) -> OriginIndex:
        """Remove and return what lies under ``prefix``, relative to it."""
        subtree = OriginIndex()
        width = len(prefix)
        for path in [path for path in self._entries if path[:width] == prefix and path != prefix]:
            subtree._entries[path[width:]] = self._entries.pop(path)
        for path in [path for path in self._aliases if path[:width] == prefix and path != prefix]:
            subtree._aliases[path[width:]] = self._aliases.pop(path)
        self._entries.pop(prefix, None)
        self._aliases.pop(prefix, None)
        return subtree

    def drop(self, prefix: ConfigPath, *, keep_root: bool = True) -> None:
        """Forget what lies under ``prefix`` (and ``prefix`` itself unless ``keep_root``)."""
        width = len(prefix)
        for table in (self._entries, self._aliases):
            for path in [path for path in table if path[:width] == prefix]:
                if path != prefix or not keep_root:
                    del table[path]

    def graft(self, prefix: ConfigPath, other: OriginIndex) -> None:
        """Put ``other`` (an index relative to ``prefix``) under ``prefix``."""
        for path, origin in other._entries.items():
            self._entries[prefix + path] = origin
        for path, (source, note) in other._aliases.items():
            self._aliases[prefix + path] = (prefix + source, note)

    def update(self, other: OriginIndex, shifts: Mapping[ConfigPath, int] | None = None) -> None:
        """Merge ``other`` (a later file) over this index: its origins win.

        ``shifts`` gives, for a list the later file appends to (``("checks",)``), how
        many entries come before its own.
        """
        moved = dict(shifts or {})
        for path, origin in other._entries.items():
            self._entries[_shifted(path, moved)] = origin
        for path, (source, note) in other._aliases.items():
            self._aliases[_shifted(path, moved)] = (_shifted(source, moved), note)

    def relabel(self, directory: Path) -> None:
        """Show each file relative to ``directory`` (the root configuration's) where it can."""
        labels: dict[str, str] = {}
        for path, origin in self._entries.items():
            label = labels.get(origin.file)
            if label is None:
                label = labels[origin.file] = _relative(origin.file, directory)
            if label != origin.file:
                self._entries[path] = replace(origin, file=label)

    def items(self) -> Iterator[tuple[ConfigPath, Origin]]:
        """The recorded origins (aliases are not expanded)."""
        return iter(self._entries.items())


def _shifted(path: ConfigPath, shifts: Mapping[ConfigPath, int]) -> ConfigPath:
    for prefix, offset in shifts.items():
        width = len(prefix)
        if path[:width] == prefix and len(path) > width and isinstance(path[width], int):
            return (*prefix, int(path[width]) + offset, *path[width + 1 :])
    return path


def _relative(file: str, directory: Path) -> str:
    try:
        return Path(os.path.relpath(file, directory)).as_posix()
    except ValueError:  # another drive (Windows): no relative form
        return file


class OriginNotes:
    """Names the file and line of the values an error is about.

    :meth:`attach` gives an error raised while validating a configuration its
    index (read back by :meth:`suffix` and :func:`qavf.redaction.exception_details`
    at the command line) and adds one note per located problem, which a traceback
    shows to a Python caller.
    """

    def __init__(self, origins: OriginIndex) -> None:
        self.origins = origins

    @classmethod
    def of(cls, exc: BaseException) -> OriginNotes | None:
        """The notes for an error that carries an index, if it does."""
        origins = OriginIndex.of(exc)
        return cls(origins) if origins is not None else None

    def attach(self, exc: BaseException) -> None:
        if OriginIndex.of(exc) is not None:
            return
        try:
            setattr(exc, EXCEPTION_ATTRIBUTE, self.origins)
        except AttributeError:  # an exception type without instance attributes
            return
        for note in self.notes(exc):
            exc.add_note(note)

    def notes(self, exc: BaseException) -> list[str]:
        """``<path>: written at <file:line:column>`` for each located problem."""
        from pydantic import ValidationError

        if not isinstance(exc, ValidationError):
            origin = self.origins.mentioned(str(exc))
            return [f"written at {origin.describe()}"] if origin is not None else []
        notes = []
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            origin = self.at(error["loc"], error["msg"])
            if origin is not None:
                shown = format_path(error["loc"]) or "<root>"
                notes.append(f"{shown}: written at {origin.describe()}")
        return list(dict.fromkeys(notes))

    def at(self, loc: Sequence[str | int], message: str = "") -> Origin | None:
        """The origin of an error at ``loc`` (or, at the root, of a path its message names)."""
        if loc:
            return self.origins.locate(loc)
        return self.origins.mentioned(message)

    def suffix(self, loc: Sequence[str | int], message: str = "") -> str:
        """`` (file:line)`` for an error at ``loc``, or nothing when it is not known."""
        origin = self.at(loc, message)
        return f" ({origin.location()})" if origin is not None else ""


class NodeOrigins:
    """Collects the origin of every key and item of a composed YAML document.

    Walks the node tree after construction (merge keys already flattened, so a
    ``<<`` merge's keys point at the anchor's lines and an explicit key wins). A
    collection reached again through an alias is recorded once and aliased.
    """

    def __init__(self, file: str) -> None:
        self.file = file

    def collect(self, root: yaml.Node | None) -> OriginIndex:
        index = OriginIndex()
        if root is None:
            return index
        first: dict[int, ConfigPath] = {}
        stack: list[tuple[yaml.Node, ConfigPath]] = [(root, ())]
        while stack:
            node, path = stack.pop()
            if not isinstance(node, yaml.MappingNode | yaml.SequenceNode):
                continue
            seen = first.get(id(node))
            if seen is not None:
                index.alias(path, seen)
                continue
            first[id(node)] = path
            if isinstance(node, yaml.MappingNode):
                for key_node, value_node in node.value:
                    if isinstance(key_node, yaml.ScalarNode):
                        child = (*path, key_node.value)
                        index.record(child, self._origin(key_node))
                        stack.append((value_node, child))
            else:
                for position, item in enumerate(node.value):
                    child = (*path, position)
                    index.record(child, self._origin(item))
                    stack.append((item, child))
        return index

    def _origin(self, node: yaml.Node) -> Origin:
        mark = node.start_mark
        return Origin(self.file, mark.line + 1, mark.column + 1)

    def json(self, text: str) -> OriginIndex:
        """The origins of a JSON document, read through the YAML composer (JSON's
        syntax is YAML flow syntax); an index without origins if it cannot be."""
        try:
            return self.collect(yaml.compose(text, Loader=yaml.SafeLoader))
        except (yaml.YAMLError, RecursionError):
            return OriginIndex()


class ReuseOrigins:
    """Keeps an index pointing at written values through ``across``, ``defaults`` and
    ``overrides``, which ``QAVFSettings`` applies to the raw configuration."""

    def __init__(self, index: OriginIndex) -> None:
        self.index = index

    def across(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
        """Each ``across`` copy points at the dataset it copies; a copy's own source
        settings at its entry in ``across``."""
        from qavf.checks.across import ACROSS_KEY, COPY_SEPARATOR

        datasets, expanded = before.get("datasets"), after.get("datasets")
        if not isinstance(datasets, Mapping) or not isinstance(expanded, Mapping):
            return
        for name, spec in datasets.items():
            if not isinstance(spec, Mapping) or spec.get(ACROSS_KEY) is None:
                continue
            across = spec[ACROSS_KEY]
            prefix = f"{name}{COPY_SEPARATOR}"
            copies = [copy for copy in expanded if str(copy).startswith(prefix)]
            for full in copies:
                copy = str(full)[len(prefix) :]
                note = f"from across copy {full}"
                target: ConfigPath = ("datasets", full)
                origin: ConfigPath = ("datasets", name)
                self.index.alias(target, origin, note)
                for key in ("path", "format"):
                    if key in spec:
                        self.index.alias((*target, "source", key), (*origin, key), note)
                if isinstance(across, list) and copy in across:
                    entry: ConfigPath = (*origin, ACROSS_KEY, across.index(copy))
                    self.index.alias((*target, "source", "connection"), entry, note)
                elif isinstance(across, Mapping) and isinstance(across.get(copy), Mapping):
                    for key in across[copy]:
                        entry = (*origin, ACROSS_KEY, copy, key)
                        self.index.alias((*target, "source", key), entry, note)

    def defaults(self, before: Mapping[str, Any]) -> None:
        """A setting an item omits points at the ``defaults`` entry that supplies it."""
        defaults = before.get("defaults")
        if not isinstance(defaults, Mapping):
            return
        for path, item in _items(before):
            check = item.get("check")
            supplied = defaults.get(check) if isinstance(check, str) else None
            if not isinstance(supplied, Mapping):
                continue
            for key in supplied:
                if isinstance(key, str) and key not in item:
                    source: ConfigPath = ("defaults", str(check), key)
                    self.index.alias((*path, key), source, f"from defaults.{check}")

    def overrides(
        self, written: Any, applied: Sequence[AppliedOverride]
    ) -> tuple[AppliedOverride, ...]:
        """Point each setting an override set at the override, and give each applied
        override the line it was written on."""
        located = []
        for record in applied:
            prefix = self._override_prefix(written, record)
            origin = self.index.find(prefix)
            note = f"override {record.target}"
            for field in record.fields:
                inner = parse_path(field)
                self.index.redirect((*parse_path(record.instance), *inner), (*prefix, *inner), note)
            located.append(replace(record, line=origin.line) if origin is not None else record)
        return tuple(located)

    @staticmethod
    def _override_prefix(written: Any, record: AppliedOverride) -> ConfigPath:
        """Where the override was written: under ``overrides``, or under the layer
        of its file when included files declared overrides too."""
        from qavf.overrides import ORIGIN_KEY, OVERRIDES_KEY

        # A section patched as a whole (``resources``) has no target key below it.
        target: ConfigPath = () if record.section == record.instance else (record.target,)
        if isinstance(written, list):
            for position, layer in enumerate(written):
                if isinstance(layer, Mapping) and layer.get(ORIGIN_KEY) == record.origin:
                    return (OVERRIDES_KEY, position, record.section, *target)
        return (OVERRIDES_KEY, record.section, *target)


def _items(raw: Mapping[str, Any]) -> Iterator[tuple[ConfigPath, Mapping[str, Any]]]:
    """Every raw check item with its path."""
    checks = raw.get("checks")
    if isinstance(checks, list):
        for position, item in enumerate(checks):
            if isinstance(item, Mapping):
                yield ("checks", position), item
    datasets = raw.get("datasets")
    if isinstance(datasets, Mapping):
        for name, spec in datasets.items():
            items = spec.get("checks") if isinstance(spec, Mapping) else None
            if isinstance(items, list):
                for position, item in enumerate(items):
                    if isinstance(item, Mapping):
                        yield ("datasets", name, "checks", position), item


__all__ = [
    "EXCEPTION_ATTRIBUTE",
    "ORIGINS_CONTEXT",
    "ConfigPath",
    "NodeOrigins",
    "Origin",
    "OriginIndex",
    "OriginNotes",
    "ReuseOrigins",
    "format_path",
    "parse_path",
]
