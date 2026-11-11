"""The fully resolved configuration, with where each value came from (``qavf resolve``).

:func:`resolve` reads a configuration the way a run does (``include``, ``vars`` and
``${ENV}``, ``defaults``, ``across`` and ``overrides``), validates it without
reading any data, and returns the values a run would use, with secrets redacted
(:func:`qavf.redaction.redact`). :class:`ResolvedConfig` renders them as YAML or
JSON; with origins, each value names the file, line and column it was written at
and how it got there (:mod:`qavf.origins`), and the applied overrides are listed.

The output is deterministic: values keep the configuration's order, and file
names are relative to the root configuration's directory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from qavf.origins import ConfigPath, Origin, OriginIndex, format_path
from qavf.overrides import AppliedOverride
from qavf.redaction import redact
from qavf.settings import QAVFSettings, load_settings

ResolvedFormat = Literal["yaml", "json"]


@dataclass(frozen=True)
class ResolvedConfig:
    """A configuration as a run would use it, redacted, with its origins."""

    configuration: dict[str, Any]
    """The resolved settings as written or derived (not every built-in default),
    secrets masked."""
    origins: dict[str, Origin] = field(default_factory=dict)
    """Where each value was written, by its path (``datasets.x.schema.min_rows``):
    every scalar, empty or list value, and a mapping whose origin is noted (an
    ``across`` copy, a substituted variable) or lies in another file than its parent."""
    overrides: tuple[AppliedOverride, ...] = ()
    """The overrides applied, in order, with what each changed."""
    source: Path | None = None
    """The root configuration file."""
    included: tuple[Path, ...] = ()
    """Every file ``include`` merged in."""

    def render(self, fmt: ResolvedFormat = "yaml", *, origins: bool = False) -> str:
        """The configuration as YAML (origins as comments) or JSON (an ``origins`` map)."""
        if fmt == "json":
            return self.to_json(origins=origins)
        return self.to_yaml(origins=origins)

    def to_json(self, *, origins: bool = False) -> str:
        """JSON text: the configuration alone, or with its ``origins`` and ``overrides``."""
        if not origins:
            return json.dumps(self.configuration, indent=2, ensure_ascii=False) + "\n"
        document = {
            "configuration": self.configuration,
            "origins": {path: origin.describe() for path, origin in self.origins.items()},
            "overrides": [_override(applied) for applied in self.overrides],
        }
        return json.dumps(document, indent=2, ensure_ascii=False) + "\n"

    def to_yaml(self, *, origins: bool = False) -> str:
        """YAML text; with ``origins``, each value's origin as a trailing comment and the
        applied overrides in a leading comment."""
        if not origins:
            return CommentedYaml({}).render(self.configuration)
        comments = {path: origin.describe() for path, origin in self.origins.items()}
        header = []
        if self.source is not None:
            header.append(f"# Resolved from {self.source.name}; origins are file:line:column.")
        if self.overrides:
            header.append(f"# Overrides applied ({len(self.overrides)}):")
            header.extend(f"#   {applied.describe()}" for applied in self.overrides)
        body = CommentedYaml(comments).render(self.configuration)
        return "\n".join(header) + ("\n" if header else "") + body


def _override(applied: AppliedOverride) -> dict[str, Any]:
    return {
        "instance": applied.instance,
        "target": applied.target,
        "section": applied.section,
        "fields": list(applied.fields),
        "origin": applied.origin,
        "line": applied.line,
    }


def resolve(config: str | os.PathLike[str] | QAVFSettings | Mapping[str, Any]) -> ResolvedConfig:
    """The fully resolved configuration, without reading any data.

    ``config`` is a YAML/JSON file (origins are known), loaded settings, or a
    configuration mapping (validated here; no origins).

    Raises:
        pydantic.ValidationError: for an invalid configuration; read from a file,
            the error carries where each invalid value was written.
        ValueError: for an unreadable configuration (with its file and line).
    """
    if isinstance(config, QAVFSettings):
        settings = config
    elif isinstance(config, Mapping):
        settings = QAVFSettings.model_validate(dict(config))
    else:
        settings = load_settings(os.fspath(config))
    values = redact(settings.model_dump(mode="json", by_alias=True, exclude_unset=True))
    index = settings.origins
    located = OriginWalk(index).collect(values) if index is not None else {}
    return ResolvedConfig(
        configuration=values,
        origins=located,
        overrides=settings.applied_overrides,
        source=settings.source_path,
        included=settings.included_paths,
    )


class OriginWalk:
    """Looks up the origin of each value of a resolved configuration."""

    def __init__(self, index: OriginIndex) -> None:
        self.index = index

    def collect(self, values: Mapping[str, Any]) -> dict[str, Origin]:
        found: dict[str, Origin] = {}
        for path, value, parent in self._walk(values):
            origin = self.index.find(path)
            if origin is None:
                continue
            leaf = not isinstance(value, Mapping) or not value
            moved = parent is None or origin.note or parent.file != origin.file
            if leaf or isinstance(value, list) or moved:
                found[format_path(path)] = origin
        return found

    def _walk(self, values: Mapping[str, Any]) -> Iterator[tuple[ConfigPath, Any, Origin | None]]:
        """Every value below the root, depth first in order, with its parent's origin."""
        stack: list[tuple[ConfigPath, Any]] = [
            ((key,), child) for key, child in reversed(list(values.items()))
        ]
        while stack:
            path, value = stack.pop()
            parent = self.index.find(path[:-1]) if len(path) > 1 else None
            yield path, value, parent
            if isinstance(value, Mapping):
                stack.extend(((*path, key), child) for key, child in reversed(list(value.items())))
            elif isinstance(value, list):
                stack.extend(
                    ((*path, index), child) for index, child in reversed(list(enumerate(value)))
                )


class CommentedYaml:
    """Block-style YAML with an optional trailing comment per value (by its path)."""

    def __init__(self, comments: Mapping[str, str]) -> None:
        self.comments = comments

    def render(self, values: Mapping[str, Any]) -> str:
        if not values:
            return "{}\n"
        lines: list[str] = []
        self._mapping(values, (), 0, lines, None)
        return "\n".join(lines) + "\n"

    def _mapping(
        self,
        values: Mapping[str, Any],
        path: ConfigPath,
        indent: int,
        lines: list[str],
        first: str | None,
    ) -> None:
        for position, (key, child) in enumerate(values.items()):
            prefix = first if position == 0 and first is not None else " " * indent
            self._entry(f"{prefix}{scalar(key)}:", (*path, key), child, indent, lines)

    def _entry(
        self, head: str, path: ConfigPath, value: Any, indent: int, lines: list[str]
    ) -> None:
        if isinstance(value, Mapping) and value:
            lines.append(self._commented(head, path))
            self._mapping(value, path, indent + 2, lines, None)
        elif isinstance(value, list) and value:
            lines.append(self._commented(head, path))
            self._sequence(value, path, indent + 2, lines)
        else:
            lines.append(self._commented(f"{head} {scalar(value)}", path))

    def _sequence(
        self, values: Sequence[Any], path: ConfigPath, indent: int, lines: list[str]
    ) -> None:
        dash = " " * indent + "- "
        for position, item in enumerate(values):
            here = (*path, position)
            if isinstance(item, Mapping) and item:
                self._mapping(item, here, indent + 2, lines, dash)
            elif isinstance(item, list) and item:
                lines.append(self._commented(dash.rstrip(), here))
                self._sequence(item, here, indent + 2, lines)
            else:
                lines.append(self._commented(f"{dash}{scalar(item)}", here))

    def _commented(self, line: str, path: ConfigPath) -> str:
        comment = self.comments.get(format_path(path))
        return f"{line}  # {comment}" if comment else line


def scalar(value: Any) -> str:
    """One YAML flow scalar (or empty collection) on a single line."""
    text = yaml.safe_dump(
        value, default_flow_style=True, allow_unicode=True, width=1_000_000, sort_keys=False
    ).rstrip("\n")
    text = text.removesuffix("\n...").rstrip("\n")
    if "\n" in text:  # a multi-line string: JSON's quoting is valid YAML on one line
        return json.dumps(value, ensure_ascii=False)
    return text


__all__ = ["CommentedYaml", "OriginWalk", "ResolvedConfig", "ResolvedFormat", "resolve", "scalar"]
