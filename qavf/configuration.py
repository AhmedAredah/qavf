"""Strict YAML/JSON configuration parsing shared by every public entry point."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_YAML_MERGE_KEY = object()


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that rejects duplicate mapping keys instead of keeping the last."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self.validated_nodes: set[int] = set()


def _validate_yaml_keys(loader: _UniqueKeyLoader, node: yaml.Node) -> None:
    """Check raw mappings before merge flattening, including referenced anchors.

    An explicit override of an inherited merge value is legal YAML; two explicit
    occurrences in the same mapping are not. Checking before flattening preserves
    that distinction, including when an anchor is consumed before its own mapping.
    """
    if id(node) in loader.validated_nodes:
        return
    loader.validated_nodes.add(id(node))
    if isinstance(node, yaml.MappingNode):
        first: dict[Any, tuple[int, int]] = {}
        for key_node, value_node in node.value:
            key = (
                _YAML_MERGE_KEY
                if key_node.tag == "tag:yaml.org,2002:merge"
                else loader.construct_object(key_node, deep=False)
            )
            location = (key_node.start_mark.line + 1, key_node.start_mark.column + 1)
            try:
                if key in first:
                    earlier = first[key]
                    shown = "<<" if key is _YAML_MERGE_KEY else key
                    raise ValueError(
                        f"duplicate key {shown!r} at line {location[0]}, column {location[1]} "
                        f"(first defined at line {earlier[0]}, column {earlier[1]}); "
                        "a repeated key would silently replace the earlier entry"
                    )
                first[key] = location
            except TypeError:
                pass  # PyYAML supplies the precise invalid-mapping-key error.
            _validate_yaml_keys(loader, value_node)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _validate_yaml_keys(loader, child)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> Any:
    _validate_yaml_keys(loader, node)
    loader.flatten_mapping(node)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"duplicate key {key!r}; a repeated key would silently replace the earlier entry"
            )
        result[key] = value
    return result


def _describe_yaml_error(exc: yaml.YAMLError) -> str:
    """One-line description of a YAML error with its 1-based line and column."""
    problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    mark = getattr(exc, "problem_mark", None)
    if mark is None:
        return str(problem)
    return f"{problem} (line {mark.line + 1}, column {mark.column + 1})"


def expand_env(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    """Recursively replace ``${NAME}`` / ``${NAME:-default}`` in strings.

    Raises:
        ValueError: for an unset variable, cyclic input, or excessive nesting.

    Shared YAML anchors retain their sharing in the expanded copy. Expanding
    each occurrence independently can turn a small alias graph into an
    exponentially large object tree.
    """
    env = os.environ if environ is None else environ
    memo: dict[int, Any] = {}
    active: set[int] = set()

    def substitute(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        raise ValueError(f"environment variable {name!r} is not set")

    def expand(item: Any) -> Any:
        if isinstance(item, str):
            return _ENV_REFERENCE.sub(substitute, item)
        if not isinstance(item, (dict, list)):
            return item
        identity = id(item)
        if identity in active:
            raise ValueError("configuration contains a cyclic reference")
        if identity in memo:
            return memo[identity]
        active.add(identity)
        try:
            expanded = (
                {key: expand(child) for key, child in item.items()}
                if isinstance(item, dict)
                else [expand(child) for child in item]
            )
            memo[identity] = expanded
            return expanded
        finally:
            active.remove(identity)

    try:
        return expand(value)
    except RecursionError as exc:
        raise ValueError("configuration is nested too deeply") from exc


def read_config_mapping(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Parse one configuration document before callers assemble inline contracts."""
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8")
    match source.suffix.lower():
        case ".yaml" | ".yml":
            try:
                raw = yaml.load(text, Loader=_UniqueKeyLoader)  # a SafeLoader subclass
            except RecursionError as exc:
                raise ValueError(f"{source}: YAML is nested too deeply") from exc
            except yaml.YAMLError as exc:
                raise ValueError(f"{source}: invalid YAML: {_describe_yaml_error(exc)}") from exc
        case ".json":
            try:
                raw = json.loads(text, object_pairs_hook=_unique_json_object)
            except RecursionError as exc:
                raise ValueError(f"{source}: JSON is nested too deeply") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source}: invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
                ) from exc
        case other:
            raise ValueError(f"unsupported configuration format {other!r}; use YAML or JSON")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: top level of the configuration must be a mapping")
    raw.pop("$schema", None)
    return expand_env(raw), source
