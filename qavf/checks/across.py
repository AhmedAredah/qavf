"""One dataset across many sources: ``across`` expanded before validation.

A dataset with ``across`` becomes one dataset per copy, ``<dataset>@<copy>``, each
with the dataset's own schema, checks, series, metrics and views, and its source
changed by the copy's overrides::

    across: [houston, dallas]              # connection names
    across:                                # copy name -> source overrides
      houston: {connection: houston}
      archive: {type: file, path: 's3://bucket/counts.parquet', secret: s3_archive}
    across: {city: [houston, dallas]}      # fills {city} in the source's path or table

``<dataset>@*`` stands for every copy wherever datasets are listed: a check's
``datasets``, ``run.require`` and the ``--dataset`` selection
(:func:`expand_dataset_names`). Like :mod:`qavf.checks.items`, this module imports
no rule code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

ACROSS_KEY = "across"
COPY_SEPARATOR = "@"
ALL_COPIES = "*"

_COPY_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]*")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_TEMPLATED_FIELDS = ("path", "table")
"""Source fields a ``{key: [values]}`` template fills."""


def copy_name(dataset: str, copy: str) -> str:
    """The name of one copy of ``dataset``: ``counts@houston``."""
    return f"{dataset}{COPY_SEPARATOR}{copy}"


class AcrossExpansion:
    """Expands every dataset's ``across`` into its copies in a raw configuration."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        connections = raw.get("connections")
        secrets = raw.get("secrets")
        self.connections = set(connections) if isinstance(connections, Mapping) else set()
        self.secrets: Mapping[str, Any] = secrets if isinstance(secrets, Mapping) else {}
        self.copies: dict[str, list[str]] = {}
        """Each expanded dataset's copy names, in order."""

    @classmethod
    def apply(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """``raw`` with copies in place of datasets using ``across`` (unchanged without)."""
        datasets = raw.get("datasets")
        if not isinstance(datasets, Mapping) or not any(
            isinstance(spec, Mapping) and spec.get(ACROSS_KEY) is not None
            for spec in datasets.values()
        ):
            return raw
        expansion = cls(raw)
        result = dict(raw)
        result["datasets"] = expansion.datasets(datasets)
        names = list(result["datasets"])
        result["datasets"] = {
            name: expansion.dataset_items(name, spec, names)
            for name, spec in result["datasets"].items()
        }
        if isinstance(raw.get("checks"), list):
            result["checks"] = expansion.items(raw["checks"], "checks", names)
        run = raw.get("run")
        if isinstance(run, Mapping) and isinstance(run.get("require"), list):
            require = expand_dataset_names(run["require"], names, "run.require")
            result["run"] = {**run, "require": require}
        return result

    def datasets(self, datasets: Mapping[str, Any]) -> dict[str, Any]:
        """Every dataset, each one with ``across`` replaced by its copies (in its place)."""
        expanded: dict[str, Any] = {}
        for name, spec in datasets.items():
            if not isinstance(spec, Mapping) or spec.get(ACROSS_KEY) is None:
                expanded[name] = spec
                continue
            names = []
            for copy, item in self.copies_of(name, spec).items():
                full = copy_name(name, copy)
                if full in datasets:
                    raise ValueError(
                        f"datasets.{name}.across: the copy {full!r} has the name of an "
                        "existing dataset; rename one"
                    )
                expanded[full] = item
                names.append(copy)
            self.copies[name] = names
        return expanded

    def copies_of(self, name: str, spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """The dataset specification of each copy of ``name``, by copy name."""
        where = f"datasets.{name}.across"
        if COPY_SEPARATOR in name:
            raise ValueError(f"{where}: a dataset using across cannot have '@' in its name")
        overrides = self.overrides(spec[ACROSS_KEY], where)
        base = {key: value for key, value in spec.items() if key != ACROSS_KEY}
        source = _source(base, where)
        result = {}
        for copy, override in overrides.items():
            item = {key: value for key, value in base.items() if key not in ("path", "format")}
            item["source"] = self.source(source, override, f"{where}.{copy}")
            result[copy] = item
        return result

    def overrides(self, across: Any, where: str) -> dict[str, dict[str, Any]]:
        """Each copy's source overrides, from any of the three forms of ``across``."""
        if isinstance(across, list):
            if not across or not all(isinstance(entry, str) for entry in across):
                raise ValueError(f"{where}: list the connection names to copy the dataset across")
            overrides: dict[str, dict[str, Any]] = {name: {"connection": name} for name in across}
        elif isinstance(across, Mapping) and _is_template(across):
            ((key, values),) = across.items()
            if not values or not all(isinstance(value, str | int) for value in values):
                raise ValueError(f"{where}.{key}: list the values that fill {{{key}}}")
            overrides = {str(value): {_TEMPLATE: {key: str(value)}} for value in values}
        elif isinstance(across, Mapping) and across:
            overrides = {}
            for copy, override in across.items():
                if override is None:
                    override = {}
                if not isinstance(override, Mapping):
                    raise ValueError(
                        f"{where}.{copy}: give the source settings that differ for this copy"
                    )
                overrides[str(copy)] = dict(override)
        else:
            raise ValueError(
                f"{where}: give a list of connection names, a mapping of copy names to source "
                "overrides, or a template such as {city: [houston, dallas]}"
            )
        for copy in overrides:
            if not _COPY_NAME.fullmatch(copy):
                raise ValueError(
                    f"{where}: copy name {copy!r} must use letters, digits, '_' or '-' only"
                )
        if len({copy.lower() for copy in overrides}) != len(overrides):
            raise ValueError(f"{where}: copy names must differ in more than case")
        return overrides

    def source(self, base: dict[str, Any], override: dict[str, Any], where: str) -> dict[str, Any]:
        """The copy's source: ``base`` changed by ``override``, checked for references."""
        override = dict(override)
        template = override.pop(_TEMPLATE, None)
        secret = override.pop("secret", None)
        if "type" in override and override["type"] != base.get("type"):
            source = override  # another kind of source: nothing of the base applies
        else:
            source = {**base, **override}
        if template is not None:
            source = _filled(source, template, where)
        connection = source.get("connection")
        if isinstance(connection, str) and connection not in self.connections:
            known = ", ".join(sorted(self.connections)) or "none are defined"
            raise ValueError(f"{where}: unknown connection {connection!r} (connections: {known})")
        if secret is not None:
            self.check_secret(secret, source, where)
        return source

    def check_secret(self, secret: Any, source: Mapping[str, Any], where: str) -> None:
        """The named secret exists and applies to the copy's location.

        Secrets apply to the URLs their ``scope`` covers, so naming one on a copy
        states which credential the copy is read with; a secret that would not
        apply is an error rather than silently unused.
        """
        if not isinstance(secret, str) or secret not in self.secrets:
            known = ", ".join(sorted(self.secrets)) or "none are defined"
            raise ValueError(f"{where}.secret: unknown secret {secret!r} (secrets: {known})")
        spec = self.secrets[secret]
        scope = spec.get("scope") if isinstance(spec, Mapping) else None
        prefixes = [scope] if isinstance(scope, str) else list(scope or [])
        location = source.get("path")
        if not prefixes or not isinstance(location, str):
            return
        if not any(location.startswith(str(prefix)) for prefix in prefixes):
            raise ValueError(
                f"{where}.secret: secret {secret!r} is scoped to {', '.join(map(str, prefixes))}"
                f" and does not apply to {location!r}"
            )

    def dataset_items(self, name: str, spec: Any, names: list[str]) -> Any:
        """``spec`` with ``<dataset>@*`` expanded in its check items."""
        if not isinstance(spec, Mapping) or not isinstance(spec.get("checks"), list):
            return spec
        return {**spec, "checks": self.items(spec["checks"], f"datasets.{name}.checks", names)}

    def items(self, items: list[Any], path: str, names: list[str]) -> list[Any]:
        """Check items with ``<dataset>@*`` expanded in their ``datasets`` lists."""
        result = []
        for index, item in enumerate(items):
            listed = item.get("datasets") if isinstance(item, Mapping) else None
            listed = [listed] if isinstance(listed, str) else listed
            if isinstance(listed, list) and all(isinstance(entry, str) for entry in listed):
                where = f"{path}[{index}].datasets"
                item = {**item, "datasets": expand_dataset_names(listed, names, where)}
            result.append(item)
        return result


_TEMPLATE = "\x00template"
"""Marks a copy's template values among its overrides (never a source field)."""


def _is_template(across: Mapping[str, Any]) -> bool:
    return len(across) == 1 and isinstance(next(iter(across.values())), list)


def _source(spec: Mapping[str, Any], where: str) -> dict[str, Any]:
    """The dataset's source settings, with the ``path`` shorthand spelled out."""
    if spec.get("path") is not None:
        source: dict[str, Any] = {"type": "file", "path": spec["path"]}
        if spec.get("format") is not None:
            source["format"] = spec["format"]
        return source
    if isinstance(spec.get("source"), Mapping):
        return dict(spec["source"])
    raise ValueError(f"{where}: the dataset needs a 'path' or 'source' to copy")


def _filled(source: dict[str, Any], values: Mapping[str, str], where: str) -> dict[str, Any]:
    """``source`` with ``{key}`` filled in its path and table."""
    filled = dict(source)
    used = False
    for field in _TEMPLATED_FIELDS:
        text = filled.get(field)
        if not isinstance(text, str):
            continue
        for key, value in values.items():
            if f"{{{key}}}" in text:
                used = True
                text = text.replace(f"{{{key}}}", value)
        if unfilled := _PLACEHOLDER.search(text):
            raise ValueError(
                f"{where}: {{{unfilled.group(1)}}} in the source's {field} is not filled; "
                f"the template gives {', '.join('{' + key + '}' for key in values)}"
            )
        filled[field] = text
    if not used:
        keys = ", ".join("{" + key + "}" for key in values)
        raise ValueError(
            f"{where}: the template key {keys} appears in neither the source's path nor its table"
        )
    return filled


def expand_dataset_names(names: Iterable[str], known: Iterable[str], where: str) -> list[str]:
    """``names`` with each ``<dataset>@*`` (or ``<dataset>@*.<view>``) replaced by every
    copy of that dataset among ``known``, in order.

    Raises:
        ValueError: for ``<dataset>@*`` naming a dataset that has no copies.
    """
    known = list(known)
    result: list[str] = []
    for name in names:
        base, wildcard, view = name.partition(f"{COPY_SEPARATOR}{ALL_COPIES}")
        if not wildcard:
            result.append(name)
            continue
        if view and not view.startswith("."):
            raise ValueError(f"{where}: {name!r} must be {base}@* or {base}@*.<view>")
        prefix = f"{base}{COPY_SEPARATOR}"
        copies = [entry for entry in known if entry.startswith(prefix) and "." not in entry]
        if not copies:
            raise ValueError(f"{where}: {name!r} names no copies; {base!r} has no 'across'")
        result.extend(f"{entry}{view}" for entry in copies if f"{entry}{view}" not in result)
    return result


__all__ = [
    "ACROSS_KEY",
    "ALL_COPIES",
    "COPY_SEPARATOR",
    "AcrossExpansion",
    "copy_name",
    "expand_dataset_names",
]
