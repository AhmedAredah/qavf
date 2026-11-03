"""Convert QAVF 0.6 configurations (rules by code) to named checks.

``qavf migrate old.yaml -o new.yaml`` rewrites a configuration once:

* ``rules.<CODE>.params`` lists become one ``checks`` item per entry, on the
  entry's dataset (or at the top level for checks relating datasets), with the
  catalog's setting renames; rule-level parameters, ``severity`` and
  ``enabled: false`` are copied onto every generated item;
* the series and metrics of temporal and spatial checks become named entries
  of their dataset (``series``/``metrics``); identical ones merge into one entry,
  named ``<value>_by_<group>``, that the items refer to;
* ``datasets.<d>.expressions`` become ``check: expression`` items and
  ``assertions`` top-level ``check: sql`` items;
* ``engine.fail_on``, ``fail_on_incomplete`` and ``require`` move to ``run``;
  ``require`` entries become check or dataset names;
* ``engine.run_unconfigured_rules``, ``include_categories`` and
  ``exclude_categories`` are removed (only configured checks run now).

PyYAML does not keep comments, so the file is rewritten cleanly; the comment
block at the top of the original file is kept.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qavf.checks.catalog import BY_RULE, CONTRACT_RULES, Decompiled, PluginDefinition
from qavf.checks.legacy import LEGACY_ENGINE_KEYS, first_legacy_key
from qavf.configuration import parse_config_text
from qavf.engine.registry import Rule, RuleRegistry, default_registry

_LIST_DRIVEN = {"INP-006": "expression", "INP-007": "sql"}
_TOP_ORDER = ("project", "datasets", "connections", "secrets", "checks", "run", "engine")


class MigrationError(ValueError):
    """A QAVF 0.6 setting that cannot be converted automatically."""


@dataclass
class Migration:
    """The converted configuration and what the conversion did."""

    config: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    """Each change made, in order, and anything that needs the user's attention."""

    @property
    def changed(self) -> bool:
        return bool(self.notes)

    def yaml(self, header: str = "") -> str:
        """The configuration as YAML, after ``header`` (comment lines)."""
        body = yaml.dump(
            self.config,
            Dumper=_CompactDumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=100,
        )
        return header + body


class _CompactDumper(yaml.SafeDumper):
    """YAML close to hand-written: short scalar lists and small mappings inline,
    multi-line text as literal blocks."""


_SCALARS = (str, int, float, bool, type(None))


def _represent_text(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


def _represent_list(dumper: yaml.SafeDumper, value: list[Any]) -> yaml.Node:
    flow = all(isinstance(item, _SCALARS) and "\n" not in str(item) for item in value)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=flow)


def _represent_mapping(dumper: yaml.SafeDumper, value: dict[Any, Any]) -> yaml.Node:
    flow = 0 < len(value) <= 4 and all(
        isinstance(item, _SCALARS) and "\n" not in str(item) for item in value.values()
    )
    return dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=flow)


_CompactDumper.add_representer(str, _represent_text)
_CompactDumper.add_representer(list, _represent_list)
_CompactDumper.add_representer(dict, _represent_mapping)


def plugin_registry(raw: Mapping[str, Any]) -> RuleRegistry:
    """Rules a configuration can use: built-ins, its plugins and ``@register`` rules."""
    plugins = raw.get("plugins") or {}
    registry = RuleRegistry().discover(
        packages=plugins.get("packages", ["qavf.rules"]),
        modules=plugins.get("modules", []),
        use_entry_points=bool(plugins.get("entry_points", True)),
    )
    for rule_cls in default_registry:
        registry.register(rule_cls)
    return registry


class Migrator:
    """Convert one QAVF 0.6 configuration mapping."""

    def __init__(
        self, registry: RuleRegistry | None = None, *, drop_disabled: bool = False
    ) -> None:
        self.registry = registry
        self.drop_disabled = drop_disabled

    def migrate(self, raw: Mapping[str, Any]) -> Migration:
        """The named-check form of ``raw`` (unchanged when it has no old keys)."""
        config = copy.deepcopy(dict(raw))
        migration = Migration(config)
        if first_legacy_key(config) is None:
            return migration
        registry = self.registry if self.registry is not None else plugin_registry(config)
        top: list[dict[str, Any]] = list(config.get("checks") or [])
        generated: dict[str, list[str]] = {}
        self._expressions(config, migration)
        self._assertions(config, top, migration)
        self._rules(config, top, generated, registry, migration)
        if top:
            config["checks"] = top
        self._engine(config, generated, migration)
        migration.config = _ordered(config)
        return migration

    # ------------------------------------------------------------ datasets
    def _expressions(self, config: dict[str, Any], migration: Migration) -> None:
        for name, spec in (config.get("datasets") or {}).items():
            if not isinstance(spec, dict) or "expressions" not in spec:
                continue
            expressions = spec.pop("expressions") or []
            items = [{"check": "expression", **dict(expression)} for expression in expressions]
            spec["checks"] = [*(spec.get("checks") or []), *items]
            migration.notes.append(
                f"datasets.{name}.expressions -> {len(items)} expression check(s) on {name}"
            )

    def _assertions(
        self, config: dict[str, Any], top: list[dict[str, Any]], migration: Migration
    ) -> None:
        if "assertions" not in config:
            return
        assertions = config.pop("assertions") or []
        top.extend({"check": "sql", **dict(assertion)} for assertion in assertions)
        migration.notes.append(f"assertions -> {len(assertions)} top-level sql check(s)")

    # --------------------------------------------------------------- rules
    def _rules(
        self,
        config: dict[str, Any],
        top: list[dict[str, Any]],
        generated: dict[str, list[str]],
        registry: RuleRegistry,
        migration: Migration,
    ) -> None:
        rules = config.pop("rules", None) or {}
        for code, settings in rules.items():
            settings = dict(settings or {})
            enabled = settings.get("enabled", True)
            severity = settings.get("severity")
            params = dict(settings.get("params") or {})
            if code in CONTRACT_RULES:
                self._contract(code, enabled, severity, params, migration)
                continue
            if code in _LIST_DRIVEN:
                self._list_driven(config, top, code, enabled, severity, migration)
                continue
            items = self._decompile(code, params, registry)
            if not items:
                migration.notes.append(
                    f"rules.{code}: nothing configured to check; removed (only configured "
                    "checks run now)"
                )
                continue
            if not enabled and self.drop_disabled:
                migration.notes.append(f"rules.{code}: disabled; {len(items)} item(s) dropped")
                continue
            names = []
            for decompiled in items:
                item = dict(decompiled.item)
                if severity is not None and "severity" not in item:
                    item["severity"] = severity
                if not enabled:
                    item["enabled"] = False
                names.append(self._place(config, top, decompiled.dataset, item, migration))
            generated[code] = names
            migration.notes.append(
                f"rules.{code} -> {len(items)} {items[0].item['check']} check(s)"
                + ("" if enabled else " with enabled: false")
            )

    def _decompile(self, code: str, params: dict[str, Any], registry: RuleRegistry) -> list[Any]:
        builtin = BY_RULE.get(code)
        if builtin is not None:
            return builtin.decompile(params)
        if code not in registry:
            raise MigrationError(f"rules.{code}: unknown rule; install its plugin before migrating")
        rule_cls: type[Rule] = registry.get(code)
        if (
            not params
            and rule_cls.Params.model_fields
            and any(field.is_required() for field in rule_cls.Params.model_fields.values())
        ):
            return []
        return [
            Decompiled(None, item.item) for item in PluginDefinition(rule_cls).decompile(params)
        ]

    @staticmethod
    def _declare(config: dict[str, Any], names: Iterable[str], migration: Migration) -> None:
        """Add an entry for each dataset a check refers to that is not configured: QAVF
        0.6 rules could name datasets supplied only at run time."""
        configured = config.setdefault("datasets", {})
        for name in names:
            if name not in configured:
                configured[name] = {}
                migration.notes.append(
                    f"datasets.{name} added: a check refers to it (supply it at run time)"
                )

    def _place(
        self,
        config: dict[str, Any],
        top: list[dict[str, Any]],
        dataset: str | None,
        item: dict[str, Any],
        migration: Migration,
    ) -> str:
        if dataset is None:
            top.append(item)
            named = item.get("datasets")
            datasets = [n for n in named if isinstance(n, str)] if isinstance(named, list) else []
            reference = item.get("reference") if item["check"] == "coverage" else None
            self._declare(config, [*datasets, *([reference] if reference else [])], migration)
            return str(item.get("name") or _default_name(item["check"], datasets))
        self._declare(config, [dataset], migration)
        spec = config["datasets"][dataset]
        for setting, block in _NAMED_BLOCKS.items():
            if isinstance(item.get(setting), dict):
                item[setting] = block.hoist(spec, item[setting])
        spec["checks"] = [*(spec.get("checks") or []), item]
        return str(item.get("name") or _default_name(item["check"], [dataset]))

    def _contract(
        self,
        code: str,
        enabled: bool,
        severity: Any,
        params: dict[str, Any],
        migration: Migration,
    ) -> None:
        notes = [f"rules.{code}: removed; schema contracts always run where they are declared"]
        if params.get("datasets") is not None:
            notes.append(f"rules.{code}.params.datasets dropped")
        if not enabled:
            notes.append(f"rules.{code} was disabled: remove the contract from the schema instead")
        if severity is not None:
            notes.append(f"rules.{code}.severity has no equivalent: set severity per column limit")
        migration.notes.extend(notes)

    def _list_driven(
        self,
        config: dict[str, Any],
        top: list[dict[str, Any]],
        code: str,
        enabled: bool,
        severity: Any,
        migration: Migration,
    ) -> None:
        check = _LIST_DRIVEN[code]
        lists = [top] + [
            spec.get("checks") or [] for spec in (config.get("datasets") or {}).values()
        ]
        for items in lists:
            for item in items:
                if item.get("check") != check:
                    continue
                if severity is not None and "severity" not in item:
                    item["severity"] = severity
                if not enabled:
                    item["enabled"] = False
        if not enabled and self.drop_disabled:
            for spec in (config.get("datasets") or {}).values():
                if spec.get("checks"):
                    spec["checks"] = [i for i in spec["checks"] if i.get("check") != check]
            top[:] = [item for item in top if item.get("check") != check]
        migration.notes.append(f"rules.{code}: settings applied to the {check} checks")

    # -------------------------------------------------------------- engine
    def _engine(
        self, config: dict[str, Any], generated: Mapping[str, list[str]], migration: Migration
    ) -> None:
        engine = config.get("engine")
        if not isinstance(engine, dict):
            return
        run = dict(config.get("run") or {})
        for key in ("fail_on", "fail_on_incomplete"):
            if key in engine:
                run[key] = engine.pop(key)
                migration.notes.append(f"engine.{key} -> run.{key}")
        if "require" in engine:
            run["require"] = self._requirements(config, engine.pop("require") or [], generated)
            migration.notes.append("engine.require -> run.require (as check or dataset names)")
        if engine.pop("run_unconfigured_rules", None) is not None:
            migration.notes.append(
                "engine.run_unconfigured_rules removed: only configured checks run"
            )
        include = engine.pop("include_categories", None)
        exclude = engine.pop("exclude_categories", None)
        if include:
            migration.notes.append(
                "engine.include_categories removed; run with " + _flags("--tags", include)
            )
        if exclude:
            migration.notes.append(
                "engine.exclude_categories removed; run with " + _flags("--skip-tags", exclude)
            )
        for key in LEGACY_ENGINE_KEYS:
            engine.pop(key, None)
        if run:
            config["run"] = run
        if not engine:
            config.pop("engine")

    @staticmethod
    def _requirements(
        config: Mapping[str, Any], entries: Iterable[str], generated: Mapping[str, list[str]]
    ) -> list[str]:
        datasets = config.get("datasets") or {}
        names: list[str] = []
        for entry in entries:
            code = entry.split("[", 1)[0].strip()
            detail = entry.split("[", 1)[1].rstrip()[:-1] if "[" in entry else ""
            if code in generated:
                names.extend(generated[code])
            elif detail and detail.split(":", 1)[0].split(".", 1)[0] in datasets:
                names.append(detail.split(":", 1)[0].split(".", 1)[0])
            else:
                names.extend(
                    name
                    for name, spec in datasets.items()
                    if isinstance(spec, Mapping) and (spec.get("schema") or spec.get("checks"))
                )
        return list(dict.fromkeys(names))


class NamedBlock:
    """Moves inline definitions into a dataset's named ``series``/``metrics`` section.

    Identical definitions on one dataset share one entry, named ``<value>_by_<group>``
    (``<value>_by_<entity>`` for a metric), with a numeric suffix when two different
    definitions would take the same name.
    """

    def __init__(self, section: str, grouping: str) -> None:
        self.section = section
        self.grouping = grouping
        """The definition key naming what the values are grouped by."""

    def name(self, definition: Mapping[str, Any]) -> str:
        value = definition.get("value")
        group = definition.get(self.grouping)
        parts = [value if isinstance(value, str) else self.section]
        groups = group if isinstance(group, list) else [group] if isinstance(group, str) else []
        if groups:
            parts.append("by_" + "_".join(str(name) for name in groups))
        return re.sub(r"\W+", "_", "_".join(parts)).strip("_") or self.section

    def hoist(self, spec: dict[str, Any], definition: dict[str, Any]) -> str:
        """The name of ``definition`` in ``spec``'s section, adding it when new."""
        defined = spec.setdefault(self.section, {})
        for name, existing in defined.items():
            if existing == definition:
                return str(name)
        base = self.name(definition)
        name, number = base, 1
        while name in defined:
            number += 1
            name = f"{base}_{number}"
        defined[name] = definition
        return name


_NAMED_BLOCKS = {
    "series": NamedBlock("series", "group_by"),
    "metric": NamedBlock("metrics", "entity"),
}
"""Check settings whose inline definitions become named dataset entries."""


def _default_name(check: str, datasets: list[str]) -> str:
    return f"{check} on {' and '.join(datasets)}" if datasets else check


def _flags(flag: str, values: Iterable[str]) -> str:
    return " ".join(f"{flag} {value}" for value in values)


def _ordered(config: dict[str, Any]) -> dict[str, Any]:
    ordered = {key: config[key] for key in _TOP_ORDER if key in config}
    return ordered | {key: value for key, value in config.items() if key not in ordered}


def leading_comments(text: str) -> str:
    """The comment block at the top of a configuration file."""
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("#") or (lines and not line.strip()):
            lines.append(line)
            continue
        break
    while lines and not lines[-1].strip():
        lines.pop()
    return "".join(lines) + ("\n" if lines else "")


def migrate_text(text: str, source: Path, *, drop_disabled: bool = False) -> tuple[Migration, str]:
    """Convert configuration ``text`` (the content of ``source``); returns the new text."""
    raw = parse_config_text(text, source, expand=False)
    migration = Migrator(drop_disabled=drop_disabled).migrate(raw)
    if source.suffix.lower() == ".json":
        import json

        return migration, json.dumps(migration.config, indent=2, default=str) + "\n"
    return migration, migration.yaml(leading_comments(text))


def migrate_file(
    path: str | Path, output: str | Path | None = None, *, drop_disabled: bool = False
) -> Migration:
    """Convert the configuration at ``path``, writing ``output`` when given."""
    source = Path(path).resolve()
    migration, text = migrate_text(
        source.read_text(encoding="utf-8"), source, drop_disabled=drop_disabled
    )
    if output is not None:
        Path(output).write_text(text, encoding="utf-8")
    return migration


__all__ = [
    "Migration",
    "MigrationError",
    "Migrator",
    "leading_comments",
    "migrate_file",
    "migrate_text",
]
