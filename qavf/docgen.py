"""Generate Markdown reference pages for rules, sources and comparison components.

Used by ``scripts/generate_docs.py`` (and a test that keeps ``docs/`` current),
and by plugin authors who want the same reference for their own rules::

    from qavf.docgen import rules_markdown
    print(rules_markdown(my_registry, title="My rules"))
"""

from __future__ import annotations

import re
import types
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from qavf.engine.registry import RuleRegistry
from qavf.sources.base import SourceRegistry

if TYPE_CHECKING:
    from qavf.plugins.registry import ComponentRegistry


def _type_name(annotation: Any) -> str:
    """Render type structure, excluding process-specific validator metadata."""
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _type_name(arguments[0])
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(argument) for argument in arguments)
    if origin is Literal:
        return f"Literal[{', '.join(repr(argument) for argument in arguments)}]"
    if origin is not None:
        return f"{_type_name(origin)}[{', '.join(_type_name(arg) for arg in arguments)}]"
    if annotation is Ellipsis:
        return "..."
    text = getattr(annotation, "__name__", None) if isinstance(annotation, type) else None
    if text is None:
        text = repr(annotation)
    text = re.sub(r"\b(?:typing|qavf\.[\w.]+|pathlib|datetime|collections\.abc)\.", "", text)
    text = text.replace("NoneType", "None").replace("<class '", "").replace("'>", "")
    return _modern_optional(text)


def _modern_optional(text: str) -> str:
    """Rewrite ``Optional[X]`` as ``X | None`` (Python versions render unions differently)."""
    marker = "Optional["
    while (start := text.find(marker)) != -1:
        depth, end = 1, start + len(marker)
        while depth:
            depth += {"[": 1, "]": -1}.get(text[end], 0)
            end += 1
        inner = text[start + len(marker) : end - 1]
        text = f"{text[:start]}{inner} | None{text[end:]}"
    return text


def _default(field: Any) -> str:
    if field.default_factory is not None:
        try:
            value = field.default_factory()
        except TypeError:
            return "*(computed)*"
    elif field.default is PydanticUndefined:
        return "**required**"
    else:
        value = field.default
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if hasattr(value, "value") and not isinstance(value, int | float | str):
        value = value.value
    if value == () or value == []:
        return "`[]`"
    if value == {}:
        return "`{}`"
    return _code_cell(repr(value) if not isinstance(value, str) else value)


def _code_cell(value: str) -> str:
    """Keep arbitrary values in one Markdown table cell and one physical line."""
    text = value.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t").replace("|", "\\|")
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    delimiter = "`" * (longest + 1)
    # Spaces disambiguate code delimiters from literal leading/trailing backticks.
    padding = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{delimiter}{padding}{text}{padding}{delimiter}"


def _escape(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def params_table(model: type[BaseModel], skip: Iterable[str] = ()) -> str:
    """A Markdown table of ``model``'s fields: name, type, default, description."""
    skipped = set(skip)
    rows = ["| Parameter | Type | Default | Description |", "|---|---|---|---|"]
    for name, field in model.model_fields.items():
        if name in skipped:
            continue
        rows.append(
            f"| {_code_cell(field.alias or name)} | {_code_cell(_type_name(field.annotation))} | "
            f"{_default(field)} | {_escape(field.description or '')} |"
        )
    return "\n".join(rows) if len(rows) > 2 else "*No parameters.*"


def _nested_models(annotation: Any, seen: set[type[BaseModel]]) -> list[type[BaseModel]]:
    """Parameter models referenced (at any depth) by a field annotation."""
    found: list[type[BaseModel]] = []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation not in seen:
            seen.add(annotation)
            found.append(annotation)
            for field in annotation.model_fields.values():
                found += _nested_models(field.annotation, seen)
        return found
    for argument in getattr(annotation, "__args__", ()) or ():
        found += _nested_models(argument, seen)
    return found


def model_reference(model: type[BaseModel], level: int = 4, skip: Iterable[str] = ()) -> list[str]:
    """Tables for ``model`` (without the ``skip`` fields) and every parameter model
    nested inside it."""
    skipped = set(skip)
    lines = [params_table(model, skipped), ""]
    seen: set[type[BaseModel]] = {model}
    for name, field in model.model_fields.items():
        if name in skipped:
            continue
        for nested in _nested_models(field.annotation, seen):
            doc = (nested.__doc__ or "").strip().splitlines()[0] if nested.__doc__ else ""
            lines += [
                f"{'#' * level} {_model_heading(nested.__name__)}",
                "",
                _escape(doc),
                "",
                params_table(nested),
                "",
            ]
    return lines


def _model_heading(name: str) -> str:
    """A nested model's heading: an inline ``series``/``metric`` definition by its setting."""
    if name.startswith("Inline_"):
        return f"Inline `{name.split('_')[1]}` definition"
    return f"`{name.lstrip('_')}`"


_PLACES = {"dataset": "`datasets.<name>.checks`", "top": "top-level `checks`"}


def rules_markdown(registry: RuleRegistry, title: str = "Check reference") -> str:
    """Markdown reference for every check in ``registry``, by check name and category.

    Each check lists its own settings (those of its check item, after the common
    fields); schema contracts, which come from ``schema`` rather than check items,
    are listed with what configures them.
    """
    from qavf.checks.catalog import definition_for
    from qavf.checks.items import COMMON_FIELDS

    rules = list(registry)
    definitions = {rule.rule_id: definition_for(rule) for rule in rules}
    lines = [
        f"# {title}",
        "",
        "<!-- Generated by scripts/generate_docs.py; do not edit by hand. -->",
        "",
        "A check item names its check (`check: missing_periods`); the code (`TMP-004`)",
        "is its stable audit ID in reports. Besides the settings listed here, every item",
        "takes the common fields `name`, `severity`, `tags`, `enabled` and `description`.",
        "The checks listing `where` take it: every check placed in a dataset's `checks`",
        "(which then assesses only the matching rows, with `min_applicable_rows`) and",
        "`compare`; the other top-level checks name a view (`<dataset>.<view>`) of each",
        "dataset instead. `qavf list-checks` prints the same catalog.",
        "",
        "| Check | Code | Goes in | Category | Default severity | Summary |",
        "|---|---|---|---|---|---|",
    ]
    for rule in rules:
        definition = definitions[rule.rule_id]
        place = _PLACES[definition.placement] if definition.configurable else "`schema`"
        lines.append(
            f"| [`{rule.check_name}`](#{_anchor(rule.check_name)}) | {rule.rule_id} | {place} | "
            f"{rule.category} | {rule.default_severity.value} | {_escape(rule.name)} |"
        )
    for category in registry.categories():
        lines += ["", f"## {category.title()}", ""]
        for rule in (r for r in rules if r.category == category):
            definition = definitions[rule.rule_id]
            lines += [
                f"### {rule.check_name}",
                "",
                f"**{_escape(rule.name)}** · code `{rule.rule_id}` · default severity "
                f"`{rule.default_severity.value}`",
                "",
                _escape(rule.description),
                "",
            ]
            if definition.configurable:
                lines += [f"Goes in {_PLACES[definition.placement]}.", ""]
                skip = tuple(
                    field
                    for field in COMMON_FIELDS
                    if field != "where" or not definition.takes_where
                )
                lines += model_reference(definition.item_model(), skip=skip)
            else:
                lines += [
                    "Comes from the dataset's `schema` (and its availability); it has no check "
                    "item and runs wherever the contract is declared.",
                    "",
                ]
            if rule.recommendation:
                lines += [f"*When it fails:* {_escape(rule.recommendation)}", ""]
    return "\n".join(lines).rstrip() + "\n"


def _anchor(heading: str) -> str:
    """The GitHub anchor of a Markdown heading."""
    return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")


def sources_markdown(registry: SourceRegistry, title: str = "Source reference") -> str:
    """Markdown reference for every source type in ``registry``."""
    lines = [
        f"# {title}",
        "",
        "<!-- Generated by scripts/generate_docs.py; do not edit by hand. -->",
        "",
    ]
    for source in registry:
        doc = (source.__doc__ or "").strip().splitlines()[0] if source.__doc__ else ""
        lines += [
            f"## `{source.kind}`",
            "",
            doc,
            "",
            params_table(source.Params, skip=("type",)),
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def comparison_markdown(
    registries: Mapping[str, ComponentRegistry[Any]], title: str = "Comparison components"
) -> str:
    """Markdown reference for comparison levels, measures, comparators and steps."""
    lines = [
        f"# {title}",
        "",
        "<!-- Generated by scripts/generate_docs.py; do not edit by hand. -->",
        "",
    ]
    for heading, registry in registries.items():
        lines += [f"## {heading}", "", f"Entry-point group: `{registry.group}`.", ""]
        for component in registry:
            doc = (component.__doc__ or "").strip().splitlines()[0] if component.__doc__ else ""
            lines += [f"### `{component.name}`", "", _escape(doc), ""]
            lines += model_reference(component.Params, level=4)
    return "\n".join(lines).rstrip() + "\n"
