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
    from qavf.compare.registry import ComponentRegistry


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


def model_reference(model: type[BaseModel], level: int = 4) -> list[str]:
    """Tables for ``model`` and every parameter model nested inside it."""
    lines = [params_table(model), ""]
    seen: set[type[BaseModel]] = {model}
    for field in model.model_fields.values():
        for nested in _nested_models(field.annotation, seen):
            doc = (nested.__doc__ or "").strip().splitlines()[0] if nested.__doc__ else ""
            lines += [
                f"{'#' * level} `{nested.__name__}`",
                "",
                _escape(doc),
                "",
                params_table(nested),
                "",
            ]
    return lines


def rules_markdown(registry: RuleRegistry, title: str = "Rule reference") -> str:
    """Markdown reference for every rule in ``registry``, grouped by category."""
    lines = [
        f"# {title}",
        "",
        "<!-- Generated by scripts/generate_docs.py; do not edit by hand. -->",
        "",
        "| ID | Category | Default severity | Name |",
        "|---|---|---|---|",
    ]
    rules = list(registry)
    for r in rules:
        link = f"[{r.rule_id}](#{r.rule_id.lower()})"
        lines.append(f"| {link} | {r.category} | {r.default_severity.value} | {r.name} |")
    for category in registry.categories():
        lines += ["", f"## {category.title()}", ""]
        for rule in (r for r in rules if r.category == category):
            lines += [
                f"### {rule.rule_id}",
                "",
                f"**{rule.name}** · default severity `{rule.default_severity.value}`",
                "",
                _escape(rule.description),
                "",
            ]
            if rule.requires:
                lines += [f"Requires datasets: {', '.join(f'`{d}`' for d in rule.requires)}.", ""]
            lines += model_reference(rule.Params)
            if rule.recommendation:
                lines += [f"*When it fails:* {_escape(rule.recommendation)}", ""]
    return "\n".join(lines).rstrip() + "\n"


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
