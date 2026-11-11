"""One resource vocabulary: the ``resources:`` section, ``engine.*`` and ``resources=``.

A run's limits can be given in three places, all naming the same engine settings:

* the configuration's ``resources:`` section (:class:`~qavf.inputs.resources.ResourceSpec`,
  readable sizes such as ``target_memory: 2GiB``);
* the matching ``engine.*`` fields (``engine.memory_budget_bytes``), which also take
  readable sizes;
* the Python ``resources=`` argument (:class:`~qavf.ResourceLimits`).

:class:`ResourceFold` folds the section and the argument into the engine settings
while a configuration is validated. A limit given in more than one place must have
the same value in each; a different value is an error naming every place (with its
file and line when the configuration was read from a file). None of them silently
wins. :class:`ResourceProvenance` remembers where each limit came from, and
:class:`EffectiveLimits` lists the limits a run uses with their sources
(``qavf explain``, the run report).

Like :mod:`qavf.overrides`, the fold works on raw mappings before validation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from qavf.inputs.resources import ResourceLimits, ResourceSpec, parse_size
from qavf.origins import ConfigPath, OriginIndex

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings

RESOURCES_CONTEXT = "qavf_resources"
"""The validation-context key carrying a call's ``resources=`` (:class:`ResourceLimits`)."""
RESOURCES_KEY = "resources"
ENGINE_KEY = "engine"
LimitSource = Literal["resources", "engine", "argument", "default"]
_ATTRIBUTION: tuple[LimitSource, ...] = ("resources", "argument", "engine")
"""Which place a limit given identically in several is attributed to first."""
_BYTE_FIELDS = frozenset(
    ResourceLimits.ENGINE_FIELDS[name]
    for name in (
        "target_memory",
        "cache_memory",
        "max_spill",
        "evidence_bytes",
        "report_bytes",
        "address_space",
        "file_size",
    )
)


@dataclass(frozen=True, slots=True)
class LimitPlace:
    """One place a resource limit was given."""

    source: LimitSource
    written: str
    """The setting as written: ``resources.target_memory``, ``engine.memory_budget_bytes``
    or ``resources=ResourceLimits(target_memory=...)``."""
    path: ConfigPath | None = None
    """Its path in the configuration (``None`` for the argument)."""

    def label(self, origins: OriginIndex | None) -> str:
        """The setting, with its file and line when known."""
        origin = origins.find(self.path) if origins is not None and self.path else None
        return f"{self.written} ({origin.location()})" if origin is not None else self.written


class ResourceProvenance:
    """Where each engine resource setting of validated settings was given."""

    __slots__ = ("_places",)

    def __init__(self, places: Mapping[str, tuple[LimitPlace, ...]] | None = None) -> None:
        self._places = dict(places or {})

    def places(self, engine_field: str) -> tuple[LimitPlace, ...]:
        """The places ``engine_field`` was given, the one it is attributed to first."""
        return self._places.get(engine_field, ())


class ResourceFold:
    """Folds ``resources:`` and a call's ``resources=`` into raw ``engine`` settings."""

    def __init__(self, origins: OriginIndex | None = None) -> None:
        self.origins = origins

    def apply(
        self, raw: dict[str, Any], argument: ResourceLimits | None = None
    ) -> tuple[dict[str, Any], ResourceProvenance]:
        """``raw`` with its engine settings completed, and where each limit came from.

        A section or engine mapping that is not valid is left for field validation
        to report at its own path.

        Raises:
            ValueError: for a limit given different values in different places,
                naming each place.
        """
        engine = _plain(raw.get(ENGINE_KEY, {}))
        section = self._section(raw.get(RESOURCES_KEY))
        if not isinstance(engine, Mapping) or (section is None and argument is None):
            return raw, ResourceProvenance(_attributed(_engine_places(engine)))
        layers: list[tuple[LimitSource, Mapping[str, Any], Mapping[str, str]]] = []
        if section is not None:
            limits = section.limits()
            layers.append(("resources", limits.engine_overrides(), limits.engine_sources()))
        if argument is not None:
            layers.append(("argument", argument.engine_overrides(), argument.engine_sources()))
        self._refuse_conflicts(engine, layers)
        merged = dict(engine)
        for _, values, _ in layers:
            merged.update(values)
        places = _engine_places(engine)
        for source, _, names in layers:
            for engine_field, name in names.items():
                places.setdefault(engine_field, []).append(_place(source, name))
        if self.origins is not None and section is not None:
            for engine_field, name in section.limits().engine_sources().items():
                note = f"from {RESOURCES_KEY}.{name}"
                self.origins.alias((ENGINE_KEY, engine_field), (RESOURCES_KEY, name), note)
        return {**raw, ENGINE_KEY: merged}, ResourceProvenance(_attributed(places))

    @staticmethod
    def _section(value: Any) -> ResourceSpec | None:
        if value is None:
            return None
        if isinstance(value, ResourceSpec):
            return value
        try:
            return ResourceSpec.model_validate(value)
        except (ValidationError, ValueError, TypeError):
            return None

    def _refuse_conflicts(
        self,
        engine: Mapping[str, Any],
        layers: list[tuple[LimitSource, Mapping[str, Any], Mapping[str, str]]],
    ) -> None:
        for engine_field in dict.fromkeys(field for _, values, _ in layers for field in values):
            given: list[tuple[LimitPlace, Any]] = []
            if engine_field in engine:
                value = _normalized(engine_field, engine[engine_field])
                if value is _INVALID:
                    continue
                given.append((_engine_place(engine_field), value))
            for source, values, names in layers:
                if engine_field in values:
                    name = names.get(engine_field, engine_field)
                    given.append((_place(source, name), values[engine_field]))
            if len({_comparable(value) for _, value in given}) > 1:
                raise ValueError(self._conflict(engine_field, given))

    def _conflict(self, engine_field: str, given: list[tuple[LimitPlace, Any]]) -> str:
        shown = "; ".join(
            f"{place.label(self.origins)} gives {_display(value)}" for place, value in given
        )
        return (
            f"resource limit engine.{engine_field} is given different values: {shown}. "
            "Give each limit in one place (resources, engine or resources=), or the same "
            "value in each"
        )


def validation_context(
    origins: OriginIndex | None, resources: ResourceLimits | None
) -> dict[str, Any] | None:
    """The ``QAVFSettings`` validation context carrying ``origins`` and a call's
    ``resources=`` (folded with the configuration's limits).

    Raises:
        TypeError: for ``resources`` that is not :class:`ResourceLimits`.
    """
    from qavf.origins import ORIGINS_CONTEXT

    context: dict[str, Any] = {}
    if origins is not None:
        context[ORIGINS_CONTEXT] = origins
    if resources is not None:
        if not isinstance(resources, ResourceLimits):
            raise TypeError("resources must be ResourceLimits")
        context[RESOURCES_CONTEXT] = resources
    return context or None


class EffectiveLimits:
    """The resource limits a run uses, in the ``resources`` vocabulary, with sources."""

    def __init__(self, settings: QAVFSettings) -> None:
        self.settings = settings

    def rows(self) -> Iterator[dict[str, Any]]:
        """One row per limit: ``name``, ``setting`` (engine field), ``value``,
        ``source`` (resources, engine, argument or default), ``written`` (each place
        it was given) and ``origin`` (file:line:column of the first, when known)."""
        engine = self.settings.engine
        provenance = self.settings.resource_provenance
        origins = self.settings.origins
        for name, engine_field in ResourceLimits.ENGINE_FIELDS.items():
            places = provenance.places(engine_field)
            origin = None
            for place in places:
                found = origins.find(place.path) if origins is not None and place.path else None
                if found is not None:
                    origin = found.describe()
                    break
            yield {
                "name": name,
                "setting": f"engine.{engine_field}",
                "value": _display_value(getattr(engine, engine_field)),
                "source": places[0].source if places else "default",
                "written": tuple(place.written for place in places),
                "origin": origin,
            }


class _Invalid:
    """A value that cannot be compared (left for field validation to report)."""


_INVALID = _Invalid()


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, exclude_unset=True)
    return value


def _engine_place(engine_field: str) -> LimitPlace:
    return LimitPlace("engine", f"{ENGINE_KEY}.{engine_field}", (ENGINE_KEY, engine_field))


def _engine_places(engine: Any) -> dict[str, list[LimitPlace]]:
    """The resource limits written in a raw ``engine`` mapping."""
    if not isinstance(engine, Mapping):
        return {}
    return {
        engine_field: [_engine_place(engine_field)]
        for engine_field in ResourceLimits.ENGINE_FIELDS.values()
        if engine_field in engine
    }


def _place(source: LimitSource, name: str) -> LimitPlace:
    """The place a layer gave the limit ``name``: the section or the argument."""
    if source == "resources":
        return LimitPlace("resources", f"{RESOURCES_KEY}.{name}", (RESOURCES_KEY, name))
    return LimitPlace("argument", f"resources=ResourceLimits({name}=...)")


def _attributed(places: Mapping[str, list[LimitPlace]]) -> dict[str, tuple[LimitPlace, ...]]:
    """Each field's places, the attributed one first; an engine value the section
    also sets is the section's (written once, reported once)."""
    result: dict[str, tuple[LimitPlace, ...]] = {}
    for engine_field, found in places.items():
        sources = {place.source for place in found}
        kept = [p for p in found if not (p.source == "engine" and "resources" in sources)]
        kept.sort(key=_rank)
        result[engine_field] = tuple(kept)
    return result


def _rank(place: LimitPlace) -> int:
    return _ATTRIBUTION.index(place.source)


def _normalized(engine_field: str, value: Any) -> Any:
    """A written engine value in the form the fold produces, or :data:`_INVALID`."""
    try:
        if engine_field in _BYTE_FIELDS and isinstance(value, str):
            return parse_size(value, engine_field)
        if engine_field == "temp_directory" and isinstance(value, str | Path):
            return Path(value)
    except ValueError:
        return _INVALID
    return value


def _comparable(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


def _display(value: Any) -> str:
    return repr(str(value)) if isinstance(value, Path) else repr(value)


def _display_value(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


__all__ = [
    "RESOURCES_CONTEXT",
    "EffectiveLimits",
    "LimitPlace",
    "ResourceFold",
    "ResourceProvenance",
    "validation_context",
]
