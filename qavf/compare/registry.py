"""Named comparison components and their discovery from code and entry points.

Alignment steps, measures, comparators and levels each live in one
:class:`ComponentRegistry`. Built-ins register with the same public decorator
plugins use; installed distributions add components through the
``qavf.compare.*`` entry-point groups. Discovery follows rule and source
discovery: duplicate entry-point names are rejected before loading, a failing
plugin is recorded (never silently dropped) and leaves no partial registration,
and ``plugins.required`` can insist on a component (``compare.levels:name``).
"""

from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from importlib.metadata import entry_points
from threading import RLock
from types import ModuleType
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from qavf.engine.registry import RuleParams, registration_transaction
from qavf.redaction import exception_details, redact_text

logger = logging.getLogger(__name__)


class Component(ABC):
    """Base of every registered comparison component.

    ``name`` selects the component in configuration and ``Params`` validates its
    options there, when the configuration is loaded.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""
    Params: ClassVar[type[RuleParams]] = RuleParams

    def __init__(self, params: RuleParams | dict[str, Any] | None = None) -> None:
        if isinstance(params, RuleParams):
            if not isinstance(params, self.Params):
                raise TypeError(
                    f"{self.name}: expected {self.Params.__name__}, got {type(params).__name__}"
                )
            self.params = params
        else:
            self.params = self.Params.model_validate(dict(params or {}))


class ComponentRegistry[C: Component]:
    """Component classes keyed by ``name``, discovered once per process.

    Lookups trigger discovery of the built-in modules and, unless disabled with
    :meth:`discover`, of the entry-point group. Use :meth:`refresh` after
    installing a distribution in a running process.
    """

    def __init__(self, kind: str, group: str, base: type, builtins: Iterable[str]) -> None:
        self.kind = kind
        self.group = group
        self.base = base
        self.builtins = tuple(builtins)
        self._components: dict[str, type[C]] = {}
        self.plugins: dict[str, str] = {}
        """Entry points tried: ``"loaded"`` or the error."""
        self._lock = RLock()
        self._builtins_loaded = False
        self._entry_points_loaded = False

    # ------------------------------------------------------------ registration
    def register[T: type](self, component: T) -> T:
        """Register a component class. Usable as a decorator.

        Raises:
            TypeError: if it is not a concrete subclass of the registry's base or has
                no ``name``.
            ValueError: if a different class already uses the same name.
        """
        if not (inspect.isclass(component) and issubclass(component, self.base)):
            raise TypeError(f"a {self.kind} must subclass {self.base.__name__}")
        if inspect.isabstract(component):
            raise TypeError(f"{component.__name__} is abstract and cannot be registered")
        name = getattr(component, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"{component.__name__} has no {self.kind} name")
        params = getattr(component, "Params", None)
        if not (isinstance(params, type) and issubclass(params, BaseModel)):
            raise TypeError(f"{component.__name__}.Params must be a pydantic model")
        with self._lock:
            existing = self._components.get(name)
            if existing is not None and not _same_class(existing, component):
                raise ValueError(
                    f"duplicate {self.kind} {name!r}: "
                    f"{existing.__module__}.{existing.__qualname__} and "
                    f"{component.__module__}.{component.__qualname__}"
                )
            self._components[name] = component  # type: ignore[assignment]
        return component

    def register_module(self, module: ModuleType) -> list[type[C]]:
        """Atomically register concrete components *defined in* ``module``."""
        found: list[type[C]] = []
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, self.base)
                and obj.__module__ == module.__name__
                and not inspect.isabstract(obj)
                and isinstance(getattr(obj, "name", None), str)
            ):
                found.append(obj)  # type: ignore[arg-type]
        with self._transaction():
            for component in found:
                self.register(component)
        return found

    # --------------------------------------------------------------- discovery
    def discover(self, *, use_entry_points: bool = True) -> ComponentRegistry[C]:
        """Load built-ins and (once) the entry-point group; return ``self``."""
        with self._lock:
            if not self._builtins_loaded:
                with self._transaction():
                    for module in self.builtins:
                        importlib.import_module(module)
                self._builtins_loaded = True
            if use_entry_points and not self._entry_points_loaded:
                self._entry_points_loaded = True
                self._discover_entry_points()
        return self

    def refresh(self) -> ComponentRegistry[C]:
        """Rescan the entry-point group (for example after installing a plugin)."""
        with self._lock:
            self._entry_points_loaded = False
        return self.discover()

    def _discover_entry_points(self) -> None:
        entries = list(entry_points(group=self.group))
        counts = Counter(entry.name for entry in entries)
        for entry in entries:
            if counts[entry.name] > 1:
                self.plugins[entry.name] = f"error: duplicate entry-point name in {self.group}"
                logger.error("Duplicate %s entry point %r", self.kind, redact_text(entry.name))
                continue
            try:
                with self._transaction():
                    loaded = entry.load()
                    if inspect.ismodule(loaded):
                        if not self.register_module(loaded):
                            raise TypeError(f"the module defines no {self.kind}")
                    else:
                        self.register(loaded)
            except Exception as exc:
                message = exception_details(exc)
                logger.error(
                    "Failed to load %s entry point %r: %s",
                    self.kind,
                    redact_text(entry.name),
                    message,
                )
                self.plugins[entry.name] = f"error: {message}"
                continue
            self.plugins[entry.name] = "loaded"

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Undo every registration a failed import made, including by decorators."""
        with self._lock:
            before = dict(self._components)
            try:
                with registration_transaction():
                    yield
            except BaseException:
                self._components = before
                raise

    # ------------------------------------------------------------------ lookup
    def get(self, name: str) -> type[C]:
        """The class registered as ``name``.

        Raises:
            KeyError: naming the available components.
        """
        self.discover()
        try:
            return self._components[name]
        except KeyError:
            available = ", ".join(sorted(self._components)) or "none"
            raise KeyError(f"unknown {self.kind} {name!r}; available: {available}") from None

    def create(self, name: str, params: RuleParams | dict[str, Any] | None = None) -> C:
        """Instantiate a component with validated parameters.

        Raises:
            ValueError: for an unknown name or invalid parameters, naming the
                component, so configuration validation reports either clearly.
        """
        try:
            component = self.get(name)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        try:
            return component(params)
        except ValidationError as exc:
            raise ValueError(f"invalid {self.kind} {name!r}: {exception_details(exc)}") from None

    def names(self) -> list[str]:
        """Registered names, sorted."""
        self.discover()
        return sorted(self._components)

    def __iter__(self) -> Iterator[type[C]]:
        self.discover()
        return iter([self._components[name] for name in sorted(self._components)])

    def __contains__(self, name: object) -> bool:
        self.discover()
        return name in self._components

    def __len__(self) -> int:
        self.discover()
        return len(self._components)


def _same_class(a: type, b: type) -> bool:
    return a is b or (a.__module__ == b.__module__ and a.__qualname__ == b.__qualname__)


def registries() -> dict[str, ComponentRegistry[Any]]:
    """Every comparison registry, keyed by its ``plugins.required`` qualifier."""
    from qavf.compare.align import ALIGNMENT_STEPS
    from qavf.compare.comparators import COMPARATORS
    from qavf.compare.levels import LEVELS
    from qavf.compare.measures import MEASURES

    return {
        "compare.alignment": ALIGNMENT_STEPS,
        "compare.comparators": COMPARATORS,
        "compare.levels": LEVELS,
        "compare.measures": MEASURES,
    }


def plugin_status(*, use_entry_points: bool = True) -> dict[str, str]:
    """Every comparison entry point tried, as ``compare.<kind>:name`` -> status."""
    status: dict[str, str] = {}
    for prefix, registry in registries().items():
        registry.discover(use_entry_points=use_entry_points)
        status |= {f"{prefix}:{name}": state for name, state in registry.plugins.items()}
    return status
