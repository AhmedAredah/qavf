"""Named components and their discovery from code and entry points.

Comparison components (``qavf.compare``) and statistical tests
(``qavf.inference``) each live in one :class:`ComponentRegistry`. Built-ins
register with the same public decorator plugins use; installed distributions
add components through the registry's entry-point group. Discovery follows rule
and source discovery: duplicate entry-point names are rejected before loading,
a failing plugin is recorded (never silently dropped) and leaves no partial
registration, and ``plugins.required`` can insist on a component
(``compare.levels:name``, ``inference.tests:name``).

Lookups never import installed plugin code. Entry points load only through
:func:`entry_point_scope` (driven by ``plugins.entry_points``, or the
``entry_points`` argument of :func:`~qavf.compare.compare_datasets`), and the
components they provide are visible only inside such a scope, so a run with
``plugins.entry_points: false`` neither imports nor resolves them.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.metadata import EntryPoint, entry_points
from threading import RLock
from types import ModuleType
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from qavf.engine.registry import RuleParams, registration_transaction
from qavf.redaction import exception_details, redact_text

logger = logging.getLogger(__name__)

_INSTALLED: ContextVar[bool] = ContextVar("qavf_component_entry_points", default=False)

FAMILIES = ("qavf.compare.registry", "qavf.inference.registry")
"""Modules whose ``registries()`` provide the built-in component registries."""


class Component(ABC):
    """Base of every registered component.

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
    """Component classes keyed by ``name``.

    Built-in and in-code registrations are always visible; lookups import only
    the built-in modules. Components from the entry-point group load once, on
    explicit :meth:`discover` (``use_entry_points=True``), and are visible only
    inside :func:`entry_point_scope` with plugins enabled. Use :meth:`refresh`
    after installing a distribution in a running process.
    """

    def __init__(self, kind: str, group: str, base: type, builtins: Iterable[str]) -> None:
        self.kind = kind
        self.group = group
        self.base = base
        self.builtins = tuple(builtins)
        self._components: dict[str, type[C]] = {}
        self._installed: dict[str, type[C]] = {}
        """Components provided by entry points, kept apart from code registrations."""
        self.plugins: dict[str, str] = {}
        """Entry points tried: ``"loaded"`` or the error."""
        self._lock = RLock()
        self._builtins_loaded = False
        self._entry_points_loaded = False
        self._loading_entry_point = False

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
            existing = self._components.get(name) or self._installed.get(name)
            if existing is not None and not _same_class(existing, component):
                raise ValueError(
                    f"duplicate {self.kind} {name!r}: "
                    f"{existing.__module__}.{existing.__qualname__} and "
                    f"{component.__module__}.{component.__qualname__}"
                )
            target = self._installed if self._loading_entry_point else self._components
            target[name] = component  # type: ignore[assignment]
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
    def discover(self, *, use_entry_points: bool = False) -> ComponentRegistry[C]:
        """Load the built-ins and, when asked, (once) the entry-point group."""
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
        return self.discover(use_entry_points=True)

    def entries(self) -> list[EntryPoint]:
        """The installed entry points of this registry's group."""
        return list(entry_points(group=self.group))

    def _discover_entry_points(self) -> None:
        entries = self.entries()
        counts = Counter(entry.name for entry in entries)
        for entry in entries:
            if counts[entry.name] > 1:
                self.plugins[entry.name] = f"error: duplicate entry-point name in {self.group}"
                logger.error("Duplicate %s entry point %r", self.kind, redact_text(entry.name))
                continue
            try:
                with self._transaction():
                    self._loading_entry_point = True
                    try:
                        loaded = entry.load()
                        if inspect.ismodule(loaded):
                            if not self.register_module(loaded):
                                raise TypeError(f"the module defines no {self.kind}")
                        else:
                            self.register(loaded)
                    finally:
                        self._loading_entry_point = False
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
            before, installed = dict(self._components), dict(self._installed)
            try:
                with registration_transaction():
                    yield
            except BaseException:
                self._components, self._installed = before, installed
                raise

    def _visible(self) -> dict[str, type[C]]:
        self.discover()
        if _INSTALLED.get():
            return {**self._installed, **self._components}
        return self._components

    # ------------------------------------------------------------------ lookup
    def get(self, name: str) -> type[C]:
        """The class registered as ``name``.

        Raises:
            KeyError: naming the available components.
        """
        visible = self._visible()
        try:
            return visible[name]
        except KeyError:
            available = ", ".join(sorted(visible)) or "none"
            hint = (
                "; installed plugins are disabled (plugins.entry_points: false)"
                if name in self._installed and not _INSTALLED.get()
                else ""
            )
            raise KeyError(f"unknown {self.kind} {name!r}; available: {available}{hint}") from None

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
        """Visible names, sorted."""
        return sorted(self._visible())

    def __iter__(self) -> Iterator[type[C]]:
        visible = self._visible()
        return iter([visible[name] for name in sorted(visible)])

    def __contains__(self, name: object) -> bool:
        return name in self._visible()

    def __len__(self) -> int:
        return len(self._visible())


def _same_class(a: type, b: type) -> bool:
    return a is b or (a.__module__ == b.__module__ and a.__qualname__ == b.__qualname__)


def registries() -> dict[str, ComponentRegistry[Any]]:
    """Every component registry, keyed by its ``plugins.required`` qualifier."""
    found: dict[str, ComponentRegistry[Any]] = {}
    for family in FAMILIES:
        found |= importlib.import_module(family).registries()
    return found


def load_entry_points() -> None:
    """Load every component entry-point group (once per process)."""
    for registry in registries().values():
        registry.discover(use_entry_points=True)


@contextmanager
def entry_point_scope(enabled: bool) -> Iterator[None]:
    """Resolve components from installed plugins inside this context, or never.

    With ``enabled``, the entry-point groups load (once) and their components
    become visible to lookups in this context. Without it, no plugin code is
    imported and only built-in and in-code components resolve. Contexts do not
    cross threads: code running a comparison on another thread re-enters a scope.
    """
    if enabled:
        load_entry_points()
    token = _INSTALLED.set(enabled)
    try:
        yield
    finally:
        _INSTALLED.reset(token)


def entry_points_enabled() -> bool:
    """Whether the current context resolves components from installed plugins."""
    return _INSTALLED.get()


def plugin_status(*, enabled: bool = True) -> dict[str, str]:
    """Component entry points tried, as ``<family>.<kind>:name`` -> status.

    Empty when plugins are disabled for the run: nothing was loaded for it.
    """
    if not enabled:
        return {}
    status: dict[str, str] = {}
    for prefix, registry in registries().items():
        status |= {f"{prefix}:{name}": state for name, state in registry.plugins.items()}
    return status
