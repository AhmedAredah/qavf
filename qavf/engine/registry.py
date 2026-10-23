"""Rule contract and the plugin registry that discovers rules automatically.

Writing a rule
--------------
Subclass :class:`Rule`, set the class metadata, optionally declare a ``Params``
model, and implement :meth:`Rule.evaluate`::

    class NoNegativeVmt(Rule):
        rule_id = "CUSTOM-001"
        name = "No negative VMT"
        description = "VMT must never be negative."
        category = "custom"
        default_severity = Severity.ERROR
        requires = ("activity",)

        def evaluate(self, ctx: DataContext) -> Finding:
            bad = ctx.get("activity").filter(pl.col("vmt") < 0)
            if bad.is_empty():
                return Finding.ok("All VMT values are non-negative.")
            return Finding.fail(f"{bad.height} negative VMT rows.", affected=bad)

A rule may also return a list of findings, one per independent check, each
with its own ``check`` label and optionally its own ``severity``.

Any concrete ``Rule`` subclass defined in a scanned module is registered; no
decorator or manual list is required. Modules whose name starts with an
underscore are treated as private helpers and skipped.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import logging
import pkgutil
import sys
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, cast

import polars as pl
from pydantic import BaseModel, ConfigDict

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.severity import Severity
from qavf.plugins.transactions import (
    LOCK as _registration_lock,
)
from qavf.plugins.transactions import (
    installed_origin,
    installed_visible,
    mapping,
    transaction,
)
from qavf.redaction import exception_details, redact_text
from qavf.settings import QAVFSettings

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "qavf.rules"


def _import_plugin(module_name: str) -> ModuleType:
    """Import configured code without exposing credential-bearing tracebacks."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"rule plugin {redact_text(module_name)!r} could not load: {exception_details(exc)}"
        ) from None


class RuleParams(BaseModel):
    """Base class for rule parameter models.

    Unknown keys are rejected, and so are NaN and infinite floats: no threshold or
    tolerance is meaningful as either (an infinite tolerance would silently accept
    any change).
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, use_attribute_docstrings=True, allow_inf_nan=False
    )


@dataclass(frozen=True, slots=True)
class Finding:
    """What a rule reports back to the engine.

    The engine turns a finding into a :class:`~qavf.reports.models.RuleResult`
    by adding identity, severity and timing, so rules never deal with those.
    """

    passed: bool
    message: str
    affected: pl.DataFrame | Sequence[Mapping[str, Any]] | None = None
    """Offending records. The engine samples and serialises them."""
    affected_count: int | None = None
    """Defaults to ``len(affected)``; set it when ``affected`` is itself a sample."""
    total_records: int | None = None
    """Items examined, counted in the same ``unit`` as ``affected_count``. Leave it
    ``None`` when there is no meaningful denominator."""
    unit: str | None = None
    """What ``affected_count`` and ``total_records`` count (plural noun: ``rows``,
    ``keys``, ``problems``, ...)."""
    affected_rows: int | None = None
    """Data rows involved, when known and ``unit`` is not ``rows`` (for ``rows`` it
    defaults to ``affected_count``)."""
    metrics: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    """The rule does not apply (nothing configured for it, or optional data absent)."""
    assessed: bool = True
    """False when something the rule was configured to check could not be checked (missing data,
    invalid SQL, nothing testable). Reported as ERROR so it can never read as a pass."""
    datasets: tuple[str, ...] = ()
    check: str | None = None
    """Identifies one of several checks a rule reports separately (e.g. a column
    limit); shown next to the rule id in reports."""
    severity: Severity | None = None
    """Overrides the rule's severity for this finding (per-check severities)."""

    def __post_init__(self) -> None:
        """Reject ambiguous plugin outcomes before the engine interprets them.

        Dataclass annotations do not validate values at runtime. In particular,
        truthy strings such as ``"false"`` must never become a passing verdict,
        and booleans or rounded numeric counts must not enter the audit ledger.
        """
        for name in ("passed", "skipped", "assessed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Finding.{name} must be a boolean")
        for name in ("affected_count", "total_records", "affected_rows"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"Finding.{name} must be a nonnegative integer or None")
            if value is not None and value < 0:
                raise ValueError(f"Finding.{name} must be nonnegative")

    @classmethod
    def ok(cls, message: str, **kwargs: Any) -> Finding:
        """The rule ran and found nothing wrong."""
        return cls(passed=True, message=message, **kwargs)

    @classmethod
    def fail(cls, message: str, **kwargs: Any) -> Finding:
        """The rule ran and found a problem."""
        return cls(passed=False, message=message, **kwargs)

    @classmethod
    def skip(cls, message: str, **kwargs: Any) -> Finding:
        """The rule does not apply to this run (nothing was configured for it)."""
        return cls(passed=True, message=message, skipped=True, **kwargs)

    @classmethod
    def unassessed(cls, message: str, **kwargs: Any) -> Finding:
        """A configured check could not be carried out; the data was not assessed."""
        return cls(passed=False, message=message, assessed=False, **kwargs)


class Rule[P: RuleParams](ABC):
    """Base class for every QA rule.

    Class attributes describe the rule; instances carry validated parameters
    and an effective severity. ``evaluate`` must be side-effect free and must
    not mutate the frames it reads, because rules run concurrently.
    """

    rule_id: ClassVar[str]
    """Stable identifier, conventionally ``PREFIX-NNN``."""
    name: ClassVar[str]
    description: ClassVar[str]
    category: ClassVar[str]
    default_severity: ClassVar[Severity] = Severity.ERROR
    requires: ClassVar[tuple[str, ...]] = ()
    """Datasets that must be available; otherwise the rule is skipped."""
    recommendation: ClassVar[str] = ""
    """What to do when the rule fails. Shown in reports."""
    Params: ClassVar[type[RuleParams]] = RuleParams

    def __init__(
        self,
        params: P | Mapping[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> None:
        if isinstance(params, RuleParams):
            if not isinstance(params, self.Params):
                raise TypeError(
                    f"{self.rule_id}: expected {self.Params.__name__}, got {type(params).__name__}"
                )
            self.params: P = cast(P, params)
        else:
            self.params = cast(P, self.Params.model_validate(dict(params or {})))
        self.severity = severity or self.default_severity

    def configuration_errors(self, settings: QAVFSettings) -> list[str]:
        """Problems with this rule's parameters that need the whole configuration to
        see (e.g. a dataset selection naming a dataset that does not exist). Checked
        when the run is planned; the default finds none."""
        return []

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        """Complete dataset dependencies, or ``None`` when execution is opaque.

        Declaring dependencies permits the engine to release data after its last
        consumer finishes. Include every dataset read directly, through SQL or
        through helpers. ``requires`` alone is not a complete dependency list.
        Unspecified plugin dependencies conservatively retain all datasets.
        """
        return None

    @abstractmethod
    def evaluate(self, ctx: DataContext) -> Finding | Sequence[Finding]:
        """Assess the data in ``ctx``.

        Return one :class:`Finding`, or several when the rule evaluates
        independent checks that should be reported (and graded) separately.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.rule_id} severity={self.severity.value}>"


_REQUIRED_METADATA = ("rule_id", "name", "description", "category")


class RuleRegistry:
    """Holds rule classes keyed by ``rule_id`` and knows how to find them."""

    def __init__(self) -> None:
        self._rules: dict[str, type[Rule]] = {}
        self._origins: dict[str, bool] = {}
        self._visibility: dict[str, bool] = {}
        self._entry_points_enabled = False
        self.plugins: dict[str, str] = {}
        """Entry points tried by :meth:`discover_entry_points`: ``"loaded"`` or the error."""

    # ------------------------------------------------------------ registration
    def register[R: Rule](self, rule_cls: type[R]) -> type[R]:
        """Register a rule class. Usable as a decorator.

        Raises:
            TypeError: if the class is abstract or lacks required metadata.
            ValueError: if a *different* class already uses the same ``rule_id``.
        """
        with _registration_lock:
            rules = mapping(self, "_rules")
            if inspect.isabstract(rule_cls):
                raise TypeError(f"{rule_cls.__name__} is abstract and cannot be registered")
            missing = [attr for attr in _REQUIRED_METADATA if not getattr(rule_cls, attr, None)]
            if missing:
                raise TypeError(
                    f"{rule_cls.__name__} is missing rule metadata: {', '.join(missing)}"
                )
            existing = rules.get(rule_cls.rule_id)
            if existing is not None and not _same_class(existing, rule_cls):
                raise ValueError(
                    f"duplicate rule id {rule_cls.rule_id!r}: "
                    f"{existing.__module__}.{existing.__qualname__} and "
                    f"{rule_cls.__module__}.{rule_cls.__qualname__}"
                )
            rules[rule_cls.rule_id] = rule_cls
            origins = mapping(self, "_origins")
            origins[rule_cls.rule_id] = origins.get(rule_cls.rule_id, True) and installed_origin()
            return rule_cls

    @property
    def _entry_points_enabled(self) -> bool:
        with _registration_lock:
            return bool(mapping(self, "_visibility").get("entry_points", False))

    @_entry_points_enabled.setter
    def _entry_points_enabled(self, value: bool) -> None:
        with _registration_lock:
            mapping(self, "_visibility")["entry_points"] = value

    def _plugin_status(self, name: str, status: str) -> None:
        with _registration_lock:
            mapping(self, "plugins")[name] = status

    def register_module(self, module: ModuleType) -> list[type[Rule]]:
        """Atomically register concrete rule classes *defined in* ``module``."""
        found = []
        with transaction():
            for obj in vars(module).values():
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Rule)
                    and obj.__module__ == module.__name__
                    and not inspect.isabstract(obj)
                    and getattr(obj, "rule_id", None)
                ):
                    found.append(self.register(obj))
        if found:
            logger.debug("Registered %d rule(s) from %s", len(found), module.__name__)
        return found

    # --------------------------------------------------------------- discovery
    def discover_package(self, package_name: str) -> list[type[Rule]]:
        """Import ``package_name`` and scan it and all its public submodules.

        A module or subpackage whose name starts with ``_`` is private: it is
        neither imported nor descended into, so nothing inside it is scanned.
        """
        with self._transaction():
            package = _import_plugin(package_name)
            found = self.register_module(package)
            search_path = getattr(package, "__path__", None)
            if search_path is not None:
                found.extend(self._discover_submodules(package_name, search_path))
            return found

    def _discover_submodules(
        self, package_name: str, search_path: Iterable[str]
    ) -> list[type[Rule]]:
        # pkgutil.walk_packages imports every subpackage to recurse into it, private
        # ones included; recursing by hand prunes private names before any import.
        found: list[type[Rule]] = []
        for info in pkgutil.iter_modules(search_path, prefix=f"{package_name}."):
            if info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            with self._transaction():
                module = _import_plugin(info.name)
                found.extend(self.register_module(module))
            subpath = getattr(module, "__path__", None)
            if info.ispkg and subpath is not None:
                found.extend(self._discover_submodules(info.name, subpath))
        return found

    def discover_path(self, path: str | Path) -> list[type[Rule]]:
        """Scan a stand-alone ``.py`` file, or every public one in a directory."""
        target = Path(path)
        if target.is_dir():
            files = sorted(p for p in target.glob("*.py") if not p.name.startswith("_"))
        elif target.is_file():
            files = [target]
        else:
            raise FileNotFoundError(f"rule plugin path not found: {redact_text(str(target))}")
        found = []
        for file in files:
            with self._transaction():
                locator = hashlib.sha256(str(file.resolve()).encode("utf-8")).hexdigest()[:16]
                module_name = f"qavf_plugin_{file.stem}_{locator}"
                module = sys.modules.get(module_name)
                if module is None:
                    spec = importlib.util.spec_from_file_location(module_name, file)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"cannot load rule plugin {file}")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                    except Exception as exc:
                        del sys.modules[module_name]
                        raise ImportError(
                            f"rule plugin {redact_text(str(file))!r} could not load: "
                            f"{exception_details(exc)}"
                        ) from None
                    except BaseException:
                        del sys.modules[module_name]
                        raise
                found.extend(self.register_module(module))
        return found

    def discover_entry_points(self, group: str = ENTRY_POINT_GROUP) -> list[type[Rule]]:
        """Load rules advertised by installed distributions.

        An entry point may resolve to a rule class or to a module to scan. A
        broken third-party plugin is logged and skipped rather than aborting
        the run.
        """
        self._entry_points_enabled = True
        found = []
        entries = list(entry_points(group=group))
        counts = Counter(entry.name for entry in entries)
        for entry in entries:
            if counts[entry.name] > 1:
                self._plugin_status(entry.name, "error: duplicate entry-point name in rules group")
                logger.error("Duplicate rule entry point %r", redact_text(entry.name))
                continue
            try:
                with transaction(installed=True):
                    loaded = entry.load()
                    if inspect.ismodule(loaded):
                        registered = self.register_module(loaded)
                    elif inspect.isclass(loaded) and issubclass(loaded, Rule):
                        registered = [self.register(loaded)]
                    else:
                        raise TypeError("it is neither a Rule nor a module")
                found.extend(registered)
            except Exception as exc:
                # Recorded (and reported), so a broken plugin cannot vanish unnoticed.
                message = exception_details(exc)
                logger.error(
                    "Failed to load rule entry point %r: %s", redact_text(entry.name), message
                )
                self._plugin_status(entry.name, f"error: {message}")
                continue
            self._plugin_status(entry.name, "loaded")
        return found

    def discover(
        self,
        packages: Iterable[str] = ("qavf.rules",),
        modules: Iterable[str] = (),
        paths: Iterable[str | Path] = (),
        use_entry_points: bool = True,
    ) -> RuleRegistry:
        """Run every discovery mechanism and return ``self`` for chaining."""
        self._entry_points_enabled = use_entry_points
        for package in packages:
            self.discover_package(package)
        for module_name in modules:
            with self._transaction():
                self.register_module(_import_plugin(module_name))
        for path in paths:
            self.discover_path(path)
        if use_entry_points:
            self.discover_entry_points()
        logger.info("Rule discovery complete: %d rule(s) registered", len(self))
        return self

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """Publish module declarations and decorator side effects together."""
        with transaction():
            yield

    def _visible(self) -> dict[str, type[Rule]]:
        with _registration_lock:
            rules = mapping(self, "_rules")
            if self._entry_points_enabled or installed_visible() or installed_origin():
                return dict(rules)
            origins = mapping(self, "_origins")
            return {name: rule for name, rule in rules.items() if not origins.get(name, False)}

    # ------------------------------------------------------------------ lookup
    def get(self, rule_id: str) -> type[Rule]:
        """Return the class registered under ``rule_id``.

        Raises:
            KeyError: if no such rule exists.
        """
        try:
            return self._visible()[rule_id]
        except KeyError:
            raise KeyError(f"unknown rule id {rule_id!r}") from None

    def create(
        self,
        rule_id: str,
        params: Mapping[str, Any] | None = None,
        severity: Severity | None = None,
    ) -> Rule:
        """Instantiate a rule with validated parameters."""
        return self.get(rule_id)(params=params, severity=severity)

    def categories(self) -> list[str]:
        """Distinct rule categories, sorted."""
        return sorted({rule.category for rule in self._visible().values()})

    def __iter__(self) -> Iterator[type[Rule]]:
        return iter(sorted(self._visible().values(), key=lambda rule: rule.rule_id))

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._visible()

    def __len__(self) -> int:
        return len(self._visible())


def _same_class(a: type, b: type) -> bool:
    """True when ``b`` is ``a`` or a re-import of the same definition."""
    return a is b or (a.__module__ == b.__module__ and a.__qualname__ == b.__qualname__)


#: Process-wide registry for ad-hoc ``@register`` use. The runner builds its
#: own registry from configuration and also includes everything registered here.
default_registry = RuleRegistry()
register = default_registry.register


@contextmanager
def registration_transaction() -> Iterator[None]:
    """Stage QAVF-owned global decorators until an import/discovery succeeds.

    A context-local staged registry keeps partial definitions invisible to other
    registries. Only publication is locked; plugin imports may start other threads.
    Nested discovery commits to its parent's stage, never directly to globals.
    Arbitrary side effects performed by plugin code are outside this boundary.
    """
    with transaction():
        yield
