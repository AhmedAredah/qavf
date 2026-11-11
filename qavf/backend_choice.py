"""Which backend a run uses, and why: an argument, the configuration, or ``auto``.

The backend is chosen in this order:

1. an explicit request: ``qavf run --backend``/``qavf explain --backend`` (source
   ``cli``) or ``validate(backend=...)``/``explain(backend=...)`` (``argument``),
   ``auto`` included;
2. the configuration's ``engine.backend`` (``config``), when it is not ``auto``;
3. ``auto`` (``auto``): Polars when installed, otherwise the DuckDB core.

A requested or configured ``polars`` without the optional extra fails when the
backend is chosen, before any data is read, with the optional-dependency error
naming the setting. :class:`BackendChoice` records the outcome; ``qavf explain``,
``qavf resolve --origins`` and the run report show it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from qavf.optional import BACKENDS, has_polars, require_polars

if TYPE_CHECKING:
    from qavf.origins import OriginIndex
    from qavf.settings import QAVFSettings

BackendSource = Literal["cli", "argument", "config", "auto"]
_CONFIG_PATH = ("engine", "backend")


@dataclass(frozen=True, slots=True)
class BackendRequest:
    """A backend asked for explicitly, and through which interface."""

    backend: str
    via: Literal["cli", "argument"] = "argument"

    @classmethod
    def of(cls, value: str | BackendRequest | None) -> BackendRequest | None:
        """``value`` as a request (``None``: nothing was requested).

        Raises:
            ValueError: for an unknown backend name.
        """
        request = value if value is None or isinstance(value, BackendRequest) else cls(value)
        if request is not None and request.backend not in BACKENDS:
            raise ValueError("backend must be 'auto', 'duckdb', or 'polars'")
        return request


@dataclass(frozen=True, slots=True)
class ConfiguredBackend:
    """The ``engine.backend`` a configuration writes, and where."""

    backend: str = "auto"
    origin: str | None = None
    """``file:line:column`` of ``engine.backend``, when read from a file."""

    @classmethod
    def of_settings(cls, settings: QAVFSettings) -> ConfiguredBackend:
        return cls(settings.engine.backend, _origin(settings.origins))

    @classmethod
    def of_config(cls, config: object) -> ConfiguredBackend:
        """The backend a configuration argument (settings, mapping or file) pins.

        A file is read (with its includes and variables) only to find the value;
        an invalid value reads as ``auto`` and is reported by full validation.
        """
        from qavf.settings import QAVFSettings

        if isinstance(config, QAVFSettings):
            return cls.of_settings(config)
        if isinstance(config, str | os.PathLike):
            from qavf.configuration import read_config_document

            document = read_config_document(Path(config))
            return cls._raw(document.raw, _origin(document.origins))
        if isinstance(config, Mapping):
            return cls._raw(dict(config), None)
        return cls()

    @classmethod
    def _raw(cls, raw: dict[str, Any], origin: str | None) -> ConfiguredBackend:
        from qavf.composition import VariableSubstitution

        try:
            engine = VariableSubstitution.apply(raw).get("engine")
        except ValueError:
            return cls()
        value = engine.get("backend") if isinstance(engine, Mapping) else None
        return cls(value, origin) if value in BACKENDS else cls()


@dataclass(frozen=True, slots=True)
class BackendChoice:
    """The backend a run uses, what was requested and where the request came from."""

    backend: str
    """``duckdb`` or ``polars``."""
    requested: str
    """``auto``, ``duckdb`` or ``polars``, as requested or configured."""
    source: BackendSource
    """``cli`` (``--backend``), ``argument`` (``backend=``), ``config``
    (``engine.backend``) or ``auto`` (nothing pinned it)."""
    origin: str | None = None
    """Where ``engine.backend`` was written, for a ``config`` source read from a file."""

    @classmethod
    def resolve(
        cls, request: str | BackendRequest | None, configured: ConfiguredBackend
    ) -> BackendChoice:
        """The backend a run uses for ``request`` and the configuration's ``engine.backend``.

        Raises:
            ValueError: for an unknown backend name.
            OptionalDependencyError: for ``polars`` without the optional extra.
        """
        explicit = BackendRequest.of(request)
        source: BackendSource
        if explicit is not None:
            requested, source, origin = explicit.backend, explicit.via, None
            capability = "--backend polars" if explicit.via == "cli" else "backend='polars'"
        elif configured.backend != "auto":
            requested, source, origin = configured.backend, "config", configured.origin
            where = f" ({configured.origin})" if configured.origin else ""
            capability = f"engine.backend: polars{where}"
        else:
            requested, source, origin = "auto", "auto", None
            capability = ""
        if requested == "polars":
            require_polars(capability)
            backend = "polars"
        elif requested == "duckdb":
            backend = "duckdb"
        else:
            backend = "polars" if has_polars() else "duckdb"
        return cls(backend, requested, source, origin)

    @classmethod
    def for_settings(
        cls, request: str | BackendRequest | None, settings: QAVFSettings
    ) -> BackendChoice:
        """:meth:`resolve` against validated settings; a choice they already record
        stands when nothing is requested."""
        if request is None and settings.backend_choice is not None:
            return settings.backend_choice
        return cls.resolve(request, ConfiguredBackend.of_settings(settings))

    @classmethod
    def for_config(cls, request: str | BackendRequest | None, config: object) -> BackendChoice:
        """:meth:`resolve` for a configuration argument of ``validate``/``explain``
        (settings, a mapping or a file), before it is fully validated."""
        from qavf.settings import QAVFSettings

        if isinstance(config, QAVFSettings):
            return cls.for_settings(request, config)
        return cls.resolve(request, ConfiguredBackend.of_config(config))

    @classmethod
    def ran(cls, settings: QAVFSettings, backend: str) -> BackendChoice:
        """How a runner of ``backend`` was chosen: as recorded on its settings, else
        from ``engine.backend`` (or ``auto``) when that names it."""
        recorded = settings.backend_choice
        if recorded is not None and recorded.backend == backend:
            return recorded
        configured = ConfiguredBackend.of_settings(settings)
        if configured.backend == backend:
            return cls(backend, backend, "config", configured.origin)
        source: BackendSource = "auto" if configured.backend == "auto" else "argument"
        return cls(backend, configured.backend, source)

    def describe(self) -> str:
        """``duckdb (engine.backend at project.yaml:4:3)``, ``polars (auto: ...)``."""
        detected = "auto: Polars is " + (
            "installed" if self.backend == "polars" else "not installed"
        )
        if self.source == "cli":
            how = f"--backend {self.requested}"
        elif self.source == "argument":
            how = f"backend={self.requested!r}"
        elif self.source == "config":
            how = f"engine.backend: {self.requested}"
            how += f" at {self.origin}" if self.origin else ""
        else:
            how = detected
        if self.requested == "auto" and self.source != "auto":
            how += f"; {detected}"
        return f"{self.backend} ({how})"

    def as_dict(self) -> dict[str, str | None]:
        return {
            "backend": self.backend,
            "requested": self.requested,
            "source": self.source,
            "origin": self.origin,
        }


def _origin(origins: OriginIndex | None) -> str | None:
    found = origins.find(_CONFIG_PATH) if origins is not None else None
    return found.describe() if found is not None else None


__all__ = ["BackendChoice", "BackendRequest", "BackendSource", "ConfiguredBackend"]
