"""Human-readable resource policies: one vocabulary for configurations and calls.

:class:`ResourceLimits` is the policy a Python call passes (``resources=``);
:class:`ResourceSpec` is the same policy written in a configuration's top-level
``resources:`` section or a standalone comparison spec. Both name the same
limits, accept the same sizes (``'64MiB'``, ``'2GB'`` or a byte count) and map
to the same engine settings (:attr:`ResourceLimits.ENGINE_FIELDS`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    GetJsonSchemaHandler,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    ValidationInfo,
    WithJsonSchema,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from qavf.config_paths import PathRole

_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(b|[kmgtp]i?b)?\s*", re.IGNORECASE)
SIZE_PATTERN = r"^\s*\d+(\.\d+)?\s*([bB]|[kKmMgGtTpP][iI]?[bB])?\s*$"
"""A size text, for editor schemas: ``2GiB`` (binary), ``2GB`` (decimal), ``512``."""
SIZE_TEXT_SCHEMA: dict[str, Any] = {
    "type": "string",
    "pattern": SIZE_PATTERN,
    "description": "A size such as 2GiB (binary) or 2GB (decimal).",
}


def parse_size(value: int | str, name: str, *, positive: bool = False) -> int:
    """A byte quantity; decimal KB and binary KiB have distinct meanings.

    Raises:
        ValueError: naming ``name``, for a boolean, a text that is not a size, a
            size that is not a whole number of bytes, or one below the minimum.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a byte count or size such as '2GiB'")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and (match := _SIZE.fullmatch(value)):
        unit = (match[2] or "b").lower()
        power = "kmgtp".find(unit[0]) + 1 if unit != "b" else 0
        # Decimal arithmetic obeys the caller's thread-local precision. Parse
        # exactly, then scale integer ratios so a resource cap cannot round up
        # (or change between construction and execution in another context).
        numerator, denominator = Decimal(match[1]).as_integer_ratio()
        result, remainder = divmod(
            numerator * (1024 if "i" in unit else 1000) ** power, denominator
        )
        if remainder:
            raise ValueError(f"{name} must resolve to a whole number of bytes")
    else:
        raise ValueError(f"{name} must be a byte count or size such as '2GiB'")
    if result < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


class SizeText:
    """Pydantic annotation of an integer byte setting that also takes a size text.

    ``engine.memory_budget_bytes: 2GiB`` is read as 2147483648 by the same parser
    as ``resources``; a boolean is still rejected by the strict integer field.
    """

    @staticmethod
    def parse(value: Any, info: ValidationInfo) -> Any:
        """A size text as its byte count; any other value unchanged."""
        if isinstance(value, str):
            return parse_size(value, info.field_name or "size")
        return value

    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"anyOf": [handler(core_schema), dict(SIZE_TEXT_SCHEMA)]}


SizeSetting = Annotated[
    StrictInt | StrictStr,
    WithJsonSchema({"anyOf": [{"type": "integer", "minimum": 0}, SIZE_TEXT_SCHEMA]}),
]
"""A size in a ``resources`` section: a byte count or a size text, kept as written."""


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Run-wide resource policy; memory is a planning target, not an OS limit.

    Byte quantities accept integers or sizes such as ``'256MiB'`` and ``'2GB'``.
    ``None`` leaves the corresponding configuration value unchanged. Limits are
    never satisfied by silently sampling input or omitting configured checks.
    ``wall_time`` (seconds), ``address_space``, ``cpu_time`` (whole CPU seconds)
    and ``file_size`` select a supervised worker on Linux, macOS or Windows. Their
    OS scopes are per process/per file; address space is not RSS. macOS cannot
    enforce ``address_space`` and Windows cannot enforce ``file_size``; both are
    rejected there. ``isolate=True`` selects the same
    worker without adding a deadline or allocation ceiling.

    A configuration's ``resources:`` section (:class:`ResourceSpec`) takes the same
    names. A limit given both here and in the configuration must have the same
    value in both; a different value is an error naming both places.
    """

    target_memory: int | str | None = None
    cache_memory: int | str | None = None
    max_spill: int | str | None = None
    max_spill_files: int | None = None
    temp_dir: str | Path | None = None
    max_workers: int | None = None
    native_threads: int | None = None
    evidence_rows: int | None = None
    evidence_bytes: int | str | None = None
    report_bytes: int | str | None = None
    isolate: bool | None = None
    wall_time: float | None = None
    address_space: int | str | None = None
    cpu_time: int | None = None
    file_size: int | str | None = None

    ENGINE_FIELDS: ClassVar[Mapping[str, str]] = {
        "target_memory": "memory_budget_bytes",
        "cache_memory": "cache_max_bytes",
        "max_spill": "spill_max_bytes",
        "max_spill_files": "max_spill_files",
        "temp_dir": "temp_directory",
        "max_workers": "max_workers",
        "native_threads": "native_threads",
        "evidence_rows": "max_affected_records",
        "evidence_bytes": "max_evidence_bytes",
        "report_bytes": "max_report_bytes",
        "isolate": "execution_mode",
        "wall_time": "run_timeout_s",
        "address_space": "process_address_space_bytes",
        "cpu_time": "process_cpu_time_s",
        "file_size": "process_file_bytes",
    }
    """Each limit and the engine setting it sets (``isolate`` and the process limits
    also set ``execution_mode``)."""
    _SIZES: ClassVar[Mapping[str, bool]] = {
        "target_memory": True,
        "cache_memory": False,
        "max_spill": False,
        "evidence_bytes": False,
        "report_bytes": True,
        "address_space": True,
        "file_size": False,
    }
    """The byte limits, and whether each must be positive."""
    _COUNTS: ClassVar[Mapping[str, int]] = {
        "max_workers": 1,
        "max_spill_files": 1,
        "native_threads": 1,
        "evidence_rows": 0,
        "cpu_time": 1,
    }
    """The whole-number limits and their minimum."""
    _PROCESS: ClassVar[tuple[str, ...]] = ("wall_time", "address_space", "cpu_time", "file_size")

    def __post_init__(self) -> None:
        self.engine_overrides()

    @classmethod
    def from_mapping(
        cls, raw: Any, *, field: str = "resources", base_dir: Path | None = None
    ) -> ResourceLimits:
        """Limits from a configuration mapping such as ``{target_memory: 64MiB}``.

        The mapping is read as a :class:`ResourceSpec`, the model of a
        configuration's ``resources:`` section. ``field`` names the setting in
        error messages. A relative ``temp_dir`` resolves against ``base_dir``
        (the configuration file's folder) when given.

        Raises:
            ValueError: naming the field path, for a non-mapping, an unknown
                field or an invalid value.
        """
        return ResourceSpec.read(raw, field).limits(base_dir)

    def given(self) -> dict[str, Any]:
        """The limits this policy sets, by name, as given."""
        return {
            name: getattr(self, name)
            for name in self.ENGINE_FIELDS
            if getattr(self, name) is not None
        }

    def engine_sources(self) -> dict[str, str]:
        """Each engine setting :meth:`engine_overrides` sets and the limit that sets it."""
        sources = {
            target: name
            for name, target in self.ENGINE_FIELDS.items()
            if getattr(self, name) is not None
        }
        if "execution_mode" not in sources:
            process = [name for name in self._PROCESS if getattr(self, name) is not None]
            if process:
                sources["execution_mode"] = process[0]
        return sources

    def engine_overrides(self) -> dict[str, Any]:
        """Validated overrides in the engine's configuration vocabulary."""
        result: dict[str, Any] = {}
        for name, positive in self._SIZES.items():
            if (value := getattr(self, name)) is not None:
                result[self.ENGINE_FIELDS[name]] = parse_size(value, name, positive=positive)
        for name, minimum in self._COUNTS.items():
            if (value := getattr(self, name)) is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}")
                result[self.ENGINE_FIELDS[name]] = value
        if self.wall_time is not None:
            if (
                isinstance(self.wall_time, bool)
                or not isinstance(self.wall_time, int | float)
                or not math.isfinite(self.wall_time)
                or self.wall_time <= 0
            ):
                raise ValueError("wall_time must be a finite positive number of seconds")
            result["run_timeout_s"] = self.wall_time
        if self.isolate is not None:
            if not isinstance(self.isolate, bool):
                raise ValueError("isolate must be a boolean")
            result["execution_mode"] = "isolated" if self.isolate else "in_process"
        if any(getattr(self, name) is not None for name in self._PROCESS):
            if self.isolate is False:
                raise ValueError("execution limits require isolate=True")
            result["execution_mode"] = "isolated"
        if self.temp_dir is not None:
            if not isinstance(self.temp_dir, str | Path) or not str(self.temp_dir).strip():
                raise ValueError("temp_dir must be a nonempty path")
            result["temp_directory"] = Path(self.temp_dir)
        return result


class ResourceSpec(BaseModel):
    """The ``resources:`` section: run-wide limits in readable units.

    The same limits as :class:`ResourceLimits` (``validate(resources=...)``), mapped
    to the same engine settings. A limit written here and in ``engine`` (or passed
    as ``resources=``) must have the same value in each place; a different value is
    an error naming both. Sizes take ``2GiB`` (binary), ``2GB`` (decimal) or a byte
    count; ``wall_time`` and ``cpu_time`` are seconds.
    """

    model_config = ConfigDict(
        extra="forbid", use_attribute_docstrings=True, hide_input_in_errors=True
    )

    target_memory: SizeSetting | None = None
    """Run memory target for tracked materializations and caches (not an OS limit);
    sets ``engine.memory_budget_bytes``."""
    cache_memory: SizeSetting | None = None
    """Maximum retained cache; zero disables optional caching (``engine.cache_max_bytes``)."""
    max_spill: SizeSetting | None = None
    """Run temporary-storage budget; zero disables spilling (``engine.spill_max_bytes``)."""
    max_spill_files: StrictInt | None = None
    """Maximum live owned spill files (``engine.max_spill_files``)."""
    temp_dir: Annotated[StrictStr, PathRole.DIRECTORY] | None = None
    """Parent of the run-owned temporary directory, relative to the configuration
    file that writes it (``engine.temp_directory``)."""
    max_workers: StrictInt | None = None
    """Concurrent rule threads (``engine.max_workers``)."""
    native_threads: StrictInt | None = None
    """Native execution threads where the backend supports it (``engine.native_threads``)."""
    evidence_rows: StrictInt | None = None
    """Offending records kept per result (``engine.max_affected_records``)."""
    evidence_bytes: SizeSetting | None = None
    """Serialized evidence per result (``engine.max_evidence_bytes``)."""
    report_bytes: SizeSetting | None = None
    """Maximum encoded report size (``engine.max_report_bytes``)."""
    isolate: StrictBool | None = None
    """Run in a supervised worker process (``engine.execution_mode: isolated``)."""
    wall_time: StrictInt | StrictFloat | None = None
    """Whole isolated worker wall time in seconds (``engine.run_timeout_s``)."""
    address_space: SizeSetting | None = None
    """OS address-space ceiling per isolated process (``engine.process_address_space_bytes``)."""
    cpu_time: StrictInt | None = None
    """OS CPU seconds per isolated process (``engine.process_cpu_time_s``)."""
    file_size: SizeSetting | None = None
    """OS maximum size of one file the isolated worker writes (``engine.process_file_bytes``)."""

    @model_validator(mode="after")
    def _admissible(self) -> ResourceSpec:
        self.limits()
        return self

    @classmethod
    def read(cls, raw: Any, field: str = "resources") -> ResourceSpec:
        """``raw`` as a section, with errors as ``ValueError`` naming ``field``.

        Raises:
            ValueError: for a non-mapping, an unknown limit or an invalid value.
        """
        if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
            raise ValueError(
                f"{field}: must be a mapping of resource limits such as "
                "{target_memory: 2GiB, max_spill: 4GiB}"
            )
        known = set(cls.model_fields)
        if unknown := sorted(set(raw) - known):
            raise ValueError(
                f"{field}.{unknown[0]}: unknown resource limit; expected one of "
                f"{', '.join(sorted(known))}"
            )
        try:
            return cls.model_validate(dict(raw))
        except ValidationError as exc:
            error = exc.errors(include_url=False)[0]
            where = ".".join(str(part) for part in error["loc"][:1])
            message = str(error["msg"]).removeprefix("Value error, ")
            raise ValueError(f"{field}{'.' + where if where else ''}: {message}") from None

    def limits(self, base_dir: Path | None = None) -> ResourceLimits:
        """The section as :class:`ResourceLimits`; a relative ``temp_dir`` resolves
        against ``base_dir`` when given."""
        values = {name: getattr(self, name) for name in self.model_fields_set}
        values = {name: value for name, value in values.items() if value is not None}
        temp_dir = (self.temp_dir or "").strip() and self.temp_dir
        if base_dir is not None and temp_dir and not Path(temp_dir).is_absolute():
            values["temp_dir"] = base_dir / temp_dir
        return ResourceLimits(**values)


__all__ = [
    "SIZE_PATTERN",
    "ResourceLimits",
    "ResourceSpec",
    "SizeSetting",
    "SizeText",
    "parse_size",
]
