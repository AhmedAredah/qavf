"""Human-readable resource policies for public validation calls."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(b|[kmgtp]i?b)?\s*", re.IGNORECASE)


def _bytes(value: int | str, name: str, *, positive: bool = False) -> int:
    """Parse a byte quantity; decimal KB and binary KiB have distinct meanings."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a byte count or size such as '2GiB'")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and (match := _SIZE.fullmatch(value)):
        unit = (match[2] or "b").lower()
        power = "kmgtp".find(unit[0]) + 1 if unit != "b" else 0
        amount = Decimal(match[1]) * (1024 if "i" in unit else 1000) ** power
        if amount != amount.to_integral_value():
            raise ValueError(f"{name} must resolve to a whole number of bytes")
        result = int(amount)
    else:
        raise ValueError(f"{name} must be a byte count or size such as '2GiB'")
    if result < (1 if positive else 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


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
    """

    target_memory: int | str | None = None
    cache_memory: int | str | None = None
    max_spill: int | str | None = None
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

    def __post_init__(self) -> None:
        self.engine_overrides()

    def engine_overrides(self) -> dict[str, Any]:
        """Validated overrides in the engine's configuration vocabulary."""
        result: dict[str, Any] = {}
        for name, target in (
            ("target_memory", "memory_budget_bytes"),
            ("cache_memory", "cache_max_bytes"),
            ("max_spill", "spill_max_bytes"),
            ("evidence_bytes", "max_evidence_bytes"),
            ("report_bytes", "max_report_bytes"),
            ("address_space", "process_address_space_bytes"),
            ("file_size", "process_file_bytes"),
        ):
            if (value := getattr(self, name)) is not None:
                result[target] = _bytes(
                    value, name, positive=name in {"target_memory", "report_bytes", "address_space"}
                )
        for name, target in (
            ("max_workers", "max_workers"),
            ("native_threads", "native_threads"),
            ("evidence_rows", "max_affected_records"),
            ("cpu_time", "process_cpu_time_s"),
        ):
            if (value := getattr(self, name)) is not None:
                minimum = 0 if name == "evidence_rows" else 1
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{name} must be an integer >= {minimum}")
                result[target] = value
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
        hard = any(
            value is not None
            for value in (self.wall_time, self.address_space, self.cpu_time, self.file_size)
        )
        if hard:
            if self.isolate is False:
                raise ValueError("execution limits require isolate=True")
            result["execution_mode"] = "isolated"
        if self.temp_dir is not None:
            if not isinstance(self.temp_dir, str | Path) or not str(self.temp_dir).strip():
                raise ValueError("temp_dir must be a nonempty path")
            result["temp_directory"] = Path(self.temp_dir)
        return result
