"""Lossless admission of optional temporal scalar implementations.

The audit format has native NumPy temporal tags and Python datetime tags.
Choose one before subclass checks can silently discard nanoseconds. This
module never imports pandas, so core validation keeps its optional dependency.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True, slots=True)
class ZonedNanosecondTimestamp:
    """Exact immutable zoned timestamp without a pandas runtime dependency."""

    utc_nanoseconds: int
    tzinfo: tzinfo

    def __post_init__(self):
        if type(self.utc_nanoseconds) is not int:
            raise ValueError("nanosecond timestamp ticks must be an integer")
        if not isinstance(self.tzinfo, timezone | ZoneInfo):
            raise ValueError("nanosecond timestamp requires a fixed offset or ZoneInfo")
        # Validate the representable date range at admission, not at rendering.
        try:
            self.as_datetime()
        except OverflowError as exc:
            raise ValueError("nanosecond timestamp is outside the audit date range") from exc

    def as_datetime(self) -> datetime:
        """The microsecond floor, for zone/fold operations, never the stored value."""
        return (
            datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=self.utc_nanoseconds // 1000)
        ).astimezone(self.tzinfo)

    @property
    def fold(self) -> int:
        return self.as_datetime().fold

    def isoformat(self) -> str:
        local = self.as_datetime()
        seconds = local.replace(microsecond=0).isoformat()
        fraction = f"{local.microsecond:06d}{self.utc_nanoseconds % 1000:03d}"
        return f"{seconds[:19]}.{fraction}{seconds[19:]}"

    def __str__(self) -> str:
        return self.isoformat()


def normalize_temporal_scalar(value: Any) -> Any:
    """Map pandas scalars to existing lossless tags, or reject unsupported values.

    Naive timestamps and durations retain their native NumPy units. Zoned
    timestamps use Python datetime when exact, and an immutable zoned tick
    value otherwise, retaining the named zone and DST fold in both cases.
    """
    pandas = sys.modules.get("pandas")
    if pandas is None:
        return value
    # Another thread can be importing the optional adapter. Its module enters
    # sys.modules before these public classes have been assigned.
    if value is getattr(pandas, "NaT", None):
        raise ValueError("audit pandas NaT is ambiguous; use None or a typed NumPy NaT")
    duration_type = getattr(pandas, "Timedelta", None)
    timestamp_type = getattr(pandas, "Timestamp", None)
    scalar: Any = value
    if isinstance(duration_type, type) and isinstance(value, duration_type):
        return scalar.to_timedelta64()
    if isinstance(timestamp_type, type) and isinstance(value, timestamp_type):
        if scalar.tzinfo is None:
            return scalar.to_datetime64()
        if scalar.nanosecond:
            zone = scalar.tzinfo
            name = getattr(zone, "key", getattr(zone, "zone", None))
            if name:
                zone = ZoneInfo(name)
            elif not isinstance(zone, timezone | ZoneInfo):
                zone = timezone(scalar.utcoffset(), scalar.tzname())
            return ZonedNanosecondTimestamp(int(scalar.value), zone)
        return scalar.to_pydatetime()
    return value
