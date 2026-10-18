"""Lossless admission of optional temporal scalar implementations.

The audit format has native NumPy temporal tags and Python datetime tags.
Choose one before subclass checks can silently discard nanoseconds. This
module never imports pandas, so core validation keeps its optional dependency.
"""

from __future__ import annotations

import sys
from typing import Any


def normalize_temporal_scalar(value: Any) -> Any:
    """Map pandas scalars to existing lossless tags, or reject unsupported values.

    Naive timestamps and durations retain their native NumPy units. Zoned
    timestamps use Python datetime only when microseconds represent them
    exactly, retaining the zone and DST fold. The existing format cannot
    represent a named zone together with a nanosecond remainder.
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
            raise ValueError(
                "audit pandas Timestamp with both a timezone and nanosecond precision "
                "is unsupported"
            )
        return scalar.to_pydatetime()
    return value
