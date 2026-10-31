"""Lossless scientific-Python scalars at configuration boundaries.

This module does not import NumPy: a NumPy scalar can only exist after NumPy is
loaded. Ordinary configuration and the public facade keep their lazy imports.
"""

from __future__ import annotations

import math
import sys
from typing import Any


def normalize_scalar(value: Any) -> Any:
    """Return the equivalent supported Python scalar, or reject precision loss.

    In particular, booleans are not numbers, unsigned integers are never routed
    through float, and extended precision or nanosecond timestamps are not
    silently narrowed by ``item()``.
    """
    numpy = sys.modules.get("numpy")
    if numpy is None or not isinstance(value, numpy.generic):
        return value
    if isinstance(value, numpy.bool_):
        return bool(value)
    # timedelta64 is also a NumPy integer subclass, but its unit is part of
    # the value and must never disappear into an unqualified Python integer.
    if isinstance(value, numpy.datetime64 | numpy.timedelta64):
        raise ValueError(
            "NumPy temporal scalars need an explicit supported temporal value; "
            "automatic conversion could discard nanoseconds"
        )
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, numpy.floating):
        converted = float(value)
        if numpy.isnan(value):
            return converted
        if numpy.isinf(value):
            return converted
        if not math.isfinite(converted) or value.as_integer_ratio() != converted.as_integer_ratio():
            raise ValueError("a NumPy floating scalar must be exactly representable as Float64")
        return converted
    if isinstance(value, numpy.str_):
        return str(value)
    if isinstance(value, numpy.bytes_):
        return bytes(value)
    raise ValueError(f"unsupported NumPy configuration scalar {type(value).__name__}")


def normalize_scalar_sequence(values: Any) -> Any:
    """Normalize literal lists without interpreting arbitrary user containers."""
    if isinstance(values, list | tuple):
        return [normalize_scalar(value) for value in values]
    return values
