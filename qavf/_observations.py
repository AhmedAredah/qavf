"""Lossless admission of caller observations before Float64 statistical kernels."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import numpy.typing as npt


def _scalar(value: object) -> float:
    """Validate the original scalar, before mixed-container dtype inference."""
    if value is None:
        return math.nan
    if isinstance(value, bool | np.bool_):
        raise ValueError("observations must be numeric, not booleans")
    if isinstance(value, float):
        return float(value)  # Already stored at Float64 precision, including NaN/inf.
    if not isinstance(value, int | float | np.integer | np.floating | Decimal):
        raise ValueError("observations must be numeric, not coercible text or objects")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("observations cannot be represented exactly as Float64") from exc
    if isinstance(value, int | np.integer):
        exact = math.isfinite(converted) and int(converted) == value
    elif isinstance(value, Decimal):
        exact = not value.is_finite() or (
            math.isfinite(converted) and Decimal.from_float(converted) == value
        )
    else:
        # Preserve the original NumPy precision: converting the source to Python
        # float before comparing would erase the very difference being tested.
        exact = not np.isfinite(value) or (
            math.isfinite(converted) and type(value)(converted) == value
        )
    if not exact:
        raise ValueError("observations cannot be represented exactly as Float64")
    return converted


def float_observations(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a 1-D Float64 view/copy only after original values prove lossless.

    Homogeneous NumPy arrays retain vectorized admission. Other array-like inputs
    first retain their original scalar objects, so a mixed float/integer list
    cannot round integers or disguise booleans during dtype inference.
    """
    raw = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"values must be 1-D, got an array with {raw.ndim} dimension(s)")
    if raw.dtype.kind not in "fiuO":
        raise ValueError("observations must be real numeric values")
    if raw.dtype.kind == "O":
        with np.errstate(over="ignore", invalid="ignore"):
            return np.fromiter((_scalar(item) for item in raw), dtype=np.float64, count=len(raw))
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(raw, dtype=np.float64)
        if raw.dtype.kind in "iu":
            # A value that rounds to the integer type's limit is not exact, and the
            # out-of-range cast back is architecture dependent (x86 wraps, ARM
            # saturates to the original value), so reject it before casting.
            limit = 2.0 ** (raw.dtype.itemsize * 8 - (raw.dtype.kind == "i"))
            if np.any(converted >= limit) or not np.array_equal(
                raw, converted.astype(raw.dtype)
            ):
                raise ValueError("integer observations cannot be represented exactly as Float64")
        elif raw.dtype.itemsize > 8 and np.any(np.isfinite(raw) & (raw != converted)):
            raise ValueError("observations cannot be represented exactly as Float64")
    return converted
