"""What every test shares: SciPy loading, sample admission and the not-assessed path.

A test function raises :class:`Unassessable` for anything that stops it from
running soundly (no SciPy, too few values, non-finite values, a failed
precondition, the work budget); :func:`assessed` turns that into a
not-assessed :class:`~qavf.inference.results.TestResult`, never a pass.
Distribution functions always come from SciPy; nothing here approximates one.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from qavf._observations import float_observations
from qavf.inference.results import TestResult, not_assessed
from qavf.stats import center_scale_observations, scale_observations

MISSING_SCIPY = "needs SciPy, the stats extra (pip install 'qavf[stats]')"


class Unassessable(Exception):
    """A test cannot run soundly; ``fields`` describe what was known (sizes, ...)."""

    def __init__(self, reason: str, **fields: Any) -> None:
        super().__init__(reason)
        self.fields = fields


def scipy_stats() -> ModuleType | None:
    """``scipy.stats``, or ``None`` without the stats extra (imported on each call)."""
    try:
        from scipy import stats
    except ImportError:
        return None
    return stats


def require_scipy() -> ModuleType:
    """``scipy.stats``; raises :class:`Unassessable` without the stats extra."""
    stats = scipy_stats()
    if stats is None:
        raise Unassessable(MISSING_SCIPY)
    return stats


def assessed[**P](method: str) -> Callable[[Callable[P, TestResult]], Callable[P, TestResult]]:
    """Report :class:`Unassessable` from the test as a not-assessed result of ``method``."""

    def decorate(test: Callable[P, TestResult]) -> Callable[P, TestResult]:
        @functools.wraps(test)
        def run(*args: P.args, **kwargs: P.kwargs) -> TestResult:
            try:
                return test(*args, **kwargs)
            except Unassessable as exc:
                return not_assessed(method, str(exc), **exc.fields)

        return run

    return decorate


def observations(values: npt.ArrayLike, role: str) -> np.ndarray:
    """1-D Float64 observations, exactly as stored; missing or infinite values refuse.

    Raises:
        Unassessable: for values that are not lossless finite numbers.
    """
    try:
        x = float_observations(values)
    except ValueError as exc:
        raise Unassessable(f"{role}: {exc}") from None
    if bad := int((~np.isfinite(x)).sum()):
        raise Unassessable(f"{role}: {bad} missing or non-finite value(s)")
    return x


def at_least(size: int, minimum: int, what: str, **fields: Any) -> None:
    """Refuse a sample smaller than the method's minimum."""
    if size < minimum:
        raise Unassessable(f"{what}: {size} value(s), fewer than the {minimum} needed", **fields)


def common_scale(*samples: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Samples shifted by one shared origin and scaled by one shared power of two.

    Location- and scale-invariant statistics (t, ranks, KS, spread) are then
    computed on values near 1, so sums of squares neither overflow nor lose a
    small variation around a large offset. Differences convert back to data
    units by multiplying with the returned unit; the origin cancels.

    Raises:
        Unassessable: when the dynamic range cannot be rescaled without loss.
    """
    sizes = [len(sample) for sample in samples]
    try:
        joined, unit, _ = center_scale_observations(np.concatenate(samples))
    except ValueError as exc:
        raise Unassessable(str(exc)) from None
    return list(np.split(joined, np.cumsum(sizes)[:-1])), unit


def common_unit(*samples: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Samples divided by one shared power of two (no shift), for scale-equivariant
    statistics whose value depends on location (sums, means against zero)."""
    lengths = [len(sample) for sample in samples]
    try:
        joined, unit = scale_observations(np.concatenate(samples))
    except ValueError as exc:
        raise Unassessable(str(exc)) from None
    return list(np.split(joined, np.cumsum(lengths)[:-1])), unit


def sizes(**samples: np.ndarray) -> dict[str, int]:
    """Sample sizes for :attr:`TestResult.n`."""
    return {role: len(sample) for role, sample in samples.items()}


def check_confidence(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    return float(confidence)
