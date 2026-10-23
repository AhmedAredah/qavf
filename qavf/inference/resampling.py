"""Seeded resampling (i.i.d. and block bootstraps), the work budget, and HAC errors.

Every resampling test draws from ``numpy.random.Generator(PCG64(seed))`` and
records the seed. Reproduce a run with the same inputs, options and locked
numerical environment; NumPy does not guarantee identical streams across
arbitrary builds or changed draw calls. The work budget caps
the estimated values processed (including permutations and leave-one-out
calculations). A test that would exceed ``max_work`` is
not assessed, with its required work reported.

For sequential observations (time buckets) the circular block bootstrap
resamples blocks of consecutive observations, wrapping at the end, so
short-range dependence survives inside each block. The default block length
is ``ceil(2 * n ** (1/3))``: the rate-optimal order ``n^(1/3)`` for variance
and distribution estimation (Hall, Horowitz and Jing 1995), with a constant
chosen by simulation so that tests on AR(1) series with autocorrelation 0.5
hold their level (``tests/test_inference_calibration.py``). Slopes of dependent series use
Newey-West (Bartlett kernel) standard errors with ``floor(4 * (n / 100) **
(2 / 9))`` lags by default (Newey and West 1994).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np

from qavf.inference._support import Unassessable

DEFAULT_SEED = 0
"""Seed of every resampling test unless configured."""
DEFAULT_RESAMPLES = 2_000
"""Bootstrap and permutation resamples unless configured."""
DEFAULT_MAX_WORK = 20_000_000
"""Largest estimated work one test may spend (values touched)."""
_CHUNK = 1_000_000
"""Values resampled per batch, bounding the memory of one test."""

type Statistic = Callable[[np.ndarray], np.ndarray]
"""A statistic computed along the last axis (``lambda a: a.mean(axis=-1)``)."""


def check_resamples(resamples: int) -> int:
    """At least two draws are needed for resampling uncertainty estimates."""
    if isinstance(resamples, bool) or not isinstance(resamples, int | np.integer) or resamples < 2:
        raise ValueError("resamples must be an integer >= 2")
    return int(resamples)


def checked_statistics(values: object, shape: tuple[int, ...]) -> np.ndarray:
    """Admit exactly one finite statistic per last-axis sample, never broadcast it."""
    result = np.asarray(values, dtype=float)
    if result.shape != shape:
        raise Unassessable(
            f"statistic returned shape {result.shape}; expected {shape} (one value per sample)"
        )
    if not np.isfinite(result).all():
        raise Unassessable("the statistic is not finite in the observations or resamples")
    return result


def evaluate(statistic: Statistic, values: np.ndarray) -> np.ndarray:
    """Evaluate a last-axis statistic in the units supplied by its caller."""
    return checked_statistics(statistic(values), values.shape[:-1])


def jackknife(sample: np.ndarray, statistic: Statistic) -> np.ndarray:
    """Leave-one-out statistics with bounded sample batches, in input order."""
    n = len(sample)
    out = np.empty(n)
    done = 0
    for step in batches(n, n):
        positions = np.arange(n - 1)[None, :]
        omitted = np.arange(done, done + step)[:, None]
        indices = positions + (positions >= omitted)
        out[done : done + step] = evaluate(statistic, sample[indices])
        done += step
    return out


def generator(seed: int) -> np.random.Generator:
    """The deterministic generator of a resampling test."""
    return np.random.Generator(np.random.PCG64(seed))


def check_work(work: int, max_work: int, what: str) -> None:
    """Refuse a test whose work exceeds the budget.

    Raises:
        Unassessable: naming the work needed and the budget.
    """
    if isinstance(max_work, bool) or not isinstance(max_work, int | np.integer) or max_work < 0:
        raise ValueError("max_work must be a non-negative integer")
    if work > max_work:
        raise Unassessable(
            f"{what} needs {work:,} units of work (values processed), more than "
            f"max_work {max_work:,}"
        )


def batches(resamples: int, n: int) -> Iterator[int]:
    """Resample counts per batch so one batch holds about :data:`_CHUNK` values."""
    size = max(1, _CHUNK // max(n, 1))
    done = 0
    while done < resamples:
        step = min(size, resamples - done)
        yield step
        done += step


def bootstrap_distribution(
    rng: np.random.Generator,
    samples: tuple[np.ndarray, ...],
    combine: Callable[..., np.ndarray],
    resamples: int,
) -> np.ndarray:
    """``combine`` of independent i.i.d. resamples of each sample, ``resamples`` times.

    ``combine`` receives one ``(batch, n_i)`` array per sample and returns the
    ``batch`` statistics.
    """
    check_resamples(resamples)
    out = np.empty(resamples)
    done = 0
    total = sum(len(sample) for sample in samples)
    for step in batches(resamples, total):
        drawn = [
            sample[rng.integers(0, len(sample), size=(step, len(sample)))] for sample in samples
        ]
        out[done : done + step] = checked_statistics(combine(*drawn), (step,))
        done += step
    return out


def permutation_p_value(
    distribution: np.ndarray, observed: float, alternative: str = "two-sided", exact: bool = False
) -> float:
    """p-value of ``observed`` against its permutation distribution.

    One-sided: the share at least as extreme, ``k / N`` over all permutations
    (``exact``) or ``(k + 1) / (B + 1)`` over random ones; ties within a relative
    ``1e-12`` count as extreme. Two-sided: twice the smaller one-sided value,
    capped at 1 (SciPy's convention).
    """
    checked_statistics(distribution, (len(distribution),))
    checked_statistics(observed, ())
    if not len(distribution):
        raise Unassessable("no resampling statistics are available")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative {alternative!r}")
    tolerance = 1e-12 * abs(observed)
    above = int(np.count_nonzero(distribution >= observed - tolerance))
    below = int(np.count_nonzero(distribution <= observed + tolerance))
    return permutation_p_from_counts(above, below, len(distribution), alternative, exact)


def permutation_p_from_counts(
    above: int, below: int, resamples: int, alternative: str, exact: bool = False
) -> float:
    """Tail counts allow permutation distributions to be consumed in bounded batches."""
    extra = 0 if exact else 1
    greater = (above + extra) / (resamples + extra)
    less = (below + extra) / (resamples + extra)
    if alternative == "greater":
        return float(greater)
    if alternative == "less":
        return float(less)
    if alternative != "two-sided":
        raise ValueError(f"unknown alternative {alternative!r}")
    return float(min(1.0, 2 * min(greater, less)))


def interval_p_value(distribution: np.ndarray, alternative: str = "two-sided") -> float:
    """p-value of ``difference = 0`` by inverting the percentile interval.

    The smallest ``1 - level`` whose interval excludes zero: ``2 * min(#(d* <= 0)
    + 1, #(d* >= 0) + 1) / (B + 1)`` capped at 1, or one tail for ``greater``
    (a positive difference) and ``less``.
    """
    checked_statistics(distribution, (len(distribution),))
    resamples = len(distribution)
    if not resamples:
        raise Unassessable("no resampling statistics are available")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative {alternative!r}")
    upper = (np.count_nonzero(distribution <= 0) + 1) / (resamples + 1)
    lower = (np.count_nonzero(distribution >= 0) + 1) / (resamples + 1)
    if alternative == "greater":
        return float(upper)
    if alternative == "less":
        return float(lower)
    return float(min(1.0, 2 * min(upper, lower)))


def default_block_length(n: int) -> int:
    """``ceil(2 * n ** (1/3))``, at most ``n``: blocks grow with the series, slower."""
    return max(1, min(n, math.ceil(2 * n ** (1 / 3) - 1e-12)))


def block_indices(
    rng: np.random.Generator, n: int, resamples: int, block: int, circular: bool = True
) -> np.ndarray:
    """``(resamples, n)`` positions of a circular (or moving) block bootstrap.

    Each resample concatenates blocks of ``block`` consecutive positions with
    uniformly drawn starts, truncated to ``n``. Circular blocks wrap around the
    end so every position is equally likely; moving blocks start at most
    ``n - block``.
    """
    if not 1 <= block <= n:
        raise ValueError(f"block length must lie in [1, {n}], got {block}")
    count = math.ceil(n / block)
    starts = rng.integers(0, n if circular else n - block + 1, size=(resamples, count))
    offsets = np.arange(block)
    positions = (starts[:, :, None] + offsets).reshape(resamples, count * block)[:, :n]
    return positions % n if circular else positions


def default_hac_lags(n: int) -> int:
    """Newey-West lag truncation ``floor(4 * (n / 100) ** (2 / 9))``."""
    return math.floor(4 * (n / 100) ** (2 / 9))


def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Least-squares ``(slope, intercept, centred x, residuals)`` of ``y`` on ``x``."""
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx == 0:
        raise Unassessable("the positions are all equal; no slope is defined")
    slope = float(xc @ (y - y.mean())) / sxx
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept, xc, y - intercept - slope * x


def newey_west_slope(x: np.ndarray, y: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """OLS slope of ``y`` on ``x`` and its Newey-West standard error.

    Bartlett weights ``1 - l / (lags + 1)``; the small-sample factor
    ``n / (n - 2)`` matches statsmodels' ``HAC`` with ``use_correction=True``.
    """
    n = len(x)
    lags = default_hac_lags(n) if lags is None else lags
    if not 0 <= lags < n:
        raise ValueError(f"HAC lags must lie in [0, {n - 1}], got {lags}")
    slope, _, xc, residual = ols_slope(x, y)
    scores = xc * residual
    meat = float(scores @ scores)
    for lag in range(1, lags + 1):
        meat += 2 * (1 - lag / (lags + 1)) * float(scores[lag:] @ scores[:-lag])
    sxx = float(xc @ xc)
    variance = meat / sxx**2 * n / (n - 2)
    return slope, math.sqrt(max(variance, 0.0))
