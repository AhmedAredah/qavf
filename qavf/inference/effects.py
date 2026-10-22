"""Effect sizes and agreement metrics: how large a difference is, not whether it is real.

Effects are oriented candidate relative to baseline: a positive Cohen's d or
rank-biserial correlation means the candidate tends to be larger, a rate or
odds ratio above 1 that its rate or odds are higher. Agreement metrics compare
paired values with the baseline as the reference: ``bias`` is the mean of
``candidate - baseline`` and ``r_squared`` how much of the baseline's variance
the candidate reproduces. Distribution distances compare two samples as a
whole: the population stability index and Jensen-Shannon divergence of shared
bins, the Kolmogorov-Smirnov and Wasserstein distances of the empirical
distribution functions, and quantile-by-quantile deltas. None of these is a
test; they never need SciPy.

Undefined values (a zero pooled spread, a zero baseline for MAPE) are
``None`` rather than infinite or NaN.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qavf._observations import float_observations
from qavf.stats import center_scale_observations


def _pair(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    b, c = float_observations(baseline), float_observations(candidate)
    if not (np.isfinite(b).all() and np.isfinite(c).all()):
        raise ValueError("effect sizes need finite values")
    return b, c


def _paired(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    b, c = _pair(baseline, candidate)
    if len(b) != len(c) or not len(b):
        raise ValueError("agreement metrics need paired, nonempty values")
    return b, c


def _scaled_pair(b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    joined, unit, _ = center_scale_observations(np.concatenate([b, c]))
    return joined[: len(b)], joined[len(b) :], unit


def cohen_d(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float | None:
    """``(mean_c - mean_b) / s_pooled`` with the ``n - 1`` pooled standard deviation."""
    b, c = _pair(baseline, candidate)
    if len(b) < 2 or len(c) < 2:
        return None
    sb, sc, _ = _scaled_pair(b, c)
    pooled = ((len(b) - 1) * sb.var(ddof=1) + (len(c) - 1) * sc.var(ddof=1)) / (len(b) + len(c) - 2)
    if pooled <= 0:
        return None
    return float((sc.mean() - sb.mean()) / math.sqrt(pooled))


def hedges_g(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float | None:
    """Cohen's d times the exact small-sample correction ``J(df)``.

    ``J(df) = Gamma(df/2) / (sqrt(df/2) * Gamma((df-1)/2))`` with
    ``df = n_b + n_c - 2`` (Hedges 1981).
    """
    d = cohen_d(baseline, candidate)
    if d is None:
        return None
    df = len(np.asarray(baseline)) + len(np.asarray(candidate)) - 2
    correction = math.exp(math.lgamma(df / 2) - math.lgamma((df - 1) / 2)) / math.sqrt(df / 2)
    return d * correction


def rank_biserial(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float | None:
    """``P(c > b) - P(c < b)`` over all pairs (ties count half each way)."""
    b, c = _pair(baseline, candidate)
    if not len(b) or not len(c):
        return None
    ordered = np.sort(b)
    below = np.searchsorted(ordered, c, side="left")
    not_above = np.searchsorted(ordered, c, side="right")
    greater = float(below.sum())
    less = float((len(b) - not_above).sum())
    return (greater - less) / (len(b) * len(c))


def cramers_v(table: npt.ArrayLike) -> float | None:
    """Cramer's V of a contingency table: ``sqrt(chi2 / (n * (min(r, k) - 1)))``."""
    counts = np.asarray(table, dtype=float)
    if counts.ndim != 2 or (counts < 0).any():
        raise ValueError("a contingency table is a 2-D array of non-negative counts")
    counts = counts[counts.sum(axis=1) > 0][:, counts.sum(axis=0) > 0]
    n = counts.sum()
    smaller = min(counts.shape) - 1
    if n == 0 or smaller < 1:
        return None
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / (n * smaller))


def rate_ratio(
    baseline_count: float,
    baseline_exposure: float,
    candidate_count: float,
    candidate_exposure: float,
) -> float | None:
    """``(c / exposure_c) / (b / exposure_b)``; ``None`` for a zero baseline rate."""
    if baseline_exposure <= 0 or candidate_exposure <= 0:
        raise ValueError("exposures must be positive")
    if baseline_count == 0:
        return None
    return (candidate_count / candidate_exposure) / (baseline_count / baseline_exposure)


def odds_ratio(table: npt.ArrayLike) -> float | None:
    """Sample odds ratio ``(a d) / (b c)`` of a 2x2 table ``[[a, b], [c, d]]``.

    Rows are baseline and candidate, columns event and non-event; ``None`` when
    a zero denominator leaves it undefined or infinite.
    """
    (a, b), (c, d) = np.asarray(table, dtype=float).reshape(2, 2)
    if b * c == 0:
        return None
    return float(a * d / (b * c))


@dataclass(frozen=True, slots=True)
class BlandAltman:
    """Mean difference (``candidate - baseline``) and its 95 % limits of agreement."""

    bias: float
    sd: float
    lower: float
    upper: float
    n: int


def bland_altman(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, multiplier: float = 1.96
) -> BlandAltman:
    """Bland-Altman bias and limits ``bias -/+ multiplier * sd`` of the differences."""
    b, c = _paired(baseline, candidate)
    if len(b) < 2:
        raise ValueError("limits of agreement need at least two pairs")
    sb, sc, unit = _scaled_pair(b, c)
    difference = sc - sb
    bias = float(difference.mean()) * unit
    sd = float(difference.std(ddof=1)) * unit
    return BlandAltman(bias, sd, bias - multiplier * sd, bias + multiplier * sd, len(b))


def concordance(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float | None:
    """Lin's concordance correlation ``2 s_bc / (s_b^2 + s_c^2 + (m_b - m_c)^2)``.

    Moments use ``1/n`` (Lin 1989); ``None`` when both sides are one constant.
    """
    b, c = _paired(baseline, candidate)
    sb, sc, _ = _scaled_pair(b, c)
    mb, mc = sb.mean(), sc.mean()
    covariance = float(((sb - mb) * (sc - mc)).mean())
    denominator = float(sb.var() + sc.var() + (mb - mc) ** 2)
    return None if denominator == 0 else 2 * covariance / denominator


def bias(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """Mean of ``candidate - baseline``."""
    b, c = _paired(baseline, candidate)
    sb, sc, unit = _scaled_pair(b, c)
    return float((sc - sb).mean()) * unit


def mae(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """Mean absolute difference."""
    b, c = _paired(baseline, candidate)
    sb, sc, unit = _scaled_pair(b, c)
    return float(np.abs(sc - sb).mean()) * unit


def rmse(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """Root mean squared difference."""
    b, c = _paired(baseline, candidate)
    sb, sc, unit = _scaled_pair(b, c)
    return math.sqrt(float(((sc - sb) ** 2).mean())) * unit


def mape(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float | None:
    """Mean absolute percentage difference relative to the baseline (``None`` for a zero)."""
    b, c = _paired(baseline, candidate)
    if (b == 0).any():
        return None
    return float(np.mean(np.abs((c - b) / b))) * 100


def smape(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """Symmetric MAPE ``mean(2 |c - b| / (|b| + |c|)) * 100``; a pair of zeros counts 0."""
    b, c = _paired(baseline, candidate)
    scale = np.abs(b) + np.abs(c)
    terms = np.divide(2 * np.abs(c - b), scale, out=np.zeros(len(b)), where=scale > 0)
    return float(terms.mean()) * 100


def r_squared(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float | None:
    """``1 - sum((c - b)^2) / sum((b - mean b)^2)``: the candidate as a prediction of
    the baseline; ``None`` for a constant baseline."""
    b, c = _paired(baseline, candidate)
    sb, sc, _ = _scaled_pair(b, c)
    total = float(((sb - sb.mean()) ** 2).sum())
    if total == 0:
        return None
    return 1 - float(((sc - sb) ** 2).sum()) / total


# ------------------------------------------------------------ distributions
def population_stability(
    pairs: Iterable[tuple[float, float]], totals: tuple[float, float], floor: float = 1e-6
) -> float:
    """Population stability index ``sum((a - e) * ln(a / e))`` over bins or categories.

    ``pairs`` are ``(baseline, candidate)`` counts per bin (streamed, in order) and
    ``totals`` the sides' sizes; ``e`` and ``a`` are the shares, raised to ``floor``
    so an empty bin stays finite. Summed with ``math.fsum``: STA-002's formula.
    """
    if totals[0] <= 0 or totals[1] <= 0:
        raise ValueError("PSI needs a nonempty sample on each side")

    def terms() -> Iterator[float]:
        for reference, current in pairs:
            expected = max(reference / totals[0], floor)
            actual = max(current / totals[1], floor)
            yield (actual - expected) * math.log(actual / expected)

    return math.fsum(terms())


def jensen_shannon(baseline_counts: npt.ArrayLike, candidate_counts: npt.ArrayLike) -> float:
    """Jensen-Shannon divergence (base 2, from 0 to 1) of two count vectors on shared bins.

    ``(KL(p || m) + KL(q || m)) / 2`` with ``m`` the mean of the two share vectors;
    SciPy's ``jensenshannon`` distance is its square root.
    """
    b, c = (np.asarray(counts, dtype=float) for counts in (baseline_counts, candidate_counts))
    if b.shape != c.shape or b.ndim != 1 or (b < 0).any() or (c < 0).any():
        raise ValueError("Jensen-Shannon needs two count vectors of the same length")
    if b.sum() <= 0 or c.sum() <= 0:
        raise ValueError("Jensen-Shannon needs a nonempty sample on each side")
    p, q = b / b.sum(), c / c.sum()
    m = (p + q) / 2

    def kl(x: np.ndarray) -> float:
        kept = x > 0
        return float(np.sum(x[kept] * np.log2(x[kept] / m[kept])))

    return min(1.0, max(0.0, (kl(p) + kl(q)) / 2))


def _distribution_functions(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Both empirical distribution functions on the pooled sorted (scaled) values."""
    b, c = _pair(baseline, candidate)
    if not len(b) or not len(c):
        raise ValueError("distribution distances need a nonempty sample on each side")
    sb, sc, unit = _scaled_pair(b, c)
    sb, sc = np.sort(sb), np.sort(sc)
    pooled = np.sort(np.concatenate([sb, sc]))
    below_b = np.searchsorted(sb, pooled, side="right") / len(sb)
    below_c = np.searchsorted(sc, pooled, side="right") / len(sc)
    return pooled, below_b, below_c, unit


def ks_distance(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """Kolmogorov-Smirnov distance: the largest gap between the two empirical CDFs."""
    _, below_b, below_c, _ = _distribution_functions(baseline, candidate)
    return float(np.abs(below_b - below_c).max())


def wasserstein(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """First Wasserstein (earth mover's) distance, in data units.

    The area between the empirical CDFs, ``sum |F_b - F_c| dx`` over the pooled
    sorted values, as SciPy's ``wasserstein_distance`` of the two samples.
    """
    pooled, below_b, below_c, unit = _distribution_functions(baseline, candidate)
    gaps = np.diff(pooled)
    return float(np.sum(np.abs(below_b[:-1] - below_c[:-1]) * gaps)) * unit


def quantile_deltas(
    baseline: npt.ArrayLike, candidate: npt.ArrayLike, probabilities: Iterable[float]
) -> list[tuple[float, float, float]]:
    """``(p, baseline quantile, candidate quantile)`` per probability (NumPy's linear)."""
    b, c = _pair(baseline, candidate)
    if not len(b) or not len(c):
        raise ValueError("quantiles need a nonempty sample on each side")
    found = []
    for p in probabilities:
        if not 0 <= p <= 1:
            raise ValueError("quantile probabilities must lie in [0, 1]")
        found.append((float(p), float(np.quantile(b, p)), float(np.quantile(c, p))))
    return found
