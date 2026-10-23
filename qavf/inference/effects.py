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
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from qavf._observations import float_observations
from qavf.inference.intervals import linear_quantiles
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
    if counts.ndim != 2 or (counts < 0).any() or not np.isfinite(counts).all():
        raise ValueError("a contingency table is a 2-D array of non-negative counts")
    if not counts.size or not counts.max() > 0:
        return None
    original = counts[np.any(counts > 0, axis=1)][:, np.any(counts > 0, axis=0)]
    smaller = min(original.shape) - 1
    if smaller < 1:
        return None
    counts = original / original.max()
    shares = counts / counts.sum()
    rows, columns = shares.sum(axis=1), shares.sum(axis=0)
    expected = rows[:, None] * columns[None, :]
    difference = shares - expected
    if (
        np.any((original > 0) & (shares == 0))
        or np.any(expected == 0)
        or np.max(np.abs(difference)) <= 16 * np.finfo(float).eps * shares.max()
    ):
        # Rare categories can carry perfect association even when a common
        # float scale erases them. Near independence, subtracting rounded
        # margins can also erase a small but representable effect.
        # Keep only the margins, not a second matrix of Python Fraction cells.
        row_totals = [Fraction(0)] * original.shape[0]
        column_totals = [Fraction(0)] * original.shape[1]
        for i, row in enumerate(original):
            for j, value in enumerate(row):
                exact = Fraction(float(value))
                row_totals[i] += exact
                column_totals[j] += exact
        total = sum(row_totals, Fraction(0))
        smaller = min(sum(value > 0 for value in row_totals),
                      sum(value > 0 for value in column_totals)) - 1
        if smaller < 1:
            return None
        statistic = sum((
            (Fraction(float(value)) * total - row_total * column_total) ** 2
            / (row_total * column_total * total * total)
            for row, row_total in zip(original, row_totals, strict=True) if row_total
            for value, column_total in zip(row, column_totals, strict=True) if column_total
        ), Fraction(0))
        squared = statistic / smaller
        if not squared:
            return 0.0
        # Convert after bringing the squared effect near one: its square can
        # underflow Float64 while the effect itself remains representable.
        exponent = squared.numerator.bit_length() - squared.denominator.bit_length()
        even = 2 * (exponent // 2)
        return math.ldexp(math.sqrt(float(squared * Fraction(2) ** -even)), even // 2)
    # Divide by the square roots separately: the expected cell probability can
    # underflow although its square root and the standardized residual do not.
    residual = difference / np.sqrt(rows[:, None]) / np.sqrt(columns[None, :])
    magnitude = float(np.max(np.abs(residual)))
    norm = math.sqrt(float(np.sum((residual / magnitude) ** 2))) * magnitude
    return min(1.0, norm / math.sqrt(smaller))


def rate_ratio(
    baseline_count: float,
    baseline_exposure: float,
    candidate_count: float,
    candidate_exposure: float,
) -> float | None:
    """``(c / exposure_c) / (b / exposure_b)``, without intermediate overflow.

    ``None`` denotes a zero baseline rate or an unrepresentably large ratio.
    """
    if not all(math.isfinite(x) and x > 0 for x in (baseline_exposure, candidate_exposure)):
        raise ValueError("exposures must be positive and finite")
    if not all(math.isfinite(x) and x >= 0 for x in (baseline_count, candidate_count)):
        raise ValueError("counts must be finite and non-negative")
    if baseline_count == 0:
        return None
    if candidate_count == 0:
        return 0.0
    # Work with binary mantissas: all products remain in [1/4, 1), even when
    # either individual rate would overflow or underflow as a float.
    cm, ce = math.frexp(candidate_count)
    bm, be = math.frexp(baseline_count)
    ebm, ebe = math.frexp(baseline_exposure)
    ecm, ece = math.frexp(candidate_exposure)
    try:
        return math.ldexp((cm * ebm) / (bm * ecm), ce + ebe - be - ece)
    except OverflowError:
        return None


def odds_ratio(table: npt.ArrayLike) -> float | None:
    """Sample odds ratio ``(a d) / (b c)`` of a 2x2 table ``[[a, b], [c, d]]``.

    Rows are baseline and candidate, columns event and non-event; ``None`` when
    a zero denominator leaves it undefined or infinite.
    """
    (a, b), (c, d) = np.asarray(table, dtype=float).reshape(2, 2)
    if not all(math.isfinite(v) and v >= 0 for v in (a, b, c, d)):
        raise ValueError("odds ratio needs finite non-negative counts")
    if b == 0 or c == 0:
        return None
    if a == 0 or d == 0:
        return 0.0
    return rate_ratio(float(b), float(a), float(d), float(c))


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
    exponent = np.frexp(np.maximum(np.abs(b), np.abs(c)))[1]
    sb, sc = np.ldexp(b, -exponent), np.ldexp(c, -exponent)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        terms = np.abs((sc - sb) / sb)
        largest = float(terms.max())
        if not math.isfinite(largest):
            return None
        result = float(np.mean(terms / largest)) * largest * 100 if largest else 0.0
    return result if math.isfinite(result) else None


def smape(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """Symmetric MAPE ``mean(2 |c - b| / (|b| + |c|)) * 100``; a pair of zeros counts 0."""
    b, c = _paired(baseline, candidate)
    exponent = np.frexp(np.maximum(np.abs(b), np.abs(c)))[1]
    sb, sc = np.ldexp(b, -exponent), np.ldexp(c, -exponent)
    scale = np.abs(sb) + np.abs(sc)
    terms = np.divide(2 * np.abs(sc - sb), scale, out=np.zeros(len(b)), where=scale > 0)
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
    if (
        b.shape != c.shape or b.ndim != 1 or (b < 0).any() or (c < 0).any()
        or not np.isfinite(b).all() or not np.isfinite(c).all()
    ):
        raise ValueError("Jensen-Shannon needs two count vectors of the same length")
    if not len(b) or not b.max() > 0 or not c.max() > 0:
        raise ValueError("Jensen-Shannon needs a nonempty sample on each side")
    b, c = b / b.max(), c / c.max()
    p, q = b / b.sum(), c / c.sum()
    combined = p + q

    def kl(x: np.ndarray) -> float:
        kept = x > 0
        return float(np.sum(x[kept] * np.log2(2 * x[kept] / combined[kept])))

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
    probabilities = list(probabilities)
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("quantile probabilities must lie in [0, 1]")
    if not probabilities:
        return []
    before, after = linear_quantiles(b, probabilities), linear_quantiles(c, probabilities)
    return [
        (float(p), float(x), float(y)) for p, x, y in zip(probabilities, before, after, strict=True)
    ]
