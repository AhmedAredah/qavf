"""Multiple-testing corrections over a family of tests.

A family is whatever is judged together: the groups of one measure, the
buckets of a temporal measure, the series of one shape check, the columns of
a record comparison, the comparisons of a rule. Bonferroni and Holm control
the family-wise error rate; Benjamini-Hochberg (``bh``) controls the false
discovery rate under independence or positive dependence, and
Benjamini-Yekutieli (``by``) under any dependence. Adjusted p-values are
monotone in the raw ones, never below them and at most 1, and agree with
``statsmodels.stats.multitest.multipletests``.

``family_size`` may exceed the p-values given: hypotheses of the family that
were not tested count as ``p = 1``, which keeps every correction valid (and
conservative) when some members could not be assessed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from qavf.inference.results import Correction, TestResult

CORRECTIONS: tuple[Correction, ...] = ("none", "bonferroni", "holm", "bh", "by")


def _harmonic(n: int) -> float:
    """H_n without an n-element array or an optional SciPy dependency.

    Above 128, Euler--Maclaurin through n^-8 has error below 1e-22;
    round-off in binary64 dominates. Small families use compensated summation.
    """
    if n <= 128:
        return math.fsum(1.0 / k for k in range(1, n + 1))
    inverse = 1.0 / n
    square = inverse * inverse
    return (
        math.log(n)
        + 0.5772156649015329
        + inverse / 2
        - square / 12
        + square**2 / 120
        - square**3 / 252
        + square**4 / 240
    )


def adjust(
    p_values: npt.ArrayLike, method: Correction, family_size: int | None = None
) -> np.ndarray:
    """Adjusted p-values of ``p_values`` (in their order) under ``method``.

    Raises:
        ValueError: for an unknown method, p-values outside [0, 1], or a
            family smaller than the p-values given.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a 1-D array of values in [0, 1]")
    if method not in CORRECTIONS:
        raise ValueError(f"unknown correction {method!r}; use one of {', '.join(CORRECTIONS)}")
    m = len(p) if family_size is None else family_size
    if isinstance(m, bool) or not isinstance(m, int | np.integer) or m < 0:
        raise ValueError("family_size must be a non-negative integer")
    if m < len(p):
        raise ValueError(f"family_size {m} is smaller than the {len(p)} p-values given")
    if not len(p) or method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(1.0, p * m)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, len(p) + 1)
    if method == "holm":
        stepped = np.maximum.accumulate((m - ranks + 1) * ranked)
    else:
        scale = _harmonic(m) if method == "by" else 1.0
        stepped = np.minimum.accumulate((m * scale / ranks * ranked)[::-1])[::-1]
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, stepped)
    return adjusted


def correct(
    results: Sequence[TestResult],
    method: Correction,
    alpha: float,
    family_size: int | None = None,
) -> list[TestResult]:
    """The family's results, assessed ones adjusted together and judged at ``alpha``.

    Each assessed result records the adjusted p-value, the method, the family
    size and whether it is significant; results that were not assessed are
    returned unchanged (they never count as significant).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    tested = [
        i for i, result in enumerate(results) if result.assessed and result.p_value is not None
    ]
    size = len(tested) if family_size is None else family_size
    adjusted = adjust([float(results[i].p_value or 0.0) for i in tested], method, size)
    out = list(results)
    for i, p in zip(tested, adjusted, strict=True):
        out[i] = (
            results[i]
            .model_copy(update={"adjusted_p": float(p), "correction": method, "family_size": size})
            .judged(alpha)
        )
    return out
