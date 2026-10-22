"""Spatial autocorrelation from explicit neighbour weights: global and local Moran's I.

Weights are given, never derived here (building them is
:mod:`qavf.spatial.weights`): a dense ``n x n`` matrix, a SciPy sparse matrix,
or an adjacency mapping ``{i: [j, ...]}`` (weight 1) or ``{i: {j: w}}``.
Values are indexed ``0..n-1`` in the same order. Inference is by
randomization: for the global statistic values are permuted over the
locations (seeded), and the p-value is ``(k + 1) / (B + 1)`` for the ``k``
permutations at least as extreme (twice the smaller tail for two-sided tests).

The local statistic (LISA, Anselin 1995) ``I_i = z_i * sum_j w_ij z_j / m2``
says where values cluster. Its inference is *conditional* randomization: each
location keeps its own value while its neighbours' values are drawn without
replacement from the other ``n - 1`` locations; one seeded set of draws serves
every location (as in PySAL), so each p-value is exact for its location while
the family stays correlated. Correct the family with
:func:`qavf.inference.multiple.correct` before naming clusters. Permutation
inference needs no SciPy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qavf.inference._support import Unassessable, assessed, at_least, observations
from qavf.inference.resampling import (
    DEFAULT_MAX_WORK,
    DEFAULT_SEED,
    batches,
    check_work,
    generator,
    permutation_p_value,
)
from qavf.inference.results import Alternative, TestResult, not_assessed
from qavf.stats import center_scale_observations

DEFAULT_PERMUTATIONS = 999


def edges(weights: Any, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, columns, weights)`` of the nonzero weights, self-pairs excluded.

    Raises:
        ValueError: for a malformed matrix or mapping, an index outside
            ``0..n-1``, or a negative or non-finite weight.
    """
    if isinstance(weights, Mapping) or (
        isinstance(weights, Sequence) and weights and not np.ndim(weights[0])
    ):
        items = weights.items() if isinstance(weights, Mapping) else enumerate(weights)
        rows, cols, values = [], [], []
        for i, neighbours in items:
            pairs = neighbours.items() if isinstance(neighbours, Mapping) else (
                (j, 1.0) for j in neighbours
            )  # fmt: skip
            for j, w in pairs:
                rows.append(int(i))
                cols.append(int(j))
                values.append(float(w))
        r, c, w = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(values)
    elif hasattr(weights, "tocoo"):
        coo = weights.tocoo()
        if coo.shape != (n, n):
            raise ValueError(f"weights must be {n} x {n}, got {coo.shape}")
        r, c, w = coo.row.astype(np.int64), coo.col.astype(np.int64), np.asarray(coo.data, float)
    else:
        matrix = np.asarray(weights, dtype=float)
        if matrix.shape != (n, n):
            raise ValueError(f"weights must be {n} x {n}, got {matrix.shape}")
        r, c = np.nonzero(matrix)
        w = matrix[r, c]
    if len(r) and (r.min() < 0 or c.min() < 0 or r.max() >= n or c.max() >= n):
        raise ValueError(f"neighbour indices must lie in 0..{n - 1}")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative")
    keep = (r != c) & (w != 0)
    return r[keep], c[keep], w[keep]


def _prepared(
    values: npt.ArrayLike, weights: Any, row_standardize: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(x, rows, columns, weights, z)``: admitted values, edges and centred values.

    Raises:
        Unassessable: for fewer than 3 values, no neighbour pair or constant values.
    """
    x = observations(values, "values")
    n = {"locations": len(x)}
    at_least(len(x), 3, "locations", n=n)
    rows, cols, w = edges(weights, len(x))
    if not len(w):
        raise Unassessable("no location has a neighbour", n=n)
    if row_standardize:
        totals = np.bincount(rows, weights=w, minlength=len(x))
        w = w / totals[rows]
    try:
        z = center_scale_observations(x)[0]
    except ValueError as exc:
        raise Unassessable(str(exc), n=n) from None
    z = z - z.mean()
    if float(z @ z) == 0:
        raise Unassessable("the values are constant; Moran's I is undefined", n=n)
    return x, rows, cols, w, z


def morans_statistic(
    values: npt.ArrayLike, weights: Any, *, row_standardize: bool = False
) -> float:
    """Moran's I of ``values`` alone, without inference (e.g. to judge its size).

    Raises:
        ValueError: when it is undefined (fewer than 3 values, no neighbour pair,
            constant values), with the reason.
    """
    try:
        prepared = _prepared(values, weights, row_standardize)
    except Unassessable as exc:
        raise ValueError(str(exc)) from None
    return _observed(*prepared)[1]


def _observed(
    x: np.ndarray, rows: np.ndarray, cols: np.ndarray, w: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """``(n / (S0 * sum z^2), I)``: the scale of every permuted statistic, and I itself."""
    scale = len(x) / float(w.sum()) / float(z @ z)
    return scale, scale * float((w * z[rows] * z[cols]).sum())


@assessed("morans_i")
def morans_i(
    values: npt.ArrayLike,
    weights: Any,
    *,
    alternative: Alternative = "two-sided",
    row_standardize: bool = False,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
) -> TestResult:
    """Moran's I of ``values`` under ``weights``, with permutation inference.

    ``I = (n / S0) * sum_ij w_ij z_i z_j / sum_i z_i^2`` for deviations ``z``
    from the mean. ``greater`` tests positive autocorrelation (similar values
    cluster), ``less`` negative. ``row_standardize`` divides each row by its
    total first. At least 3 values; constant values or no neighbour pairs are
    not assessed; locations without neighbours are noted.
    """
    x, rows, cols, w, z = _prepared(values, weights, row_standardize)
    n = {"locations": len(x)}
    scale, observed = _observed(x, rows, cols, w, z)
    check_work(permutations * len(w), max_work, "Moran's I permutations")
    rng = generator(seed)
    distribution = np.empty(permutations)
    done = 0
    for step in batches(permutations, max(len(w), len(x))):
        shuffled = rng.permuted(np.broadcast_to(z, (step, len(z))), axis=1)
        distribution[done : done + step] = scale * (shuffled[:, rows] * shuffled[:, cols] * w).sum(
            axis=1
        )
        done += step
    p = permutation_p_value(distribution, observed, alternative)
    islands = len(x) - len(np.unique(rows))
    sd = float(distribution.std(ddof=1))
    notes = [
        f"expected I {-1 / (len(x) - 1):.6g} under no autocorrelation",
        f"permutation z {(observed - distribution.mean()) / sd:.6g}"
        if sd
        else "permutation z: n/a",
    ]
    if islands:
        notes.append(f"{islands} location(s) without neighbours")
    return TestResult(
        method="morans_i",
        statistic=observed,
        statistic_name="I",
        p_value=float(p),
        alternative=alternative,
        effect=observed,
        effect_name="morans_i",
        n=n,
        assumptions=["exchangeable values under no autocorrelation", "weights given as input"],
        dependence="spatial",
        seed=seed,
        resamples=permutations,
        notes=notes,
    )


QUADRANTS = {(True, True): "HH", (False, False): "LL", (True, False): "HL", (False, True): "LH"}
"""``(value above the mean, neighbours above the mean)``: HH and LL are clusters
(hot and cold spots), HL and LH spatial outliers."""


@dataclass(frozen=True)
class LocalMoran:
    """Each location's local Moran's I (LISA) test and its quadrant."""

    results: list[TestResult]
    """One per location, in location order; islands and unusable data are not assessed."""
    quadrants: list[str | None]
    """``HH``, ``LL``, ``HL`` or ``LH`` per location (``None``: an island, a value at
    the mean, or not assessed)."""

    def significant(self) -> list[int]:
        """Locations whose (corrected, once judged) test is significant."""
        return [i for i, result in enumerate(self.results) if result.significant]


def local_morans_i(
    values: npt.ArrayLike,
    weights: Any,
    *,
    alternative: Alternative = "greater",
    row_standardize: bool = True,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    max_work: int = DEFAULT_MAX_WORK,
) -> LocalMoran:
    """Local Moran's I of every location, with conditional permutation p-values.

    ``I_i = z_i * sum_j w_ij z_j / m2`` for deviations ``z`` from the mean and
    ``m2 = sum z^2 / n``, with row-standardized weights by default. ``greater``
    (the default) tests positive local autocorrelation: a cluster of similar
    values (HH or LL); ``less`` a spatial outlier. Nothing raises: an island, or
    data the statistic cannot use (fewer than 3 values, constant values, the
    work budget), gives not-assessed results with the reason. The p-values are
    raw; correct the family before judging it.
    """
    try:
        x, rows, cols, w, z = _prepared(values, weights, row_standardize)
        counts = np.bincount(rows, minlength=len(x))
        check_work(permutations * int(counts.sum()), max_work, "local Moran's I permutations")
    except Unassessable as exc:
        size = len(np.atleast_1d(np.asarray(values, dtype=object)))
        result = not_assessed("local_morans_i", str(exc), **exc.fields)
        return LocalMoran([result] * size, [None] * size)
    n = len(x)
    m2 = float(z @ z) / n
    lag = np.bincount(rows, weights=w * z[cols], minlength=n)
    local = z * lag / m2
    order = np.argsort(rows, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)])
    rng = generator(seed)
    widest = int(counts.max())
    # One draw of neighbour positions per permutation, shared by every location.
    draws = np.stack([rng.permutation(n - 1)[:widest] for _ in range(permutations)])
    results: list[TestResult] = []
    quadrants: list[str | None] = []
    for i in range(n):
        k = int(counts[i])
        sizes = {"locations": n, "neighbours": k}
        if not k:
            reason = "the location has no neighbours"
            results.append(not_assessed("local_morans_i", reason, n=sizes))
            quadrants.append(None)
            continue
        own = w[order[starts[i] : starts[i + 1]]]
        positions = draws[:, :k]
        positions = positions + (positions >= i)
        distribution = z[i] * (z[positions] * own).sum(axis=1) / m2
        p = permutation_p_value(distribution, float(local[i]), alternative)
        quadrant = None if z[i] == 0 or lag[i] == 0 else QUADRANTS[(z[i] > 0, lag[i] > 0)]
        quadrants.append(quadrant)
        results.append(
            TestResult(
                method="local_morans_i",
                statistic=float(local[i]),
                statistic_name="I_i",
                p_value=float(p),
                alternative=alternative,
                effect=float(local[i]),
                effect_name="local_morans_i",
                n=sizes,
                assumptions=[
                    "exchangeable values under no autocorrelation",
                    "conditional randomization (the location's own value fixed)",
                    "weights given as input",
                ],
                dependence="spatial",
                seed=seed,
                resamples=permutations,
                notes=[f"quadrant {quadrant}" if quadrant else "quadrant: none"],
            )
        )
    return LocalMoran(results, quadrants)
