"""Spatial autocorrelation from explicit neighbour weights: Moran's I.

Weights are given, never derived here (building them from geometry is a
separate step): a dense ``n x n`` matrix, a SciPy sparse matrix, or an
adjacency mapping ``{i: [j, ...]}`` (weight 1) or ``{i: {j: w}}``. Values are
indexed ``0..n-1`` in the same order. Inference is by conditional
randomization: values are permuted over the locations (seeded), and the
p-value is ``(k + 1) / (B + 1)`` for the ``k`` permutations at least as
extreme (twice the smaller tail for two-sided tests). Permutation
inference needs no SciPy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from qavf.inference.results import Alternative, TestResult
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
    spread = float(z @ z)
    if spread == 0:
        raise Unassessable("the values are constant; Moran's I is undefined", n=n)
    scale = len(x) / float(w.sum()) / spread
    observed = scale * float((w * z[rows] * z[cols]).sum())
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
