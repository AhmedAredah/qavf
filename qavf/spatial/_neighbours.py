"""Bounded neighbour construction with optional exact spatial indexing."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from qavf._adjacency import CompactAdjacency
from qavf.spatial.distance import EARTH_RADIUS_M, distance

DEFAULT_MAX_EDGES = 1_000_000
DEFAULT_MAX_WORK = 20_000_000
_BLOCK_PAIRS = 262_144


@dataclass(slots=True)
class _Admission:
    """Bound explicit coordinate/distance work and retained directed edges."""

    max_work: int
    max_edges: int
    work: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_work, bool)
            or not isinstance(self.max_work, int | np.integer)
            or self.max_work < 1
        ):
            raise ValueError("max_work must be a positive integer")
        if (
            isinstance(self.max_edges, bool)
            or not isinstance(self.max_edges, int | np.integer)
            or self.max_edges < 0
        ):
            raise ValueError("max_edges must be a non-negative integer")
        self.max_work, self.max_edges = int(self.max_work), int(self.max_edges)

    def add_work(self, count: int) -> None:
        if self.work + count > self.max_work:
            raise ValueError(f"neighbour construction exceeds max_work={self.max_work:,}")
        self.work += count

    def edges(self, count: int) -> None:
        if count > self.max_edges:
            raise ValueError(
                f"neighbour graph requires {count:,} edges, above max_edges={self.max_edges:,}"
            )


def _coordinates(x: Any, y: Any, n: int, geographic: bool) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1 or len(xs) != n:
        raise ValueError("coordinates must be two 1-D arrays matching the location ids")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise ValueError("coordinates must be finite")
    if geographic and ((ys < -90).any() or (ys > 90).any()):
        raise ValueError("geographic latitudes must lie in -90..90 degrees")
    if geographic and ((xs < -180).any() or (xs > 180).any()):
        raise ValueError("geographic longitudes must lie in -180..180 degrees")
    return xs, ys


def _points(xs: np.ndarray, ys: np.ndarray, geographic: bool) -> tuple[np.ndarray, int]:
    if not geographic:
        magnitude = max(float(np.max(np.abs(xs), initial=0)), float(np.max(np.abs(ys), initial=0)))
        exponent = math.frexp(magnitude)[1]
        # Exact power-of-two scaling protects tree squared norms from overflow.
        return np.column_stack((np.ldexp(xs, -exponent), np.ldexp(ys, -exponent))), exponent
    lon, lat = np.radians(xs), np.radians(ys)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))), 0


def _tree(points: np.ndarray) -> Any:
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    return cKDTree(points)


def _radius(value: float) -> float:
    # Candidate admission includes floating-point projection/norm uncertainty;
    # final acceptance/order always uses QAVF's original distance function.
    return float(np.nextafter(value + max(abs(value), 1.0) * 64 * np.finfo(float).eps, np.inf))


def _blocks(n: int) -> Iterator[slice]:
    size = max(1, _BLOCK_PAIRS // max(n, 1))
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def _nearest_row(indices: np.ndarray, distances: np.ndarray, k: int) -> np.ndarray:
    """Partial selection with exact distance-then-index tie semantics."""
    if np.isnan(distances).any():
        raise ValueError("coordinates have undefined neighbour distances")
    boundary = np.partition(distances, k - 1)[k - 1]
    if not np.isfinite(boundary):
        raise ValueError("coordinates cannot be represented safely for neighbour distances")
    lower = np.flatnonzero(distances < boundary)
    tied = np.flatnonzero(distances == boundary)
    remaining = k - len(lower)
    tied = tied[np.argsort(indices[tied], kind="stable")[:remaining]]
    return np.sort(indices[np.concatenate((lower, tied))])


def nearest_graph(
    x: Any,
    y: Any,
    n: int,
    k: int,
    *,
    geographic: bool,
    max_edges: int,
    max_work: int,
) -> CompactAdjacency:
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or not 1 <= k < n:
        raise ValueError(f"k must lie in 1..{n - 1} for {n} locations, got {k}")
    k = int(k)
    admission = _Admission(max_work, max_edges)
    admission.edges(n * k)
    admission.add_work(n)
    xs, ys = _coordinates(x, y, n, geographic)
    points, _ = _points(xs, ys, geographic)
    tree = _tree(points)
    rows: list[np.ndarray] = []
    if tree is None:
        admission.add_work(n * n)
        for block in _blocks(n):
            values = distance(
                xs[block, None], ys[block, None], xs[None, :], ys[None, :], geographic=geographic
            )
            for offset, i in enumerate(range(block.start, block.stop)):
                candidates = np.concatenate((np.arange(i), np.arange(i + 1, n)))
                rows.append(_nearest_row(candidates, values[offset, candidates], k))
        return CompactAdjacency.from_rows(rows)
    # The result allocation is bounded by the already-admitted n*k edge count.
    limits, _ = tree.query(points, k=k + 1, eps=0, workers=1)
    for i in range(n):
        radius = _radius(float(limits[i, -1]))
        count = int(tree.query_ball_point(points[i], radius, return_length=True))
        admission.add_work(count)
        candidates = np.asarray(tree.query_ball_point(points[i], radius), dtype=np.int64)
        candidates = candidates[candidates != i]
        values = distance(xs[i], ys[i], xs[candidates], ys[candidates], geographic=geographic)
        if len(candidates) < k or not np.isfinite(values).all():
            raise ValueError("coordinates cannot be represented safely for neighbour distances")
        rows.append(_nearest_row(candidates, values, k))
    return CompactAdjacency.from_rows(rows)


def band_graph(
    x: Any,
    y: Any,
    n: int,
    threshold: float,
    *,
    geographic: bool,
    max_edges: int,
    max_work: int,
) -> CompactAdjacency:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("distance band must be finite and non-negative")
    admission = _Admission(max_work, max_edges)
    admission.add_work(n)
    xs, ys = _coordinates(x, y, n, geographic)
    if not n:
        return CompactAdjacency.from_rows([])
    points, exponent = _points(xs, ys, geographic)
    tree = _tree(points)
    rows: list[np.ndarray] = []
    pairs = 0
    if tree is None:
        admission.add_work(n * n)
        for block in _blocks(n):
            values = distance(
                xs[block, None], ys[block, None], xs[None, :], ys[None, :], geographic=geographic
            )
            if np.isnan(values).any():
                raise ValueError("coordinates have undefined neighbour distances")
            for offset, i in enumerate(range(block.start, block.stop)):
                near = np.flatnonzero(values[offset] <= threshold)
                near = near[near != i]
                pairs += len(near)
                admission.edges(pairs)
                rows.append(near)
        return CompactAdjacency.from_rows(rows)
    if geographic:
        radius = 2 * math.sin(min(threshold / EARTH_RADIUS_M, math.pi) / 2)
    else:
        try:
            radius = math.ldexp(threshold, -exponent)
        except OverflowError:
            radius = math.inf
    radius = _radius(radius)
    # Counts avoid constructing an unbounded Python list of all graph edges.
    counts = tree.query_ball_point(points, radius, return_length=True, workers=1)
    admission.add_work(int(np.sum(counts)))
    for i in range(n):
        # Even when the radius admits numerical boundary candidates, no one
        # row may materialize beyond the explicit work allowance.
        candidates = np.asarray(tree.query_ball_point(points[i], radius), dtype=np.int64)
        candidates = candidates[candidates != i]
        values = distance(xs[i], ys[i], xs[candidates], ys[candidates], geographic=geographic)
        if np.isnan(values).any():
            raise ValueError("coordinates have undefined neighbour distances")
        near = np.sort(candidates[values <= threshold])
        pairs += len(near)
        admission.edges(pairs)
        rows.append(near)
    return CompactAdjacency.from_rows(rows)
