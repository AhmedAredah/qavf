"""Compact, read-only sparse adjacency shared by geometry and numerical kernels."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class _Row(Mapping[int, float]):
    indices: np.ndarray
    weights: np.ndarray | None

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        return (int(i) for i in self.indices)

    def __getitem__(self, key: int) -> float:
        position = int(np.searchsorted(self.indices, key))
        if position >= len(self.indices) or self.indices[position] != key:
            raise KeyError(key)
        return 1.0 if self.weights is None else float(self.weights[position])


class CompactAdjacency(Mapping[int, Mapping[int, float]]):
    """CSR adjacency preserving the public ``{row: {column: weight}}`` interface.

    Binary graphs need only offsets and neighbour indices. Arrays are detached
    and made read-only on admission; each row is a small view, not a dictionary
    with one Python allocation per edge. Indices within every row are sorted.
    """

    __slots__ = ("_indices", "_offsets", "_values")

    def __init__(
        self, offsets: np.ndarray, indices: np.ndarray, values: np.ndarray | None = None
    ) -> None:
        offsets = np.asarray(offsets)
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("sparse adjacency offsets must be integers")
        offsets = offsets.astype(np.int64, copy=False)
        indices = np.asarray(indices)
        if (
            offsets.ndim != 1
            or not len(offsets)
            or offsets[0] != 0
            or np.any(offsets[1:] < offsets[:-1])
            or offsets[-1] != len(indices)
            or indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise ValueError("invalid sparse adjacency offsets or indices")
        n = len(offsets) - 1
        if len(indices) and (indices.min() < 0 or indices.max() >= n):
            raise ValueError("sparse adjacency contains an out-of-range neighbour")
        if values is not None:
            values = np.asarray(values, dtype=float)
            if values.shape != indices.shape or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("sparse adjacency weights must be finite and non-negative")
        if len(indices) > 1:
            unordered = indices[1:] <= indices[:-1]
            boundaries = offsets[1:-1]
            boundaries = boundaries[(boundaries > 0) & (boundaries < len(indices))]
            unordered[boundaries - 1] = False
            if unordered.any():
                raise ValueError("sparse adjacency rows must contain sorted unique neighbours")
        self._offsets = offsets.copy()
        self._indices = indices.astype(np.int32 if n < 2**31 else np.int64, copy=True)
        self._values = None if values is None else values.copy()
        for array in (self._offsets, self._indices, self._values):
            if array is not None:
                array.flags.writeable = False

    @classmethod
    def from_rows(cls, rows: Sequence[np.ndarray]) -> CompactAdjacency:
        counts = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        indices = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
        return cls(offsets, indices)

    @classmethod
    def from_edges(
        cls, n: int, rows: np.ndarray, columns: np.ndarray, values: np.ndarray | None = None
    ) -> CompactAdjacency:
        """Build CSR from admitted edge arrays; duplicate weights are summed safely.

        Callers admit input cardinality before allocating these arrays. Binary
        duplicates collapse to one edge. Weighted duplicates use a compensated
        sum, refusing an aggregate that cannot be represented as a finite float.
        """
        if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 0:
            raise ValueError("sparse adjacency size must be a non-negative integer")
        rows, columns = np.asarray(rows), np.asarray(columns)
        if (
            rows.ndim != 1
            or rows.shape != columns.shape
            or not np.issubdtype(rows.dtype, np.integer)
            or not np.issubdtype(columns.dtype, np.integer)
        ):
            raise ValueError("sparse adjacency endpoints must be equal-length integer arrays")
        if len(rows) and (
            rows.min() < 0 or columns.min() < 0 or rows.max() >= n or columns.max() >= n
        ):
            raise ValueError("sparse adjacency contains an out-of-range endpoint")
        if values is not None:
            values = np.asarray(values, dtype=float)
            if values.shape != rows.shape or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("sparse adjacency weights must be finite and non-negative")
        order = np.lexsort((columns, rows))
        rows, columns = rows[order], columns[order]
        first = np.ones(len(rows), dtype=bool)
        first[1:] = (rows[1:] != rows[:-1]) | (columns[1:] != columns[:-1])
        starts = np.flatnonzero(first)
        if values is not None:
            values = values[order]
            combined = values[starts].copy()
            ends = np.append(starts[1:], len(rows))
            for group in np.flatnonzero(ends - starts > 1):
                try:
                    combined[group] = math.fsum(values[starts[group] : ends[group]])
                except OverflowError:
                    raise ValueError("duplicate edge weights overflow a finite sum") from None
            values = combined
        rows, columns = rows[first], columns[first]
        counts = np.bincount(rows.astype(np.int64, copy=False), minlength=n)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        return cls(offsets, columns, values)

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __getitem__(self, key: int) -> Mapping[int, float]:
        if not isinstance(key, int | np.integer) or not 0 <= key < len(self):
            raise KeyError(key)
        a, b = self._offsets[key : key + 2]
        return _Row(self._indices[a:b], None if self._values is None else self._values[a:b])

    @property
    def pair_count(self) -> int:
        return len(self._indices)

    @property
    def nbytes(self) -> int:
        return (
            self._offsets.nbytes
            + self._indices.nbytes
            + (0 if self._values is None else self._values.nbytes)
        )

    def row_counts(self) -> np.ndarray:
        return np.diff(self._offsets)

    def edge_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Arrays for numerical consumers, with no intermediate Python edge lists."""
        rows = np.repeat(np.arange(len(self), dtype=np.int64), self.row_counts())
        values = np.ones(self.pair_count) if self._values is None else self._values
        return rows, self._indices, values

    def standardized(self) -> CompactAdjacency:
        counts = self.row_counts()
        occupied = counts > 0
        if self._values is None:
            return CompactAdjacency(
                self._offsets, self._indices, np.repeat(1.0 / counts[occupied], counts[occupied])
            )
        scaled = np.empty(self.pair_count)
        retained = np.ones(self.pair_count, dtype=bool)
        for row in np.flatnonzero(occupied):
            a, b = self._offsets[row : row + 2]
            normalized = normalized_weights(self._values[a:b])
            if normalized is None:
                retained[a:b] = False
                counts[row] = 0
            else:
                scaled[a:b] = normalized
        # Match the Mapping implementation: zero-total rows become islands,
        # while explicit zero edges in a positive-total row remain present.
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        return CompactAdjacency(offsets, self._indices[retained], scaled[retained])

    def subset(self, keep: Sequence[int]) -> CompactAdjacency:
        """Induced graph in the given unique vertex order, without Python edge objects."""
        selected = checked_subset(keep, len(self))
        inverse = np.full(len(self), -1, dtype=np.int64)
        inverse[selected] = np.arange(len(selected))
        rows, columns, values = self.edge_arrays()
        mapped_rows, mapped_columns = inverse[rows], inverse[columns]
        retained = (mapped_rows >= 0) & (mapped_columns >= 0)
        return CompactAdjacency.from_edges(
            len(selected),
            mapped_rows[retained],
            mapped_columns[retained],
            None if self._values is None else values[retained],
        )


def checked_subset(keep: Sequence[int], n: int) -> np.ndarray:
    """Validate an induced graph selection before integer conversion or gathering."""
    items = list(keep)
    if any(
        isinstance(i, bool) or not isinstance(i, int | np.integer) or not 0 <= i < n for i in items
    ):
        raise ValueError("subset indices must be integers within the graph")
    selected = np.asarray(items, dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("subset indices must be unique")
    return selected


def normalized_weights(values: np.ndarray) -> np.ndarray | None:
    """Preserve small weights: stable original-unit sum, exact overflow fallback.

    Scaling every row first can double-round a representable subnormal weight
    to zero. Finite totals therefore use direct division. Only sums outside the
    float range require the slower exact fallback; binary rows never call this.
    """
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite before standardization")
    try:
        total = math.fsum(values)
    except OverflowError:
        exact = sum((Fraction(float(value)) for value in values), Fraction())
        return (
            np.array([float(Fraction(float(value)) / exact) for value in values]) if exact else None
        )
    return values / total if total else None
