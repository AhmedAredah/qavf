"""Spatial neighbour weights for :func:`qavf.inference.morans_i` and its local form.

:class:`SpatialWeights` holds the sparse form the inference functions accept,
``{i: {j: w}}`` over locations ``0..n-1`` (``ids`` names them), and reports
*islands*, locations without a neighbour, explicitly. Builders:

* :func:`contiguity`: polygons that share a boundary, from geometry in DuckDB.
  ``queen`` neighbours share at least one point (a vertex or an edge), ``rook``
  neighbours a boundary of positive length (an edge). Overlapping polygons are
  neighbours under both rules. A bounding-box prefilter limits the exact tests.
* :func:`grid_contiguity`: the same rules for grid cells, from their integer
  positions alone (no extension): queen cells differ by at most one in each
  direction, rook cells by one in exactly one direction.
* :func:`nearest`: the ``k`` nearest neighbours of each point or centroid
  (ties at the ``k``-th distance broken by index), directed.
* :func:`distance_band`: every pair within a distance (inclusive), symmetric.
* :func:`from_table`: explicit ``from``/``to``/``weight`` pairs.

Distances are haversine metres for a geographic CRS and Euclidean CRS units
otherwise (:mod:`qavf.spatial.distance`). Contiguity and band weights are 1
(binary); :meth:`SpatialWeights.standardized` divides each row by its total.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from qavf.engine.registry import RuleParams
from qavf.rules.helpers import plural
from qavf.spatial.distance import distance
from qavf.spatial.extension import fetch
from qavf.spatial.hierarchy import Crosswalk

Rule = Literal["queen", "rook"]
_CHUNK = 2_000_000
"""Pairwise distances computed per batch."""


@dataclass(frozen=True)
class SpatialWeights:
    """Sparse neighbour weights ``{i: {j: w}}`` over the locations ``ids``."""

    ids: pl.Series
    """Location ids; index ``i`` of the weights is ``ids[i]``."""
    neighbours: Mapping[int, Mapping[int, float]]
    """Every location's neighbours and weights (islands map to ``{}``)."""
    method: str
    row_standardized: bool = False

    @property
    def n(self) -> int:
        return self.ids.len()

    @property
    def islands(self) -> list[int]:
        """Locations without a neighbour."""
        return [i for i in range(self.n) if not self.neighbours.get(i)]

    @property
    def pairs(self) -> int:
        """Directed neighbour pairs."""
        return sum(len(row) for row in self.neighbours.values())

    def counts(self) -> list[int]:
        """Neighbours of each location."""
        return [len(self.neighbours.get(i, {})) for i in range(self.n)]

    def standardized(self) -> SpatialWeights:
        """Each location's weights divided by their total (islands keep none)."""
        rows = {}
        for i in range(self.n):
            row = self.neighbours.get(i, {})
            total = sum(row.values())
            rows[i] = {j: w / total for j, w in row.items()} if total else {}
        return replace(self, neighbours=rows, row_standardized=True)

    def subset(self, keep: Sequence[int]) -> SpatialWeights:
        """The weights among the locations ``keep``, reindexed in that order.

        Neighbours outside ``keep`` are dropped (row standardization, if wanted,
        is applied again afterwards).
        """
        index = {old: new for new, old in enumerate(keep)}
        rows = {
            index[i]: {index[j]: w for j, w in self.neighbours.get(i, {}).items() if j in index}
            for i in keep
        }
        weights = SpatialWeights(self.ids.gather(list(keep)), rows, self.method)
        return weights.standardized() if self.row_standardized else weights

    def describe(self) -> str:
        islands = len(self.islands)
        text = f"{self.method} weights, {plural(self.pairs, 'neighbour pair')}"
        if self.row_standardized:
            text += ", row-standardized"
        return text + (f", {plural(islands, 'island')}" if islands else "")


def _symmetric(ids: pl.Series, pairs: pl.DataFrame, method: str) -> SpatialWeights:
    """Binary weights from undirected ``i``/``j`` index pairs."""
    rows: dict[int, dict[int, float]] = {i: {} for i in range(ids.len())}
    for i, j in pairs.select("i", "j").iter_rows():
        if i != j:
            rows[i][j] = rows[j][i] = 1.0
    return SpatialWeights(ids, {i: dict(sorted(row.items())) for i, row in rows.items()}, method)


def contiguity(con: Any, table: str, ids: pl.Series, rule: Rule = "queen") -> SpatialWeights:
    """Queen or rook contiguity of the polygons of a loaded layer (``zone_order`` = index)."""
    shared = (
        "ST_Intersects(a.geom, b.geom)"
        if rule == "queen"
        else "ST_Length(ST_Intersection(ST_Boundary(a.geom), ST_Boundary(b.geom))) > 0 "
        "OR ST_Area(ST_Intersection(a.geom, b.geom)) > 0"
    )
    pairs = fetch(
        con,
        f'SELECT a.zone_order AS i, b.zone_order AS j FROM "{table}" a JOIN "{table}" b '
        "ON a.zone_order < b.zone_order AND a.xmin <= b.xmax AND b.xmin <= a.xmax "
        f"AND a.ymin <= b.ymax AND b.ymin <= a.ymax WHERE ST_Intersects(a.geom, b.geom) "
        f"AND ({shared})",
    ).with_columns(pl.col("i").cast(pl.Int64), pl.col("j").cast(pl.Int64))
    return _symmetric(ids, pairs, rule)


def grid_contiguity(ids: pl.Series, ix: pl.Series, iy: pl.Series, rule: Rule = "queen"):
    """Queen or rook contiguity of grid cells at integer positions ``ix``, ``iy``."""
    cells = pl.DataFrame({"i": pl.int_range(0, ids.len(), eager=True), "ix": ix, "iy": iy})
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    if rule == "rook":
        offsets = [(dx, dy) for dx, dy in offsets if abs(dx) + abs(dy) == 1]
    shifts = pl.DataFrame({"dx": [o[0] for o in offsets], "dy": [o[1] for o in offsets]})
    moved = cells.join(shifts, how="cross").select(
        "i", (pl.col("ix") + pl.col("dx")).alias("ix"), (pl.col("iy") + pl.col("dy")).alias("iy")
    )
    pairs = moved.join(cells.rename({"i": "j"}), on=["ix", "iy"]).filter(pl.col("i") < pl.col("j"))
    return _symmetric(ids, pairs, f"grid {rule}")


def _distances(x: np.ndarray, y: np.ndarray, rows: slice, geographic: bool) -> np.ndarray:
    """Distances from the points in ``rows`` to every point (a ``rows x n`` block)."""
    return distance(x[rows, None], y[rows, None], x[None, :], y[None, :], geographic=geographic)


def _blocks(n: int) -> list[slice]:
    size = max(1, _CHUNK // max(n, 1))
    return [slice(start, min(start + size, n)) for start in range(0, n, size)]


def _coordinates(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xs.shape != ys.shape or xs.ndim != 1:
        raise ValueError("coordinates must be two 1-D arrays of equal length")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise ValueError("coordinates must be finite")
    return xs, ys


def nearest(ids: pl.Series, x: Any, y: Any, k: int, *, geographic: bool) -> SpatialWeights:
    """The ``k`` nearest other locations of each one (binary, directed).

    Raises:
        ValueError: for ``k`` outside ``1..n-1`` or non-finite coordinates.
    """
    xs, ys = _coordinates(x, y)
    n = len(xs)
    if not 1 <= k < n:
        raise ValueError(f"k must lie in 1..{n - 1} for {plural(n, 'location')}, got {k}")
    rows: dict[int, dict[int, float]] = {}
    for block in _blocks(n):
        d = _distances(xs, ys, block, geographic)
        for offset, i in enumerate(range(block.start, block.stop)):
            d[offset, i] = np.inf
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        for offset, i in enumerate(range(block.start, block.stop)):
            rows[i] = {int(j): 1.0 for j in sorted(order[offset])}
    return SpatialWeights(ids, rows, f"{k} nearest neighbours")


def distance_band(
    ids: pl.Series, x: Any, y: Any, threshold: float, *, geographic: bool
) -> SpatialWeights:
    """Every pair of locations at most ``threshold`` apart (binary, symmetric)."""
    xs, ys = _coordinates(x, y)
    rows: dict[int, dict[int, float]] = {}
    for block in _blocks(len(xs)):
        d = _distances(xs, ys, block, geographic)
        for offset, i in enumerate(range(block.start, block.stop)):
            near = np.flatnonzero(d[offset] <= threshold)
            rows[i] = {int(j): 1.0 for j in near if j != i}
    return SpatialWeights(ids, rows, f"distance band {threshold:g}")


def from_table(ids: pl.Series, table: pl.DataFrame) -> SpatialWeights:
    """Weights from ``from``/``to``/``weight`` pairs of location ids (directed).

    Raises:
        ValueError: for ids that are not locations, or non-positive weights.
    """
    index = pl.DataFrame({"id": ids, "i": pl.int_range(0, ids.len(), eager=True)})
    try:
        frame = table.with_columns(pl.col("from", "to").cast(ids.dtype, strict=True))
    except Exception:
        raise ValueError(f"weight table ids cannot all be represented as {ids.dtype}") from None
    joined = frame.join(index.rename({"id": "from"}), on="from", how="left").join(
        index.rename({"id": "to", "i": "j"}), on="to", how="left"
    )
    if unknown := joined.filter(pl.col("i").is_null() | pl.col("j").is_null()).height:
        raise ValueError(f"{plural(unknown, 'weight pair')} name locations that are not zones")
    if joined.filter(~(pl.col("weight") > 0) | ~pl.col("weight").is_finite()).height:
        raise ValueError("weights must be finite and positive")
    rows: dict[int, dict[int, float]] = {i: {} for i in range(ids.len())}
    for i, j, w in joined.select("i", "j", "weight").iter_rows():
        if i != j:
            rows[i][j] = rows[i].get(j, 0.0) + float(w)
    return SpatialWeights(ids, rows, "weight table")


class WeightsSpec(RuleParams):
    """How zones' neighbours are found: contiguity, nearest neighbours, a band or a table."""

    method: Literal["queen", "rook", "knn", "distance_band", "table"] = "queen"
    """``queen``/``rook`` contiguity (zone polygons, or grid cells), ``knn`` and
    ``distance_band`` (zone centroids or cell centres), or a ``table`` of pairs."""
    k: int = Field(default=4, ge=1)
    """Neighbours per zone for ``knn``."""
    distance: float | None = Field(default=None, gt=0)
    """Band for ``distance_band``: metres for a geographic CRS, else CRS units."""
    row_standardize: bool = True
    """Divide each zone's weights by their total."""
    table: Crosswalk | None = None
    """For ``table``: ``from``/``to`` (``weight``) pairs of zone ids, as a crosswalk is
    written (``{dataset: name}``, ``{path: file}`` or ``{pairs: [[a, b], ...]}``); the
    weights need not sum to 1."""

    @model_validator(mode="after")
    def _complete(self) -> WeightsSpec:
        if self.method == "distance_band" and self.distance is None:
            raise ValueError("distance_band weights need a 'distance'")
        if (self.method == "table") != (self.table is not None):
            raise ValueError("'table' is required for, and only used by, method: table")
        return self
