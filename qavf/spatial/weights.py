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

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from qavf._adjacency import CompactAdjacency, checked_subset, normalized_weights
from qavf.engine.registry import RuleParams
from qavf.keys import cast_key
from qavf.rules.helpers import plural
from qavf.sources.sql import quote_identifier
from qavf.spatial._neighbours import (
    DEFAULT_MAX_EDGES,
    DEFAULT_MAX_WORK,
    _Admission,
    band_graph,
    nearest_graph,
)
from qavf.spatial.extension import register
from qavf.spatial.hierarchy import Crosswalk

Rule = Literal["queen", "rook"]


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
        if isinstance(self.neighbours, CompactAdjacency):
            return np.flatnonzero(self.neighbours.row_counts() == 0).tolist()
        return [i for i in range(self.n) if not self.neighbours.get(i)]

    @property
    def pairs(self) -> int:
        """Directed neighbour pairs."""
        if isinstance(self.neighbours, CompactAdjacency):
            return self.neighbours.pair_count
        return sum(len(row) for row in self.neighbours.values())

    def counts(self) -> list[int]:
        """Neighbours of each location."""
        if isinstance(self.neighbours, CompactAdjacency):
            return self.neighbours.row_counts().tolist()
        return [len(self.neighbours.get(i, {})) for i in range(self.n)]

    def standardized(self) -> SpatialWeights:
        """Each location's weights divided by their total (islands keep none)."""
        if isinstance(self.neighbours, CompactAdjacency):
            return replace(self, neighbours=self.neighbours.standardized(), row_standardized=True)
        rows = {}
        for i in range(self.n):
            row = self.neighbours.get(i, {})
            scaled = normalized_weights(np.fromiter(row.values(), dtype=float, count=len(row)))
            rows[i] = dict(zip(row, map(float, scaled), strict=True)) if scaled is not None else {}
        return replace(self, neighbours=rows, row_standardized=True)

    def subset(self, keep: Sequence[int]) -> SpatialWeights:
        """The weights among the locations ``keep``, reindexed in that order.

        Neighbours outside ``keep`` are dropped (row standardization, if wanted,
        is applied again afterwards).
        """
        keep = checked_subset(keep, self.n).tolist()
        if isinstance(self.neighbours, CompactAdjacency):
            weights = SpatialWeights(
                self.ids.gather(list(keep)), self.neighbours.subset(keep), self.method
            )
            return weights.standardized() if self.row_standardized else weights
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


def _symmetric(ids: pl.Series, left: np.ndarray, right: np.ndarray, method: str) -> SpatialWeights:
    """Binary compact weights from already-admitted undirected index pairs."""
    return SpatialWeights(
        ids,
        CompactAdjacency.from_edges(
            len(ids), np.concatenate((left, right)), np.concatenate((right, left))
        ),
        method,
    )


def _rule(rule: Rule) -> None:
    if rule not in ("queen", "rook"):
        raise ValueError("contiguity rule must be 'queen' or 'rook'")


@contextmanager
def _selected_zones(con: Any, table: str, keep: np.ndarray | None):
    """Remap a narrow induced graph before any pair work, releasing its binding."""
    quoted = quote_identifier(table)
    if keep is None:
        yield quoted
        return
    name = "__qavf_neighbours_" + uuid4().hex
    register(con, name, pl.DataFrame({"original": keep, "selected": np.arange(len(keep))}))
    try:
        yield (
            f"(SELECT m.selected AS zone_order, z.geom, z.xmin, z.xmax, z.ymin, z.ymax "
            f"FROM {quoted} z JOIN {quote_identifier(name)} m ON z.zone_order=m.original)"
        )
    finally:
        con.unregister(name)


def contiguity(
    con: Any,
    table: str,
    ids: pl.Series,
    rule: Rule = "queen",
    *,
    keep: Sequence[int] | None = None,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_work: int = DEFAULT_MAX_WORK,
    check_cancelled: Callable[[], None] | None = None,
) -> SpatialWeights:
    """Contiguity of a loaded layer, optionally induced by unique ``keep`` indices.

    ``max_work`` admits selected locations plus unordered bounding-box candidate
    pairs before exact geometry tests. It does not count internal join/index
    instructions. ``max_edges`` bounds retained directed pairs. Native query
    working memory follows the connection's policy; Python transfers are batched.
    """
    _rule(rule)
    admission = _Admission(max_work, max_edges)
    if check_cancelled is not None:
        check_cancelled()
    admission.add_work(len(ids) if keep is None else len(keep))
    selected = None if keep is None else checked_subset(keep, len(ids))
    ids = ids if selected is None else ids.gather(selected)
    if len(ids) < 2:
        return _symmetric(ids, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), rule)
    shared = (
        "ST_Intersects(a.geom, b.geom)"
        if rule == "queen"
        else "ST_Length(ST_Intersection(ST_Boundary(a.geom), ST_Boundary(b.geom))) > 0 "
        "OR ST_Area(ST_Intersection(a.geom, b.geom)) > 0"
    )
    with _selected_zones(con, table, selected) as zones:
        bbox = (
            f"FROM {zones} a JOIN {zones} b ON a.zone_order < b.zone_order "
            "AND a.xmin <= b.xmax AND b.xmin <= a.xmax "
            "AND a.ymin <= b.ymax AND b.ymin <= a.ymax"
        )
        remaining = admission.max_work - admission.work
        count = con.execute(
            f"SELECT count(*) FROM (SELECT 1 {bbox} LIMIT {remaining + 1}) candidates"
        ).fetchone()[0]
        admission.add_work(count)
        if check_cancelled is not None:
            check_cancelled()
        # At most one over-budget pair is transferred, never the complete dense
        # result. The native operator's own working state remains connection-owned.
        cap = admission.max_edges // 2
        cursor = con.execute(
            f"SELECT a.zone_order AS i, b.zone_order AS j {bbox} "
            f"WHERE ST_Intersects(a.geom, b.geom) AND ({shared}) LIMIT {cap + 1}"
        )
        reader = (
            cursor.to_arrow_reader(65536)
            if hasattr(cursor, "to_arrow_reader")
            else cursor.fetch_record_batch(65536)
        )
        left, right = [], []
        count = 0
        try:
            for batch in reader:
                if check_cancelled is not None:
                    check_cancelled()
                count += batch.num_rows
                admission.edges(2 * count)
                left.append(np.asarray(batch.column(0)).astype(np.int64, copy=True))
                right.append(np.asarray(batch.column(1)).astype(np.int64, copy=True))
        finally:
            reader.close()
    if check_cancelled is not None:
        check_cancelled()
    return _symmetric(
        ids,
        np.concatenate(left) if left else np.empty(0, dtype=np.int64),
        np.concatenate(right) if right else np.empty(0, dtype=np.int64),
        rule,
    )


def grid_contiguity(
    ids: pl.Series,
    ix: pl.Series,
    iy: pl.Series,
    rule: Rule = "queen",
    *,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_work: int = DEFAULT_MAX_WORK,
):
    """Grid contiguity, admitting locations/offset probes and directed pairs first."""
    _rule(rule)
    admission = _Admission(max_work, max_edges)
    if len(ix) != len(ids) or len(iy) != len(ids):
        raise ValueError("grid coordinates must match the location ids")
    for values in (ix, iy):
        if not values.dtype.is_integer() or values.null_count():
            raise ValueError("grid coordinates must be non-null integers")
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    if rule == "rook":
        offsets = [(dx, dy) for dx, dy in offsets if abs(dx) + abs(dy) == 1]
    admission.add_work(len(ids) * (1 + len(offsets)))
    # Wider integer arithmetic prevents +1/-1 from joining opposite Int64 ends.
    cells = pl.DataFrame(
        {
            "i": pl.int_range(0, ids.len(), eager=True),
            "ix": ix.cast(pl.Int128),
            "iy": iy.cast(pl.Int128),
        }
    )
    if any(cells[c].min() == -(2**127) or cells[c].max() == 2**127 - 1 for c in ("ix", "iy")):
        raise ValueError("grid coordinates need room for a one-cell integer offset")
    shifts = pl.DataFrame({"dx": [o[0] for o in offsets], "dy": [o[1] for o in offsets]})
    counts = cells.group_by("ix", "iy").len()
    candidates = (
        counts.join(shifts, how="cross")
        .select(
            (pl.col("ix") + pl.col("dx")).alias("ix"),
            (pl.col("iy") + pl.col("dy")).alias("iy"),
            pl.col("len").alias("a"),
        )
        .join(counts, on=["ix", "iy"])
    )
    directed = candidates.select((pl.col("a").cast(pl.Int64) * pl.col("len")).sum()).item()
    admission.edges(directed or 0)
    moved = cells.join(shifts, how="cross").select(
        "i", (pl.col("ix") + pl.col("dx")).alias("ix"), (pl.col("iy") + pl.col("dy")).alias("iy")
    )
    pairs = moved.join(cells.rename({"i": "j"}), on=["ix", "iy"]).filter(pl.col("i") < pl.col("j"))
    return _symmetric(ids, pairs["i"].to_numpy(), pairs["j"].to_numpy(), f"grid {rule}")


def nearest(
    ids: pl.Series,
    x: Any,
    y: Any,
    k: int,
    *,
    geographic: bool,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_work: int = DEFAULT_MAX_WORK,
) -> SpatialWeights:
    """The exact ``k`` nearest other locations, with index-stable distance ties.

    Uses an optional SciPy spatial index and refines candidates with QAVF's
    distance formula. Without SciPy, bounded blocks use partial selection.
    ``max_work`` bounds admitted coordinate/distance evaluations, not CPU
    instructions within the index; ``max_edges`` bounds retained directed pairs.
    """
    rows = nearest_graph(
        x, y, len(ids), k, geographic=geographic, max_edges=max_edges, max_work=max_work
    )
    return SpatialWeights(ids, rows, f"{k} nearest neighbours")


def distance_band(
    ids: pl.Series,
    x: Any,
    y: Any,
    threshold: float,
    *,
    geographic: bool,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_work: int = DEFAULT_MAX_WORK,
) -> SpatialWeights:
    """Every pair within the inclusive band, using bounded compact adjacency.

    Work and edge admission have the same meanings as :func:`nearest`.
    """
    rows = band_graph(
        x,
        y,
        len(ids),
        threshold,
        geographic=geographic,
        max_edges=max_edges,
        max_work=max_work,
    )
    return SpatialWeights(ids, rows, f"distance band {threshold:g}")


def from_table(
    ids: pl.Series,
    table: pl.DataFrame,
    *,
    max_edges: int = DEFAULT_MAX_EDGES,
    max_work: int = DEFAULT_MAX_WORK,
) -> SpatialWeights:
    """Weights from ``from``/``to``/``weight`` pairs of location ids (directed).

    Raises:
        ValueError: for unknown/duplicate location ids, non-positive/nonfinite
            weights, duplicate-sum overflow, or work/edge admission refusal.

    Work admits locations plus input rows, including self pairs and duplicates.
    Edges count unique directed nonself pairs after positive weights are summed.
    Native/numeric construction state is bounded by the admitted input rows.
    """
    admission = _Admission(max_work, max_edges)
    admission.add_work(len(ids) + table.height)
    if ids.null_count() or ids.n_unique() != len(ids):
        raise ValueError("weight table location ids must be present and unique")
    index = pl.DataFrame({"id": ids, "i": pl.int_range(0, ids.len(), eager=True)})
    try:
        frame = table.select("from", "to", "weight").with_columns(
            cast_key(table[c], ids.dtype) for c in ("from", "to")
        )
    except ValueError:
        raise ValueError(
            f"weight table ids cannot all be represented as {ids.dtype} without changing identity"
        ) from None
    if not (frame["weight"].dtype.is_numeric() or frame["weight"].dtype == pl.Boolean):
        raise ValueError("weights must be finite and positive")
    frame = frame.with_columns(pl.col("weight").cast(pl.Float64))
    joined = frame.join(index.rename({"id": "from"}), on="from", how="left").join(
        index.rename({"id": "to", "i": "j"}), on="to", how="left"
    )
    if unknown := joined.filter(pl.col("i").is_null() | pl.col("j").is_null()).height:
        raise ValueError(f"{plural(unknown, 'weight pair')} name locations that are not zones")
    if joined.filter(
        pl.col("weight").is_null() | ~(pl.col("weight") > 0) | ~pl.col("weight").is_finite()
    ).height:
        raise ValueError("weights must be finite and positive")
    joined = joined.filter(pl.col("i") != pl.col("j")).select("i", "j", "weight")
    admission.edges(joined.select("i", "j").n_unique())
    rows = CompactAdjacency.from_edges(
        len(ids),
        joined["i"].to_numpy(),
        joined["j"].to_numpy(),
        joined["weight"].cast(pl.Float64).to_numpy(),
    )
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
    max_edges: int = Field(default=DEFAULT_MAX_EDGES, ge=0)
    """Maximum retained unique directed pairs for every graph builder."""
    max_work: int = Field(default=DEFAULT_MAX_WORK, ge=1)
    """Maximum admitted work: locations plus distance candidates, polygon bbox
    pairs, grid offset probes, or explicit table rows, depending on the builder.
    Native operator/index instructions and geometry vertex complexity are not counted."""
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
