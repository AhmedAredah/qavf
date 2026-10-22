"""Regular grid cells: a spatial bucketing with deterministic labels and an expected grid.

The spatial counterpart of :mod:`qavf.temporal.buckets`. A :class:`Grid` cuts
the plane into squares of ``size`` CRS units (metres in a projected CRS,
degrees in a geographic one) from ``origin``. A point ``(x, y)`` lies in the
cell at *position* ``(floor((x - x0) / size), floor((y - y0) / size))``, whose
*label* is the text ``"<ix>_<iy>"`` (``"-3_12"``). Cells are half-open: a point
on a cell's west or south edge belongs to it. Labels are computed natively by
Polars expressions in binary64, so a point within one rounding of an edge is
placed by binary64 division; positions and labels are exact integers.

:func:`grid_cells` is the expected grid over an ``extent``, in position order
(rows of ``iy``, then ``ix``): coverage gaps are the expected cells a side
lacks. Points are transformed to the grid's ``crs`` first when it differs from
theirs (that needs DuckDB's spatial extension; a grid in the points' own CRS
does not). Hexagonal (H3) cells would need a community extension and are not
offered.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import polars as pl
from pydantic import Field, model_validator

from qavf.engine.registry import RuleParams
from qavf.rules.helpers import fresh_internal
from qavf.spatial.assign import Assigned, coordinate_keys
from qavf.spatial.crs import Crs, same_crs, transform_sql
from qavf.spatial.extension import fetch, register


class Grid(RuleParams):
    """Square cells of ``size`` CRS units from ``origin``, optionally over an ``extent``."""

    size: float = Field(gt=0)
    """Cell width in CRS units (metres for a projected CRS, degrees for a geographic one)."""
    crs: Crs | None = None
    """CRS of the grid; points are transformed to it. The points' own CRS when omitted."""
    origin: tuple[float, float] = (0.0, 0.0)
    """``(x, y)`` of a cell corner."""
    extent: tuple[float, float, float, float] | None = None
    """``(xmin, ymin, xmax, ymax)``: the cells expected to be present (all cells it overlaps)."""
    max_cells: int = Field(default=1_000_000, ge=1)
    """Largest expected grid; a larger one is not assessed."""

    @model_validator(mode="after")
    def _ordered(self) -> Grid:
        if self.extent is not None:
            xmin, ymin, xmax, ymax = self.extent
            if not (xmin < xmax and ymin < ymax):
                raise ValueError("a grid extent needs xmin < xmax and ymin < ymax")
        return self

    def describe(self) -> str:
        return f"{self.size:g}-unit grid" + (f" in {self.crs}" if self.crs else "")


def _index(value: pl.Expr, origin: float, size: float) -> pl.Expr:
    return ((value - origin) / size).floor().cast(pl.Int64, strict=False)


def cell_positions(x: pl.Expr, y: pl.Expr, grid: Grid) -> tuple[pl.Expr, pl.Expr]:
    """``(ix, iy)`` of each point; null for missing or non-finite coordinates."""
    finite = x.is_finite() & y.is_finite()
    return tuple(  # type: ignore[return-value]
        pl.when(finite).then(_index(value, origin, grid.size))
        for value, origin in ((x, grid.origin[0]), (y, grid.origin[1]))
    )


def label_of(ix: pl.Expr, iy: pl.Expr) -> pl.Expr:
    """The ``"<ix>_<iy>"`` label of integer positions (null when either is)."""
    return pl.format("{}_{}", ix, iy)


def cell_labels(x: pl.Expr, y: pl.Expr, grid: Grid) -> pl.Expr:
    """The cell label of each point (null for missing or non-finite coordinates)."""
    return label_of(*cell_positions(x, y, grid))


def label_positions(labels: pl.Series) -> pl.DataFrame:
    """``ix`` and ``iy`` of cell labels (the inverse of :func:`label_of`).

    Raises:
        ValueError: for text that is not a cell label.
    """
    parts = labels.cast(pl.String).str.split_exact("_", 1)
    frame = pl.DataFrame(
        {
            "ix": parts.struct.field("field_0").cast(pl.Int64, strict=False),
            "iy": parts.struct.field("field_1").cast(pl.Int64, strict=False),
        }
    )
    bad = frame.select((pl.col("ix").is_null() | pl.col("iy").is_null()).sum()).item()
    if bad - labels.null_count():
        raise ValueError(f"{bad:,} value(s) are not grid cell labels ('<ix>_<iy>')")
    return frame


def cell_centres(labels: pl.Series, grid: Grid) -> tuple[Any, Any]:
    """Centre coordinates of the cells (NumPy arrays, in the grid's CRS)."""
    positions = label_positions(labels)
    size = grid.size
    x = grid.origin[0] + (positions["ix"].to_numpy() + 0.5) * size
    y = grid.origin[1] + (positions["iy"].to_numpy() + 0.5) * size
    return x, y


def grid_cells(grid: Grid) -> pl.Series:
    """Every cell overlapping the grid's ``extent``, in position order (``iy``, then ``ix``).

    Raises:
        ValueError: without an extent, or for more than ``max_cells`` cells.
    """
    if grid.extent is None:
        raise ValueError("the grid has no 'extent' to expect cells in")
    xmin, ymin, xmax, ymax = grid.extent
    (ox, oy), size = grid.origin, grid.size
    first = (math.floor((xmin - ox) / size), math.floor((ymin - oy) / size))
    last = (math.ceil((xmax - ox) / size) - 1, math.ceil((ymax - oy) / size) - 1)
    columns, rows = last[0] - first[0] + 1, last[1] - first[1] + 1
    if columns * rows > grid.max_cells:
        raise ValueError(
            f"the expected grid has {columns * rows:,} cells, more than max_cells "
            f"{grid.max_cells:,}"
        )
    ix = pl.int_range(first[0], last[0] + 1, dtype=pl.Int64, eager=True).alias("ix")
    iy = pl.int_range(first[1], last[1] + 1, dtype=pl.Int64, eager=True).alias("iy")
    cells = iy.to_frame().join(ix.to_frame(), how="cross")
    return cells.select(label_of(pl.col("ix"), pl.col("iy")).alias("cell"))["cell"]


def assign_cells(
    plan: pl.LazyFrame,
    grid: Grid,
    *,
    x: str,
    y: str,
    crs: str,
    column: str,
    collect: Callable[[pl.LazyFrame], pl.DataFrame],
    connection: Callable[[], Any] | None = None,
) -> Assigned:
    """Add ``column``, each row's cell label; rows without usable coordinates get none.

    In the points' own CRS the labels are one native expression. Otherwise the
    distinct points are transformed in DuckDB (``connection()`` opens a spatial
    connection as a context manager) and their labels joined back.

    Raises:
        ValueError: for a ``column`` the plan already has.
        SpatialUnavailable: when a transform is needed without the extension.
    """
    schema = plan.collect_schema()
    if column in schema:
        raise ValueError(f"the zone column {column!r} already exists")
    kx, ky = coordinate_keys(x, y)
    if grid.crs is None or same_crs(crs, grid.crs):
        labelled = plan.with_columns(cell_labels(kx, ky, grid).alias(column))
    else:
        assert connection is not None
        taken = [*schema, column]
        names = [fresh_internal(f"cell_{role}", taken) for role in ("x", "y")]
        distinct = collect(plan.select(kx.alias(names[0]), ky.alias(names[1])).unique())
        points = distinct.select(pl.col(names[0]).alias("px"), pl.col(names[1]).alias("py"))
        with connection() as con:
            register(con, "qavf_cells", points.with_row_index("pid"))
            point = transform_sql("ST_Point(px, py)", crs, grid.crs)
            moved = fetch(
                con,
                f"SELECT pid, ST_X(p) AS tx, ST_Y(p) AS ty FROM (SELECT pid, {point} AS p FROM "
                "qavf_cells WHERE px IS NOT NULL AND py IS NOT NULL AND isfinite(px) AND "
                "isfinite(py)) ORDER BY pid",
            )
            con.unregister("qavf_cells")
        cells = moved.select(
            pl.col("pid").cast(pl.UInt32),
            cell_labels(pl.col("tx"), pl.col("ty"), grid).alias(column),
        )
        mapping = distinct.with_row_index("pid").join(cells, on="pid").drop("pid")
        labelled = plan.with_columns(kx.alias(names[0]), ky.alias(names[1])).join(
            mapping.lazy(), on=names, how="left"
        )
        labelled = labelled.drop(names)
    counts = collect(labelled.select(pl.len(), pl.col(column).is_null().sum())).row(0)
    return Assigned(labelled, int(counts[0]), int(counts[1]), 0, 0)
