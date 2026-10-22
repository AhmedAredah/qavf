"""DuckDB's ``spatial`` extension: one loading path, failing closed when it is absent.

Everything that needs geometry (point-in-polygon, zone layers read from GeoJSON
or shapefiles, polygon contiguity, CRS transforms, polygon areas) runs in a
DuckDB connection with the extension loaded. :func:`spatial_connection` opens
one under the run's native resource policy (threads, memory, spill); when the
extension cannot be loaded it raises :class:`SpatialUnavailable`, which every
caller reports as *not assessed*, never as a pass. Attribute zones, grid cells
in the data's own CRS, crosswalks, and weights from grid labels or tables need
no extension.

``LOAD spatial`` is tried first; when the extension is not installed, DuckDB's
own ``INSTALL spatial`` (the signed default repository) is tried once, exactly
as DuckDB's autoloading would. The Docker image and ``scripts/verify_release.py
extensions`` pre-install it, so runs there need no network.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
import pyarrow as pa

from qavf.redaction import redact_text

if TYPE_CHECKING:
    from qavf.engine.context import DataContext


class SpatialUnavailable(RuntimeError):
    """DuckDB's spatial extension cannot be loaded; geometry checks are not assessed."""


def load_spatial(con: Any) -> None:
    """Load the spatial extension into ``con``, installing it once if needed.

    Raises:
        SpatialUnavailable: with DuckDB's reason when neither works.
    """
    try:
        con.execute("LOAD spatial")
        return
    except duckdb.Error as first:
        reason = str(first)
    try:
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
    except duckdb.Error as exc:
        raise SpatialUnavailable(
            "DuckDB's spatial extension could not be loaded "
            f"({redact_text(reason.splitlines()[0])}; installing it failed: "
            f"{redact_text(str(exc).splitlines()[0])}); install it with "
            "`INSTALL spatial` or `scripts/verify_release.py extensions`"
        ) from None


@contextmanager
def spatial_connection(ctx: DataContext | None = None) -> Iterator[Any]:
    """A DuckDB connection with the spatial extension loaded, closed on exit.

    With ``ctx`` the connection follows the run's native resource policy.

    Raises:
        SpatialUnavailable: when the extension cannot be loaded.
    """
    if ctx is not None:
        from qavf.sources.native import connect

        con: Any = connect(ctx.env)
    else:
        con = duckdb.connect()
    try:
        load_spatial(con)
        yield con
    finally:
        con.close()


def register(con: Any, name: str, frame: pl.DataFrame) -> None:
    """Bind ``frame`` as a relation; columns are exported to Arrow one by one first."""
    table = pa.Table.from_arrays([column.to_arrow() for column in frame], names=frame.columns)
    con.register(name, table)


def fetch(con: Any, sql: str, parameters: list[Any] | None = None) -> pl.DataFrame:
    """Run ``sql`` and return its result as a Polars frame."""
    return con.execute(sql, parameters or []).pl()
