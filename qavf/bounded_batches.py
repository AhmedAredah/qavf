"""Read a DuckDB result as a bounded stream of Arrow record batches.

Both the native backend's copy of a borrowed query and the staging of a coherent
database snapshot read their rows this way, so neither holds a complete result in
this process: one batch is alive at a time, between which the run's cancellation
is checked and the row count is admitted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from qavf.duckdb_results import record_batch_reader

if TYPE_CHECKING:
    import pyarrow as pa


class BoundedBatches:
    """One pass over a DuckDB relation as Arrow batches of at most ``batch_rows`` rows.

    ``admit`` receives the cumulative row count after every batch and raises to
    reject the result (a row budget); callers limit their query to ``budget + 1``
    rows, so the source stops reading one row past the budget and the rejection is
    exact. ``check_cancelled`` runs before each batch is read and before it is
    handed on. Closing ends the reader, even part way, so the connection that ran
    the relation is left without an unfinished result; the relation itself is
    only read, never closed.
    """

    BATCH_ROWS = 65_536

    def __init__(
        self,
        relation: Any,
        admit: Callable[[int], None],
        check_cancelled: Callable[[], None],
        batch_rows: int = BATCH_ROWS,
    ) -> None:
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        check_cancelled()
        self._reader = record_batch_reader(relation, batch_rows)
        self._admit = admit
        self._check_cancelled = check_cancelled
        self.rows = 0
        self.batches = 0

    @property
    def schema(self) -> pa.Schema:
        return self._reader.schema

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        while True:
            self._check_cancelled()
            try:
                batch = self._reader.read_next_batch()
            except StopIteration:
                return
            self.rows += batch.num_rows
            self.batches += 1
            self._admit(self.rows)
            self._check_cancelled()
            yield batch

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> BoundedBatches:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["BoundedBatches"]
