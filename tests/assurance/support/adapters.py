"""Represent identical typed rows through each public input transport."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa

from qavf import Query

BACKENDS = ("eager", "lazy", "arrow", "parquet", "query")
PARTITIONED_BACKENDS = ("arrow", "parquet", "query", "partitioned")


def _adapt(frame: pl.DataFrame, backend: str, path: Path, partition_size: int) -> object:
    if backend == "eager":
        return frame
    if backend == "lazy":
        return frame.lazy()
    if backend == "arrow":
        # Preserve physical batch boundaries and the schema of empty tables.
        table = frame.to_arrow()
        batches = table.to_batches(max_chunksize=partition_size)
        return pa.Table.from_batches(batches, schema=table.schema)
    if backend == "partitioned":
        path.mkdir()
        for index, start in enumerate(range(0, max(frame.height, 1), partition_size)):
            frame.slice(start, partition_size).write_parquet(path / f"part-{index:04d}.parquet")
        return pl.scan_parquet(path / "*.parquet")
    if backend in ("parquet", "query"):
        frame.write_parquet(path, row_group_size=partition_size)
        if backend == "parquet":
            return path
        return Query("SELECT * FROM read_parquet($source)", parameters={"source": str(path)})
    raise ValueError(f"Unknown assurance input backend: {backend}")


@contextmanager
def adapted_frames(
    frames: Mapping[str, pl.DataFrame],
    backend: str | Mapping[str, str],
    *,
    partition_size: int = 3,
) -> Iterator[dict[str, object]]:
    """Keep temporary sources alive throughout validation and remove them afterwards.

    A mapping chooses separate transports per dataset, exercising mixed execution.
    Partition size controls Arrow batches, Parquet row groups and partition files.
    """
    if partition_size < 1:
        raise ValueError("partition_size must be positive")
    with TemporaryDirectory(prefix="qavf-assurance-") as directory:
        yield {
            name: _adapt(
                frame,
                backend[name] if isinstance(backend, Mapping) else backend,
                Path(directory) / f"{index}.parquet",
                partition_size,
            )
            for index, (name, frame) in enumerate(frames.items())
        }
