"""Stage a coherent database snapshot in owned spill files, one bounded batch at a time.

A snapshot reads every member inside one transaction. Holding each member's
complete result in memory until the transaction commits would bound the group by
memory; instead each member streams to its own Arrow IPC file under the run's
spill budget, and only the file handles are published, together, after commit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from qavf.duckdb_results import unique_names
from qavf.engine.cleanup import attempt_cleanup, raise_cleanup_errors

if TYPE_CHECKING:
    from qavf.bounded_batches import BoundedBatches
    from qavf.engine.spill import SpillFile
    from qavf.sources.base import SourceError


class MemberPreparationError(Exception):
    """A member's column names could not be resolved; the whole snapshot fails."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


@dataclass(frozen=True, slots=True)
class StagedMember:
    """One member's complete, owned rows, written but not yet published."""

    file: SpillFile
    schema: dict[str, pl.DataType]
    """The member's Polars schema after canonical and explicit renames."""
    rows: int
    size: int
    """The decoded Arrow bytes of the rows: the cache's estimate when they are read back."""


class SnapshotStage:
    """One snapshot attempt: its staged members, failed members and start time.

    The stage owns every file it creates until :meth:`detach` hands the members to
    publication; :meth:`close` deletes whatever it still owns, so a failed, retried
    or cancelled attempt leaves no files behind.
    """

    def __init__(
        self,
        new_file: Callable[[], SpillFile],
        prepare: Callable[[str, pl.DataFrame], pl.DataFrame],
        started_at: datetime,
    ) -> None:
        self._new_file = new_file
        self._prepare = prepare
        self.started_at = started_at
        self.members: dict[str, StagedMember] = {}
        self.failures: dict[str, SourceError] = {}
        self._files: list[SpillFile] = []

    def stage_batches(self, name: str, batches: BoundedBatches) -> StagedMember:
        """Write ``batches`` to one owned IPC file, renamed as the member's frame would be."""
        import pyarrow as pa

        source = batches.schema
        names = unique_names(source.names)
        empty = pl.from_arrow(source.empty_table().rename_columns(names))
        assert isinstance(empty, pl.DataFrame)
        prepared = self._prepared(name, empty)
        target = pa.schema(
            [
                field.with_name(column)
                for field, column in zip(source, prepared.columns, strict=True)
            ],
            metadata=source.metadata,
        )
        owned = self._own()
        size = 0
        try:
            with owned.writer() as sink, pa.ipc.new_file(sink, target) as writer:
                for batch in batches:
                    writer.write_batch(pa.RecordBatch.from_arrays(batch.columns, schema=target))
                    size += batch.nbytes
        except BaseException:
            # A failed member may be skipped while the others continue: free its bytes now.
            self._discard(owned)
            raise
        return self._staged(name, StagedMember(owned, dict(prepared.schema), batches.rows, size))

    def stage_frame(self, name: str, frame: pl.DataFrame) -> StagedMember:
        """Write a member that its source can only read whole (a plugin's ``read``)."""
        prepared = self._prepared(name, frame)
        owned = self._own()
        try:
            with owned.writer() as sink:
                prepared.write_ipc(sink, compression="uncompressed")
        except BaseException:
            self._discard(owned)
            raise
        member = StagedMember(
            owned, dict(prepared.schema), prepared.height, int(prepared.estimated_size())
        )
        return self._staged(name, member)

    def spills(self) -> Mapping[str, tuple[SpillFile, int]]:
        """Each staged member's file and decoded size, for publication."""
        return {name: (member.file, member.size) for name, member in self.members.items()}

    def detach(self) -> None:
        """Hand every staged file to its publisher; closing no longer deletes them."""
        self._files.clear()

    def close(self) -> None:
        """Delete every file this attempt still owns."""
        files, self._files = self._files, []
        self.members.clear()
        raise_cleanup_errors(
            "snapshot staging files could not be released",
            attempt_cleanup(owned.close for owned in files),
        )

    def _prepared(self, name: str, frame: pl.DataFrame) -> pl.DataFrame:
        try:
            return self._prepare(name, frame)
        except Exception as exc:
            raise MemberPreparationError(exc) from exc

    def _own(self) -> SpillFile:
        owned = self._new_file()
        self._files.append(owned)
        return owned

    def _discard(self, owned: SpillFile) -> None:
        self._files.remove(owned)
        owned.close()

    def _staged(self, name: str, member: StagedMember) -> StagedMember:
        self.members[name] = member
        return member


__all__ = ["MemberPreparationError", "SnapshotStage", "StagedMember"]
