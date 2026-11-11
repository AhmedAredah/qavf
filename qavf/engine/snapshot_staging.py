"""Stage a coherent database snapshot in memory while it fits, in owned spill files after.

A snapshot reads every member inside one transaction, and nothing is published
until that transaction commits. A group whose decoded size stays within the
memory allowance keeps its members in memory, so a small snapshot pays no spill
write and read. The first member that would exceed the allowance switches the
stage to spill: the members already buffered are written to owned Arrow IPC
files and every remaining batch streams straight to its member's file, so a
large group is bounded by spill space, not memory. Either way only complete
members are published, together, after commit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from qavf.duckdb_results import unique_names
from qavf.engine.cleanup import attempt_cleanup, raise_cleanup_errors

if TYPE_CHECKING:
    import pyarrow as pa

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
    """One member's complete, owned rows, held in memory or written, not yet published."""

    schema: dict[str, pl.DataType]
    """The member's Polars schema after canonical and explicit renames."""
    rows: int
    size: int
    """The decoded Arrow bytes of the rows: the cache's estimate when they are read back."""
    file: SpillFile | None = None
    frame: pl.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class _HeldBatches:
    """The leading batches of a member that were kept in memory, and whether that was all."""

    batches: list[pa.RecordBatch]
    size: int
    complete: bool


class SnapshotStage:
    """One snapshot attempt: its staged members, failed members and start time.

    Members stay in memory while the attempt's running decoded size stays within
    ``memory_bytes``; once a member would exceed it, the stage spills (see the
    module docstring) and stays spilled for the rest of the attempt, so the group
    is published either wholly from memory or wholly from files. The stage owns
    every file it creates until :meth:`detach` hands the members to publication;
    :meth:`close` deletes whatever it still owns, so a failed, retried or
    cancelled attempt leaves no files behind.
    """

    def __init__(
        self,
        new_file: Callable[[], SpillFile],
        prepare: Callable[[str, pl.DataFrame], pl.DataFrame],
        started_at: datetime,
        memory_bytes: int = 0,
    ) -> None:
        self._new_file = new_file
        self._prepare = prepare
        self.started_at = started_at
        self.memory_bytes = memory_bytes
        self.members: dict[str, StagedMember] = {}
        self.failures: dict[str, SourceError] = {}
        self.spilled = False
        """Whether the group outgrew memory; once set, every member is a spill file."""
        self._held_bytes = 0
        self._files: list[SpillFile] = []

    def stage_batches(self, name: str, batches: BoundedBatches) -> StagedMember:
        """Keep ``batches`` in memory while they fit, else stream them to an owned file."""
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
        schema = dict(prepared.schema)
        renamed = self._renamed(batches, target)
        held = self._hold(renamed)
        if held.complete:
            frame = pl.from_arrow(pa.Table.from_batches(held.batches, schema=target))
            assert isinstance(frame, pl.DataFrame)
            self._held_bytes += held.size
            return self._staged(name, StagedMember(schema, batches.rows, held.size, frame=frame))
        self._spill_held()
        owned = self._own()
        size = held.size
        try:
            with owned.writer() as sink, pa.ipc.new_file(sink, target) as writer:
                for batch in held.batches:
                    writer.write_batch(batch)
                held.batches.clear()
                for batch in renamed:
                    writer.write_batch(batch)
                    size += batch.nbytes
        except BaseException:
            # A failed member may be skipped while the others continue: free its bytes now.
            self._discard(owned)
            raise
        return self._staged(name, StagedMember(schema, batches.rows, size, file=owned))

    def stage_frame(self, name: str, frame: pl.DataFrame) -> StagedMember:
        """Stage a member that its source can only read whole (a plugin's ``read``)."""
        prepared = self._prepared(name, frame)
        size = int(prepared.estimated_size())
        schema = dict(prepared.schema)
        if not self.spilled and self._held_bytes + size <= self.memory_bytes:
            self._held_bytes += size
            return self._staged(name, StagedMember(schema, prepared.height, size, frame=prepared))
        self._spill_held()
        member = StagedMember(schema, prepared.height, size, file=self._write_frame(prepared))
        return self._staged(name, member)

    def frames(self) -> Mapping[str, pl.DataFrame]:
        """Each staged member's frame, for publication of a group kept in memory."""
        assert not self.spilled
        return {
            name: member.frame for name, member in self.members.items() if member.frame is not None
        }

    def spills(self) -> Mapping[str, tuple[SpillFile, int]]:
        """Each staged member's file and decoded size, for publication of a spilled group."""
        assert self.spilled
        return {
            name: (member.file, member.size)
            for name, member in self.members.items()
            if member.file is not None
        }

    def detach(self) -> None:
        """Hand every staged file to its publisher; closing no longer deletes them."""
        self._files.clear()

    def close(self) -> None:
        """Delete every file this attempt still owns and drop its held frames."""
        files, self._files = self._files, []
        self.members.clear()
        self._held_bytes = 0
        raise_cleanup_errors(
            "snapshot staging files could not be released",
            attempt_cleanup(owned.close for owned in files),
        )

    def _hold(self, batches: Iterator[pa.RecordBatch]) -> _HeldBatches:
        """Take batches while the group still fits in memory; stop at the first that does not."""
        held: list[pa.RecordBatch] = []
        size = 0
        for batch in batches:
            held.append(batch)
            size += batch.nbytes
            if self.spilled or self._held_bytes + size > self.memory_bytes:
                return _HeldBatches(held, size, complete=False)
        return _HeldBatches(held, size, complete=not self.spilled)

    @staticmethod
    def _renamed(batches: Iterable[pa.RecordBatch], target: pa.Schema) -> Iterator[pa.RecordBatch]:
        import pyarrow as pa

        for batch in batches:
            yield pa.RecordBatch.from_arrays(batch.columns, schema=target)

    def _spill_held(self) -> None:
        """Write every member kept in memory to its own file; the stage stays spilled."""
        if self.spilled:
            return
        self.spilled = True
        for name, member in list(self.members.items()):
            assert member.frame is not None
            owned = self._write_frame(member.frame)
            self.members[name] = dataclasses.replace(member, file=owned, frame=None)
        self._held_bytes = 0

    def _write_frame(self, frame: pl.DataFrame) -> SpillFile:
        owned = self._own()
        try:
            with owned.writer() as sink:
                frame.write_ipc(sink, compression="uncompressed")
        except BaseException:
            self._discard(owned)
            raise
        return owned

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
