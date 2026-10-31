"""Observation storage policy independent of dataset orchestration and readers."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

import polars as pl

from qavf.engine.cache_ownership import compact_polars_with_size
from qavf.engine.retention import RetainedBudget
from qavf.engine.spill import SpillBudget, SpillFile


@dataclass(frozen=True, slots=True)
class ObservationStorage:
    """The only capabilities needed to admit and write observation chunks."""

    directory: Path
    retained: RetainedBudget
    working: RetainedBudget
    spill: SpillBudget
    check_cancelled: Callable[[], None]

    def new_file(self) -> SpillFile:
        self.check_cancelled()
        return SpillFile(self.directory / f"{uuid4().hex}.arrow", self.spill)

    def write(self, frame: pl.DataFrame) -> SpillFile:
        owned = self.new_file()
        try:
            with owned.writer() as writer:
                frame.write_ipc(writer, compression="uncompressed")
            self.check_cancelled()
            return owned
        except BaseException:
            owned.close()
            raise

    def sink(self, plan: pl.LazyFrame) -> SpillFile:
        owned = self.new_file()
        try:
            with owned.writer() as writer:
                plan.sink_ipc(
                    writer, compression="uncompressed", maintain_order=True, engine="streaming"
                )
            self.check_cancelled()
            return owned
        except BaseException:
            owned.close()
            raise


class BatchSpool:
    """Coalesce producer fragments within shared working bytes and row/chunk caps.

    Files contain complete independent IPC chunks. No producer buffer is borrowed
    across calls. Oversized batches are written synchronously without a retention
    copy; native writer working memory remains subject to process-level policy.
    """

    MAX_ROWS = 65_536
    MAX_FRAGMENTS = 64

    def __init__(self, storage: ObservationStorage, publish: Callable[[SpillFile], None]) -> None:
        self.storage, self.publish = storage, publish
        self.frames: list[pl.DataFrame] = []
        self._levels: list[int] = []
        self.bytes = self.rows = 0

    def append(self, frame: pl.DataFrame) -> None:
        self.storage.check_cancelled()
        if self.rows + frame.height > self.MAX_ROWS:
            self.flush()
        estimate = int(frame.estimated_size()) + sys.getsizeof(frame) + 8
        if estimate > self.storage.working.limit or frame.height >= self.MAX_ROWS:
            self.flush()
            self.publish(self.storage.write(frame))
            return
        if not self.storage.working.reserve(estimate):
            self.flush()
            if not self.storage.working.reserve(estimate):
                self.publish(self.storage.write(frame))
                return
        charge = estimate
        try:
            copied, size = compact_polars_with_size(frame)
            actual = size + sys.getsizeof(copied) + 8
            if actual > charge:
                if not self.storage.working.reserve(actual - charge):
                    self.flush()
                    self.publish(self.storage.write(frame))
                    return
            else:
                self.storage.working.release(charge - actual)
            charge = actual
            self.frames.append(cast(pl.DataFrame, copied))
            self._levels.append(0)
            self.bytes += charge
            self.rows += frame.height
            charge = 0
            # Binary merging bounds both wrapper count and work: each row is
            # copied at most logarithmically within a fixed-size output chunk.
            # Merging the growing prefix on every small fragment is quadratic.
            self._merge_tail()
            if len(self.frames) >= self.MAX_FRAGMENTS:
                self.flush()
            if self.rows >= self.MAX_ROWS:
                self.flush()
        finally:
            self.storage.working.release(charge)

    @staticmethod
    def _size(frame: pl.DataFrame) -> int:
        return int(frame.estimated_size()) + sys.getsizeof(frame) + 8

    def _merge_tail(self) -> None:
        while len(self._levels) >= 2 and self._levels[-1] == self._levels[-2]:
            left, right = self.frames[-2:]
            merged = pl.concat([left, right], rechunk=True)
            change = self._size(merged) - self._size(left) - self._size(right)
            if change > 0 and not self.storage.working.reserve(change):
                return
            if change < 0:
                self.storage.working.release(-change)
            self.bytes += change
            self.frames[-2:] = [merged]
            self._levels[-2:] = [self._levels[-1] + 1]

    def flush(self) -> None:
        if not self.frames:
            return
        self.storage.check_cancelled()
        merged = pl.concat(self.frames, rechunk=True)
        self.publish(self.storage.write(merged))
        self.close()

    def close(self) -> None:
        self.frames.clear()
        self._levels.clear()
        self.storage.working.release(self.bytes)
        self.bytes = self.rows = 0
