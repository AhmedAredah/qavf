"""Narrow owned-batch capability with a compatibility adapter for source plugins."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from qavf.sources.base import DataSource, SourceError


class BatchReader(Protocol):
    @property
    def schema(self) -> Mapping[str, pl.DataType]: ...

    @property
    def row_budget(self) -> int | None: ...

    def open(self) -> AbstractContextManager[Iterator[pl.DataFrame]]: ...

    def record_rows(self, rows: int) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceBatchReader:
    """Existing plugins retain their DataSource hooks and source-policy scopes."""

    source: DataSource
    schema: Mapping[str, pl.DataType]

    @classmethod
    def adapt(cls, source: DataSource) -> SourceBatchReader:
        schema = source.batch_schema()
        if schema is None:
            raise SourceError("source does not declare a batch schema")
        return cls(source, schema)

    @property
    def row_budget(self) -> int | None:
        return self.source.params.row_budget

    def open(self) -> AbstractContextManager[Iterator[pl.DataFrame]]:
        return self.source.open_batches()

    def record_rows(self, rows: int) -> None:
        self.source.load_notes["batch_rows"] = rows
