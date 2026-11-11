"""The file and row within it of evidence rows from multi-file inputs.

Evidence names a row by ``_row``, its 0-based position in the dataset as read.
For a dataset read from several files, :class:`RowLocator` adds ``_file`` and
``_file_row`` (0-based within that file) to each evidence row. Only the bounded
evidence is mapped, through one cached count of rows per file; the count runs
only when such evidence exists.
"""

from __future__ import annotations

import bisect
import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.registry import Finding
from qavf.redaction import exception_details, redact_text

if TYPE_CHECKING:
    from qavf.engine.ports import SourceLookup

logger = logging.getLogger(__name__)

ROW = "_row"
FILE = "_file"
FILE_ROW = "_file_row"


class FileIndex:
    """Maps a dataset row position to its file and its row within that file."""

    def __init__(self, files: Sequence[tuple[str, int]]) -> None:
        self.files = [name for name, _ in files]
        self.starts = [0, *itertools.accumulate(rows for _, rows in files)]

    def locate(self, position: int) -> tuple[str | None, int | None]:
        index = bisect.bisect_right(self.starts, position) - 1
        if not 0 <= index < len(self.files):
            return None, None
        return self.files[index], position - self.starts[index]

    def annotate(
        self, affected: pl.DataFrame | Sequence[Mapping[str, Any]], row_column: str = ROW
    ) -> pl.DataFrame | list[dict[str, Any]]:
        from qavf.rules.helpers import annotate, fresh_internal, present

        if isinstance(affected, pl.DataFrame):
            located = [self.locate(int(p)) for p in affected[row_column].to_list()]
            file_column = fresh_internal("file", affected.columns)
            file_row_column = fresh_internal("file_row", [*affected.columns, file_column])
            located_frame = affected.with_columns(
                pl.Series(file_column, [file for file, _ in located], dtype=pl.String),
                pl.Series(file_row_column, [row for _, row in located], dtype=pl.Int64),
            )
            return present(located_frame, {file_column: FILE, file_row_column: FILE_ROW})
        records = []
        for record in affected:
            file, row = self.locate(int(record[row_column]))
            records.append(annotate(record, **{FILE: file, FILE_ROW: row}))
        return records


class RowLocator:
    """Adds the file and the row within it to a finding's evidence, when it can."""

    def __init__(self, ctx: SourceLookup) -> None:
        self.ctx = ctx

    def locate(self, finding: Finding) -> Finding:
        """``finding`` with located evidence; unchanged when there is nothing to
        locate or locating fails (evidence keeps ``_row`` either way)."""
        if len(finding.datasets) != 1 or not self._positioned(finding.affected, finding.row_column):
            return finding
        dataset = finding.datasets[0]
        source = self.ctx.source(dataset)
        if source is None:
            return finding
        try:
            files = source.file_rows()
            if not files:
                return finding
            assert finding.affected is not None
            assert finding.row_column is not None
            return replace(
                finding, affected=FileIndex(files).annotate(finding.affected, finding.row_column)
            )
        except Exception as exc:
            logger.warning(
                "%s", redact_text(f"Files of evidence rows in {dataset}: {exception_details(exc)}")
            )
            return finding

    @staticmethod
    def _positioned(
        affected: pl.DataFrame | Sequence[Mapping[str, Any]] | None,
        row_column: str | None = ROW,
    ) -> bool:
        if affected is None or len(affected) == 0 or row_column is None:
            return False
        if isinstance(affected, pl.DataFrame):
            return row_column in affected.columns
        return all(isinstance(record, Mapping) and row_column in record for record in affected)
