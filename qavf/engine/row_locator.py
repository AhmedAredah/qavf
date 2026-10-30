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
    from qavf.engine.context import DataContext

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
        self, affected: pl.DataFrame | Sequence[Mapping[str, Any]]
    ) -> pl.DataFrame | list[dict[str, Any]]:
        if isinstance(affected, pl.DataFrame):
            located = [self.locate(int(p)) for p in affected[ROW].to_list()]
            return affected.with_columns(
                pl.Series(FILE, [file for file, _ in located], dtype=pl.String),
                pl.Series(FILE_ROW, [row for _, row in located], dtype=pl.Int64),
            )
        records = []
        for record in affected:
            file, row = self.locate(int(record[ROW]))
            records.append({**record, FILE: file, FILE_ROW: row})
        return records


class RowLocator:
    """Adds the file and the row within it to a finding's evidence, when it can."""

    def __init__(self, ctx: DataContext) -> None:
        self.ctx = ctx

    def locate(self, finding: Finding) -> Finding:
        """``finding`` with located evidence; unchanged when there is nothing to
        locate or locating fails (evidence keeps ``_row`` either way)."""
        if len(finding.datasets) != 1 or not self._positioned(finding.affected):
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
            return replace(finding, affected=FileIndex(files).annotate(finding.affected))
        except Exception as exc:
            logger.warning(
                "%s", redact_text(f"Files of evidence rows in {dataset}: {exception_details(exc)}")
            )
            return finding

    @staticmethod
    def _positioned(affected: pl.DataFrame | Sequence[Mapping[str, Any]] | None) -> bool:
        if affected is None or len(affected) == 0:
            return False
        if isinstance(affected, pl.DataFrame):
            return ROW in affected.columns
        return all(isinstance(record, Mapping) and ROW in record for record in affected)
