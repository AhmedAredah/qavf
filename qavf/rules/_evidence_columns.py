"""Which data columns failing-row evidence of a column check shows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from qavf.settings import SchemaSpec


@dataclass(frozen=True, slots=True)
class EvidenceColumns:
    """The data columns of a dataset's failing-row evidence.

    With ``evidence_columns: checked`` (the default) a check's evidence shows the
    columns it depends on (the checked column, the columns of its ``where`` and its
    unit column) and the ``primary_key``; unrelated columns are never read for it.
    ``all`` shows whole rows. Row locators (``_row``, ``_file``, ``_file_row``) and
    ``violates`` are added to either.
    """

    schema: SchemaSpec
    available: tuple[str, ...]
    """The dataset's columns, in their order."""

    @property
    def whole_rows(self) -> bool:
        return self.schema.evidence_columns == "all"

    def of(self, dependencies: Iterable[str]) -> list[str]:
        """The shown columns of a check depending on ``dependencies``, in dataset order."""
        if self.whole_rows:
            return list(self.available)
        wanted = {*dependencies, *self.schema.primary_key}
        return [column for column in self.available if column in wanted]
