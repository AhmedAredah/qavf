"""Dataset views on the native DuckDB backend.

A view (``datasets.roads.views.active: {where: "status = 'open'"}``) is queried as
``"roads.active"`` wherever SQL names a dataset. The native backend registers each
as a SQL view over the loaded dataset, ``SELECT <columns or *> FROM "<dataset>"
WHERE <where>``, the rows and columns the Polars backend's view gives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import duckdb

from qavf.redaction import redact_text
from qavf.sources.sql import quote_identifier as qi

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings, ViewSpec

logger = logging.getLogger(__name__)


class NativeViews:
    """Registers and drops the configured views of loaded datasets on one connection."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, settings: QAVFSettings) -> None:
        self.connection = connection
        self.settings = settings
        self.failures: dict[str, str] = {}
        """Views that could not be registered, with the reason."""

    def names(self, dataset: str) -> list[str]:
        """The relation names of ``dataset``'s views (``<dataset>.<view>``)."""
        spec = self.settings.datasets.get(dataset)
        return [f"{dataset}.{view}" for view in spec.views] if spec is not None else []

    @staticmethod
    def statement(dataset: str, name: str, spec: ViewSpec) -> str:
        """The ``CREATE TEMP VIEW`` statement of view ``name`` of ``dataset``."""
        columns = ", ".join(qi(column) for column in spec.columns) if spec.columns else "*"
        where = f" WHERE ({spec.where})" if spec.where is not None else ""
        return (
            f"CREATE OR REPLACE TEMP VIEW {qi(name)} AS SELECT {columns} FROM {qi(dataset)}{where}"
        )

    def create(self, dataset: str) -> None:
        """Register every view of ``dataset``, which is already queryable by its name."""
        spec = self.settings.datasets.get(dataset)
        if spec is None:
            return
        for view, definition in spec.views.items():
            name = f"{dataset}.{view}"
            try:
                self.connection.execute(self.statement(dataset, name, definition))
            except duckdb.Error as exc:
                # A query naming the view then reports it as missing, as an unloadable
                # view is on the other backend; the reason is kept for diagnostics.
                self.failures[name] = redact_text(str(exc))
                logger.warning("view %s could not be registered: %s", name, self.failures[name])

    def drop(self, dataset: str) -> None:
        """Drop the views of ``dataset`` (before the dataset itself is released)."""
        for name in self.names(dataset):
            self.connection.execute(f"DROP VIEW IF EXISTS {qi(name)}")


__all__ = ["NativeViews"]
