"""Row positions of duplicated keys, for INP-004 evidence.

Key summaries count each key once, so their samples hold no rows. A
:class:`DuplicateRows` lookup adds the first row positions of each sampled key,
on the same backend and under the same missing-value policy as the summary:
:class:`NativeDuplicateRows` in SQL, :class:`LazyDuplicateRows` on a Polars plan.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.sql import missing_sql
from qavf.redaction import exception_details, redact_text
from qavf.rules._execution import normalized_plan
from qavf.rules.helpers import fresh_internal
from qavf.sources.sql import quote_identifier

logger = logging.getLogger(__name__)

ROWS_PER_KEY = 10
"""Row positions listed for each duplicated key in the evidence."""


class DuplicateRows(ABC):
    """Adds to a sample of duplicated keys the first row positions of each key.

    Positions are 0-based, in the order the dataset was read, like ``_row`` in
    other evidence. The lookup reads the keys again; if that fails, the sample
    keeps its counts without positions.
    """

    def __init__(self, ctx: DataContext, dataset: str, columns: Sequence[str], name: str) -> None:
        self.ctx = ctx
        self.dataset = dataset
        self.columns = list(columns)
        self.name = name
        """The evidence column holding the positions."""

    @staticmethod
    def of(
        ctx: DataContext,
        dataset: str,
        columns: Sequence[str],
        name: str,
        *,
        selected: pl.DataFrame | None = None,
        observation: Observation | None = None,
    ) -> DuplicateRows:
        """The lookup for how the keys were summarized: in SQL unless they were read."""
        if observation is None and selected is None and ctx.execution_kind(dataset) == "native_sql":
            return NativeDuplicateRows(ctx, dataset, columns, name)
        return LazyDuplicateRows(ctx, dataset, columns, name, selected, observation)

    def attach(self, sample: pl.DataFrame) -> pl.DataFrame:
        if sample.is_empty():
            return sample
        try:
            found = self.locate(sample)
        except Exception as exc:
            logger.warning(
                "%s",
                redact_text(
                    f"Row positions of duplicated keys in {self.dataset}: {exception_details(exc)}"
                ),
            )
            return sample
        return sample.join(
            found, on=self.columns, how="left", nulls_equal=True, maintain_order="left"
        )

    @abstractmethod
    def locate(self, sample: pl.DataFrame) -> pl.DataFrame:
        """Each sampled key with its first :data:`ROWS_PER_KEY` positions."""

    def _row_name(self, sample: pl.DataFrame) -> str:
        return fresh_internal("row", [*sample.columns, self.name])


class NativeDuplicateRows(DuplicateRows):
    """Repeats the summary's filter and order in SQL, so it finds the sampled keys."""

    def locate(self, sample: pl.DataFrame) -> pl.DataFrame:
        q = quote_identifier
        row = q(self._row_name(sample))
        keys = ", ".join(q(c) for c in self.columns)
        present = " AND ".join(f"{q(c)} IS NOT NULL" for c in self.columns)
        order = ", ".join(f"{q(c)} NULLS FIRST" for c in self.columns)
        return self.ctx.sql(
            f"SELECT {keys}, min({row}, {ROWS_PER_KEY}) AS {q(self.name)} "
            f"FROM (SELECT {self._normalized_keys()}, row_number() OVER () - 1 AS {row} "
            f"FROM {q(self.dataset)}) AS source WHERE {present} "
            f"GROUP BY {keys} HAVING count(*) > 1 ORDER BY {order} LIMIT {sample.height}",
            datasets=(self.dataset,),
        )

    def _normalized_keys(self) -> str:
        """Key columns with declared missing values as null, as in the summary."""
        actual = self.ctx.schema(self.dataset)
        spec = self.ctx.specs.get(self.dataset)
        contracts = spec.schema_spec.columns if spec and spec.schema_spec else {}
        q = quote_identifier
        return ", ".join(
            f"CASE WHEN {missing_sql(c, actual[c], contracts.get(c))} THEN NULL "
            f"ELSE {q(c)} END AS {q(c)}"
            for c in self.columns
        )


class LazyDuplicateRows(DuplicateRows):
    """Semi-joins the sampled keys back to the rows that were summarized."""

    def __init__(
        self,
        ctx: DataContext,
        dataset: str,
        columns: Sequence[str],
        name: str,
        selected: pl.DataFrame | None,
        observation: Observation | None,
    ) -> None:
        super().__init__(ctx, dataset, columns, name)
        self.selected = selected
        self.observation = observation

    def locate(self, sample: pl.DataFrame) -> pl.DataFrame:
        plan = KeyPositionsPlan(self, sample.select(self.columns), self._row_name(sample))
        if self.observation is not None:
            return self.observation.collect(plan(self.observation.scan()))
        if self.selected is not None:
            return plan(self.selected.lazy()).collect()
        return self.ctx.reduce(self.dataset, plan)


class KeyPositionsPlan:
    """The Polars plan from a source to each wanted key's first row positions."""

    def __init__(self, lookup: DuplicateRows, wanted: pl.DataFrame, row: str) -> None:
        self.lookup = lookup
        self.wanted = wanted
        self.row = row

    def __call__(self, source: pl.LazyFrame) -> pl.LazyFrame:
        columns = self.lookup.columns
        keys = normalized_plan(self.lookup.ctx, self.lookup.dataset, source.select(columns))
        return (
            keys.with_row_index(self.row)
            .join(self.wanted.lazy(), on=columns, how="semi")
            .group_by(columns)
            .agg(pl.col(self.row).sort().head(ROWS_PER_KEY).alias(self.lookup.name))
        )
