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
from typing import Literal

import polars as pl

from qavf.duplicate_evidence import DuplicateEvidencePolicy
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.redaction import exception_details, redact_text
from qavf.rules._execution import _native_key_type
from qavf.rules._key_values import CanonicalKeyValues
from qavf.rules.helpers import fresh_internal
from qavf.sources.sql import quote_identifier, sql_literal

logger = logging.getLogger(__name__)

ROWS_PER_KEY = 10
"""Row positions listed for each duplicated key in the evidence."""


class DuplicateRows(ABC):
    """Adds to a sample of duplicated keys the first row positions of each key.

    Positions are 0-based, in the order the dataset was read, like ``_row`` in
    other evidence. The lookup reads the keys again; if that fails, the sample
    keeps its counts without positions.
    """

    def __init__(
        self,
        ctx: DataContext,
        dataset: str,
        columns: Sequence[str],
        name: str,
        limit: int | Literal["all"] = ROWS_PER_KEY,
    ) -> None:
        self.ctx = ctx
        self.dataset = dataset
        self.columns = list(columns)
        self.name = name
        """The evidence column holding the positions."""
        self.limit = limit
        self.available = True
        self.complete = True
        self.truncated = False

    @staticmethod
    def of(
        ctx: DataContext,
        dataset: str,
        columns: Sequence[str],
        name: str,
        *,
        selected: pl.DataFrame | None = None,
        observation: Observation | None = None,
        limit: int | Literal["all"] = ROWS_PER_KEY,
    ) -> DuplicateRows:
        """The lookup for how the keys were summarized: in SQL unless they were read."""
        if observation is None and selected is None and ctx.execution_kind(dataset) == "native_sql":
            return NativeDuplicateRows(ctx, dataset, columns, name, limit)
        return LazyDuplicateRows(ctx, dataset, columns, name, selected, observation, limit)

    def attach(self, sample: pl.DataFrame, frequency: str | None = None) -> pl.DataFrame:
        if sample.is_empty():
            return sample
        try:
            found = self.locate(sample)
        except Exception as exc:
            self.available = self.complete = False
            logger.warning(
                "%s",
                redact_text(
                    f"Row positions of duplicated keys in {self.dataset}: {exception_details(exc)}"
                ),
            )
            return sample
        joins = PositionJoinKeys(sample, self.columns)
        result = (
            joins.apply(sample.lazy())
            .join(
                joins.apply(found.lazy()).select([*joins.names, self.name]),
                on=joins.names,
                how="left",
                nulls_equal=True,
                maintain_order="left",
            )
            .drop(joins.auxiliary)
            .collect()
        )
        if frequency is not None:
            self.available = result[self.name].null_count() == 0
            records = result.to_dicts()
            positions = [record[self.name] or [] for record in records]
            accumulated = DuplicateEvidencePolicy(self.limit, self.position_budget).accumulator(
                records,
                frequency=frequency,
                positions=self.name,
            )
            for index, values in enumerate(positions):
                for value in values:
                    if not accumulated.append(index, value):
                        break
            self.complete = accumulated.metrics["duplicate_positions_complete"] and self.available
            self.truncated = accumulated.metrics["duplicate_positions_truncated"]
            # Rebuild only the position lists. Python datetime cannot round-trip
            # nanosecond key values, so the original typed key columns stay intact.
            result = result.with_columns(
                pl.Series(
                    self.name,
                    [record[self.name] for record in records],
                    dtype=result.schema[self.name],
                )
            )
        return result

    def bounded_limit(self, sample: pl.DataFrame) -> int:
        """Bound native/Python materialization before exact byte trimming.

        Positions become Python integers and list pointers in the report. A
        conservative 40 bytes per position also stays within the runner's
        retained-memory evidence limit, independently of source cardinality.
        """
        budget = max(1, self.position_budget // (40 * max(1, sample.height)))
        return budget if self.limit == "all" else min(self.limit, budget)

    @property
    def position_budget(self) -> int:
        # Counts/completeness metadata share the report's evidence byte budget.
        return max(0, self.ctx.evidence_bytes - 2048)

    @property
    def metrics(self) -> dict:
        """Whether sampled keys have positions, and whether those are exhaustive."""
        return {
            "duplicate_rows_per_key": self.limit,
            "duplicate_positions_available": self.available,
            "duplicate_positions_complete": self.complete,
            "duplicate_positions_truncated": self.truncated,
        }

    @abstractmethod
    def locate(self, sample: pl.DataFrame) -> pl.DataFrame:
        """Each sampled key with its configured first positions, or all positions."""

    def _row_name(self, sample: pl.DataFrame) -> str:
        return fresh_internal("row", [*sample.columns, self.name])


class NativeDuplicateRows(DuplicateRows):
    """Collect positions only for the bounded sample, using a null-safe semi-join."""

    def key_source(self, row_name: str) -> tuple[str, dict[str, pl.DataType]]:
        return CanonicalKeyValues(self.ctx, self.dataset, self.columns).sql(
            exclude_na=True,
            row_column=row_name,
        )

    def locate(self, sample: pl.DataFrame) -> pl.DataFrame:
        q = quote_identifier
        row_name = self._row_name(sample)
        row = q(row_name)
        keys = ", ".join(f"source.{q(c)}" for c in self.columns)
        source, schema = self.key_source(row_name)
        # The small sample alone crosses back into SQL. Explicit types retain
        # exact decimals/temporal values and make all-null samples unambiguous.
        literals = sample.select(self.columns).with_columns(
            pl.col(column).cast(pl.String)
            for column in self.columns
            if isinstance(schema[column], (pl.Datetime, pl.Duration))
        )
        values = ", ".join(
            "("
            + ", ".join(
                f"from_hex({sql_literal(value.hex())})"
                if isinstance(value, bytes)
                else (
                    f"CAST({sql_literal(str(value) if value is not None else None)} "
                    f"AS {_native_key_type(schema[column])})"
                )
                for column, value in zip(self.columns, record, strict=True)
            )
            + ")"
            for record in literals.iter_rows()
        )
        wanted_columns = ", ".join(q(c) for c in self.columns)
        join = " AND ".join(
            f"source.{q(c)} IS NOT DISTINCT FROM wanted.{q(c)}" for c in self.columns
        )
        positions = f"min({row}, {self.bounded_limit(sample)})"
        return self.ctx.sql(
            f"SELECT {keys}, {positions} AS {q(self.name)} "
            f"FROM ({source}) AS source "
            f"SEMI JOIN (VALUES {values}) AS wanted({wanted_columns}) ON {join} "
            f"GROUP BY {keys}",
            datasets=(self.dataset,),
            normalize_missing=False,
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
        limit: int | Literal["all"] = ROWS_PER_KEY,
    ) -> None:
        super().__init__(ctx, dataset, columns, name, limit)
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

    def __init__(self, lookup: LazyDuplicateRows, wanted: pl.DataFrame, row: str) -> None:
        self.lookup = lookup
        self.wanted = wanted
        self.row = row

    def __call__(self, source: pl.LazyFrame) -> pl.LazyFrame:
        columns = self.lookup.columns
        collector = self.lookup.observation.collect if self.lookup.observation is not None else None
        keys = CanonicalKeyValues(self.lookup.ctx, self.lookup.dataset, columns).lazy(
            source.with_row_index(self.row),
            collector,
            exclude_na=True,
            preserve=(self.row,),
        )
        positions = pl.col(self.row).sort().head(self.lookup.bounded_limit(self.wanted))
        joins = PositionJoinKeys(self.wanted, columns)
        return (
            joins.apply(keys)
            .join(
                joins.apply(self.wanted.lazy()).select(joins.names),
                on=joins.names,
                how="semi",
                nulls_equal=True,
            )
            .group_by(columns)
            .agg(positions.alias(self.lookup.name))
        )


class PositionJoinKeys:
    """Lossless join aliases for Decimal/Int128 on supported older Polars releases.

    Polars 1.32 groups these types but cannot hash-join them. Their canonical
    strings are exact, unlike floats; original typed columns remain evidence.
    """

    def __init__(self, sample: pl.DataFrame, columns: Sequence[str]) -> None:
        self.names: list[str] = []
        self.expressions: list[pl.Expr] = []
        self.auxiliary: list[str] = []
        for column in columns:
            dtype = sample.schema[column]
            if isinstance(dtype, pl.Decimal) or dtype == pl.Int128:
                name = fresh_internal("join_" + column, [*sample.columns, *self.auxiliary])
                self.auxiliary.append(name)
                self.expressions.append(pl.col(column).cast(pl.String).alias(name))
            else:
                name = column
            self.names.append(name)

    def apply(self, plan: pl.LazyFrame) -> pl.LazyFrame:
        return plan.with_columns(self.expressions)
