"""Value-level conformance checks of INP-002, where column types alone cannot decide.

A column whose type differs from its declaration may still hold values of the
declared type: text read from a CSV, whole numbers stored as floats. Each
:class:`ColumnConformance` decides when it applies and reports exactly the values
that do not conform. :class:`SourceChecks` reports what the source itself found
while reading: values its typed read could not convert, lines it left out, and
header names it had to invent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.rules._declared_types import convert, converts_text, described
from qavf.rules._row_summary import RowSummary, row_aliases, summarize_rows
from qavf.rules._temporal import temporal_evidence
from qavf.rules.helpers import plural, present
from qavf.settings import ColumnSpec, SchemaSpec
from qavf.sources.base import DataSource, TypeViolations
from qavf.sources.sql import quote_identifier, sql_literal

from . import _column_values as values


@dataclass(frozen=True)
class DatasetView:
    """A dataset as INP-002 sees it: its contract, actual types and row count."""

    ctx: DataContext
    name: str
    schema: SchemaSpec
    actual: Mapping[str, pl.DataType]
    rows: int

    def context(self, column: str) -> list[str]:
        """Primary-key columns that identify a row in evidence about ``column``."""
        return [c for c in self.schema.primary_key or [] if c != column and c in self.actual]

    def unassessed(self, check: str, exc: Exception) -> Finding:
        return Finding.unassessed(
            f"Not assessed: {check}: {type(exc).__name__}: {exc}.",
            check=check,
            datasets=(self.name,),
        )


class ColumnConformance(ABC):
    """Values of one column that are not of its declared type."""

    def __init__(self, view: DatasetView, column: str) -> None:
        self.view = view
        self.column = column
        self.spec = view.schema.columns[column]

    @staticmethod
    @abstractmethod
    def applies(spec: ColumnSpec, actual: pl.DataType) -> bool:
        """Whether this check decides conformance for a column read as ``actual``."""

    @property
    def check(self) -> str:
        return f"{self.view.name}.{self.column}: dtype {self.spec.dtype}"

    def finding(self) -> Finding | None:
        """The failing values; ``None`` when every value conforms."""
        try:
            return self._assess()
        except Exception as exc:
            return self.view.unassessed(self.check, exc)

    @abstractmethod
    def _assess(self) -> Finding | None: ...


class TextValues(ColumnConformance):
    """A column read as text holding values of its declared type, except a few cells.

    A column whose every value converts conforms: it is text only because of
    declared missing markers, and INP-003 checks the converted values.
    """

    @staticmethod
    def applies(spec: ColumnSpec, actual: pl.DataType) -> bool:
        return converts_text(spec, actual)

    def _assess(self) -> Finding | None:
        view, column = self.view, self.column
        frame = view.ctx.project(view.name, [*view.context(column), column])
        invalid = convert(frame[column], self.spec).invalid
        count = int(invalid.sum())
        if not count:
            return None
        kind = described(str(self.spec.dtype))
        return Finding.fail(
            f"{plural(count, 'value')} of {frame.height:,} in {view.name}.{column} "
            f"{'is' if count == 1 else 'are'} not {kind}, so the column was read as "
            f"{view.actual[column]}.",
            check=self.check,
            affected=values._rows(frame, invalid, f"not {kind}", view.ctx.evidence_limit),
            affected_count=count,
            total_records=frame.height,
            unit="rows",
            datasets=(view.name,),
        )


class NonIntegralValues(ColumnConformance):
    """A float column declared ``int``: spreadsheets store every number as a float.

    One streaming reduction with a bounded sample; NaN is missing, not a fraction.
    """

    LABEL: ClassVar[str] = "not an int"

    def __init__(self, view: DatasetView, column: str) -> None:
        super().__init__(view, column)
        self.row, self.problem = row_aliases([*view.actual])

    @staticmethod
    def applies(spec: ColumnSpec, actual: pl.DataType) -> bool:
        return spec.dtype == "int" and actual.is_float()

    def _assess(self) -> Finding | None:
        summary = self._summary()
        if not summary.invalid:
            return None
        view = self.view
        whole = "is not a whole number" if summary.invalid == 1 else "are not whole numbers"
        return Finding.fail(
            f"{plural(summary.invalid, 'value')} of {summary.rows:,} in {view.name}."
            f"{self.column} {whole} in a column declared int.",
            check=self.check,
            affected=temporal_evidence(
                present(summary.sample, {self.row: "_row", self.problem: "violates"})
            ),
            affected_count=summary.invalid,
            total_records=summary.rows,
            unit="rows",
            datasets=(view.name,),
        )

    def _summary(self) -> RowSummary:
        view = self.view
        sample = [*view.context(self.column), self.column, self.row, self.problem]
        if view.ctx.execution_kind(view.name) == "native_sql":
            return summarize_rows(
                view.ctx,
                view.name,
                problem=self.problem,
                labels=(self.LABEL,),
                sample_columns=sample,
                sample_order=self.row,
                sql=self._classified_sql(),
            )
        return summarize_rows(
            view.ctx,
            view.name,
            problem=self.problem,
            labels=(self.LABEL,),
            sample_columns=sample,
            build=self._classified_plan,
        )

    def _classified_sql(self) -> str:
        q = quote_identifier
        value = q(self.column)
        selected = ", ".join(q(c) for c in [*self.view.context(self.column), self.column])
        return (
            f"SELECT {selected}, row_number() OVER () - 1 AS {q(self.row)}, "
            f"CASE WHEN {value} IS NOT NULL AND NOT isnan({value}) AND "
            f"(isinf({value}) OR {value} <> trunc({value})) THEN {sql_literal(self.LABEL)} "
            f"END AS {q(self.problem)} FROM {q(self.view.name)}"
        )

    def _classified_plan(self, source: pl.LazyFrame) -> pl.LazyFrame:
        value = pl.col(self.column)
        fractional = (
            value.is_not_null()
            & ~value.is_nan()
            & (value.is_infinite() | (value != value.floor()))
        )
        return (
            source.select([*self.view.context(self.column), self.column])
            .with_row_index(self.row)
            .with_columns(pl.when(fractional).then(pl.lit(self.LABEL)).alias(self.problem))
        )


COLUMN_CHECKS: tuple[type[ColumnConformance], ...] = (NonIntegralValues, TextValues)
"""Value-level checks, consulted in order before a type mismatch is reported."""


def column_conformance(view: DatasetView, column: str) -> ColumnConformance | None:
    """The check that decides ``column``'s conformance from its values, if any."""
    spec, actual = view.schema.columns[column], view.actual[column]
    kind = next((check for check in COLUMN_CHECKS if check.applies(spec, actual)), None)
    return kind(view, column) if kind is not None else None


class SourceChecks:
    """What a dataset's source found while reading, as INP-002 findings and problems."""

    def __init__(self, view: DatasetView, source: DataSource, rename: Mapping[str, str]) -> None:
        self.view = view
        self.source = source
        self.rename = dict(rename)

    def findings(self) -> list[Finding]:
        found = self._typed_read_values()
        unreadable = self._unreadable_lines()
        return [*found, unreadable] if unreadable is not None else found

    def problems(self) -> list[dict[str, Any]]:
        """Header names the reader had to invent (a diagnosis that never blocks)."""
        try:
            return self.source.header_problems()
        except Exception:
            return []

    def _typed_read_values(self) -> list[Finding]:
        """Values the source's typed read could not convert (they read as missing)."""
        original = {new: old for old, new in self.rename.items()}
        context = [original.get(c, c) for c in self.view.schema.primary_key or []]
        try:
            violations = self.source.type_violations(context)
        except Exception as exc:
            return [self.view.unassessed(f"{self.view.name}: declared types", exc)]
        findings = []
        for name, found in violations.items():
            column = self.rename.get(name, name)
            spec = self.view.schema.columns.get(column)
            if found.count and spec is not None and spec.dtype is not None:
                findings.append(self._typed_read_finding(column, spec, found, context))
        return findings

    def _typed_read_finding(
        self, column: str, spec: ColumnSpec, found: TypeViolations, context: list[str]
    ) -> Finding:
        view = self.view
        kind = described(str(spec.dtype)) + (
            f" written as {spec.format!r}" if spec.format else ""
        )
        evidence = [
            {
                "_row": entry["_row"],
                **{self.rename.get(c, c): entry[c] for c in context if c in entry},
                column: entry["value"],
                "violates": f"not {kind}",
            }
            for entry in found.sample[: view.ctx.evidence_limit]
        ]
        one = found.count == 1
        return Finding.fail(
            f"{plural(found.count, 'value')} of {view.rows:,} in {view.name}.{column} "
            f"{'is' if one else 'are'} not {kind}, so {'it is' if one else 'they are'} "
            "read as missing.",
            check=f"{view.name}.{column}: dtype {spec.dtype}",
            affected=evidence,
            affected_count=found.count,
            total_records=view.rows,
            unit="rows",
            datasets=(view.name,),
        )

    def _unreadable_lines(self) -> Finding | None:
        """Lines a lossy read left out, beyond the source's ``max_rejected_rows``."""
        rejected = self.source.rejected_rows()
        allowed = int(getattr(self.source.params, "max_rejected_rows", 0))
        if rejected is None or rejected.count <= allowed:
            return None
        count = f"{rejected.count:,}" if rejected.exact else f"at least {rejected.count:,}"
        one = rejected.count == 1
        return Finding.fail(
            f"{count} {'line' if one else 'lines'} of {self.view.name} could not be read and "
            f"{'was' if one else 'were'} left out (max_rejected_rows: {allowed}).",
            check=f"{self.view.name}: unreadable lines",
            affected=list(rejected.sample[: self.view.ctx.evidence_limit]),
            affected_count=rejected.count,
            unit="lines",
            datasets=(self.view.name,),
        )
