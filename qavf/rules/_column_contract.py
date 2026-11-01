"""Normalized column checks and backend-independent finding semantics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl

from qavf.engine.registry import Finding
from qavf.settings import COLUMN_CONSTRAINTS, ROW_CONSTRAINTS, ColumnCheck, ColumnSpec, SchemaSpec
from qavf.value_policy import is_text


def require_statistic_population(statistics: Iterable[str], usable: int) -> None:
    """Undefined numeric statistics are unassessed, never fabricated violations.

    Invoke after the real reduction; zero-row binder preflights are not data.
    Empty distinct cardinality remains a defined count of zero.
    """
    for name in sorted(statistics):
        minimum = 2 if name == "std" else 0 if name == "distinct_count" else 1
        if usable < minimum:
            raise ValueError(
                f"{name} is undefined: {usable} usable value(s), at least {minimum} required"
            )


@dataclass(frozen=True, slots=True)
class ColumnPlan:
    """One independently assessed column contract and its input dependencies."""

    column: str
    spec: ColumnSpec
    kind: str
    check: ColumnCheck | None = None

    @property
    def dependencies(self) -> tuple[str, ...]:
        columns = [self.column]
        if self.kind == "units" and self.spec.unit_column:
            columns.append(self.spec.unit_column)
        if self.check is not None and self.check.where:
            columns.extend(pl.sql_expr(self.check.where).meta.root_names())
        return tuple(dict.fromkeys(columns))

    @property
    def exact_only(self) -> bool:
        """Whether only the exact kernel assesses this contract: ``not_null`` lists
        its missing rows, which no reduction here retains."""
        return self.check is not None and self.check.not_null

    def single_schema(self) -> SchemaSpec:
        """Only this contract, for projected fallback through the eager kernel."""
        raw = self.spec.model_dump()
        defaults = ColumnSpec().model_dump()
        for name in (
            *ROW_CONSTRAINTS,
            *COLUMN_CONSTRAINTS,
            "normalize",
            "convert_options",
            "each_allowed_once",
        ):
            raw[name] = defaults[name]
        raw.update(
            checks=[],
            nullable=True,
            allow_infinite=True,
            unit_column=None,
            allowed_units=None,
            unit=None,
        )
        if self.kind == "not null":
            raw["nullable"] = False
        elif self.kind == "finite":
            raw["allow_infinite"] = False
        elif self.kind == "units":
            raw.update(
                unit=self.spec.unit,
                unit_column=self.spec.unit_column,
                allowed_units=self.spec.allowed_units,
            )
        else:
            raw["checks"] = [self.check]
        return SchemaSpec(columns={self.column: ColumnSpec.model_validate(raw)})


def column_plans(schema: SchemaSpec, actual: dict[str, pl.DataType]) -> list[ColumnPlan]:
    plans = []
    for column, spec in schema.columns.items():
        if column not in actual:
            continue
        if not spec.nullable:
            plans.append(ColumnPlan(column, spec, "not null"))
        if actual[column].is_float() and not spec.allow_infinite:
            plans.append(ColumnPlan(column, spec, "finite"))
        if spec.unit_column is not None:
            plans.append(ColumnPlan(column, spec, "units"))
        plans.extend(ColumnPlan(column, spec, "check", check) for check in spec.all_checks())
    return plans


def _unsafe_unit_cast(item: ColumnPlan, actual: dict[str, pl.DataType]) -> bool:
    """Empty preflight cannot validate a data-dependent non-text cast to String."""
    return item.kind == "units" and not is_text(actual.get(str(item.spec.unit_column), pl.Null()))


@dataclass(frozen=True, slots=True)
class ColumnOutcome:
    """Backend-independent applicability and finding semantics, never row execution."""

    dataset: str
    column: str
    label: str
    check: ColumnCheck | None

    def applicability_problem(self, rows: int) -> str | None:
        if self.check and rows < self.check.min_applicable_rows:
            return (
                f"only {rows} row(s) match 'where: {self.check.where}', "
                f"fewer than min_applicable_rows {self.check.min_applicable_rows}"
            )
        return None

    def require_applicable(self, rows: int) -> None:
        if problem := self.applicability_problem(rows):
            raise ValueError(problem)

    def finding(
        self, count: int, applicable: int, problems: list[str], sample: Any = None
    ) -> Finding | None:
        if not count and not problems:
            return None
        identity = f"{self.dataset}.{self.column}"
        severity = self.check.severity if self.check else None
        if not count:
            return Finding.fail(
                f"{identity} fails '{self.label}': {'; '.join(problems)}.",
                check=f"{identity}: {self.label}",
                severity=severity,
                affected=[
                    {"dataset": self.dataset, "column": self.column, "problem": p} for p in problems
                ],
                unit="problems",
                datasets=(self.dataset,),
            )
        return Finding.fail(
            f"{count:,} value{'s' if count != 1 else ''} of {applicable:,} "
            f"in {identity} {'violates' if count == 1 else 'violate'} '{self.label}'."
            + (f" Column-level: {'; '.join(problems)}." if problems else ""),
            check=f"{identity}: {self.label}",
            severity=severity,
            affected=sample,
            affected_count=count,
            total_records=applicable,
            unit="rows",
            metrics={"column_problems": problems} if problems else {},
            datasets=(self.dataset,),
        )
