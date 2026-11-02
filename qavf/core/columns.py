"""Column QA limits (INP-003) as SQL, one result per column check.

A column check is a column's own limits or one entry of its ``checks`` list. Its
row-level limits are unioned in SQL, so a row that breaks several of them is
counted once, and evidence names the limits each row breaks in ``violates``.
Column-level limits (missing fraction, statistics, distinct count) are folded
into the same result. Not-null, finite and unit contracts are checks of their
own. Counts are exact SQL counts; one fused query counts every check, and a
failure of that query falls back to counting each check on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from qavf import input_rules as wording
from qavf.core.aggregates import ExactAggregates
from qavf.core.contracts import condition_dependencies
from qavf.core.key_policy import FLOATS, INTEGERS
from qavf.core.policy import (
    NativeValueDomain,
    decimal_places_condition,
    native_bound_condition,
    native_literal,
    normalized,
)
from qavf.core.schema import dtype_name, evidence_name
from qavf.engine.severity import Status
from qavf.settings import ROW_CONSTRAINTS
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import sql_literal as lit

AGGREGATES = ("min", "max", "mean", "median", "std", "sum")
STATISTICS = (*AGGREGATES, "distinct_count")
_STREAMED = frozenset(("sum", "mean", "std"))


def _numeric(dtype):
    return dtype in INTEGERS or dtype in FLOATS or dtype.startswith("DECIMAL")


def _temporal(dtype):
    return dtype in ("DATE", "TIME") or dtype.startswith("TIMESTAMP")


def _within(value, bounds):
    """Whether an observed statistic lies inside its configured range."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    return (bounds.ge is None or value >= bounds.ge) and (bounds.le is None or value <= bounds.le)


def _exceeds(limit, missing, applicable):
    """Whether the applicable population exceeds its missing-value limit, exactly."""
    return (Fraction(missing, applicable) if applicable else Fraction(0)) > Fraction(limit)


def exact_kernel(check, dtype):
    """Whether the full backend decides this check with its exact kernel.

    The two backends then word an unassessable check the same way: the exact
    kernel lists every reason, the SQL reductions name the first one.
    """
    if check is None:
        return False
    constraints = check.constraints()
    statistics = {name for name in constraints if name in STATISTICS}
    rows = {name for name in constraints if name in ROW_CONSTRAINTS}
    floating = dtype in FLOATS
    if check.not_null:
        return True
    if not (
        check.normalize
        or statistics & {"median", "std"}
        or (floating and statistics & {"sum", "mean"})
        or (_temporal(dtype) and rows)
        or (floating and rows & {"integral", "max_decimals"})
    ):
        return False
    row_check = check.regex is not None and set(constraints) <= {
        *ROW_CONSTRAINTS,
        "max_null_fraction",
    }
    streamed = (
        floating and bool(statistics) and set(constraints) <= {*_STREAMED, "max_null_fraction"}
    )
    return not (row_check or streamed)


@dataclass
class ColumnContract:
    """One column check: its applicable rows, row-level limits and identity."""

    dataset: str
    column: str
    label: str
    descriptors: tuple[dict[str, Any], ...]
    scope: str
    check: Any = None
    severity: Any = None
    predicates: list[tuple[str, str]] = field(default_factory=list)
    dependencies: tuple[str, ...] = ()
    exact: bool = False
    problem: str | None = None
    """Why the check could not be prepared or counted."""
    unassessed: str | None = None
    """The complete message of a check that cannot be assessed at all."""
    missing: str = "FALSE"
    """The SQL of a missing value, for missing-fraction limits."""
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def identity(self):
        return f"{self.dataset}.{self.column}"

    @property
    def name(self):
        return f"{self.identity}: {self.label}"

    @property
    def statistics(self):
        if self.check is None:
            return {}
        return {k: v for k, v in self.check.constraints().items() if k in STATISTICS}

    @property
    def missing_fraction(self):
        return None if self.check is None else self.check.max_null_fraction

    @staticmethod
    def union(predicates):
        return " OR ".join(f"({sql})" for _, sql in predicates) or "FALSE"

    def aggregates(self, usable):
        """Exact reductions of this check, by name."""
        scope, bad = self.scope, self.union(self.predicates)
        reductions = {
            "applicable": f"count(*) FILTER (WHERE {scope})",
            "bad": f"count(*) FILTER (WHERE ({scope}) AND ({bad}))",
        }
        if self.missing_fraction is not None:
            reductions["missing"] = f"count(*) FILTER (WHERE ({scope}) AND ({self.missing}))"
            reductions["bad_or_missing"] = (
                f"count(*) FILTER (WHERE ({scope}) AND (({bad}) OR ({self.missing})))"
            )
        if self.statistics:
            reductions["usable"] = f"count({usable}) FILTER (WHERE {scope})"
        return reductions


class ColumnChecks:
    """INP-003 for one loaded dataset, in the full backend's order and wording."""

    RULE = "INP-003"

    def __init__(self, checks):
        self.checks = checks
        self.out = checks.out
        self.con = checks.con
        self.data = checks.data
        self.schema = checks.schema
        self.values = checks.values

    def run(self):
        contracts = []
        for column in self._columns():
            spec = self.schema.columns[column]
            if column not in self.data.columns:
                if spec.required:
                    self._absent(column, spec)
                continue
            contracts.extend(self._contracts(column, spec))
        self._count(contracts)
        for contract in contracts:
            self._finish(contract)

    def _columns(self):
        """Declared columns; those with N/A tokens last, assessed on their own population."""
        columns = list(self.schema.columns)
        applicable = [c for c in columns if c in self.values and self.schema.columns[c].na_values]
        return [c for c in columns if c not in applicable] + applicable

    def _absent(self, column, spec):
        name = self.data.name
        self.out.report(
            self.RULE,
            name,
            Status.ERROR,
            wording.required_column_absent(name, column, self.checks.available),
            check=f"{name}.{column}",
            check_id="column.available",
            datasets=(name,),
            columns=(column,),
            parameters={"required": spec.required},
            checks=(self._descriptor(column, "column.available", {"required": spec.required}),),
        )

    def _descriptor(self, column, check_id, parameters):
        return {
            "check_id": check_id,
            "datasets": (self.data.name,),
            "columns": (column,),
            "parameters": parameters,
        }

    # ------------------------------------------------------------- preparation
    def _contracts(self, column, spec):
        value = self.values[column]
        base = f"NOT ({value.na})" if spec.na_values else "TRUE"
        contracts = []
        if not spec.nullable:
            contracts.append(
                self._simple(column, "not null", base, ("missing", value.missing), spec)
            )
        if value.dtype in FLOATS and not spec.allow_infinite:
            contracts.append(
                self._simple(column, "finite", base, ("infinite", f"isinf({value.value})"), spec)
            )
        if spec.unit_column is not None:
            contracts.append(self._units(column, spec, base))
        checks = spec.all_checks()
        descriptors = {}
        for check in checks:
            kind = "each_allowed_once" if check.each_allowed_once else "constraints"
            descriptors.setdefault(wording.check_label(check), []).append(
                self._descriptor(
                    column,
                    f"column.{kind}",
                    check.model_dump(mode="python", exclude_defaults=True),
                )
            )
        for check in checks:
            label = wording.check_label(check)
            contracts.append(self._check(column, check, label, tuple(descriptors[label]), base))
        return contracts

    def _simple(self, column, label, scope, predicate, spec):
        parameters = (
            {"nullable": spec.nullable} if label == "not null" else {"allow_infinite": False}
        )
        check_id = f"column.{label.replace(' ', '_')}"
        return ColumnContract(
            self.data.name,
            column,
            label,
            (self._descriptor(column, check_id, parameters),),
            scope,
            predicates=[predicate],
            dependencies=(column,),
        )

    def _units(self, column, spec, scope):
        value = self.values[column]
        unit_column = spec.unit_column
        parameters = {
            "unit": spec.unit,
            "unit_column": unit_column,
            "allowed_units": spec.allowed_units,
        }
        contract = ColumnContract(
            self.data.name,
            column,
            "units",
            (self._descriptor(column, "column.units", parameters),),
            scope,
            dependencies=(column, unit_column),
        )
        if unit_column not in self.data.columns:
            contract.unassessed = wording.unit_column_absent(unit_column, self.data.name, column)
            return contract
        units = self.values[unit_column].value if unit_column in self.values else qi(unit_column)
        units = f"CAST({units} AS VARCHAR)"
        allowed = ", ".join(lit(unit) for unit in spec.allowed_units or [str(spec.unit)])
        contract.predicates.append(
            (
                "unit",
                f"({value.value}) IS NOT NULL AND ({units} IS NULL OR {units} NOT IN ({allowed}))",
            )
        )
        return contract

    def _check(self, column, check, label, descriptors, base):
        value = self.values[column]
        scope = base
        if check.where:
            scope = f"({base}) AND ({self.checks.selection(check.where)})"
        contract = ColumnContract(
            self.data.name,
            column,
            label,
            descriptors,
            scope,
            check=check,
            severity=check.severity,
            dependencies=(column, *condition_dependencies(check.where, self.data.columns)),
            exact=self.checks.exact_kernel or exact_kernel(check, value.dtype),
            missing=value.missing,
        )
        try:
            contract.predicates = self._row_limits(column, check, scope)
            if not contract.exact:
                for name in contract.statistics:
                    if name != "distinct_count" and not _numeric(value.dtype):
                        raise ValueError(
                            f"{name} requires a numeric column, found {dtype_name(value.dtype)}"
                        )
        except Exception as exc:
            contract.problem = f"{type(exc).__name__}: {exc}"
        return contract

    def _row_limits(self, column, check, scope):
        """The check's row-level limits as (reason, SQL of a breaking row)."""
        value = self.values[column]
        dtype = value.dtype
        expression = normalized(value.value, check.normalize)
        predicates = []
        constraints = check.constraints()
        if dtype in FLOATS and any(
            isinstance(v, float) and math.isnan(v) for v in constraints.get("notin") or ()
        ):
            raw = f"isnan(try_cast({qi(column)} AS DOUBLE))"
            predicates.append(("notin NaN", raw))
        for kind, expected in constraints.items():
            if kind not in ROW_CONSTRAINTS:
                continue
            bad = self._row_limit(kind, expected, expression, dtype, check.normalize)
            predicates.append(
                (wording.row_limit_label(kind, expected), f"({expression}) IS NOT NULL AND ({bad})")
            )
        if check.unique:
            duplicated = (
                f"{expression} IN (SELECT {expression} FROM {qi(self.data.table)} "
                f"WHERE {scope} GROUP BY {expression} HAVING count(*) > 1)"
            )
            predicates.append(("unique", f"({expression}) IS NOT NULL AND ({duplicated})"))
        if check.not_null:
            predicates.append(("missing", value.missing))
        return predicates

    @staticmethod
    def _row_limit(kind, expected, expression, dtype, methods=()):
        if kind in ("ge", "le", "gt", "lt"):
            return f"NOT ({native_bound_condition(expression, dtype, kind, expected)})"
        if kind in ("isin", "notin"):
            usable, foreign = NativeValueDomain(dtype).members(expected)
            if foreign:
                raise ValueError(
                    f"Options are incompatible with the converted column type: {foreign!r}"
                )
            allowed = [normalized(native_literal(v), methods) for v in usable]
            if not allowed:
                return "TRUE" if kind == "isin" else "FALSE"
            return f"{expression} {'NOT IN' if kind == 'isin' else 'IN'} ({', '.join(allowed)})"
        if kind == "regex":
            from qavf.regex_policy import require_native_regex

            require_native_regex(expected)
            return f"NOT regexp_full_match({expression}, {lit(expected)})"
        if kind in ("min_length", "max_length"):
            return f"length({expression}) {'<' if kind == 'min_length' else '>'} {expected}"
        if kind == "integral":
            return f"{expression} != trunc({expression})"
        return f"NOT ({decimal_places_condition(expression, dtype, expected)})"

    # ---------------------------------------------------------------- counting
    def _expression(self, contract):
        """The checked values, normalized as the check compares them."""
        methods = contract.check.normalize if contract.check is not None else ()
        return normalized(self.values[contract.column].value, methods)

    def _count(self, contracts):
        """Count every check in one query; on failure, each check on its own."""
        ready = [
            contract
            for contract in contracts
            if contract.problem is None and contract.unassessed is None
        ]
        try:
            self._reduce(ready)
        except Exception:
            # A bad contract must not suppress independent valid contracts.
            for contract in ready:
                try:
                    self._reduce([contract])
                except Exception as exc:
                    contract.problem = f"{type(exc).__name__}: {exc}"

    def _reduce(self, contracts):
        layout = [
            (contract, contract.aggregates(self._expression(contract))) for contract in contracts
        ]
        expressions = [sql for _, reductions in layout for sql in reductions.values()]
        if not expressions:
            return
        row = iter(
            self.con.execute(
                f"SELECT {', '.join(expressions)} FROM {qi(self.data.table)}"
            ).fetchone()
        )
        for contract, reductions in layout:
            contract.counts = {name: int(next(row) or 0) for name in reductions}

    # --------------------------------------------------------------- reporting
    def _finish(self, contract):
        spec = self.schema.columns[contract.column]
        if spec.na_values and self._all_not_applicable(contract):
            self._error(contract, wording.all_not_applicable(contract.name))
            return
        if contract.unassessed is not None:
            self._error(contract, contract.unassessed)
            return
        if contract.problem is not None:
            self._error(
                contract,
                wording.check_not_assessed(
                    contract.label, contract.dataset, contract.column, contract.problem
                ),
            )
            return
        try:
            self._assess(contract)
        except Exception as exc:
            self._error(
                contract,
                wording.check_not_assessed(
                    contract.label,
                    contract.dataset,
                    contract.column,
                    f"{type(exc).__name__}: {exc}",
                ),
            )

    def _all_not_applicable(self, contract):
        rows, na = self.data.rows, self.checks.population.get(contract.column, 0)
        return bool(
            rows and na == rows and contract.check and set(contract.statistics) & set(AGGREGATES)
        )

    def _assess(self, contract):
        counts = contract.counts
        applicable = counts["applicable"]
        check = contract.check
        if check is not None and check.where and applicable < check.min_applicable_rows:
            reason = wording.too_few_applicable(applicable, check.where, check.min_applicable_rows)
            self._unassessed(contract, [reason], [])
            return
        problems = []
        missing_rows = False
        limit = contract.missing_fraction
        if limit is not None and _exceeds(limit, counts["missing"], applicable):
            problems.append(wording.missing_fraction_problem(counts["missing"], applicable, limit))
            missing_rows = True
        unusable = self._statistics(contract, problems)
        if unusable:
            self._unassessed(contract, unusable, problems)
            return
        predicates = list(contract.predicates)
        if missing_rows:
            predicates.append(("missing", contract.missing))
        count = counts["bad_or_missing"] if missing_rows else counts["bad"]
        if not count and not problems:
            self.out.passed(self.RULE, contract.dataset, contract.name, *contract.descriptors)
            return
        if not count:
            self._column_level(contract, problems)
        else:
            self._rows(contract, count, applicable, problems, predicates)

    def _statistics(self, contract, problems):
        """Assess column-level statistics; return why any is undefined."""
        statistics = contract.statistics
        if not statistics:
            return []
        value = self.values[contract.column]
        usable = contract.counts.get("usable", 0)
        if not contract.exact:
            for name in sorted(statistics):
                minimum = wording.statistic_minimum(name)
                if usable < minimum:
                    return [wording.statistic_undefined(name, usable, minimum)]
        unusable = []
        try:
            for name, bounds in statistics.items():
                if name != "distinct_count" and not _numeric(value.dtype):
                    unusable.append(
                        f"{name} requires a numeric column, found {dtype_name(value.dtype)}"
                    )
                    continue
                observed = ExactAggregates(
                    self.con,
                    self.data.table,
                    self._expression(contract),
                    value.dtype,
                    contract.scope,
                ).statistic(name)
                if observed is None:
                    unusable.append(f"{name} is undefined: insufficient usable values")
                elif not _within(observed, bounds):
                    problems.append(wording.outside_bounds(name, observed, bounds))
        except Exception as exc:
            if not contract.exact:
                raise
            # Isolated to this check, as the exact kernel isolates it.
            unusable.append(f"evaluation failed: {type(exc).__name__}: {exc}")
        return unusable

    def _unassessed(self, contract, reasons, problems):
        if not contract.exact:
            reasons = [f"ValueError: {reason}" for reason in reasons]
        message = wording.check_not_assessed(
            contract.label, contract.dataset, contract.column, "; ".join(reasons)
        )
        if contract.exact and problems:
            message += f" Violations found meanwhile: {'; '.join(problems)}."
        self._error(contract, message)

    def _error(self, contract, message):
        self._report(contract, Status.ERROR, message)

    def _report(self, contract, status, message, **fields):
        descriptor = contract.descriptors[0]
        metrics = fields.pop("metrics", {})
        spec = self.schema.columns[contract.column]
        if spec.na_values:
            na = self.checks.population.get(contract.column, 0)
            metrics = {
                **metrics,
                "na_count": na,
                "applicable_count": self.data.rows - na,
                "population_count": self.data.rows,
            }
        self.out.report(
            self.RULE,
            contract.dataset,
            status,
            message,
            check=contract.name,
            check_id=descriptor["check_id"],
            datasets=descriptor["datasets"],
            columns=descriptor["columns"],
            parameters=descriptor["parameters"],
            checks=contract.descriptors,
            severity=contract.severity,
            metrics=metrics,
            **fields,
        )

    def _column_level(self, contract, problems):
        self._report(
            contract,
            Status.FAIL,
            wording.column_fails(contract.identity, contract.label, problems),
            records=[
                {"dataset": contract.dataset, "column": contract.column, "problem": problem}
                for problem in problems
            ],
            unit="problems",
        )

    def _rows(self, contract, count, applicable, problems, predicates):
        self._report(
            contract,
            Status.FAIL,
            wording.column_values_violate(
                count, applicable, contract.identity, contract.label, problems
            ),
            records=self._evidence(contract, predicates),
            count=count,
            total=applicable,
            unit="rows",
            metrics={"column_problems": problems} if problems else {},
        )

    def _evidence(self, contract, predicates):
        """The first breaking rows as read, with the limits each one breaks."""
        limit = self.out.evidence_limit
        if not limit:
            return None
        data = self.data
        shown = self.checks.shown(contract.dependencies)
        reason = evidence_name("violates", [*shown, data.row_column])
        reasons = ", ".join(f"CASE WHEN {sql} THEN {lit(label)} END" for label, sql in predicates)
        return self.out.records(
            f"SELECT {self.checks.converted(shown)}{', ' if shown else ''}"
            f"{qi(data.row_column)}, concat_ws('; ', {reasons}) AS {qi(reason)} "
            f"FROM {qi(data.table)} WHERE ({contract.scope}) AND "
            f"({ColumnContract.union(predicates)}) "
            f"ORDER BY {qi(data.row_column)} LIMIT {limit}"
        )
