"""Exact eager execution and bounded evidence for column contracts."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding
from qavf.input_rules import (
    check_label,
    check_not_assessed,
    column_values_violate,
    missing_fraction_problem,
    outside_bounds,
    unit_column_absent,
)
from qavf.redaction import exception_details, redact_text
from qavf.rules._column_batch import exceeds_missing_fraction
from qavf.rules._column_contract import ColumnOutcome, ColumnPlan
from qavf.rules._evidence_columns import EvidenceColumns
from qavf.rules._predicates import (
    compatibility_error,
    is_nan,
    row_tests,
)
from qavf.rules.helpers import (
    missing_expr,
)
from qavf.settings import (
    ROW_CONSTRAINTS,
    ColumnCheck,
    SchemaSpec,
)
from qavf.value_policy import normalized_text

from . import _column_values as values

logger = logging.getLogger(__name__)


def evaluate_eager(
    ctx: DataContext,
    dataset: str,
    schema: SchemaSpec,
    frame: pl.DataFrame | None = None,
) -> tuple[list[Finding], list[str]]:
    """Assess ``schema`` on ``frame`` (by default the whole dataset)."""
    frame = ctx.get(dataset) if frame is None else frame
    rows = frame.height
    failures: list[Finding] = []
    passed: list[str] = []
    evidence = EvidenceColumns(schema, tuple(frame.columns))

    # Missing-value policy: NaN counts as missing, exactly like null, for every
    # limit. ``prepared`` therefore replaces NaN with null in float columns, and
    # all checks except the explicit NaN handling below run on it.
    required_columns = set(schema.columns)
    for spec in schema.columns.values():
        if spec.unit_column:
            required_columns.add(spec.unit_column)
        for check in spec.all_checks():
            if check.where:
                required_columns.update(pl.sql_expr(check.where).meta.root_names())
    float_columns = [
        c for c in required_columns if c in frame.columns and frame[c].dtype.is_float()
    ]
    normalizers: list[pl.Expr] = []
    blocked_columns: dict[str, str] = {}
    empty = frame.clear()
    for column in required_columns & set(frame.columns):
        try:
            marker = missing_expr(frame, column, schema.columns.get(column))
            normalizer = pl.when(marker).then(None).otherwise(pl.col(column)).alias(column)
            # Validate each dependency without data. One incompatible marker must
            # not suppress independently assessable sibling contracts.
            empty.select(normalizer)
            normalizers.append(normalizer)
        except Exception as exc:
            blocked_columns[column] = f"{type(exc).__name__}: {exc}"
    prepared = frame.with_columns(normalizers)

    def fail(column: str, label: str, check: ColumnCheck | None, message: str, **kw: Any) -> None:
        failures.append(
            Finding.fail(
                message,
                check=f"{dataset}.{column}: {label}",
                severity=check.severity if check else None,
                datasets=(dataset,),
                **kw,
            )
        )

    def blocked(column: str, label: str, check: ColumnCheck, message: str) -> None:
        failures.append(
            Finding.unassessed(
                message,
                check=f"{dataset}.{column}: {label}",
                severity=check.severity,
                datasets=(dataset,),
            )
        )

    for column, spec in schema.columns.items():
        if column not in frame.columns:
            if spec.required:
                failures.append(values._missing_column(dataset, column, frame.columns))
            continue
        if column in blocked_columns:
            failures.append(
                Finding.unassessed(
                    f"Not assessed: {dataset}.{column}: {blocked_columns[column]}.",
                    check=f"{dataset}.{column}",
                    datasets=(dataset,),
                )
            )
            continue
        is_float = column in float_columns
        series = prepared[column]
        missing = series.null_count()
        missing_mask = series.is_null()  # null, NaN, configured sentinels and blanks

        if not spec.nullable:
            if missing:
                fail(
                    column,
                    "not null",
                    None,
                    column_values_violate(missing, rows, f"{dataset}.{column}", "not null"),
                    affected=values._rows(
                        frame, missing_mask, "missing", ctx.evidence_limit, evidence.of([column])
                    ),
                    affected_count=missing,
                    total_records=rows,
                    unit="rows",
                )
            else:
                passed.append(f"{dataset}.{column}: not null")
        if is_float and not spec.allow_infinite:
            infinite = int(series.is_infinite().sum())
            if infinite:
                fail(
                    column,
                    "finite",
                    None,
                    column_values_violate(infinite, rows, f"{dataset}.{column}", "finite"),
                    affected=values._rows(
                        frame,
                        pl.col(column).is_infinite(),
                        "infinite",
                        ctx.evidence_limit,
                        evidence.of([column]),
                    ),
                    affected_count=infinite,
                    total_records=rows,
                    unit="rows",
                )
            else:
                passed.append(f"{dataset}.{column}: finite")

        if spec.unit_column is not None:
            if spec.unit_column in blocked_columns:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: units on {dataset}.{column}: "
                        f"{blocked_columns[spec.unit_column]}.",
                        check=f"{dataset}.{column}: units",
                        datasets=(dataset,),
                    )
                )
            else:
                try:
                    _units(
                        dataset, column, spec, frame, rows, failures, passed, ctx, prepared,
                        evidence.of(ColumnPlan(column, spec, "units").dependencies),
                    )  # fmt: skip
                except Exception as exc:
                    failures.append(
                        Finding.unassessed(
                            f"Not assessed: units on {dataset}.{column}: "
                            f"{type(exc).__name__}: {exc}.",
                            check=f"{dataset}.{column}: units",
                            datasets=(dataset,),
                        )
                    )

        for check in spec.all_checks():
            label = check_label(check)
            constraints = check.constraints()
            problems: list[str] = []
            # ``problems`` are violations found in the data; ``unusable`` explains why
            # (part of) the check could not be evaluated, which is never a violation.
            unusable: list[str] = []
            # ``where`` limits the check to some rows; everything below uses them only.
            applicable: pl.Series | None = None
            scope, scope_missing, scope_rows = series, missing, rows
            if check.where is not None:
                try:
                    for dependency in pl.sql_expr(check.where).meta.root_names():
                        if dependency in blocked_columns:
                            raise ValueError(blocked_columns[dependency])
                    applicable = prepared.select(
                        pl.sql_expr(check.where).fill_null(False).alias("m")
                    )["m"]
                    scope = series.filter(applicable)
                    scope_missing = int(missing_mask.filter(applicable).sum())
                    scope_rows = scope.len()
                except Exception as exc:
                    unusable.append(f"where condition failed: {type(exc).__name__}: {exc}")
                else:
                    problem = ColumnOutcome(dataset, column, label, check).applicability_problem(
                        scope_rows
                    )
                    if problem:
                        unusable.append(problem)
            # Column-level limits are evaluated directly, on exact native values.
            # Missing rows are evidence when values must be present or are too often absent.
            missing_rows = check.not_null
            try:
                if check.max_null_fraction is not None and exceeds_missing_fraction(
                    check, scope_missing, scope_rows
                ):
                    missing_rows = True
                    problems.append(
                        missing_fraction_problem(scope_missing, scope_rows, check.max_null_fraction)
                    )
                for statistic in (
                    "min",
                    "max",
                    "mean",
                    "median",
                    "std",
                    "sum",
                    "distinct_count",
                ):
                    bounds = constraints.get(statistic)
                    if bounds is None:
                        continue
                    if statistic != "distinct_count" and not series.dtype.is_numeric():
                        unusable.append(
                            f"{statistic} requires a numeric column, found {series.dtype}"
                        )
                        continue
                    value = values._statistic(scope, statistic)
                    if value is None:
                        unusable.append(f"{statistic} is undefined: insufficient usable values")
                        continue
                    if not values._within(value, bounds):
                        problems.append(outside_bounds(statistic, value, bounds))
            except Exception as exc:
                # Isolated to this check: sibling checks are still evaluated.
                logger.warning(
                    "%s",
                    redact_text(
                        f"Check {dataset}.{column} {label!r} failed: {exception_details(exc)}"
                    ),
                )
                unusable.append(f"evaluation failed: {type(exc).__name__}: {exc}")
            predicates: list[tuple[str, pl.Expr]] = []
            if is_float and any(is_nan(v) for v in constraints.get("notin") or []):
                predicates.append(("notin NaN", pl.lit(frame[column].is_nan())))
            for name, raw in constraints.items():
                if name not in ROW_CONSTRAINTS:
                    continue
                try:
                    error = compatibility_error(name, raw, series.dtype)
                    if error:
                        unusable.append(error)
                        continue
                    row_label, test = row_tests(
                        {name: raw}, series.dtype, normalize=check.normalize
                    )[name]
                    bad = pl.col(column).is_not_null() & ~test(column).fill_null(True)
                    predicates.append((row_label, bad))
                except Exception as exc:
                    unusable.append(f"{name}: {type(exc).__name__}: {exc}")
            if check.unique:
                try:
                    predicates.append(("unique", _duplicated(column, series.dtype, check)))
                except Exception as exc:
                    unusable.append(f"unique: {type(exc).__name__}: {exc}")
            if missing_rows:
                predicates.append(("missing", pl.lit(missing_mask)))
            if unusable:
                found = f" Violations found meanwhile: {'; '.join(problems)}." if problems else ""
                blocked(
                    column,
                    label,
                    check,
                    check_not_assessed(label, dataset, column, "; ".join(unusable)) + found,
                )
                continue
            if applicable is not None:
                predicates = [(name, bad & pl.lit(applicable)) for name, bad in predicates]
            try:
                count, affected = values._row_limits(
                    frame,
                    prepared,
                    predicates,
                    ctx.evidence_limit,
                    evidence.of(ColumnPlan(column, spec, "check", check).dependencies),
                )
            except Exception as exc:
                blocked(
                    column,
                    label,
                    check,
                    f"Not assessed: '{label}' on {dataset}.{column} could not be evaluated: "
                    f"{type(exc).__name__}: {exc}.",
                )
                continue
            finding = ColumnOutcome(dataset, column, label, check).finding(
                count, scope_rows, problems, affected
            )
            if finding is not None:
                failures.append(finding)
            else:
                passed.append(f"{dataset}.{column}: {label}")
    return failures, passed


def _duplicated(column: str, dtype: pl.DataType, check: ColumnCheck) -> pl.Expr:
    """Rows whose value occurs more than once; missing values are never duplicates.

    Callers restrict the result to the check's ``where`` rows, so only values in
    scope are compared with each other.
    """
    value = normalized_text(pl.col(column), check.normalize, dtype)
    if check.where is not None:
        value = pl.when(pl.sql_expr(check.where).fill_null(False)).then(value)
    return value.is_not_null() & value.is_duplicated()


def _units(
    dataset: str,
    column: str,
    spec: Any,
    frame: pl.DataFrame,
    rows: int,
    failures: list[Finding],
    passed: list[str],
    ctx: DataContext,
    prepared: pl.DataFrame,
    shown: list[str],
) -> None:
    """Every row with a value carries an allowed unit in ``spec.unit_column``."""
    check = f"{dataset}.{column}: units"
    unit_column = spec.unit_column
    allowed = list(spec.allowed_units or [spec.unit])
    if unit_column not in frame.columns:
        failures.append(
            Finding.unassessed(
                unit_column_absent(unit_column, dataset, column),
                check=check,
                datasets=(dataset,),
            )
        )
        return
    units = pl.col(unit_column).cast(pl.String)
    wrong = pl.lit(prepared[column].is_not_null()) & (units.is_null() | ~units.is_in(allowed))
    mask = prepared.select(wrong.alias("wrong"))["wrong"]
    count = int(mask.sum())
    if not count:
        passed.append(check)
        return
    failures.append(
        Finding.fail(
            column_values_violate(count, rows, f"{dataset}.{column}", "units"),
            check=check,
            affected=values._rows(frame, mask, "unit", ctx.evidence_limit, shown),
            affected_count=count,
            total_records=rows,
            unit="rows",
            datasets=(dataset,),
        )
    )
