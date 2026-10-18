"""Freshness: the newest data is recent enough, relative to the run (or a given time)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

import polars as pl
from pydantic import Field

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.rules._temporal import duration_seconds, temporal_scalar, temporal_ticks, tick_seconds
from qavf.rules.helpers import missing_expr, outcome, plural, usable
from qavf.settings import QAVFSettings
from qavf.sources.sql import quote_identifier

CATEGORY = "input"


class FreshnessCheck(RuleParams):
    """One dataset column whose newest value must be recent."""

    dataset: str
    column: str
    """A date or datetime column, e.g. a timestamp or reporting date."""
    max_age: timedelta = Field(gt=timedelta(0))
    """Largest acceptable age of the newest value, as an ISO 8601 duration (``P2D``,
    ``PT6H``) or seconds."""
    max_future_skew: timedelta = Field(default=timedelta(0), ge=timedelta(0))
    """Largest allowed timestamp lead over the reference clock; defaults to zero."""
    name: str | None = None


class FreshnessParams(RuleParams):
    """Parameters for :class:`FreshnessRule`."""

    checks: tuple[FreshnessCheck, ...] = ()
    """Columns to check; the rule skips until one is configured."""
    now: datetime | None = None
    """The reference time; the start of the run when omitted. Set it to re-check a past
    run reproducibly. A time without a time zone is UTC."""


def _as_utc(value: date | datetime) -> datetime:
    """Dates are midnight UTC; datetimes without a time zone are UTC."""
    if not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class FreshnessRule(Rule):
    """The newest value of each configured date/time column is recent enough."""

    rule_id: ClassVar[str] = "INP-008"
    name: ClassVar[str] = "Data freshness"
    description: ClassVar[str] = (
        "The newest value of each configured date or datetime column is no older than its "
        "max_age, measured from the run (or a configured reference time)."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Stale data usually means an upstream job did not run or a file was not refreshed. "
        "Check the producing pipeline's last run before trusting downstream results."
    )
    Params: ClassVar[type[RuleParams]] = FreshnessParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        return DataRequirements(frozenset(check.dataset for check in self.params.checks))

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        p = self.params
        now = _as_utc(p.now) if p.now is not None else ctx.reference_time
        failures: list[Finding] = []
        passed: list[str] = []
        for check in p.checks:
            try:
                label = check.name or f"{check.dataset}.{check.column}"
                if reason := usable(ctx, check.dataset, (check.column,)):
                    failures.append(
                        Finding.unassessed(
                            f"Not assessed: {label}: {reason}.",
                            check=label,
                            datasets=(check.dataset,),
                        )
                    )
                    continue
                dtype = ctx.schema(check.dataset)[check.column]
                if not (dtype == pl.Date or isinstance(dtype, pl.Datetime)):
                    failures.append(
                        Finding.unassessed(
                            f"Not assessed: {label}: freshness needs a date or datetime column, "
                            f"found {dtype}.",
                            check=label,
                            datasets=(check.dataset,),
                        )
                    )
                    continue
                try:
                    spec = ctx.specs.get(check.dataset)
                    column_spec = (
                        spec.schema_spec.columns.get(check.column)
                        if spec is not None and spec.schema_spec is not None
                        else None
                    )
                    missing = missing_expr(
                        pl.DataFrame(schema={check.column: dtype}), check.column, column_spec
                    )
                    kind = ctx.execution_kind(check.dataset)
                    reduction = pl.col(check.column).filter(~missing).cast(pl.Int64).max()
                    if kind == "native_sql":
                        quoted = quote_identifier(check.column)
                        physical = (
                            f"epoch_{dtype.time_unit}({quoted})"
                            if isinstance(dtype, pl.Datetime)
                            else f"date_diff('day', DATE '1970-01-01', {quoted})"
                        )
                        projection = ctx.sql_missing_projection(
                            check.dataset, {check.column: dtype}
                        )
                        latest = ctx.sql(
                            f"SELECT max({physical}) AS latest "
                            f"FROM (SELECT {projection} FROM {quote_identifier(check.dataset)}) "
                            "AS normalized",
                            datasets=(check.dataset,),
                        ).item()
                    else:
                        def select_latest(
                            frame: pl.LazyFrame, expr: pl.Expr = reduction
                        ) -> pl.LazyFrame:
                            return frame.select(expr)

                        latest = ctx.reduce(check.dataset, select_latest).item()
                except Exception as exc:
                    failures.append(
                        Finding.unassessed(
                            f"Not assessed: {label}: {type(exc).__name__}: {exc}.",
                            check=label,
                            datasets=(check.dataset,),
                        )
                    )
                    continue
                if latest is None:
                    failures.append(
                        Finding.unassessed(
                            f"Not assessed: {label}: the column has no values.",
                            check=label,
                            datasets=(check.dataset,),
                        )
                    )
                    continue
                age = (temporal_ticks(now, dtype) - int(latest)) * tick_seconds(dtype)
                latest_value = temporal_scalar(int(latest), dtype)
                record = {
                    "dataset": check.dataset,
                    "column": check.column,
                    "latest": latest_value,
                    "reference_time": now,
                    "age_hours": round(float(age) / 3600, 3),
                    "max_age_hours": round(check.max_age.total_seconds() / 3600, 3),
                }
                if age < -duration_seconds(check.max_future_skew):
                    failures.append(
                        Finding.fail(
                            f"{label}: the newest value ({latest_value}) is ahead of the reference "
                            f"time by {float(-age):g} seconds, more than allowed "
                            f"{check.max_future_skew}.",
                            check=label,
                            affected=[record],
                            unit="columns",
                            datasets=(check.dataset,),
                        )
                    )
                elif age > duration_seconds(check.max_age):
                    failures.append(
                        Finding.fail(
                            f"{label}: the newest value ({latest_value}) is {float(age):g} seconds "
                            "old, more than the "
                            f"allowed {check.max_age}.",
                            check=label,
                            affected=[record],
                            unit="columns",
                            datasets=(check.dataset,),
                        )
                    )
                else:
                    passed.append(label)
            except Exception as exc:
                failures.append(
                    Finding.unassessed(
                        f"Not assessed: {check.dataset}.{check.column}: "
                        f"{type(exc).__name__}: {exc}.",
                        check=check.name or f"{check.dataset}.{check.column}",
                        datasets=(check.dataset,),
                    )
                )
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), 'column')} fresh.",
            "No freshness checks are configured.",
        )
