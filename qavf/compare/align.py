"""The alignment pipeline: ordered steps that make two datasets comparable.

Each step transforms one side's lazy relation and may refuse it. A refusal is an
:class:`AlignmentError`, which makes the comparison not assessed: a lossy cast,
an absent mapped column or an unmapped category is never coerced silently.
Steps that need a data proof (no value lost) ask the context to count the
offending rows natively; no row-level Python state is retained.

Writing a step::

    @register_alignment_step
    class Trim(AlignmentStep):
        name = "trim"
        Params = TrimParams          # RuleParams subclass: columns: list[str]

        def apply(self, plan, context):
            return plan.with_columns(pl.col(c).str.strip_chars() for c in self.params.columns)

It is then configured as ``steps: [{step: trim, columns: [name]}]``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl
from pydantic import Field

from qavf.compare.registry import Component, ComponentRegistry
from qavf.compare.spec import CastSpec, ComparisonSpec, Recode, Side, UnitConversion
from qavf.engine.registry import RuleParams
from qavf.regression._plan import mapped_schema
from qavf.typed_values import checked_cast, exact_scale, logical_type


class AlignmentError(ValueError):
    """The datasets cannot be aligned as configured; the comparison is not assessed."""


@dataclass
class AlignmentContext:
    """What a step may know about the side it transforms."""

    side: Side
    dataset: str
    count: Callable[[pl.LazyFrame], int]
    """Count the rows of a plan derived from this side (evaluated natively)."""
    units: dict[str, str | None] = field(default_factory=dict)
    """Declared column units under current names; steps keep it up to date."""
    nullable: dict[str, bool] = field(default_factory=dict)
    """Declared ``nullable`` flags under current names."""

    def refuse(self, message: str) -> AlignmentError:
        return AlignmentError(f"{self.side} {self.dataset!r}: {message}")

    def require_none(self, plan: pl.LazyFrame, predicate: pl.Expr, message: str) -> None:
        """Refuse when any row matches ``predicate``; ``message`` takes the count."""
        found = self.count(plan.filter(predicate))
        if found:
            raise self.refuse(message.format(count=f"{found:,}"))

    def rename(self, mapping: Mapping[str, str]) -> None:
        """Carry declared metadata over a column rename."""
        for old, new in mapping.items():
            if old in self.units:
                self.units[new] = self.units.pop(old)
            if old in self.nullable:
                self.nullable[new] = self.nullable.pop(old)


class AlignmentStep(Component):
    """One transformation of a side's relation; ``apply`` must be side-effect free."""

    sides: ClassVar[tuple[Side, ...]] = ("baseline", "candidate")
    """Sides the step applies to; ``apply`` is not called for the others."""

    @abstractmethod
    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        """Return the transformed relation, or raise :class:`AlignmentError`."""

    def describe(self) -> str:
        """One line for the comparison's alignment record."""
        options = self.params.model_dump(exclude_defaults=True)
        return f"{self.name}: {options}" if options else self.name


ALIGNMENT_STEPS: ComponentRegistry[AlignmentStep] = ComponentRegistry(
    "alignment step", "qavf.compare.alignment", AlignmentStep, builtins=(__name__,)
)
register_alignment_step = ALIGNMENT_STEPS.register


def _require_columns(plan: pl.LazyFrame, columns: Sequence[str], context: AlignmentContext) -> None:
    if absent := [column for column in columns if column not in plan.collect_schema()]:
        raise context.refuse(f"lacks column(s) {', '.join(absent)}")


def _applies(selection: str, side: Side) -> bool:
    return selection in ("both", side)


# ----------------------------------------------------------------- built-ins
class ColumnMapParams(RuleParams):
    columns: dict[str, str] = Field(default_factory=dict)
    """Baseline name -> candidate name."""


@register_alignment_step
class ColumnMapStep(AlignmentStep):
    """Rename baseline columns to their candidate names."""

    name: ClassVar[str] = "columns"
    description: ClassVar[str] = "Baseline columns take their candidate names."
    sides: ClassVar[tuple[Side, ...]] = ("baseline",)
    Params: ClassVar[type[RuleParams]] = ColumnMapParams
    params: ColumnMapParams

    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        try:
            _, mapping = mapped_schema(plan.collect_schema(), self.params.columns, "columns")
        except ValueError as exc:
            raise context.refuse(str(exc).removeprefix("columns: ")) from None
        context.rename(mapping)
        return plan.rename(mapping)


class CastParams(RuleParams):
    columns: dict[str, CastSpec] = Field(default_factory=dict)


@register_alignment_step
class CastStep(AlignmentStep):
    """Convert column types losslessly; any changed value refuses the alignment."""

    name: ClassVar[str] = "cast"
    description: ClassVar[str] = "Lossless type harmonization."
    Params: ClassVar[type[RuleParams]] = CastParams
    params: CastParams

    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        selected = {c: s for c, s in self.params.columns.items() if _applies(s.side, context.side)}
        _require_columns(plan, list(selected), context)
        schema = plan.collect_schema()
        converted, lost = [], []
        for column, spec in selected.items():
            target = logical_type(spec.dtype, spec.scale)
            value, flag = checked_cast(pl.col(column), schema[column], target)
            converted.append(value.alias(column))
            lost.append((column, spec, flag))
        for column, spec, flag in lost:
            context.require_none(
                plan,
                flag,
                f"casting {column!r} to {spec.dtype} would change {{count}} value(s); "
                "harmonize the type in its source or choose a type that holds every value",
            )
        return plan.with_columns(converted) if converted else plan


class UnitParams(RuleParams):
    columns: dict[str, UnitConversion] = Field(default_factory=dict)


@register_alignment_step
class UnitStep(AlignmentStep):
    """Scale columns by an exact factor (unit or scale conversion)."""

    name: ClassVar[str] = "units"
    description: ClassVar[str] = "Unit and scale conversion by an exact factor."
    Params: ClassVar[type[RuleParams]] = UnitParams
    params: UnitParams

    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        selected = {c: s for c, s in self.params.columns.items() if s.side == context.side}
        _require_columns(plan, list(selected), context)
        schema = plan.collect_schema()
        scaled = []
        for column, spec in selected.items():
            declared = context.units.get(column)
            if spec.from_unit is not None and declared is not None and declared != spec.from_unit:
                raise context.refuse(
                    f"{column!r} is declared in {declared!r}, not {spec.from_unit!r}"
                )
            try:
                scaled.append(
                    exact_scale(pl.col(column), schema[column], spec.factor).alias(column)
                )
            except ValueError as exc:
                raise context.refuse(f"cannot scale {column!r}: {exc}") from None
            context.units[column] = spec.to_unit
        return plan.with_columns(scaled) if scaled else plan


class RecodeParams(RuleParams):
    columns: dict[str, Recode] = Field(default_factory=dict)


def _typed(values: Sequence[Any], dtype: pl.DataType, what: str) -> pl.Series:
    try:
        return pl.Series(list(values)).cast(dtype, strict=True)
    except Exception:
        raise ValueError(f"{what} cannot all be represented as {dtype}") from None


@register_alignment_step
class RecodeStep(AlignmentStep):
    """Replace categorical values through an explicit map."""

    name: ClassVar[str] = "recode"
    description: ClassVar[str] = "Categorical recoding through a value map."
    Params: ClassVar[type[RuleParams]] = RecodeParams
    params: RecodeParams

    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        selected = {c: s for c, s in self.params.columns.items() if _applies(s.side, context.side)}
        _require_columns(plan, list(selected), context)
        schema = plan.collect_schema()
        recoded = []
        for column, spec in selected.items():
            dtype = schema[column]
            pairs = spec.pairs()
            try:
                old = _typed([pair[0] for pair in pairs], dtype, f"recode values of {column!r}")
            except ValueError as exc:
                raise context.refuse(str(exc)) from None
            new = pl.Series([pair[1] for pair in pairs])
            target = new.dtype if new.dtype != pl.Null else dtype
            value = pl.col(column)
            unmatched = value.is_not_null() & ~value.is_in(old.implode())
            mapped = value.replace_strict(old, new, default=None, return_dtype=target)
            if spec.unmapped == "error":
                context.require_none(
                    plan, unmatched, f"{{count}} value(s) of {column!r} have no recode mapping"
                )
            if spec.unmapped == "keep":
                kept, lost = checked_cast(value, dtype, target)
                context.require_none(
                    plan,
                    unmatched & lost,
                    f"{{count}} unmapped value(s) of {column!r} cannot be kept as {target}",
                )
                mapped = pl.when(unmatched).then(kept).otherwise(mapped)
            recoded.append(mapped.alias(column))
        return plan.with_columns(recoded) if recoded else plan


class DeriveParams(RuleParams):
    columns: dict[str, str] = Field(default_factory=dict)
    """New column name -> Polars SQL expression."""


@register_alignment_step
class DeriveStep(AlignmentStep):
    """Compute the same new columns on both sides."""

    name: ClassVar[str] = "derive"
    description: ClassVar[str] = "Derived columns from one expression on both sides."
    Params: ClassVar[type[RuleParams]] = DeriveParams
    params: DeriveParams

    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        schema = plan.collect_schema()
        if clash := [name for name in self.params.columns if name in schema]:
            raise context.refuse(f"derived column(s) {', '.join(clash)} already exist")
        return plan.with_columns(
            pl.sql_expr(expression).alias(name) for name, expression in self.params.columns.items()
        )


class FilterParams(RuleParams):
    baseline_where: str | None = None
    candidate_where: str | None = None
    where: str | None = None


@register_alignment_step
class FilterStep(AlignmentStep):
    """Keep the rows in scope: a per-side condition and a shared one."""

    name: ClassVar[str] = "filter"
    description: ClassVar[str] = "Per-side and shared row filters."
    Params: ClassVar[type[RuleParams]] = FilterParams
    params: FilterParams

    def apply(self, plan: pl.LazyFrame, context: AlignmentContext) -> pl.LazyFrame:
        own = (
            self.params.baseline_where
            if context.side == "baseline"
            else self.params.candidate_where
        )
        for condition in (own, self.params.where):
            if condition is not None:
                plan = plan.filter(pl.sql_expr(condition))
        return plan


def build_pipeline(spec: ComparisonSpec) -> list[AlignmentStep]:
    """The configured steps in their documented order."""
    configured: list[tuple[str, dict[str, Any]]] = []
    for name, value in (
        ("columns", spec.columns),
        ("cast", spec.cast),
        ("units", spec.units),
        ("recode", spec.recode),
        ("derive", spec.derive),
    ):
        if value:
            configured.append((name, {"columns": value}))
    configured.extend((step.step, step.params) for step in spec.steps)
    filters = {
        "baseline_where": spec.baseline_where,
        "candidate_where": spec.candidate_where,
        "where": spec.where,
    }
    if any(value is not None for value in filters.values()):
        configured.append(("filter", {k: v for k, v in filters.items() if v is not None}))
    return [ALIGNMENT_STEPS.create(name, params) for name, params in configured]


def align(
    plan: pl.LazyFrame, steps: Sequence[AlignmentStep], context: AlignmentContext
) -> pl.LazyFrame:
    """Apply each step meant for the context's side, in order."""
    for step in steps:
        if context.side in step.sides:
            plan = step.apply(plan, context)
    return plan
