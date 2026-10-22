"""Configuration of one dataset comparison, validated when the configuration loads.

Column references after ``columns`` (the baseline-to-candidate name map) use
candidate names: keys, casts, unit conversions, recodes, derived columns,
filters, tolerances and every level's parameters. Named components (levels,
measures, comparators, alignment steps) are looked up in their registries and
their parameters validated here, so a typo fails before any data is read.
"""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, PrivateAttr, field_validator, model_validator

from qavf.engine.registry import RuleParams
from qavf.engine.severity import Severity
from qavf.inference.decision import DEFAULT_ALPHA, DEFAULT_POLICY
from qavf.inference.resampling import DEFAULT_MAX_WORK, DEFAULT_SEED
from qavf.inference.results import Correction, DecisionPolicy
from qavf.rules.regression_checks import ColumnTolerance
from qavf.settings import DType, NumericBound, _check_where

Side = Literal["baseline", "candidate"]
SideSelection = Literal["baseline", "candidate", "both"]

DEFAULT_LEVELS = ("structure", "coverage", "records")
"""Levels compared when ``levels`` is omitted (``structure`` alone without keys)."""


def _repeated(values: list[str] | tuple[str, ...]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


class CastSpec(RuleParams):
    """A lossless type conversion of one column (``{dtype: int}`` or just ``int``)."""

    dtype: DType
    scale: int | None = Field(default=None, ge=0, le=38)
    """Decimal places for ``dtype: decimal``."""
    side: SideSelection = "both"
    """Which dataset's column is converted."""

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        return {"dtype": data} if isinstance(data, str) else data

    @model_validator(mode="after")
    def _scale(self) -> CastSpec:
        if (self.dtype == "decimal") != (self.scale is not None):
            raise ValueError("'scale' is required for, and only used by, dtype: decimal")
        return self


class UnitConversion(RuleParams):
    """Multiply one side's column by ``factor`` so both sides share a unit."""

    factor: NumericBound
    """Exact for integers and decimals; a float factor means its decimal spelling."""
    side: Side = "baseline"
    from_unit: str | None = None
    """The unit the column must be declared in (``schema.columns.<c>.unit``), if declared."""
    to_unit: str | None = None
    """The unit after conversion, compared with the other side's declared unit."""

    @model_validator(mode="after")
    def _factor(self) -> UnitConversion:
        value = self.factor
        finite = value.is_finite() if isinstance(value, Decimal) else math.isfinite(value)
        if not finite or not value:
            raise ValueError("'factor' must be a finite nonzero number")
        return self


class Recode(RuleParams):
    """Replace categorical values, e.g. codes renamed between versions."""

    values: dict[str, Any] | list[tuple[Any, Any]]
    """``old: new`` (mapping keys are text; they are converted to the column type)
    or a list of ``[old, new]`` pairs for typed old values."""
    side: SideSelection = "baseline"
    unmapped: Literal["keep", "null", "error"] = "keep"
    """Values without a mapping: kept (when losslessly representable), made null,
    or reported (the comparison is then not assessed)."""

    @model_validator(mode="after")
    def _pairs(self) -> Recode:
        if not self.values:
            raise ValueError("'values' must map at least one value")
        olds = [old for old, _ in self.pairs()]
        if len({repr(old) for old in olds}) != len(olds):
            raise ValueError("'values' maps the same old value more than once")
        return self

    def pairs(self) -> list[tuple[Any, Any]]:
        """``(old, new)`` pairs in configuration order."""
        if isinstance(self.values, dict):
            return list(self.values.items())
        return [tuple(pair) for pair in self.values]


class _Configured(RuleParams):
    """A named component whose own parameters may be written inline."""

    @model_validator(mode="before")
    @classmethod
    def _inline(cls, data: Any) -> Any:
        selector = cls._selector()
        if isinstance(data, str):
            return {selector: data}
        if isinstance(data, dict) and "params" not in data:
            own = set(cls.model_fields) - {"params"}
            data = {
                **{key: value for key, value in data.items() if key in own},
                "params": {key: value for key, value in data.items() if key not in own},
            }
        return data

    @classmethod
    def _selector(cls) -> str:
        raise NotImplementedError


class StepConfig(_Configured):
    """A plugin (or built-in) alignment step: ``{step: name, ...its parameters}``."""

    step: str
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _selector(cls) -> str:
        return "step"

    @model_validator(mode="after")
    def _known(self) -> StepConfig:
        from qavf.compare.align import ALIGNMENT_STEPS

        ALIGNMENT_STEPS.create(self.step, self.params)
        return self


class TestConfig(_Configured):
    """A significance test: ``welch_t``, ``{test: bootstrap, resamples: 999}`` or ``none``."""

    __test__ = False  # not a pytest test class

    test: str
    """A registered test (``qavf.inference.TESTS``), or ``none``: no test applies."""
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _selector(cls) -> str:
        return "test"

    @model_validator(mode="after")
    def _known(self) -> TestConfig:
        if self.test == "none":
            if self.params:
                raise ValueError("'test: none' takes no parameters")
        else:
            from qavf.inference.registry import TESTS

            TESTS.create(self.test, self.params)
        return self

    def create(self) -> Any:
        """The test component, or ``None`` for ``none``."""
        if self.test == "none":
            return None
        from qavf.inference.registry import TESTS

        return TESTS.create(self.test, self.params)


class InferenceParams(RuleParams):
    """A level's own significance settings; the comparison's when omitted."""

    decision: DecisionPolicy | None = None
    """``material_and_significant`` (default), ``material``, ``significant`` or
    ``equivalence``."""
    alpha: float | None = Field(default=None, gt=0, lt=1)
    correction: Correction | None = None
    """Multiple-testing correction across each check's family of tests."""
    seed: int | None = Field(default=None, ge=0)
    max_work: int | None = Field(default=None, ge=1)


class LevelConfig(_Configured):
    """A comparison level: ``records`` or ``{level: groups, group_by: [...], ...}``."""

    level: str
    name: str | None = None
    """Label in reports; the level name when omitted."""
    severity: Severity | None = None
    """Severity of this level's failures; the rule's when omitted."""
    params: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def _selector(cls) -> str:
        return "level"

    @model_validator(mode="after")
    def _known(self) -> LevelConfig:
        from qavf.compare.levels import LEVELS

        LEVELS.create(self.level, self.params)
        return self

    @property
    def label(self) -> str:
        return self.name or self.level


class ComparisonSpec(RuleParams):
    """Two datasets compared at several levels after an alignment pipeline.

    Alignment runs in this order: ``columns``, ``cast``, ``units``, ``recode``,
    ``derive``, plugin ``steps``, then the ``where`` filters.
    """

    _datasets: tuple[str, ...] = PrivateAttr(default=())

    name: str | None = None
    """Comparison label in reports (``baseline vs candidate`` when omitted)."""
    baseline: str
    """The reference dataset (e.g. the previous version)."""
    candidate: str
    """The dataset compared with it (e.g. the new version)."""
    keys: tuple[str, ...] = ()
    """Columns identifying a record on both sides (candidate names). Coverage and
    record levels need them."""
    columns: dict[str, str] = Field(default_factory=dict)
    """Baseline columns renamed in the candidate (``baseline_name: candidate_name``)."""
    cast: dict[str, CastSpec] = Field(default_factory=dict)
    """Lossless type conversions; a value that would change makes the comparison
    not assessed."""
    units: dict[str, UnitConversion] = Field(default_factory=dict)
    recode: dict[str, Recode] = Field(default_factory=dict)
    derive: dict[str, str] = Field(default_factory=dict)
    """New columns computed on both sides from a Polars SQL expression."""
    steps: tuple[StepConfig, ...] = ()
    """Further alignment steps, e.g. from plugins, applied after ``derive``."""
    baseline_where: str | None = None
    """Compare only baseline rows matching this Polars SQL condition."""
    candidate_where: str | None = None
    where: str | None = None
    """Scope condition applied to both sides."""
    ignore_columns: tuple[str, ...] = ()
    rtol: float = Field(default=1e-6, ge=0)
    """Default relative tolerance for values and measures."""
    atol: float = Field(default=1e-9, ge=0)
    tolerances: dict[str, ColumnTolerance] = Field(default_factory=dict)
    """Per-column tolerances, e.g. ``{share: {atol: 0.001}, code: {exact: true}}``."""
    levels: tuple[LevelConfig, ...] | None = None
    """Levels to compare; ``structure, coverage, records`` (``structure`` without keys)."""
    min_rows: int = Field(default=1, ge=0)
    """Fewest aligned rows on each side for the comparison to be assessed."""
    decision: DecisionPolicy = DEFAULT_POLICY
    """How materiality and significance combine: ``material_and_significant``
    (fail only when a difference is material and significant), ``material``
    (no tests; for a core install without the stats extra), ``significant`` or
    ``equivalence`` (pass only when equivalence within the tolerance is shown)."""
    alpha: float = Field(default=DEFAULT_ALPHA, gt=0, lt=1)
    """Significance level, applied after the correction."""
    correction: Correction = "bh"
    """Multiple-testing correction across each check's family (groups, buckets,
    series, columns): ``bh`` (Benjamini-Hochberg), ``by``, ``holm``,
    ``bonferroni`` or ``none``."""
    seed: int = Field(default=DEFAULT_SEED, ge=0)
    """Seed of resampling tests; the same seed reproduces a comparison exactly."""
    max_work: int = Field(default=DEFAULT_MAX_WORK, ge=1)
    """Largest work (resamples x values, or values collected) of one test; a larger
    test is not assessed rather than slow."""

    @field_validator("derive")
    @classmethod
    def _expressions(cls, value: dict[str, str]) -> dict[str, str]:
        for name, expression in value.items():
            _check_where(expression, f"derived column {name!r} expression")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> ComparisonSpec:
        if repeated := _repeated(list(self.columns.values())):
            raise ValueError(
                f"'columns' gives several baseline columns the name(s) {', '.join(repeated)}"
            )
        if repeated := _repeated(self.keys):
            raise ValueError(f"keys repeat column(s): {', '.join(repeated)}")
        if clash := sorted(set(self.keys) & set(self.ignore_columns)):
            raise ValueError(f"key column(s) {', '.join(clash)} cannot also be ignored")
        if clash := sorted(set(self.tolerances) & (set(self.keys) | set(self.ignore_columns))):
            raise ValueError(
                f"tolerances cannot apply to key or ignored column(s): {', '.join(clash)}"
            )
        for condition in (self.baseline_where, self.candidate_where, self.where):
            if condition is not None:
                _check_where(condition)
        levels = self.selected_levels()
        if repeated := _repeated([level.label for level in levels]):
            raise ValueError(
                f"levels repeat the name(s) {', '.join(repeated)}; give each a distinct 'name'"
            )
        from qavf.compare.align import ALIGNMENT_STEPS
        from qavf.compare.levels import LEVELS

        components = [LEVELS.create(level.level, level.params) for level in levels]
        problems = [
            f"level {level.label!r}: {problem}"
            for level, component in zip(levels, components, strict=True)
            for problem in component.configuration_errors(self)
        ]
        if problems:
            raise ValueError("; ".join(problems))
        named = [self.baseline, self.candidate]
        for step in self.steps:
            named.extend(sorted(ALIGNMENT_STEPS.create(step.step, step.params).datasets()))
        for component in components:
            named.extend(sorted(component.datasets()))
        self._datasets = tuple(dict.fromkeys(named))
        return self

    def selected_levels(self) -> tuple[LevelConfig, ...]:
        """The configured levels, or the defaults for this comparison."""
        if self.levels is not None:
            return self.levels
        names = DEFAULT_LEVELS if self.keys else DEFAULT_LEVELS[:1]
        return tuple(LevelConfig(level=name) for name in names)

    @property
    def label(self) -> str:
        """Report label of this comparison."""
        return self.name or f"{self.baseline} vs {self.candidate}"

    def datasets(self) -> tuple[str, ...]:
        """Every configured dataset the comparison reads: both sides, then the further
        datasets its steps and levels name (zone layers, crosswalks), in order.

        Resolved when the configuration loads, while its plugin components are visible.
        """
        return self._datasets

    def tolerance(self, column: str | None) -> tuple[float, float]:
        """``(rtol, atol)`` for a column: its own tolerance, else the comparison's."""
        override = self.tolerances.get(column) if column is not None else None
        if override is None:
            return self.rtol, self.atol
        if override.exact:
            return 0.0, 0.0
        return (
            self.rtol if override.rtol is None else override.rtol,
            self.atol if override.atol is None else override.atol,
        )
