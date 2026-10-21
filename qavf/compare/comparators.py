"""Comparators turn a baseline and a candidate value into a difference.

Values arrive as exact rationals (integers, decimals, stored binary floats and
rational strings all convert without rounding), so ``diff`` of two large
integers is exact. Comparators return a rational or a float, or ``None`` when
the difference is undefined (a ratio to zero). Whether a change matters is a
separate judgement, :class:`Materiality`, made with the package's exact
tolerance arithmetic. Statistical significance is a later, separate judgement;
results carry a typed slot for it.

Writing a comparator::

    @register_comparator
    class PercentChange(Comparator):
        name = "pct_change"

        def compare(self, baseline, candidate, context):
            return (candidate - baseline) / abs(baseline) * 100 if baseline else None
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, ClassVar

from qavf.compare.registry import Component, ComponentRegistry
from qavf.numeric import exact_rational, numeric_close

Value = Fraction | float | None


@dataclass(frozen=True, slots=True)
class ComparatorContext:
    """Side-wide facts some comparators need, e.g. totals for share changes."""

    baseline_total: Fraction | None = None
    candidate_total: Fraction | None = None


class Comparator(Component):
    """Map ``(baseline, candidate)`` exact values to a difference."""

    @abstractmethod
    def compare(self, baseline: Fraction, candidate: Fraction, context: ComparatorContext) -> Value:
        """The difference, or ``None`` where it is undefined."""


COMPARATORS: ComponentRegistry[Comparator] = ComponentRegistry(
    "comparator", "qavf.compare.comparators", Comparator, builtins=(__name__,)
)
register_comparator = COMPARATORS.register


@register_comparator
class Diff(Comparator):
    """``candidate - baseline``."""

    name: ClassVar[str] = "diff"

    def compare(self, baseline: Fraction, candidate: Fraction, context: ComparatorContext) -> Value:
        return candidate - baseline


@register_comparator
class AbsDiff(Comparator):
    """``|candidate - baseline|``."""

    name: ClassVar[str] = "abs_diff"

    def compare(self, baseline: Fraction, candidate: Fraction, context: ComparatorContext) -> Value:
        return abs(candidate - baseline)


@register_comparator
class RelDiff(Comparator):
    """``(candidate - baseline) / |baseline|``; undefined for a zero baseline."""

    name: ClassVar[str] = "rel_diff"

    def compare(self, baseline: Fraction, candidate: Fraction, context: ComparatorContext) -> Value:
        return (candidate - baseline) / abs(baseline) if baseline else None


@register_comparator
class Ratio(Comparator):
    """``candidate / baseline``; undefined for a zero baseline."""

    name: ClassVar[str] = "ratio"

    def compare(self, baseline: Fraction, candidate: Fraction, context: ComparatorContext) -> Value:
        return candidate / baseline if baseline else None


@register_comparator
class LogRatio(Comparator):
    """``ln(candidate / baseline)``; defined when both are positive."""

    name: ClassVar[str] = "log_ratio"

    def compare(self, baseline: Fraction, candidate: Fraction, context: ComparatorContext) -> Value:
        if baseline <= 0 or candidate <= 0:
            return None
        # Logarithms of numerator and denominator separately avoid float overflow.
        return (
            math.log(candidate.numerator)
            - math.log(candidate.denominator)
            - math.log(baseline.numerator)
            + math.log(baseline.denominator)
        )


@register_comparator
class ShareChange(Comparator):
    """Change of the value's share of its side's total (``c/C - b/B``)."""

    name: ClassVar[str] = "share_change"

    def compare(self, baseline: Fraction, candidate: Fraction, context: ComparatorContext) -> Value:
        if not context.baseline_total or not context.candidate_total:
            return None
        return candidate / context.candidate_total - baseline / context.baseline_total


@dataclass(frozen=True, slots=True)
class Materiality:
    """Whether a change matters: beyond ``max(atol, rtol * max(|b|, |c|))``.

    Numbers are compared exactly (:func:`qavf.numeric.numeric_close`); other
    values (text, dates) by equality. A value present on one side only is
    always material.
    """

    rtol: float = 0.0
    atol: float = 0.0

    def material(self, baseline: Any, candidate: Any) -> bool:
        if baseline is None and candidate is None:
            return False
        if baseline is None or candidate is None:
            return True
        b, c = exact_rational(baseline), exact_rational(candidate)
        if b is None or c is None:
            nan = [isinstance(v, float) and math.isnan(v) for v in (baseline, candidate)]
            return not all(nan) and bool(baseline != candidate)
        return not numeric_close(b, c, self.rtol, self.atol)
