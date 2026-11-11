"""Rules whose tests a multiple-testing correction judges, alone or in a family.

A :class:`CorrectedRule` evaluates in two steps. :meth:`CorrectedRule.measure`
reads the data and returns its raw tests (:class:`FamilyTests`): the p-values,
the family size they count (untested hypotheses count as ``p = 1``) and what is
needed to judge them. :meth:`FamilyTests.finish` then judges them with a
:class:`FamilyCorrection`. Alone, a rule corrects its own tests
(:meth:`CorrectedRule.evaluate`). In an inference family the runner measures
every member first, corrects all of their p-values together
(:class:`PooledFamily`) and only then lets each member finish.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule
from qavf.inference.multiple import adjust
from qavf.inference.results import Correction


@dataclass(frozen=True, slots=True)
class FamilyCorrection:
    """A member's tests after the correction: adjusted p-values in the order of
    :attr:`FamilyTests.p_values` (``NaN`` for a p-value left out as non-finite)."""

    method: Correction
    adjusted: np.ndarray
    family_size: int
    """The tests the correction counted (every member's, in a family)."""

    @classmethod
    def alone(cls, tests: FamilyTests, method: Correction) -> FamilyCorrection:
        """The member's own correction, as without a family."""
        return cls(method, adjust(tests.p_values, method, tests.family_size), tests.family_size)


class FamilyTests(ABC):
    """One check item's raw tests, awaiting their correction."""

    @property
    @abstractmethod
    def p_values(self) -> np.ndarray:
        """Raw p-values of the tests that ran, in a fixed order."""

    @property
    @abstractmethod
    def family_size(self) -> int:
        """Tests counted by the correction; those that could not run count as ``p = 1``."""

    @abstractmethod
    def finish(self, ctx: DataContext, correction: FamilyCorrection) -> list[Finding]:
        """The item's findings, its tests judged with ``correction``."""

    @property
    def methods(self) -> frozenset[str]:
        """Corrections the tests asked for while measured (none recorded: empty)."""
        return frozenset()

    def close(self) -> None:
        """Release what the tests hold until they are judged."""
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CorrectedRule(Rule):
    """A rule whose verdicts depend on a multiple-testing correction; its check items
    take ``inference: {family: ...}``."""

    def family_corrections(self) -> frozenset[str]:
        """The corrections this instance's tests ask for (empty: it runs no test)."""
        return frozenset({str(self.params.correction)})  # type: ignore[attr-defined]

    def family_method(self) -> Correction:
        """The correction this instance applies alone."""
        method: Correction = self.params.correction  # type: ignore[attr-defined]
        return method

    @abstractmethod
    def measure(self, ctx: DataContext) -> FamilyTests | list[Finding]:
        """The item's raw tests; findings instead when nothing can be tested."""

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        """Measure, then judge the tests with this item's own correction."""
        measured = self.measure(ctx)
        if not isinstance(measured, FamilyTests):
            return measured
        with measured:
            return measured.finish(ctx, FamilyCorrection.alone(measured, self.family_method()))


@dataclass(frozen=True, slots=True)
class PooledFamily:
    """One family's correction over the tests of the members measured in a run."""

    name: str
    method: Correction
    family_size: int
    members: tuple[str, ...]
    """Members whose tests were corrected together, in configuration order."""
    tests: Mapping[str, int]
    """Each member's p-values in the family."""
    non_finite: int = 0
    """P-values left out of the family because they are not finite."""

    @classmethod
    def correct(
        cls, name: str, method: Correction, measured: Sequence[tuple[str, FamilyTests]]
    ) -> tuple[PooledFamily, dict[str, FamilyCorrection]]:
        """The family's correction and each member's share of it, by member id.

        Every member's p-values are corrected as one family of the members' summed
        family sizes; a non-finite p-value is left out (adjusted ``NaN``) and does
        not count.
        """
        arrays = [np.asarray(tests.p_values, dtype=float) for _, tests in measured]
        pooled = np.concatenate(arrays) if arrays else np.empty(0)
        finite = np.isfinite(pooled)
        non_finite = int((~finite).sum())
        size = sum(tests.family_size for _, tests in measured) - non_finite
        adjusted = np.full(len(pooled), math.nan)
        adjusted[finite] = adjust(pooled[finite], method, size)
        corrections: dict[str, FamilyCorrection] = {}
        offset = 0
        for (member, _), values in zip(measured, arrays, strict=True):
            share = adjusted[offset : offset + len(values)]
            corrections[member] = FamilyCorrection(method, share, size)
            offset += len(values)
        family = cls(
            name,
            method,
            size,
            tuple(member for member, _ in measured),
            {member: len(values) for (member, _), values in zip(measured, arrays, strict=True)},
            non_finite,
        )
        return family, corrections


__all__ = ["CorrectedRule", "FamilyCorrection", "FamilyTests", "PooledFamily"]
