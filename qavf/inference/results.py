"""Typed, serializable outcomes of statistical tests and of families of tests.

A :class:`TestResult` is one test: the statistic, the p-value (and, once a
family is corrected, the adjusted p-value with the family size and method),
an effect size with its confidence interval, the sample sizes, the assumptions
relied on, how dependence was handled and the resampling seed. A test that
could not run is ``assessed=False`` with a ``reason``; it is never a pass.

A :class:`Significance` summarizes one family judged together under a
decision policy (a measure across groups, a shape check across series); it is
the type of every ``significance`` slot of comparison results.

Both are audit snapshots: non-finite statistics (an infinite t for separated
constant samples) survive JSON transport with their values intact.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

from pydantic import Field, field_serializer, model_validator

from qavf.audit_codec import encode_audit as encode_non_finite
from qavf.snapshots import SnapshotModel

Alternative = Literal["two-sided", "less", "greater"]
"""Direction of the alternative: ``less`` means the candidate is smaller."""
Dependence = Literal["none", "block", "hac", "effective_n", "spatial"]
"""How dependence between observations was handled: assumed independent, a
block bootstrap, Newey-West (HAC) standard errors, an effective sample size,
or spatial permutation."""
Correction = Literal["none", "bonferroni", "holm", "bh", "by"]
"""Multiple-testing correction: Bonferroni, Holm, Benjamini-Hochberg (``bh``)
or Benjamini-Yekutieli (``by``)."""
DecisionPolicy = Literal["material", "significant", "material_and_significant", "equivalence"]
"""How materiality and significance combine (see :mod:`qavf.inference.decision`)."""

_FLOATS = ("statistic", "p_value", "adjusted_p", "effect", "ci_low", "ci_high")


class TestResult(SnapshotModel):
    """The outcome of one statistical test."""

    __test__: ClassVar[bool] = False  # not a pytest test class
    audit_fields: ClassVar[tuple[str, ...]] = _FLOATS

    method: str
    """The test (``welch_t``, ``mann_whitney``, ...)."""
    assessed: bool = True
    reason: str | None = None
    """Why the test was not assessed (too few values, no SciPy, work budget, ...)."""
    label: str | None = None
    """What was tested, e.g. a group or a column, when the result is one of a family."""
    statistic: float | None = None
    statistic_name: str | None = None
    p_value: float | None = None
    adjusted_p: float | None = None
    """The p-value after the family's correction; ``None`` until corrected."""
    correction: Correction | None = None
    family_size: int | None = None
    """Tests adjusted together (assessed tests of the family)."""
    alpha: float | None = None
    significant: bool | None = None
    """Adjusted (else raw) p-value below ``alpha``; ``None`` until judged."""
    alternative: Alternative = "two-sided"
    effect: float | None = None
    """Effect size in ``effect_name`` units (candidate relative to baseline)."""
    effect_name: str | None = None
    ci_low: float | None = None
    """Confidence interval of the effect (``None``: not computed)."""
    ci_high: float | None = None
    confidence: float | None = None
    n: dict[str, int] = Field(default_factory=dict)
    """Sample size per side (``baseline``/``candidate``), or ``pairs``, ``values``..."""
    assumptions: list[str] = Field(default_factory=list)
    """What the method checked or relied on."""
    dependence: Dependence = "none"
    seed: int | None = None
    """Seed of a resampling test; the same seed reproduces the result exactly."""
    resamples: int | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _probability_state(self) -> TestResult:
        if self.alpha is not None and not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        probabilities = (self.p_value, self.adjusted_p)
        invalid = any(
            p is not None and (not math.isfinite(p) or not 0 <= p <= 1) for p in probabilities
        )
        if self.assessed and (self.p_value is None or invalid):
            object.__setattr__(self, "assessed", False)
            object.__setattr__(self, "reason", "test has no finite p-value in [0, 1]")
        judged = (
            bool(self.p < self.alpha)
            if self.assessed and self.p is not None and self.alpha is not None
            else None
        )
        object.__setattr__(self, "significant", judged)
        return self

    @field_serializer(*_FLOATS, when_used="json")
    def _encode(self, value: Any) -> Any:
        return encode_non_finite(value)

    @property
    def p(self) -> float | None:
        """The p-value a decision uses: adjusted when corrected, else raw."""
        return self.adjusted_p if self.adjusted_p is not None else self.p_value

    def judged(self, alpha: float) -> TestResult:
        """This result with ``alpha`` and ``significant`` set from :attr:`p`."""
        significant = None if not self.assessed or self.p is None else bool(self.p < alpha)
        return self.model_copy(update={"alpha": alpha, "significant": significant})


def not_assessed(method: str, reason: str, **fields: Any) -> TestResult:
    """A test that could not run: never a pass."""
    return TestResult(method=method, assessed=False, reason=reason, **fields)


class Significance(SnapshotModel):
    """A family of tests judged together, e.g. one measure across groups.

    ``failed`` counts the family's members failing under ``decision``;
    ``not_assessed`` those whose test should apply but could not run. When no
    sound test exists for the quantity (``applicable=False``), the decision
    falls back to materiality and ``reason`` says so.
    """

    method: str
    """The test, or ``none`` when no test applies."""
    decision: DecisionPolicy
    alpha: float
    correction: Correction
    applicable: bool = True
    reason: str | None = None
    family_size: int = 0
    """Tests assessed and adjusted together."""
    significant: int = 0
    """Members significant after correction (equivalence shown, for ``equivalence``)."""
    failed: int = 0
    not_assessed: int = 0
    tests: list[TestResult] = Field(default_factory=list)
    """The family's leading tests (in family order), bounded by the evidence limit."""
