"""Combining materiality with significance into one decision.

Materiality asks whether a difference is large enough to matter (a tolerance);
significance whether it is larger than chance alone explains (a test, after
the family's multiple-testing correction). Four policies combine them:

=============================  ==============================================
``material_and_significant``   (default) fail only when the difference is
                               material **and** significant; an immaterial
                               difference passes without a test
``material``                   fail when material (no test is run)
``significant``                fail when significant, however small
``equivalence``                pass only when equivalence within the
                               materiality margin is shown (a TOST-style
                               test whose p-value is below alpha)
=============================  ==============================================

Two situations are kept apart. A quantity with **no sound test** (a minimum,
a maximum, a distinct count) is judged by materiality under every policy,
and the decision says so (``basis="materiality"``). A test that **should
apply but cannot run** (no SciPy, too few values, a failed precondition, the
work budget) leaves the decision ``not_assessed`` whenever the policy needs
it: never a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qavf.inference.results import DecisionPolicy, TestResult

POLICIES: tuple[DecisionPolicy, ...] = (
    "material_and_significant",
    "material",
    "significant",
    "equivalence",
)
DEFAULT_POLICY: DecisionPolicy = "material_and_significant"
DEFAULT_ALPHA = 0.05

Outcome = Literal["pass", "fail", "not_assessed"]
Basis = Literal["materiality", "significance", "both", "equivalence"]


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome for one compared quantity, the evidence it rests on, and why."""

    outcome: Outcome
    basis: Basis
    reason: str

    @property
    def failed(self) -> bool:
        return self.outcome == "fail"

    @property
    def assessed(self) -> bool:
        return self.outcome != "not_assessed"


def needs_test(policy: DecisionPolicy, material: bool) -> bool:
    """Whether ``policy`` needs a test result to decide (``material`` alone never does)."""
    if policy == "material":
        return False
    if policy == "material_and_significant":
        return material
    return True


def decide(
    policy: DecisionPolicy,
    material: bool,
    test: TestResult | None,
    *,
    alpha: float = DEFAULT_ALPHA,
    applicable: bool = True,
) -> Decision:
    """Decide one quantity under ``policy``.

    ``test`` is the quantity's (corrected) test result: a difference test, or
    for ``equivalence`` an equivalence test. ``applicable=False`` declares that
    no sound test exists for the quantity; the decision then falls back to
    materiality. ``alpha`` applies when the result was not judged already.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown decision policy {policy!r}; use one of {', '.join(POLICIES)}")
    if not applicable or policy == "material":
        why = "judged by materiality" if policy == "material" else (
            "no sound test applies; judged by materiality"
        )  # fmt: skip
        return Decision("fail" if material else "pass", "materiality", why)
    if not needs_test(policy, material):
        return Decision("pass", "materiality", "not material; no test needed")
    if test is None or not test.assessed:
        reason = "no test result" if test is None else f"test not assessed: {test.reason}"
        return Decision("not_assessed", "significance", reason)
    significant = test.significant
    if significant is None:
        significant = test.p is not None and test.p < alpha
    p = f"p = {test.p:.3g}" if test.p is not None else "no p-value"
    if policy == "equivalence":
        if significant:
            return Decision("pass", "equivalence", f"equivalence shown ({p})")
        return Decision("fail", "equivalence", f"equivalence not shown ({p})")
    if policy == "significant":
        return Decision(
            "fail" if significant else "pass",
            "significance",
            f"significant ({p})" if significant else f"not significant ({p})",
        )
    if significant:
        return Decision("fail", "both", f"material and significant ({p})")
    return Decision("pass", "both", f"material but not significant ({p})")
