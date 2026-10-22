"""Statistical tests as named, pluggable components (``test: name`` in configuration).

A test component declares what it ``needs`` from a caller (raw samples of
both sides, their summary moments, event counts with exposures, successes
with trials, or matched pairs) and turns a :class:`TestInput` into a
:class:`~qavf.inference.results.TestResult`. Built-ins wrap the functions of
this package; plugins add their own with :func:`register_test` or the
``qavf.inference.tests`` entry-point group, gated by ``plugins.entry_points``
exactly like comparison components.

Writing a test::

    @register_test
    class MedianShift(InferenceTest):
        name = "median_shift"
        needs = "samples"

        def run(self, data):
            return bootstrap_difference(data.baseline, data.candidate,
                                        statistic=statistic_for("median"), name="median",
                                        seed=data.seed, max_work=data.max_work)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from qavf.engine.registry import RuleParams
from qavf.inference.independent import Moments
from qavf.inference.resampling import DEFAULT_MAX_WORK, DEFAULT_SEED
from qavf.inference.results import Alternative, TestResult, not_assessed
from qavf.plugins.registry import Component, ComponentRegistry

Needs = Literal["samples", "moments", "counts", "proportions", "paired"]
"""What a test reads from :class:`TestInput`."""


@dataclass(frozen=True, slots=True)
class TestInput:
    """Everything a test may read; a caller fills what the test ``needs``.

    ``baseline``/``candidate`` are raw samples (``samples``) or matched values in
    pair order (``paired``); ``moments`` the summary of each side; ``counts`` and
    ``totals`` the events and exposures (``counts``) or successes and trials
    (``proportions``), each ``(baseline, candidate)``.
    """

    __test__: ClassVar[bool] = False  # not a pytest test class

    baseline: np.ndarray | None = None
    candidate: np.ndarray | None = None
    moments: tuple[Moments, Moments] | None = None
    counts: tuple[float, float] | None = None
    totals: tuple[float, float] | None = None
    statistic: str = "mean"
    """The statistic a resampling test compares: mean, median, sum, std or quantile."""
    probability: float | None = None
    """The quantile probability for ``statistic='quantile'``."""
    margin: tuple[float, float] | None = None
    """The equivalence margin ``(low, high)`` of the difference."""
    confidence: float = 0.95
    seed: int = DEFAULT_SEED
    max_work: int = DEFAULT_MAX_WORK


class TestParams(RuleParams):
    """Options shared by every built-in test."""

    alternative: Alternative = "two-sided"
    """``greater``: the candidate is larger; ``less``: smaller."""


class InferenceTest(Component):
    """A registered test: ``run`` returns a result; ``equivalent`` its TOST form."""

    __test__: ClassVar[bool] = False  # not a pytest test class
    Params: ClassVar[type[RuleParams]] = TestParams
    needs: ClassVar[Needs] = "samples"
    supports_equivalence: ClassVar[bool] = False
    """Whether :meth:`equivalent` can show equivalence within a margin."""

    @abstractmethod
    def run(self, data: TestInput) -> TestResult:
        """Test for a difference; never raise for data the test cannot use."""

    def equivalent(self, data: TestInput) -> TestResult:
        """Test equivalence within ``data.margin``; not assessed unless supported."""
        return not_assessed(self.name, f"{self.name} has no equivalence form")

    @property
    def alternative(self) -> Alternative:
        return getattr(self.params, "alternative", "two-sided")


TESTS: ComponentRegistry[InferenceTest] = ComponentRegistry(
    "statistical test",
    "qavf.inference.tests",
    InferenceTest,
    builtins=("qavf.inference.components",),
)
register_test = TESTS.register


def registries() -> dict[str, ComponentRegistry[Any]]:
    """The inference registries, keyed by their ``plugins.required`` qualifier."""
    return {"inference.tests": TESTS}
