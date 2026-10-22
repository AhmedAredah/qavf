"""Built-in test components: the package's test functions, selectable by name."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from qavf.engine.registry import RuleParams
from qavf.inference.counts import poisson_rate, two_proportion
from qavf.inference.equivalence import (
    bootstrap_equivalence,
    tost_from_moments,
    tost_paired,
)
from qavf.inference.independent import (
    anderson_darling,
    bootstrap_difference,
    brown_forsythe,
    ks_two_sample,
    mann_whitney,
    permutation,
    statistic_for,
    welch_t_from_moments,
)
from qavf.inference.paired import (
    paired_bootstrap,
    paired_permutation,
    paired_t,
    sign_test,
    wilcoxon_signed_rank,
)
from qavf.inference.registry import InferenceTest, Needs, TestInput, TestParams, register_test
from qavf.inference.resampling import DEFAULT_RESAMPLES
from qavf.inference.results import TestResult, not_assessed


class ResamplingParams(TestParams):
    """Options of a resampling test."""

    resamples: int = Field(default=DEFAULT_RESAMPLES, ge=99)
    """Random resamples (more: steadier p-values, more work)."""


class BootstrapParams(ResamplingParams):
    interval: Literal["percentile", "bca"] = "percentile"
    """Confidence interval of the difference (BCa costs ``n^2`` jackknife work)."""


def _missing(test: InferenceTest, data: TestInput, *fields: str) -> TestResult | None:
    absent = [field for field in fields if getattr(data, field) is None]
    if absent:
        return not_assessed(test.name, f"{test.name} needs {', '.join(absent)}")
    return None


def _samples(test: InferenceTest, data: TestInput) -> TestResult | None:
    return _missing(test, data, "baseline", "candidate")


@register_test
class WelchT(InferenceTest):
    """Welch's t-test of the difference of means (from each side's moments)."""

    name: ClassVar[str] = "welch_t"
    needs: ClassVar[Needs] = "moments"
    supports_equivalence: ClassVar[bool] = True

    def run(self, data: TestInput) -> TestResult:
        if (missing := _missing(self, data, "moments")) is not None:
            return missing
        assert data.moments is not None
        b, c = data.moments
        return welch_t_from_moments(b, c, alternative=self.alternative, confidence=data.confidence)

    def equivalent(self, data: TestInput) -> TestResult:
        if (missing := _missing(self, data, "moments", "margin")) is not None:
            return missing
        assert data.moments is not None and data.margin is not None
        return tost_from_moments(*data.moments, data.margin)


@register_test
class MannWhitney(InferenceTest):
    """Mann-Whitney U: does one side tend to have larger values?"""

    name: ClassVar[str] = "mann_whitney"

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return mann_whitney(data.baseline, data.candidate, alternative=self.alternative)  # type: ignore[arg-type]


@register_test
class KolmogorovSmirnov(InferenceTest):
    """Two-sample Kolmogorov-Smirnov test of any difference of distribution."""

    name: ClassVar[str] = "ks"
    Params: ClassVar[type[RuleParams]] = RuleParams

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return ks_two_sample(data.baseline, data.candidate)  # type: ignore[arg-type]


@register_test
class AndersonDarling(InferenceTest):
    """Anderson-Darling k-sample test of a common distribution (tail-sensitive)."""

    name: ClassVar[str] = "anderson_darling"
    Params: ClassVar[type[RuleParams]] = RuleParams

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return anderson_darling(data.baseline, data.candidate)  # type: ignore[arg-type]


@register_test
class BrownForsythe(InferenceTest):
    """Brown-Forsythe test of a difference of spread."""

    name: ClassVar[str] = "brown_forsythe"
    Params: ClassVar[type[RuleParams]] = RuleParams

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return brown_forsythe(data.baseline, data.candidate)  # type: ignore[arg-type]


@register_test
class Permutation(InferenceTest):
    """Permutation test of the difference of the measure's statistic."""

    name: ClassVar[str] = "permutation"
    Params: ClassVar[type[RuleParams]] = ResamplingParams
    uses_statistic: ClassVar[bool] = True

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return permutation(
            data.baseline,  # type: ignore[arg-type]
            data.candidate,  # type: ignore[arg-type]
            statistic=statistic_for(data.statistic, data.probability),
            alternative=self.alternative,
            resamples=self.params.resamples,  # type: ignore[attr-defined]
            seed=data.seed,
            max_work=data.max_work,
        )


@register_test
class Bootstrap(InferenceTest):
    """Bootstrap of the difference of the measure's statistic (sum, median, quantile...)."""

    name: ClassVar[str] = "bootstrap"
    Params: ClassVar[type[RuleParams]] = BootstrapParams
    uses_statistic: ClassVar[bool] = True
    supports_equivalence: ClassVar[bool] = True

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return bootstrap_difference(
            data.baseline,  # type: ignore[arg-type]
            data.candidate,  # type: ignore[arg-type]
            statistic=statistic_for(data.statistic, data.probability),
            name=data.statistic,
            alternative=self.alternative,
            interval=self.params.interval,  # type: ignore[attr-defined]
            confidence=data.confidence,
            resamples=self.params.resamples,  # type: ignore[attr-defined]
            seed=data.seed,
            max_work=data.max_work,
        )

    def equivalent(self, data: TestInput) -> TestResult:
        if (missing := _missing(self, data, "baseline", "candidate", "margin")) is not None:
            return missing
        return bootstrap_equivalence(
            data.baseline,  # type: ignore[arg-type]
            data.candidate,  # type: ignore[arg-type]
            data.margin,  # type: ignore[arg-type]
            statistic=statistic_for(data.statistic, data.probability),
            name=data.statistic,
            resamples=self.params.resamples,  # type: ignore[attr-defined]
            seed=data.seed,
            max_work=data.max_work,
        )


@register_test
class Poisson(InferenceTest):
    """Exact conditional test of two Poisson rates (counts per exposure)."""

    name: ClassVar[str] = "poisson"
    minimum: ClassVar[int] = 0
    needs: ClassVar[Needs] = "counts"

    def run(self, data: TestInput) -> TestResult:
        if (missing := _missing(self, data, "counts")) is not None:
            return missing
        assert data.counts is not None
        return poisson_rate(
            data.counts,
            data.totals or (1.0, 1.0),
            alternative=self.alternative,
            confidence=data.confidence,
        )


@register_test
class TwoProportion(InferenceTest):
    """Pooled two-proportion z-test of two shares."""

    name: ClassVar[str] = "two_proportion"
    minimum: ClassVar[int] = 1
    needs: ClassVar[Needs] = "proportions"

    def run(self, data: TestInput) -> TestResult:
        if (missing := _missing(self, data, "counts", "totals")) is not None:
            return missing
        assert data.counts is not None and data.totals is not None
        return two_proportion(
            data.counts, data.totals, alternative=self.alternative, confidence=data.confidence
        )


@register_test
class PairedT(InferenceTest):
    """Paired t-test of the mean difference of matched values."""

    name: ClassVar[str] = "paired_t"
    needs: ClassVar[Needs] = "paired"
    supports_equivalence: ClassVar[bool] = True

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return paired_t(
            data.baseline,  # type: ignore[arg-type]
            data.candidate,  # type: ignore[arg-type]
            alternative=self.alternative,
            confidence=data.confidence,
        )

    def equivalent(self, data: TestInput) -> TestResult:
        if (missing := _missing(self, data, "baseline", "candidate", "margin")) is not None:
            return missing
        return tost_paired(data.baseline, data.candidate, data.margin)  # type: ignore[arg-type]


@register_test
class Wilcoxon(InferenceTest):
    """Wilcoxon signed-rank test of matched values."""

    name: ClassVar[str] = "wilcoxon"
    minimum: ClassVar[int] = 1
    needs: ClassVar[Needs] = "paired"

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return wilcoxon_signed_rank(data.baseline, data.candidate, alternative=self.alternative)  # type: ignore[arg-type]


@register_test
class Sign(InferenceTest):
    """Exact sign test of matched values."""

    name: ClassVar[str] = "sign"
    minimum: ClassVar[int] = 1
    needs: ClassVar[Needs] = "paired"

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return sign_test(data.baseline, data.candidate, alternative=self.alternative)  # type: ignore[arg-type]


@register_test
class PairedPermutation(InferenceTest):
    """Sign-flip permutation test of the mean difference of matched values."""

    name: ClassVar[str] = "paired_permutation"
    needs: ClassVar[Needs] = "paired"
    Params: ClassVar[type[RuleParams]] = ResamplingParams

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return paired_permutation(
            data.baseline,  # type: ignore[arg-type]
            data.candidate,  # type: ignore[arg-type]
            alternative=self.alternative,
            resamples=self.params.resamples,  # type: ignore[attr-defined]
            seed=data.seed,
            max_work=data.max_work,
        )


@register_test
class PairedBootstrap(InferenceTest):
    """Bootstrap of the mean (or another statistic) of matched differences."""

    name: ClassVar[str] = "paired_bootstrap"
    needs: ClassVar[Needs] = "paired"
    Params: ClassVar[type[RuleParams]] = BootstrapParams

    def run(self, data: TestInput) -> TestResult:
        if (missing := _samples(self, data)) is not None:
            return missing
        return paired_bootstrap(
            data.baseline,  # type: ignore[arg-type]
            data.candidate,  # type: ignore[arg-type]
            statistic=statistic_for(data.statistic, data.probability),
            name=data.statistic,
            alternative=self.alternative,
            interval=self.params.interval,  # type: ignore[attr-defined]
            confidence=data.confidence,
            resamples=self.params.resamples,  # type: ignore[attr-defined]
            seed=data.seed,
            max_work=data.max_work,
        )
