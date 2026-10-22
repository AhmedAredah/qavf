"""Statistical inference: tests, corrections, intervals, effects and decisions.

An independent library: arrays (or summary counts) in, typed
:class:`TestResult` out, usable by dataset comparisons and by any rule. It
needs SciPy (the ``stats`` extra) for distribution functions; without it a
test is reported as not assessed with the reason, never as a pass.

Quick start::

    from qavf.inference import correct, decide, welch_t

    result = welch_t(baseline_values, candidate_values)
    [result] = correct([result], "bh", alpha=0.05)
    decision = decide("material_and_significant", material=True, test=result)

Modules: :mod:`~qavf.inference.paired`, :mod:`~qavf.inference.independent`,
:mod:`~qavf.inference.equivalence`, :mod:`~qavf.inference.categorical`,
:mod:`~qavf.inference.counts`, :mod:`~qavf.inference.trend`,
:mod:`~qavf.inference.spatial` (tests); :mod:`~qavf.inference.multiple`
(corrections); :mod:`~qavf.inference.intervals` and
:mod:`~qavf.inference.resampling` (intervals, bootstraps, HAC, the work
budget); :mod:`~qavf.inference.effects` (effect sizes and agreement metrics);
:mod:`~qavf.inference.decision` (materiality with significance). Tests are
registered components (:data:`TESTS`, :func:`register_test`, the
``qavf.inference.tests`` entry-point group).
"""

from qavf.inference.categorical import (
    chi_square,
    contingency,
    fisher_exact,
    g_test,
    mcnemar,
    paired_table,
    stuart_maxwell,
)
from qavf.inference.counts import poisson_rate, two_proportion
from qavf.inference.decision import DEFAULT_ALPHA, DEFAULT_POLICY, Decision, decide, needs_test
from qavf.inference.equivalence import (
    bootstrap_equivalence,
    tost_from_moments,
    tost_independent,
    tost_paired,
)
from qavf.inference.independent import (
    Moments,
    anderson_darling,
    bootstrap_difference,
    brown_forsythe,
    ks_two_sample,
    mann_whitney,
    permutation,
    statistic_for,
    welch_t,
    welch_t_from_moments,
)
from qavf.inference.multiple import adjust, correct
from qavf.inference.paired import (
    paired_bootstrap,
    paired_permutation,
    paired_t,
    sign_test,
    wilcoxon_signed_rank,
)
from qavf.inference.registry import TESTS, InferenceTest, TestInput, register_test
from qavf.inference.results import (
    Alternative,
    Correction,
    DecisionPolicy,
    Dependence,
    Significance,
    TestResult,
    not_assessed,
)
from qavf.inference.spatial import LocalMoran, local_morans_i, morans_i
from qavf.inference.trend import (
    compare_correlations,
    correlation_test,
    kendall_tau,
    lag_test,
    mann_kendall,
    rank_correlation_test,
    seasonal_kendall,
    slope_difference,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POLICY",
    "TESTS",
    "Alternative",
    "Correction",
    "Decision",
    "DecisionPolicy",
    "Dependence",
    "InferenceTest",
    "LocalMoran",
    "Moments",
    "Significance",
    "TestInput",
    "TestResult",
    "adjust",
    "anderson_darling",
    "bootstrap_difference",
    "bootstrap_equivalence",
    "brown_forsythe",
    "chi_square",
    "compare_correlations",
    "contingency",
    "correct",
    "correlation_test",
    "decide",
    "fisher_exact",
    "g_test",
    "kendall_tau",
    "ks_two_sample",
    "lag_test",
    "local_morans_i",
    "mann_kendall",
    "mann_whitney",
    "mcnemar",
    "morans_i",
    "needs_test",
    "not_assessed",
    "paired_bootstrap",
    "paired_permutation",
    "paired_t",
    "paired_table",
    "permutation",
    "poisson_rate",
    "rank_correlation_test",
    "register_test",
    "seasonal_kendall",
    "sign_test",
    "slope_difference",
    "statistic_for",
    "stuart_maxwell",
    "tost_from_moments",
    "tost_independent",
    "tost_paired",
    "two_proportion",
    "welch_t",
    "welch_t_from_moments",
    "wilcoxon_signed_rank",
]
