"""Categorical data: contingency tables of independent sides, and paired categories.

Independent sides (rows baseline and candidate, columns categories):

* :func:`chi_square`: homogeneity of the category shares (Pearson);
* :func:`g_test`: the same by the likelihood ratio;
* :func:`fisher_exact`: an exact test of a 2x2 table.

Paired categories (rows the baseline category of a record, columns its
candidate category):

* :func:`mcnemar`: a binary category changing more one way than the other;
* :func:`stuart_maxwell`: marginal homogeneity of ``k`` categories.

:func:`contingency` and :func:`paired_table` build the tables from labels.
Categories absent on both sides are left out; a side without observations is
not assessed. A table with no changes at all has ``p = 1``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qavf.inference._support import Unassessable, assessed, require_scipy
from qavf.inference.effects import cramers_v, odds_ratio
from qavf.inference.results import Alternative, TestResult


def contingency(baseline: Sequence[Any], candidate: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
    """A ``2 x k`` table of category counts per side, and its sorted categories."""
    categories = sorted(set(baseline) | set(candidate), key=lambda v: (str(type(v)), v))
    index = {category: i for i, category in enumerate(categories)}
    table = np.zeros((2, len(categories)), dtype=np.int64)
    for row, labels in enumerate((baseline, candidate)):
        for label in labels:
            table[row, index[label]] += 1
    return table, categories


def paired_table(baseline: Sequence[Any], candidate: Sequence[Any]) -> tuple[np.ndarray, list[Any]]:
    """A ``k x k`` table: row the baseline category of a pair, column its candidate one."""
    if len(baseline) != len(candidate):
        raise ValueError("paired categories need one candidate label per baseline label")
    categories = sorted(set(baseline) | set(candidate), key=lambda v: (str(type(v)), v))
    index = {category: i for i, category in enumerate(categories)}
    table = np.zeros((len(categories), len(categories)), dtype=np.int64)
    for before, after in zip(baseline, candidate, strict=True):
        table[index[before], index[after]] += 1
    return table, categories


def _counts(table: npt.ArrayLike, square: bool = False) -> np.ndarray:
    counts = np.asarray(table)
    if counts.ndim != 2 or counts.dtype.kind not in "iuf" or (counts < 0).any():
        raise ValueError("a contingency table is a 2-D array of non-negative counts")
    if not np.array_equal(counts, np.round(counts)):
        raise ValueError("contingency counts must be whole numbers")
    if not counts.size or not np.isfinite(counts).all():
        raise Unassessable("a contingency table needs nonempty finite counts")
    # Exact tests and all marginal sums below use signed Int64. Check before
    # conversion: a UInt64 value or infinity must never wrap into a negative
    # count and then be mistaken for an empty category.
    maximum = np.iinfo(np.int64).max
    largest = int(counts.max())
    if largest > maximum:
        raise Unassessable("contingency counts exceed the supported Int64 range")
    if largest * counts.size > maximum and sum(int(value) for value in counts.flat) > maximum:
        raise Unassessable("contingency total exceeds the supported Int64 range")
    if square and counts.shape[0] != counts.shape[1]:
        raise ValueError("a paired table must be square (k x k)")
    return counts.astype(np.int64)


def _homogeneity(table: npt.ArrayLike, method: str, lambda_: str | None) -> TestResult:
    counts = _counts(table)
    n = {f"row_{i}": int(total) for i, total in enumerate(counts.sum(axis=1))}
    if (counts.sum(axis=1) == 0).any():
        raise Unassessable("a side (row) has no observations", n=n)
    kept = counts[:, counts.sum(axis=0) > 0]
    notes = []
    if kept.shape[1] < counts.shape[1]:
        notes.append(f"{counts.shape[1] - kept.shape[1]} empty categories left out")
    effect = {"effect": cramers_v(kept), "effect_name": "cramers_v"}
    if kept.shape[1] < 2:
        return TestResult(
            method=method, statistic=0.0, p_value=1.0, n=n, notes=[*notes, "one category"],
            **effect,
        )  # fmt: skip
    result = require_scipy().chi2_contingency(kept, correction=False, lambda_=lambda_)
    expected = np.asarray(result.expected_freq)
    if small := int((expected < 5).sum()):
        notes.append(
            f"{small} cell(s) expect fewer than 5 counts; the chi-square "
            "approximation may be poor (consider fisher_exact for 2x2)"
        )
    return TestResult(
        method=method,
        statistic=float(result.statistic),
        statistic_name="G" if lambda_ else "chi2",
        p_value=float(result.pvalue),
        n=n,
        assumptions=["independent observations", f"{int(result.dof)} degrees of freedom"],
        notes=notes,
        **effect,
    )


@assessed("chi_square")
def chi_square(table: npt.ArrayLike) -> TestResult:
    """Pearson chi-square test that every row has the same category shares."""
    return _homogeneity(table, "chi_square", None)


@assessed("g_test")
def g_test(table: npt.ArrayLike) -> TestResult:
    """Likelihood-ratio (G) test that every row has the same category shares."""
    return _homogeneity(table, "g_test", "log-likelihood")


@assessed("fisher_exact")
def fisher_exact(table: npt.ArrayLike, *, alternative: Alternative = "two-sided") -> TestResult:
    """Fisher's exact test of a 2x2 table ``[[a, b], [c, d]]``.

    Rows are baseline and candidate, columns event and non-event. The effect is
    the sample odds ratio of the event (candidate over baseline), with SciPy's
    normal-approximation interval of its logarithm; ``greater`` means the
    candidate's odds are higher.
    """
    counts = _counts(table)
    if counts.shape != (2, 2):
        raise ValueError("Fisher's exact test needs a 2x2 table")
    stats = require_scipy()
    n = {"baseline": int(counts[0].sum()), "candidate": int(counts[1].sum())}
    if (counts.sum(axis=1) == 0).any():
        raise Unassessable("a side has no observations", n=n)
    # SciPy's alternative refers to the odds ratio of the first row over the second.
    flipped = {"greater": "less", "less": "greater"}.get(alternative, alternative)
    result = stats.fisher_exact(counts, alternative=flipped)
    effect = odds_ratio(counts[::-1])
    low = high = None
    if effect is not None and effect > 0:
        from scipy.stats.contingency import odds_ratio as scipy_odds_ratio

        interval = scipy_odds_ratio(counts[::-1], kind="sample").confidence_interval(0.95)
        low, high = float(interval.low), float(interval.high)
    return TestResult(
        method="fisher_exact",
        statistic=effect,
        statistic_name="odds_ratio",
        p_value=float(result.pvalue),
        alternative=alternative,
        effect=effect,
        effect_name="odds_ratio",
        ci_low=low,
        ci_high=high,
        confidence=0.95 if low is not None else None,
        n=n,
        assumptions=["independent observations", "fixed margins (conditional test)"],
    )


@assessed("mcnemar")
def mcnemar(table: npt.ArrayLike, *, exact: bool = True) -> TestResult:
    """McNemar's test of a paired binary category ``[[a, b], [c, d]]``.

    Rows are the baseline category (event, non-event), columns the candidate
    one; only the discordant pairs ``b`` (event lost) and ``c`` (event gained)
    matter. Exact binomial by default, else chi-square with continuity
    correction. The effect is ``(c - b) / n``, the change of the event share.
    """
    counts = _counts(table, square=True)
    if counts.shape != (2, 2):
        raise ValueError("McNemar's test needs a 2x2 table")
    stats = require_scipy()
    lost, gained, total = int(counts[0, 1]), int(counts[1, 0]), int(counts.sum())
    n = {"pairs": total, "discordant": lost + gained}
    if not total:
        raise Unassessable("no pairs", n=n)
    common = {"method": "mcnemar", "effect": (gained - lost) / total, "effect_name": "share_change"}
    if not lost + gained:
        return TestResult(statistic=0.0, p_value=1.0, n=n, notes=["no discordant pairs"], **common)
    if exact:
        p = float(stats.binomtest(min(lost, gained), lost + gained, 0.5).pvalue)
        statistic, name = float(min(lost, gained)), "discordant_min"
    else:
        statistic = (abs(lost - gained) - 1) ** 2 / (lost + gained)
        p, name = float(stats.chi2.sf(statistic, 1)), "chi2"
    return TestResult(
        statistic=statistic,
        statistic_name=name,
        p_value=p,
        n=n,
        assumptions=["independent pairs"],
        notes=["exact binomial" if exact else "chi-square with continuity correction"],
        **common,
    )


@assessed("stuart_maxwell")
def stuart_maxwell(table: npt.ArrayLike) -> TestResult:
    """Stuart-Maxwell test of marginal homogeneity of a paired ``k x k`` table.

    ``d`` is the difference between the row and column margins and ``S`` its
    covariance under homogeneity (``S_ii = n_i. + n_.i - 2 n_ii``,
    ``S_ij = -(n_ij + n_ji)``); the statistic ``d' S^-1 d`` over the first
    ``k - 1`` categories is chi-square with ``k - 1`` degrees of freedom.
    Categories that never change are left out. The effect is the largest
    change of a category's share.
    """
    counts = _counts(table, square=True)
    stats = require_scipy()
    total = int(counts.sum())
    n = {"pairs": total, "categories": counts.shape[0]}
    if not total:
        raise Unassessable("no pairs", n=n)
    changes = counts + counts.T
    np.fill_diagonal(changes, 0)
    keep = changes.sum(axis=1) > 0
    shares = (counts.sum(axis=0) - counts.sum(axis=1)) / total
    common = {
        "method": "stuart_maxwell",
        "effect": float(np.abs(shares).max()),
        "effect_name": "max_share_change",
    }
    if keep.sum() < 2:
        return TestResult(statistic=0.0, p_value=1.0, n=n, notes=["no category changes"], **common)
    kept = counts[np.ix_(keep, keep)].astype(float)
    k = kept.shape[0]
    d = (kept.sum(axis=1) - kept.sum(axis=0))[:-1]
    s = -(kept + kept.T)
    np.fill_diagonal(s, kept.sum(axis=1) + kept.sum(axis=0) - 2 * np.diag(kept))
    s = s[:-1, :-1]
    try:
        statistic = float(d @ np.linalg.solve(s, d))
    except np.linalg.LinAlgError:
        raise Unassessable(
            "paired category changes have a singular covariance; "
            "a rank-aware marginal-homogeneity test is required",
            n=n,
        ) from None
    notes = [f"{int((~keep).sum())} unchanged categories left out"] if not keep.all() else []
    return TestResult(
        statistic=statistic,
        statistic_name="chi2",
        p_value=float(stats.chi2.sf(statistic, k - 1)),
        n=n,
        assumptions=["independent pairs", f"{k - 1} degrees of freedom"],
        notes=notes,
        **common,
    )
