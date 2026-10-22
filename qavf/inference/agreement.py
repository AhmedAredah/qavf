"""Agreement of two measurements of the same records: the candidate fitted on the baseline.

Two versions that agree lie on the identity line ``candidate = baseline``.
:func:`fit_line` fits ``candidate = intercept + slope * baseline`` by ordinary
least squares (the baseline taken as exact) or by Deming regression (both
sides measured with error, in a known variance ratio); :func:`identity_test`
tests ``slope = 1`` and ``intercept = 0`` jointly. The summary metrics of
agreement (Bland-Altman limits, Lin's concordance, MAE, RMSE, ...) are in
:mod:`qavf.inference.effects`.

Sums of squares are taken around the means after one shared power-of-two
scaling, so neither large values nor a small variation around a large offset
are lost; the intercept is reported in data units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from qavf.inference._support import (
    Unassessable,
    assessed,
    at_least,
    check_confidence,
    common_unit,
    observations,
    require_scipy,
)
from qavf.inference.results import TestResult

FitMethod = Literal["ols", "deming"]


@dataclass(frozen=True, slots=True)
class LineFit:
    """``candidate = intercept + slope * baseline`` with the estimates' covariance."""

    method: FitMethod
    slope: float
    intercept: float
    covariance: tuple[tuple[float, float], tuple[float, float]] | None
    """Covariance of ``(intercept, slope)``: analytic for OLS, jackknife for Deming;
    ``None`` with fewer than 3 pairs."""
    n: int

    @property
    def se_slope(self) -> float | None:
        return None if self.covariance is None else math.sqrt(max(self.covariance[1][1], 0.0))

    @property
    def se_intercept(self) -> float | None:
        return None if self.covariance is None else math.sqrt(max(self.covariance[0][0], 0.0))


def _paired(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    b, c = observations(baseline, "baseline"), observations(candidate, "candidate")
    if len(b) != len(c):
        raise Unassessable("a fit needs one candidate value per baseline value")
    return b, c


def _deming_slope(sbb: np.ndarray, scc: np.ndarray, sbc: np.ndarray, ratio: float) -> np.ndarray:
    """Deming's slope from centred sums; ``ratio`` is the candidate's error variance over
    the baseline's (1: orthogonal regression)."""
    gap = scc - ratio * sbb
    with np.errstate(divide="ignore", invalid="ignore"):
        return (gap + np.sqrt(gap * gap + 4 * ratio * sbc * sbc)) / (2 * sbc)


def fit_line(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    method: FitMethod = "ols",
    error_ratio: float = 1.0,
) -> LineFit:
    """Fit ``candidate`` on ``baseline`` (at least 2 pairs).

    Raises:
        Unassessable: for unusable values, fewer than 2 pairs, a constant baseline
            (OLS) or uncorrelated sides (Deming), where the line is undefined.
    """
    if method not in ("ols", "deming"):
        raise ValueError(f"method must be 'ols' or 'deming', got {method!r}")
    if not error_ratio > 0:
        raise ValueError("error_ratio must be positive")
    b, c = _paired(baseline, candidate)
    n = len(b)
    at_least(n, 2, "pairs", n={"pairs": n})
    (sb, sc), unit = common_unit(b, c)
    mb, mc = float(sb.mean()), float(sc.mean())
    x, y = sb - mb, sc - mc
    sbb, scc, sbc = float(x @ x), float(y @ y), float(x @ y)
    if method == "ols":
        if sbb == 0:
            raise Unassessable("the baseline is constant; the fit is undefined", n={"pairs": n})
        slope = sbc / sbb
    else:
        if sbc == 0:
            raise Unassessable("the sides are uncorrelated; Deming's line is undefined")
        slope = float(_deming_slope(np.array(sbb), np.array(scc), np.array(sbc), error_ratio))
    intercept = mc - slope * mb
    covariance = None
    if n >= 3:
        if method == "ols":
            residual = float(((y - slope * x) ** 2).sum()) / (n - 2)
            covariance = (
                (residual * (1 / n + mb * mb / sbb), -residual * mb / sbb),
                (-residual * mb / sbb, residual / sbb),
            )
        else:
            covariance = _jackknife(x, y, mb, mc, n, error_ratio)
    if covariance is not None:
        scale = ((unit * unit, unit), (unit, 1.0))
        covariance = tuple(  # type: ignore[assignment]
            tuple(value * factor for value, factor in zip(row, factors, strict=True))
            for row, factors in zip(covariance, scale, strict=True)
        )
    return LineFit(method, slope, intercept * unit, covariance, n)


def _jackknife(
    x: np.ndarray, y: np.ndarray, mb: float, mc: float, n: int, ratio: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Jackknife covariance of Deming's ``(intercept, slope)``, from leave-one-out sums.

    Leaving pair ``i`` out of centred values moves the means by ``-x_i / (n - 1)``,
    so ``Sxx' = Sxx - x_i^2 - x_i^2 / (n - 1)`` (and alike for the other sums):
    every leave-one-out fit costs ``O(1)``.
    """
    sbb, scc, sbc = x @ x, y @ y, x @ y
    k = n - 1
    loo_bb = sbb - x * x - x * x / k
    loo_cc = scc - y * y - y * y / k
    loo_bc = sbc - x * y - x * y / k
    slopes = _deming_slope(loo_bb, loo_cc, loo_bc, ratio)
    intercepts = (mc - y / k) - slopes * (mb - x / k)
    if not (np.isfinite(slopes).all() and np.isfinite(intercepts).all()):
        raise Unassessable("a leave-one-out Deming fit is undefined")
    estimates = np.column_stack([intercepts, slopes])
    centred = estimates - estimates.mean(axis=0)
    covariance = centred.T @ centred * (k / n)
    return (
        (float(covariance[0, 0]), float(covariance[0, 1])),
        (float(covariance[1, 0]), float(covariance[1, 1])),
    )


@assessed("identity_fit")
def identity_test(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    *,
    method: FitMethod = "ols",
    error_ratio: float = 1.0,
    confidence: float = 0.95,
) -> TestResult:
    """Joint Wald test that ``slope = 1`` and ``intercept = 0`` (the identity line).

    ``F = d' V^-1 d / 2`` with ``d = (intercept, slope - 1)`` and ``V`` the
    estimates' covariance (analytic for OLS, jackknife for Deming), referred to
    ``F(2, n - 2)``. The effect is the slope, with its ``t(n - 2)`` interval; the
    intercept and its interval are noted. At least 3 pairs; a perfect fit on the
    identity line has ``p = 1``, a perfect fit off it ``p = 0``.
    """
    check_confidence(confidence)
    b, c = _paired(baseline, candidate)
    n = {"pairs": len(b)}
    at_least(len(b), 3, "pairs", n=n)
    fit = fit_line(b, c, method=method, error_ratio=error_ratio)
    assert fit.covariance is not None
    stats = require_scipy()
    d = np.array([fit.intercept, fit.slope - 1.0])
    v = np.array(fit.covariance)
    df = len(b) - 2
    if not np.isfinite(v).all():
        raise Unassessable("the fit's covariance is not finite", n=n)
    if np.allclose(v, 0, rtol=0, atol=0):
        exact = bool(np.all(d == 0))
        statistic, p = (0.0, 1.0) if exact else (math.inf, 0.0)
    else:
        try:
            statistic = float(d @ np.linalg.solve(v, d)) / 2
        except np.linalg.LinAlgError:
            statistic = float(d @ np.linalg.pinv(v) @ d) / 2
        p = float(stats.f.sf(statistic, 2, df))
    half = float(stats.t.ppf(0.5 + confidence / 2, df))
    se_slope, se_intercept = fit.se_slope or 0.0, fit.se_intercept or 0.0
    low, high = fit.intercept - half * se_intercept, fit.intercept + half * se_intercept
    how = (
        "ordinary least squares of the candidate on the baseline"
        if method == "ols"
        else f"Deming regression, error variance ratio {error_ratio:g} (jackknife covariance)"
    )
    return TestResult(
        method="identity_fit",
        statistic=statistic,
        statistic_name="F",
        p_value=p,
        effect=fit.slope,
        effect_name="slope",
        ci_low=fit.slope - half * se_slope,
        ci_high=fit.slope + half * se_slope,
        confidence=confidence,
        n=n,
        assumptions=[how, "independent pairs", "slope = 1 and intercept = 0 under the null"],
        notes=[f"intercept {fit.intercept:.6g} ({low:.6g} to {high:.6g})"],
    )
