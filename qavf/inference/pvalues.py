"""p-values of standard reference distributions (SciPy's), for scalars or arrays.

``alternative`` is ``two-sided``, ``greater`` (large statistics are extreme) or
``less``. Two-sided p-values double the smaller tail and never exceed 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qavf.inference._support import require_scipy
from qavf.inference.results import Alternative


def _tails(distribution: Any, statistic: Any, alternative: Alternative, *shape: float) -> Any:
    if alternative == "greater":
        return distribution.sf(statistic, *shape)
    if alternative == "less":
        return distribution.cdf(statistic, *shape)
    return np.minimum(1.0, 2 * distribution.sf(np.abs(statistic), *shape))


def t_p_value(statistic: Any, df: float, alternative: Alternative = "two-sided") -> Any:
    """p-value(s) of Student t statistic(s) with ``df`` degrees of freedom."""
    stats = require_scipy()
    p = _tails(stats.t, statistic, alternative, df)
    return float(p) if np.ndim(p) == 0 else p


def normal_p_value(statistic: Any, alternative: Alternative = "two-sided") -> Any:
    """p-value(s) of standard normal statistic(s)."""
    stats = require_scipy()
    p = _tails(stats.norm, statistic, alternative)
    return float(p) if np.ndim(p) == 0 else p
