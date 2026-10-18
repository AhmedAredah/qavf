"""Statistical outlier detection primitives used by the QA/QC rules.

Two equivalent interfaces are provided:

* array functions (:func:`detect_outliers` and friends) take a 1-D array,
  ignore NaNs and return an :class:`OutlierResult` aligned with the input;
* :func:`flag_outliers` applies the same methods to every group of a Polars
  frame in one vectorised pass, which is what the rules use.

Samples that are too small or have no spread yield no flags rather than
raising, so both can be applied to arbitrary groups without special-casing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import polars as pl

from qavf._observations import float_observations
from qavf.numeric import checked_float

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Scales the MAD so it estimates the standard deviation of a normal sample.
_MAD_NORMAL_CONSISTENCY = 0.6745
#: Scales the mean absolute deviation into a standard-deviation estimate (normal data).
_MEAN_AD_NORMAL_CONSISTENCY = 1.253314


class OutlierMethod(StrEnum):
    """Supported outlier detection methods."""

    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    IQR = "iqr"


#: Conventional threshold for each method (z, modified z, IQR multiplier).
DEFAULT_THRESHOLDS: dict[OutlierMethod, float] = {
    OutlierMethod.ZSCORE: 3.0,
    OutlierMethod.MODIFIED_ZSCORE: 3.5,
    OutlierMethod.IQR: 1.5,
}

#: Fewest non-null observations for which each method is meaningful.
MIN_SAMPLES: dict[OutlierMethod, int] = {
    OutlierMethod.ZSCORE: 3,
    OutlierMethod.MODIFIED_ZSCORE: 3,
    OutlierMethod.IQR: 4,
}


@dataclass(frozen=True, slots=True)
class OutlierResult:
    """Outcome of an outlier scan.

    Attributes:
        mask: True where the observation is an outlier.
        scores: Method-specific score per observation (z-score, modified
            z-score, or distance beyond the nearest fence in IQR units).
        lower: Lower acceptance bound in data units (NaN when undefined).
        upper: Upper acceptance bound in data units (NaN when undefined).
        method: Method that produced the result.
        threshold: Threshold that was applied.
    """

    mask: BoolArray
    scores: FloatArray
    lower: float
    upper: float
    method: OutlierMethod
    threshold: float

    @property
    def count(self) -> int:
        """Number of flagged observations."""
        return int(self.mask.sum())


def zscore_min_samples(threshold: float) -> int:
    """Fewest values for which a z-score can exceed ``threshold`` at all.

    With the sample standard deviation the largest possible |z| in ``n`` values is
    ``(n - 1) / sqrt(n)`` (one value against ``n - 1`` equal others), so for ``n = 10``
    no value can ever pass the default threshold of 3. Smaller samples cannot be
    tested by the z-score and are reported as untested rather than as clean.
    """
    limit = check_threshold(threshold)
    # Solve b²(n-1)² > a²n exactly for threshold=a/b. Integer arithmetic
    # avoids both a threshold-dependent loop and overflow for large finite floats.
    a, b = limit.as_integer_ratio()
    aa, bb = a * a, b * b
    discriminant = aa * aa + 4 * aa * bb
    boundary = (aa + 2 * bb + math.isqrt(discriminant)) // (2 * bb)
    return max(MIN_SAMPLES[OutlierMethod.ZSCORE], boundary + 1)


def check_threshold(threshold: float) -> float:
    """A method threshold must be a finite number > 0 (NaN would flag nothing in one
    implementation and everything in another).

    Raises:
        ValueError: otherwise.
    """
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float | np.integer | np.floating)
        or not (math.isfinite(threshold) and threshold > 0)
    ):
        raise ValueError(f"threshold must be a finite number > 0, got {threshold!r}")
    return float(threshold)


def checked_float_series(series: pl.Series) -> pl.Series:
    """Convert numeric observations only when finite values round-trip exactly.

    Statistical kernels use Float64; silently rounding large integers or fixed
    precision decimals can erase the variation being checked.
    """
    if not series.dtype.is_numeric():
        raise ValueError(f"statistical values must be numeric, found {series.dtype}")
    converted = series.cast(pl.Float64)
    if series.dtype.is_decimal():
        for value in series.drop_nulls():
            checked_float(value)
    if series.dtype.is_integer():
        restored = converted.cast(series.dtype, strict=False)
        lost = series.is_not_null() & (restored.is_null() | (restored != series))
        if lost.any():
            raise ValueError(
                "values cannot be represented exactly as Float64; "
                "rescale or centre them explicitly before statistical assessment"
            )
    return converted


def _observations(values: npt.ArrayLike, threshold: float) -> tuple[FloatArray, float]:
    """Validated 1-D float observations and threshold for the array detectors."""
    return float_observations(values), check_threshold(threshold)


def _power_of_two_scale(magnitude: float) -> float:
    """A power of two near ``magnitude`` (1.0 for zero).

    Every score here is invariant to rescaling the data, so each sample is
    divided by this before its statistics are computed: sums and squares then
    cannot overflow (values near 1e308) or lose precision (subnormal values),
    and dividing by a power of two is exact, so ordinary data gives bit-identical
    results.
    """
    if magnitude == 0 or not math.isfinite(magnitude):
        return 1.0
    _, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, exponent - 1)


def _scaled(
    x: FloatArray, finite: BoolArray, scale: float | None = None
) -> tuple[FloatArray, float]:
    if scale is None:
        scale = _power_of_two_scale(float(np.abs(x[finite]).max()))
    # Multiply by 1 / scale in two exact steps: the reciprocal of a subnormal scale
    # is not representable, and dividing by it overflows on some platforms.
    exponent = math.frexp(scale)[1] - 1  # scale == 2 ** exponent
    half = -exponent // 2
    y = x * math.ldexp(1.0, half) * math.ldexp(1.0, -exponent - half)
    if not np.array_equal(y[finite] * scale, x[finite]):
        raise ValueError("observation dynamic range cannot be rescaled without precision loss")
    return y, scale


def scale_observations(
    values: FloatArray, *, scale: float | None = None
) -> tuple[FloatArray, float]:
    """Power-of-two scaling for finite statistical kernels; reject information loss."""
    finite = np.isfinite(values)
    if values.ndim != 1 or not finite.all() or not len(values):
        raise ValueError("scaling requires a nonempty finite one-dimensional sample")
    if scale is not None and (
        not math.isfinite(scale) or scale <= 0 or math.frexp(scale)[0] != 0.5
    ):
        raise ValueError("a shared scale must be a finite positive power of two")
    return _scaled(values, finite, scale)


def _centered_scaled(x: FloatArray, finite: BoolArray) -> tuple[FloatArray, float, float]:
    """Subtract a nearby origin before statistics, then scale without data loss.

    Centering first preserves small variation around a large common offset.
    Zero is used for a sample spanning zero, avoiding subtraction overflow.
    The reconstruction checks make unsupported dynamic range explicit.
    """
    lo, hi = float(x[finite].min()), float(x[finite].max())
    origin = lo if lo >= 0 else hi if hi <= 0 else 0.0
    shifted = x - origin
    if not np.array_equal((shifted + origin)[finite], x[finite]):
        origin, shifted = 0.0, x
    y, scale = _scaled(shifted, finite)
    return y, scale, origin


def center_scale_observations(values: FloatArray) -> tuple[FloatArray, float, float]:
    """Preserve offsets and spread in a finite one-dimensional statistical sample."""
    finite = np.isfinite(values)
    if values.ndim != 1 or not finite.all() or not len(values):
        raise ValueError("centering requires a nonempty finite one-dimensional sample")
    return _centered_scaled(values, finite)


def _in_data_units(value: float, scale: float, origin: float = 0.0) -> float:
    with np.errstate(over="ignore"):
        return float(np.float64(value) * scale + origin)  # may be infinite: beyond float range


def _empty(values: FloatArray, method: OutlierMethod, threshold: float) -> OutlierResult:
    return OutlierResult(
        mask=np.zeros(values.shape, dtype=bool),
        scores=np.full(values.shape, np.nan),
        lower=float("nan"),
        upper=float("nan"),
        method=method,
        threshold=threshold,
    )


def zscore_outliers(values: npt.ArrayLike, threshold: float = 3.0) -> OutlierResult:
    """Flag observations whose absolute z-score exceeds ``threshold``.

    Uses the sample standard deviation (``ddof=1``). Sensitive to masking by
    the outliers themselves; prefer :func:`modified_zscore_outliers` for small
    samples. A sample whose values are all identical has no outliers.
    """
    x, threshold = _observations(values, threshold)
    finite = np.isfinite(x)
    if finite.sum() < zscore_min_samples(threshold):  # too few for any value to be flagged
        return _empty(x, OutlierMethod.ZSCORE, threshold)
    y, scale, origin = _centered_scaled(x, finite)
    mean = math.fsum(y[finite]) / int(finite.sum())
    std = math.sqrt(math.fsum((float(v) - mean) ** 2 for v in y[finite]) / (int(finite.sum()) - 1))
    if not (math.isfinite(mean) and math.isfinite(std)):
        return _empty(x, OutlierMethod.ZSCORE, threshold)  # numerically untestable
    if std == 0.0:  # every value identical
        return _identical(x, finite, mean * scale + origin, OutlierMethod.ZSCORE, threshold)
    scores = np.where(finite, (y - mean) / std, np.nan)
    mask = finite & (np.abs(scores) > threshold)
    return OutlierResult(
        mask,
        scores,
        _in_data_units(mean - threshold * std, scale, origin),
        _in_data_units(mean + threshold * std, scale, origin),
        OutlierMethod.ZSCORE,
        threshold,
    )


def modified_zscore_outliers(values: npt.ArrayLike, threshold: float = 3.5) -> OutlierResult:
    """Flag observations using the median/MAD modified z-score (Iglewicz & Hoaglin).

    Robust to the outliers it is looking for, which makes it the better choice
    for the handful-of-entities samples typical of regional data.

    When more than half the values are tied the MAD is zero, which would make
    every departure from the tie look infinitely large or (if skipped) invisible.
    The standard fallback is used instead: the mean absolute deviation from the
    median, scaled by 1.253314 so it also estimates the standard deviation of a
    normal sample. Only a sample whose values are all identical has no spread.
    """
    x, threshold = _observations(values, threshold)
    finite = np.isfinite(x)
    if finite.sum() < MIN_SAMPLES[OutlierMethod.MODIFIED_ZSCORE]:
        return _empty(x, OutlierMethod.MODIFIED_ZSCORE, threshold)
    y, unit, origin = _centered_scaled(x, finite)
    median = float(np.median(y[finite]))
    deviations = np.abs(y[finite] - median)
    mad = float(np.median(deviations))
    if mad > 0.0:
        scale = mad / _MAD_NORMAL_CONSISTENCY
    else:
        mean_deviation = float(deviations.mean())
        if mean_deviation == 0.0:  # every value identical
            return _identical(
                x, finite, median * unit + origin, OutlierMethod.MODIFIED_ZSCORE, threshold
            )
        scale = _MEAN_AD_NORMAL_CONSISTENCY * mean_deviation
    if not (math.isfinite(median) and math.isfinite(scale)):
        return _empty(x, OutlierMethod.MODIFIED_ZSCORE, threshold)  # numerically untestable
    scores = np.where(finite, (y - median) / scale, np.nan)
    mask = finite & (np.abs(scores) > threshold)
    return OutlierResult(
        mask,
        scores,
        _in_data_units(median - threshold * scale, unit, origin),
        _in_data_units(median + threshold * scale, unit, origin),
        OutlierMethod.MODIFIED_ZSCORE,
        threshold,
    )


def iqr_outliers(values: npt.ArrayLike, threshold: float = 1.5) -> OutlierResult:
    """Flag observations outside Tukey's fences ``[Q1 - k*IQR, Q3 + k*IQR]``.

    ``scores`` is the signed distance beyond the nearest fence in IQR units
    (zero inside the fences). When Q1 equals Q3 the fences still exist (both
    equal Q1), so every value away from it is flagged; its score is then
    infinite, signed by direction, because there is no spread to measure in.
    """
    x, threshold = _observations(values, threshold)
    finite = np.isfinite(x)
    if finite.sum() < MIN_SAMPLES[OutlierMethod.IQR]:
        return _empty(x, OutlierMethod.IQR, threshold)
    y, unit, origin = _centered_scaled(x, finite)
    q1, q3 = (float(q) for q in np.percentile(y[finite], [25, 75]))
    spread = q3 - q1
    lower, upper = q1 - threshold * spread, q3 + threshold * spread
    if not all(math.isfinite(v) for v in (lower, upper)):
        return _empty(x, OutlierMethod.IQR, threshold)  # numerically untestable
    with np.errstate(divide="ignore", invalid="ignore"):
        above = (y - upper) / spread if spread else np.where(y > upper, np.inf, 0.0)
        below = (y - lower) / spread if spread else np.where(y < lower, -np.inf, 0.0)
    scores = np.where(y > upper, above, np.where(y < lower, below, 0.0))
    scores = np.where(finite, scores, np.nan)
    mask = finite & ((y < lower) | (y > upper))
    return OutlierResult(
        mask,
        scores,
        _in_data_units(lower, unit, origin),
        _in_data_units(upper, unit, origin),
        OutlierMethod.IQR,
        threshold,
    )


def _identical(
    x: FloatArray, finite: BoolArray, centre: float, method: OutlierMethod, threshold: float
) -> OutlierResult:
    """Result for a sample whose finite values are all equal: tested, nothing flagged."""
    return OutlierResult(
        mask=np.zeros(x.shape, dtype=bool),
        scores=np.where(finite, 0.0, np.nan),
        lower=centre,
        upper=centre,
        method=method,
        threshold=threshold,
    )


@dataclass(frozen=True, slots=True)
class OutlierColumns:
    """Names of the result columns :func:`flag_outliers` appended.

    Each is the plain name (``score``, ``is_outlier``, …) unless the input frame
    already has a column of that name, in which case a ``qavf_``-prefixed name
    is used so no user column is ever overwritten.
    """

    score: str
    lower_bound: str
    upper_bound: str
    is_outlier: str
    group_size: str
    group_median: str


@dataclass(frozen=True, slots=True)
class OutlierScan:
    """Result of :func:`flag_outliers`."""

    frame: pl.DataFrame
    """The usable input rows, every original column unchanged, plus the result columns."""
    columns: OutlierColumns
    """Where the result columns are in :attr:`frame`."""

    @property
    def flags(self) -> pl.Series:
        """Boolean outlier flag per row of :attr:`frame`."""
        return self.frame[self.columns.is_outlier]

    @property
    def scores(self) -> pl.Series:
        """Method score per row (null where the group could not be tested)."""
        return self.frame[self.columns.score]

    @property
    def tested(self) -> int:
        """Rows that belonged to a testable group."""
        return int(self.scores.is_not_null().sum())

    @property
    def outliers(self) -> pl.DataFrame:
        """The flagged rows, without the flag and group-size bookkeeping columns."""
        return self.frame.filter(self.flags).drop(self.columns.is_outlier, self.columns.group_size)


_RESULT_ROLES = ("score", "lower_bound", "upper_bound", "is_outlier", "group_size", "group_median")


def _fresh_name(name: str, taken: set[str]) -> str:
    while name in taken:
        name = f"qavf_{name}"
    taken.add(name)
    return name


def minimum_samples(method: OutlierMethod | str, threshold: float) -> int:
    """Fewest values per group for ``method`` at ``threshold`` to be able to flag one."""
    resolved = OutlierMethod(method)
    if resolved is OutlierMethod.ZSCORE:
        return zscore_min_samples(threshold)
    return MIN_SAMPLES[resolved]


def flag_outliers(
    frame: pl.DataFrame,
    value: str,
    group_by: Sequence[str] = (),
    method: OutlierMethod | str = OutlierMethod.MODIFIED_ZSCORE,
    threshold: float | None = None,
    min_group_size: int = 0,
) -> OutlierScan:
    """Vectorised, grouped counterpart of :func:`detect_outliers`.

    Scans ``value`` within each ``group_by`` group in a single Polars pass, so
    it scales to large tables with many groups where calling
    :func:`detect_outliers` per group would not. Rows with a null or non-finite
    value are dropped. Semantics match the array functions exactly.

    The computation runs on a private projection of the value and group
    columns, so the verdict never depends on what the input columns are called
    and no input column is read after it could have been overwritten.

    Args:
        frame: Input table.
        value: Numeric column to scan.
        group_by: Columns defining the groups; the whole frame when empty.
        method: Detection method.
        threshold: Method threshold; defaults to :data:`DEFAULT_THRESHOLDS`.
        min_group_size: Groups with fewer usable values are never flagged. The
            method's own minimum (:data:`MIN_SAMPLES`) always applies.
    """
    resolved = OutlierMethod(method)
    limit = check_threshold(DEFAULT_THRESHOLDS[resolved] if threshold is None else threshold)
    if isinstance(min_group_size, bool) or not isinstance(min_group_size, int):
        raise ValueError(f"min_group_size must be an integer, got {min_group_size!r}")
    if min_group_size < 0:
        raise ValueError(f"min_group_size must be >= 0, got {min_group_size}")
    minimum = min(frame.height + 1, max(min_group_size, minimum_samples(resolved, limit)))

    # A private namespace: nothing below can collide with the caller's columns.
    float_values = checked_float_series(frame[value])
    taken = set(frame.columns)
    row = _fresh_name("__qavf_row", taken)
    keys = [f"__qavf_group_{index}" for index in range(len(group_by))]
    work = frame.select(
        float_values.alias("__x"),
        *(pl.col(column).alias(alias) for column, alias in zip(group_by, keys, strict=True)),
    ).with_row_index(row)
    work = work.filter(pl.col("__x").is_not_null() & pl.col("__x").is_finite())

    def over(expr: pl.Expr) -> pl.Expr:
        return expr.over(keys) if keys else expr

    # The same power-of-two rescaling as the array functions (see _power_of_two_scale):
    # statistics are computed on __y = __x / __unit; bounds go back to data units.
    # Polars divides by multiplying with the reciprocal, which overflows for a
    # subnormal divisor, so __x is multiplied by 2**-e in two exact, finite steps.
    lo, hi = over(pl.col("__x").min()), over(pl.col("__x").max())
    work = work.with_columns(
        pl.when(lo >= 0).then(lo).when(hi <= 0).then(hi).otherwise(0.0).alias("__origin")
    ).with_columns((pl.col("__x") - pl.col("__origin")).alias("__shifted"))
    work = work.with_columns(
        pl.when(over(((pl.col("__shifted") + pl.col("__origin")) != pl.col("__x")).any()))
        .then(0.0)
        .otherwise(pl.col("__origin"))
        .alias("__origin")
    ).with_columns((pl.col("__x") - pl.col("__origin")).alias("__shifted"))
    magnitude = over(pl.col("__shifted").abs().max())
    # log2(max_float) rounds to 1024; its power is not representable.
    exponent = pl.when(magnitude > 0).then(magnitude.log(2).floor()).otherwise(0.0)
    exponent = exponent.clip(-1074.0, 1023.0)
    half = (-exponent / 2).floor()
    work = work.with_columns(
        pl.lit(2.0).pow(exponent).alias("__unit"),
        (pl.col("__shifted") * pl.lit(2.0).pow(half) * pl.lit(2.0).pow(-exponent - half)).alias(
            "__y"
        ),
    )
    x = pl.col("__y")
    unit = pl.col("__unit")

    staged = work.with_columns(
        over(pl.len()).alias("__size"),
        over(x.median()).alias("__median"),
        over(
            (
                ((x * unit) != pl.col("__shifted"))
                | ((pl.col("__shifted") + pl.col("__origin")) != pl.col("__x"))
            ).any()
        ).alias("__precision_lost"),
    )
    # Zero spread is handled exactly as in the array functions: identical values
    # are tested and clean; a tied majority does not hide departures from it.
    spread = pl.col("__spread")
    match resolved:
        case OutlierMethod.ZSCORE:
            staged = staged.with_columns(
                over(x.mean()).alias("__centre"), over(x.std(ddof=1)).alias("__spread")
            )
            score = pl.when(spread > 0).then((x - pl.col("__centre")) / spread).otherwise(0.0)
            lower = pl.col("__centre") - limit * spread
            upper = pl.col("__centre") + limit * spread
            outlier = score.abs() > limit
        case OutlierMethod.MODIFIED_ZSCORE:
            deviation = (x - pl.col("__median")).abs()
            staged = staged.with_columns(
                over(deviation.median()).alias("__mad"), over(deviation.mean()).alias("__mean_ad")
            ).with_columns(
                pl.col("__median").alias("__centre"),
                pl.when(pl.col("__mad") > 0)
                .then(pl.col("__mad") / _MAD_NORMAL_CONSISTENCY)
                .otherwise(_MEAN_AD_NORMAL_CONSISTENCY * pl.col("__mean_ad"))
                .alias("__spread"),
            )
            score = pl.when(spread > 0).then((x - pl.col("__centre")) / spread).otherwise(0.0)
            lower = pl.col("__centre") - limit * spread
            upper = pl.col("__centre") + limit * spread
            outlier = score.abs() > limit
        case OutlierMethod.IQR:
            staged = staged.with_columns(
                over(x.quantile(0.25, interpolation="linear")).alias("__q1"),
                over(x.quantile(0.75, interpolation="linear")).alias("__q3"),
            ).with_columns((pl.col("__q3") - pl.col("__q1")).alias("__spread"))
            lower = pl.col("__q1") - limit * spread
            upper = pl.col("__q3") + limit * spread
            score = (
                pl.when(x > upper)
                .then(pl.when(spread > 0).then((x - upper) / spread).otherwise(float("inf")))
                .when(x < lower)
                .then(pl.when(spread > 0).then((x - lower) / spread).otherwise(float("-inf")))
                .otherwise(0.0)
            )
            outlier = (x < lower) | (x > upper)
    # A group whose statistics are not finite cannot be tested (never a verdict).
    testable = (
        (pl.col("__size") >= minimum)
        & ~pl.col("__precision_lost")
        & pl.col("__spread").is_finite()
        & lower.is_finite()
        & upper.is_finite()
    )

    names = OutlierColumns(**{role: _fresh_name(role, taken) for role in _RESULT_ROLES})
    results = staged.select(
        pl.col(row),
        pl.when(testable).then(score).otherwise(None).alias(names.score),
        pl.when(testable)
        .then(lower * unit + pl.col("__origin"))
        .otherwise(None)
        .alias(names.lower_bound),
        pl.when(testable)
        .then(upper * unit + pl.col("__origin"))
        .otherwise(None)
        .alias(names.upper_bound),
        (testable & outlier).fill_null(False).alias(names.is_outlier),
        pl.col("__size").alias(names.group_size),
        (pl.col("__median") * unit + pl.col("__origin")).alias(names.group_median),
    )
    # Filtering and window expressions preserve row order. Reattach by alignment,
    # avoiding a full indexed join and sort of every original column.
    original = frame.filter(pl.col(value).cast(pl.Float64).is_finite().fill_null(False))
    return OutlierScan(original.hstack(results.drop(row)), names)


def detect_outliers(
    values: npt.ArrayLike,
    method: OutlierMethod | str = OutlierMethod.MODIFIED_ZSCORE,
    threshold: float | None = None,
) -> OutlierResult:
    """Dispatch to the detector for ``method``.

    Args:
        values: 1-D numeric observations; NaNs are ignored.
        method: Detection method.
        threshold: Method threshold; defaults to :data:`DEFAULT_THRESHOLDS`.
    """
    resolved = OutlierMethod(method)
    limit = DEFAULT_THRESHOLDS[resolved] if threshold is None else threshold
    match resolved:
        case OutlierMethod.ZSCORE:
            return zscore_outliers(values, limit)
        case OutlierMethod.MODIFIED_ZSCORE:
            return modified_zscore_outliers(values, limit)
        case OutlierMethod.IQR:
            return iqr_outliers(values, limit)
