# Statistical inference

`qavf.inference` is a small statistics library shared by dataset comparisons and
rules. It takes arrays (or summary counts) and returns a typed `TestResult`. On top of
the tests it provides multiple-testing corrections, analytic and bootstrap intervals,
block bootstraps and HAC errors for dependent data, effect sizes and agreement
metrics, and the policies that combine **materiality** (is a difference large enough
to matter?) with **significance** (is it larger than chance explains?).

Distribution functions come from SciPy, the `stats` extra (`pip install
'qavf[stats]'`). Without SciPy, methods that need its distribution functions return
**not assessed** with that reason. Permutation-only methods (including global and
local Moran's I), percentile bootstraps and the effect sizes need no SciPy. Importing
array-based inference does not initialize the engine or report renderers; plugin
registry objects are loaded when requested.

## Calling the library

```python
import numpy as np
from qavf.inference import correct, decide, welch_t

rng = np.random.default_rng(1)
north = rng.normal(10, 1, 40)
groups = {
    "north": (north, north + rng.normal(0.8, 0.2, 40)),   # a real shift
    "south": (north, rng.normal(10, 1, 40)),              # new draws, same population
}
result = welch_t(*groups["north"])
print(result.p_value, result.effect, (result.ci_low, result.ci_high))

family = correct([welch_t(b, c) for b, c in groups.values()], "bh", alpha=0.05)
for (name, (b, c)), tested in zip(groups.items(), family):
    material = abs(c.mean() - b.mean()) > 0.1
    print(name, decide("material_and_significant", material, tested).outcome)
```

### Results

`TestResult` holds `method`, `statistic` (and `statistic_name`), `p_value`,
`adjusted_p` with `correction`, `family_size`, `alpha` and `significant` once a family
is corrected, `effect` and `effect_name` (always candidate relative to baseline),
`ci_low`/`ci_high` with `confidence`, `n` per side (or `pairs`), `alternative`,
`assumptions`, `dependence` (`none`, `block`, `hac`, `effective_n`, `spatial`), `seed`
and `resamples` for resampling tests, and `assessed`/`reason`/`notes`. It is an audit
snapshot: non-finite statistics (an infinite t for two different constant samples)
survive JSON transport intact. An assessed result requires a finite raw p-value in
`[0, 1]` and, when present, a finite adjusted p-value in the same range. Missing or
malformed probabilities make the result **not assessed**, with a reason; they remain
in the audit record. This invariant is also checked when copying a result and when
making a decision. `significant` is derived from the effective p-value and valid
`alpha`, so a stale supplied flag cannot override the probability.

`Significance` summarizes a family judged together (one measure across groups, one
shape check across series): the test, the decision policy, alpha, the correction, the
family size, how many members are significant, fail and are not assessed, and the
leading tests (bounded by the evidence limit). It is the type of every `significance`
slot in comparison results.

## Tests

Every test checks its preconditions and returns not assessed with the reason when
they fail: too few values, non-finite values, mismatched pairs, a dynamic range that
cannot be rescaled losslessly, SciPy missing, or work beyond `max_work`.
Location- and scale-invariant tests run on values shifted by one shared origin and
divided by one shared power of two (`qavf.stats.center_scale_observations`), so a
small variation around a large offset is not lost; effects and intervals are
reported in data units.

| Function | Component | Tests | Minimum | Explicit edge cases |
|---|---|---|---|---|
| `paired_t` | `paired_t` | mean of `candidate - baseline` | 2 pairs | no change: p = 1; one nonzero constant difference: p = 0 |
| `wilcoxon_signed_rank` | `wilcoxon` | differences symmetric about 0 (exact up to 50 nonzero differences without ties) | 1 pair | zero differences dropped (counted); none left: p = 1 |
| `sign_test` | `sign` | share of positive differences (exact binomial) | 1 pair | no nonzero difference: p = 1 |
| `paired_permutation` | `paired_permutation` | sign-flip permutation of the mean difference (exact when `2^n <= resamples`) | 2 pairs | |
| `paired_bootstrap` | `paired_bootstrap` | bootstrap of a statistic of the differences | 2 pairs | |
| `tost_paired`, `tost_independent`, `tost_from_moments` | (equivalence form of `paired_t`, `welch_t`) | TOST: difference inside a margin | 2 each | no spread: p = 0 inside, 1 outside |
| `bootstrap_equivalence` | (equivalence form of `bootstrap`) | bootstrap TOST of a statistic difference | 2 each | |
| `welch_t`, `welch_t_from_moments` | `welch_t` | difference of means, unequal variances | 2 each | equal constants: p = 1; different constants: p = 0 |
| `mann_whitney` | `mann_whitney` | one side tends to be larger (exact when a side has at most 8 values and no ties) | 2 each | all tied: p = 1 |
| `ks_two_sample` | `ks` | any distribution difference (SciPy `auto`) | 2 each | |
| `anderson_darling` | `anderson_darling` | common distribution, tail-sensitive (p-value capped to [0.001, 0.25] by SciPy's table, noted) | 2 each | all tied: p = 1 |
| `brown_forsythe` | `brown_forsythe` | equal spread (Levene around medians) | 2 each | no spread: p = 1; constant deviations that differ: p = 0 |
| `permutation` | `permutation` | difference of a statistic by relabelling (exact when the relabellings fit `resamples`) | 2 each | |
| `bootstrap_difference` | `bootstrap` | difference of a statistic (mean, median, sum, std, quantile) | 2 each | non-finite resamples: not assessed |
| `chi_square`, `g_test` | | homogeneity of category shares (Pearson, likelihood ratio) | a nonempty row per side | empty categories dropped; one category: p = 1; expected counts below 5 noted |
| `fisher_exact` | | exact 2x2 test, odds ratio with its interval | | |
| `mcnemar` | | paired binary change (exact binomial, or chi-square with continuity) | 1 pair | no discordant pair: p = 1 |
| `stuart_maxwell` | | marginal homogeneity of paired categories | 1 pair | unchanged categories dropped; none changing: p = 1; singular covariance: not assessed |
| `two_proportion` | `two_proportion` | two shares, pooled z | 1 trial each | every trial alike: p = 1 |
| `poisson_rate` | `poisson` | two rates per exposure, exact conditional (binomial) | | no events: p = 1; common exposure rescaling preserves the test; unrepresentable exposure ratio: not assessed |
| `kendall_tau` | | rank correlation with time (TMP-002) | | undefined tau: not assessed |
| `mann_kendall` | | monotone trend, tie-corrected, with Sen's slope and interval | 3 values | constant: p = 1 |
| `seasonal_kendall` | | Mann-Kendall summed over seasons, seasonal Sen slope | a season with 2 values | |
| `slope_difference` | | difference of two series' slopes on shared positions | 4 positions | exactly linear difference: p = 0 (or 1 for no slope) |
| `correlation_test` | | Fisher z of a correlation against `rho` | 4 pairs (effective size above 3) | constant series: not assessed |
| `compare_correlations` | | Fisher z of two independent correlations (`variance=1.06` for Spearman's) | 4 pairs each | |
| `identity_test` | | joint F test that the candidate-on-baseline line has slope 1 and intercept 0 (OLS, or Deming with a jackknife covariance) | 3 pairs | on the line exactly: p = 1; a perfect fit off it: p = 0 |
| `lag_test` | | gain in correlation of a lag over lag 0 (block bootstrap) | `minimum` positions | constant series: not assessed |
| `rank_correlation_test` | | Fisher z of Spearman's correlation against `rho` (Fieller-Hartley-Pearson variance `1.06 / (n - 3)`) | 4 pairs | constant side: not assessed |
| `morans_i` | | spatial autocorrelation from given weights (permutation) | 3 locations | constant values or no neighbours: not assessed |
| `local_morans_i` | | local Moran's I (LISA) per location with its quadrant (HH, LL, HL, LH), conditional permutation; returns a `LocalMoran` of raw p-values to correct | 3 locations | islands and constant values: not assessed |

Components (the second column) are the tests a comparison measure can select with
`test:`; the [component reference](comparison_reference.md#statistical-tests) lists
their parameters.

## Multiple testing

`adjust(p, method, family_size=None)` returns adjusted p-values for `bonferroni`,
`holm`, `bh` (Benjamini-Hochberg) and `by` (Benjamini-Yekutieli), checked against
[`statsmodels.stats.multitest.multipletests`][multipletests] to floating-point
precision; `correct(results, method, alpha)`
applies one to the assessed results of a family and records `adjusted_p`, the method,
the family size and `significant` on each. Adjusted p-values are monotone in the raw
ones, never below them and at most 1. A `family_size` larger than the p-values given
counts the members that were not tested as `p = 1`, which keeps every correction valid
when some members could not be assessed or did not need a test. Padding is implicit:
only the supplied `k` probabilities are stored and sorted, even when the family size
`m` is much larger. Holm/BH/BY therefore use `O(k log k)` time and `O(k)` memory;
Bonferroni is linear in `k`. BY computes its harmonic factor without an `m`-element
array. The returned array owns its storage.

Bonferroni and Holm control the family-wise error rate; BH the false discovery rate
under independence or positive dependence; BY under any dependence.

## Intervals, resampling and dependence

* **Analytic intervals**: `t_interval`, `normal_interval`.
* **Bootstrap intervals**: `percentile_interval` and `bca_interval` (bias-corrected
  and accelerated; the jackknife acceleration uses SciPy's per-sample formula for
  several samples).
* **Seeds**: every resampling test draws from `numpy.random.Generator(PCG64(seed))`
  and records the seed and resample count. Reproduce an audit with the same inputs,
  options, seed and locked numerical environment; [NumPy's compatibility policy][random]
  does not promise identical streams across every build, platform or changed call
  sequence. The batching regressions check the supported seeded behavior. Comparisons
  offset the seed by each member's position. The default seed is 0 and the default
  resample count 2,000 (spatial tests default to 999 permutations). Exact enumeration
  does not depend on a seed.
* **Work budget**: a test needing more than `max_work` (20,000,000 by default)
  estimated values processed is not assessed, with required and allowed work in its
  reason. Bootstrap and permutation work counts draws times observations; BCa adds
  leave-one-out work. Global Moran charges `B * (n + edges)` and local Moran charges
  `B * (n - 1 + edges)`, including locations without neighbours. Effective sample size
  admits the sum of its overlapping lag correlations before doing the quadratic
  default-lag calculation. These are algorithmic work estimates, not wall-clock or
  process-memory limits; arbitrary callback cost and external libraries can differ.
* **Memory**: test callers process bootstrap samples, exact permutations and
  leave-one-out samples in bounded batches. Exact enumeration remains exact when it
  fits the work budget. One-dimensional distributions can still use `O(B)` storage
  for intervals; local Moran retains tail counts and one result per location, and
  releases full permutation buffers after each draw. Batch limits bound sample
  matrices, not the entire process. The public `block_indices` allocation helper
  returns the full shape explicitly requested by its caller.
* **Block bootstrap**: `block_indices` draws circular (or moving) blocks. The default
  block length is `ceil(2 * n ** (1/3))`: the rate-optimal order `n^(1/3)`, with a
  constant chosen by simulation.
* **HAC**: `newey_west_slope` gives the Newey-West (Bartlett) standard error of an OLS
  slope with `floor(4 * (n / 100) ** (2 / 9))` lags by default and statsmodels'
  small-sample factor.

Sequential buckets are rarely independent, so the series tests are dependence-aware
by default:

| Where | Default | Alternatives |
|---|---|---|
| `slope_difference` (`series_shape` trend) | `block`: null-imposed circular block bootstrap standard error, blocks of `l` and `2l` extrapolated to remove the `O(1/l)` bias, Student t with `n - 2` df | `hac`, `none` |
| `series_shape` correlation | `effective_n`: Bartlett's effective sample size `n / (1 + 2 sum r_x(k) r_y(k))` over `n // 5` lags | `none` (the standalone `correlation_test` default) |
| `lag_test` (`series_shape` lag) | circular block bootstrap of the correlation gain | |
| `morans_i` | spatial permutation | |
| `local_morans_i` | conditional permutation (each location's value fixed, one draw shared by all locations) | |

Tests of groups and buckets treat the rows within one group as independent.

### Custom resampling statistics

A callback computes along the **last axis**: a one-dimensional sample must produce
one scalar, and a `(batch, n)` array must produce exactly `(batch,)`. All observed,
resampled and leave-one-out statistics must be finite. Wrong shapes and nonfinite
outputs produce not-assessed results; they are never broadcast into plausible
probabilities.

`permutation` and `bootstrap_difference` evaluate the callback on baseline and
candidate separately and compare `statistic(candidate) - statistic(baseline)`.
`paired_bootstrap` evaluates the callback on `candidate - baseline`. Arbitrary
callbacks receive those values in their **original measurement units**. For example,
`lambda a: (a > 10).mean(axis=-1)` retains its threshold, and a variance statistic
retains squared units. Only the known built-in mean uses internal numerical scaling.

```python
from qavf.inference import paired_bootstrap, permutation

fraction_above_limit = lambda a: (a > 10).mean(axis=-1)
test = permutation([1, 2, 3], [20, 21, 22], statistic=fraction_above_limit)
# Exact difference of exceedance fractions: 1.0, with a two-sided p-value of 0.1.

spread = paired_bootstrap(
    [0, 0, 0, 0, 0], [1, 5, 2, 8, 4], statistic=lambda a: a.var(axis=-1), name="variance"
)
```

Callbacks should be deterministic and independent of batch size. Use `interval="percentile"`
or `interval="bca"`; any other name is a configuration error. Resample counts must be
integers of at least two. Bootstrap p-values invert the percentile interval even when
BCa is requested for the reported confidence interval. QAVF's exact/random permutation
tail conventions are tested against [SciPy's permutation test][permutation].

### Calibration

`tests/test_inference_calibration.py` checks, with fixed seeds, that the false-rejection
rate under the null lies within 3.5 binomial standard deviations of alpha = 0.05.
Observed rates (400 replications each): Welch 0.043, Mann-Whitney 0.053, KS 0.050,
Anderson-Darling 0.070, Brown-Forsythe 0.050, paired t 0.035, Wilcoxon 0.055, sign
0.035, Poisson 0.025, two-proportion 0.050, permutation 0.048, bootstrap 0.033,
Mann-Kendall 0.040 and Fisher z 0.068; exact tests (sign, Poisson, KS) are
conservative by design.

On AR(1) series with autocorrelation 0.5 (1,000 replications), the slope-difference
test rejected a true null at these rates:

| Buckets | `block` | `hac` | `none` (i.i.d.) |
|---|---|---|---|
| 30 | 0.115 | 0.209 | 0.257 |
| 60 | 0.075 | 0.161 | 0.262 |
| 120 | 0.059 | 0.146 | 0.263 |
| 240 | 0.038 | 0.100 | 0.258 |

The i.i.d. test rejects about five times too often whatever the length, which is why
dependence handling matters; HAC errors stay liberal; the block bootstrap holds its
level from about 100 buckets and is moderately liberal for short series. The Fisher z
test of two independent AR(1) series (autocorrelation 0.7, 60 buckets) rejected at
0.053 with the effective sample size and 0.258 without. Benjamini-Hochberg kept the
false discovery rate at 0.033 in a family of 20 tests with 5 real effects.

Under spatial randomness (`tests/test_spatial_inference.py`, rook weights, fixed
seeds), global Moran's I on an 8 x 8 lattice (199 permutations, 2,000 replications)
rejected at 0.038 two-sided and 0.041 one-sided; local Moran's I on a 6 x 6 lattice
(1,999 permutations, 1,000 replications) rejected 0.050 of its 36,000 location tests
before correction, and some location of a lattice was named significant in 0.026 of
the replications after Benjamini-Hochberg (0.023 with Holm, 0 with
Benjamini-Yekutieli). The Spearman rank test rejected at 0.044 with 25 pairs and with
10 pairs (4,000 replications).

## Effect sizes and agreement

`qavf.inference.effects` (no SciPy needed): `cohen_d`, `hedges_g` (exact small-sample
correction), `rank_biserial`, `cramers_v`, `rate_ratio`, `odds_ratio`; and for paired
values `bland_altman` (bias and limits of agreement), `concordance` (Lin's CCC),
`bias`, `mae`, `rmse`, `mape`, `smape` and `r_squared` (the candidate as a prediction
of the baseline). Undefined values (a zero spread, a zero baseline for MAPE) are
`None`.

Distribution distances compare two samples as a whole: `population_stability` (PSI of
per-bin counts, STA-002's formula with its `1e-6` floor), `jensen_shannon` (the
divergence in bits on shared bins; SciPy's `jensenshannon` distance is its square
root), `ks_distance`, `wasserstein` (earth mover's distance in data units, as SciPy's
`wasserstein_distance`) and `quantile_deltas`. `qavf.statistics.ordered.quantile_cutpoints`
gives the bins: the reference's quantiles with open support tails.

`qavf.inference.agreement.fit_line` fits `candidate = intercept + slope * baseline` by
OLS (the baseline taken as exact) or by Deming regression (`error_ratio`: the
candidate's error variance over the baseline's; 1 is orthogonal regression), with the
covariance of the estimates (analytic, or a jackknife computed from leave-one-out sums
in linear time); `identity_test` tests it against the identity line.

## Decision policies

`decide(policy, material, test)` combines materiality and significance:

| Policy | A difference fails when | Notes |
|---|---|---|
| `material_and_significant` (default) | it is material **and** significant after the correction | an immaterial difference passes without a test |
| `material` | it is material | runs no test; for a core install without the stats extra |
| `significant` | it is significant, however small | |
| `equivalence` | equivalence within the materiality margin is **not** shown | passes only when a TOST-style test shows the difference inside `max(atol, rtol * max(abs(b), abs(c)))` |

Two situations stay apart. A quantity with **no sound test** (a minimum, maximum or
distinct count) is judged by materiality under every policy, and the decision says so
(`basis="materiality"`). A test that **should apply but cannot run** (no SciPy, too few
values, a failed precondition, the work budget) makes the decision **not assessed**
whenever the policy needs it; it is never a pass.

## In dataset comparisons

Comparisons and each of their levels take `decision`, `alpha` (0.05), `correction`
(`bh`), `seed` (0) and `max_work`; a level's own value overrides the comparison's.
Each check is one family: the groups of one measure, the buckets of one temporal
measure, the zones of one spatial measure, of the `shares` check or of `lisa`, the
series of one shape check, the columns (and `shift_by` groups) of the record shift
check. Failing members carry `test`, `p_value`, `adjusted_p`, `effect`, `ci_low`,
`ci_high` and the decision's reason as evidence; the check's `significance` and
`metrics` summarize the family. Under `material`, no test runs and every slot stays
empty, exactly as before inference.

The `spatial` level's pattern checks use the spatial tests: `rank` the Spearman
test, `moran` the global Moran's I of the per-zone differences (one-sided: are they
clustered?), and `lisa` the local Moran's I of each zone, corrected across zones, so
the zones in significant clusters of material change are named
([spatial data](spatial.md#spatial)).

Each measure names its default test, which `test:` overrides (`test: none` declares
that no test applies):

| Measure | Default test | Reads per group |
|---|---|---|
| `mean` | `welch_t` | count, mean and variance, reduced natively |
| `median` | `mann_whitney` | the values |
| `quantile` | `bootstrap` (of the quantile) | the values |
| `sum` of an integer column | `poisson` | the sums |
| `sum` of other columns | `bootstrap` (of the sum) | the values |
| `count` | `poisson` | the counts |
| `share` | `two_proportion` (of the side's total) | rows or integer sums |
| `null_count` | `two_proportion` (of the group's rows) | missing values and rows |
| `std` | `brown_forsythe` | the values |
| `min`, `max`, `distinct_count`, `weighted_mean` | none: judged by materiality | |

A distribution comparison is `test: ks` or `test: anderson_darling` on any numeric
measure of the column. Paired tests apply to the `records` level only; a test whose
input a measure cannot provide (a Poisson test of means, a bootstrap of a count) is a
configuration error. See [dataset comparison](comparison.md#significance) for each
level.

## In rules

TMP-002 (Kendall), TMP-003 (studentized residuals) and STA-002 (KS, and PSI with the
shared binning) use the shared tests, distances and corrections. Each has a `correction` (TMP-002 `none`, TMP-003 and STA-002
`bonferroni`, their behaviour before) and reports `p_value`, `adjusted_p` and `effect`
in its evidence; their default results are unchanged.

## Writing a test plugin

A test is a registered component. Declare what it reads (`needs`: `samples`,
`moments`, `counts`, `proportions` or `paired`) and implement `run`; `equivalent`
adds an equivalence form.

```python
from typing import ClassVar

from qavf.inference import InferenceTest, TestInput, TestResult, register_test
from qavf.inference.independent import bootstrap_difference


@register_test
class TrimmedMean(InferenceTest):
    """Bootstrap of the difference of 10% trimmed means."""

    name: ClassVar[str] = "trimmed_mean"
    needs: ClassVar = "samples"

    def run(self, data: TestInput) -> TestResult:
        from scipy.stats import trim_mean

        return bootstrap_difference(
            data.baseline, data.candidate,
            statistic=lambda a: trim_mean(a, 0.1, axis=-1), name="trimmed_mean",
            seed=data.seed, max_work=data.max_work,
        )
```

```yaml
measures: [{measure: mean, column: speed, test: trimmed_mean}]
```

Packaged plugins advertise tests under the `qavf.inference.tests` entry-point group.
They follow `plugins.entry_points` exactly like comparison components, are recorded in
`metadata.plugins` as `inference.tests:<name>`, and `plugins.required` accepts
`inference.tests:<name>` ([plugins](plugins.md#statistical-tests)).

## Limitations

* Rows within a group or bucket are treated as independent; a temporal comparison of
  autocorrelated rows inside one bucket can be liberal.
* Short autocorrelated series (fewer than about 100 buckets) remain moderately liberal
  even with the block bootstrap (see the calibration table).
* Spearman's correlation has no test in `series_shape` (its buckets are dependent);
  a low Spearman correlation alone is judged by its threshold there. Peaks, profiles
  and level shifts have no test. The `spatial` level tests the zones' Spearman
  correlation with `rank_correlation_test`, treating zones as independent pairs.
* Moran's I and LISA take explicit neighbour weights; `qavf.spatial.weights` builds
  them from polygons, grid cells, centroids or tables ([spatial data](spatial.md)).
* A singular Stuart–Maxwell covariance, including some disconnected category-change
  patterns, needs a rank-aware method with appropriate degrees of freedom. QAVF
  reports it as not assessed instead of substituting an unsupported p-value.
* Numerical scaling cannot represent every possible dynamic range. Poisson tests
  normalize exposures before forming their null share; exposure-ratio underflow is
  explicit not assessed. Effect ratios too large for a float are `None`, and
  unbounded interval endpoints remain infinite.
* LISA p-values are no smaller than `1 / (permutations + 1)`; a large family of zones
  needs more permutations for the correction to name any cluster.


## Numerical and implementation references

The regression oracles are `tests/test_review_inference_fixes.py`,
`tests/test_inference_oracles.py`, and `tests/test_spatial_inference.py`. They cover
explicit family padding, exact relabellings, arbitrary callback units, seeded batch
invariance, invalid probabilities and bounded intermediate allocations. They complement
the calibration tests; allocation bounds are not a proof of universal optimality.

[multipletests]: https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
[permutation]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
[random]: https://numpy.org/doc/stable/reference/random/compatibility.html
