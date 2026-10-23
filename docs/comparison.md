# Dataset comparison

`compare_datasets` (and the `CMP-001` rule) compares a *candidate* dataset with a
*baseline*: a new version of an output with the previous one, a migrated table with
its source, or two producers of the same quantity. The datasets may differ in column
names, types, units and codes; an **alignment pipeline** makes them comparable first,
then each configured **level** compares them at one level of detail.

```python
from qavf import compare_datasets

result = compare_datasets(
    "v1/inventory.parquet",
    "v2/inventory.parquet",
    keys=["county", "year", "pollutant"],
    columns={"emissions_t": "emissions"},          # baseline name -> candidate name
    units={"emissions": {"factor": 1000, "from_unit": "t", "to_unit": "kg"}},
    levels=[
        "structure",
        "coverage",
        {"level": "records", "top_n": 5},
        {"level": "groups", "group_by": ["pollutant"],
         "measures": ["count", "sum(emissions)", {"measure": "quantile", "column": "emissions", "p": 0.9}],
         "rank": {"measure": "sum(emissions)", "top_k": 5}},
    ],
    rtol=1e-6,
)
print(result.summary())
for level in result.levels:
    for check in level.checks:
        print(level.name, check.check, check.passed, check.message)
```

Inputs take every form `validate` accepts: paths, `File`, `Dataset` (its schema's
`missing_values`, `unit` and `nullable` declarations apply), `Query`, `Loader` and
Polars, pandas or Arrow frames. `resources=ResourceLimits(...)` applies the same
resource policy, including isolated execution for file and query inputs.

By default a difference fails only when it is **material and statistically
significant** ([significance](#significance)). Significance needs the `stats` extra
(SciPy); a core install should set `decision="material"`, which judges materiality
alone exactly as before and runs no test.

## Reusable typed configuration

Keep the simple keyword API for one-off comparisons. For a reusable contract,
pass an existing `ComparisonSpec` or a configuration mapping:

```python
from qavf import compare_datasets
from qavf.compare import ComparisonSpec

contract = ComparisonSpec(
    baseline="reference", candidate="release", keys=("site",),
    decision="material", rtol=0, atol=0,
)
result = compare_datasets("before.parquet", "after.parquet", configuration=contract)
```

The convenience function binds the contract to its two supplied inputs; the
contract's dataset names do not select external datasets. `configuration=` cannot
be combined with `keys`, `columns`, `levels`, `name`, or other comparison options.
Use one form so a requested check is never silently replaced. `resources` and
`entry_points` remain independent run controls.

## Fail-closed results

A `DatasetComparison` holds one typed result per level (`StructureResult`,
`CoverageResult`, `RecordsResult`, `GroupsResult`, `TemporalResult`, `ShapeResult`,
`SpatialResult`, `LocationsResult`, `DistributionResult`, `RelationshipsResult`,
`ContributionResult`),
each with graded checks. A check
has an exact `affected_count` and bounded `affected_records`: evidence follows the
run's `max_affected_records` and `max_evidence_bytes` limits and is ordered
deterministically, while counts stay complete.

Nothing that could not be compared reads as agreement. A dataset that cannot be read,
a mapped column the baseline lacks, a lossy cast, an unmapped category with
`unmapped: error`, fewer aligned rows than `min_rows` (1 by default), duplicate keys
for the record level or a non-numeric column for a numeric measure all leave the
affected checks **not assessed**. `result.passed` is true only when every check was
assessed and passed. An empty `levels=[]` selection is invalid. A plugin level
returning no checks, or a grouped measure with no shared members, is not assessed.
Location movement/geometry checks need matched features with usable locations;
`min_rows=0` permits empty input datasets but does not invent paired assessments.

## Alignment

Steps run in this order; after `columns`, every column reference (keys, tolerances,
level parameters) uses **candidate names**.

| Option | Step | Example |
|---|---|---|
| `columns` | Rename baseline columns to candidate names. A mapped column the baseline lacks refuses the alignment. | `{AADT: aadt}` |
| `cast` | Convert types losslessly (`int`, `float`, `decimal` with `scale`, `str`, `bool`, `date`, `datetime`) on `both` sides or one. A value that would change refuses the alignment. | `{site: str, rate: {dtype: decimal, scale: 4}}` |
| `units` | Multiply one side's column by an exact factor; integers and decimals stay exact. `from_unit` must match the declared `unit`; `to_unit` becomes it. | `{mass: {factor: 1000, to_unit: kg}}` |
| `recode` | Replace values through a map; unmapped values are kept, made null or refused. | `{fc: {values: {"1": interstate}}}` |
| `derive` | Add the same Polars SQL expression on both sides. | `{total: "cars + trucks"}` |
| `steps` | Plugin (or built-in) steps, `{step: name, ...}`, e.g. `shift_time` ([calendar alignment](#calendar-alignment)) or `assign_zones` ([in space](#in-space)). | `[{step: trim, columns: [name]}]` |
| `baseline_where`, `candidate_where`, `where` | Keep only rows in scope on one or both sides. | `"year >= 2020"` |

Dataset-level `rename`, `missing_values` and `canonical_columns` apply before the
pipeline, exactly as for every other rule.

## Levels

| Level | Compares | Checks |
|---|---|---|
| `structure` | Columns added or removed after mapping, suggested renames (similar names and similar value samples, never applied), type, declared-unit and nullability changes, row counts (exact unless `rows: {rtol: ...}`), key uniqueness per side | `columns`, `types`, `units`, `nullability`, `rows`, `keys` |
| `coverage` | Keys only in the baseline or only in the candidate, overall and by `group_by`; changes of each column's missing-value rate beyond `max_null_rate_change` | `keys`, `null rates` |
| `records` | Keyed match through the regression engine: rows added, removed, changed, unchanged; per-column change counts and rates; largest absolute and relative differences; null and NaN transitions; categorical transitions (`max_transitions` most frequent, with the total count); the `top_n` largest numeric changes beyond tolerance as evidence; a systematic shift of each numeric column (paired test, unless `decision: material`) | `rows`, `values`, `shift` |
| `groups` | Measures per `group_by` group on each side; groups on one side only; material changes per measure; optional top-k rank changes | `groups`, one per measure, `rank: <measure>` |
| `temporal` | Measures per time bucket (and `group_by` group) on each side: buckets on one side only, expected-grid gaps on each side, material changes per measure with the largest first | `buckets`, one per measure, `gaps` |
| `series_shape` | Each group's bucketed series on both sides: correlation, trend, cyclic profile, peak, lag, level shifts | `series`, `correlation`, `trend`, `profile`, `peak`, `lag`, `level shift` (and `too short`) |
| `spatial` | Measures per zone (an attribute or `assign_zones`): zones on one side only, expected zones a side lacks, material changes per measure, rank correlation, share redistribution, global Moran's I of the changes and LISA clusters of change | `zones`, one per measure, `expected`, `rank`, `shares`, `moran`, `lisa` |
| `locations` | Features matched by key: located on one side only, distance moved (metres for longitude/latitude), area and shape changes of geometries | `located`, `moved`, `area`, `shape` |
| `distribution` | Each column's distribution per member (group, zone, time bucket): PSI, KS, Anderson-Darling, Wasserstein, Jensen-Shannon, quantile deltas; categories that appeared or disappeared | `groups` (or `buckets`), one per column, `<column>: categories` |
| `relationships` | Correlation changes of column pairs, ratio invariants, agreement of keyed records (Bland-Altman, CCC, error metrics, the candidate-on-baseline line) | `correlation: <method>`, `ratio: <name>`, `agreement: <column>` |
| `contribution` | Each member's contribution to the total change along each dimension, the minimal set explaining a share, mix versus rate of declared rates | `total`, `contribution: <dimension>`, `rate: <name>` |

`allow_added` / `allow_removed` accept one-sided columns, keys or rows where they are
expected. Every level's parameters are listed in the
[component reference](comparison_reference.md).

### Measures and comparators

Measures: `count`, `sum`, `mean`, `median`, `quantile` (`p`), `min`, `max`, `std`,
`distinct_count`, `null_count`, `weighted_mean` (`weight`) and `share` (of the side's
total). Write `sum(x)` or `{measure: quantile, column: x, p: 0.9, name: p90}`.

Sums and means of integers and decimals are exact (overflow-checked 128-bit and
38-digit accumulators); sums and means of floats use the exact binary-float
accumulator; quantiles select order statistics natively and interpolate exactly. Only
`std` is computed in binary64. Missing values (null, NaN, declared markers) are left
out of every measure.

Comparators describe a change: `diff`, `abs_diff`, `rel_diff`, `ratio`, `log_ratio`
and `share_change` (change of the group's share of its side's total), `diff` and
`rel_diff` by default. They operate on exact rationals, so the difference of two large
integers is exact.

### Materiality

A change is material beyond `max(atol, rtol * max(|b|, |c|))`, evaluated exactly.
Tolerances resolve from the most specific setting: a measure's own `rtol`/`atol`/
`exact`, then the comparison's per-column `tolerances`, then the level's `rtol`/`atol`,
then the comparison's `rtol` (1e-6) and `atol` (1e-9). Record values use the
comparison's and per-column tolerances, with the regression engine's exact arithmetic.

Materiality is one half of the default decision; [significance](#significance) is
the other.

## Over time

Two versions of a time series can be compared at any granularity, by any groups,
and by shape:

```python
result = compare_datasets(
    "v1/counts.parquet",
    "v2/counts.parquet",
    levels=[
        {"level": "temporal", "time_column": "timestamp", "bucket": "day",
         "time_zone": "America/Chicago", "group_by": ["site"], "measures": ["sum(volume)"]},
        {"level": "temporal", "name": "hourly profile", "time_column": "timestamp",
         "bucket": "hour_of_day", "measures": ["mean(volume)"]},
        {"level": "series_shape", "time_column": "timestamp", "bucket": "hour",
         "group_by": ["site"], "measure": "sum(volume)",
         "profile": {"profile": "hour_of_day"}},
    ],
)
```

### Buckets

`bucket` turns the time column into a label per row, computed natively (the same
buckets serve the TMP series rules, see [rules](rules.md)):

| Kind | Values | Label |
|---|---|---|
| Calendar | `second`, `minute`, `hour`, `day`, `week` (`week_start`), `month`, `quarter`, `year`; multiples as `15 minutes`, `6h`, `{unit: month, every: 3}` | the bucket's first `Date` (a day or longer) or starting instant |
| Cyclic profile | `hour_of_day` (0-23), `day_of_week` (ISO 1 = Monday), `day_of_year`, `month_of_year`, `quarter_of_year`, `season` (meteorological; `hemisphere`, or your own `seasons: {wet: [5, 6, 7, 8, 9, 10], dry: [...]}`) | the position in the cycle |

Multiples of days and weeks count from 1970-01-01 (weeks from the `week_start` on or
before it), months, quarters and years from year 0, so `2 months` start in January,
March, ...; sub-day widths must divide a day. Without `bucket`, time values are used
as they are, so numeric periods such as a year work unchanged.

**Time zones.** Buckets follow the local calendar of `time_zone`: naive timestamps
are wall-clock times there, aware ones are converted (an aware column keeps its own
zone when `time_zone` is not set). A spring-forward day has 23 hours and a fall-back
day 25, and the two physical 01:00 hours of a fall-back are two hour buckets, labelled
with their own instants. A bucket that merely contains a transition (`6 hours` from
midnight) stays one bucket. Labels are exact (dates, instants on an integer grid,
small integers, season names), never float time.

Sparse sub-day sequence positions for naive, UTC and fixed-offset timestamps use
exact integer tick differences, without allocating intermediate calendar buckets.
Zones with daylight-saving transitions retain the validated explicit-grid path and
its grid limit. Missing-period diagnostics retain exact counts while limiting
displayed gap labels.

`Date`, `Datetime` (any unit, naive or aware) and text columns (with
`time_format`, a `strftime` pattern) can be bucketed; the two sides may differ (a
`Date` and a `Datetime` side bucket alike by day). A row whose time is missing or
cannot be placed (unparseable text, or a naive local time that a DST change makes
ambiguous for its bucket or skips) makes the level **not assessed**, with counts;
exclude such rows with `where` if that is intended. A calendar bucket of a numeric
column, an hourly bucket of dates, or sides whose labels differ in type are not
assessed either.

**Two positions.** Each bucket has two exact integer positions, which differ only
for sub-day buckets across a DST change:

- the *sequence* position counts buckets of the exact grid: consecutive buckets are
  one apart, both fall-back hours count and the skipped spring hour does not. It
  measures spacing in time;
- the *calendar* position is the local wall-clock slot: an hourly bucket's position
  mod 24 is its local hour, the two fall-back hours are two observations of slot 1
  (never merged into one bucket) and slot 2 is absent on the spring-forward day. It
  is the position for cycles and seasons.

| Consumer | Uses |
|---|---|
| TMP-001 rate of change, TMP-002 and TMP-003 trends | sequence (rate per bucket of elapsed sequence) |
| TMP-007 season (`position mod season_length`) | calendar (hourly with 24: the local hour) |
| TMP-004 missing periods, `temporal` gaps | the exact bucket grid (`bucket_grid`) |
| TMP-005, TMP-006, TMP-008 | bucket order only |
| `temporal` per-bucket measures | bucket labels (no position) |
| `series_shape` correlation, trend, lag, peak shift, level shift | sequence, counted from the same first bucket on both sides |
| `series_shape` profile | calendar slot of the `profile` |
| `shift_time` | the time values themselves (calendar offsets on the local wall clock) |

### `temporal`

Both sides are bucketed and reduced per `group_by` and bucket with the
[measures and comparators](#measures-and-comparators) of `groups`, then compared
bucket by bucket with the same materiality. Checks:

- `buckets`: (group, bucket) keys on one side only (`missing_buckets: ignore` to accept);
- one check per measure: material changes, the largest first, bounded evidence;
- `gaps`: buckets of the expected calendar grid a side lacks, for each of its groups,
  over the sides' common span (or `start`/`end`). Cyclic profiles expect the whole
  cycle (day 366 is never expected). `allow_gaps: true` reports without failing;
  `gaps: false` leaves the check out. A grid larger than `max_buckets` is not
  assessed.

### `series_shape`

Each `group_by` group is a series of one `measure` per bucket on each side. On the
buckets both sides have (at least `min_buckets`, and two level-shift segments):

| Check | Compares | Threshold |
|---|---|---|
| `correlation` | Pearson and Spearman correlation | `min_pearson`, `min_spearman` (0.9) |
| `trend` | least-squares slope per bucket on each side | `rtol` (0.1), `atol` |
| `profile` | mean per cycle position (`profile: {profile: day_of_week}`), normalized; half the summed absolute share differences | `max_distance` (0.05) |
| `peak` | the bucket and value of each side's peak (around the cycle for profiles) | `max_shift` (0 buckets), `rtol`, `atol` |
| `lag` | the cross-correlation lag within `max_lag` that matches best; lag 1 means the candidate follows the baseline by one bucket | `allowed` (0) |
| `level shift` | the TMP-008 two-segment screen on each side: a shift in one version only | `threshold`, `min_segment`, `min_relative_shift`, `max_offset` |

Set a check to `null` to leave it out. Correlations, slopes and profiles are exact
rationals, and thresholds are decided exactly; two flat series agree, a flat series
against a varying one does not. A series without enough common buckets is reported
in a `too short` check that is **not assessed**, never passed; a series on one side
only fails the `series` check (`missing_series: ignore` to accept).

`ShapeResult.series` lists each series' metrics (bounded by the evidence limit) and
its tests (`SeriesShape.significance`); `TemporalResult.measures` and
`ShapeResult.assessments` carry each family's `significance`
([significance](#significance)).

### Calendar alignment

The `shift_time` step compares one version at a shifted time, for example this
year's candidate with last year's baseline:

```yaml
steps:
  - {step: shift_time, column: day, offset: {years: 1}}          # V2 2025 vs V1 2024
  - {step: shift_time, column: day, offset: {years: 1, same_weekday: true}}  # 364 days
  - {step: shift_time, column: ts, offset: {hours: 1}, unaligned: drop}
```

`offset` is how far the candidate is ahead: its time moves back by it (`side:
baseline` moves the baseline forward instead). Years, months, weeks and days are
calendar steps on the local wall clock (`time_zone`), hours, minutes and seconds
elapsed time. A value without an exact counterpart (29 February a year earlier, a
31st a month earlier, a local time that a DST change skips or repeats) refuses the
alignment, or with `unaligned: drop` is dropped and counted in
`DatasetComparison.alignment`. As an alignment step it applies before every level, so
keyed `records` can be compared year over year too.

## In space

Zones are group keys: an attribute (`county`, `taz`, a link id) compares directly,
and the `assign_zones` step gives points the id of the zone polygon (or grid cell)
they fall in, before any level runs, so `groups`, `temporal`, `records` and the
`spatial` level can all group by zone:

```python
result = compare_datasets(
    "v1/trips.parquet",
    "v2/trips.parquet",
    keys=["trip_id"],
    steps=[{"step": "assign_zones", "x": "lon", "y": "lat",
            "layer": {"path": "zones/taz.geojson", "id": "taz"}}],
    levels=[
        {"level": "spatial", "measures": ["count"], "weights": {"method": "rook"}},
        {"level": "temporal", "time_column": "day", "bucket": "week", "group_by": ["zone"]},
        {"level": "locations", "x": "lon", "y": "lat", "max_distance": 25},
    ],
)
```

Point-in-polygon, zone layers from GeoJSON or shapefiles, CRS transforms and
polygon contiguity need DuckDB's `spatial` extension; without it those checks are
not assessed, never passed, while attribute zones keep working. Zone layers,
assignment policies (boundaries, overlaps, unassigned points), grids, crosswalks,
neighbour weights and both levels are described in [spatial data](spatial.md).

## Distributions

The `distribution` level compares each column's whole distribution, overall or per
member: a `group_by` group (a zone column is a group column too) and, with `time`, a
time bucket within it.

```python
levels=[{"level": "distribution", "columns": ["speed", {"column": "fc", "kind": "categorical"}],
         "group_by": ["zone"], "time": {"time_column": "day", "bucket": "month"},
         "thresholds": {"max_psi": 0.2, "max_ks": 0.1, "max_js": 0.05},
         "quantiles": {"probabilities": [0.1, 0.5, 0.9]}}]
```

| Kind | Metrics (materiality) | Test (significance) |
|---|---|---|
| numeric | PSI over the baseline's `bins` quantile bins with open tails (STA-002's binning; a baseline with few distinct values gets one bin per value), KS distance, Wasserstein distance (in data units), Jensen-Shannon divergence on the same bins, quantile deltas at `probabilities` | `ks` (default) or `anderson_darling`; the other statistic is reported too |
| categorical | PSI and Jensen-Shannon over the categories | `chi_square` (default) or `g_test` of homogeneity |

A member is material when a metric passes its threshold (`max_psi` 0.2, `max_ks` 0.1,
`max_js` 0.05, `max_wasserstein` off; `null` leaves one out) or a quantile moves
beyond the column's tolerance. Each column's members form one family for the
correction. Categories that appeared or disappeared within a member are reported in
`<column>: categories` (`new_categories: ignore` to report without failing). A member
with fewer than `min_values` values on a side, values beyond Float64's exact range, or
more values than `max_work` leaves the check not assessed. `DistributionResult.members`
keeps each member's metrics, bin shares, quantiles and test (bounded by the evidence
limit). Jensen-Shannon is the divergence in bits (0 to 1); SciPy's `jensenshannon` is
its square root.

## Relationships

The `relationships` level compares how columns relate, per member (`group_by`,
`time`), with three optional parts:

```python
levels=[{"level": "relationships", "group_by": ["county"],
         "correlation": {"columns": ["aadt", "vmt", "crashes"], "max_change": 0.1},
         "ratios": ["fatal / crashes", {"numerator": "nox", "denominator": "vmt", "rtol": 0.01}],
         "agreement": {"columns": ["aadt"], "fit": "deming", "min_ccc": 0.99,
                       "max_slope_deviation": 0.05, "max_abs_bias": 50}}]
```

* `correlation`: every pair of `columns` is correlated on each side (Pearson, and
  Spearman on average ranks), exactly, over the member's complete rows. A change
  beyond `max_change` is material; Fisher's z test of two independent correlations
  decides significance (Spearman with the Fieller-Hartley-Pearson variance 1.06),
  one family per method across pairs and members.
* `ratios`: `sum(numerator) / sum(denominator)` per member on each side, exact,
  material beyond the ratio's tolerance, tested as two Poisson rates (the
  denominators as exposures) when the numerators sum to whole counts; otherwise
  judged by materiality (`test: none`).
* `agreement` needs `keys`: records are matched by key (members by the candidate's
  values) and each column's pairs give the Bland-Altman bias and 95 % limits of
  agreement, Lin's concordance correlation, MAE, RMSE, bias, MAPE, sMAPE, R² and the
  line `candidate = intercept + slope * baseline` (OLS, or Deming with
  `error_ratio`). Thresholds per metric (`max_abs_bias`, `max_loa`, `min_ccc` 0.99,
  `max_mae`, `max_rmse`, `max_mape`, `max_smape`, `min_r_squared`,
  `max_slope_deviation` 0.05, `max_abs_intercept`) decide materiality, and the joint
  F test of `slope = 1, intercept = 0` (jackknife covariance for Deming)
  significance.

These tests have no equivalence form: under `decision: equivalence` the checks are not
assessed. A member without enough complete rows, a constant column or a zero
denominator is not assessed, never passed.

## Contributions

The `contribution` level says what explains the total change of an additive measure
(`count`, `sum(x)`, `null_count(x)`), along one or more dimensions:

```python
levels=[{"level": "contribution", "measure": "sum(fatal)", "share": 0.8,
         "dimensions": ["zone", ["zone", "fc"],
                        {"columns": ["zone"], "time": {"time_column": "day", "bucket": "month"}}],
         "rates": [{"numerator": "fatal", "denominator": "vmt", "dimension": "zone"}]}]
```

For each dimension every member's change (`candidate - baseline`, a member on one side
only counting 0 on the other) is its contribution; the contributions sum exactly to
the total change (they are exact rationals, kept as `exact_change`). They are sorted
largest first with their share of the total change and the cumulative share, and the
smallest set of members explaining `share` of it is named (`minimal_set`; taking the
largest changes in the total's direction first is optimal). The `contribution:
<dimension>` check judges each member's change as a `groups` measure would (materiality
and significance, one family per dimension); `total` judges the whole.

### Mix and rate

A rate such as fatalities per vehicle-mile can change because the strata's own rates
changed (**rate** effect) or because the denominator moved between strata with
different rates (**mix** effect). With weights `w = D / sum(D)` and rates `r = N / D`
per stratum, the overall rate is `R = sum(w r)` and, stratum by stratum,

```text
w_c r_c - w_b r_b = (w_c - w_b) r_b          mix
                  + w_b (r_c - r_b)          rate
                  + (w_c - w_b)(r_c - r_b)   interaction
```

so `mix + rate + interaction = R_c - R_b` exactly (`method: symmetric` gives
Kitagawa's two-term form, each difference weighted by the mean of the other factor).
A worked example with two road classes:

| | baseline fatal / vmt | rate | weight | candidate fatal / vmt | rate | weight |
|---|---|---|---|---|---|---|
| A | 10 / 100 | 0.1 | 1/2 | 60 / 300 | 0.2 | 3/4 |
| B | 40 / 100 | 0.4 | 1/2 | 40 / 100 | 0.4 | 1/4 |
| all | 50 / 200 | **0.25** | | 100 / 400 | **0.25** | |

The overall rate did not change, but not because nothing happened: travel shifted to
the safer class A (mix `1/4 * 0.1 - 1/4 * 0.4 = -0.075`) while class A became twice as
dangerous (rate `1/2 * 0.1 = +0.05`; interaction `1/4 * 0.1 = +0.025`). The parts are
exact rationals (`RateDecompositionResult.exact`) and the `rate: <name>` check judges
the overall rate's change (a Poisson test of the totals for whole-number numerators).

## Reports

When a report holds CMP-001 results, the HTML report has a **Dataset comparisons**
section with, per comparison, a summary card (its verdict, a verdict per level with
check counts, the alignment steps and notes) and a drill-down per level: each check's
message, evidence and significance (test, p, adjusted p, effect and interval), and
inline SVG charts:

| Level | Charts |
|---|---|
| `contribution` | a waterfall from the baseline total through the largest contributions to the candidate total; a heat map of relative change for two-column dimensions (zone x month); a waterfall of each rate's mix, rate and interaction |
| `temporal` | a heat map of relative change by group and bucket; baseline (solid) and candidate (dashed) lines per bucket for the first groups |
| `distribution` | each side's share per bin or category (candidate hatched) and the quantile deltas, for the first members |
| `spatial` | a choropleth of each zone's relative change, from the zone layer's (or grid's) outlines simplified when the level ran (`map`, `map_vertices`) |

Charts read without colour (values are labelled, decreases and the candidate hatched,
the candidate's line dashed) and follow the light or dark theme. A chart with more
marks than `HtmlLimits.max_chart_points`, or beyond the document's `max_chart_bytes`,
is replaced by "Chart omitted: too large"; a map without outlines says why (for
example, DuckDB's spatial extension is unavailable). Levels keep at most
`chart_points` values for charts (500 by default, within the evidence byte limit).

The JSON report keeps the typed `DatasetComparison` with the first CMP-001 result of
each comparison (`metrics["comparison"]`; `QAReport.comparisons()` reads them back),
trimmed to the evidence limit with what was left out listed in its `omitted`; when
`engine.max_report_bytes` trims the report, the omission is disclosed in
`metadata.resource_usage.report_details_omitted` and the section is built from the
findings alone. A `DatasetComparison` from `compare_datasets` writes the same section
as its own page and its JSON:

```python
result.write_html("comparison.html")
result.write_json("comparison.json", max_bytes=16_777_216)  # DatasetComparison.model_validate_json
```

```text
qavf compare-datasets v1/counts.parquet v2/counts.parquet -c spec.yaml --html out.html --json out.json
```

The spec holds the options of `compare_datasets` (`keys`, `columns`, `levels`, `decision`,
...; paths relative to the working directory). The exit code is 0 when every check was
assessed and passed, 1 when one failed or was not assessed, and 2 for a usage or
configuration error.

## Significance

A material difference can still be chance: two versions drawn from the same process
differ in every group. Unless `decision` is `material`, each check also tests its
differences and decides with a [policy](inference.md#decision-policies):

| `decision` | A difference fails when |
|---|---|
| `material_and_significant` (default) | it is material **and** significant after the correction |
| `material` | it is material (no test runs; for a core install without the stats extra) |
| `significant` | it is significant, however small |
| `equivalence` | equivalence within the tolerance is **not** shown (TOST-style) |

Each check is one family for the multiple-testing `correction` (`bh` by default;
`by`, `holm`, `bonferroni`, `none`) at level `alpha` (0.05): the groups of a measure,
the buckets of a temporal measure, the zones of a spatial measure or of `lisa`, the
series of a shape check, the columns of the record shift. Members that needed no test
(immaterial under the default) count as `p = 1`, which keeps the correction valid.

| Level | What is tested |
|---|---|
| `groups`, `temporal` | each group or bucket's change of a measure, from both sides' values in it, by the measure's default test (`mean`: Welch's t, `median`: Mann-Whitney, `count` and integer `sum`: Poisson, other `sum` and `quantile`: bootstrap, `share` and `null_count`: two-proportion, `std`: Brown-Forsythe), or `test:` per measure |
| `records` | the `shift` check: per numeric column (and `shift_by` group), a paired test (`test: paired_t` by default; `wilcoxon`, `sign`, `paired_permutation`, `paired_bootstrap`) of `candidate - baseline` over matched rows, with the mean difference judged by the column's tolerance; value changes stay facts, reported whatever the policy |
| `series_shape` | `correlation`: Fisher z of the Pearson correlation against `min_pearson` with Bartlett's effective sample size; `trend`: the slope difference with a block bootstrap (`dependence: block`, or `hac`, `none`); `lag`: a block bootstrap of the best lag's correlation gain; `profile`, `peak` and `level shift` have no test and are judged by threshold |
| `spatial` | each zone's change of a measure, as for `groups`; `rank`: Fisher z of Spearman's correlation against `min_spearman`; `shares`: two-proportion per zone (counts); `moran`: a permutation test of the global Moran's I of the changes; `lisa`: conditional permutation tests of the local Moran's I per zone, corrected across zones |
| `distribution` | each member's distribution change, by `ks` or `anderson_darling` (numbers) or `chi_square`/`g_test` (categories), one family per column |
| `relationships` | `correlation`: Fisher z of two correlations per pair and member; `ratios`: two Poisson rates per member; `agreement`: the joint F test of slope 1 and intercept 0 per member |
| `contribution` | each member's change as for `groups`, one family per dimension; `total`: the whole; `rate`: two Poisson rates of the totals |

`min`, `max`, `distinct_count` and `weighted_mean` have no sound test: they are
judged by materiality under every policy, and their `significance` says so
(`applicable: false`). A test that should apply but cannot run (no SciPy, too few
values, values that are not counts, beyond `max_work`) leaves the check **not
assessed**, never passed.

Set them on a comparison (or override them on one level), and pick a test per
measure with `test:`, as in the [CMP-001 example](#in-a-configuration-cmp-001):

```text
decision: material_and_significant     # the default
correction: bh
alpha: 0.05
seed: 0                                # resampling tests reproduce exactly
measures: [sum(aadt), {measure: median, column: speed, test: bootstrap},
           {measure: mean, column: speed, name: speed distribution, test: ks}]
```

Evidence of a failing group, bucket or column adds `test`, `p_value`, `adjusted_p`,
`effect`, `ci_low`, `ci_high` and the decision's reason, within the evidence limits;
the check's `significance` and `metrics` summarize the family (test, policy, alpha,
correction, family size, significant and not assessed counts), and so do
`MeasureComparison.significance`, `ColumnChanges.significance`,
`RecordsResult.significance`, `ShapeAssessment.significance` and
`SeriesShape.significance`. The tests, their defaults and calibration are described
in [statistical inference](inference.md).

## In a configuration: CMP-001

```yaml
datasets:
  v1: {path: releases/2024/counts.parquet, rename: {SiteID: site}}
  v2: {path: releases/2025/counts.parquet}
rules:
  CMP-001:
    params:
      comparisons:
        - name: counts 2024 vs 2025
          baseline: v1
          candidate: v2
          keys: [site]
          columns: {aadt_k: aadt}
          units: {aadt: {factor: 1000}}
          decision: material_and_significant  # the default; material runs no test
          correction: bh
          levels:
            - structure
            - {level: coverage, allow_added: true}
            - {level: records, shift_by: [county]}
            - level: groups
              severity: WARNING
              group_by: [county]
              measures: [sum(aadt), {measure: median, column: aadt, test: bootstrap}]
            - {level: temporal, time_column: counted_on, bucket: month, measures: [sum(aadt)]}
            - {level: series_shape, time_column: counted_on, bucket: month, measure: sum(aadt),
               trend: {dependence: block}, decision: significant, alpha: 0.01}
```

Each check becomes one result, `CMP-001 [counts 2024 vs 2025: records: values]`, so
severities (per level with `severity`), `engine.require`, evidence limits, redaction
and the HTML/JSON reports work as for every rule. Datasets keep their sources,
secrets, load policies and isolation settings.

## Regression comparisons with renamed columns

The keyed regression API takes the same mapping: `compare_frames`, `compare_outputs`
and `assert_outputs_match` accept `column_map={"old": "new"}`, REG-001 baselines
accept `rename: {old: new}`, and `qavf compare` accepts `--map OLD=NEW` (repeatable).
Keys, ignored columns and tolerances then use the new names.

## Extending

Alignment steps, measures, comparators, levels and statistical tests are registered
components; see [writing plugins](plugins.md#comparison-components).

### Dependency and reduction contracts

`ComparisonLevel.required_columns()` may return a `frozenset` of every aligned
column the level reads. The engine adds comparison keys and known alignment/filter
dependencies before taking coherent source observations. The default `None`
preserves all columns: retain it for schema-wide checks and plugins whose dependency
set is not known. Plugins must not omit dependencies from a declared set. Unknown
alignment steps also retain full input columns. Rename collisions and derived-column
collisions are checked against the full source schema before projection. A proven
identity transformation reuses its observation instead of writing another snapshot.

`Measure.reduction(data)` optionally returns a native scalar aggregate expression.
Compatible measures and their statistical extras are collected together per side;
`compute(data)` remains the fallback for custom algorithms and exact reductions.
Only advertise an expression when it preserves the measure's null, dtype, overflow,
and exactness semantics. A count/min/max expression can be fused; replacing an exact
sum with an unchecked native sum cannot.

### Work and display limits

Correlation admission counts values across all selected column pairs and methods
before allocating pair plans. Raw pair samples are processed one pair at a time and
only requested methods run. Agreement admission counts every requested column;
integer/decimal OLS metrics use exact differences and sufficient statistics. A test
or Deming fit requiring binary64 receives losslessly centred values or becomes not
assessed. Display metrics round only after exact materiality decisions where the
metric is rational; square-root metrics retain their stated numerical semantics.

Numeric distribution distances retain every shared bin. Display detail is
separately limited to `2 * bins` named categories plus an explicit `Other` bin whose
shares include the remaining mass. Categorical displays use the same disclosure.
Groups with no usable categorical values remain in assessment coverage.

Location summaries collect aggregate counts and bounded evidence; point joins and
geometry batches use owned observations. Record descriptions reuse a matched
observation when several consumers need the same join. Spatial distance builders
receive the weight configuration's `max_edges` and `max_work` limits.

Comparison HTML streams repeated sections and builds charts only as the renderer
reaches them. Exhausted chart budgets prevent later builders from running. A first
oversized SVG still has to be built to determine its encoded size; point limits and
output byte limits are different controls. JSON detail trimming visits nested
significance families and discloses only details actually removed. Verdicts and
complete counts remain present even when optional evidence is omitted.
