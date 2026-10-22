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

## Fail-closed results

A `DatasetComparison` holds one typed result per level (`StructureResult`,
`CoverageResult`, `RecordsResult`, `GroupsResult`, `TemporalResult`, `ShapeResult`),
each with graded checks. A check
has an exact `affected_count` and bounded `affected_records`: evidence follows the
run's `max_affected_records` and `max_evidence_bytes` limits and is ordered
deterministically, while counts stay complete.

Nothing that could not be compared reads as agreement. A dataset that cannot be read,
a mapped column the baseline lacks, a lossy cast, an unmapped category with
`unmapped: error`, fewer aligned rows than `min_rows` (1 by default), duplicate keys
for the record level or a non-numeric column for a numeric measure all leave the
affected checks **not assessed**. `result.passed` is true only when every check was
assessed and passed.

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
| `steps` | Plugin (or built-in) steps, `{step: name, ...}`, e.g. `shift_time` ([calendar alignment](#calendar-alignment)). | `[{step: trim, columns: [name]}]` |
| `baseline_where`, `candidate_where`, `where` | Keep only rows in scope on one or both sides. | `"year >= 2020"` |

Dataset-level `rename`, `missing_values` and `canonical_columns` apply before the
pipeline, exactly as for every other rule.

## Levels

| Level | Compares | Checks |
|---|---|---|
| `structure` | Columns added or removed after mapping, suggested renames (similar names and similar value samples, never applied), type, declared-unit and nullability changes, row counts (exact unless `rows: {rtol: ...}`), key uniqueness per side | `columns`, `types`, `units`, `nullability`, `rows`, `keys` |
| `coverage` | Keys only in the baseline or only in the candidate, overall and by `group_by`; changes of each column's missing-value rate beyond `max_null_rate_change` | `keys`, `null rates` |
| `records` | Keyed match through the regression engine: rows added, removed, changed, unchanged; per-column change counts and rates; largest absolute and relative differences; null and NaN transitions; categorical transitions (`max_transitions` most frequent, with the total count); the `top_n` largest numeric changes beyond tolerance as evidence | `rows`, `values` |
| `groups` | Measures per `group_by` group on each side; groups on one side only; material changes per measure; optional top-k rank changes | `groups`, one per measure, `rank: <measure>` |
| `temporal` | Measures per time bucket (and `group_by` group) on each side: buckets on one side only, expected-grid gaps on each side, material changes per measure with the largest first | `buckets`, one per measure, `gaps` |
| `series_shape` | Each group's bucketed series on both sides: correlation, trend, cyclic profile, peak, lag, level shifts | `series`, `correlation`, `trend`, `profile`, `peak`, `lag`, `level shift` (and `too short`) |

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

Results carry an empty `significance` slot for statistical tests, which a later
release adds as a separate judgement from materiality.

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

`ShapeResult.series` lists each series' metrics (bounded by the evidence limit), and
both levels keep typed results with empty `significance` slots (per measure and per
series) for the statistical tests of a later release.

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
          levels:
            - structure
            - {level: coverage, allow_added: true}
            - records
            - {level: groups, severity: WARNING, group_by: [county], measures: [sum(aadt)]}
            - {level: temporal, time_column: counted_on, bucket: month, measures: [sum(aadt)]}
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

Alignment steps, measures, comparators and levels are registered components; see
[writing plugins](plugins.md#comparison-components).
