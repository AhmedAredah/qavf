# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- Add spatial comparison. `qavf.spatial` holds the shared building blocks: zone layers
  from a configured dataset (WKB or WKT), GeoParquet, GeoJSON or a shapefile, validated
  (`ST_IsValid`; invalid geometry is refused unless `repair: true`), with declared CRSs
  (a file's own CRS must agree); x-first (`always_xy`) transforms; one haversine
  distance for floats, arrays and Polars expressions; point-to-zone assignment through
  a DuckDB spatial join with a bounding-box prefilter and explicit policies for
  boundaries (`intersects` or `contains`, ties to the lowest zone id), overlapping
  zones (`error`, `first`, `all`) and unassigned points (`max_unassigned`); regular
  grid cells with labels and an expected grid; crosswalks and hierarchies with split
  weights proven to sum to 1, and exact aggregation up a hierarchy; and neighbour
  weights (queen and rook contiguity, nearest neighbours, distance bands, tables) with
  islands reported.
- Add the `assign_zones` alignment step (a layer per side when zones were re-drawn,
  grid cells, or attribute zones, plus a crosswalk), the `spatial` level (measures per
  zone, one-sided and expected zones, Spearman rank correlation, share redistribution,
  global Moran's I of the changes and LISA clusters of change, corrected for multiple
  testing) and the `locations` level (distance moved by keyed features, area and shape
  changes of geometries). `assign_zones` with `temporal` compares zones bucket by
  bucket. Steps and levels name the further datasets they read (`datasets()`), which
  CMP-001 requires and validates.
- Add `local_morans_i` (LISA with conditional permutation p-values and quadrants),
  `morans_statistic` and `rank_correlation_test` (Spearman, Fisher z with the
  Fieller-Hartley-Pearson variance) to `qavf.inference`.
- GEO-001 takes an optional `within` zone layer or study area per point set (a
  separate check; `boundary: intersects` or `contains`), and SPA-003 an optional
  `zone_layer` whose ids are the expected entities. Existing configurations are
  unchanged.
- Geometry needs DuckDB's `spatial` extension: the Docker image and
  `scripts/verify_release.py extensions` (now also run by the CI test job) install it.
  Without it, every check that needs geometry is not assessed, with the reason;
  attribute zones, grids in the data's CRS and tables work without it.

- Add `qavf.inference`, a statistical inference library shared by comparisons and
  rules: paired (t, Wilcoxon, sign, permutation, bootstrap), equivalence (TOST),
  independent-sample (Welch t, Mann-Whitney, KS, Anderson-Darling, Brown-Forsythe,
  permutation, bootstrap), categorical (chi-square, Fisher, G, McNemar,
  Stuart-Maxwell), count and rate (two-proportion, exact conditional Poisson), trend
  and series (Mann-Kendall with Sen slope, seasonal Kendall, slope differences, Fisher
  z, lag significance) and spatial (Moran's I from given weights) tests with a typed
  `TestResult`; Bonferroni, Holm, Benjamini-Hochberg and Benjamini-Yekutieli
  corrections; analytic, percentile and BCa intervals; seeded block bootstraps,
  Newey-West errors and a work budget; effect sizes and agreement metrics; and
  decision policies combining materiality with significance. Without the `stats`
  extra a test is not assessed. Tests are pluggable (`register_test`, the
  `qavf.inference.tests` entry-point group, `plugins.required: inference.tests:name`).
- Comparisons now decide `material_and_significant` by default: a difference fails
  only when it is material and statistically significant after a Benjamini-Hochberg
  correction across each check's family. `decision`, `alpha`, `correction`, `seed`
  and `max_work` are set per comparison or level, and `test:` per measure; each
  measure has a default test (`mean`: Welch, `median`: Mann-Whitney, `count` and
  integer `sum`: Poisson, other `sum` and `quantile`: bootstrap, `share` and
  `null_count`: two-proportion, `std`: Brown-Forsythe), and `min`, `max` and
  `distinct_count` are judged by materiality alone. `records` gains a `shift` check
  of paired tests per column (and `shift_by` group); `series_shape` tests correlation
  (Fisher z with an effective sample size), trend (block bootstrap or HAC) and lag
  (block bootstrap). Evidence carries p-values, adjusted p-values, effects and
  intervals, and every `significance` slot is filled. A core install without the
  `stats` extra should set `decision: material`, the previous behaviour.
- TMP-002, TMP-003 and STA-002 use the shared tests and corrections, gain an optional
  `correction` (defaults unchanged) and report `p_value`, `adjusted_p` and `effect`
  in their evidence.
- Move the component registry to `qavf.plugins.registry`, shared by comparison
  components and statistical tests; `qavf.compare.registry` re-exports it.
- Add dataset comparison: `compare_datasets` (also `qavf.compare`) and the `CMP-001`
  rule compare a candidate dataset with a baseline after an alignment pipeline
  (column mapping, lossless casts, exact unit scaling, recoding, derived columns,
  per-side and shared filters) at four levels: structure (with suggested renames),
  key coverage, keyed records and grouped measures with comparators, materiality and
  rank changes. Unassessable checks (missing mapped columns, lossy casts, too few rows)
  are reported as not assessed.
- Make alignment steps, measures, comparators and levels extensible through
  registration decorators and the `qavf.compare.alignment`, `qavf.compare.measures`,
  `qavf.compare.comparators` and `qavf.compare.levels` entry-point groups;
  `plugins.required` accepts `compare.<kind>:name`.
- Map renamed columns in regression comparisons: `column_map=` for `compare_frames`,
  `compare_outputs` and `assert_outputs_match`, `rename` for REG-001 baselines and
  `qavf compare --map OLD=NEW`. Comparison results count changed values per column
  (`column_changes`).
- Compare datasets over time: the `temporal` level compares measures per calendar
  bucket (second to year, with multiples such as `15 minutes`) or cyclic profile
  (hour of day to season) within groups, with one-sided buckets and expected-grid
  gaps; the `series_shape` level compares each group's series by correlation
  (Pearson and Spearman), trend slope, cyclic profile, peak timing and size,
  cross-correlation lag and level shifts; the `shift_time` alignment step compares a
  version at a shifted time (year over year, same weekday, or a lag). Buckets follow
  the local calendar of a time zone, so DST days have 23 or 25 hours and the two
  fall-back hours stay apart; rows that cannot be placed and series too short to
  judge are not assessed.
- Bucket the TMP series rules: `bucket`, `time_zone` and `time_format` let TMP-001 to
  TMP-008 use Date, Datetime and text time columns; numeric time columns behave as
  before. TMP-004 expects every bucket of the calendar grid of a bucketed series.
  Rates and trends use a bucket's exact sequence position and TMP-007 seasons its
  local calendar slot, so hourly seasons stay on the local hour across DST changes.
- Add `qavf.temporal` (time buckets, bucket grids, explicit sequence and calendar
  positions and calendar offsets as native expressions) and `qavf.statistics.series`
  (exact least squares, Pearson and Spearman correlation, cross-correlation lags and
  the level-shift screen).
- Comparison steps can record data-dependent outcomes with `AlignmentContext.note`.
- Add general exact helpers: checked casts and exact scaling (`qavf.typed_values`),
  exact quantile interpolation (`qavf.statistics.ordered`), grouped exact means and
  quantiles, and byte-bounded evidence collection.

## [0.4.1] - 2026-10-18

- Add a `secrets` configuration section for cloud storage: AWS S3 and S3-compatible
  services (Cloudflare R2, MinIO, Wasabi, GCS interoperability) and Azure Blob/ADLS,
  with static keys or the platform credential chain. Secrets apply to DuckDB reads and
  to remote Parquet read through Polars, and are masked in reports and logs.
- Mask `key_id`, `connection_string`, account keys and SAS tokens wherever settings
  or messages are redacted.
- Include the `aws` and `azure` DuckDB extensions in the container image.
- Never stop an isolated worker before its wall-time deadline: on Windows a process
  wait could wake a few nanoseconds early and end the worker before its allowance.

## [0.4.0] - 2026-10-18

First public release.

- Validation API (`validate`, `QARunner`) and CLI for files, globs, URLs, databases,
  SQL queries, Polars/pandas/Arrow frames and Python loaders.
- 28 built-in rules: input contracts, column limits, keys and foreign keys, row
  expressions and SQL assertions, freshness, temporal and spatial checks, outlier and
  drift detection, reconciliation and regression against accepted baselines.
- Fail-closed assessment: checks that cannot be assessed make a run incomplete rather
  than passing, with exact counts and bounded, deterministic evidence.
- HTML and JSON reports with full run provenance and redacted credentials.
- Resource budgets for memory, spill storage, evidence and report size, and optional
  isolated execution with OS-enforced limits on Linux, macOS and Windows.
- Plugin discovery for rules and sources from packages, files and entry points.
