# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.0] - 2026-10-30

### Added

- **Stray values no longer hide a column's limits.** When a column declared `int`,
  `float`, `decimal`, `bool`, `date` or `datetime` is read as text because of a few
  cells, INP-002 reports exactly those cells with their rows and values, and INP-003
  checks every limit on the remaining values instead of reporting them not assessed.
  Text `missing_values` such as `N/A` now also work on these columns, and
  `blank_as_missing` is accepted with any `dtype`.
- **`text_columns`** for CSV file sources reads the named columns as written, so codes
  and identifiers that look numeric keep their leading zeros. Columns declared
  `dtype: str` are added automatically. `qavf init` drafts integer columns written
  with leading zeros (`007`) as text.
- **`normalize`** compares text in `isin`, `notin` and `unique` after normalizing it:
  `case` ignores letter case and `trim` ignores surrounding whitespace.
- An absent column names its likely spelling in the data (`ID` for `id`, `Site ID` for
  `site_id`, a close misspelling) in INP-002, INP-003 and INP-004.
- `.xlsm` workbooks are read as Excel; a legacy `.xls` workbook explains how to read it.
- **Unreadable CSV lines are located.** When a CSV line cannot be read (a value of the
  wrong type past the reader's type sample, extra or missing fields, a wrong encoding),
  checks report how many lines fail, the first one's line number, column and text, and
  what to set, instead of the reader's raw error. The lines are found with one bounded
  pass, only after a read has failed.
- **Lossy reads are reported.** With `store_rejects` or `ignore_errors`, INP-002 fails
  when more lines were left out than the new `max_rejected_rows` (default 0), and lists
  them by file and line.
- INP-002 reports duplicate and blank CSV header names that the reader renamed (`v_1`,
  `column3`), and a header row the reader did not recognize.
- **Declared types are read, not guessed.** In a CSV, a column declared `int`, `float`,
  `bool` or `date` (or `datetime` with a `format`) is read as text and converted in the
  same scan, so the reader's sample of the first rows never decides its type. A value
  that is not of the type is missing for every check, and INP-002 lists it with its row
  and text. Polars and DuckDB recognize exactly the same text for each type.
- `format` for `date` and `datetime` columns written another way than ISO 8601, and
  `true_values`/`false_values` for `bool` columns (`Y`/`N`).
- A column declared `int` accepts floats that are whole numbers, as spreadsheets store
  every number; INP-002 lists the rows holding a fraction.
- A text cell in an Excel number or date column names the cell and how to read it.
- **Every row-level finding locates its rows.** Row expressions (INP-006) list
  `_row` for their violating rows, found only after an expression fails. Evidence from
  a dataset read from several files adds `_file` and `_file_row`. A failing
  `max_null_fraction` lists the missing rows. The README describes how to read
  evidence.
- `not_null` in a check makes a field required only in the check's rows, e.g.
  `{where: "status = 'closed'", not_null: true}`, and lists the rows missing it.

### Fixed

- A CSV value past the reader's type sample could be silently changed: in a column typed
  as an integer from its first rows, `2.5` was read as 3, `0x10` as 16 and `1e3` as 1000,
  and every check passed. Declared columns are now converted strictly (see above).
- **Decimal text is converted exactly.** A `decimal` column read from text takes the
  scale its values need (up to 38 digits); a value that does not fit is invalid and
  listed by INP-002, never rounded into the limits.
- Header names that only look generated (`column0`) or hold a delimiter (`"rate,count"`)
  are no longer reported as renamed when the file really spells them that way.
- A metric normalized by its own dataset (`normalize_by` on the same dataset) reads its
  numerator and denominator from one snapshot of a changing source.
- **Comparison error limits are hard limits.** `max_rmse`, `max_mae`, bias, agreement,
  concordance and R² limits fail on their value under every decision policy; the
  identity-line test decides only the slope and intercept limits. Previously an
  exceeded error limit could pass when the fitted line stayed on the identity.
- A comparison whose requested columns cannot all be compared (a type change, say) is
  `incomplete` with a "values not compared" check, instead of passing on the rest.
- **A report never replaces an input.** `qavf run` and `qavf compare-datasets` refuse a
  report path naming one of the run's data or configuration files, including through
  a link, before anything is written.
- Configuration that reuses one object in many places is no longer expanded into a
  tree: shared parts stay shared, a configuration expanding past 1,000,000 values is
  refused, and a cyclic one is reported instead of overflowing the stack.
- Streamed JSON reports equal Pydantic's own output for plugin result models,
  including their extra fields and custom serializers.
- Layers, crosswalks and weight tables treat a NaN identifier as missing, as every
  other check does.
- Comparison charts keep groups apart whose labels look alike (`('a, b', 'c')` and
  `('a', 'b, c')`, floats equal to six digits, a null and the text `—`), and maps find
  float zone ids.
- A temporary file whose removal fails (a Windows sharing lock) stays charged to the
  spill budget and is removed by the next cleanup; run cleanup attempts every step and
  reports what failed.
- **A rule calling `sys.exit` no longer ends the host program.** It is reported as an
  ERROR (not assessed) in sequential, thread-pool and timed execution alike, including
  a generator rule failing while its findings are read. `KeyboardInterrupt` still
  stops the run.
- `refresh()` of a component registry rebuilds its installed components: one whose
  entry point was removed, or whose replacement failed to load, is no longer
  available.
- Whole counts reach statistical tests as exact integers (`2**53 + 1` is no longer
  `2**53`). The exact Poisson test reports totals above `2**53`, and SciPy results that
  are not probabilities, as not assessed instead of raising.

### Changed

- A value that is not of a column's declared type counts as missing for every check
  other than INP-002, whichever way the data is read.
- Zoned timestamps load and compare in UTC on every machine. They previously took the
  host's time zone, so the same file and bounds could pass on one machine and fail on
  another.
- INP-004 duplicate-key evidence lists the first row positions of each key (`rows`).
- A column's `unique` limit reports the rows holding repeated values, like other row
  limits, instead of only stating that values repeat.
- **Column-check evidence shows the checked columns.** A failing column check's rows
  list the checked column, the columns of its `where` and unit column, and the
  `primary_key`, with `_row` and `violates`; unrelated columns are no longer read for
  each failing check. Set `evidence_columns: all` in a dataset's `schema` for whole
  rows, as before.
- `qavf.spatial.compose` documents its exactness per step: each step's weights are
  exact and rounded once, so a chain of several steps is rounded once per step.

### Performance

- Declared-text checks convert columns in the same batched scan as the other checks,
  without loading a converted copy of the column.
- Parquet file row positions in evidence come from the file footers, not a scan.
- `series_shape` lag searches and level-shift screens are admitted against `max_work`
  before they run, with TMP-008's cost function, across every series and side; a
  series beyond the budget is a not-assessed `resource limit` check.
- A pandas frame or Arrow table that states more rows than its `row_budget` is
  refused before it is converted.
- Charts cut labels and tooltips to 200 characters before they are built.
- A spilled dataset is read natively, without a Python copy of its file, and outside
  the cache and context locks, so other datasets and operations do not wait for it;
  parallel readers of one dataset still read it once.
- Unordered evidence samples of SQL queries (row expressions) count and keep their
  first rows in one streamed pass instead of staging the whole result in a
  materialized CTE; ordered samples (SQL assertions) keep the CTE.
- An induced subgraph of a spatial weights graph reads only the selected rows.
- Regression comparisons decide 64-bit integer closeness under a tolerance natively
  and exactly, with unchanged results (3.5 to 10 times faster on the measured
  100,000-row changed columns).
- Nested integer and text cells stop at their first difference.
- Agreement and distribution metrics prepare each member's values once.
- Sub-day sequence positions take typed bounds instead of sorting every label.

## [0.5.0] - 2026-10-23

### Added

- **Dataset comparison.** `compare_datasets`, the `CMP-001` rule and
  `qavf compare-datasets` compare a candidate dataset with a baseline, including
  datasets whose columns differ. An alignment pipeline handles this: column mapping,
  lossless casts, unit scaling, recoding, derived columns, filters, time shifts and
  zone assignment. Comparison levels: structure, key coverage, keyed records,
  grouped measures, temporal buckets and series shape, spatial zones and locations,
  distributions, relationships and contributions (with an exact mix-versus-rate
  decomposition). Checks that cannot be assessed are reported as such, never passed.
- **Statistical inference.** `qavf.inference` provides paired, independent-sample,
  equivalence, categorical, count, trend and spatial tests, multiple-testing
  corrections, intervals, bootstraps, effect sizes and agreement metrics. By default
  a comparison fails only when a difference is both material and significant after
  Benjamini-Hochberg correction. TMP-002, TMP-003 and STA-002 use the same tests
  and report `p_value`, `adjusted_p` and `effect`.
- **Spatial building blocks** (`qavf.spatial`). Zone layers come from datasets,
  GeoParquet, GeoJSON or shapefiles. Point-to-zone assignment has explicit boundary
  and overlap policies. Also grids, crosswalks, hierarchies, neighbour weights,
  Moran's I and LISA. GEO-001 accepts a `within` zone layer and SPA-003 a
  `zone_layer`. Geometry needs DuckDB's `spatial` extension.
- **Temporal building blocks** (`qavf.temporal`, `qavf.statistics.series`).
  Calendar buckets follow local time zones and keep DST days intact. TMP-001 to
  TMP-008 accept `bucket`, `time_zone` and `time_format` for Date, Datetime and
  text time columns.
- Reports show a comparison section: a summary, per-check evidence and inline SVG
  charts within `HtmlLimits`. `DatasetComparison` can write its own HTML and JSON
  reports.
- Regression comparisons map renamed columns (`column_map=`, `rename`,
  `qavf compare --map OLD=NEW`).
- A new `spatial` extra (pyproj and SciPy) adds indexed neighbour searches and
  authoritative CRS resolution. `all` now includes it.
- Alignment steps, measures, comparators, levels and statistical tests are
  pluggable through registration decorators and entry points. The shared component
  registry lives in `qavf.plugins.registry`.

### Changed

- Fewer false passes: invalid or empty assessments, unsupported numeric values,
  lossy casts, undefined statistics and duplicate keys are now failures or
  explicitly unassessed. Some verdicts that were wrongly passing will change.
- New default resource limits, with explicit refusal instead of partial results:
  - polygon assignment: 1,000,000 point keys, 2,000,000 mappings, 20,000,000
    candidate pairs;
  - neighbour graphs: 1,000,000 edges;
  - profiling category state: 1 MiB.

  Raise these deliberately for larger workloads.
- Unknown CRS classifications need `qavf[spatial]`. The minimum supported pyproj
  is 3.6.1.
- Invalid fractional byte sizes are rejected rather than rounded. Colliding HTML and
  JSON report paths are refused before writing (CLI exit code 2). Typed Parquet
  reads refuse explicit cloud credential-provider chains.
- A core install without the `stats` extra should set `decision: material` for
  comparisons.

### Performance

- Neighbour weights use compact sparse arrays.
- Point-to-zone matching runs in batches, resolving overlaps natively.
- Map outlines are checked against their vertex budget before they are built.
- Compatible lazy column checks are evaluated in one scan.
- Rule provenance no longer scans every installed package.
- Fewer redundant source evaluations and serializations, plus bounded work for
  resampling, correlations, graphs and reports.

### Fixed

- Exact numeric handling: Int128 scaling, Decimal rounding, nanosecond and missing
  group identities, subnormal ratios and floating-point threshold boundaries.
- Statistics: BCa acceleration, contingency counts, resampling stability, DST bucket
  grids and dependent time ordering.
- Execution: deadline monitoring through cleanup, cancellation at native operation
  entry, plugin registration atomicity and retained-memory accounting.
- Compatibility with the minimum supported Polars and DuckDB versions.

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
